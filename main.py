import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

from data import load_prices
from clustering import cluster_assets, get_pca_for_visualization
from pair_selection import find_pairs
from signals import compute_spread, generate_signals
from backtest import backtest
from half_life import compute_half_life
from hurst import compute_hurst
import visualizations as viz

from config import *

def _prepare_data(tickers: List[str], start_date: str, end_date: str, in_sample_end: str, out_of_sample_start: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # update to receive both closes and opens
    closes, opens = load_prices(tickers, start_date, end_date)

    # prices in/out splits for Closes
    closes_in = closes.loc[:in_sample_end]
    closes_out = closes.loc[out_of_sample_start:]

    # Opens for the out-of-sample backtest period
    opens_out = opens.loc[out_of_sample_start:]

    return closes_in, closes_out, opens_out, closes

def _get_cluster_pairs(cluster_id: Any, cluster: List[str], prices_in: pd.DataFrame, funnel_stats: List[Dict[str, Any]]) -> List[Tuple[Tuple[str, str], Any]]:
    """
    identifies valid pairs within a cluster based on cointegration, half-life, and hurst exponent.
    Returns a list of ((y_ticker, x_ticker), half_life).
    """
    candidate_pairs = find_pairs(prices_in, cluster)
    funnel_stats.append({
        "cluster": cluster_id,
        "step": "cointegration",
        "num_pairs": len(candidate_pairs)
    })

    passed_pairs = []
    for y_tkr, x_tkr in candidate_pairs:
        y_in = prices_in[y_tkr]
        x_in = prices_in[x_tkr]

        beta_in = (y_in.rolling(ROLLING_BETA).cov(x_in) / x_in.rolling(ROLLING_BETA).var()).shift(1)
        spread_in = compute_spread(y_in, x_in, beta_in)

        # half life filter
        hl = compute_half_life(spread_in)
        if hl is None or hl < MIN_HALF_LIFE or hl > MAX_HALF_LIFE:
            funnel_stats.append({
                "cluster": cluster_id, "pair": f"{y_tkr}-{x_tkr}",
                "step": "half_life_rejected", "half_life": hl
            })
            continue

        # hurst exponent filter
        hurst = compute_hurst(spread_in)
        if hurst is None or hurst >= MAX_HURST_EXPONENT:
            funnel_stats.append({
                "cluster": cluster_id, "pair": f"{y_tkr}-{x_tkr}",
                "step": "hurst_rejected", "hurst": hurst
            })
            continue

        # if a pair passes all filters
        funnel_stats.append({
            "cluster": cluster_id, "pair": f"{y_tkr}-{x_tkr}",
            "step": "passed_all_filters", "half_life": hl, "hurst": hurst
        })
        passed_pairs.append(((y_tkr, x_tkr), hl))

    return passed_pairs

def _run_backtests(all_passed_pairs: List[Tuple[Tuple[str, str], Any]], prices_out: pd.DataFrame, opens_out: pd.DataFrame, allocated_capital: float, all_pnl: List[pd.Series], all_trades: List[pd.DataFrame]):
    """
    Runs backtest for all selected pairs using the allocated capital.
    Now accepts opens_out for execution.
    """
    print(f"Allocating ${allocated_capital:,.2f} per pair across {len(all_passed_pairs)} pairs.")

    for (y_tkr, x_tkr), hl in all_passed_pairs:
        # Closes for signal generation
        y_out = prices_out[y_tkr]
        x_out = prices_out[x_tkr]

        # Opens for execution
        y_out_open = opens_out[y_tkr]
        x_out_open = opens_out[x_tkr]

        beta_out = (y_out.rolling(ROLLING_BETA).cov(x_out) / x_out.rolling(ROLLING_BETA).var()).shift(1)
        spread_out = compute_spread(y_out, x_out, beta_out)

        # generate signals using Close prices
        signal = generate_signals(spread_out, ZSCORE_WINDOW, ENTRY_Z, EXIT_Z, half_life=hl)

        pnl, trades = backtest(y_out, x_out, y_out_open, x_out_open, beta_out, signal, allocated_capital)

        if pnl.notna().sum() > 0:
            all_pnl.append(pnl)
            trades["pair"] = f"{y_tkr}-{x_tkr}"
            trades["half_life"] = hl
            all_trades.append(trades)


def main():
    # load tickers
    tickers: List[str] = STOCK_TICKERS

    # load historical prices and split in-sample / out-of-sample
    prices_in, prices_out, opens_out, all_prices = _prepare_data(tickers, START_DATE, END_DATE, IN_SAMPLE_END, OUT_OF_SAMPLE_START)

    # cluster stocks
    clusters: Dict[Any, List[str]] = cluster_assets(prices_in, n_clusters=6)

    # init logs
    all_pnl: List[pd.Series] = []
    all_trades: List[pd.DataFrame] = []
    funnel_stats: List[Dict[str, Any]] = []

    # get valid pairs
    print("\nFINDING PAIRS")
    all_passed_pairs: List[Tuple[Tuple[str, str], Any]] = []

    for cluster_id, cluster in clusters.items():
        pairs_in_cluster = _get_cluster_pairs(cluster_id, cluster, prices_in, funnel_stats)
        all_passed_pairs.extend(pairs_in_cluster)

    if not all_passed_pairs:
        raise RuntimeError("No pairs passed selection criteria.")

    # init capital and divide equally among pairs
    allocated_capital_per_pair = CAPITAL / len(all_passed_pairs)

    # backtesting
    print("\nBACKTESTING")
    _run_backtests(all_passed_pairs, prices_out, opens_out, allocated_capital_per_pair, all_pnl, all_trades)

    # calc portoflio stats
    if not all_pnl:
        raise RuntimeError("No trades generated during backtest.")

    portfolio: pd.Series = pd.concat(all_pnl, axis=1).fillna(0).sum(axis=1)
    cum: pd.Series = portfolio.cumsum()

    # max drawdown calculation
    equity_curve: pd.Series = CAPITAL + cum
    running_max: pd.Series = equity_curve.cummax()
    drawdown: pd.Series = running_max - equity_curve
    max_drawdown_dollars: float = drawdown.max()
    peak_equity: float = running_max.max()
    max_drawdown_pct: float = max_drawdown_dollars / peak_equity if peak_equity > 0 else 0


    # logging and summary
    trade_log: pd.DataFrame = pd.concat(all_trades).reset_index(drop=True)
    trade_log.to_csv("trade_log.csv", index=False)

    funnel_df: pd.DataFrame = pd.DataFrame(funnel_stats)
    funnel_df.to_csv("trade_funnel.csv", index=False)

    pair_summary: pd.DataFrame = trade_log.groupby("pair").agg(
        total_pnl=("pnl", "sum"),
        num_trades=("pnl", "count"),
        avg_pnl=("pnl", "mean")
    )
    pair_summary["sharpe"] = pair_summary["total_pnl"] / (trade_log.groupby("pair")["pnl"].std() * np.sqrt(252))
    pair_summary["max_drawdown"] = trade_log.groupby("pair")["pnl"].apply(lambda x: (x.cumsum().cummax() - x.cumsum()).max())
    pair_summary = pair_summary.sort_values(by="total_pnl", ascending=False)
    pair_summary.to_csv("pair_summary.csv")

    # print reports
    print("\nOUTCOME")
    print(f"Sharpe       : {portfolio.mean()/portfolio.std()*np.sqrt(252):.2f}")
    print(f"Max Drawdown : {max_drawdown_pct:.2%}")
    print(f"Total PnL    : {cum.iloc[-1]:,.0f}")
    print(f"Trades       : {(portfolio != 0).sum()}")


    # visualizations
    pca_df = get_pca_for_visualization(all_prices)
    viz.plot_clusters(pca_df, clusters)
    viz.plot_equity_curve(portfolio, CAPITAL)
    viz.plot_daily_pnl_histogram(portfolio)
    viz.plot_pnl_by_pair(pair_summary)
    viz.plot_sharpe_by_pair(pair_summary)
    viz.plot_pair_funnel(funnel_df)
    viz.plot_half_life_distribution(funnel_df)

    # take top 5 pairs and plot them
    top_5_pairs: pd.Index = pair_summary.head(5).index
    for pair in top_5_pairs:
        viz.plot_individual_pair_dynamics(pair, prices_out, trade_log, ROLLING_BETA, ZSCORE_WINDOW)

    print("Visualizations saved to 'reports/' directory.")

if __name__ == "__main__":
    main()