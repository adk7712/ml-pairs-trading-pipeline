import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import os
from typing import List, Dict, Any

plt.style.use('seaborn-v0_8-whitegrid')

# plot and save the equity curve and drawdowns
def plot_equity_curve(portfolio: pd.Series, capital: float, output_path: str = "reports/equity_curve.png"):
    equity_curve: pd.Series = capital + portfolio.cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # equity Curve
    ax1.plot(equity_curve.index, equity_curve, label='Equity Curve', color='dodgerblue')
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True)

    # drawdown pct
    running_max: pd.Series = equity_curve.cummax()
    drawdown: pd.Series = ((equity_curve - running_max) / running_max) * 100
    ax2.fill_between(drawdown.index, drawdown, 0, color='salmon', alpha=0.5)
    ax2.set_title('Drawdown (%)')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# plot and save the daily pnl histogram.
def plot_daily_pnl_histogram(portfolio: pd.Series, output_path: str = "reports/daily_pnl_histogram.png"):
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio, bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Daily PnL')
    plt.xlabel('Daily PnL ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


# plot and save the total pnl contributed by each pair.
def plot_pnl_by_pair(pair_summary: pd.DataFrame, output_path: str = "reports/pnl_by_pair.png"):
    plt.figure(figsize=(12, 8))
    # filter for pairs with non-zero pnl and sort
    top_pairs: pd.DataFrame = pair_summary[pair_summary['total_pnl'] != 0].sort_values(by='total_pnl', ascending=False)

    colors: List[str] = ['lightgreen' if pnl > 0 else 'salmon' for pnl in top_pairs['total_pnl']]

    sns.barplot(x=top_pairs['total_pnl'], y=top_pairs.index, hue=top_pairs.index, palette=colors)
    plt.title('Total PnL Contribution by Pair')
    plt.xlabel('Total PnL ($)')
    plt.ylabel('Pair')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# plot and save the sharpe ratio for each pair.
def plot_sharpe_by_pair(pair_summary: pd.DataFrame, output_path: str = "reports/sharpe_by_pair.png"):

    # filter for pairs with a valid sharpe ratio and sort
    top_pairs: pd.DataFrame = pair_summary.dropna(subset=['sharpe']).sort_values(by='sharpe', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_pairs['sharpe'], y=top_pairs.index, hue=top_pairs.index, palette='viridis')
    plt.title('Sharpe Ratio by Pair')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Pair')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# plot the distribution of half-lives for both accepted and rejected pairs
def plot_half_life_distribution(funnel_df: pd.DataFrame, output_path: str = "reports/half_life_distribution.png"):
    passed_hl: pd.Series = funnel_df[funnel_df['step'] == 'passed_all_filters']['half_life'].dropna()
    rejected_hl: pd.Series = funnel_df[funnel_df['step'] == 'half_life_rejected']['half_life'].dropna()

    plt.figure(figsize=(12, 7))
    sns.histplot(passed_hl, color="lightgreen", label='Passed', kde=True, bins=30)
    sns.histplot(rejected_hl, color="salmon", label='Rejected', kde=True, bins=30)

    plt.title('Distribution of Half-Lives')
    plt.xlabel('Half-Life (Days)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# plot bar chart showing the number of pairs at each stage of the selection funnel
def plot_pair_funnel(funnel_df: pd.DataFrame, output_path: str = "reports/pair_funnel.png"):

    cointegrated_count = funnel_df[funnel_df['step'] == 'cointegration']['num_pairs'].sum()

    # after cointegration, we have to count how many unique pairs entered the next stages
    # a pair is a candidate for half-life if it passed coint test
    # can count unique pairs from the log steps.
    hl_candidates = funnel_df[funnel_df['step'].isin(['half_life_rejected', 'hurst_rejected', 'passed_all_filters'])]['pair'].nunique()

    # pairs that passed half-life are candidates for the hurst filter
    hurst_candidates = funnel_df[funnel_df['step'].isin(['hurst_rejected', 'passed_all_filters'])]['pair'].nunique()

    # pairs that passed all the filters
    passed_all_count = funnel_df[funnel_df['step'] == 'passed_all_filters']['pair'].nunique()

    funnel_data = {
        'Step': ['1. Cointegrated', '2. Passed Half-Life', '3. Passed Hurst'],
        'Count': [cointegrated_count, hurst_candidates, passed_all_count]
    }

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Step', y='Count', data=pd.DataFrame(funnel_data), hue='Step', palette='coolwarm')
    plt.title('Pair Selection Funnel')
    plt.ylabel('Number of Pairs')

    for container in ax.containers:
        ax.bar_label(container)

    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# plot detailed dynamics for a single pair -> prices, spread, z-score, and trades.
# recalculates beta, spread, and z-score internally
def plot_individual_pair_dynamics(pair_name: str, prices_out: pd.DataFrame, trade_log: pd.DataFrame, ROLLING_BETA: int, ZSCORE_WINDOW: int, output_dir: str = "reports/pair_dynamics"):

    os.makedirs(output_dir, exist_ok=True)

    y_tkr: str
    x_tkr: str
    y_tkr, x_tkr = pair_name.split('-')
    y_price: pd.Series = prices_out[y_tkr]
    x_price: pd.Series = prices_out[x_tkr]

    # recalculate beta, spread and z-score for the out-of-sample period
    beta: pd.Series = (y_price.rolling(ROLLING_BETA).cov(x_price) / x_price.rolling(ROLLING_BETA).var()).shift(1)
    spread: pd.Series = y_price - beta * x_price

    # rolling z score
    rolling_mean: pd.Series = spread.rolling(window=ZSCORE_WINDOW).mean()
    rolling_std: pd.Series = spread.rolling(window=ZSCORE_WINDOW).std()
    z_score: pd.Series = (spread - rolling_mean) / rolling_std

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})

    # top plot -> spread w Bollinger Bands
    ax1.plot(spread.index, spread, label='Spread', color='dodgerblue', alpha=0.8)
    ax1.plot(rolling_mean, label='Rolling Mean', color='orange', linestyle='--')
    ax1.fill_between(spread.index, rolling_mean - 2 * rolling_std, rolling_mean + 2 * rolling_std, color='gray', alpha=0.2, label='Bollinger Bands (2 std)')

    # trades on spread plot
    pair_trades: pd.DataFrame = trade_log[trade_log['pair'] == pair_name]

    # identify entry points
    long_entries: pd.DataFrame = pair_trades[(pair_trades['position'] > 0) & (pair_trades['position'].shift(1) == 0)]
    short_entries: pd.DataFrame = pair_trades[(pair_trades['position'] < 0) & (pair_trades['position'].shift(1) == 0)]

    # identify exit points
    exits: pd.DataFrame = pair_trades[(pair_trades['position'] == 0) & (pair_trades['position'].shift(1) != 0)]

    ax1.plot(long_entries['date'], spread.loc[long_entries['date']], '^', markersize=10, color='lightgreen', label='Long Entry')
    ax1.plot(short_entries['date'], spread.loc[short_entries['date']], 'v', markersize=10, color='salmon', label='Short Entry')
    ax1.plot(exits['date'], spread.loc[exits['date']], 'o', markersize=8, color='gray', label='Exit')

    ax1.set_title(f'Spread & Trades for {pair_name}')
    ax1.set_ylabel('Spread Value')

    ax1.legend()
    ax1.grid(True)

    # bottom plot -> z score w signals
    ax2.plot(z_score.index, z_score, label='Z-Score', color='black')
    ax2.axhline(2.0, color='red', linestyle='--', label='Entry Threshold (+2.0)')
    ax2.axhline(-2.0, color='red', linestyle='--')
    ax2.axhline(0.5, color='green', linestyle='--', label='Exit Threshold (+/-0.5)')
    ax2.axhline(-0.5, color='green', linestyle='--')

    ax2.set_title('Z-Score & Trading Thresholds')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pair_name}_dynamics.png"))
    plt.close()

# plot and save a scatter plot of asset clusters based on PCA.
def plot_clusters(pca_df: pd.DataFrame, clusters: Dict[Any, List[str]], output_path: str = "reports/asset_clusters.png"):

    # create a mapping from ticker to cluster label
    ticker_to_cluster = {ticker: cluster_id for cluster_id, tickers in clusters.items() for ticker in tickers}

    # add cluster labels to the PCA dataframe
    pca_df['cluster'] = pca_df.index.map(ticker_to_cluster)

    plt.figure(figsize=(14, 10))

    # get unique clusters
    unique_clusters = sorted(pca_df['cluster'].dropna().unique())
    palette = sns.color_palette('viridis', n_colors=len(unique_clusters))
    cluster_colors = dict(zip(unique_clusters, palette))

    # create a scatter plot without automatic legend
    sns.scatterplot(
        x='PC1',
        y='PC2',
        hue='cluster',
        data=pca_df,
        palette=cluster_colors,
        s=100,
        alpha=0.8,
        legend='full'
    )

    # annotate each point with respective ticker
    for ticker, row in pca_df.iterrows():
        plt.text(
            row['PC1'] + 0.01,
            row['PC2'],
            ticker,
            fontsize=9
        )

    plt.title('Asset Clusters using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()