import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from config import RISK_PER_TRADE, COMMISSION_BPS

# backtest a single pair using Cash and Share tracking with lazy hedging
# executes trades at OPEN prices and subtracts transaction costs.
def backtest(y_close: pd.Series, x_close: pd.Series, y_open: pd.Series, x_open: pd.Series, beta: pd.Series, signal: pd.Series, capital: float) -> Tuple[pd.Series, pd.DataFrame]:
    pnl: List[float] = []
    trades_list: List[Dict[str, Any]] = []

    # calc spread and volatility for sizing (using close prices)
    spread: pd.Series = y_close - beta * x_close
    vol: pd.Series = spread.rolling(60).std()

    # portfolio state management
    shares_y: float = 0.0
    shares_x: float = 0.0
    cash: float = capital
    prev_portfolio_value: float = capital

    # lazy hedging state
    current_position_state: int = 0 # 0, 1, or -1

    # align data
    dates = spread.index

    for t in range(1, len(spread)):
        current_date = dates[t]

        # signal generated at t - 1 (based on close)
        sig = int(signal.iloc[t - 1])

        # determine target
        target_shares_y = shares_y # default to holding
        target_shares_x = shares_x

        if sig != current_position_state:
            # state change
            if sig == 0:
                # exit
                target_shares_y = 0.0
                target_shares_x = 0.0
            else:
                # entry/flip
                current_vol = vol.iloc[t - 1]

                if np.isnan(current_vol) or current_vol == 0:
                    target_shares_y = 0.0
                    target_shares_x = 0.0
                else:
                    # calc size based on volatility
                    target_size = (capital * RISK_PER_TRADE) / current_vol
                    current_beta = beta.iloc[t - 1]

                    target_shares_y = target_size * sig
                    target_shares_x = -target_shares_y * current_beta

            current_position_state = sig

        # exec trades at OPEN prices
        price_y_trade = y_open.iloc[t]
        price_x_trade = x_open.iloc[t]

        delta_y = target_shares_y - shares_y
        delta_x = target_shares_x - shares_x

        # simulate transaction fees
        traded_value_y = abs(delta_y * price_y_trade)
        traded_value_x = abs(delta_x * price_x_trade)
        commission = (traded_value_y + traded_value_x) * (COMMISSION_BPS / 10000.0)

        # cost using OPEN prices
        trade_cost = (delta_y * price_y_trade) + (delta_x * price_x_trade)
        cash -= (trade_cost + commission) # Subtract trade cost and commission

        # update holdings
        shares_y = target_shares_y
        shares_x = target_shares_x

        # calc portfolio value at CLOSE prices
        price_y_mark = y_close.iloc[t]
        price_x_mark = x_close.iloc[t]

        current_portfolio_value = cash + (shares_y * price_y_mark) + (shares_x * price_x_mark)

        daily_pnl = current_portfolio_value - prev_portfolio_value
        pnl.append(daily_pnl)

        prev_portfolio_value = current_portfolio_value

        # add trade to logs
        if shares_y != 0 or delta_y != 0:
            trades_list.append({
                "date": current_date,
                "pnl": daily_pnl,
                "commission": commission,
                "position": current_position_state,
                "shares_y": shares_y,
                "shares_x": shares_x,
                "y_price_close": price_y_mark,
                "x_price_close": price_x_mark,
                "y_price_open": price_y_trade,
                "x_price_open": price_x_trade,
                "spread": spread.iloc[t],
                "portfolio_value": current_portfolio_value
            })

    return pd.Series(pnl, index=spread.index[1:]), pd.DataFrame(trades_list)
