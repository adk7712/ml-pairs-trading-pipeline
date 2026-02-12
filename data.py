import yfinance as yf
import pandas as pd
import warnings
from typing import List, Union, Tuple

def load_prices(tickers: Union[str, List[str]], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        raw_prices: pd.DataFrame = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

        if raw_prices.empty:
            return pd.DataFrame(), pd.DataFrame()

        # handle multi index vs single index
        if isinstance(raw_prices.columns, pd.MultiIndex):
            closes: pd.DataFrame = raw_prices['Close']
            opens: pd.DataFrame = raw_prices['Open']
        else:
            # single ticker
            closes = raw_prices[['Close']]
            opens = raw_prices[['Open']]

            # rename columns
            if isinstance(tickers, list) and len(tickers) == 1:
                closes.columns = tickers
                opens.columns = tickers
            elif isinstance(tickers, str):
                closes.columns = [tickers]
                opens.columns = [tickers]
            else:
                closes.columns = ['Close_Price']
                opens.columns = ['Open_Price']

    except Exception as e:
        raise ConnectionError(f"Failed to download prices for {tickers}: {e}. Please check your internet connection or ticker symbols.") from e

    # ffill missing values
    closes = closes.ffill()

    # drop columns that are entirely nan
    closes = closes.dropna(axis=1, how='all')

    # make sure opens match the valid columns and fill
    opens = opens.ffill()

    # only keep columns that survived the 'Close' dropna check to ensure matching pairs
    opens = opens[closes.columns]

    return closes, opens
