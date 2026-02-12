import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from config import MIN_COINT_OBS, PVAL_THRESHOLD
import warnings
from typing import List, Tuple

def find_pairs(prices: pd.DataFrame, tickers: List[str], pval_threshold: float = PVAL_THRESHOLD) -> List[Tuple[str, str]]:
    """
    Returns a list of candidate pairs that pass the cointegration test.
    """
    pairs: List[Tuple[str, str]] = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            s1, s2 = tickers[i], tickers[j]

            # clean and check length of series before cointegration test
            y_series: pd.Series = prices[s1].dropna()
            x_series: pd.Series = prices[s2].dropna()

            # align series to ensure they have the same index and length
            aligned_y, aligned_x = y_series.align(x_series, join='inner')

            # drop any nans that might occur due to alignment and ensure they have enough observations
            aligned_y = aligned_y.dropna()
            aligned_x = aligned_x.dropna()

            if len(aligned_y) < MIN_COINT_OBS or len(aligned_x) < MIN_COINT_OBS:
                warnings.warn(f"Cointegration test for {s1}-{s2} skipped: Insufficient observations after dropping NaNs.")
                continue

            try:
                _, pval, _ = coint(aligned_y, aligned_x) # Use aligned series
                if pval < pval_threshold:
                    pairs.append((s1, s2))
            except ValueError as e:
                warnings.warn(f"Cointegration test for {s1}-{s2} failed due to ValueError: {e}")
                continue
            except Exception as e:
                warnings.warn(f"Cointegration test for {s1}-{s2} failed: {e}")
                continue
    return pairs
