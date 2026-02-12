import numpy as np
import pandas as pd
from typing import Optional

# compute the Hurst Exponent using Rescaled Range (R/S) analysis
# H < 0.5 -> Mean-reverting
# H = 0.5 -> Random Walk
# H > 0.5 -> Trending
def compute_hurst(series: pd.Series, max_lags: int = 100) -> Optional[float]:

    if not isinstance(series, pd.Series) or series.empty:
        return None

    series = series.dropna()
    if len(series) < 20:
        return None

    try:
        max_lags = min(max_lags, len(series) // 2)
        lags = range(2, max_lags)

        # calculate rescaled range
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

        # linear regression
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0]

    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as e:
        return None
