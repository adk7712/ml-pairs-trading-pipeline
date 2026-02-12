import numpy as np
import pandas as pd
import warnings
from typing import Optional

# estimate half-life of mean reversion using Ornstein-Uhlenbeck approximation.
def compute_half_life(spread: pd.Series) -> Optional[float]:

    delta: pd.Series = spread.diff().dropna()
    lagged: pd.Series = spread.shift(1).dropna()
    lagged, delta = lagged.align(delta, join='inner')

    if len(lagged) < 2:
        return None
    try:
        # np.polyfit returns array, take the first element (slope)
        beta_val: float = np.polyfit(lagged, delta, 1)[0]
        if beta_val == 0:
            return None
        half_life: float = -np.log(2) / beta_val
        if half_life <= 0:
            return None
        return half_life
    except np.linalg.LinAlgError as e:
        warnings.warn(f"Half-life calculation failed due to linear algebra error: {e}")
        return None # calculation failed
    except Exception as e:
        warnings.warn(f"Half-life calculation failed: {e}")
        return None
