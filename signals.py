import numpy as np
import pandas as pd
from typing import Optional
from config import STOP_LOSS_Z_THRESHOLD, STOP_EXIT_FACTOR

def compute_spread(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    return y - beta * x

def zscore(series: pd.Series, window: int) -> pd.Series:
    mean: pd.Series = series.rolling(window).mean()
    std: pd.Series = series.rolling(window).std()
    return (series - mean) / std

def generate_signals(spread: pd.Series, window: int, entry_z: float, exit_z: float, half_life: Optional[float] = None) -> pd.Series:
    z: pd.Series = zscore(spread, window)
    signal: pd.Series = pd.Series(0, index=z.index, dtype=int)

    current_position: int = 0 # init current trading position [-1 -> short, 0 -> flat, 1 -> long]
    days_in_position: int = 0 # track how many days the current position has been open for time-based stop loss

    # determine max holding period based on half-life
    max_days: Optional[int] = None
    if half_life is not None and half_life > 0:
        max_days = int(half_life * STOP_EXIT_FACTOR)

    # iterate through z-scores to generate trading signals
    for t in range(1, len(z)):
        # if Z-score is nan, maintain the last known position and update days_in_position if a position is held.
        if pd.isna(z.iloc[t]):
            signal.iloc[t] = current_position
            if current_position != 0:
                days_in_position += 1
            else:
                days_in_position = 0
            continue

        # update days in position for existing trades
        # increment if a position is currently open, else reset
        if current_position != 0:
            days_in_position += 1
        else:
            days_in_position = 0

        # time based stop loss
        # if a position has been held for longer than max_days, close the position
        if current_position != 0 and max_days is not None and days_in_position > max_days:
            current_position = 0 # close position
            days_in_position = 0 # reset counter
            signal.iloc[t] = 0 # signal to be flat
            continue # skip further signal generation for this day

        # z score stop loss
        # if the absolute z-score exceeds the STOP_LOSS_Z_THRESHOLD, close the position
        if current_position != 0 and abs(z.iloc[t]) > STOP_LOSS_Z_THRESHOLD:
            current_position = 0 # close position
            days_in_position = 0 # reset counter
            signal.iloc[t] = 0 # signal to be flat
            continue # skip further signal generation for this day

        # Normal entry/exit logic
        # if flat (current_position == 0):
        #   - short if z score is above entry_z (spread wide, expected to contract)
        #   - enter long if z score is below -entry_z (spread narrow, expected to expand)
        # if currently in a position (current_position != 0):
        #   - exit if absolute z score falls below exit_z (spread has mean-reverted sufficiently)
        if z.iloc[t] > entry_z and current_position == 0: # entry condition for short
            current_position = -1
            days_in_position = 0 # reset on new entry
        elif z.iloc[t] < -entry_z and current_position == 0: # entry condition for long
            current_position = 1
            days_in_position = 0 # reset on new entry
        elif abs(z.iloc[t]) < exit_z and current_position != 0: # exit condition
            current_position = 0
            days_in_position = 0 # reset on exit

        # update signal series with the determined current position for the day
        signal.iloc[t] = current_position
    return signal
