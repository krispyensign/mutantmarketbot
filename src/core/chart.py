"""Functions for generating charts."""

import typing
import pandas as pd
import numpy as np
from numba import jit  # type: ignore
from typing import Any
from numpy.typing import NDArray


@jit(nopython=True)
def heiken_ashi_numpy(
    c_open: NDArray[Any],
    c_high: NDArray[Any],
    c_low: NDArray[Any],
    c_close: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Generate Heikin Ashi candlesticks for a given numpy arrays."""
    ha_close = (c_open + c_high + c_low + c_close) / 4
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (c_open[0] + c_close[0]) / 2
    for i in range(1, len(c_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum(np.maximum(ha_open, ha_close), c_high)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), c_low)
    return ha_open, ha_high, ha_low, ha_close


def ohlc(
    df: pd.DataFrame, timeFrame: str = "5Min", isSwapped: bool = False
) -> tuple[pd.DataFrame, typing.Any]:
    """Resample the input DataFrame into OHLC format for specified time intervals.

    This function takes a DataFrame containing raw trading data, calculates the
    price, bid price, and ask price based on the amounts, and then resamples
    these prices into Open-High-Low-Close (OHLC) format for the specified time
    intervals. The function handles both buy and sell scenarios to compute
    bid and ask prices.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame with columns 'amount0' and 'amount1' representing trading data.
    timeFrame : str, optional
        The time interval for resampling, by default '5Min'.
    isSwapped: bool, optional
        If True, swap the columns 'amount0' and 'amount1'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing resampled OHLC data for price, bid price, and ask price.

    """
    if isSwapped:
        df.rename(columns={"amount0": "amount1", "amount1": "amount0"}, inplace=True)

    # if amount0 is negative, then it is a buy so calculate the ask price
    # i.e. buy ETH sell BOBO
    # i.e. ETH = 0.15307 BOBO = 1040599933.89 price per ETH = 6,798,196,471.483635 BOBO
    df["ask_price"] = abs(df["amount1"] / df["amount0"]).where(df["amount0"] < 0)

    # if amount0 is positive, then it is a sell so calculate the bid price
    # i.e. sell ETH buy BOBO
    # i.e. ETH = 0.19187 BOBO = 1280811348.79 price per ETH = 6,675,412,251.993537 BOBO
    df["bid_price"] = abs(df["amount1"] / df["amount0"]).where(df["amount0"] > 0)

    df.ffill(inplace=True)
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2

    del df["amount0"]
    del df["amount1"]

    # resample to 5 minute intervals
    df_p = df["mid_price"].resample(timeFrame).ohlc()
    df_p.ffill(inplace=True)

    df_b = df["bid_price"].resample(timeFrame).ohlc()
    df_b.ffill(inplace=True)

    df_a = df["ask_price"].resample(timeFrame).ohlc()
    df_a.ffill(inplace=True)

    df_ohlc = pd.DataFrame()

    df_ohlc["open"] = df_p["open"]
    df_ohlc["high"] = df_p["high"]
    df_ohlc["low"] = df_p["low"]
    df_ohlc["close"] = df_p["close"]

    df_ohlc["bid_open"] = df_b["open"]
    df_ohlc["bid_high"] = df_b["high"]
    df_ohlc["bid_low"] = df_b["low"]
    df_ohlc["bid_close"] = df_b["close"]

    df_ohlc["ask_open"] = df_a["open"]
    df_ohlc["ask_high"] = df_a["high"]
    df_ohlc["ask_low"] = df_a["low"]
    df_ohlc["ask_close"] = df_a["close"]

    df_json = df_ohlc.reset_index()

    return df_ohlc, df_json.to_json(orient="records")
