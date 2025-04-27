"""Functions for processing and generating trading signals."""

from dataclasses import dataclass
from typing import Any
import talib
import pandas as pd

from core.chart import heiken_ashi_numpy
from core.calc import (
    entry_price,
    exit_total,
    take_profit,
    stop_loss as sl,
)

import numpy as np
from numpy.typing import NDArray
from numba import jit  # type: ignore

ASK_COLUMN = "ask_close"
BID_COLUMN = "bid_close"


@dataclass
class KernelConfig:
    """A dataclass containing the configuration for the kernel."""

    signal_buy_column: str
    signal_exit_column: str
    source_column: str
    wma_period: int = 20
    take_profit: float = 0
    stop_loss: float = 0

    def __str__(self):
        """Return a string representation of the SignalConfig object."""
        return f"so:{self.source_column}, sib:{self.signal_buy_column}, sie:{self.signal_exit_column}, sl:{self.stop_loss}, tp:{self.take_profit}"


@jit(nopython=True)
def wma_signals(
    buy_data: NDArray[Any],
    exit_data: NDArray[Any],
    wma_data: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Generate trading signals based on a comparison of the Heikin-Ashi highs and lows to the wma."""
    signals = np.zeros(len(buy_data)).astype(np.int64)


    # Generate signals using numpy
    buy_signals = buy_data > wma_data
    exit_signals = exit_data < wma_data

    for i in range(1, len(signals)):
        if buy_signals[i]:
            signals[i] = 1
        elif exit_signals[i] and signals[i - 1] != 1:
            signals[i] = 0

    trigger = np.diff(signals).astype(np.int64)
    trigger = np.concatenate((np.zeros(1) , trigger))

    return signals, trigger


def kernel(
    df: pd.DataFrame,
    include_incomplete: bool,
    config: KernelConfig,
) -> pd.DataFrame:
    """Process a DataFrame containing trading data.

    This function processes a DataFrame containing trading data and generate trading signals
    using candlesticks and weighted moving average (wma).

    TODO: support other pipelines

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trading data.
    include_incomplete:
        Whether to include the last candle in the output DataFrame.
    config : KernelConfig
        A dataclass containing the configuration for the kernel.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed trading data.

    """
    if not include_incomplete:
        df = df.iloc[:-1].copy()
    else:
        df = df.copy()

    # calculate the Heikin-Ashi candlesticks
    df["ha_open"], df["ha_high"], df["ha_low"], df["ha_close"] = heiken_ashi_numpy(
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )

    # calculate the Heikin-Ashi candlesticks for the bid prices
    df["ha_bid_open"], df["ha_bid_high"], df["ha_bid_low"], df["ha_bid_close"] = (
        heiken_ashi_numpy(
            df["bid_open"].to_numpy(),
            df["bid_high"].to_numpy(),
            df["bid_low"].to_numpy(),
            df["bid_close"].to_numpy(),
        )
    )

    # calculate the Heikin-Ashi candlesticks for the ask prices
    df["ha_ask_open"], df["ha_ask_high"], df["ha_ask_low"], df["ha_ask_close"] = (
        heiken_ashi_numpy(
            df["ask_open"].to_numpy(),
            df["ask_high"].to_numpy(),
            df["ask_low"].to_numpy(),
            df["ask_close"].to_numpy(),
        )
    )
    
    # calculate the ATR for the trailing stop loss
    df["atr"] = talib.ATR(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        timeperiod=config.wma_period,
    )
    df["wma"] = talib.WMA(df[config.source_column].to_numpy(), config.wma_period)

    # signal using the close prices
    # signal and trigger interval could appears as this:
    # 0 0 1 1 1 0 0 - 1 above or 0 below the wma
    # 0 0 1 0 0 -1 0 - diff gives actual trigger
    # NOTE: usage of close prices differs online than in offline trading
    df["signal"], df["trigger"]= wma_signals(
        df[config.signal_buy_column].to_numpy(),
        df[config.signal_exit_column].to_numpy(),
        df["wma"].to_numpy(),
    )

    # calculate the entry prices:
    df["internal_bit_mask"], df["entry_price"], df["position_value"] = (
        entry_price(
            df[ASK_COLUMN].to_numpy(),
            df[BID_COLUMN].to_numpy(),
            df["signal"].to_numpy(),
            df["trigger"].to_numpy(),
        )
    )

    # for internally managed take profits
    if config.take_profit > 0:
        df["signal"], df["trigger"] = take_profit(
            df["position_value"].to_numpy(),
            df["atr"].to_numpy(),
            df["signal"].to_numpy(),
            config.take_profit,
        )
        df["internal_bit_mask"], df["entry_price"], df["position_value"] = (
            entry_price(
                df[ASK_COLUMN].to_numpy(),
                df[BID_COLUMN].to_numpy(),
                df["signal"].to_numpy(),
                df["trigger"].to_numpy(),
            )
        )

    if config.stop_loss > 0:
        df["signal"], df["trigger"] = sl(
            df["position_value"].to_numpy(),
            df["atr"].to_numpy(),
            df["signal"].to_numpy(),
            config.stop_loss,
        )
        df["internal_bit_mask"], df["entry_price"], df["position_value"] = (
            entry_price(
                df[ASK_COLUMN].to_numpy(),
                df[BID_COLUMN].to_numpy(),
                df["signal"].to_numpy(),
                df["trigger"].to_numpy(),
            )
        )

    # calculate the exit total
    exit_total(df)

    return df
