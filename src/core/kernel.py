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

EDGE = False

ASK_COLUMN = "ask_close"
BID_COLUMN = "bid_open" if EDGE else "bid_close"


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
    """Generate trading signals based on a comparison highs and lows to the wma."""
    signals = np.where(buy_data > wma_data, 1, 0)
    trigger = np.diff(signals).astype(np.int64)
    trigger = np.concatenate((np.zeros(1), trigger))

    signals = np.where((exit_data < wma_data) & (trigger != 1), 0, signals)
    trigger = np.diff(signals).astype(np.int64)
    trigger = np.concatenate((np.zeros(1), trigger))

    return signals, trigger


@jit(nopython=True)
def kernel_stage_1(  # noqa: PLR0913
    buy_data: NDArray[Any],
    exit_data: NDArray[Any],
    wma_data: NDArray[Any],
    ask_data: NDArray[Any],
    bid_data: NDArray[Any],
    atr: NDArray[Any],
    take_profit_conf: float,
    stop_loss_conf: float,
):
    """Perform the first stage of the kernel.

    This function takes in arrays of high and low prices, a weighted moving average
    (wma) array, ask and bid prices, average true range (atr), and take profit and
    stop loss values. It then generates trading signals using the wma and prices,
    calculates the entry prices, and applies any take profit or stop loss strategies.

    Parameters
    ----------
    buy_data : NDArray[Any]
        The array of high prices.
    exit_data : NDArray[Any]
        The array of low prices.
    wma_data : NDArray[Any]
        The array of weighted moving average (wma) values.
    ask_data : NDArray[Any]
        The array of ask prices.
    bid_data : NDArray[Any]
        The array of bid prices.
    atr : NDArray[Any]
        The array of average true range (atr) values.
    take_profit_conf : float
        The take profit value as a multiplier of the atr.
    stop_loss_conf : float
        The stop loss value as a multiplier of the atr.

    Returns
    -------
    tuple[NDArray[Any], NDArray[Any], NDArray[Any]]
        A tuple containing the signal, trigger, and position value arrays.

    """
    # signal using the close prices
    # signal and trigger interval could appears as this:
    # 0 0 1 1 1 0 0 - 1 above or 0 below the wma
    # 0 0 1 0 0 -1 0 - diff gives actual trigger
    # NOTE: usage of close prices differs online than in offline trading
    signal, trigger = wma_signals(
        buy_data,
        exit_data,
        wma_data,
    )

    # calculate the entry prices:
    _, _, position_value = entry_price(
        ask_data,
        bid_data,
        signal,
        trigger,
    )

    # for internally managed take profits
    if take_profit_conf > 0:
        signal, trigger = take_profit(
            position_value,
            atr,
            signal,
            trigger,
            take_profit_conf,
        )
        _, _, position_value = entry_price(
            ask_data,
            bid_data,
            signal,
            trigger,
        )

    if stop_loss_conf > 0:
        signal, trigger = sl(
            position_value,
            atr,
            signal,
            stop_loss_conf,
        )
        _, _, position_value = entry_price(
            ask_data,
            bid_data,
            signal,
            trigger,
        )

    return signal, trigger, position_value


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

    if (
        "ha" in config.signal_buy_column
        or "ha" in config.signal_exit_column
        or "ha" in config.source_column
    ):
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

    # calculate the entry and exit signals
    df["signal"], df["trigger"], df["position_value"] = kernel_stage_1(
        df[config.signal_buy_column].to_numpy(),
        df[config.signal_exit_column].to_numpy(),
        df["wma"].to_numpy(),
        df[ASK_COLUMN].to_numpy(),
        df[BID_COLUMN].to_numpy(),
        df["atr"].to_numpy(),
        config.take_profit,
        config.stop_loss,
    )

    # calculate the exit total
    exit_total(df)

    return df
