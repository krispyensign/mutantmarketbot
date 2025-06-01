"""Functions for processing and generating trading signals."""

from dataclasses import dataclass
from functools import cached_property
from typing import Any
import pandas as pd

from core.calc import (
    entry_price,
    take_profit,
    stop_loss as sl,
)

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from numba import jit  # type: ignore

USE_QUASI = True
USE_EXIT_BOUND = True


class EdgeCategory(Enum):
    """Enumeration class for edge categories."""

    Quasi = 2
    Deterministic = 4


@dataclass
class KernelConfig:
    """A dataclass containing the configuration for the kernel."""

    signal_buy_column: str = ""
    signal_exit_column: str = ""
    source_column: str = ""
    wma_period: int = 20
    take_profit: float = 0.0
    stop_loss: float = 0.0

    @cached_property
    def edge(self) -> EdgeCategory:
        """Return the edge of the kernel.

        Returns
        -------
        float
            The edge of the kernel.

        """
        if "open" in self.signal_exit_column and USE_QUASI:
            return EdgeCategory.Quasi

        return EdgeCategory.Deterministic

    @cached_property
    def ask_column(self) -> str:
        """Return the name of the column in the DataFrame for the ask prices.

        Returns
        -------
        str
            The name of the column in the DataFrame for the ask prices.

        """
        return "ask_close"

    @cached_property
    def bid_column(self) -> str:
        """Return the name of the column in the DataFrame for the bid prices.

        Returns
        -------
        str
            The name of the column in the DataFrame for the bid prices.

        """
        if self.edge == EdgeCategory.Deterministic:
            return "bid_close"
        else:
            return "bid_open"

    def __str__(self) -> str:
        """Return a string representation of the SignalConfig object."""
        return f"edge:{self.edge}, so:{self.source_column}, sib:{self.signal_buy_column}, sie:{self.signal_exit_column}, sl:{self.stop_loss}, tp:{self.take_profit}"


@jit(nopython=True)  # type: ignore
def wma_exit_signals(
    buy_data: NDArray[np.float64],
    exit_data: NDArray[np.float64],
    wma_data: NDArray[np.float64],
    should_roll: np.bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Calculate the weighted moving average."""
    if should_roll:
        np.roll(wma_data, 1)
        wma_data[0] = np.nan

    # if USE_QUASI or USE_EXIT_BOUND:
    signals = np.zeros(len(buy_data)).astype(np.bool)
    buy_signals = np.where(buy_data > wma_data, np.True_, np.False_)
    exit_signals = np.where(exit_data > wma_data, np.True_, np.False_)
    for i in range(1, len(buy_signals)):
        An1 = buy_signals[i - 1]
        An = buy_signals[i]
        B = exit_signals[i]
        signals[i] = not An1 and An or An1 and B
    # else:
    #     signals = np.where(buy_data > wma_data, 1, 0)
    #     trigger = np.diff(signals.astype(np.int64))
    #     trigger = np.concatenate((np.zeros(1), trigger))

    #     signals = np.where((exit_data < wma_data) & (trigger != 1), 0, signals)
    trigger = np.diff(signals.astype(np.int64))
    trigger = np.concatenate((np.zeros(1), trigger))

    return signals.astype(np.int64), trigger.astype(np.int64)


@jit(nopython=True)  # type: ignore
def wma_signals_no_exit(
    buy_data: NDArray[np.float64],
    wma_data: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Calculate the weighted moving average."""
    signals = np.where(buy_data > wma_data, 1, 0)
    trigger = np.diff(signals)
    trigger = np.concatenate((np.zeros(1), trigger))

    return signals.astype(np.int64), trigger.astype(np.int64)


@jit(nopython=True)  # type: ignore
def kernel_stage_1(
    buy_data: NDArray[Any],
    exit_data: NDArray[Any],
    wma_data: NDArray[Any],
    ask_data: NDArray[Any],
    bid_data: NDArray[Any],
    bid_high_data: NDArray[Any],
    bid_low_data: NDArray[Any],
    atr: NDArray[Any],
    take_profit_conf: np.float64,
    stop_loss_conf: np.float64,
    use_exit: np.bool,
    should_roll: np.bool,
    erase: np.bool,
    digits: np.int64,
) -> tuple[
    NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]
]:
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
    use_exit : bool
        Whether to use exit data or not.
    erase: bool
        Whether to erase trades or not.
    should_roll: bool
        Whether to roll the wma or not.

    Returns
    -------
    tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]
        A tuple containing the signal, trigger, position, exit value, exit total, and running total arrays.

    """
    # signal using the close prices
    # signal and trigger interval could appears as this:
    # 0 0 1 1 1 0 0 - 1 above or 0 below the wma
    # 0 0 1 0 0 -1 0 - diff gives actual trigger
    # NOTE: usage of close prices differs online than in offline trading
    if use_exit:
        signal, trigger = wma_exit_signals(
            buy_data,
            exit_data,
            wma_data,
            should_roll,
        )
    else:
        signal, trigger = wma_signals_no_exit(buy_data, wma_data)

    # calculate the entry prices:
    position_value, position_high_value, position_low_value, entry_atr = entry_price(
        ask_data,
        bid_data,
        bid_high_data,
        bid_low_data,
        atr,
        signal,
        trigger,
    )

    # for internally managed take profits
    if take_profit_conf > 0:
        signal, trigger, tp_array = take_profit(
            position_high_value,
            entry_atr,
            signal,
            take_profit_conf,
            trigger,
            digits,
        )
        position_value, position_high_value, position_low_value, entry_atr = (
            entry_price(
                ask_data,
                bid_data,
                bid_high_data,
                bid_low_data,
                atr,
                signal,
                trigger,
            )
        )
        position_value = np.where(
            position_high_value > tp_array, tp_array, position_value
        )

    if stop_loss_conf > 0:
        signal, trigger, sl_array = sl(
            position_low_value,
            entry_atr,
            signal,
            stop_loss_conf,
            trigger,
            digits,
        )
        position_value, position_high_value, position_low_value, entry_atr = (
            entry_price(
                ask_data,
                bid_data,
                bid_high_data,
                bid_low_data,
                atr,
                signal,
                trigger,
            )
        )
        position_value = np.where(
            position_low_value < sl_array, sl_array, position_value
        )

    if erase:
        for i in range(3, len(signal)):
            if signal[i - 2] == 0 and signal[i - 1] == 1 and signal[i - 0] == 0:
                signal[i - 1] = 0
        trigger = np.diff(signal)
        trigger = np.concatenate((np.zeros(1), trigger)).astype(np.int64)
        position_value, position_high_value, position_low_value, entry_atr = (
            entry_price(
                ask_data,
                bid_data,
                bid_high_data,
                bid_low_data,
                atr,
                signal,
                trigger,
            )
        )

    exit_value = np.where(trigger == -1, position_value, 0)
    et = np.cumsum(exit_value)
    running_total = et + position_value * signal

    return signal, trigger, position_value, exit_value, et, running_total


def kernel(
    df: pd.DataFrame,
    config: KernelConfig,
    digits: np.int64,
) -> pd.DataFrame:
    """Process a DataFrame containing trading data.

    This function processes a DataFrame containing trading data and generate trading signals
    using candlesticks and weighted moving average (wma).

    TODO: support other pipelines

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing trading data.
    config : KernelConfig
        A dataclass containing the configuration for the kernel.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed trading data.

    """
    # calculate the entry and exit signals
    df["wma"] = df[f"wma_{config.source_column}"]
    should_roll = (
        "open" not in config.source_column and config.edge != EdgeCategory.Deterministic
    )
    (
        df["signal"],
        df["trigger"],
        df["position_value"],
        df["exit_value"],
        df["exit_total"],
        df["running_total"],
    ) = kernel_stage_1(
        df[config.signal_buy_column].to_numpy(),
        df[config.signal_exit_column].to_numpy(),
        df["wma"].to_numpy(),
        df[config.ask_column].to_numpy(),
        df[config.bid_column].to_numpy(),
        df["bid_high"].to_numpy(),
        df["bid_low"].to_numpy(),
        df["atr"].to_numpy(),
        config.take_profit,
        config.stop_loss,
        config.signal_buy_column != config.signal_exit_column,
        should_roll,
        config.edge == EdgeCategory.Quasi,
        digits,
    )

    return df
