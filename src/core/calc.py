"""Functions for calculating trading signals."""

from typing import Any
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from numba import jit  # type: ignore

ASK_COLUMN = "ask_close"
BID_COLUMN = "bid_close"


def exit_total(df: pd.DataFrame) -> None:
    """Calculate the cumulative total of all trades and the running total of the portfolio.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'exit_total' and 'running_total' columns added.

    Notes
    -----
    The 'exit_total' column is the cumulative total of all trades, and the 'running_total' column
    is the cumulative total of the portfolio, including the current trade.

    """
    df["exit_value"] = df["position_value"] * ((df["trigger"] == -1).astype(int))
    df["exit_total"] = df["exit_value"].cumsum()
    df["running_total"] = df["exit_total"] + (df["position_value"] * df["signal"])
    df["wins"] = (df["exit_value"] > 0).astype(int).cumsum()
    df["losses"] = (df["exit_value"] < 0).astype(int).cumsum()
    df["min_exit_total"] = df["exit_total"].expanding().min()


def take_profit(
    position_value: NDArray[Any], 
    atr: NDArray[Any], 
    signal: NDArray[Any], 
    take_profit_value: float
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Apply a take profit strategy to trading signals.

    Parameters
    ----------
    position_value : np.ndarray
        The array of position values.
    atr : np.ndarray
        The array of average true range (atr).
    signal : np.ndarray
        The array of trading signals.
    take_profit_value : float
        The take profit value as a multiplier of the atr.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the updated 'signal' and 'trigger' arrays.

    Notes
    -----
    The 'signal' array is set to 0 where the 'position_value' array is greater than the
    'atr' array times the take profit value. The 'trigger' array is set to the difference
    between the 'signal' array and the previous value of the 'signal' array.

    """
    signal = np.where(position_value > atr * take_profit_value, 0, signal)
    trigger = np.diff(signal).astype(int)
    trigger = np.concatenate([[0], trigger])
    return signal, trigger


@jit(nopython=True)
def stop_loss(
    position_value: NDArray[Any], 
    atr: NDArray[Any], 
    signal: NDArray[Any], 
    stop_loss_value: float
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Apply a stop loss strategy to trading signals.

    This function takes arrays of position values, average true range (atr), signals, 
    and a stop loss multiplier to determine when to stop out of trades. If the position 
    value falls below the stop loss threshold, the signal is set to 0. The function 
    calculates the trigger as the difference between consecutive signal values.

    Parameters
    ----------
    position_value : NDArray[Any]
        Numpy array containing the position values.
    atr : NDArray[Any]
        Numpy array containing the average true range values.
    signal : NDArray[Any]
        Numpy array containing the trading signals.
    stop_loss_value : float
        The stop loss value as a multiplier of the atr.

    Returns
    -------
    tuple[NDArray[Any], NDArray[Any]]
        A tuple containing the updated signal and the trigger arrays.

    """
    stop_loss_array = stop_loss_value * atr
    for i in range(len(position_value)):
        if position_value[i] < stop_loss_array[i]:
            signal[i] = 0
    trigger = np.diff(signal).astype(np.int64)
    trigger = np.concatenate((np.zeros(1) , trigger))
    return signal, trigger


@jit(nopython=True)
def forward_fill(arr: NDArray) -> NDArray:
    """Forward fills NaN values in a 1D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        A 1D NumPy array containing potentially NaN values.

    Returns
    -------
        np.ndarray: A new 1D NumPy array with NaN values forward filled.

    """
    last_valid = np.nan
    result = np.empty_like(arr)
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            last_valid = arr[i]
        result[i] = last_valid
    return result


@jit(nopython=True)
def entry_price(
    entry: NDArray[Any], exit: NDArray[Any], signal: NDArray[Any], trigger: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Calculate the entry price for a given trading signal.

    Parameters
    ----------
    entry : np.ndarray
        The entry price array.
    exit : np.ndarray
        The exit price array.
    signal : np.ndarray
        The signal array.
    trigger : np.ndarray
        The trigger array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the entry price, exit price, and position value arrays.

    """
    internal_bit_mask = np.logical_or(signal, trigger)
    entry_price = np.where(trigger == 1, entry, np.nan)
    entry_price = forward_fill(entry_price) * internal_bit_mask
    position_value = (exit - entry_price) * internal_bit_mask

    return internal_bit_mask, entry_price, position_value



