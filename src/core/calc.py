"""Functions for calculating trading signals."""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from numba import jit  # type: ignore


@jit(nopython=True)  # type: ignore
def take_profit(
    position_value: NDArray[Any],
    atr: NDArray[Any],
    signal: NDArray[Any],
    take_profit_value: float,
    trigger: NDArray[Any],
    digits: np.int64,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Apply a take profit strategy to trading signals.

    Parameters
    ----------
    position_value : np.ndarray
        The array of position values.
    atr : np.ndarray
        The array of average true range (atr).
    signal : np.ndarray
        The array of trading signals.
    trigger : np.ndarray
        The array of trigger values.
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
    take_profit_array = np.round(take_profit_value * atr, digits)
    signal = np.where((position_value > take_profit_array) & (trigger != 1), 0, signal)
    trigger = np.diff(signal)
    trigger = np.concatenate((np.zeros(1), trigger))
    return signal.astype(np.int64), trigger.astype(np.int64), take_profit_array


@jit(nopython=True)  # type: ignore
def stop_loss(
    position_value: NDArray[Any],
    atr: NDArray[Any],
    spread: NDArray[Any],
    signal: NDArray[Any],
    stop_loss_value: float,
    trigger: NDArray[Any],
    digits: np.int64,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
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
    trigger: NDArray[Any]
        Numpy array containing the triggers.
    stop_loss_value : float
        The stop loss value as a multiplier of the atr.

    Returns
    -------
    tuple[NDArray[Any], NDArray[Any]]
        A tuple containing the updated signal and the trigger arrays.

    """
    inc = 10 ** (-digits)
    stop_loss_array = np.round(-stop_loss_value * atr, digits)
    stop_loss_array = np.where( spread >= np.abs(stop_loss_array), - spread - inc, stop_loss_array) 
    signal = np.where((position_value < stop_loss_array) & (trigger != 1), 0, signal)
    trigger = np.diff(signal)
    trigger = np.concatenate((np.zeros(1), trigger))
    return signal.astype(np.int64), trigger.astype(np.int64), stop_loss_array


@jit(nopython=True)  # type: ignore
def forward_fill(arr: NDArray[Any]) -> NDArray[Any]:
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


@jit(nopython=True)  # type: ignore
def entry_price(
    entry: NDArray[np.float64],
    exit: NDArray[np.float64],
    exit_high: NDArray[np.float64],
    exit_low: NDArray[np.float64],
    atr: NDArray[Any],
    signal: NDArray[np.int64],
    trigger: NDArray[np.int64],
    spread: NDArray[np.float64], 
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Calculate the entry price for a given trading signal."""
    internal_bit_mask = np.logical_or(signal, trigger)
    entry_price = np.where(trigger == 1, entry, np.nan)
    entry_price = forward_fill(entry_price) * internal_bit_mask
    entry_spread = np.where(trigger == 1, spread, np.nan)
    entry_spread = forward_fill(entry_spread) * internal_bit_mask

    position_value = (exit - entry_price) * internal_bit_mask
    position_high_value = (exit_high - entry_price) * internal_bit_mask
    position_low_value = (exit_low - entry_price) * internal_bit_mask

    entry_atr = np.where(trigger == 1, atr, np.nan)
    entry_atr = forward_fill(entry_atr) * internal_bit_mask


    return position_value, position_high_value, position_low_value, entry_atr, entry_spread  # type: ignore
