"""Functions for calculating trading signals."""

import pandas as pd
import numpy as np
import talib

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


def take_profit(df: pd.DataFrame, take_profit: float) -> None:
    """Apply a take profit strategy to the trading data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    take_profit : float
        The take profit value as a multiplier of the atr.
    entry_column : str
        The column name for the entry price.
    exit_column : str
        The column name for the exit price.

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'signal' and 'trigger' columns updated.

    Notes
    -----
    The 'signal' column is set to 0 where the 'value' column is greater than the 'atr'
    column times the take profit value. The 'trigger' column is set to the difference
    between the 'signal' and the previous value of the 'signal' column. The entry_price
    is then re-calculated.

    """
    df["take_profit"] = take_profit * df["atr"]
    df.loc[
        (df["position_value"] > df["take_profit"]) & (df["trigger"] != 1), "signal"
    ] = 0
    df["trigger"] = df["signal"].diff().fillna(0).astype(int)


def stop_loss(df: pd.DataFrame, stop_loss: float) -> None:
    """Apply a stop loss strategy to the trading data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    stop_loss : float
        The stop loss value as a multiplier of the atr.
    entry_column : str
        The column name for the entry price.
    exit_column : str
        The column name for the exit price.

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'signal' and 'trigger' columns updated.

    Notes
    -----
    The 'signal' column is set to 0 where the 'value' column is greater than the 'atr'
    column times the stop loss value. The 'trigger' column is set to the difference
    between the 'signal' and the previous value of the 'signal' column. The entry_price
    is then re-calculated.

    """
    df["stop_loss"] = stop_loss * df["atr"]
    df.loc[(df["position_value"] < df["stop_loss"]) & (df["trigger"] != 1), "signal"] = 0
    df["trigger"] = df["signal"].diff().fillna(0).astype(int)


def entry_price(df: pd.DataFrame) -> None:
    """Calculate the entry price for a given trading signal.

    When the trigger is 1, use the ask_open price (buy signal), so when the trigger is -1, it means
    the buy signal is no longer valid, and the bid_open price at this point would be the same as
    the bid_close price at the previous point (where the signal was 1) since the closing price is
    trailing we can use the bid_open in the immediate next row.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    entry_column : str
        The column name for the entry price.
    exit_column : str
        The column name for the exit price.

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'internal_bit_mask', 'entry_price', and 'position_value' columns updated.

    Notes
    -----
    The 'internal_bit_mask' column is set to the bit-wise OR of the 'signal' and 'trigger'
    columns. The 'entry_price' column is set to the 'ask_high' column when the 'trigger'
    column is 1, and the 'value' column is set to the difference between the 'bid_open'
    and 'entry_price' columns times the 'internal_bit_mask' column.

    """
    df["internal_bit_mask"] = df["signal"] | abs(df["trigger"])
    df["entry_price"] = np.where(df["trigger"] == 1, df[ASK_COLUMN], np.nan)
    df["entry_price"] = df["entry_price"].ffill() * df["internal_bit_mask"]
    df["position_value"] = (df[BID_COLUMN] - df["entry_price"]) * df[
        "internal_bit_mask"
    ]


def atr(df: pd.DataFrame, wma_period: int) -> None:
    """Calculate the Average True Range for a given DataFrame and add it to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    wma_period : int
        The period for the weighted moving average.

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'atr' column added.

    Notes
    -----
    The 'atr' column is the Average True Range of the trading data, calculated using
    the TA-Lib library.

    """
    df["atr"] = talib.ATR(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        timeperiod=wma_period,
    )
