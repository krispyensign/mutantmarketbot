"""Functions for reporting trading results."""

from datetime import timedelta
import pandas as pd
import logging


logger = logging.getLogger("reporting")

ENTRY_COLUMN = "ask_close"
EXIT_COLUMN = "bid_close"


def report(
    df: pd.DataFrame,
    instrument: str,
    signal_buy_column: str,
    signal_exit_column: str,
    length: int = 2,
):
    """Print a report of the trading results.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    instrument : str
        The instrument being traded.
    signal_buy_column : str
        The column name for the buy signal data.
    signal_exit_column : str
        The column name for the exit signal data.
    length : int, optional
        The number of rows to print, by default 2

    """
    df_ticks = df.reset_index()[
        [
            "signal",
            "trigger",
            "atr",
            "wma",
            signal_buy_column,
            signal_exit_column,
            ENTRY_COLUMN,
            EXIT_COLUMN,
            "position_value",
            "exit_value",
            "running_total",
            "exit_total",
            "timestamp",
        ]
    ]
    df_ticks["timestamp"] = pd.to_datetime(df_ticks["timestamp"])
    df_ticks["completed_datetime"] = (
        (timedelta(minutes=5) + df_ticks["timestamp"]).dt
    ).strftime("%Y-%m-%d %H:%M:%S")
    df_orders = df_ticks.copy()
    df_orders = df_orders[df_orders["trigger"] != 0]
    round_amount = 3 if "JPY" in instrument else 5
    logger.info("recent trades")
    logger.info(
        "\n"
        + df_orders.tail(length)
        .round(round_amount)
        .to_string(index=False, header=True, justify="left")
    )
    logger.debug("current status")
    logger.debug(
        "\n"
        + df_ticks.tail(length)
        .round(round_amount)
        .to_string(index=False, header=True, justify="left")
    )
