"""Functions for reporting trading results."""

from datetime import timedelta
import pandas as pd
import logging

from core.kernel import KernelConfig


def report(
    df: pd.DataFrame,
    instrument: str,
    kernel_conf: KernelConfig,
    length: int = 3,
):
    """Print a report of the trading results.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    instrument : str
        The instrument being traded.
    kernel_conf : KernelConfig
        The kernel configuration.
    length : int, optional
        The number of rows to print, by default 2

    """
    logger = logging.getLogger("reporting")
    df_ticks = df.reset_index()[
        [
            "signal",
            "trigger",
            "atr",
            "wma",
            kernel_conf.signal_buy_column,
            kernel_conf.signal_exit_column,
            kernel_conf.ask_column,
            kernel_conf.bid_column,
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
    # recent_ticks = df_ticks.where(df_ticks["timestamp"] > APP_START_TIME)
    # logger.info("total %s", recent_ticks["exit_value"].sum())
    df_orders = df_ticks.copy()
    df_orders = df_orders[df_orders["trigger"] != 0]
    round_amount = 3 if "JPY" in instrument else 5
    logger.info("recent trades")
    logger.info(
        "\n"
        + df_orders.tail(length * 2)
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
