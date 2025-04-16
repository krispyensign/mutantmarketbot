"""Functions for processing and generating trading signals."""

from dataclasses import dataclass
import talib

from core.chart import heikin_ashi
from core.calc import entry_price, exit_total, take_profit, atr, trailing_stop_loss
import pandas as pd


def wma_signals(
    df: pd.DataFrame,
    source_column: str = "open",
    signal_buy_column: str = "bid_low",
    wma_period: int = 20,
) -> None:
    """Generate trading signals based on a comparison of the Heikin-Ashi highs and lows to the wma.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the trading data.
    first_time_frame_run : bool, optional
        Whether this is the first time frame run.
    source_column : str, optional
        The column name for the source data.
    signal_buy_column : str, optional
        The column name for the buy signal data.
    wma_period : int, optional
        The period for the weighted moving average, by default 20

    Returns
    -------
    pd.Dataframe
        The DataFrame with the 'signal' column added.

    Notes
    -----
    The 'signal' column is added to the DataFrame and is set to 1 where the Heikin-Ashi high
    is greater than the wma, and 0 where the Heikin-Ashi low is less than the wma.

    """
    df["signal"] = 0
    df["wma"] = talib.WMA(df[source_column].to_numpy(), wma_period)

    # check if the buy column is greater than the wma
    # B > W  S < W  Result
    # T      T      X
    # T      F      Buy
    # F      T      Exit
    # F      F      X

    # check if the buy column is greater than the wma
    df.loc[df[signal_buy_column] > df["wma"], "signal"] = 1
    df["trigger"] = df["signal"].diff().fillna(0).astype(int)


@dataclass
class KernelConfig:
    """A dataclass containing the configuration for the kernel."""

    signal_buy_column: str
    source_column: str
    entry_column: str
    exit_column: str
    wma_period: int = 20
    take_profit: float = 0
    stop_loss: float = 0


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

    # calculate the ATR for the trailing stop loss
    heikin_ashi(df)
    atr(df, config.wma_period)

    # signal using the close prices
    # signal and trigger interval could appears as this:
    # 0 0 1 1 1 0 0 - 1 above or 0 below the wma
    # 0 0 1 0 0 -1 0 - diff gives actual trigger
    # NOTE: usage of close prices differs online than in offline trading
    wma_signals(
        df,
        signal_buy_column=config.signal_buy_column,
        wma_period=config.wma_period,
        source_column=config.source_column,
    )

    # calculate the entry prices:
    entry_price(df, entry_column=config.entry_column, exit_column=config.exit_column)

    # recalculate the entry prices after a take profit
    # for internally managed take profits
    if config.take_profit > 0:
        take_profit(
            df,
            config.take_profit,
            entry_column=config.entry_column,
            exit_column=config.exit_column,
        )

    if config.stop_loss > 0:
        trailing_stop_loss(
            df,
            config.stop_loss,
            entry_column=config.entry_column,
            exit_column=config.exit_column,
        )

    # calculate the exit total
    exit_total(df)

    return df
