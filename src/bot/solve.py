"""Backtest the trading strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools
import subprocess
import numpy as np
import pandas as pd
import talib
import v20  # type: ignore
from core.chart import heiken_ashi_numpy
from numpy.typing import NDArray
from numba import jit  # type: ignore

from core.kernel import EdgeCategory, KernelConfig, kernel_stage_1
from bot.exchange import (
    getOandaOHLC,
    OandaContext,
)

import logging

APP_START_TIME = datetime.now()


def get_git_info() -> tuple[str, bool, Exception | None]:
    """Get commit hash and whether the working tree is clean.

    Returns a tuple (str, bool). The first element is the commit hash. The second
    element is a boolean indicating whether the working tree is clean (i.e., there
    are no pending changes).

    If there is an error (e.g., not in a Git repository), the function returns None.
    """
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()

        # Get porcelain status
        porcelain_status = subprocess.check_output(
            ["git", "status", "--porcelain"], encoding="utf-8"
        ).strip()

        return commit_hash, porcelain_status == "", None
    except subprocess.CalledProcessError as e:
        return "", False, e


class PerfTimer:
    """PerfTimer class."""

    def __init__(self, app_start_time: datetime, logger: logging.Logger):
        """Initialize a PerfTimer object."""
        self.app_start_time = app_start_time
        self.logger = logger
        pass

    def __enter__(self):
        """Start the timer."""
        self.start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer."""
        self.end = datetime.now()
        self.logger.info(f"run interval: {self.end - self.start}")
        self.logger.info("up time: %s", (self.end - self.app_start_time))
        self.logger.info("last run time: %s", self.end.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class ChartConfig:
    """ChartConfig class."""

    instrument: str
    granularity: str
    candle_count: int
    datefrom: datetime | None = None


@dataclass
class SolverConfig:
    """SolverConfig class."""

    take_profit: list[float]
    stop_loss: list[float]
    source_columns: list[str]
    verifier: str

    def get_column_pairs(self) -> tuple[itertools.product, int]:
        """Get column pairs."""
        return itertools.product(
            self.source_columns,
            self.source_columns,
            self.source_columns,
            self.take_profit,
            self.stop_loss,
        ), len(self.source_columns) ** 3 * len(self.take_profit) * len(self.stop_loss)


@dataclass
class BacktestResult:
    """BacktestResult class."""

    instrument: str
    kernel_conf: KernelConfig
    exit_total: np.float64


def preprocess(df: pd.DataFrame, wma_period: int, convert: bool) -> dict[str, NDArray] | pd.DataFrame:
    # calculate the ATR for the trailing stop loss
    """Preprocess the DataFrame to calculate various technical indicators.

    This function calculates the Average True Range (ATR), Weighted Moving Averages (WMA)
    for open, high, low, and close prices, and Heikin-Ashi candlesticks for both original
    and bid/ask prices in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing columns for open, high, low, close, ask, and bid prices.
    wma_period : int
        The period to be used for calculating the Weighted Moving Averages (WMA).

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional columns for ATR, WMA, and Heikin-Ashi
        candlesticks.

    """
    df["atr"] = talib.ATR(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        timeperiod=wma_period,
    )

    df["wma_open"] = talib.WMA(df["open"].to_numpy(), timeperiod=wma_period)
    df["wma_high"] = talib.WMA(df["high"].to_numpy(), timeperiod=wma_period)
    df["wma_low"] = talib.WMA(df["low"].to_numpy(), timeperiod=wma_period)
    df["wma_close"] = talib.WMA(df["close"].to_numpy(), timeperiod=wma_period)

    df["wma_ask_open"] = talib.WMA(df["ask_open"].to_numpy(), timeperiod=wma_period)
    df["wma_ask_high"] = talib.WMA(df["ask_high"].to_numpy(), timeperiod=wma_period)
    df["wma_ask_low"] = talib.WMA(df["ask_low"].to_numpy(), timeperiod=wma_period)
    df["wma_ask_close"] = talib.WMA(df["ask_close"].to_numpy(), timeperiod=wma_period)

    df["wma_bid_open"] = talib.WMA(df["bid_open"].to_numpy(), timeperiod=wma_period)
    df["wma_bid_high"] = talib.WMA(df["bid_high"].to_numpy(), timeperiod=wma_period)
    df["wma_bid_low"] = talib.WMA(df["bid_low"].to_numpy(), timeperiod=wma_period)
    df["wma_bid_close"] = talib.WMA(df["bid_close"].to_numpy(), timeperiod=wma_period)

    # calculate the Heikin-Ashi candlesticks
    df["ha_open"], df["ha_high"], df["ha_low"], df["ha_close"] = heiken_ashi_numpy(
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    df["wma_ha_open"] = talib.WMA(df["ha_open"].to_numpy(), timeperiod=wma_period)
    df["wma_ha_high"] = talib.WMA(df["ha_high"].to_numpy(), timeperiod=wma_period)
    df["wma_ha_low"] = talib.WMA(df["ha_low"].to_numpy(), timeperiod=wma_period)
    df["wma_ha_close"] = talib.WMA(df["ha_close"].to_numpy(), timeperiod=wma_period)

    # calculate the Heikin-Ashi candlesticks for the bid prices
    df["ha_bid_open"], df["ha_bid_high"], df["ha_bid_low"], df["ha_bid_close"] = (
        heiken_ashi_numpy(
            df["bid_open"].to_numpy(),
            df["bid_high"].to_numpy(),
            df["bid_low"].to_numpy(),
            df["bid_close"].to_numpy(),
        )
    )
    df["wma_ha_bid_open"] = talib.WMA(
        df["ha_bid_open"].to_numpy(), timeperiod=wma_period
    )
    df["wma_ha_bid_high"] = talib.WMA(
        df["ha_bid_high"].to_numpy(), timeperiod=wma_period
    )
    df["wma_ha_bid_low"] = talib.WMA(df["ha_bid_low"].to_numpy(), timeperiod=wma_period)
    df["wma_ha_bid_close"] = talib.WMA(
        df["ha_bid_close"].to_numpy(), timeperiod=wma_period
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
    df["wma_ha_ask_open"] = talib.WMA(
        df["ha_ask_open"].to_numpy(), timeperiod=wma_period
    )
    df["wma_ha_ask_high"] = talib.WMA(
        df["ha_ask_high"].to_numpy(), timeperiod=wma_period
    )
    df["wma_ha_ask_low"] = talib.WMA(df["ha_ask_low"].to_numpy(), timeperiod=wma_period)
    df["wma_ha_ask_close"] = talib.WMA(
        df["ha_ask_close"].to_numpy(), timeperiod=wma_period
    )

    if convert:
        result_dict: dict[str, NDArray[np.float64]] = {}
        for k, v in df.to_dict('series').items():
            result_dict[str(k)] = v.to_numpy()

        return result_dict

    return df


def _solve_run(
    kernel_conf_in: KernelConfig, config_tuple: tuple | None, ask_column: NDArray[np.float64], atr: NDArray[np.float64], df: dict
) -> tuple[KernelConfig, np.float64] | None:
    # run the backtest
    if config_tuple is not None:
        kernel_conf = _map_kernel_conf(kernel_conf_in, config_tuple)
    else:
        kernel_conf = kernel_conf_in
    wma = df[f"wma_{kernel_conf.source_column}"]
    (
        _,
        _,
        _,
        exit_value,
        exit_total,
        _,
    ) = kernel_stage_1(
        df[kernel_conf.signal_buy_column],
        df[kernel_conf.signal_exit_column],
        wma,
        ask_column,
        df[kernel_conf.bid_column],
        atr,
        kernel_conf.take_profit,
        kernel_conf.stop_loss,
        kernel_conf.signal_buy_column != kernel_conf.signal_exit_column,
        kernel_conf.edge == EdgeCategory.Quasi,
    )

    # filter invalid results
    if _is_invalid_scenario(exit_value, exit_total):
        return None

    return kernel_conf, exit_total[-1]


def solve(
    chart_config: ChartConfig,
    kernel_conf_in: KernelConfig,
    token: str,
    backtest_config: SolverConfig,
) -> tuple[BacktestResult, BacktestResult] | None:
    """Run a backtest of the trading strategy.

    Parameters
    ----------
    chart_config : ChartConfig
        The chart configuration.
    kernel_conf_in : KernelConfig
        The kernel configuration.
    token : str
        The Oanda API token.
    backtest_config : BacktestConfig
        The backtest configuration.

    Returns
    -------
    KernelConfig | None
        The best kernel configuration.

    Notes
    -----
    The backtest will run for a large number of combinations of source and signal
    columns. The best combination will be saved to best_df and the results will be
    printed to the log file.

    """
    logger = logging.getLogger("backtest")
    logger.info("starting backtest")
    commit, porcelain, err = get_git_info()
    if err is not None:
        logger.error("failed to get git info: %s", err)
        return None

    logger.info("git info: %s %s", commit, porcelain)

    # get data and preprocess
    orig_df: dict[str, NDArray[np.float64]] = preprocess(
        _get_data(chart_config, token, logger), kernel_conf_in.wma_period, True
    ) # type: ignore

    # get verifier data and preprocess
    verifier_orig_df: dict[str, NDArray[np.float64]] = preprocess(
        _get_data(chart_config, token, logger, backtest_config.verifier),
        kernel_conf_in.wma_period, True
    ) # type: ignore

    best_result: tuple[BacktestResult, BacktestResult] | None = None
    column_pairs, column_pair_len = backtest_config.get_column_pairs()
    logger.info(f"total_combinations: {column_pair_len}")
    total_found = 0
    found_results: list[BacktestResult] = []
    best_total = 0.0
    count = 0
    filter_start_time = datetime.now()

    df = orig_df
    atr = df["atr"]
    ask_column = df["ask_close"]
    for config_tuple in column_pairs:
        # log progress
        count = _log_progress(
            logger, column_pair_len, total_found, count, filter_start_time
        )

        # run
        result = _solve_run(kernel_conf_in, config_tuple, ask_column, atr, df)
        if result is None:
            continue

        # save result if valid
        total_found += 1
        kernel_conf, et = result
        found_results.append(
            BacktestResult(
                instrument=chart_config.instrument,
                kernel_conf=kernel_conf,
                exit_total=et,
            )
        )

    count = 0
    total_found = 0
    df = verifier_orig_df.copy()
    filter_start_time = datetime.now()
    atr = df["atr"]
    ask_column = df["ask_close"]
    filter_count = (
        len(found_results)
        * len(backtest_config.take_profit)
        * len(backtest_config.stop_loss)
    )
    gen = (
        (filter_result, tp, sl)
        for filter_result in found_results
        for tp in backtest_config.take_profit
        for sl in backtest_config.stop_loss
    )
    for f in gen:
        # log progress and filter invalid results
        count = _log_progress(
            logger, filter_count, total_found, count, filter_start_time
        )

        # run
        filter_result, tp, sl = f
        kernel_conf = KernelConfig(
            signal_buy_column=filter_result.kernel_conf.signal_buy_column,
            signal_exit_column=filter_result.kernel_conf.signal_exit_column,
            source_column=filter_result.kernel_conf.source_column,
            wma_period=kernel_conf_in.wma_period,
            take_profit=tp,
            stop_loss=sl,
        )
        result = _solve_run(kernel_conf, None, ask_column, atr, df)
        if result is None:
            continue

        kernel_conf, et = result
        total_found += 1
        total = et + filter_result.exit_total
        if total > best_total:
            logger.debug(
                f"found result: {kernel_conf.source_column} {kernel_conf.signal_buy_column} {kernel_conf.signal_exit_column} {kernel_conf.take_profit} {kernel_conf.stop_loss} {round(et, 5)} {round(total, 5)}")
            best_result = (
                filter_result,
                BacktestResult(
                    kernel_conf=kernel_conf,
                    exit_total=et,
                    instrument=backtest_config.verifier,
                ),
            )
            best_total = total

    logger.info("total_found: %s", total_found)
    if total_found == 0:
        logger.error("no combinations found")
        return None

    return best_result


def _log_progress(
    logger: logging.Logger,
    column_pair_len: int,
    total_found: int,
    count: int,
    start_time: datetime,
) -> int:
    if count == 0:
        logger.info("starting pass")
    count += 1
    if count % 10000 == 0:
        time_now = datetime.now()
        time_diff = time_now - start_time
        throughput = count / time_diff.total_seconds()
        remaining = timedelta(seconds=(column_pair_len - count) / throughput)
        logger.info(
            "heartbeat: %s %s%% %s/%s %s/s %s remaining",
            total_found,
            round(100 * count / column_pair_len, 2),
            count,
            column_pair_len,
            round(throughput, 2),
            remaining,
        )
    return count


def _get_data(
    chart_config: ChartConfig,
    token: str,
    logger: logging.Logger,
    instrument: str | None = None,
) -> pd.DataFrame:
    """Get data from Oanda and return it as a DataFrame.

    Parameters
    ----------
    chart_config : ChartConfig
        The configuration for the chart.
    token : str
        The Oanda API token.
    logger : logging.Logger
        The logger to use.
    instrument : str | None, optional
        The instrument to get data for. The default is None, which means to use the instrument from the chart config.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the data.

    """
    ctx = OandaContext(
        v20.Context("api-fxpractice.oanda.com", token=token),
        None,
        token,
        chart_config.instrument if instrument is None else instrument,
    )

    orig_df = getOandaOHLC(
        ctx, count=chart_config.candle_count, granularity=chart_config.granularity
    )
    logger.info(
        "count: %s granularity: %s",
        chart_config.candle_count,
        chart_config.granularity,
    )

    return orig_df


def _map_kernel_conf(
    kernel_conf_in: KernelConfig, config_tuple: tuple[str, str, str, float, float]
) -> KernelConfig:
    kernel_conf = KernelConfig(
        wma_period=kernel_conf_in.wma_period,
        signal_buy_column=config_tuple[0],
        signal_exit_column=config_tuple[1],
        source_column=config_tuple[2],
        take_profit=config_tuple[3],
        stop_loss=config_tuple[4],
    )

    return kernel_conf


@jit(nopython=True)
def _is_invalid_scenario(
    exit_value: NDArray[np.float64], exit_total: NDArray[np.float64]
) -> np.bool:
    wins: np.float64 = np.where(exit_value > 0, 1, 0).astype(np.int64).sum()
    final_exit_total: np.float64 = exit_total[-1]
    return (
        wins == 0
        or final_exit_total < 0
        or (
            exit_value.min() < 0 and np.abs(exit_value.min()) > np.abs(final_exit_total)
        )
    )
