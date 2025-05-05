"""Backtest the trading strategy."""

from dataclasses import dataclass
from datetime import datetime
import itertools
import subprocess
import pandas as pd
import talib
import v20  # type: ignore
from core.chart import heiken_ashi_numpy

from core.kernel import KernelConfig, kernel
from bot.exchange import (
    getOandaOHLC,
    OandaContext,
)

import logging

from bot.reporting import report

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


@dataclass
class BacktestConfig:
    """BacktestConfig class."""

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

    kernel_conf: KernelConfig
    best_df: pd.DataFrame
    rec: pd.Series


def _preprocess(df: pd.DataFrame, wma_period: int) -> pd.DataFrame:
    # calculate the ATR for the trailing stop loss
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

    return df


def solve(  # noqa: C901, PLR0915
    chart_config: ChartConfig,
    kernel_conf_in: KernelConfig,
    token: str,
    backtest_config: BacktestConfig,
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
    start_time = datetime.now()

    # get data and preprocess
    orig_df = _preprocess(
        _get_data(chart_config, token, logger), kernel_conf_in.wma_period
    )

    # get verifier data and preprocess
    verifier_orig_df = _preprocess(
        _get_data(chart_config, token, logger, backtest_config.verifier),
        kernel_conf_in.wma_period,
    )

    best_result: tuple[BacktestResult, BacktestResult] | None = None
    column_pairs, column_pair_len = backtest_config.get_column_pairs()
    logger.info(f"total_combinations: {column_pair_len}")
    total_found = 0
    found_results: list[BacktestResult] = []
    best_total = 0.0
    count = 0

    with PerfTimer(start_time, logger):
        df = orig_df.copy()
        for config_tuple in column_pairs:
            # run the backtest
            kernel_conf = _map_kernel_conf(kernel_conf_in, config_tuple)
            df["wma"] = df[f"wma_{kernel_conf.source_column}"]
            df = kernel(df, config=kernel_conf)
            rec = df.iloc[-1]

            # log progress and filter invalid results
            count = _log_progress(
                logger, column_pair_len, total_found, count, APP_START_TIME
            )
            if _is_invalid_rec(rec, df):
                continue

            # if this configuration has already been found, skip
            found_results.append(
                BacktestResult(
                    kernel_conf=kernel_conf,
                    best_df=df.copy(),
                    rec=rec.copy(),
                )
            )

            total_found += 1
            total = rec.exit_total
            _log_found(logger, df, rec, kernel_conf)
            _recycle_df(df)

        found_filters = _generate_filters(
            kernel_conf_in, backtest_config, found_results
        )
        count = 0
        df = verifier_orig_df.copy()
        filter_start_time = datetime.now()
        for found in found_filters:
            # run the backtest
            df["wma"] = verifier_orig_df[f"wma_{found[1].source_column}"]
            df = kernel(df, config=found[1])
            rec = df.iloc[-1]

            # log progress and filter invalid results
            count = _log_progress(
                logger, column_pair_len, total_found, count, filter_start_time
            )
            if _is_invalid_rec(rec, df):
                continue

            total_found += 1
            total = rec.exit_total + found[0].rec.exit_total
            if total >= best_total:
                _log_new_max(logger, df, rec, found, total)
                best_result = (
                    found[0],
                    BacktestResult(
                        kernel_conf=found[1],
                        best_df=df.copy(),
                        rec=rec,
                    ),
                )
                best_total = total
            else:
                _log_found(logger, df, rec, found[1])

            _recycle_df(df)

    logger.info("total_found: %s", total_found)
    if total_found == 0:
        logger.error("no combinations found")
        return None

    if best_result is not None:
        q_res, v_res = best_result
        report(q_res.best_df, chart_config.instrument, q_res.kernel_conf, 5)
        report(v_res.best_df, backtest_config.verifier, v_res.kernel_conf, 5)
    else:
        logger.error("no best found")
        return None

    return best_result


def _recycle_df(df):
    df.drop(
        columns=[
            "signal",
            "trigger",
            "position_value",
            "exit_value",
            "exit_total",
            "running_total",
        ],
        inplace=True,
    )


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
    if count % 1000 == 0:
        time_now = datetime.now()
        time_diff = time_now - start_time
        throughput = count / time_diff.total_seconds()
        time_left = time_diff * (column_pair_len - total_found) / count
        logger.info(
            "heartbeat: %s %s%% %s/%s %s/s %s left",
            total_found,
            round(100 * count / column_pair_len, 2),
            count,
            column_pair_len,
            round(throughput, 2),
            time_left,
        )
    return count


def _get_data(
    chart_config: ChartConfig,
    token: str,
    logger: logging.Logger,
    instrument: str | None = None,
):
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


def _log_found(
    logger: logging.Logger, df: pd.DataFrame, rec: pd.Series, kernel_conf: KernelConfig
):
    wins = (df["exit_value"] > 0).astype(int).cumsum()
    losses = (df["exit_value"] < 0).astype(int).cumsum()
    logger.debug(
        "found qt:%s qm:%s qe:%s w:%s l:%s %s",
        round(rec.exit_total, 5),
        round(df["exit_total"].min(), 5),
        round(df["exit_value"].min(), 5),
        wins.iloc[-1],
        losses.iloc[-1],
        kernel_conf,
    )


def _log_new_max(
    logger: logging.Logger,
    df: pd.DataFrame,
    rec: pd.Series,
    found: tuple[BacktestResult, KernelConfig],
    total: float,
):
    wins = (df["exit_value"] > 0).astype(int).cumsum()
    losses = (df["exit_value"] < 0).astype(int).cumsum()
    logger.debug(
        "new vmax found t:%s qt:%s qm:%s qe:%s w:%s l:%s %s",
        round(total, 5),
        round(rec.exit_total, 5),
        round(df["exit_total"].min(), 5),
        round(df["exit_value"].min(), 5),
        wins.iloc[-1],
        losses.iloc[-1],
        found[1],
    )


def _generate_filters(
    kernel_conf_in: KernelConfig,
    backtest_config: BacktestConfig,
    found_results: list[BacktestResult],
) -> list[tuple[BacktestResult, KernelConfig]]:
    found_filters: list[tuple[BacktestResult, KernelConfig]] = []
    for filter_result in found_results:
        for tp in backtest_config.take_profit:
            for sl in backtest_config.stop_loss:
                found_filters.append(
                    (
                        filter_result,
                        KernelConfig(
                            filter_result.kernel_conf.signal_buy_column,
                            filter_result.kernel_conf.signal_exit_column,
                            filter_result.kernel_conf.source_column,
                            kernel_conf_in.wma_period,
                            tp,
                            sl,
                        ),
                    )
                )

    return found_filters


def _is_invalid_rec(rec, df):
    wins = (df["exit_value"] > 0).astype(int).cumsum()
    return (
        wins.iloc[-1] == 0
        or rec.exit_total < 0
        or (
            df["exit_total"].min() < 0
            and abs(df["exit_total"].min()) > abs(rec.exit_total)
        )
    )
