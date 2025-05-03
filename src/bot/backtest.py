"""Backtest the trading strategy."""

from dataclasses import dataclass
from datetime import datetime
import itertools
import subprocess
import pandas as pd
import v20  # type: ignore
from alive_progress import alive_it  # type: ignore

from core.kernel import KernelConfig, kernel
from bot.exchange import (
    getOandaOHLC,
    OandaContext,
)

import logging

from bot.reporting import report

APP_START_TIME = datetime.now()


def get_git_info() -> tuple[str, bool] | None:
    """Get commit hash and whether the working tree is clean.

    Returns a tuple (str, bool). The first element is the commit hash. The second
    element is a boolean indicating whether the working tree is clean (i.e., there
    are no pending changes).

    If there is an error (e.g., not in a Git repository), the function returns None.
    """
    logger = logging.getLogger("backtest")
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()

        # Get porcelain status
        porcelain_status = subprocess.check_output(
            ["git", "status", "--porcelain"], encoding="utf-8"
        ).strip()

        return commit_hash, porcelain_status == ""
    except subprocess.CalledProcessError as e:
        logger.error("Failed to get Git info: %s", e)
        # Handle errors, e.g., when not in a Git repository
        return None


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


def backtest(  # noqa: C901, PLR0915
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
    git_info = get_git_info()
    if git_info is None:
        logger.error("Failed to get Git info")
        return None
    logger.info("git info: %s %s", git_info[0], git_info[1])
    start_time = datetime.now()
    orig_df, verifier_df_orig = _get_data(chart_config, token, backtest_config, logger)

    best_result: tuple[BacktestResult, BacktestResult] | None = None

    column_pairs, column_pair_len = backtest_config.get_column_pairs()
    logger.info(f"total_combinations: {column_pair_len}")
    total_found = 0
    found_results: list[BacktestResult] = []
    best_total = 0.0
    with PerfTimer(start_time, logger):
        for config_tuple in alive_it(column_pairs, total=column_pair_len):
            kernel_conf = _map_kernel_conf(kernel_conf_in, config_tuple)
            df = kernel(
                orig_df.copy(),
                config=kernel_conf,
            )
            rec = df.iloc[-1]

            # if there are no wins, the total is worse, or the min total is worse then skip
            if _is_invalid_rec(rec):
                continue

            # if this configuration has already been found, skip
            found_results.append(
                BacktestResult(
                    kernel_conf=kernel_conf,
                    best_df=df,
                    rec=rec,
                )
            )

            total_found += 1
            _log_found(logger, df, rec)
            total = rec.exit_total

        found_filters = _generate_filters(
            kernel_conf_in, backtest_config, found_results
        )
        for found in alive_it(found_filters, total=len(found_filters)):
            df = kernel(
                verifier_df_orig.copy(),
                config=found[1],
            )
            rec = df.iloc[-1]

            # if there are no wins, the total is worse, or the min total is worse then skip
            if _is_invalid_rec(rec):
                continue
            
            total_found += 1
            total = rec.exit_total + found[0].rec.exit_total
            if total >= best_total:
                _log_new_max(logger, df, rec, found, total)
                best_result = (
                    found[0],
                    BacktestResult(
                        kernel_conf=found[1],
                        best_df=df,
                        rec=rec,
                    ),
                )
                best_total = total
            else:
                _log_found(logger, df, rec)

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


def _map_kernel_conf(kernel_conf_in, config_tuple):
    kernel_conf = KernelConfig(
        wma_period=kernel_conf_in.wma_period,
        signal_buy_column=config_tuple[0],
        signal_exit_column=config_tuple[1],
        source_column=config_tuple[2],
        take_profit=config_tuple[3],
        stop_loss=config_tuple[4],
    )

    return kernel_conf


def _log_found(logger, df, rec):
    logger.debug(
        "found qt:%s qm:%s qe:%s w:%s l:%s",
        rec.exit_total,
        rec.min_exit_total,
        df["exit_value"].min(),
        rec.wins,
        rec.losses,
    )


def _get_data(chart_config, token, backtest_config, logger):
    ctx = OandaContext(
        v20.Context("api-fxpractice.oanda.com", token=token),
        None,
        token,
        chart_config.instrument,
    )

    orig_df = getOandaOHLC(
        ctx, count=chart_config.candle_count, granularity=chart_config.granularity
    )
    logger.info(
        "count: %s granularity: %s",
        chart_config.candle_count,
        chart_config.granularity,
    )

    verifier_ctx = OandaContext(
        v20.Context("api-fxpractice.oanda.com", token=token),
        None,
        token,
        backtest_config.verifier,
    )
    verifier_df_orig = getOandaOHLC(
        verifier_ctx,
        count=chart_config.candle_count,
        granularity=chart_config.granularity,
    )

    return orig_df, verifier_df_orig


def _log_new_max(logger, df, rec, found, total):
    logger.debug(
        "new vmax found t:%s qt:%s qm:%s qe:%s w:%s l:%s %s",
        round(total, 5),
        round(rec.exit_total, 5),
        round(rec.min_exit_total, 5),
        round(df["exit_value"].min(), 5),
        rec.wins,
        rec.losses,
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


def _is_invalid_rec(rec):
    return (
        rec.wins == 0
        or rec.exit_total < 0
        or (rec.min_exit_total < 0 and abs(rec.min_exit_total) > abs(rec.exit_total))
    )
