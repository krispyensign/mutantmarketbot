"""Backtest the trading strategy."""

from dataclasses import dataclass
from datetime import datetime
import itertools
import subprocess
from typing import Any
import pandas as pd
import v20  # type: ignore
from alive_progress import alive_it  # type: ignore

from core.kernel import KernelConfig, kernel, EdgeCategory
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
    disable_fast: bool

    def get_column_pairs(self) -> tuple[itertools.product, int]:
        """Get column pairs."""
        return itertools.product(
            self.source_columns,
            self.source_columns,
            self.source_columns,
            self.take_profit,
            self.stop_loss,
        ), len(self.source_columns) ** 3 * len(self.take_profit) * len(self.stop_loss)


def backtest(  # noqa: C901, PLR0915
    chart_config: ChartConfig,
    kernel_conf_in: KernelConfig,
    token: str,
    backtest_config: BacktestConfig,
) -> KernelConfig | None:
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

    best_df: pd.DataFrame | None = None
    best_rec: pd.Series[Any] | None = None
    best_conf: KernelConfig | None = None

    column_pairs, column_pair_len = backtest_config.get_column_pairs()
    logger.info(f"total_combinations: {column_pair_len}")
    total_found = 0
    best_total = -99.9
    with PerfTimer(start_time, logger):
        for (
            source_column_name,
            signal_buy_column_name,
            signal_exit_column_name,
            take_profit_multiplier,
            stop_loss_multiplier,
        ) in alive_it(column_pairs, total=column_pair_len):
            kernel_conf = KernelConfig(
                signal_buy_column=signal_buy_column_name,
                signal_exit_column=signal_exit_column_name,
                source_column=source_column_name,
                wma_period=kernel_conf_in.wma_period,
                take_profit=take_profit_multiplier,
                stop_loss=stop_loss_multiplier,
            )
            if backtest_config.disable_fast and kernel_conf.edge == EdgeCategory.Fast:
                continue
            df = kernel(
                orig_df.copy(),
                config=kernel_conf,
            )
            rec = df.iloc[-1]
            if best_rec is None or best_conf is None or best_df is None:
                best_rec = rec
                best_conf = kernel_conf
                best_df = df.copy()

            # if there are no wins, the total is worse, or the min total is worse then skip
            if (
                rec.wins == 0
                or rec.exit_total < 0
                or (
                    rec.min_exit_total < 0
                    and abs(rec.min_exit_total) > abs(rec.exit_total)
                )
            ):
                continue

            verifier_df = kernel(
                verifier_df_orig.copy(),
                config=kernel_conf,
            )
            vrec = verifier_df.iloc[-1]
            if (
                vrec.wins == 0
                or vrec.exit_total < 0
                or (
                    vrec.min_exit_total < 0
                    and abs(vrec.min_exit_total) > abs(vrec.exit_total)
                )
            ):
                continue

            total_found += 1
            total = (
                rec.exit_total
                + rec.min_exit_total
                + vrec.exit_total
                + vrec.min_exit_total
            )
            if total >= best_total:
                logger.debug(
                    "new max found t:%s qt:%s qm:%s vt:%s vm:%s qe:%s ve:%s w:%s l:%s %s",
                    round(total, 5),
                    round(rec.exit_total, 5),
                    round(rec.min_exit_total, 5),
                    round(vrec.exit_total, 5),
                    round(vrec.min_exit_total, 5),
                    round(df["exit_value"].min(), 5),
                    round(verifier_df["exit_value"].min(), 5),
                    vrec.wins + rec.wins,
                    vrec.losses + rec.losses,
                    kernel_conf,
                )
                best_rec = rec
                best_conf = kernel_conf
                best_df = df.copy()
                best_total = total

    logger.info("total_found: %s", total_found)
    if total_found == 0:
        logger.error("no combinations found")
        return None

    logger.debug(
        "best max found %s %s",
        best_conf,
        best_rec,
    )
    if best_df is not None and best_conf is not None:
        report(
            best_df,
            chart_config.instrument,
            best_conf,
            length=10,
        )

    return best_conf
