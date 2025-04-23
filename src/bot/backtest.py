"""Backtest the trading strategy."""

from dataclasses import dataclass
from datetime import datetime
import itertools
import pandas as pd
import v20  # type: ignore
from alive_progress import alive_it  # type: ignore

from bot.constants import (
    SOURCE_COLUMNS,
    TP,
    SL,
)
from core.kernel import KernelConfig, kernel
from bot.exchange import (
    getOandaOHLC,
    OandaContext,
)

import logging

from bot.reporting import report

logger = logging.getLogger("backtest")
APP_START_TIME = datetime.now()


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
class SignalConfig:
    """SignalConfig class."""

    source_column: str
    signal_buy_column: str
    signal_exit_column: str
    stop_loss: float
    take_profit: float

    def __str__(self):
        """Return a string representation of the SignalConfig object."""
        return f"so:{self.source_column}, sib:{self.signal_buy_column}, sie:{self.signal_exit_column}, sl:{self.stop_loss}, tp:{self.take_profit}"


@dataclass
class Record:
    """Record class."""

    signal: int
    trigger: int
    losses: int
    wins: int
    exit_total: float
    min_exit_total: float
    take_profit: float

    def __str__(self) -> str:
        """Return a string representation of the Record object."""
        return f"w:{self.wins} l:{self.losses}, q:{round(self.exit_total, 5)}, q_min:{round(self.min_exit_total, 5)}"


@dataclass
class ChartConfig:
    """ChartConfig class."""

    instrument: str
    granularity: str
    wma_period: int
    candle_count: int


def backtest(chart_config: ChartConfig, token: str) -> SignalConfig | None:
    """Run a backtest of the trading strategy.

    Parameters
    ----------
    chart_config : ChartConfig
        The chart configuration.
    token : str
        The Oanda API token.

    Notes
    -----
    The backtest will run for a large number of combinations of source and signal
    columns. The best combination will be saved to best_df and the results will be
    printed to the log file.

    """
    logger.info("starting backtest")
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
        "count: %s granularity: %s wma_period: %s",
        chart_config.candle_count,
        chart_config.granularity,
        chart_config.wma_period,
    )

    best_max_conf = SignalConfig("", "", "", 0.0, 0.0)
    not_worst_conf = SignalConfig("", "", "", 0.0, 0.0)
    best_df = pd.DataFrame()
    not_worst_df = pd.DataFrame()
    best_rec = Record(0, 0, 0, 0, -99.0, -99.0, 0.0)
    not_worst_rec = Record(0, 0, 0, 0, -99.0, -99.0, 0.0)

    column_pairs = itertools.product(
        SOURCE_COLUMNS, SOURCE_COLUMNS, SOURCE_COLUMNS, TP, SL
    )
    column_pair_len = (
        len(SOURCE_COLUMNS)
        * len(SOURCE_COLUMNS)
        * len(SOURCE_COLUMNS)
        * len(TP)
        * len(SL)
    )
    logger.info(f"total_combinations: {column_pair_len}")
    total_found = 0
    with PerfTimer(start_time, logger):
        for (
            source_column_name,
            signal_buy_column_name,
            signal_exit_column_name,
            take_profit_multiplier,
            stop_loss_multiplier,
        ) in alive_it(column_pairs, total=column_pair_len):
            # since we are taking the edge only open is deterministic
            if "open" not in signal_exit_column_name:
                continue

            if stop_loss_multiplier > take_profit_multiplier:
                continue

            kernel_conf = KernelConfig(
                signal_buy_column=signal_buy_column_name,
                signal_exit_column=signal_exit_column_name,
                source_column=source_column_name,
                wma_period=chart_config.wma_period,
                take_profit=take_profit_multiplier,
                stop_loss=stop_loss_multiplier,
            )
            df = kernel(
                orig_df.copy(),
                include_incomplete=False,
                config=kernel_conf,
            )

            signal_conf = SignalConfig(
                source_column_name,
                signal_buy_column_name,
                signal_exit_column_name,
                stop_loss_multiplier,
                take_profit_multiplier,
            )

            rec = Record(
                signal=df["signal"].iloc[-1],
                trigger=df["trigger"].iloc[-1],
                losses=df["losses"].iloc[-1],
                wins=df["wins"].iloc[-1],
                exit_total=df["exit_total"].iloc[-1],
                min_exit_total=df["min_exit_total"].iloc[-1],
                take_profit=df["take_profit"].iloc[-1] + df["entry_price"].iloc[-1],
            )

            if rec.wins == 0:
                continue
            else:
                total_found += 1

            if rec.min_exit_total > not_worst_rec.min_exit_total:
                logger.debug(
                    "new min found %s %s",
                    rec,
                    signal_conf,
                )
                not_worst_rec = rec
                not_worst_conf = signal_conf
                not_worst_df = df.copy()

            if rec.exit_total > best_rec.exit_total:
                logger.debug(
                    "new max found %s %s",
                    rec,
                    signal_conf,
                )
                best_rec = rec
                best_max_conf = signal_conf
                best_df = df.copy()

    logger.info("total_found: %s", total_found)
    if total_found == 0:
        logger.error("no winning combinations found")
        return None

    logger.debug(
        "best max found %s %s",
        best_max_conf,
        best_rec,
    )
    report(
        best_df,
        chart_config.instrument,
        best_max_conf.signal_buy_column,
        best_max_conf.signal_exit_column,
    )

    logger.debug(
        "not worst found %s %s",
        not_worst_conf,
        not_worst_rec,
    )
    report(
        not_worst_df,
        chart_config.instrument,
        not_worst_conf.signal_buy_column,
        not_worst_conf.signal_exit_column,
    )

    # choose the least worst combination to minimize loss
    if (not_worst_rec.wins - not_worst_rec.losses) > (best_rec.wins - best_rec.losses):
        logger.info("best min selected")
        return not_worst_conf

    logger.info("best max selected")
    return best_max_conf
