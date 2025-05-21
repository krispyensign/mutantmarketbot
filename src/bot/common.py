"""Common data structures for the bot."""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Generator
import uuid

import numpy as np
from core.kernel import KernelConfig

APP_START_TIME = datetime.now()
FRIDAY = 5
SUNDAY = 7
FIVE_PM = 21
HALF_MINUTE = 30


@dataclass
class TradeConfig:
    """Configuration for the bot."""

    amount: float
    bot_id: uuid.UUID


@dataclass
class ChartConfig:
    """ChartConfig class."""

    instrument: str
    granularity: str
    candle_count: int
    verifier: str
    date_from: datetime | None = None


@dataclass
class SolverConfig:
    """SolverConfig class."""

    take_profit: list[float]
    stop_loss: list[float]
    source_columns: list[str]
    solver_interval: int = 3600
    force_edge: str = ""
    sample_size: int = 100
    train_size: int = 80

    def get_configs(
        self, kernel_conf: KernelConfig
    ) -> tuple[Generator[KernelConfig], int]:
        """Get column pairs."""
        if kernel_conf.signal_buy_column == "":
            gen = (
                KernelConfig(
                    signal_buy_column=sb,
                    signal_exit_column=se,
                    source_column=so,
                    take_profit=tp,
                    stop_loss=sl,
                    wma_period=kernel_conf.wma_period,
                )
                for so in self.source_columns
                for sb in self.source_columns
                for se in self.source_columns
                for tp in self.take_profit
                for sl in self.stop_loss
            )
            return gen, len(self.source_columns) ** 3 * len(self.take_profit) * len(
                self.stop_loss
            )
        else:
            gen = (
                KernelConfig(
                    signal_buy_column=kernel_conf.signal_buy_column,
                    signal_exit_column=kernel_conf.signal_exit_column,
                    source_column=kernel_conf.source_column,
                    wma_period=kernel_conf.wma_period,
                    take_profit=tp,
                    stop_loss=sl,
                )
                for tp in self.take_profit
                for sl in self.stop_loss
            )
            return gen, len(self.take_profit) * len(self.stop_loss)


@dataclass
class BacktestResult:
    """BacktestResult class."""

    instrument: str
    kernel_conf: KernelConfig
    exit_total: np.float64
    ratio: np.float64
    wins: np.int64
    losses: np.int64

    def __str__(self) -> str:
        """Return a string representation of the BacktestResult object."""
        return (
            f"result: {self.kernel_conf} "
            f"et:{round(self.exit_total, 5)} "
            f"r:{round(self.ratio, 5)} "
            f"wins:{self.wins} "
            f"losses:{self.losses}"
        )


@dataclass
class BotConfig:
    """Configuration for the bot."""

    chart_conf: ChartConfig
    kernel_conf: KernelConfig
    trade_conf: TradeConfig
    solver_conf: SolverConfig
    backtest_only: bool


@dataclass
class OandaConfig:
    """Configuration for Oanda."""

    token: str
    account_id: str


class PerfTimer:
    """PerfTimer class."""

    def __init__(self, app_start_time: datetime, logger: logging.Logger):
        """Initialize a PerfTimer object."""
        self.app_start_time = app_start_time
        self.logger = logger
        pass

    def __enter__(self) -> "PerfTimer":
        """Start the timer."""
        self.start = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type:ignore
        """Stop the timer."""
        self.end = datetime.now()
        self.logger.info(f"run interval: {self.end - self.start}")
        self.logger.info("up time: %s", (self.end - self.app_start_time))
        self.logger.info("last run time: %s", self.end.strftime("%Y-%m-%d %H:%M:%S"))
