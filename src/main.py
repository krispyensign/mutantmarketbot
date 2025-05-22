"""Main module."""

from datetime import datetime
import logging
import logging.config
import sys

import v20  # type: ignore
import yaml

from bot.common import (
    BotConfig,
    ChartConfig,
    OandaConfig,
    PerfTimer,
    SolverConfig,
    TradeConfig,
)
from bot.solve import segmented_solve
from bot.bot import bot
from core.kernel import KernelConfig
import os
from bot.exchange import OandaContext

TOKEN = os.environ.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")

USAGE = """
    mutantmarketbot

      Usage:
        python main.py solve <my_config>.yaml
        python main.py bot <my_config>.yaml
        python main.py backtest <my_config>.yaml

      ENV:
        OANDA_TOKEN=<token>
        OANDA_ACCOUNT_ID=<account_id>

      """


if __name__ == "__main__":
    start_time = datetime.now()
    if TOKEN is None or ACCOUNT_ID is None:
        print(sys.argv)
        print(USAGE)
        sys.exit(1)
    if "help" in sys.argv[1] or "--help" in sys.argv[1]:
        print(USAGE)
        sys.exit(0)
    elif "solve" in sys.argv[1]:
        # load config
        conf = yaml.safe_load(open(sys.argv[2]))
        chart_conf = ChartConfig(**conf["chart"])
        kernel_conf = KernelConfig(**conf["kernel"])
        solver_conf = SolverConfig(**conf["solver"])
        oanda_conf = OandaConfig(
            token=TOKEN,
            account_id=ACCOUNT_ID,
        )
        chart_conf.candle_count = solver_conf.sample_size + solver_conf.train_size

        # setup logging
        logging_conf = conf["logging"]
        logging.config.dictConfig(logging_conf)
        logger = logging.getLogger("main")

        # create Oanda context
        ctx = OandaContext(
            ctx=v20.Context("api-fxpractice.oanda.com", token=oanda_conf.token),
            account_id=oanda_conf.account_id,
            token=oanda_conf.token,
            instrument=chart_conf.instrument,
        )

        # configure take profit and stop loss
        tp = conf["take_profit"] if "take_profit" in conf else [0.0]
        sl = conf["stop_loss"] if "stop_loss" in conf else [0.0]

        # run
        with PerfTimer(start_time, logger):
            result = segmented_solve(chart_conf, kernel_conf, TOKEN, solver_conf)
        if result == 0.0:
            sys.exit(1)

        logger.info("%s", result)

    elif sys.argv[1] in ["bot", "backtest"]:
        # load config
        conf = yaml.safe_load(open(sys.argv[2]))

        chart_conf = ChartConfig(**conf["chart"])
        kernel_conf = KernelConfig(**conf["kernel"])
        trade_conf = TradeConfig(**conf["trade"])
        solver_conf = SolverConfig(**conf["solver"])

        # setup logging
        logging_conf = conf["logging"]
        logging.config.dictConfig(logging_conf)
        logger = logging.getLogger("main")

        # create Oanda context
        ctx = OandaContext(
            ctx=v20.Context("api-fxpractice.oanda.com", token=TOKEN),
            account_id=ACCOUNT_ID,
            token=TOKEN,
            instrument=chart_conf.instrument,
        )
        # run
        bot(
            oanda_ctx=ctx,
            bot_conf=BotConfig(
                chart_conf=chart_conf,
                kernel_conf=kernel_conf,
                trade_conf=trade_conf,
                solver_conf=solver_conf,
                backtest_only=sys.argv[1] == "backtest",
            ),
        )
    else:
        print(sys.argv)
        print(USAGE)
