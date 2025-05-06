"""Main module."""

from datetime import datetime
import logging
import logging.config
import sys

import yaml

from bot.backtest import BacktestConfig, ChartConfig, PerfTimer, solve
from bot.bot import TradeConfig, bot
from core.kernel import KernelConfig
import os
import cProfile
import pstats
import io
from pstats import SortKey

TOKEN = os.environ.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")

USAGE = """
    mutantmarketbot
      Usage: 
        python main.py backtest <my_config>.yaml
        python main.py bot <my_config>.yaml
        python main.py observe <my_config>.yaml
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

    if "backtest" in sys.argv[1]:
        # load config
        conf = yaml.safe_load(open(sys.argv[2]))
        chart_conf = ChartConfig(**conf["chart_config"])
        kernel_conf = KernelConfig(**conf["kernel_config"])
        backtest_conf = BacktestConfig(**conf["backtest_config"])

        # setup logging
        logging_conf = conf["logging"]
        logging.config.dictConfig(logging_conf)
        logger = logging.getLogger("main")

        # configure take profit and stop loss
        tp = conf["take_profit"] if "take_profit" in conf else [0.0]
        sl = conf["stop_loss"] if "stop_loss" in conf else [0.0]

        # run
        pr = cProfile.Profile()
        pr.enable()
        with PerfTimer(start_time, logger):
            try:
                result = solve(chart_conf, kernel_conf, TOKEN, backtest_conf)
            except Exception as err:
                logger.error(err)
                pr.disable()
                s = io.StringIO()
                sortby = SortKey.CUMULATIVE
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats(50)
                print(s.getvalue())
            if result is None:
                sys.exit(1)

        logger.info("ins: %s %s", result[0].instrument, result[0].kernel_conf)
        logger.info("ins: %s %s", result[1].instrument, result[1].kernel_conf)

    elif sys.argv[1] in ["bot", "observe"]:
        # load config
        conf = yaml.safe_load(open(sys.argv[2]))
        chart_conf = ChartConfig(**conf["chart_config"])
        kernel_conf = KernelConfig(**conf["kernel_config"])
        trade_conf = TradeConfig(**conf["trade_config"])

        # setup logging
        logging_conf = conf["logging"]
        logging.config.dictConfig(logging_conf)
        logger = logging.getLogger("main")

        # run
        bot(
            token=TOKEN,
            account_id=ACCOUNT_ID,
            chart_conf=chart_conf,
            kernel_conf=kernel_conf,
            trade_conf=trade_conf,
            observe_only=sys.argv[1] == "observe",
        )
    else:
        print(sys.argv)
        print(USAGE)
