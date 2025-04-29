"""Main module."""

import logging
import logging.config
import sys

import yaml

from bot.backtest import ChartConfig, backtest
from bot.bot import TradeConfig, bot
from core.kernel import KernelConfig
import os

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
    if TOKEN is None or ACCOUNT_ID is None:
        print(sys.argv)
        print(USAGE)
        sys.exit(1)

    if "backtest" in sys.argv[1]:
        # load config
        conf = yaml.safe_load(open(sys.argv[2]))
        chart_conf = ChartConfig(**conf["chart_config"])
        kernel_conf = KernelConfig(**conf["kernel_config"])

        # setup logging
        logging_conf = conf["logging"]
        logging.config.dictConfig(logging_conf)
        logger = logging.getLogger("main")

        # configure take profit and stop loss
        tp = conf["take_profit"] if "take_profit" in conf else [0.0]
        sl = conf["stop_loss"] if "stop_loss" in conf else [0.0]

        # run
        result = backtest(chart_conf, kernel_conf, token=TOKEN, take_profit=tp, stop_loss=sl)
        logger.info(result)
        if result is None:
            sys.exit(1)

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
