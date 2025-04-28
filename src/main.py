"""Main module."""

import logging
import sys

import yaml

from bot.backtest import ChartConfig, backtest
from bot.bot import TradeConfig, bot
from core.kernel import KernelConfig
import os

logging.root.handlers = []

TOKEN = os.environ.get("OANDA_TOKEN")
ACCOUNT_ID = os.environ.get("OANDA_ACCOUNT_ID")

USAGE = """
    mutantmarketbot
      Usage: 
        python main.py backtest EUR_USD <my_config>.yaml
        python main.py bot <my_config>.yaml [observe]
      ENV:
        OANDA_TOKEN=<token>
        OANDA_ACCOUNT_ID=<account_id>
      """


def get_logger(file_name: str):
    """Get logger for main module."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
        handlers=[logging.FileHandler(file_name), logging.StreamHandler()],
    )
    logger = logging.getLogger("main")
    # logger.addHandler(logging.FileHandler())
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    return logger


if __name__ == "__main__":
    if TOKEN is None or ACCOUNT_ID is None:
        print(sys.argv)
        print(USAGE)
        sys.exit(1)
    if "backtest" in sys.argv[1]:
        instrument = sys.argv[2]
        os.makedirs("logs", exist_ok=True)
        logger = get_logger(f"logs/mmbot-backtest-{instrument}.log")
        conf = yaml.safe_load(open(sys.argv[3]))
        chart_conf = ChartConfig(instrument, **conf["chart_config"])
        tp = [0.0]
        sl = [0.0]
        if "take_profit" in conf:
            tp = conf["take_profit"]
        if "stop_loss" in conf:
            sl = conf["stop_loss"]

        result = backtest(chart_conf, token=TOKEN, take_profit=tp, stop_loss=sl)
        logger.info(result)
        if result is None:
            sys.exit(1)
    elif "bot" in sys.argv[1]:
        conf = yaml.safe_load(open(sys.argv[2]))
        chart_conf = ChartConfig(**conf["chart_config"])
        kernel_conf = KernelConfig(**conf["kernel_config"])
        trade_conf = TradeConfig(**conf["trade_config"])
        id = str(trade_conf.bot_id).split("-")[1]
        instrument = chart_conf.instrument
        os.makedirs("logs", exist_ok=True)
        logger = get_logger(f"logs/mmbot-{instrument}-{id}.log")
        if "observe" in sys.argv:
            observe_only = True
        else:
            observe_only = False
        bot(
            token=TOKEN,
            account_id=ACCOUNT_ID,
            chart_conf=chart_conf,
            kernel_conf=kernel_conf,
            trade_conf=trade_conf,
            observe_only=observe_only,
        )
    else:
        print(sys.argv)
        print(USAGE)
