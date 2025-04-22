"""Get OHLC data from an exchange and convert it into a pandas DataFrame."""

import uuid
import v20  # type: ignore
import pandas as pd
import logging

logger = logging.getLogger("exchange")


class OandaContext:
    """OandaContext class."""

    def __init__(
        self, ctx: v20.Context, account_id: str | None, token: str, instrument: str
    ):
        """Initialize a OandaContext object."""
        self.ctx = ctx
        self.account_id = account_id if account_id is not None else ""
        self.token = token
        self.instrument = instrument


def getOandaBalance(ctx: OandaContext) -> float:
    """Get the current balance from Oanda.

    Parameters
    ----------
    ctx : OandaContext
        The Oanda API context.

    Returns
    -------
    float
        The current balance in the Oanda account.

    """
    resp = ctx.ctx.account.get(ctx.account_id)
    if resp.body["account"]:
        account: v20.account.Account = resp.body["account"]
        return account.balance

    return 0


def getOandaOHLC(
    ctx: OandaContext, granularity: str = "M5", count: int = 288
) -> pd.DataFrame:
    # create dataframe with candles
    """Get OHLC data from Oanda and convert it into a pandas DataFrame.

    Parameters
    ----------
    ctx : OandaContext
        The Oanda API context.
    instrument : str
        The instrument to get the OHLC data for.
    granularity : str, optional
        The granularity of the OHLC data, by default "M5".
    count : int, optional
        The number of candles to get, by default 288.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the OHLC data with the following columns:

        - timestamp
        - open
        - high
        - low
        - close
        - bid_open
        - bid_high
        - bid_low
        - bid_close
        - ask_open
        - ask_high
        - ask_low
        - ask_close

    """
    df = pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "bid_open",
            "bid_high",
            "bid_low",
            "bid_close",
            "ask_open",
            "ask_high",
            "ask_low",
            "ask_close",
        ]
    )

    resp = ctx.ctx.instrument.candles(
        instrument=ctx.instrument,
        granularity=granularity,
        price="MAB",
        count=count,
    )
    if resp.body["candles"]:
        candles: v20.instrument.Candlesticks = resp.body["candles"]
        candle: v20.instrument.Candlestick
        for i, candle in enumerate(candles):
            df.loc[i] = {  # type: ignore
                "timestamp": candle.time,
                "open": candle.mid.o,
                "high": candle.mid.h,
                "low": candle.mid.l,
                "close": candle.mid.c,
                "bid_open": candle.bid.o,
                "bid_high": candle.bid.h,
                "bid_low": candle.bid.l,
                "bid_close": candle.bid.c,
                "ask_open": candle.ask.o,
                "ask_high": candle.ask.h,
                "ask_low": candle.ask.l,
                "ask_close": candle.ask.c,
            }
        logger.info("retrieved %s candles", len(candles))

    return df


def place_order(
    ctx: OandaContext,
    amount: float,
    id: uuid.UUID,
    take_profit: float = 0.0,
    trailing_distance: float = 0.0,
) -> int:
    """Place an order on the Oanda API.

    Parameters
    ----------
    ctx : OandaContext
        The Oanda API context.
    amount : float
        The amount of the instrument to buy or sell.
    id : uuid.UUID
        The UUID of the app.
    take_profit : float
        The take profit price for the order.
    trailing_distance : float
        The trailing distance for the order.

    Returns
    -------
    int
        The order ID of the placed order.

    """
    # place the order
    decimals = 5
    if ctx.instrument.split("_")[1] == "JPY":
        decimals = 3

    client_extensions = v20.transaction.ClientExtensions(
        id=str(id),
        tag="mutant"
    )
    order: v20.order.MarketOrder = v20.order.MarketOrder(
        instrument=ctx.instrument,
        units=amount,
        clientExtensions=client_extensions,
        tradeClientExtensions=client_extensions,
    )
    if take_profit > 0.0:
        takeProfitOnFill = v20.transaction.TakeProfitDetails(
            price=f"{round(take_profit, decimals)}"
        )
        order.takeProfitOnFill = takeProfitOnFill

    if trailing_distance > 0.0:
        trailingStopLossOnFill = v20.transaction.TrailingStopLossDetails(
            distance=f"{round(trailing_distance, decimals)}"
        )
        order.trailingStopLossOnFill = trailingStopLossOnFill

    logger.info(order.json())
    resp: v20.response.Response = ctx.ctx.order.create(
        ctx.account_id,
        order=order,
    )

    if resp.body is None:
        raise Exception("No response body")

    # get the trade id from the response body and return it if it exists
    trade_id: int
    if resp.body is not None and "orderFillTransaction" in resp.body:
        result: v20.transaction.OrderFillTransaction = resp.body["orderFillTransaction"]
        trade: v20.trade.TradeOpen = result.tradeOpened
        trade_id = trade.tradeID
    else:
        if resp.body is not None and "orderRejectTransaction" in resp.body:
            reject_result: v20.transaction.MarketOrderRejectTransaction = resp.body[
                "orderRejectTransaction"
            ]
            logger.error(reject_result.summary())
            logger.error(reject_result.json())
            raise Exception(reject_result.reason)

        raise Exception(resp.body.__str__())

    return trade_id


def close_order(ctx: OandaContext, trade_id: int) -> None:
    """Close an order on the Oanda API.

    Parameters
    ----------
    ctx : OandaContext
        The Oanda API context.
    trade_id : str
        The trade ID of the order to close.

    """
    resp = ctx.ctx.trade.close(ctx.account_id, trade_id)

    if resp.body is not None:
        if "orderRejectTransaction" in resp.body:
            raise Exception(resp.body.to_json())


def get_open_trade(ctx: OandaContext, id: uuid.UUID) -> int:
    """Get the first open trade.

    Parameters
    ----------
    ctx : OandaContext
        The Oanda API context.
    id : uuid.UUID
        The UUID of the app.

    Returns
    -------
    int
        The trade ID of the first open trade for the app.

    """
    resp = ctx.ctx.trade.list_open(ctx.account_id)
    trades: list[v20.trade.Trade] = []
    if resp.body is not None:
        if "trades" in resp.body:
            trades = resp.body["trades"]
            if len(trades) > 0:
                for t in trades:
                    if t.clientExtensions.id == str(id):
                        return t.id

    return -1
