from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from urllib.request import Request, urlopen


_longbridge_http_preferred = "https://openapi.longbridge.cn"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alpaca_get(path: str) -> object:
    request = Request(
        "https://paper-api.alpaca.markets" + path,
        headers={
            "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
        },
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


async def read_alpaca_paper_account() -> dict:
    """Read account and positions only; this module defines no order operation."""
    account, positions = await asyncio.gather(
        asyncio.to_thread(_alpaca_get, "/v2/account"),
        asyncio.to_thread(_alpaca_get, "/v2/positions"),
    )
    return {
        "provider": "Alpaca Paper", "environment": "paper", "status": "ok",
        "currency": account.get("currency", "USD"),
        "cash": float(account.get("cash", 0)),
        "equity": float(account.get("equity", 0)),
        "day_start_equity": float(account.get("last_equity", 0)),
        "buying_power": float(account.get("buying_power", 0)),
        "positions": [{
            "symbol": item["symbol"], "name": item["symbol"],
            "quantity": float(item.get("qty", 0)),
            "market_value": float(item.get("market_value", 0)),
            "unrealized_pl": float(item.get("unrealized_pl", 0)),
            "currency": account.get("currency", "USD"),
        } for item in positions],
        "updated_at": _now(),
    }


async def read_longbridge_paper_account() -> dict:
    """Try both official Longbridge HTTP access points and remember the winner."""
    global _longbridge_http_preferred
    endpoints = [_longbridge_http_preferred]
    endpoints.append("https://openapi.longbridge.com" if endpoints[0].endswith(".cn")
                     else "https://openapi.longbridge.cn")
    for endpoint in endpoints:
        try:
            result = await asyncio.wait_for(
                _read_longbridge_paper_account(endpoint), timeout=12.0)
            _longbridge_http_preferred = endpoint
            result["access_point"] = endpoint.rsplit(".", 1)[-1]
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
    raise ConnectionError(
        "长桥账户的中国内地与国际接入点均暂时不可用，程序将自动重试"
    )


async def _read_longbridge_paper_account(http_url: str) -> dict:
    """Read Longbridge simulated balances and positions; never submit orders."""
    from longbridge.openapi import AsyncQuoteContext, AsyncTradeContext, Config, PortfolioContext

    domain = "openapi-quote.longbridge.cn" if http_url.endswith(".cn") else "openapi-quote.longbridge.com"
    config = Config.from_apikey(
        os.environ["LONGBRIDGE_APP_KEY"],
        os.environ["LONGBRIDGE_APP_SECRET"],
        os.environ["LONGBRIDGE_ACCESS_TOKEN"],
        http_url=http_url,
        quote_ws_url=f"wss://{domain}",
        enable_print_quote_packages=False,
        enable_papertrading=True,
    )
    context = AsyncTradeContext.create(config)
    balances, response = await asyncio.gather(
        context.account_balance(), context.stock_positions())
    preferred = next((item for item in balances if item.currency == "HKD"),
                     balances[0] if balances else None)
    account_currency = preferred.currency if preferred else "HKD"
    fx_by_currency = {account_currency: 1.0}
    try:
        rates = await asyncio.to_thread(lambda: PortfolioContext(config).exchange_rate())
        fx_by_currency.update({item.other_currency: float(item.average_rate)
                               for item in rates.exchanges
                               if item.base_currency == account_currency})
    except Exception:
        pass
    raw_positions = [item for channel in response.channels for item in channel.positions]
    quote_by_symbol = {}
    valuation_status = "unavailable"
    if raw_positions:
        try:
            quote_context = AsyncQuoteContext.create(config)
            quotes = await asyncio.wait_for(
                quote_context.quote([item.symbol for item in raw_positions]), timeout=4.0)
            quote_by_symbol = {item.symbol: item for item in quotes}
            valuation_status = "ok"
        except Exception:
            # Return balances/positions promptly; the monitor retries valuation separately.
            quote_by_symbol = {}
    positions = []
    for item in raw_positions:
        quantity, cost = float(item.quantity), float(item.cost_price)
        quote = quote_by_symbol.get(item.symbol)
        current = None if quote is None else float(quote.last_done)
        previous_close = None if quote is None else float(quote.prev_close)
        market_value = None if current is None else current * quantity
        unrealized = None if current is None else (current - cost) * quantity
        ratio = None if current is None or cost <= 0 else current / cost - 1.0
        today_pl = (None if current is None or previous_close is None else
                    (current - previous_close) * quantity)
        today_ratio = (None if current is None or not previous_close else
                       current / previous_close - 1.0)
        positions.append({
            "symbol": item.symbol, "name": item.symbol_name,
            "market": "HK" if item.symbol.endswith(".HK") else "US",
            "quantity": quantity, "cost_price": cost, "current_price": current,
            "previous_close": previous_close,
            "market_value": market_value, "unrealized_pl": unrealized,
            "unrealized_pl_ratio": ratio, "today_pl": today_pl,
            "today_pl_ratio": today_ratio, "currency": item.currency,
        })
    account_equity = float(preferred.net_assets) if preferred else 0.0
    today_pl_account_currency = 0.0
    today_pl_complete = True
    for item in positions:
        fx = fx_by_currency.get(item["currency"])
        account_value = (None if item["market_value"] is None or fx is None else
                         item["market_value"] * fx)
        item["market_value_account_currency"] = account_value
        item["position_weight"] = (account_value / account_equity
                                   if account_value is not None and account_equity > 0 else None)
        item["position_weight_scope"] = "net_assets"
        if item["today_pl"] is None or fx is None:
            today_pl_complete = False
        else:
            today_pl_account_currency += item["today_pl"] * fx
    day_start_equity = (account_equity - today_pl_account_currency
                        if account_equity > 0 and today_pl_complete else 0.0)
    return {
        "provider": "长桥 OpenAPI 模拟账户", "environment": "paper", "status": "ok",
        "currency": account_currency,
        "cash": float(preferred.total_cash) if preferred else 0.0,
        "equity": float(preferred.net_assets) if preferred else 0.0,
        "day_start_equity": day_start_equity,
        "buying_power": float(preferred.buy_power) if preferred else 0.0,
        "positions": positions, "valuation_status": valuation_status,
        "fx_rates": fx_by_currency,
        "updated_at": _now(),
    }


async def monitor_account(market: str, reader, update, interval: float = 15.0) -> None:
    announced = False
    last_success: dict | None = None
    while True:
        try:
            last_success = await reader()
            update(market, last_success)
            if not announced:
                print(f"{market} 模拟账户只读监控已连接")
                announced = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            friendly_error = (str(error) if isinstance(error, ConnectionError) else
                              f"{type(error).__name__}：长桥账户连接失败，将自动重试")
            failure = {
                "status": "stale" if last_success else "error",
                "error": friendly_error,
                "retrying": True, "failed_at": _now(),
            }
            if last_success:
                failure = {**last_success, **failure}
            else:
                failure["positions"] = []
            update(market, failure)
            if not announced:
                print(f"{market} 模拟账户暂时读取失败：{type(error).__name__}；将自动重试")
                announced = True
        retry_soon = bool(last_success and last_success.get("valuation_status") != "ok")
        await asyncio.sleep(5.0 if retry_soon else interval)
