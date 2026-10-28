from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.parse import urlsplit

import pandas as pd
import requests

from trading_core import (
    build_indicator_bundle,
    build_signal_summary,
    calculate_ml_probability,
    clean_symbol,
    score_indicator_bundle,
)
from sp500_universe import get_rotating_candidates, get_rotation_period_start

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets/v2"
STUDENTS = ("Kunal", "Navroop", "Kavya")
STUDENT_BUDGET = 100_000.0
MAX_POSITION_PCT = 0.10
MAX_NEW_BUYS = 4
MAX_GROUP_BUYS_PER_STUDENT = 10
MAX_NEW_SELLS = 3
BARS_SYMBOLS_PER_REQUEST = 100
BAR_CACHE_TTL_SECONDS = 300
BAR_CACHE_MAX_BATCHES = 20
RATE_LIMIT_RETRIES = 3
ALPACA_REQUEST_INTERVAL_SECONDS = 0.5
_BAR_CACHE: OrderedDict[
    tuple[str, int, str, tuple[str, ...]],
    tuple[float, Dict[str, pd.DataFrame], Dict[str, str]],
] = OrderedDict()
_BAR_CACHE_LOCK = threading.Lock()
_BAR_FETCH_LOCK = threading.Lock()
_REQUEST_RATE_LOCK = threading.Lock()
_LAST_ALPACA_REQUEST_AT = 0.0


@dataclass
class ProposedOrder:
    symbol: str
    side: str
    qty: int
    reference_price: float
    estimated_value: float
    final_score: float
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    student: str = "Kunal"
    account_name: str = "primary"


def _headers(api_key: str, secret_key: str) -> Dict[str, str]:
    if not api_key or not secret_key:
        raise ValueError("Enter both Alpaca paper API credentials.")
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }


def _trading_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url.strip())
    if (
        parsed.scheme != "https"
        or parsed.hostname != "paper-api.alpaca.markets"
        or parsed.path.rstrip("/") not in {"", "/v2"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Trading endpoints must use https://paper-api.alpaca.markets.")
    root = base_url.rstrip("/")
    if not root.endswith("/v2"):
        root += "/v2"
    return f"{root}/{path.lstrip('/')}"


def _request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    *,
    params: Dict[str, Any] | None = None,
    payload: Dict[str, Any] | None = None,
) -> Any:
    global _LAST_ALPACA_REQUEST_AT
    hostname = urlsplit(url).hostname or "unknown host"
    is_alpaca_request = hostname.endswith("alpaca.markets")
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        if is_alpaca_request:
            with _REQUEST_RATE_LOCK:
                wait = (
                    _LAST_ALPACA_REQUEST_AT
                    + ALPACA_REQUEST_INTERVAL_SECONDS
                    - time.monotonic()
                )
                if wait > 0:
                    time.sleep(wait)
                _LAST_ALPACA_REQUEST_AT = time.monotonic()
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=payload,
            timeout=25,
        )
        if response.status_code != 429:
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        if attempt == RATE_LIMIT_RETRIES:
            retry_after = response.headers.get("Retry-After")
            raise RuntimeError(
                f"Rate limit from {hostname}{urlsplit(url).path} remained exceeded after "
                f"{RATE_LIMIT_RETRIES} retries"
                + (f" (Retry-After: {retry_after}s)." if retry_after else ".")
            )
        retry_after_value = response.headers.get("Retry-After", "")
        try:
            delay = max(0.0, float(retry_after_value))
        except ValueError:
            delay = float(2 ** attempt)
        time.sleep(delay)
    raise RuntimeError("Alpaca request retry loop ended unexpectedly.")


def _clear_bar_cache() -> None:
    with _BAR_CACHE_LOCK:
        _BAR_CACHE.clear()


def _fetch_batch_bars(
    symbols: Sequence[str],
    lookback_days: int,
    timeframe: str,
    headers: Dict[str, str],
) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    with _BAR_FETCH_LOCK:
        return _fetch_batch_bars_locked(symbols, lookback_days, timeframe, headers)


def _fetch_batch_bars_locked(
    symbols: Sequence[str],
    lookback_days: int,
    timeframe: str,
    headers: Dict[str, str],
) -> tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    frames: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    for offset in range(0, len(symbols), BARS_SYMBOLS_PER_REQUEST):
        batch = symbols[offset:offset + BARS_SYMBOLS_PER_REQUEST]
        key_fingerprint = hashlib.sha256(
            headers.get("APCA-API-KEY-ID", "").encode("utf-8")
        ).hexdigest()
        cache_key = (
            key_fingerprint,
            lookback_days,
            timeframe,
            tuple(batch),
        )
        with _BAR_CACHE_LOCK:
            cached = _BAR_CACHE.get(cache_key)
            if cached and time.monotonic() - cached[0] < BAR_CACHE_TTL_SECONDS:
                _BAR_CACHE.move_to_end(cache_key)
                frames.update(
                    {symbol: frame.copy(deep=True) for symbol, frame in cached[1].items()}
                )
                errors.update(cached[2])
                continue
            if cached:
                del _BAR_CACHE[cache_key]
        params = {
            "symbols": ",".join(batch),
            "timeframe": timeframe,
            "start": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "end": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "limit": 10000,
            "feed": "iex",
        }
        raw_bars: Dict[str, List[Dict[str, Any]]] = {}
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        while True:
            page_params = dict(params)
            if page_token:
                if page_token in seen_page_tokens:
                    raise RuntimeError("Alpaca returned a repeated market-data page token.")
                seen_page_tokens.add(page_token)
                page_params["page_token"] = page_token
            payload = _request_json(
                "GET",
                f"{DATA_BASE_URL}/stocks/bars",
                headers,
                params=page_params,
            )
            for symbol, bars in payload.get("bars", {}).items():
                raw_bars.setdefault(symbol, []).extend(bars)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        batch_frames: Dict[str, pd.DataFrame] = {}
        batch_errors: Dict[str, str] = {}
        for symbol in batch:
            bars = raw_bars.get(symbol, [])
            if not bars:
                batch_errors[symbol] = "No bars returned for symbol."
                continue
            frame = pd.DataFrame(bars).rename(
                columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            )
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(frame.columns):
                batch_errors[symbol] = "Alpaca response is missing required OHLCV fields."
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
            frame = frame.set_index("timestamp").sort_index()
            if frame.empty:
                batch_errors[symbol] = "No valid OHLCV rows after validation."
                continue
            batch_frames[symbol] = frame[["open", "high", "low", "close", "volume"]]
        with _BAR_CACHE_LOCK:
            _BAR_CACHE[cache_key] = (
                time.monotonic(),
                {symbol: frame.copy(deep=True) for symbol, frame in batch_frames.items()},
                batch_errors.copy(),
            )
            _BAR_CACHE.move_to_end(cache_key)
            while len(_BAR_CACHE) > BAR_CACHE_MAX_BATCHES:
                _BAR_CACHE.popitem(last=False)
        frames.update(batch_frames)
        errors.update(batch_errors)
    return frames, errors


def _news_score(symbol: str, api_key: str) -> Dict[str, Any]:
    if not api_key:
        return {"news_score": 50.0, "sentiment": "neutral", "status": "neutral_fallback"}
    try:
        payload = _request_json(
            "GET",
            "https://newsapi.org/v2/everything",
            {"X-Api-Key": api_key},
            params={"q": f'"{symbol}" stock', "sortBy": "publishedAt", "pageSize": 10, "language": "en"},
        )
        articles = payload.get("articles", [])
        if not articles:
            return {"news_score": 50.0, "sentiment": "neutral", "status": "no_articles"}
        positive = ("beat", "surge", "upgrade", "growth", "strong", "bullish", "profit", "record")
        negative = ("drop", "miss", "downgrade", "weak", "bearish", "loss", "lawsuit", "decline")
        score_sum = 0
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            score_sum += int(any(word in text for word in positive))
            score_sum -= int(any(word in text for word in negative))
        score = max(0.0, min(100.0, 50.0 + 50.0 * score_sum / len(articles)))
        label = "positive" if score > 50 else "negative" if score < 50 else "neutral"
        return {"news_score": score, "sentiment": label, "status": "ok"}
    except Exception as exc:
        logger.warning("News request failed for %s: %s", symbol, exc)
        return {"news_score": 50.0, "sentiment": "neutral", "status": "neutral_fallback"}


def _load_account_state(
    headers: Dict[str, str],
    base_url: str = PAPER_BASE_URL,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    account = _request_json("GET", _trading_url(base_url, "account"), headers)
    positions = _request_json("GET", _trading_url(base_url, "positions"), headers)
    open_orders = _request_json(
        "GET",
        _trading_url(base_url, "orders"),
        headers,
        params={"status": "open", "limit": 500},
    )
    if account.get("trading_blocked") or account.get("account_blocked"):
        raise RuntimeError("Alpaca paper account is blocked from trading.")
    return account, positions, open_orders


def _configured_accounts(
    api_key: str,
    secret_key: str,
    base_url: str,
    secondary_api_key: str = "",
    secondary_secret_key: str = "",
    secondary_base_url: str = PAPER_BASE_URL,
) -> List[Dict[str, Any]]:
    accounts = [
        {
            "name": "primary",
            "headers": _headers(api_key, secret_key),
            "base_url": base_url,
        }
    ]
    if bool(secondary_api_key) != bool(secondary_secret_key):
        raise ValueError(
            "Configure both ALPACA_SECONDARY_API_KEY and "
            "ALPACA_SECONDARY_SECRET_KEY, or leave both empty."
        )
    if secondary_api_key:
        if api_key == secondary_api_key and secret_key == secondary_secret_key:
            raise ValueError("Primary and secondary Alpaca credentials must be different.")
        accounts.append(
            {
                "name": "secondary",
                "headers": _headers(secondary_api_key, secondary_secret_key),
                "base_url": secondary_base_url,
            }
        )
    for account in accounts:
        _trading_url(str(account["base_url"]), "account")
    return accounts


def _require_distinct_accounts(accounts: Sequence[Dict[str, Any]]) -> None:
    account_ids = [
        str(account.get("id", ""))
        for account in accounts
        if account.get("id")
    ]
    if len(account_ids) != len(set(account_ids)):
        raise ValueError(
            "The configured Alpaca credentials resolve to the same paper account. "
            "Use keys for two distinct paper accounts to combine their buying power."
        )


def _score_symbol(
    symbol: str,
    frame: pd.DataFrame,
    news_api_key: str,
) -> Dict[str, Any]:
    indicators = build_indicator_bundle(frame)
    technical_score = score_indicator_bundle(indicators)
    ml_output = calculate_ml_probability(frame)
    news = _news_score(symbol, news_api_key)
    natr = float(indicators.get("volatility", {}).get("natr", 0.0) or 0.0)
    risk_score = max(0.0, min(100.0, 100.0 - natr * 10.0))
    summary = build_signal_summary(
        technical_score,
        float(ml_output.get("ml_score", 50.0)),
        float(news.get("news_score", 50.0)),
        risk_score,
    )
    return {
        "symbol": symbol,
        **summary,
        "prediction_probability": float(ml_output.get("prediction_probability", 0.5)),
        "model": ml_output.get("model", "unknown"),
        "news_status": news.get("status", "unknown"),
        "indicators": indicators,
        "price": float(frame["close"].iloc[-1]),
        "atr": float(indicators.get("volatility", {}).get("atr", 0.0) or 0.0),
        "rationale": (
            f"Technical indicators score {summary['technical_score']:.1f}; "
            f"five-day model probability {float(ml_output.get('prediction_probability', 0.5)):.1%}; "
            f"news is {news.get('sentiment', 'neutral')}; ATR is {float(indicators.get('volatility', {}).get('atr', 0.0) or 0.0):.2f}."
        ),
    }


def _build_proposals(
    scores: List[Dict[str, Any]],
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    account: Dict[str, Any],
    *,
    budget: float,
    max_position_pct: float = MAX_POSITION_PCT,
    max_new_buys: int = MAX_NEW_BUYS,
    max_new_sells: int = MAX_NEW_SELLS,
) -> List[ProposedOrder]:
    held = {str(item["symbol"]).upper(): item for item in positions}
    open_symbols = {str(item.get("symbol", "")).upper() for item in open_orders}
    projected_position_value = sum(float(item.get("market_value", 0.0) or 0.0) for item in positions)
    cash = float(account.get("cash", 0.0) or 0.0)
    buying_capacity = max(0.0, min(cash, budget - projected_position_value))
    per_position_cap = budget * max_position_pct
    proposals: List[ProposedOrder] = []

    sells = [
        item for item in scores
        if item["decision"] == "SELL" and item["symbol"] in held and item["symbol"] not in open_symbols
    ]
    sells.sort(key=lambda item: item["final_score"])
    for item in sells[:max_new_sells]:
        position = held[item["symbol"]]
        position_qty = float(position.get("qty", 0.0) or 0.0)
        qty = int(math.floor(position_qty)) if position_qty > 0 else 0
        if qty <= 0:
            continue
        price = float(item["price"])
        proposals.append(
            ProposedOrder(
                symbol=item["symbol"],
                side="sell",
                qty=qty,
                reference_price=price,
                estimated_value=qty * price,
                final_score=float(item["final_score"]),
                reason="SELL threshold reached for an existing position.",
            )
        )

    buys = [
        item for item in scores
        if item["decision"] in {"BUY", "STRONG BUY"}
        and item["symbol"] not in held
        and item["symbol"] not in open_symbols
    ]
    buys.sort(key=lambda item: item["final_score"], reverse=True)
    for item in buys[:max_new_buys]:
        price = float(item["price"])
        if price <= 0:
            continue
        max_position_value = min(
            per_position_cap,
            max(0.0, budget - projected_position_value),
            buying_capacity,
        )
        qty = int(max_position_value // price)
        if qty <= 0:
            continue
        estimated_value = qty * price
        atr = float(item.get("atr", 0.0) or 0.0)
        proposals.append(
            ProposedOrder(
                symbol=item["symbol"],
                side="buy",
                qty=qty,
                reference_price=price,
                estimated_value=estimated_value,
                final_score=float(item["final_score"]),
                reason=item["rationale"],
                stop_loss=round(max(0.01, price - 2 * atr), 2),
                take_profit=round(price + 3 * atr, 2),
            )
        )
        projected_position_value += estimated_value
        buying_capacity -= estimated_value
    return proposals


def analyze_portfolio(
    *,
    student: str,
    symbols: Sequence[str],
    api_key: str,
    secret_key: str,
    lookback_days: int = 180,
    timeframe: str = "1Day",
    news_api_key: str = "",
) -> Dict[str, Any]:
    if student not in STUDENTS:
        raise ValueError(f"Choose one of the configured students: {', '.join(STUDENTS)}.")
    cleaned_symbols = list(dict.fromkeys(clean_symbol(symbol) for symbol in symbols))
    if not cleaned_symbols:
        raise ValueError("Enter at least one stock symbol.")
    if not 10 <= len(cleaned_symbols) <= 15:
        raise ValueError("Enter 10–15 distinct symbols for the selected student.")

    headers = _headers(api_key, secret_key)
    frames, data_errors = _fetch_batch_bars(cleaned_symbols, lookback_days, timeframe, headers)
    if not frames:
        raise RuntimeError("No selected symbols had valid market data: " + json.dumps(data_errors))
    account, positions, open_orders = _load_account_state(headers)

    scores = []
    analysis_errors = dict(data_errors)
    for symbol, frame in frames.items():
        try:
            scores.append(_score_symbol(symbol, frame, news_api_key))
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Analysis failed for %s", symbol)
            analysis_errors[symbol] = str(exc)

    proposals = _build_proposals(
        scores,
        positions,
        open_orders,
        account,
        budget=STUDENT_BUDGET,
    )
    return {
        "student": student,
        "student_budget": STUDENT_BUDGET,
        "account": {
            "status": account.get("status"),
            "equity": float(account.get("equity", 0.0) or 0.0),
            "cash": float(account.get("cash", 0.0) or 0.0),
            "buying_power": float(account.get("buying_power", 0.0) or 0.0),
        },
        "positions": positions,
        "open_orders": open_orders,
        "scores": scores,
        "proposals": [asdict(order) for order in proposals],
        "errors": analysis_errors,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _reconcile_bucket_orders(
    headers: Dict[str, str],
    recent_orders: List[Dict[str, Any]] | None = None,
    *,
    account_name: str = "primary",
    base_url: str = PAPER_BASE_URL,
) -> None:
    from portfolio_ledger import apply_order_update, get_bucket_state

    buckets = get_bucket_state()
    pending_ids = []
    for bucket in buckets.values():
        pending_ids.extend(
            order["order_id"]
            for order in bucket["pending_orders"]
            if order.get("order_id")
            and order.get("account_name", "primary") == account_name
        )
    if not pending_ids:
        return

    if recent_orders is None:
        recent_orders = _request_json(
            "GET",
            _trading_url(base_url, "orders"),
            headers,
            params={"status": "all", "limit": 500, "direction": "desc"},
        )
    if not isinstance(recent_orders, list):
        raise RuntimeError("Alpaca returned an invalid order-list response during reconciliation.")
    orders_by_id = {
        str(order.get("id", "")): order
        for order in recent_orders
        if order.get("id")
    }
    for order_id in pending_ids:
        order = orders_by_id.get(order_id)
        if order is None:
            order = _request_json(
                "GET",
                _trading_url(base_url, f"orders/{order_id}"),
                headers,
            )
        apply_order_update(order)


def _sell_signal_reason(item: Dict[str, Any], average_price: float) -> str | None:
    reasons: List[str] = []
    if str(item.get("decision", "")).upper() == "SELL":
        reasons.append("model signal is SELL")
    price = float(item.get("price", 0.0) or 0.0)
    atr = float(item.get("atr", 0.0) or 0.0)
    if average_price > 0 and price > 0 and atr > 0:
        if price <= average_price - 2 * atr:
            reasons.append("price reached the 2-ATR stop-loss")
        if price >= average_price + 3 * atr:
            reasons.append("price reached the 3-ATR take-profit")
    if not reasons:
        return None
    return "Sell strategy triggered: " + "; ".join(reasons) + "."


def _build_group_proposals(
    scores_by_symbol: Dict[str, Dict[str, Any]],
    student_watchlists: Dict[str, Sequence[str]],
    buckets: Dict[str, Dict[str, Any]],
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    account: Dict[str, Any],
) -> List[ProposedOrder]:
    from portfolio_ledger import get_open_bucket_symbol_pairs, get_pending_buy_reservations

    real_positions = {str(item["symbol"]).upper(): item for item in positions}
    real_open_symbols = {str(item.get("symbol", "")).upper() for item in open_orders}
    pending_bucket_symbols = get_open_bucket_symbol_pairs()
    account_equity = float(account.get("equity", 0.0) or 0.0)
    account_cash = float(account.get("cash", 0.0) or 0.0)
    buying_power = float(account.get("buying_power", account_cash) or 0.0)
    actual_position_value = sum(
        float(item.get("market_value", 0.0) or 0.0)
        for item in positions
    )
    pending_reservations = get_pending_buy_reservations()
    global_capacity = max(
        0.0,
        min(
            account_cash - pending_reservations,
            account_equity - actual_position_value - pending_reservations,
            buying_power,
        ),
    )
    per_position_cap = STUDENT_BUDGET * MAX_POSITION_PCT
    proposals: List[ProposedOrder] = []
    planning_cash: Dict[str, float] = {}
    planning_position_value: Dict[str, float] = {}
    for student in STUDENTS:
        bucket = buckets[student]
        pending_cash = sum(
            float(order["reserved_cash"])
            for order in bucket["pending_orders"]
            if order["side"] == "buy"
        )
        planning_cash[student] = max(0.0, float(bucket["cash"]) - pending_cash)
        planning_position_value[student] = sum(
            float(holding["qty"])
            * (
                abs(
                    float(
                        real_positions.get(holding["symbol"], {}).get(
                            "market_value",
                            0.0,
                        )
                        or 0.0
                    )
                )
                / max(
                    abs(
                        float(
                            real_positions.get(holding["symbol"], {}).get(
                                "qty",
                                0.0,
                            )
                            or 0.0
                        )
                    ),
                    1e-9,
                )
                if holding["symbol"] in real_positions
                else float(holding["average_price"])
            )
            for holding in bucket["holdings"]
        )

    # Positions are sold from the account that owns them; untracked broker positions
    # can be proposed as unassigned sells without changing a student's virtual ledger.
    buy_candidates: List[tuple[float, str, str, Dict[str, Any]]] = []
    owners_by_position: Dict[tuple[str, str], tuple[str, Dict[str, Any]]] = {}
    for student in STUDENTS:
        bucket = buckets[student]
        allowed_symbols = set(student_watchlists[student])
        owned = {
            item["symbol"]: item
            for item in bucket["holdings"]
        }
        for symbol, holding in owned.items():
            owners_by_position[
                (str(holding.get("account_name", "primary")), symbol)
            ] = (student, holding)
        for symbol in allowed_symbols:
            item = scores_by_symbol.get(symbol)
            if not item:
                continue
            if (
                symbol not in owned
                and symbol not in real_positions
                and symbol not in real_open_symbols
                and (student, symbol) not in pending_bucket_symbols
            ):
                buy_candidates.append((-float(item["final_score"]), student, symbol, item))

    used_symbols: set[str] = set()
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        item = scores_by_symbol.get(symbol)
        if not item or symbol in real_open_symbols:
            continue
        position_qty = float(position.get("qty", 0.0) or 0.0)
        qty = int(math.floor(position_qty)) if position_qty > 0 else 0
        if qty <= 0:
            continue
        account_name = str(position.get("_account_name", "primary"))
        owner = owners_by_position.get((account_name, symbol))
        if owner:
            student, bucket_position = owner
            if (student, symbol) in pending_bucket_symbols:
                continue
            pending_sell_qty = sum(
                float(order["requested_qty"])
                for order in buckets[student]["pending_orders"]
                if order["symbol"] == symbol
                and order["side"] == "sell"
                and order.get("account_name", "primary") == account_name
            )
            qty = min(
                qty,
                int(math.floor(max(0.0, float(bucket_position["qty"]) - pending_sell_qty))),
            )
        else:
            student = "Unassigned"
        if qty <= 0:
            continue
        avg_entry = float(position.get("avg_entry_price", 0.0) or 0.0)
        if avg_entry <= 0 and owner:
            avg_entry = float(owner[1].get("average_price", 0.0) or 0.0)
        sell_reason = _sell_signal_reason(item, avg_entry)
        if sell_reason is None:
            continue
        price = float(item["price"])
        proposals.append(
            ProposedOrder(
                student=student,
                symbol=symbol,
                side="sell",
                qty=qty,
                reference_price=price,
                estimated_value=qty * price,
                final_score=float(item["final_score"]),
                reason=sell_reason,
                account_name=account_name,
            )
        )
        used_symbols.add(symbol)

    ranked_candidates: Dict[str, List[tuple[float, str, str, Dict[str, Any]]]] = {
        student: [] for student in STUDENTS
    }
    for candidate in sorted(buy_candidates):
        ranked_candidates[candidate[1]].append(candidate)

    round_robin_buy_candidates: List[
        tuple[float, str, str, Dict[str, Any]]
    ] = []
    candidate_offsets = {student: 0 for student in STUDENTS}
    while True:
        added_candidate = False
        for student in STUDENTS:
            offset = candidate_offsets[student]
            if offset < len(ranked_candidates[student]):
                round_robin_buy_candidates.append(
                    ranked_candidates[student][offset]
                )
                candidate_offsets[student] += 1
                added_candidate = True
        if not added_candidate:
            break

    student_buy_counts = {student: 0 for student in STUDENTS}
    for _, student, symbol, item in round_robin_buy_candidates:
        if (
            symbol in used_symbols
            or student_buy_counts[student] >= MAX_GROUP_BUYS_PER_STUDENT
            or len([order for order in proposals if order.side == "buy"]) >= MAX_GROUP_BUYS_PER_STUDENT * len(STUDENTS)
        ):
            continue
        if global_capacity <= 0:
            break
        bucket = buckets[student]
        price = float(item["price"])
        if price <= 0:
            continue
        available_bucket_cash = planning_cash[student]
        owned_value = planning_position_value[student]
        bucket_capacity = max(
            0.0,
            min(
                available_bucket_cash,
                STUDENT_BUDGET - owned_value,
                per_position_cap,
            ),
        )
        order_value = min(bucket_capacity, global_capacity)
        qty = int(order_value // price)
        if qty <= 0:
            continue
        estimated_value = qty * price
        atr = float(item.get("atr", 0.0) or 0.0)
        proposals.append(
            ProposedOrder(
                student=student,
                symbol=symbol,
                side="buy",
                qty=qty,
                reference_price=price,
                estimated_value=estimated_value,
                final_score=float(item["final_score"]),
                reason=(
                    f"{item['rationale']} Buy score threshold bypassed by the "
                    "10-stock-per-student selection setting."
                ),
                stop_loss=round(max(0.01, price - 2 * atr), 2),
                take_profit=round(price + 3 * atr, 2),
            )
        )
        used_symbols.add(symbol)
        student_buy_counts[student] += 1
        global_capacity -= estimated_value
        planning_cash[student] -= estimated_value
        planning_position_value[student] += estimated_value
    return proposals


def _assign_recommendations(
    scores: Dict[str, Dict[str, Any]],
    buckets: Dict[str, Dict[str, Any]],
    positions: List[Dict[str, Any]],
    open_orders: List[Dict[str, Any]],
    recommendation_count: int,
) -> Dict[str, List[Dict[str, Any]]]:
    occupied = {
        str(position.get("symbol", "")).upper()
        for position in positions
    }
    occupied.update(
        str(order.get("symbol", "")).upper()
        for order in open_orders
    )
    occupied.update(
        holding["symbol"].upper()
        for bucket in buckets.values()
        for holding in bucket["holdings"]
    )
    candidates = sorted(
        (
            item
            for symbol, item in scores.items()
            if symbol not in occupied
        ),
        key=lambda item: (
            -float(item["final_score"]),
            item["symbol"],
        ),
    )
    recommendations = {student: [] for student in STUDENTS}
    for item in candidates:
        student = min(
            STUDENTS,
            key=lambda name: (len(recommendations[name]), STUDENTS.index(name)),
        )
        if len(recommendations[student]) >= recommendation_count:
            continue
        recommendations[student].append(item)
        if all(
            len(recommendations[name]) >= recommendation_count
            for name in STUDENTS
        ):
            break
    return recommendations


def analyze_group_portfolio(
    *,
    api_key: str,
    secret_key: str,
    recommendation_count: int = 10,
    lookback_days: int = 180,
    timeframe: str = "1Day",
    news_api_key: str = "",
    base_url: str = PAPER_BASE_URL,
    secondary_api_key: str = "",
    secondary_secret_key: str = "",
    secondary_base_url: str = PAPER_BASE_URL,
) -> Dict[str, Any]:
    from portfolio_ledger import get_bucket_state

    if not 10 <= recommendation_count <= 15:
        raise ValueError("Choose between 10 and 15 recommendations per student.")
    rotation_time = datetime.now(timezone.utc)
    rotating_candidates = get_rotating_candidates(rotation_time)
    accounts = _configured_accounts(
        api_key,
        secret_key,
        base_url,
        secondary_api_key,
        secondary_secret_key,
        secondary_base_url,
    )
    account_summaries: List[Dict[str, Any]] = []
    positions: List[Dict[str, Any]] = []
    open_orders: List[Dict[str, Any]] = []
    for account_config in accounts:
        broker_account, account_positions, account_orders = _load_account_state(
            account_config["headers"],
            account_config["base_url"],
        )
        _reconcile_bucket_orders(
            account_config["headers"],
            account_name=account_config["name"],
            base_url=account_config["base_url"],
        )
        positions.extend(
            {**position, "_account_name": account_config["name"]}
            for position in account_positions
        )
        open_orders.extend(
            {**order, "_account_name": account_config["name"]}
            for order in account_orders
        )
        account_summaries.append(
            {
                "name": account_config["name"],
                "id": str(broker_account.get("id", "")),
                "status": broker_account.get("status"),
                "equity": float(broker_account.get("equity", 0.0) or 0.0),
                "cash": float(broker_account.get("cash", 0.0) or 0.0),
                "buying_power": float(broker_account.get("buying_power", 0.0) or 0.0),
            }
        )
    _require_distinct_accounts(account_summaries)
    aggregate_account = {
        "status": ", ".join(str(item["status"]) for item in account_summaries),
        "equity": sum(item["equity"] for item in account_summaries),
        "cash": sum(item["cash"] for item in account_summaries),
        "buying_power": sum(item["buying_power"] for item in account_summaries),
    }
    all_symbols = list(
        dict.fromkeys(
            [
                *rotating_candidates,
                *(
                    clean_symbol(str(position["symbol"]))
                    for position in positions
                    if position.get("symbol")
                ),
            ]
        )
    )
    frames, data_errors = _fetch_batch_bars(
        all_symbols,
        lookback_days,
        timeframe,
        accounts[0]["headers"],
    )
    if not frames:
        raise RuntimeError("No S&P 500 symbols or held positions had valid market data: " + json.dumps(data_errors))
    buckets = get_bucket_state()
    current_prices = {
        symbol: float(item["close"].iloc[-1])
        for symbol, item in frames.items()
    }
    for bucket in buckets.values():
        bucket["position_value"] = sum(
            float(position["qty"])
            * current_prices.get(position["symbol"], float(position["average_price"]))
            for position in bucket["holdings"]
        )

    scores: Dict[str, Dict[str, Any]] = {}
    errors = dict(data_errors)
    for symbol, frame in frames.items():
        try:
            scores[symbol] = _score_symbol(symbol, frame, "")
        except (ValueError, KeyError, TypeError) as exc:
            logger.exception("Analysis failed for %s", symbol)
            errors[symbol] = str(exc)

    if news_api_key and scores:
        news_candidates = sorted(
            scores.values(),
            key=lambda item: (-float(item["final_score"]), item["symbol"]),
        )[:recommendation_count * len(STUDENTS)]
        news_fetch_available = True
        for item in news_candidates:
            news = (
                _news_score(item["symbol"], news_api_key)
                if news_fetch_available
                else {
                    "news_score": 50.0,
                    "sentiment": "neutral",
                    "status": "neutral_fallback",
                }
            )
            if news["status"] == "neutral_fallback":
                news_fetch_available = False
            item["news_score"] = float(news["news_score"])
            item["news_status"] = str(news["status"])
            summary = build_signal_summary(
                float(item["technical_score"]),
                float(item["ml_score"]),
                float(item["news_score"]),
                float(item["risk_score"]),
            )
            item.update(summary)
            item["rationale"] = (
                f"Technical score {item['technical_score']:.1f}; "
                f"five-day model probability {float(item['prediction_probability']):.1%}; "
                f"news sentiment {news['sentiment']}; "
                f"ATR {float(item['atr']):.2f}."
            )

    recommendations = _assign_recommendations(
        scores,
        buckets,
        positions,
        open_orders,
        recommendation_count,
    )
    selected_watchlists = {
        student: [item["symbol"] for item in recommendations[student]]
        for student in STUDENTS
    }
    proposals = _build_group_proposals(
        scores,
        selected_watchlists,
        buckets,
        positions,
        open_orders,
        aggregate_account,
    )
    return {
        "student_budgets": {student: STUDENT_BUDGET for student in STUDENTS},
        "recommendation_count": recommendation_count,
        "universe_size": len(rotating_candidates),
        "rotation_period": get_rotation_period_start(rotation_time).isoformat(),
        "account": {
            "status": ", ".join(str(item["status"]) for item in account_summaries),
            "equity": sum(item["equity"] for item in account_summaries),
            "cash": sum(item["cash"] for item in account_summaries),
            "buying_power": sum(item["buying_power"] for item in account_summaries),
            "accounts": account_summaries,
        },
        "student_watchlists": selected_watchlists,
        "recommendations": recommendations,
        "buckets": buckets,
        "positions": positions,
        "open_orders": open_orders,
        "scores": list(scores.values()),
        "proposals": [asdict(order) for order in proposals],
        "errors": errors,
        "lookback_days": lookback_days,
        "timeframe": timeframe,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def credential_fingerprint(api_key: str, secret_key: str, *additional_parts: str) -> str:
    """Return a non-reversible identifier for matching approval to the same credentials."""
    values = (api_key, secret_key, *additional_parts)
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def select_buy_proposals(
    proposals: Sequence[Dict[str, Any]],
    selected_buys: Sequence[tuple[str, str]],
) -> List[Dict[str, Any]]:
    selected = {
        (student, symbol.upper())
        for student, symbol in selected_buys
    }
    return [
        proposal
        for proposal in proposals
        if str(proposal.get("side", "")).lower() == "buy"
        and (
            str(proposal.get("student", "")),
            str(proposal.get("symbol", "")).upper(),
        )
        in selected
    ]


def select_sell_proposals(
    proposals: Sequence[Dict[str, Any]],
    selected_sells: Sequence[tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    selected = {
        (student, symbol.upper(), account_name)
        for student, symbol, account_name in selected_sells
    }
    return [
        proposal
        for proposal in proposals
        if str(proposal.get("side", "")).lower() == "sell"
        and (
            str(proposal.get("student", "")),
            str(proposal.get("symbol", "")).upper(),
            str(proposal.get("account_name", "primary")),
        )
        in selected
    ]


def execute_approved_orders(
    analysis: Dict[str, Any],
    *,
    api_key: str,
    secret_key: str,
    approved: bool,
    base_url: str = PAPER_BASE_URL,
    secondary_api_key: str = "",
    secondary_secret_key: str = "",
    secondary_base_url: str = PAPER_BASE_URL,
) -> List[Dict[str, Any]]:
    from portfolio_ledger import (
        apply_order_update,
        get_bucket_position,
        get_bucket_state,
        get_pending_buy_reservations,
        get_pending_sell_qty,
        record_pending_order,
    )

    if not approved:
        raise PermissionError("Order execution requires explicit approval.")
    account_configs = _configured_accounts(
        api_key,
        secret_key,
        base_url,
        secondary_api_key,
        secondary_secret_key,
        secondary_base_url,
    )
    account_states: Dict[str, Dict[str, Any]] = {}
    positions: List[Dict[str, Any]] = []
    open_orders: List[Dict[str, Any]] = []
    for account_config in account_configs:
        account, account_positions, account_orders = _load_account_state(
            account_config["headers"],
            account_config["base_url"],
        )
        _reconcile_bucket_orders(
            account_config["headers"],
            account_name=account_config["name"],
            base_url=account_config["base_url"],
        )
        positions.extend(
            {**position, "_account_name": account_config["name"]}
            for position in account_positions
        )
        open_orders.extend(
            {**order, "_account_name": account_config["name"]}
            for order in account_orders
        )
        account_states[account_config["name"]] = {
            **account_config,
            "account": account,
            "cash": float(account.get("cash", 0.0) or 0.0),
            "equity": float(account.get("equity", 0.0) or 0.0),
            "buying_power": float(
                account.get("buying_power", account.get("cash", 0.0)) or 0.0
            ),
            "position_value": sum(
                abs(float(item.get("market_value", 0.0) or 0.0))
                for item in account_positions
            ),
        }
    _require_distinct_accounts(
        [
            {
                "id": state["account"].get("id"),
            }
            for state in account_states.values()
        ]
    )
    buckets = get_bucket_state()

    # Recheck all account state immediately before submitting any paper orders.
    current_positions = {
        str(item["symbol"]).upper(): item
        for item in positions
    }
    positions_by_account: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: {} for name in account_states
    }
    for position in positions:
        account_name = str(position.get("_account_name", "primary"))
        positions_by_account.setdefault(account_name, {})[
            str(position["symbol"]).upper()
        ] = position
    pending_symbols = {str(item.get("symbol", "")).upper() for item in open_orders}
    pending_reservations = {
        name: get_pending_buy_reservations(name)
        for name in account_states
    }
    bucket_available_cash: Dict[str, float] = {}
    bucket_position_values: Dict[str, float] = {}
    for student, bucket in buckets.items():
        pending_cash = sum(
            float(order["reserved_cash"])
            for order in bucket["pending_orders"]
            if order["side"] == "buy"
        )
        bucket_available_cash[student] = max(0.0, float(bucket["cash"]) - pending_cash)
        bucket_position_values[student] = 0.0
        for holding in bucket["holdings"]:
            symbol = holding["symbol"]
            account_name = str(holding.get("account_name", "primary"))
            position = positions_by_account.get(account_name, {}).get(symbol)
            if position is not None:
                actual_qty = abs(float(position.get("qty", 0.0) or 0.0))
                actual_market_value = abs(float(position.get("market_value", 0.0) or 0.0))
                price = actual_market_value / max(actual_qty, 1e-9)
            else:
                price = float(holding["average_price"])
            bucket_position_values[student] += float(holding["qty"]) * price
    results: List[Dict[str, Any]] = []
    buy_symbols = list(
        dict.fromkeys(
            clean_symbol(proposal["symbol"])
            for proposal in analysis.get("proposals", [])
            if str(proposal.get("side", "")).lower() == "buy"
        )
    )
    latest_trades: Dict[str, Any] = {}
    if buy_symbols:
        snapshot_payload = _request_json(
            "GET",
            f"{DATA_BASE_URL}/stocks/snapshots",
            account_configs[0]["headers"],
            params={"symbols": ",".join(buy_symbols), "feed": "iex"},
        )
        latest_trades = snapshot_payload

    for proposal in analysis.get("proposals", []):
        student = str(proposal.get("student", ""))
        symbol = clean_symbol(proposal["symbol"])
        side = str(proposal["side"]).lower()
        if student not in STUDENTS and not (
            student == "Unassigned" and side == "sell"
        ):
            results.append(
                {
                    "student": student,
                    "symbol": symbol,
                    "status": "rejected",
                    "reason": "Proposal has no valid student bucket attribution.",
                }
            )
            continue
        qty = int(proposal["qty"])
        price = float(proposal["reference_price"])
        if qty <= 0 or price <= 0 or side not in {"buy", "sell"}:
            results.append({"student": student, "symbol": symbol, "status": "rejected", "reason": "Invalid order proposal."})
            continue
        if symbol in pending_symbols:
            results.append({"student": student, "symbol": symbol, "status": "rejected", "reason": "An open order already exists."})
            continue
        if side == "buy":
            snapshot = latest_trades.get(symbol, {})
            latest_trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
            live_price = float(latest_trade.get("p", latest_trade.get("price", 0.0)) or 0.0)
            if live_price <= 0:
                results.append({"symbol": symbol, "status": "rejected", "reason": "No fresh Alpaca trade price; refusing market order."})
                continue
            price = live_price
            value = qty * price
            if symbol in current_positions:
                results.append({"student": student, "symbol": symbol, "status": "rejected", "reason": "A shared-account position appeared after analysis; refusing duplicate buy."})
                continue
            owned_position = get_bucket_position(student, symbol)
            owned_value = (
                owned_position["qty"] * owned_position["average_price"]
                if owned_position
                else 0.0
            )
            bucket_allowed_value = min(
                STUDENT_BUDGET * MAX_POSITION_PCT,
                max(0.0, STUDENT_BUDGET - owned_value),
                max(0.0, bucket_available_cash[student]),
                max(0.0, STUDENT_BUDGET - bucket_position_values[student]),
            )
            selected_account: Dict[str, Any] | None = None
            selected_account_capacity = 0.0
            account_capacities: List[tuple[int, Dict[str, Any], float]] = []
            desired_qty = min(qty, int(bucket_allowed_value // price))
            for name in ("primary", "secondary"):
                account_state = account_states.get(name)
                if account_state is None:
                    continue
                available = max(
                    0.0,
                    min(
                        account_state["cash"] - pending_reservations[name],
                        account_state["equity"]
                        - account_state["position_value"]
                        - pending_reservations[name],
                        account_state["buying_power"],
                    ),
                )
                account_capacities.append(
                    (int(min(bucket_allowed_value, available) // price), account_state, available)
                )
            selected_capacity = next(
                (
                    candidate
                    for candidate in account_capacities
                    if candidate[0] >= desired_qty and desired_qty > 0
                ),
                None,
            )
            if selected_capacity is None and account_capacities:
                selected_capacity = max(
                    account_capacities,
                    key=lambda candidate: candidate[0],
                )
            if selected_capacity is not None and selected_capacity[0] > 0:
                _, selected_account, selected_account_capacity = selected_capacity
            if selected_account is None:
                results.append(
                    {
                        "student": student,
                        "symbol": symbol,
                        "status": "rejected",
                        "reason": "No configured paper account has sufficient available buying power.",
                    }
                )
                continue
            allowed_value = min(bucket_allowed_value, selected_account_capacity)
            qty = min(qty, int(allowed_value // price))
            value = qty * price
            if qty <= 0:
                results.append({"student": student, "symbol": symbol, "status": "rejected", "reason": "Fresh price leaves no whole-share quantity within the paper-account and student-bucket limits."})
                continue
        else:
            value = qty * price
            bucket_position = (
                get_bucket_position(student, symbol)
                if student in STUDENTS
                else None
            )
            account_name = str(
                proposal.get(
                    "account_name",
                    bucket_position.get("account_name", "primary")
                    if bucket_position
                    else "primary",
                )
            )
            selected_account = account_states.get(account_name)
            position = (
                positions_by_account.get(account_name, {}).get(symbol)
                if selected_account
                else None
            )
            actual_qty = (
                abs(float(position.get("qty", 0.0) or 0.0))
                if position
                else 0.0
            )
            if position and float(position.get("qty", 0.0) or 0.0) < 0:
                actual_qty = 0.0
            pending_sell_qty = (
                get_pending_sell_qty(student, symbol)
                if bucket_position
                else 0.0
            )
            owned_qty = max(
                0.0,
                (bucket_position["qty"] if bucket_position else 0.0) - pending_sell_qty,
            )
            exceeds_attributed_qty = bool(
                bucket_position
                and qty > int(math.floor(owned_qty))
            )
            if (
                selected_account is None
                or position is None
                or qty > int(math.floor(actual_qty))
                or exceeds_attributed_qty
            ):
                results.append({"student": student, "symbol": symbol, "status": "rejected", "reason": "Sell quantity exceeds the available broker or attributed position."})
                continue

        order_payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        try:
            order_result = _request_json(
                "POST",
                _trading_url(selected_account["base_url"], "orders"),
                selected_account["headers"],
                payload=order_payload,
            )
            order_id = str(order_result.get("id", ""))
            if not order_id:
                raise RuntimeError(f"Alpaca accepted {symbol} order without returning an order id.")
            if student in STUDENTS:
                record_pending_order(
                    order_id=order_id,
                    student=student,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    reference_price=price,
                    account_name=selected_account["name"],
                )
            results.append({"student": student, "symbol": symbol, "status": "submitted", "account": selected_account["name"], "order": order_result})
            pending_symbols.add(symbol)
            if side == "buy":
                selected_account["buying_power"] = max(
                    0.0,
                    selected_account["buying_power"] - value,
                )
                pending_reservations[selected_account["name"]] += value
                bucket_available_cash[student] -= value
                bucket_position_values[student] += value
            if str(order_result.get("status", "")).lower() in {
                "filled",
                "canceled",
                "cancelled",
                "rejected",
                "expired",
            }:
                apply_order_update(order_result)
        except requests.RequestException as exc:
            results.append({"student": student, "symbol": symbol, "status": "error", "error": str(exc)})
    _write_execution_logs(analysis, results)
    return results


def fetch_portfolio_dashboard(
    api_key: str,
    secret_key: str,
    *,
    base_url: str = PAPER_BASE_URL,
    secondary_api_key: str = "",
    secondary_secret_key: str = "",
    secondary_base_url: str = PAPER_BASE_URL,
) -> Dict[str, Any]:
    account_configs = _configured_accounts(
        api_key,
        secret_key,
        base_url,
        secondary_api_key,
        secondary_secret_key,
        secondary_base_url,
    )
    positions: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    account_summaries: List[Dict[str, Any]] = []
    for account_config in account_configs:
        account = _request_json(
            "GET",
            _trading_url(account_config["base_url"], "account"),
            account_config["headers"],
        )
        account_positions = _request_json(
            "GET",
            _trading_url(account_config["base_url"], "positions"),
            account_config["headers"],
        )
        account_orders = _request_json(
            "GET",
            _trading_url(account_config["base_url"], "orders"),
            account_config["headers"],
            params={"status": "all", "limit": 500, "direction": "desc"},
        )
        if not isinstance(account_positions, list) or not isinstance(account_orders, list):
            raise RuntimeError("Alpaca returned an invalid positions or orders response.")
        _reconcile_bucket_orders(
            account_config["headers"],
            recent_orders=account_orders,
            account_name=account_config["name"],
            base_url=account_config["base_url"],
        )
        positions.extend(
            {**position, "_account_name": account_config["name"]}
            for position in account_positions
        )
        orders.extend(
            {**order, "_account_name": account_config["name"]}
            for order in account_orders
        )
        account_summaries.append(
            {
                "name": account_config["name"],
                "id": str(account.get("id", "")),
                "status": account.get("status"),
                "equity": float(account.get("equity", 0.0) or 0.0),
                "cash": float(account.get("cash", 0.0) or 0.0),
                "buying_power": float(account.get("buying_power", 0.0) or 0.0),
            }
        )
    _require_distinct_accounts(account_summaries)
    open_orders = [
        order
        for order in orders
        if str(order.get("status", "")).lower()
        in {
            "new",
            "accepted",
            "pending_new",
            "accepted_for_bidding",
            "partially_filled",
            "pending_cancel",
            "pending_replace",
            "held",
            "stopped",
            "calculated",
        }
    ]
    from portfolio_ledger import get_bucket_state

    return {
        "account": {
            "status": ", ".join(str(item["status"]) for item in account_summaries),
            "equity": sum(item["equity"] for item in account_summaries),
            "cash": sum(item["cash"] for item in account_summaries),
            "buying_power": sum(item["buying_power"] for item in account_summaries),
        },
        "accounts": account_summaries,
        "positions": positions,
        "open_orders": open_orders,
        "orders": orders,
        "buckets": get_bucket_state(),
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_execution_logs(analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")
    logs_dir = Path("logs")
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_label = analysis.get("student", "group").replace(" ", "_").lower()
    report_path = reports_dir / f"trading_run_{report_label}_{timestamp}.md"
    report_path.write_text(
        "# Paper trading run\n\n"
        f"- Student bucket: {analysis.get('student', 'Shared group portfolio')}\n"
        f"- Per-student virtual budget: ${STUDENT_BUDGET:,.2f}\n"
        f"- Run time: {analysis['created_at']}\n"
        "- Mode: Alpaca paper trading\n\n"
        "## Decisions and scores\n\n"
        + "\n".join(
            f"- {s['symbol']}: {s['decision']} — score {s['final_score']:.2f} "
            f"(technical {s['technical_score']:.1f}, ML {s['ml_score']:.1f}, "
            f"news {s['news_score']:.1f}, risk {s['risk_score']:.1f})"
            for s in analysis["scores"]
        )
        + "\n\n## Orders\n\n"
        + json.dumps(results, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )

    csv_path = logs_dir / "paper_orders.csv"
    fieldnames = ["timestamp", "student", "symbol", "status", "side", "qty", "order_id", "error"]
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for result in results:
            order = result.get("order", {})
            writer.writerow(
                {
                    "timestamp": analysis["created_at"],
                    "student": result.get("student", analysis.get("student", "")),
                    "symbol": result.get("symbol", ""),
                    "status": result.get("status", ""),
                    "side": order.get("side", ""),
                    "qty": order.get("qty", ""),
                    "order_id": order.get("id", ""),
                    "error": result.get("error", result.get("reason", "")),
                }
            )
