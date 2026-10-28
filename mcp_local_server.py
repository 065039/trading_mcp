from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from trading_core import build_indicator_bundle, calculate_ml_probability, clean_symbol

load_dotenv()

mcp = FastMCP("TradingAgentsMCP")
OHLCV_CACHE: Dict[str, Dict[str, Any]] = {}

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"


def _get_headers() -> Dict[str, str]:
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca API credentials. Copy .env.example to .env and add valid "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY values before running the workflow."
        )
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


def _ensure_ohlcv_cache(symbol: str) -> pd.DataFrame:
    symbol = clean_symbol(symbol)
    if symbol not in OHLCV_CACHE:
        raise RuntimeError(f"No cached OHLCV data for {symbol}. Call fetch_ohlcv_batch first.")
    return OHLCV_CACHE[symbol]["df"]


def _normalize_symbol_argument(value: Any) -> tuple[list[str], bool]:
    """Accept one ticker, a ticker list, or mappings some LLM tool callers emit."""
    if isinstance(value, str):
        return [clean_symbol(value)], False

    if isinstance(value, Mapping):
        if "symbol" in value:
            return _normalize_symbol_argument(value["symbol"])
        if "symbols" in value:
            return _normalize_symbol_argument(value["symbols"])
        raw_symbols = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        raw_symbols = list(value)
    else:
        raise ValueError("Expected a ticker string, a list of ticker strings, or a ticker-keyed mapping.")

    if not raw_symbols:
        raise ValueError("At least one ticker symbol is required.")
    cleaned = list(dict.fromkeys(clean_symbol(item) for item in raw_symbols))
    return cleaned, True


def _batch_tool_result(
    symbols: list[str],
    is_batch: bool,
    operation: Any,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for symbol in symbols:
        try:
            results[symbol] = operation(symbol)
        except (ValueError, RuntimeError, requests.RequestException, KeyError) as exc:
            errors[symbol] = str(exc)

    if not is_batch:
        symbol = symbols[0]
        if symbol in errors:
            raise RuntimeError(f"{symbol}: {errors[symbol]}")
        return results[symbol]
    return {"results": results, "errors": errors}


@mcp.tool()
def fetch_ohlcv(symbol: str, lookback_days: int = 180, timeframe: str = "1Day") -> Dict[str, Any]:
    symbol = clean_symbol(symbol)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "feed": "iex",
    }
    headers = _get_headers()
    response = requests.get(f"{ALPACA_DATA_URL}/stocks/bars", headers=headers, params=params, timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    bars = response.json().get("bars", {}).get(symbol, [])
    if not bars:
        raise ValueError(f"No OHLCV data found for {symbol}.")
    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df = df.set_index("t").sort_index()
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    OHLCV_CACHE[symbol] = {"df": df, "timeframe": timeframe, "lookback_days": lookback_days}
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_fetched": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "columns": list(df.columns),
        "sample": df.head(3).reset_index().rename(columns={"t": "timestamp"}).astype(str).to_dict(orient="records"),
        "status": "ok",
    }


@mcp.tool()
def fetch_ohlcv_batch(symbols: List[str], lookback_days: int = 180, timeframe: str = "1Day") -> Dict[str, Any]:
    cleaned = [clean_symbol(item) for item in symbols]
    results = {symbol: fetch_ohlcv(symbol, lookback_days=lookback_days, timeframe=timeframe) for symbol in cleaned}
    return {"status": "ok", "symbols": cleaned, "bars_by_symbol": results, "cached_symbols": list(OHLCV_CACHE.keys())}


@mcp.tool()
def get_latest_prices(symbols: List[str]) -> Dict[str, Any]:
    cleaned = [clean_symbol(item) for item in symbols]
    prices = {symbol: round(float(_ensure_ohlcv_cache(symbol)["close"].iloc[-1]), 4) for symbol in cleaned}
    return {"prices": prices}


@mcp.tool()
def calculate_all_indicators(symbols: str | List[str]) -> Dict[str, Any]:
    requested_symbols = [symbols] if isinstance(symbols, str) else symbols
    indicators_by_symbol = {
        cleaned_symbol: build_indicator_bundle(_ensure_ohlcv_cache(cleaned_symbol))
        for cleaned_symbol in dict.fromkeys(clean_symbol(item) for item in requested_symbols)
    }
    if isinstance(symbols, str):
        cleaned_symbol = clean_symbol(symbols)
        return {"symbol": cleaned_symbol, "indicators": indicators_by_symbol[cleaned_symbol]}
    return {"indicators_by_symbol": indicators_by_symbol}


@mcp.tool()
def calculate_trend_indicators(symbol: str) -> Dict[str, Any]:
    df = _ensure_ohlcv_cache(symbol)
    return {"symbol": clean_symbol(symbol), "trend": build_indicator_bundle(df)["trend"]}


@mcp.tool()
def calculate_momentum_indicators(symbol: str) -> Dict[str, Any]:
    df = _ensure_ohlcv_cache(symbol)
    return {"symbol": clean_symbol(symbol), "momentum": build_indicator_bundle(df)["momentum"]}


@mcp.tool()
def calculate_volume_indicators(symbol: str) -> Dict[str, Any]:
    df = _ensure_ohlcv_cache(symbol)
    return {"symbol": clean_symbol(symbol), "volume": build_indicator_bundle(df)["volume"]}


@mcp.tool()
def calculate_volatility_indicators(symbol: str) -> Dict[str, Any]:
    df = _ensure_ohlcv_cache(symbol)
    return {"symbol": clean_symbol(symbol), "volatility": build_indicator_bundle(df)["volatility"]}


@mcp.tool()
def calculate_statistical_indicators(symbol: str) -> Dict[str, Any]:
    df = _ensure_ohlcv_cache(symbol)
    return {"symbol": clean_symbol(symbol), "statistical": build_indicator_bundle(df)["statistical"]}


@mcp.tool()
def generate_features(symbols: List[str]) -> Dict[str, Any]:
    """Generate cached OHLCV features for a list of tickers."""
    requested_symbols, _ = _normalize_symbol_argument(symbols)

    def generate_one(ticker: str) -> Dict[str, Any]:
        df = _ensure_ohlcv_cache(ticker)
        bundle = build_indicator_bundle(df)
        features = df.copy()
        features["return_1d"] = features["close"].pct_change().fillna(0.0)
        features["return_5d"] = features["close"].pct_change(5).fillna(0.0)
        for section in bundle.values():
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    features[f"indicator_{key}"] = value
        return {
            "symbol": ticker,
            "feature_count": len(features.columns),
            "features": features.tail(5).to_dict(orient="records"),
        }

    return _batch_tool_result(requested_symbols, True, generate_one)


@mcp.tool()
def train_prediction_model(symbols: List[str]) -> Dict[str, Any]:
    cleaned = [clean_symbol(item) for item in symbols]
    frames = [OHLCV_CACHE[symbol]["df"] for symbol in cleaned if symbol in OHLCV_CACHE]
    if not frames:
        raise RuntimeError("No cached OHLCV data available for model training.")
    combined = pd.concat(frames, axis=0, keys=cleaned)
    combined = combined.reset_index(level=0).rename(columns={"level_0": "symbol"})
    result = calculate_ml_probability(combined, forward_days=5)
    return {"symbols": cleaned, "model": result.get("model"), "prediction_probability": result.get("prediction_probability"), "ml_score": result.get("ml_score")}


@mcp.tool()
def predict_return_probability(symbols: List[str]) -> Dict[str, Any]:
    """Predict five-day return probability for a list of tickers."""
    requested_symbols, _ = _normalize_symbol_argument(symbols)

    def predict_one(ticker: str) -> Dict[str, Any]:
        df = _ensure_ohlcv_cache(ticker).reset_index().rename(columns={"t": "timestamp"})
        result = calculate_ml_probability(df, forward_days=5)
        return {
            "symbol": ticker,
            "prediction_probability": result.get("prediction_probability"),
            "ml_score": result.get("ml_score"),
            "model": result.get("model"),
        }

    return _batch_tool_result(requested_symbols, True, predict_one)


@mcp.tool()
def get_stock_news(symbol: str) -> Dict[str, Any]:
    symbol = clean_symbol(symbol)
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return {"symbol": symbol, "articles": [], "status": "neutral_fallback", "summary": "No news API key configured; using neutral score."}
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": f"{symbol} stock"},
            headers={"X-API-Key": api_key},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return {"symbol": symbol, "articles": data.get("articles", [])[:5], "status": "ok"}
    except Exception as exc:  # pragma: no cover
        return {"symbol": symbol, "articles": [], "status": "error", "error": str(exc)}


@mcp.tool()
def calculate_news_sentiment(symbols: List[str]) -> Dict[str, Any]:
    """Calculate news sentiment for a list of tickers."""
    requested_symbols, _ = _normalize_symbol_argument(symbols)

    def score_one(ticker: str) -> Dict[str, Any]:
        payload = get_stock_news(ticker)
        if payload.get("status") in {"neutral_fallback", "error"}:
            return {
                "symbol": ticker,
                "news_score": 50.0,
                "sentiment": "neutral",
                "articles": payload.get("articles", []),
                "status": payload.get("status"),
            }
        articles = payload.get("articles", [])
        if not articles:
            return {"symbol": ticker, "news_score": 50.0, "sentiment": "neutral", "articles": []}
        sentiment = 0.0
        for article in articles[:5]:
            title = (article.get("title") or "").lower()
            if any(word in title for word in ["beat", "surge", "upgrade", "growth", "strong", "bullish"]):
                sentiment += 1.0
            if any(word in title for word in ["drop", "miss", "downgrade", "weak", "bearish", "loss"]):
                sentiment -= 1.0
        score = 50 + (sentiment / max(1, len(articles))) * 50
        return {
            "symbol": ticker,
            "news_score": round(max(0.0, min(100.0, score)), 2),
            "sentiment": "positive" if score > 50 else "negative" if score < 50 else "neutral",
            "articles": articles[:5],
        }

    return _batch_tool_result(requested_symbols, True, score_one)


@mcp.tool()
def get_account_info() -> Dict[str, Any]:
    response = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=_get_headers(), timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_positions() -> Dict[str, Any]:
    response = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=_get_headers(), timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_open_orders() -> Dict[str, Any]:
    response = requests.get(f"{ALPACA_BASE_URL}/v2/orders?status=open", headers=_get_headers(), timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def calculate_portfolio_exposure() -> Dict[str, Any]:
    positions = get_positions()
    total_value = sum(float(item.get("market_value", 0.0) or 0.0) for item in positions)
    return {"positions": positions, "total_value": total_value, "exposure_percent": round((total_value / max(1.0, total_value)) * 100.0, 2)}


@mcp.tool()
def place_buy_order(symbol: str, qty: int) -> Dict[str, Any]:
    payload = {"symbol": clean_symbol(symbol), "qty": int(qty), "side": "buy", "type": "market", "time_in_force": "day"}
    response = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=_get_headers(), json=payload, timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def place_sell_order(symbol: str, qty: int) -> Dict[str, Any]:
    payload = {"symbol": clean_symbol(symbol), "qty": int(qty), "side": "sell", "type": "market", "time_in_force": "day"}
    response = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=_get_headers(), json=payload, timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def cancel_order(order_id: str) -> Dict[str, Any]:
    response = requests.delete(f"{ALPACA_BASE_URL}/v2/orders/{order_id}", headers=_get_headers(), timeout=20)
    if response.status_code == 401:
        raise PermissionError(
            "Alpaca rejected the configured credentials. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in your local .env file."
        )
    response.raise_for_status()
    return response.json()


@mcp.tool()
def cancel_all_orders(symbol: str | None = None) -> Dict[str, Any]:
    orders = get_open_orders()
    cancelled = []
    for item in orders:
        if symbol is not None and item.get("symbol") != clean_symbol(symbol):
            continue
        cancelled.append(cancel_order(item["id"]))
    return {"cancelled": cancelled, "symbol": symbol}


if __name__ == "__main__":
    mcp.run(transport="stdio")
