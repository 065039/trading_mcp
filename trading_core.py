from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StockSignal:
    symbol: str
    technical_score: float = 0.0
    ml_score: float = 0.0
    news_score: float = 0.0
    risk_score: float = 0.0
    final_score: float = 0.0
    decision: str = "SELL"
    conviction: str = "Low"
    rationale: str = ""


@dataclass
class OrderProposal:
    symbol: str
    side: str
    qty: int
    reason: str
    estimated_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def clean_symbol(symbol: Any) -> str:
    if symbol is None:
        raise ValueError("Ticker symbol is required.")
    cleaned = str(symbol).strip().upper().replace(" ", "").replace(".", "")
    if not cleaned:
        raise ValueError("Ticker symbol is empty after cleaning.")
    return cleaned


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def normalize_0_100(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    numeric = safe_float(value, 0.0)
    if high == low:
        return 50.0
    return clamp((numeric - low) / (high - low) * 100.0, 0.0, 100.0)


def decision_from_score(score: float) -> str:
    if score >= 80:
        return "STRONG BUY"
    if score >= 70:
        return "BUY"
    if score >= 55:
        return "HOLD"
    if score >= 40:
        return "WEAK HOLD"
    return "SELL"


def conviction_from_score(score: float) -> str:
    if score >= 80:
        return "Very high"
    if score >= 70:
        return "High"
    if score >= 55:
        return "Moderate"
    if score >= 40:
        return "Low"
    return "Very low"


def build_signal_summary(technical_score: float, ml_score: float, news_score: float, risk_score: float) -> Dict[str, Any]:
    tech = clamp(technical_score, 0.0, 100.0)
    ml = clamp(ml_score, 0.0, 100.0)
    news = clamp(news_score, 0.0, 100.0)
    risk = clamp(risk_score, 0.0, 100.0)
    final = 0.60 * tech + 0.20 * ml + 0.10 * news + 0.10 * risk
    return {
        "technical_score": round(tech, 2),
        "ml_score": round(ml, 2),
        "news_score": round(news, 2),
        "risk_score": round(risk, 2),
        "final_score": round(final, 2),
        "decision": decision_from_score(final),
        "conviction": conviction_from_score(final),
    }


def calculate_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    if len(closes) < period + 1:
        return pd.Series(np.nan, index=closes.index)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def calculate_macd_series(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram}


def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_bollinger_width(df: pd.DataFrame, window: int = 20) -> pd.Series:
    if not {"close"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std(ddof=0)
    upper = rolling_mean + 2 * rolling_std
    lower = rolling_mean - 2 * rolling_std
    return ((upper - lower) / rolling_mean.replace(0, np.nan)) * 100.0


def calculate_donchian_channel(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
    if not {"high", "low"}.issubset(df.columns):
        return {"upper": pd.Series(np.nan, index=df.index), "lower": pd.Series(np.nan, index=df.index)}
    return {
        "upper": df["high"].rolling(window).max(),
        "lower": df["low"].rolling(window).min(),
    }


def calculate_returns_series(series: pd.Series, period: int = 1) -> pd.Series:
    return series.pct_change(periods=period).fillna(0.0)


def calculate_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def calculate_rolling_skewness(series: pd.Series, window: int = 20) -> pd.Series:
    if window <= 1:
        return pd.Series(0.0, index=series.index)
    return series.rolling(window).skew().fillna(0.0)


def build_indicator_bundle(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"trend": {}, "momentum": {}, "volume": {}, "volatility": {}, "statistical": {}}
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    volumes = df["volume"].astype(float)
    returns = calculate_returns_series(closes, 1)
    zscore = calculate_rolling_zscore(returns, 20)
    skewness = calculate_rolling_skewness(returns, 20)
    rsi = calculate_rsi_series(closes, 14)
    macd = calculate_macd_series(closes)
    roc = closes.pct_change(periods=12).fillna(0.0) * 100.0
    low_14 = lows.rolling(14).min()
    high_14 = highs.rolling(14).max()
    stoch_k = ((closes - low_14) / (high_14 - low_14).replace(0, np.nan)) * 100.0

    price_change = closes.diff(1)
    abs_change = price_change.abs()
    double_smooth = price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    abs_double_smooth = abs_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    tsi = (double_smooth / abs_double_smooth.replace(0, np.nan)) * 100.0

    typical_price = (highs + lows + closes) / 3.0
    money_flow = typical_price * volumes
    positive_flow = pd.Series(np.where((closes > closes.shift(1)).fillna(False), money_flow, 0.0), index=df.index)
    negative_flow = pd.Series(np.where((closes < closes.shift(1)).fillna(False), money_flow, 0.0), index=df.index)
    cumulative_money_flow = (positive_flow - negative_flow).cumsum().fillna(0.0)
    money_flow_index = pd.Series(np.nan, index=df.index)
    if len(df) >= 14:
        flow = money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_14 = money_flow.where(typical_price < typical_price.shift(1), 0.0)
        pos_flow = flow.rolling(14).sum()
        neg_flow = negative_14.rolling(14).sum()
        money_flow_index = 100 - (100 / (1 + pos_flow / neg_flow.replace(0, np.nan)))

    obv = np.sign(closes.diff().fillna(0)) * volumes
    obv = pd.Series(obv, index=df.index).cumsum().fillna(0.0)
    pvo = pd.Series(np.nan, index=df.index)
    if len(df) >= 26:
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        pvo = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)) * 100.0

    atr = calculate_atr_series(df, 14)
    natr = (atr / closes.replace(0, np.nan) * 100.0).fillna(0.0)
    bbw = calculate_bollinger_width(df, 20)
    dc = calculate_donchian_channel(df, 20)

    trend = {
        "ema20": safe_float(closes.ewm(span=20, adjust=False).mean().iloc[-1], 0.0),
        "ema50": safe_float(closes.ewm(span=50, adjust=False).mean().iloc[-1], 0.0),
        "ema200": safe_float(closes.ewm(span=200, adjust=False).mean().iloc[-1], 0.0),
        "adx": 25.0,
    }
    momentum = {
        "rsi": safe_float(rsi.iloc[-1], 50.0),
        "macd": safe_float(macd["macd_line"].iloc[-1], 0.0),
        "roc": safe_float(roc.iloc[-1], 0.0),
        "stochastic": safe_float(stoch_k.iloc[-1], 50.0),
        "tsi": safe_float(tsi.iloc[-1], 0.0),
    }
    volume = {
        "obv": safe_float(obv.iloc[-1], 0.0),
        "cmf": safe_float(cumulative_money_flow.iloc[-1], 0.0),
        "mfi": safe_float(money_flow_index.iloc[-1], 50.0),
        "pvo": safe_float(pvo.iloc[-1], 0.0),
    }
    volatility = {
        "atr": safe_float(atr.iloc[-1], 0.0),
        "natr": safe_float(natr.iloc[-1], 0.0),
        "bollinger_band_width": safe_float(bbw.iloc[-1], 0.0),
        "donchian_upper": safe_float(dc["upper"].iloc[-1], closes.iloc[-1]),
        "donchian_lower": safe_float(dc["lower"].iloc[-1], closes.iloc[-1]),
    }
    statistical = {
        "returns": safe_float(returns.iloc[-1], 0.0),
        "rolling_z_score": safe_float(zscore.iloc[-1], 0.0),
        "rolling_skewness": safe_float(skewness.iloc[-1], 0.0),
    }
    return {"trend": trend, "momentum": momentum, "volume": volume, "volatility": volatility, "statistical": statistical}


def score_indicator_bundle(indicator_bundle: Dict[str, Any]) -> float:
    try:
        trend = indicator_bundle.get("trend", {})
        momentum = indicator_bundle.get("momentum", {})
        volume = indicator_bundle.get("volume", {})
        volatility = indicator_bundle.get("volatility", {})
        statistical = indicator_bundle.get("statistical", {})

        rsi = clamp(safe_float(momentum.get("rsi", 50.0), 50.0), 0.0, 100.0)
        macd = safe_float(momentum.get("macd", 0.0), 0.0)
        roc = normalize_0_100(safe_float(momentum.get("roc", 0.0), 0.0), -20.0, 20.0)
        stoch = clamp(safe_float(momentum.get("stochastic", 50.0), 50.0), 0.0, 100.0)
        tsi = normalize_0_100(safe_float(momentum.get("tsi", 0.0), 0.0), -20.0, 20.0)
        ema20 = safe_float(trend.get("ema20", 0.0), 0.0)
        ema50 = safe_float(trend.get("ema50", 0.0), 0.0)
        ema200 = safe_float(trend.get("ema200", 0.0), 0.0)
        trend_rating = 50.0
        if ema20 > ema50 > ema200:
            trend_rating += 20.0
        if ema20 < ema50 < ema200:
            trend_rating -= 20.0
        if macd > 0:
            trend_rating += 10.0
        else:
            trend_rating -= 10.0
        trend_rating = clamp(trend_rating, 0.0, 100.0)

        volume_score = 50.0
        if safe_float(volume.get("mfi", 50.0), 50.0) > 50:
            volume_score += 15.0
        else:
            volume_score -= 15.0
        if safe_float(volume.get("obv", 0.0), 0.0) > 0:
            volume_score += 10.0
        volume_score = clamp(volume_score, 0.0, 100.0)

        volatility_score = 50.0
        if safe_float(volatility.get("natr", 0.0), 0.0) < 3.0:
            volatility_score += 10.0
        else:
            volatility_score -= 10.0
        volatility_score = clamp(volatility_score, 0.0, 100.0)

        return_score = clamp(normalize_0_100(safe_float(statistical.get("returns", 0.0), 0.0), -0.05, 0.05), 0.0, 100.0)
        score = 0.30 * trend_rating + 0.35 * ((rsi + stoch) / 2.0) + 0.10 * roc + 0.10 * tsi + 0.10 * volume_score + 0.05 * return_score
        return clamp(score, 0.0, 100.0)
    except Exception:
        return 50.0


def make_prediction_features(df: pd.DataFrame, indicator_bundle: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    result = df.copy()
    result["return_1d"] = result["close"].pct_change().fillna(0.0)
    result["return_5d"] = result["close"].pct_change(5).fillna(0.0)
    result["volatility_10d"] = result["close"].rolling(10).std().fillna(0.0)
    result["sma_10"] = result["close"].rolling(10).mean().fillna(0.0)
    result["sma_20"] = result["close"].rolling(20).mean().fillna(0.0)
    result["momentum_5d"] = result["close"].pct_change(5).fillna(0.0)
    result["volume_ratio"] = result["volume"].div(result["volume"].rolling(20).mean()).fillna(1.0)
    if indicator_bundle:
        for section_name, section in indicator_bundle.items():
            for key, value in section.items():
                if isinstance(value, (int, float)):
                    result[f"{section_name}_{key}"] = value
    return result.dropna(axis=1, how="all")


def calculate_ml_probability(df: pd.DataFrame, forward_days: int = 5) -> Dict[str, Any]:
    if df.empty or "close" not in df.columns:
        return {"prediction_probability": 0.5, "ml_score": 50.0, "model": "fallback_rule_based"}

    features = make_prediction_features(df)
    features["target"] = (features["close"].shift(-forward_days) / features["close"] - 1.0) > 0
    features = features.dropna(subset=["target"]).copy()
    feature_columns = [c for c in features.columns if c not in {"target", "symbol", "timestamp", "date"}]
    if len(features) < 10 or len(feature_columns) == 0:
        return {"prediction_probability": 0.5, "ml_score": 50.0, "model": "fallback_rule_based"}

    X = features[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = features["target"].astype(int)
    split_idx = max(1, int(len(X) * 0.8))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    try:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=120, random_state=42, min_samples_leaf=2)
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        pred = float(np.mean(probability)) if len(probability) else 0.5
        score = clamp(pred * 100.0, 0.0, 100.0)
        return {"prediction_probability": pred, "ml_score": score, "model": "RandomForestClassifier"}
    except Exception:
        momentum = safe_float(features["return_5d"].iloc[-1], 0.0)
        pred = clamp(0.5 + momentum * 2.5, 0.0, 1.0)
        score = clamp(pred * 100.0, 0.0, 100.0)
        return {"prediction_probability": float(pred), "ml_score": float(score), "model": "fallback_rule_based"}


def portfolio_position_size(portfolio_value: float, max_pct: float = 0.10) -> float:
    return max(0.0, float(portfolio_value) * max_pct)


def calculate_max_buy_qty(entry_price: float, available_cash: float, max_position_pct: float = 0.10) -> int:
    if entry_price <= 0:
        return 0
    max_value = min(float(available_cash), portfolio_position_size(available_cash, max_position_pct))
    return max(0, int(max_value // entry_price))


def generate_order_proposals(candidates: Iterable[Dict[str, Any]], available_cash: float, max_new_buys: int = 4, max_new_sells: int = 3) -> List[OrderProposal]:
    proposals: List[OrderProposal] = []
    buy_candidates = sorted([c for c in candidates if c.get("side") == "BUY"], key=lambda x: x.get("final_score", 0.0), reverse=True)[:max_new_buys]
    sell_candidates = sorted([c for c in candidates if c.get("side") == "SELL"], key=lambda x: x.get("final_score", 0.0), reverse=True)[:max_new_sells]
    cash_left = float(available_cash)

    for candidate in buy_candidates:
        price = safe_float(candidate.get("price"), 0.0)
        if price <= 0:
            continue
        max_qty = max(0, int(cash_left // price))
        qty = min(max_qty, int(candidate.get("max_qty", max_qty) or max_qty))
        if qty > 0:
            proposals.append(
                OrderProposal(
                    symbol=candidate["symbol"],
                    side="BUY",
                    qty=qty,
                    reason=candidate.get("rationale", "Score-based buy"),
                    estimated_price=price,
                    stop_loss=round(price * 0.95, 2),
                    take_profit=round(price * 1.08, 2),
                )
            )
            cash_left -= qty * price

    for candidate in sell_candidates:
        price = safe_float(candidate.get("price"), 0.0)
        qty = max(0, int(candidate.get("qty", 0)))
        if qty > 0 and price > 0:
            proposals.append(
                OrderProposal(
                    symbol=candidate["symbol"],
                    side="SELL",
                    qty=qty,
                    reason=candidate.get("rationale", "Risk-control sell"),
                    estimated_price=price,
                    stop_loss=None,
                    take_profit=None,
                )
            )
    return proposals


def write_markdown_report(path: str, title: str, content: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out)


def alarm(message: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"[{ts}] {message}"
