from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from trading_core import build_signal_summary, clean_symbol, clamp


@dataclass
class PortfolioPlan:
    buy_symbols: List[str]
    sell_symbols: List[str]
    hold_symbols: List[str]
    max_position_size: float
    min_cash_reserve: float


def build_portfolio_plan(scores: List[Dict[str, Any]], available_cash: float, current_positions: Optional[List[str]] = None, max_position_size: float = 0.10, min_cash_reserve: float = 0.0, max_new_buys: int = 4) -> PortfolioPlan:
    current_positions = set(str(item) for item in (current_positions or [])) if isinstance(current_positions, list) else set()
    ranked = sorted(scores, key=lambda item: item.get("final_score", 0.0), reverse=True)
    buy = []
    hold = []
    sell = []
    for item in ranked:
        symbol = clean_symbol(item.get("symbol", ""))
        decision = str(item.get("decision", "SELL")).upper()
        if decision in {"STRONG BUY", "BUY"} and symbol not in current_positions:
            buy.append(symbol)
        elif decision in {"SELL", "WEAK HOLD"}:
            sell.append(symbol)
        else:
            hold.append(symbol)

    buy = buy[:max_new_buys]
    if available_cash <= 0:
        buy = []
    return PortfolioPlan(
        buy_symbols=buy,
        sell_symbols=sell,
        hold_symbols=hold,
        max_position_size=max_position_size,
        min_cash_reserve=min_cash_reserve,
    )


def build_decision_payload(symbol: str, technical: float, ml: float, news: float, risk: float, rationale: str, decision: str) -> Dict[str, Any]:
    summary = build_signal_summary(technical, ml, news, risk)
    payload = {
        "symbol": clean_symbol(symbol),
        "technical_score": summary["technical_score"],
        "ml_score": summary["ml_score"],
        "news_score": summary["news_score"],
        "risk_score": summary["risk_score"],
        "final_score": summary["final_score"],
        "decision": decision or summary["decision"],
        "conviction": summary["conviction"],
        "rationale": rationale,
    }
    return payload
