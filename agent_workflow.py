from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from crewai_workflow import run_crewai_workflow
from trading_core import clean_symbol


def run_portfolio_cycle(
    symbols: List[str],
    lookback_days: int = 180,
    timeframe: str = "1Day",
    paper_trading_mode: bool = False,
    human_approval: bool = False,
    indicator_groups: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Run the CrewAI analysis through the local MCP server without submitting orders."""
    del paper_trading_mode, human_approval, indicator_groups
    cleaned_symbols = list(dict.fromkeys(clean_symbol(symbol) for symbol in symbols))
    crew_result = run_crewai_workflow(
        cleaned_symbols,
        lookback_days=lookback_days,
        timeframe=timeframe,
        allow_order_tools=False,
    )

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"portfolio_analysis_{datetime.utcnow():%Y%m%d_%H%M%S}.md"
    report_path.write_text(
        "# CrewAI portfolio analysis\n\n"
        f"- Symbols: {', '.join(cleaned_symbols)}\n"
        f"- Lookback: {lookback_days} days\n"
        f"- Timeframe: {timeframe}\n"
        "- Execution: analysis-only; order tools were not provided to agents\n\n"
        "## Crew output\n\n"
        f"{crew_result}\n",
        encoding="utf-8",
    )
    return {
        "symbols": cleaned_symbols,
        "crew_result": str(crew_result),
        "report_path": str(report_path),
        "order_execution_enabled": False,
    }
