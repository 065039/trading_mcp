from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

from trading_config import load_config
from trading_core import clean_symbol


SERVER_PATH = Path(__file__).resolve().with_name("mcp_local_server.py")

DATA_TOOL_NAMES = {"fetch_ohlcv_batch", "get_latest_prices"}
TECHNICAL_TOOL_NAMES = {
    "calculate_all_indicators",
    "calculate_trend_indicators",
    "calculate_momentum_indicators",
    "calculate_volume_indicators",
    "calculate_volatility_indicators",
    "calculate_statistical_indicators",
}
QUANT_TOOL_NAMES = {"generate_features", "train_prediction_model", "predict_return_probability"}
NEWS_TOOL_NAMES = {"get_stock_news", "calculate_news_sentiment"}
PORTFOLIO_TOOL_NAMES = {
    "get_account_info",
    "get_positions",
    "get_open_orders",
    "calculate_portfolio_exposure",
}
ORDER_TOOL_NAMES = {"place_buy_order", "place_sell_order", "cancel_order", "cancel_all_orders"}


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if not name:
        raise RuntimeError(f"MCP adapter returned a tool without a name: {tool!r}")
    return str(name)


def _select_tools(
    tools: Sequence[Any],
    names: set[str],
    *,
    optional: bool = False,
) -> list[Any]:
    by_name = {_tool_name(tool): tool for tool in tools}
    missing = names - by_name.keys()
    if missing and not optional:
        raise RuntimeError(
            "Local MCP server is missing required tools: "
            + ", ".join(sorted(missing))
        )
    return [by_name[name] for name in sorted(names & by_name.keys())]


def _build_crew(
    symbols: list[str],
    lookback_days: int,
    timeframe: str,
    mcp_tools: Sequence[Any],
    *,
    allow_order_tools: bool,
    user_approved_orders: bool,
) -> Crew:
    config = load_config()
    llm = LLM(
        model=config.llm_model,
        base_url=config.ollama_base_url,
        temperature=0.1,
    )

    data_tools = _select_tools(mcp_tools, DATA_TOOL_NAMES)
    technical_tools = _select_tools(mcp_tools, TECHNICAL_TOOL_NAMES)
    quant_tools = _select_tools(mcp_tools, QUANT_TOOL_NAMES)
    news_tools = _select_tools(mcp_tools, NEWS_TOOL_NAMES)
    portfolio_tools = _select_tools(mcp_tools, PORTFOLIO_TOOL_NAMES)
    risk_tools = _select_tools(mcp_tools, PORTFOLIO_TOOL_NAMES)
    if allow_order_tools:
        if not config.paper_trading_mode:
            raise ValueError(
                "Order tools require PAPER_TRADING_MODE=true. "
                "No live-trading endpoint is supported."
            )
        if not user_approved_orders:
            raise ValueError(
                "Order tools require explicit user approval for this run."
            )
        if config.alpaca_base_url.rstrip("/") != "https://paper-api.alpaca.markets":
            raise ValueError(
                "Order tools are restricted to https://paper-api.alpaca.markets."
            )
        risk_tools += _select_tools(mcp_tools, ORDER_TOOL_NAMES)

    market_agent = Agent(
        role="Market Data Agent",
        goal="Fetch and validate all requested OHLCV data with one batch tool call.",
        backstory="You verify market data quality before downstream analysis starts.",
        tools=data_tools,
        verbose=True,
        llm=llm,
    )
    technical_agent = Agent(
        role="Technical Analyst",
        goal="Calculate and report the selected trend, momentum, volume, volatility, and statistical indicators.",
        backstory="You calculate indicator values from the MCP server's cached OHLCV bars.",
        tools=technical_tools,
        verbose=True,
        llm=llm,
    )
    quant_agent = Agent(
        role="Quant and ML Analyst",
        goal="Generate features and estimate the probability of a positive five-day forward return.",
        backstory="You use the model tools and report their outputs without implying certainty.",
        tools=quant_tools,
        verbose=True,
        llm=llm,
    )
    news_agent = Agent(
        role="News Analyst",
        goal="Retrieve recent stock news and report the normalized sentiment score.",
        backstory="You treat news as one input and use the configured neutral fallback when unavailable.",
        tools=news_tools,
        verbose=True,
        llm=llm,
    )
    signal_agent = Agent(
        role="Signal Analyst",
        goal="Explain component signals and apply the deterministic scores supplied in the task context; do not invent scores.",
        backstory="You communicate model and indicator evidence while Python remains the source of truth for arithmetic.",
        verbose=True,
        llm=llm,
    )
    portfolio_agent = Agent(
        role="Portfolio Strategist",
        goal="Review current account, holdings, open orders, and exposure before suggesting portfolio actions.",
        backstory="You respect cash reserves, position limits, and existing holdings.",
        tools=portfolio_tools,
        verbose=True,
        llm=llm,
    )
    risk_agent = Agent(
        role="Risk and Execution Agent",
        goal=(
            "Check account, positions, orders, and exposure. Never submit or cancel an order "
            "unless order tools were explicitly enabled for this run and the user has approved execution."
        ),
        backstory="You are the final conservative paper-trading gatekeeper.",
        tools=risk_tools,
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    symbols_text = ", ".join(symbols)
    fetch_task = Task(
        description=(
            f"Call fetch_ohlcv_batch exactly once for these symbols: {symbols_text}. "
            f"Use lookback_days={lookback_days} and timeframe={timeframe}. "
            "Report successful, missing, and invalid symbols. Do not fetch symbols one at a time."
        ),
        expected_output="Batch fetch summary with per-symbol status, date range, and bar count.",
        agent=market_agent,
    )
    technical_task = Task(
        description=(
            f"Using the cached data for {symbols_text}, call calculate_all_indicators once with the full "
            "symbol list in its symbols argument. Report all indicator groups by symbol, including any "
            "missing or invalid values."
        ),
        expected_output="Indicator values grouped by symbol and indicator family.",
        agent=technical_agent,
        context=[fetch_task],
    )
    quant_task = Task(
        description=(
            f"For the valid symbols in {symbols_text}, call generate_features once and "
            "predict_return_probability once, passing the symbols as a JSON array of ticker strings. "
            "Do not pass a ticker-to-null dictionary. Report each symbol's features, probability, "
            "score, and model name. Do not train on shuffled time-series data."
        ),
        expected_output="Per-symbol feature summary, model probability, normalized ML score, and model metadata.",
        agent=quant_agent,
        context=[fetch_task],
    )
    news_task = Task(
        description=(
            f"Call calculate_news_sentiment once for the valid symbols in {symbols_text}, "
            "passing them as a JSON array of ticker strings. Do not pass a ticker-to-null dictionary. "
            "Report neutral fallback explicitly when news is unavailable."
        ),
        expected_output="Per-symbol normalized news score, sentiment label, and fallback status.",
        agent=news_agent,
        context=[fetch_task],
    )
    signal_task = Task(
        description=(
            "Synthesize technical, ML, and news findings from the prior tasks. "
            "Do not invent or recalculate a final score; label any score not provided by deterministic "
            "Python code as unavailable. Explain evidence and uncertainty."
        ),
        expected_output="Per-symbol evidence summary without fabricated numeric final scores.",
        agent=signal_agent,
        context=[technical_task, quant_task, news_task],
    )
    portfolio_task = Task(
        description=(
            "Inspect account information, current positions, open orders, and portfolio exposure. "
            "Rank only the prior task's valid candidates. Respect 10% max position size and 10% "
            "minimum cash reserve; explain why constraints prevent any proposed action."
        ),
        expected_output="Portfolio-aware buy, hold, and sell recommendations with constraints.",
        agent=portfolio_agent,
        context=[signal_task],
    )
    risk_task = Task(
        description=(
            "Perform the final risk review of the portfolio recommendations. Check available cash, "
            "positions, open orders, and exposure. Do not submit orders unless this run explicitly "
            "enabled order tools and user approval has already been obtained. Otherwise return "
            "proposed actions only."
        ),
        expected_output="Risk checks, rejected actions, and paper-order proposals if applicable.",
        agent=risk_agent,
        context=[portfolio_task],
    )
    return Crew(
        agents=[
            market_agent,
            technical_agent,
            quant_agent,
            news_agent,
            signal_agent,
            portfolio_agent,
            risk_agent,
        ],
        tasks=[
            fetch_task,
            technical_task,
            quant_task,
            news_task,
            signal_task,
            portfolio_task,
            risk_task,
        ],
        process=Process.sequential,
        verbose=True,
    )


@contextmanager
def connected_crewai_crew(
    symbols: list[str],
    lookback_days: int = 180,
    timeframe: str = "1Day",
    *,
    allow_order_tools: bool = False,
    user_approved_orders: bool = False,
) -> Iterator[Crew]:
    """Keep the local MCP subprocess alive while its CrewAI tools are in use."""
    config = load_config()
    if not SERVER_PATH.is_file():
        raise FileNotFoundError(f"Local MCP server not found: {SERVER_PATH}")

    environment = os.environ.copy()
    environment.update(
        {
            "ALPACA_API_KEY": config.alpaca_api_key,
            "ALPACA_SECRET_KEY": config.alpaca_secret_key,
            "ALPACA_BASE_URL": config.alpaca_base_url,
            "NEWS_API_KEY": config.news_api_key,
        }
    )
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        env=environment,
    )
    cleaned_symbols = list(dict.fromkeys(clean_symbol(symbol) for symbol in symbols))
    if not cleaned_symbols:
        raise ValueError("At least one ticker symbol is required.")

    with MCPServerAdapter(server_parameters) as mcp_tools:
        crew = _build_crew(
            cleaned_symbols,
            lookback_days,
            timeframe,
            mcp_tools,
            allow_order_tools=allow_order_tools,
            user_approved_orders=user_approved_orders,
        )
        yield crew


def run_crewai_workflow(
    symbols: list[str],
    lookback_days: int = 180,
    timeframe: str = "1Day",
    *,
    allow_order_tools: bool = False,
    user_approved_orders: bool = False,
) -> Any:
    """Connect to the local MCP server and run the Crew before closing the connection."""
    with connected_crewai_crew(
        symbols,
        lookback_days,
        timeframe,
        allow_order_tools=allow_order_tools,
        user_approved_orders=user_approved_orders,
    ) as crew:
        return crew.kickoff()
