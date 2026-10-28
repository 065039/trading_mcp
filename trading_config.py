from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


@dataclass
class AppConfig:
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_secondary_api_key: str = ""
    alpaca_secondary_secret_key: str = ""
    alpaca_secondary_base_url: str = "https://paper-api.alpaca.markets/v2"
    news_api_key: str = ""
    llm_model: str = "ollama/qwen2.5:1.5b"
    ollama_base_url: str = "http://localhost:11434"
    paper_trading_mode: bool = False
    max_position_size: float = 0.10
    min_cash_reserve: float = 0.0
    max_new_buys: int = 4
    max_new_sells: int = 3
    max_portfolio_positions: int = 12
    lookback_days: int = 180
    timeframe: str = "1Day"


def load_config() -> AppConfig:
    return AppConfig(
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        alpaca_secondary_api_key=os.getenv("ALPACA_SECONDARY_API_KEY", ""),
        alpaca_secondary_secret_key=os.getenv("ALPACA_SECONDARY_SECRET_KEY", ""),
        alpaca_secondary_base_url=os.getenv(
            "ALPACA_SECONDARY_BASE_URL",
            "https://paper-api.alpaca.markets/v2",
        ),
        news_api_key=os.getenv("NEWS_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "ollama/qwen2.5:1.5b"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        paper_trading_mode=str(os.getenv("PAPER_TRADING_MODE", "false")).lower() == "true",
        max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.10")),
        min_cash_reserve=float(os.getenv("MIN_CASH_RESERVE", "0.0")),
        max_new_buys=int(os.getenv("MAX_NEW_BUYS", "4")),
        max_new_sells=int(os.getenv("MAX_NEW_SELLS", "3")),
        max_portfolio_positions=int(os.getenv("MAX_PORTFOLIO_POSITIONS", "12")),
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "180")),
        timeframe=os.getenv("TIMEFRAME", "1Day"),
    )
