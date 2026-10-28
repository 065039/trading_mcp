import pandas as pd

from trading_core import (
    build_indicator_bundle,
    build_signal_summary,
    clean_symbol,
    generate_order_proposals,
    make_prediction_features,
    portfolio_position_size,
)


def test_clean_symbol():
    assert clean_symbol(" aapl ") == "AAPL"
    assert clean_symbol("aapl.") == "AAPL"


def test_batch_ohlcv_parsing():
    df = pd.DataFrame(
        [
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 2000},
            {"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.5, "volume": 2500},
            {"open": 102.5, "high": 104.0, "low": 101.0, "close": 103.0, "volume": 2200},
        ],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    bundle = build_indicator_bundle(df)
    assert "trend" in bundle
    assert "momentum" in bundle
    assert "statistical" in bundle


def test_build_signal_summary():
    summary = build_signal_summary(technical_score=85, ml_score=75, news_score=70, risk_score=80)
    assert summary["decision"] in {"BUY", "STRONG BUY", "HOLD"}
    assert 0 <= summary["final_score"] <= 100


def test_prediction_features():
    df = pd.DataFrame(
        {
            "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "open": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "high": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "low": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "volume": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        },
        index=pd.date_range("2024-01-01", periods=12, freq="D"),
    )
    features = make_prediction_features(df)
    assert "return_5d" in features.columns
    assert "volume_ratio" in features.columns


def test_portfolio_position_size():
    assert portfolio_position_size(100000, 0.1) == 10000


def test_generate_order_proposals():
    proposals = generate_order_proposals([
        {"symbol": "AAPL", "side": "BUY", "final_score": 82, "price": 100, "max_qty": 20, "rationale": "Test buy"},
        {"symbol": "MSFT", "side": "SELL", "final_score": 30, "price": 200, "qty": 5, "rationale": "Test sell"},
    ], available_cash=5000, max_new_buys=4, max_new_sells=3)
    assert len(proposals) >= 1
    assert proposals[0].side in {"BUY", "SELL"}
