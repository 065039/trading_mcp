# Portfolio analysis

{
  "market_data": {
    "status": "ok",
    "symbols": [
      "AAPL",
      "MSFT"
    ],
    "cache": [
      "AAPL",
      "MSFT"
    ]
  },
  "technical_results": {
    "AAPL": {
      "trend": {},
      "momentum": {},
      "volume": {},
      "volatility": {},
      "statistical": {}
    },
    "MSFT": {
      "trend": {},
      "momentum": {},
      "volume": {},
      "volatility": {},
      "statistical": {}
    }
  },
  "quant_results": {
    "AAPL": {
      "features": {
        "symbol": "AAPL",
        "feature_count": 0,
        "features": []
      },
      "probability": {
        "symbol": "AAPL",
        "prediction_probability": 0.5,
        "ml_score": 50.0,
        "model": "fallback_rule_based"
      }
    },
    "MSFT": {
      "features": {
        "symbol": "MSFT",
        "feature_count": 0,
        "features": []
      },
      "probability": {
        "symbol": "MSFT",
        "prediction_probability": 0.5,
        "ml_score": 50.0,
        "model": "fallback_rule_based"
      }
    }
  },
  "news_results": {
    "AAPL": {
      "symbol": "AAPL",
      "news_score": 50.0,
      "sentiment": "neutral",
      "articles": []
    },
    "MSFT": {
      "symbol": "MSFT",
      "news_score": 50.0,
      "sentiment": "neutral",
      "articles": []
    }
  },
  "signal_results": {
    "AAPL": {
      "symbol": "AAPL",
      "technical_score": 50.0,
      "ml_score": 50.0,
      "news_score": 50.0,
      "risk_score": 60.0,
      "final_score": 51.0,
      "decision": "WEAK HOLD",
      "conviction": "Low",
      "rationale": "Deterministic weighted signal across technical, ML, news, and risk factors."
    },
    "MSFT": {
      "symbol": "MSFT",
      "technical_score": 50.0,
      "ml_score": 50.0,
      "news_score": 50.0,
      "risk_score": 60.0,
      "final_score": 51.0,
      "decision": "WEAK HOLD",
      "conviction": "Low",
      "rationale": "Deterministic weighted signal across technical, ML, news, and risk factors."
    }
  }
}
