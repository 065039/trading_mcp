from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

import portfolio_ledger
from trading_engine import (
    STUDENT_BUDGET,
    _clear_bar_cache,
    _fetch_batch_bars,
    _news_score,
    _reconcile_bucket_orders,
    _request_json,
    _trading_url,
    fetch_portfolio_dashboard,
    select_buy_proposals,
    _assign_recommendations,
    _build_proposals,
    _build_group_proposals,
    analyze_portfolio,
    analyze_group_portfolio,
    execute_approved_orders,
    select_sell_proposals,
)
from sp500_universe import (
    SCAN_CANDIDATE_COUNT,
    SP500_TICKERS,
    get_rotating_candidates,
    get_rotation_period_start,
)


class TradingEngineTests(unittest.TestCase):
    def test_news_api_failure_uses_neutral_score(self) -> None:
        with patch(
            "trading_engine._request_json",
            side_effect=RuntimeError("NewsAPI rate limit exceeded"),
        ):
            result = _news_score("AAPL", "test-news-key")

        self.assertEqual(result["news_score"], 50.0)
        self.assertEqual(result["sentiment"], "neutral")
        self.assertEqual(result["status"], "neutral_fallback")

    def test_selected_buy_proposals_filter_by_student_symbol_and_side(self) -> None:
        proposals = [
            {"student": "Kunal", "symbol": "AAPL", "side": "buy"},
            {"student": "Navroop", "symbol": "AAPL", "side": "buy"},
            {"student": "Kunal", "symbol": "MSFT", "side": "sell"},
        ]

        selected = select_buy_proposals(proposals, [("Kunal", "aapl"), ("Kunal", "MSFT")])

        self.assertEqual(
            selected,
            [{"student": "Kunal", "symbol": "AAPL", "side": "buy"}],
        )

    def test_selected_sell_proposals_match_student_symbol_and_account(self) -> None:
        proposals = [
            {
                "student": "Unassigned",
                "symbol": "AAPL",
                "side": "sell",
                "account_name": "secondary",
            },
            {"student": "Kunal", "symbol": "AAPL", "side": "sell", "account_name": "primary"},
        ]

        selected = select_sell_proposals(
            proposals,
            [("Unassigned", "aapl", "secondary")],
        )

        self.assertEqual(selected, [proposals[0]])

    def test_portfolio_dashboard_fetches_holdings_orders_and_reconciles(self) -> None:
        account = {"status": "ACTIVE", "equity": "120000", "cash": "50000", "buying_power": "50000"}
        positions = [{"symbol": "AAPL", "qty": "2", "unrealized_pl": "10"}]
        orders = [
            {"id": "open-1", "symbol": "MSFT", "status": "partially_filled"},
            {"id": "done-1", "symbol": "NVDA", "status": "filled"},
        ]
        with (
            patch(
                "trading_engine._request_json",
                side_effect=[account, positions, orders],
            ) as request,
            patch("trading_engine._reconcile_bucket_orders") as reconcile,
            patch("portfolio_ledger.get_bucket_state", return_value={"Kunal": {"holdings": []}}),
        ):
            dashboard = fetch_portfolio_dashboard("paper-key", "paper-secret")

        self.assertEqual(request.call_count, 3)
        reconcile.assert_called_once_with(
            request.call_args_list[0].kwargs["headers"],
            recent_orders=orders,
            account_name="primary",
            base_url="https://paper-api.alpaca.markets",
        )
        self.assertEqual(
            dashboard["positions"],
            [{**positions[0], "_account_name": "primary"}],
        )
        self.assertEqual(
            dashboard["open_orders"],
            [{**orders[0], "_account_name": "primary"}],
        )
        self.assertEqual(
            dashboard["orders"],
            [{**order, "_account_name": "primary"} for order in orders],
        )

    def test_trading_url_accepts_host_or_v2_base_without_duplicate_path(self) -> None:
        self.assertEqual(
            _trading_url("https://paper-api.alpaca.markets", "orders"),
            "https://paper-api.alpaca.markets/v2/orders",
        )
        self.assertEqual(
            _trading_url("https://paper-api.alpaca.markets/v2", "/orders"),
            "https://paper-api.alpaca.markets/v2/orders",
        )
        with self.assertRaises(ValueError):
            _trading_url("https://api.alpaca.markets/v2", "orders")

    def test_bundled_s_and_p_universe_has_at_least_500_unique_symbols(self) -> None:
        self.assertGreaterEqual(len(SP500_TICKERS), 500)
        self.assertEqual(len(SP500_TICKERS), len(set(SP500_TICKERS)))

    def test_rotating_candidates_change_every_two_days(self) -> None:
        period_start = get_rotation_period_start(
            pd.Timestamp("2026-10-16").to_pydatetime()
        )
        day_one = pd.Timestamp(period_start).to_pydatetime()
        day_two = day_one + pd.Timedelta(days=1)
        next_period = day_one + pd.Timedelta(days=2)
        candidates = get_rotating_candidates(day_one)

        self.assertEqual(len(candidates), SCAN_CANDIDATE_COUNT)
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertTrue(set(candidates).issubset(SP500_TICKERS))
        self.assertEqual(candidates, get_rotating_candidates(day_two))
        self.assertNotEqual(candidates, get_rotating_candidates(next_period))

    def test_bucket_performance_reports_realized_and_unrealized_profit(self) -> None:
        performance = portfolio_ledger.calculate_bucket_performance(
            {
                "Kunal": {
                    "initial_budget": 100_000,
                    "cash": 99_000,
                    "holdings": [
                        {"symbol": "AAPL", "qty": 10, "average_price": 100},
                    ],
                },
                "Navroop": {
                    "initial_budget": 100_000,
                    "cash": 100_500,
                    "holdings": [],
                },
            },
            {"AAPL": 120},
        )

        self.assertEqual(performance["Kunal"]["realized_profit"], 0)
        self.assertEqual(performance["Kunal"]["unrealized_profit"], 200)
        self.assertEqual(performance["Kunal"]["total_profit"], 200)
        self.assertEqual(performance["Navroop"]["realized_profit"], 500)
        self.assertEqual(performance["Navroop"]["total_profit"], 500)

    def test_buy_proposals_respect_per_position_cap_without_cash_reserve(self) -> None:
        candidates = [
            {
                "symbol": f"STK{index}",
                "decision": "BUY",
                "final_score": 90 - index,
                "price": 100.0,
                "atr": 2.0,
                "rationale": "test",
            }
            for index in range(6)
        ]

        proposals = _build_proposals(
            candidates,
            [],
            [],
            {"cash": STUDENT_BUDGET},
            budget=STUDENT_BUDGET,
            max_new_buys=6,
        )

        self.assertTrue(proposals)
        self.assertTrue(all(proposal.estimated_value <= 10_000 for proposal in proposals))
        self.assertLessEqual(sum(proposal.estimated_value for proposal in proposals), 100_000)

    def test_buy_proposals_can_deploy_full_student_budget(self) -> None:
        candidates = [
            {
                "symbol": f"STK{index}",
                "decision": "BUY",
                "final_score": 90 - index,
                "price": 1_000.0,
                "atr": 2.0,
                "rationale": "test",
            }
            for index in range(10)
        ]

        proposals = _build_proposals(
            candidates,
            [],
            [],
            {"cash": STUDENT_BUDGET},
            budget=STUDENT_BUDGET,
            max_new_buys=10,
        )

        self.assertEqual(sum(proposal.estimated_value for proposal in proposals), 100_000)

    def test_open_order_blocks_duplicate_symbol_buy(self) -> None:
        candidates = [
            {
                "symbol": "AAPL",
                "decision": "BUY",
                "final_score": 85,
                "price": 100.0,
                "atr": 2.0,
                "rationale": "test",
            }
        ]

        proposals = _build_proposals(
            candidates,
            [],
            [{"symbol": "AAPL"}],
            {"cash": STUDENT_BUDGET},
            budget=STUDENT_BUDGET,
        )

        self.assertEqual(proposals, [])

    def test_sell_proposal_cannot_exceed_existing_whole_share_quantity(self) -> None:
        proposals = _build_proposals(
            [
                {
                    "symbol": "AAPL",
                    "decision": "SELL",
                    "final_score": 20,
                    "price": 100.0,
                    "atr": 2.0,
                    "rationale": "test",
                }
            ],
            [{"symbol": "AAPL", "qty": "4.7", "market_value": "470"}],
            [],
            {"cash": 99_000},
            budget=STUDENT_BUDGET,
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].side, "sell")
        self.assertEqual(proposals[0].qty, 4)

    def test_analysis_fetches_account_and_enforces_student_watchlist(self) -> None:
        symbols = [f"STK{index}" for index in range(10)]
        frames = {symbol: object() for symbol in symbols}
        account = {"status": "ACTIVE", "equity": "100000", "cash": "100000", "buying_power": "100000"}
        with (
            patch("trading_engine._headers", return_value={}),
            patch("trading_engine._fetch_batch_bars", return_value=(frames, {})),
            patch("trading_engine._load_account_state", return_value=(account, [], [])),
            patch(
                "trading_engine._score_symbol",
                side_effect=lambda symbol, frame, key: {
                    "symbol": symbol,
                    "technical_score": 50,
                    "ml_score": 50,
                    "news_score": 50,
                    "risk_score": 50,
                    "final_score": 50,
                    "decision": "HOLD",
                    "prediction_probability": 0.5,
                    "model": "test",
                    "news_status": "neutral_fallback",
                    "indicators": {},
                    "price": 100,
                    "atr": 1,
                    "rationale": "test",
                },
            ),
        ):
            result = analyze_portfolio(
                student="Kunal",
                symbols=symbols,
                api_key="paper-key",
                secret_key="paper-secret",
            )

        self.assertEqual(result["student_budget"], 100_000)
        self.assertEqual(len(result["scores"]), 10)

    def test_group_analysis_aggregates_two_accounts_for_proposal_planning(self) -> None:
        symbol = get_rotating_candidates()[0]
        frame = pd.DataFrame({"close": [100.0]})
        primary = {
            "id": "primary-account",
            "status": "ACTIVE",
            "equity": "100000",
            "cash": "50000",
            "buying_power": "50000",
        }
        secondary = {
            "id": "secondary-account",
            "status": "ACTIVE",
            "equity": "200000",
            "cash": "150000",
            "buying_power": "150000",
        }
        fetch_bars = patch(
            "trading_engine._fetch_batch_bars",
            return_value=({symbol: frame}, {}),
        )
        with (
            fetch_bars,
            patch(
                "trading_engine._load_account_state",
                side_effect=[
                    (primary, [{"symbol": "AAPL", "market_value": "10000"}], []),
                    (secondary, [{"symbol": "MSFT", "market_value": "20000"}], []),
                ],
            ),
            patch("trading_engine._reconcile_bucket_orders"),
            patch(
                "portfolio_ledger.get_bucket_state",
                return_value={
                    student: {"holdings": [], "cash": STUDENT_BUDGET}
                    for student in ("Kunal", "Navroop", "Kavya")
                },
            ),
            patch(
                "trading_engine._score_symbol",
                return_value={
                    "symbol": symbol,
                    "technical_score": 50,
                    "ml_score": 50,
                    "news_score": 50,
                    "risk_score": 50,
                    "final_score": 50,
                    "decision": "HOLD",
                    "price": 100,
                    "atr": 1,
                    "rationale": "test",
                },
            ),
            patch(
                "trading_engine._assign_recommendations",
                return_value={
                    student: [] for student in ("Kunal", "Navroop", "Kavya")
                },
            ),
            patch("trading_engine._build_group_proposals", return_value=[]) as build_proposals,
        ):
            result = analyze_group_portfolio(
                api_key="primary-key",
                secret_key="primary-secret",
                secondary_api_key="secondary-key",
                secondary_secret_key="secondary-secret",
            )

        aggregate_account = build_proposals.call_args.args[5]
        scanned_symbols = fetch_bars.call_args.args[0]
        self.assertIn("AAPL", scanned_symbols)
        self.assertIn("MSFT", scanned_symbols)
        self.assertEqual(aggregate_account["equity"], 300_000)
        self.assertEqual(aggregate_account["cash"], 200_000)
        self.assertEqual(aggregate_account["buying_power"], 200_000)
        self.assertEqual(result["account"]["equity"], 300_000)
        self.assertEqual(
            {position["_account_name"] for position in result["positions"]},
            {"primary", "secondary"},
        )

    def test_analysis_rejects_invalid_student_or_watchlist_size(self) -> None:
        with self.assertRaises(ValueError):
            analyze_portfolio(
                student="not-a-student",
                symbols=["AAPL"] * 10,
                api_key="key",
                secret_key="secret",
            )
        with self.assertRaises(ValueError):
            analyze_portfolio(
                student="Kunal",
                symbols=["AAPL"],
                api_key="key",
                secret_key="secret",
            )

    def test_market_data_uses_one_multi_symbol_request(self) -> None:
        symbols = ["AAPL", "MSFT"]
        rows = [
            {"t": "2026-10-01T00:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000},
            {"t": "2026-10-02T00:00:00Z", "o": 101, "h": 103, "l": 100, "c": 102, "v": 1200},
        ]
        response = {"bars": {"AAPL": rows, "MSFT": rows}}
        with patch("trading_engine._request_json", return_value=response) as request:
            frames, errors = _fetch_batch_bars(symbols, 180, "1Day", {})

        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["params"]["symbols"], "AAPL,MSFT")
        self.assertEqual(set(frames), set(symbols))
        self.assertEqual(errors, {})

    def test_market_data_follows_pagination_for_all_requested_symbols(self) -> None:
        _clear_bar_cache()
        symbols = ["AAPL", "MSFT"]
        first_rows = [
            {"t": "2026-10-01T00:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000},
        ]
        second_rows = [
            {"t": "2026-10-02T00:00:00Z", "o": 101, "h": 103, "l": 100, "c": 102, "v": 1200},
        ]
        responses = [
            {"bars": {"AAPL": first_rows}, "next_page_token": "next-page"},
            {"bars": {"MSFT": second_rows}, "next_page_token": None},
        ]
        with patch(
            "trading_engine._request_json",
            side_effect=responses,
        ) as request:
            frames, errors = _fetch_batch_bars(symbols, 180, "1Day", {})

        self.assertEqual(request.call_count, 2)
        self.assertNotIn("page_token", request.call_args_list[0].kwargs["params"])
        self.assertEqual(
            request.call_args_list[1].kwargs["params"]["page_token"],
            "next-page",
        )
        self.assertEqual(set(frames), set(symbols))
        self.assertEqual(errors, {})
        _clear_bar_cache()

    def test_market_data_cache_avoids_repeating_identical_alpaca_requests(self) -> None:
        _clear_bar_cache()
        symbols = ["CACHE1"]
        rows = [
            {"t": "2026-10-01T00:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000},
        ]
        with patch(
            "trading_engine._request_json",
            return_value={"bars": {"CACHE1": rows}},
        ) as request:
            first, first_errors = _fetch_batch_bars(symbols, 180, "1Day", {"APCA-API-KEY-ID": "test"})
            second, second_errors = _fetch_batch_bars(symbols, 180, "1Day", {"APCA-API-KEY-ID": "test"})

        request.assert_called_once()
        self.assertEqual(first_errors, {})
        self.assertEqual(second_errors, {})
        self.assertEqual(first["CACHE1"].iloc[-1]["close"], 101)
        self.assertEqual(second["CACHE1"].iloc[-1]["close"], 101)
        _clear_bar_cache()

    def test_rate_limited_requests_retry_after_header(self) -> None:
        rate_limited = Mock(status_code=429, headers={"Retry-After": "0"}, content=b"")
        success = Mock(
            status_code=200,
            headers={},
            content=b"{}",
        )
        success.json.return_value = {"ok": True}
        with (
            patch("trading_engine.requests.request", side_effect=[rate_limited, success]) as request,
            patch("trading_engine.time.sleep") as sleep,
        ):
            result = _request_json("GET", "https://example.test", {})

        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_exhausted_rate_limit_names_limited_endpoint(self) -> None:
        responses = [
            Mock(status_code=429, headers={"Retry-After": "0"}, content=b"")
            for _ in range(4)
        ]
        with (
            patch("trading_engine.requests.request", side_effect=responses),
            patch("trading_engine.time.sleep"),
            patch("trading_engine.ALPACA_REQUEST_INTERVAL_SECONDS", 0.0),
            self.assertRaisesRegex(
                RuntimeError,
                r"Rate limit from data\.alpaca\.markets/v2/stocks/bars",
            ),
        ):
            _request_json(
                "GET",
                "https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL",
                {},
            )

    def test_pending_orders_are_reconciled_with_one_batch_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"):
                portfolio_ledger.record_pending_order(
                    order_id="order-1",
                    student="Kunal",
                    symbol="AAA",
                    side="buy",
                    qty=2,
                    reference_price=10,
                )
                portfolio_ledger.record_pending_order(
                    order_id="order-2",
                    student="Navroop",
                    symbol="BBB",
                    side="buy",
                    qty=3,
                    reference_price=20,
                )
                orders = [
                    {
                        "id": "order-1",
                        "status": "filled",
                        "filled_qty": "2",
                        "filled_avg_price": "10",
                    },
                    {
                        "id": "order-2",
                        "status": "filled",
                        "filled_qty": "3",
                        "filled_avg_price": "20",
                    },
                ]
                with patch("trading_engine._request_json", return_value=orders) as request:
                    _reconcile_bucket_orders({})
                buckets = portfolio_ledger.get_bucket_state()

        request.assert_called_once()
        self.assertEqual(buckets["Kunal"]["holdings"][0]["symbol"], "AAA")
        self.assertEqual(buckets["Navroop"]["holdings"][0]["symbol"], "BBB")

    def test_market_data_splits_large_universes_into_bounded_batches(self) -> None:
        symbols = [f"S{index:03}" for index in range(101)]
        rows = [
            {"t": "2026-10-01T00:00:00Z", "o": 100, "h": 102, "l": 99, "c": 101, "v": 1000},
        ]

        def response_for_batch(*args, **kwargs):
            requested = kwargs["params"]["symbols"].split(",")
            return {"bars": {symbol: rows for symbol in requested}}

        with patch("trading_engine._request_json", side_effect=response_for_batch) as request:
            frames, errors = _fetch_batch_bars(symbols, 180, "1Day", {})

        self.assertEqual(request.call_count, 2)
        self.assertEqual([len(call.kwargs["params"]["symbols"].split(",")) for call in request.call_args_list], [100, 1])
        self.assertEqual(set(frames), set(symbols))
        self.assertEqual(errors, {})

    def test_recommendations_are_ranked_and_assigned_distinctly(self) -> None:
        scores = {
            f"TICK{index}": {
                "symbol": f"TICK{index}",
                "decision": "BUY" if index < 6 else "SELL" if index == 12 else "HOLD",
                "final_score": 90 - index,
            }
            for index in range(15)
        }
        buckets = {
            student: {"holdings": []}
            for student in ("Kunal", "Navroop", "Kavya")
        }
        recommendations = _assign_recommendations(
            scores,
            buckets,
            [{"symbol": "TICK14"}],
            [{"symbol": "TICK13"}],
            recommendation_count=13,
        )
        picked = [
            item["symbol"]
            for student_recommendations in recommendations.values()
            for item in student_recommendations
        ]

        self.assertEqual([len(recommendations[student]) for student in ("Kunal", "Navroop", "Kavya")], [5, 4, 4])
        self.assertEqual(len(picked), len(set(picked)))
        self.assertNotIn("TICK13", picked)
        self.assertNotIn("TICK14", picked)
        self.assertIn("TICK12", picked)

    def test_execution_reprices_and_keeps_order_within_student_position_cap(self) -> None:
        analysis = {
            "student_budget": STUDENT_BUDGET,
            "student": "Kunal",
            "created_at": "2026-10-11T00:00:00Z",
            "scores": [],
            "proposals": [
                {
                    "symbol": "AAPL",
                    "student": "Kunal",
                    "side": "buy",
                    "qty": 100,
                    "reference_price": 100.0,
                    "estimated_value": 10_000.0,
                    "final_score": 80.0,
                    "reason": "test",
                }
            ],
        }
        account = {"status": "ACTIVE", "cash": "100000", "equity": "100000"}
        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"),
                patch("trading_engine._headers", return_value={}),
                patch("trading_engine._load_account_state", return_value=(account, [], [])),
                patch(
                    "trading_engine._request_json",
                    side_effect=[
                        {"AAPL": {"latestTrade": {"p": 120.0}}},
                        {"id": "paper-order-1", "symbol": "AAPL", "qty": "83", "side": "buy", "status": "accepted"},
                    ],
                ) as request,
                patch("trading_engine._write_execution_logs"),
            ):
                result = execute_approved_orders(
                    analysis,
                    api_key="paper-key",
                    secret_key="paper-secret",
                    approved=True,
                )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(request.call_args.kwargs["payload"]["qty"], 83)
        self.assertEqual(
            request.call_args.args[1],
            "https://paper-api.alpaca.markets/v2/orders",
        )

    def test_execution_routes_order_to_secondary_when_primary_has_no_capacity(self) -> None:
        analysis = {
            "created_at": "2026-10-16T00:00:00Z",
            "scores": [],
            "proposals": [
                {
                    "symbol": "AAPL",
                    "student": "Kunal",
                    "side": "buy",
                    "qty": 100,
                    "reference_price": 100.0,
                    "estimated_value": 10_000.0,
                    "final_score": 80.0,
                    "reason": "test",
                }
            ],
        }
        primary_state = (
            {"id": "primary-id", "status": "ACTIVE", "cash": "0", "equity": "100000", "buying_power": "0"},
            [{"symbol": "MSFT", "qty": "1000", "market_value": "100000"}],
            [],
        )
        secondary_state = (
            {"id": "secondary-id", "status": "ACTIVE", "cash": "200000", "equity": "200000", "buying_power": "200000"},
            [],
            [],
        )
        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"),
                patch(
                    "trading_engine._load_account_state",
                    side_effect=[primary_state, secondary_state],
                ),
                patch("trading_engine._reconcile_bucket_orders"),
                patch(
                    "trading_engine._request_json",
                    side_effect=[
                        {"AAPL": {"latestTrade": {"p": 100.0}}},
                        {"id": "secondary-order-1", "symbol": "AAPL", "qty": "100", "side": "buy", "status": "accepted"},
                    ],
                ) as request,
                patch("trading_engine._write_execution_logs"),
            ):
                result = execute_approved_orders(
                    analysis,
                    api_key="primary-key",
                    secret_key="primary-secret",
                    approved=True,
                    secondary_api_key="secondary-key",
                    secondary_secret_key="secondary-secret",
                    secondary_base_url="https://paper-api.alpaca.markets/v2",
                )
                pending_order = portfolio_ledger.get_bucket_state()["Kunal"]["pending_orders"][0]

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(result[0]["account"], "secondary")
        self.assertEqual(
            request.call_args.args[1],
            "https://paper-api.alpaca.markets/v2/orders",
        )
        self.assertEqual(
            request.call_args.kwargs["headers"]["APCA-API-KEY-ID"],
            "secondary-key",
        )
        self.assertEqual(pending_order["account_name"], "secondary")

    def test_execution_submits_unassigned_sell_to_the_holding_account(self) -> None:
        analysis = {
            "created_at": "2026-10-16T00:00:00Z",
            "scores": [],
            "proposals": [
                {
                    "symbol": "AAPL",
                    "student": "Unassigned",
                    "account_name": "secondary",
                    "side": "sell",
                    "qty": 10,
                    "reference_price": 120.0,
                    "estimated_value": 1_200.0,
                    "final_score": 20.0,
                    "reason": "sell signal",
                }
            ],
        }
        primary_state = (
            {"id": "primary-id", "status": "ACTIVE", "cash": "100000", "equity": "100000"},
            [],
            [],
        )
        secondary_state = (
            {"id": "secondary-id", "status": "ACTIVE", "cash": "100000", "equity": "100000"},
            [
                {
                    "symbol": "AAPL",
                    "qty": "10",
                    "market_value": "1200",
                    "avg_entry_price": "100",
                }
            ],
            [],
        )
        with TemporaryDirectory() as temp_dir:
            with (
                patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"),
                patch(
                    "trading_engine._load_account_state",
                    side_effect=[primary_state, secondary_state],
                ),
                patch("trading_engine._reconcile_bucket_orders"),
                patch(
                    "trading_engine._request_json",
                    return_value={
                        "id": "unassigned-sell-1",
                        "symbol": "AAPL",
                        "qty": "10",
                        "side": "sell",
                        "status": "accepted",
                    },
                ) as request,
                patch("trading_engine._write_execution_logs"),
            ):
                result = execute_approved_orders(
                    analysis,
                    api_key="primary-key",
                    secret_key="primary-secret",
                    approved=True,
                    secondary_api_key="secondary-key",
                    secondary_secret_key="secondary-secret",
                )
                buckets = portfolio_ledger.get_bucket_state()

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(result[0]["account"], "secondary")
        self.assertEqual(
            request.call_args.args[1],
            "https://paper-api.alpaca.markets/v2/orders",
        )
        self.assertTrue(all(not bucket["pending_orders"] for bucket in buckets.values()))

    def test_group_proposals_assign_each_buy_to_one_student_bucket(self) -> None:
        symbols = {
            "Kunal": ["AAPL", "MSFT"],
            "Navroop": ["NVDA", "AMZN"],
            "Kavya": ["GOOGL", "TSLA"],
        }
        scores = {
            symbol: {
                "symbol": symbol,
                "decision": "BUY",
                "final_score": 90.0 - index,
                "price": 100.0,
                "atr": 2.0,
                "rationale": "test",
            }
            for index, symbol in enumerate(
                ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
            )
        }
        buckets = {
            student: {
                "cash": STUDENT_BUDGET,
                "holdings": [],
                "pending_orders": [],
            }
            for student in ("Kunal", "Navroop", "Kavya")
        }
        with (
            patch("portfolio_ledger.get_open_bucket_symbol_pairs", return_value=set()),
            patch("portfolio_ledger.get_pending_buy_reservations", return_value=0.0),
        ):
            proposals = _build_group_proposals(
                scores,
                symbols,
                buckets,
                [],
                [],
                {"cash": 100_000, "buying_power": 100_000, "equity": 100_000},
            )

        self.assertTrue(proposals)
        self.assertEqual(len({proposal.symbol for proposal in proposals}), len(proposals))
        self.assertEqual(
            {proposal.student for proposal in proposals},
            {"Kunal", "Navroop", "Kavya"},
        )
        for student in ("Kunal", "Navroop", "Kavya"):
            student_orders = [p for p in proposals if p.student == student]
            self.assertLessEqual(sum(p.estimated_value for p in student_orders), 100_000)
            self.assertTrue(all(p.estimated_value <= 10_000 for p in student_orders))
        self.assertLessEqual(sum(p.estimated_value for p in proposals), 100_000)

    def test_group_buy_proposal_bypasses_score_threshold_for_selected_ticker(self) -> None:
        scores = {
            "LOW": {
                "symbol": "LOW",
                "decision": "SELL",
                "final_score": 20.0,
                "price": 100.0,
                "atr": 2.0,
                "rationale": "test",
            }
        }
        buckets = {
            student: {
                "cash": STUDENT_BUDGET,
                "holdings": [],
                "pending_orders": [],
            }
            for student in ("Kunal", "Navroop", "Kavya")
        }
        with (
            patch("portfolio_ledger.get_open_bucket_symbol_pairs", return_value=set()),
            patch("portfolio_ledger.get_pending_buy_reservations", return_value=0.0),
        ):
            proposals = _build_group_proposals(
                scores,
                {"Kunal": ["LOW"], "Navroop": [], "Kavya": []},
                buckets,
                [],
                [],
                {"cash": 100_000, "buying_power": 100_000, "equity": 100_000},
            )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].symbol, "LOW")
        self.assertEqual(proposals[0].side, "buy")
        self.assertIn("threshold bypassed", proposals[0].reason)

    def test_sell_strategy_proposes_exits_for_all_triggered_positions(self) -> None:
        scores = {
            "STOP": {
                "symbol": "STOP",
                "decision": "HOLD",
                "final_score": 50,
                "price": 74.0,
                "atr": 13.0,
                "rationale": "stop-triggered test",
            },
            "SIGNAL": {
                "symbol": "SIGNAL",
                "decision": "SELL",
                "final_score": 20,
                "price": 90.0,
                "atr": 1.0,
                "rationale": "sell-signal test",
            },
        }
        buckets = {
            student: {
                "cash": STUDENT_BUDGET,
                "holdings": [],
                "pending_orders": [],
            }
            for student in ("Kunal", "Navroop", "Kavya")
        }
        positions = [
            {
                "symbol": "STOP",
                "qty": "10",
                "avg_entry_price": "100",
                "market_value": "740",
                "_account_name": "secondary",
            },
            {
                "symbol": "SIGNAL",
                "qty": "5",
                "avg_entry_price": "100",
                "market_value": "450",
                "_account_name": "primary",
            },
        ]
        with (
            patch("portfolio_ledger.get_open_bucket_symbol_pairs", return_value=set()),
            patch("portfolio_ledger.get_pending_buy_reservations", return_value=0.0),
        ):
            proposals = _build_group_proposals(
                scores,
                {"Kunal": [], "Navroop": [], "Kavya": []},
                buckets,
                positions,
                [],
                {"cash": 100_000, "buying_power": 100_000, "equity": 100_000},
            )

        self.assertEqual(
            {(order.symbol, order.qty, order.student, order.account_name) for order in proposals},
            {
                ("STOP", 10, "Unassigned", "secondary"),
                ("SIGNAL", 5, "Unassigned", "primary"),
            },
        )
        self.assertIn("2-ATR stop-loss", proposals[0].reason)

    def test_group_buy_proposals_allow_up_to_ten_per_student(self) -> None:
        symbols = [f"LOW{index}" for index in range(12)]
        scores = {
            symbol: {
                "symbol": symbol,
                "decision": "HOLD",
                "final_score": 60.0 - index,
                "price": 9_000.0,
                "atr": 100.0,
                "rationale": "below-threshold test",
            }
            for index, symbol in enumerate(symbols)
        }
        buckets = {
            student: {
                "cash": STUDENT_BUDGET,
                "holdings": [],
                "pending_orders": [],
            }
            for student in ("Kunal", "Navroop", "Kavya")
        }
        with (
            patch("portfolio_ledger.get_open_bucket_symbol_pairs", return_value=set()),
            patch("portfolio_ledger.get_pending_buy_reservations", return_value=0.0),
        ):
            proposals = _build_group_proposals(
                scores,
                {"Kunal": symbols, "Navroop": [], "Kavya": []},
                buckets,
                [],
                [],
                {"cash": 300_000, "buying_power": 300_000, "equity": 300_000},
            )

        kunal_buys = [
            proposal
            for proposal in proposals
            if proposal.student == "Kunal" and proposal.side == "buy"
        ]
        self.assertEqual(len(kunal_buys), 10)
        self.assertTrue(all(proposal.estimated_value <= 10_000 for proposal in kunal_buys))

    def test_shared_account_capacity_is_distributed_across_students(self) -> None:
        student_symbols = {
            student: [f"{student[:2].upper()}{index}" for index in range(10)]
            for student in ("Kunal", "Navroop", "Kavya")
        }
        scores = {
            symbol: {
                "symbol": symbol,
                "decision": "BUY",
                "final_score": 100.0 - index,
                "price": 1_000.0,
                "atr": 10.0,
                "rationale": "test",
            }
            for symbols in student_symbols.values()
            for index, symbol in enumerate(symbols)
        }
        buckets = {
            student: {
                "cash": STUDENT_BUDGET,
                "holdings": [],
                "pending_orders": [],
            }
            for student in ("Kunal", "Navroop", "Kavya")
        }
        with (
            patch("portfolio_ledger.get_open_bucket_symbol_pairs", return_value=set()),
            patch("portfolio_ledger.get_pending_buy_reservations", return_value=0.0),
        ):
            proposals = _build_group_proposals(
                scores,
                student_symbols,
                buckets,
                [],
                [],
                {"cash": 100_000, "buying_power": 100_000, "equity": 100_000},
            )

        buys_by_student = {
            student: [
                proposal
                for proposal in proposals
                if proposal.student == student and proposal.side == "buy"
            ]
            for student in student_symbols
        }
        self.assertEqual(
            {student: len(student_buys) for student, student_buys in buys_by_student.items()},
            {"Kunal": 3, "Navroop": 3, "Kavya": 3},
        )
        self.assertLessEqual(
            sum(proposal.estimated_value for proposal in proposals),
            100_000,
        )

    def test_ledger_reconciles_filled_order_to_student_bucket(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"):
                portfolio_ledger.record_pending_order(
                    order_id="order-1",
                    student="Navroop",
                    symbol="NVDA",
                    side="buy",
                    qty=10,
                    reference_price=100,
                )
                portfolio_ledger.apply_order_update(
                    {
                        "id": "order-1",
                        "status": "filled",
                        "filled_qty": "10",
                        "filled_avg_price": "101",
                    }
                )
                buckets = portfolio_ledger.get_bucket_state()

        self.assertEqual(buckets["Navroop"]["cash"], 100_000 - 1_010)
        self.assertEqual(buckets["Navroop"]["holdings"][0]["symbol"], "NVDA")
        self.assertEqual(buckets["Navroop"]["holdings"][0]["qty"], 10)
        self.assertEqual(buckets["Kunal"]["holdings"], [])

    def test_ledger_records_completed_fill_above_remaining_bucket_cash(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch.object(portfolio_ledger, "DB_PATH", Path(temp_dir) / "ledger.sqlite3"):
                with portfolio_ledger._connect() as connection:
                    connection.execute(
                        "UPDATE student_buckets SET cash = 50 WHERE student = ?",
                        ("Navroop",),
                    )
                portfolio_ledger.record_pending_order(
                    order_id="over-budget-fill",
                    student="Navroop",
                    symbol="NVDA",
                    side="buy",
                    qty=1,
                    reference_price=49,
                )

                portfolio_ledger.apply_order_update(
                    {
                        "id": "over-budget-fill",
                        "status": "filled",
                        "filled_qty": "1",
                        "filled_avg_price": "51",
                    }
                )
                buckets = portfolio_ledger.get_bucket_state()

        self.assertEqual(buckets["Navroop"]["cash"], -1)
        self.assertEqual(buckets["Navroop"]["holdings"][0]["symbol"], "NVDA")
        self.assertEqual(buckets["Navroop"]["holdings"][0]["qty"], 1)
        self.assertEqual(buckets["Navroop"]["pending_orders"], [])


if __name__ == "__main__":
    unittest.main()
