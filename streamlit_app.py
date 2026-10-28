from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from portfolio_ledger import calculate_bucket_performance, get_bucket_state
from trading_config import load_config
from trading_engine import (
    MAX_GROUP_BUYS_PER_STUDENT,
    STUDENTS,
    analyze_group_portfolio,
    credential_fingerprint,
    execute_approved_orders,
    fetch_portfolio_dashboard,
    select_buy_proposals,
    select_sell_proposals,
)

config = load_config()

st.set_page_config(page_title="Trading Agents Paper Desk", layout="wide")
st.title("Trade Algorithmically using Trading Agents")
st.warning(
    "The group portfolio uses one or two Alpaca paper accounts. Each student has a tracked virtual "
    "$100,000 bucket ($300,000 combined); orders are routed to the configured account with "
    "available buying power."
)

with st.sidebar:
    secondary_credentials_partial = bool(config.alpaca_secondary_api_key) != bool(
        config.alpaca_secondary_secret_key
    )
    credentials_configured = bool(
        config.alpaca_api_key
        and config.alpaca_secret_key
        and not secondary_credentials_partial
    )
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        st.error(
            "Alpaca credentials are not configured. Add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY to the project .env file, then restart Streamlit."
        )
    elif secondary_credentials_partial:
        st.error(
            "Configure both secondary Alpaca credentials in .env, or leave both empty."
        )
    elif config.alpaca_secondary_api_key:
        st.success("Primary and secondary Alpaca paper accounts are configured.")

    st.header("S&P 500 recommendations")
    st.caption(
        "The app scores a rotating 60-stock S&P 500 slice plus current holdings and ranks "
        "available symbols for each student."
    )
    lookback_days = st.number_input(
        "Lookback days",
        min_value=60,
        max_value=500,
        value=180,
        step=30,
    )
    timeframe = st.selectbox("Timeframe", ["1Day"], index=0)

if st.button(
    "Analyze group portfolio",
    type="primary",
    disabled=not credentials_configured,
):
    try:
        analysis = analyze_group_portfolio(
            api_key=config.alpaca_api_key,
            secret_key=config.alpaca_secret_key,
            recommendation_count=10,
            lookback_days=int(lookback_days),
            timeframe=timeframe,
            news_api_key=config.news_api_key,
            base_url=config.alpaca_base_url,
            secondary_api_key=config.alpaca_secondary_api_key,
            secondary_secret_key=config.alpaca_secondary_secret_key,
            secondary_base_url=config.alpaca_secondary_base_url,
        )
        st.session_state["trading_analysis"] = analysis
        st.session_state["trading_execution"] = None
        st.session_state["trading_analysis_owner"] = {
            "lookback_days": int(lookback_days),
            "timeframe": timeframe,
            "credential_fingerprint": credential_fingerprint(
                config.alpaca_api_key,
                config.alpaca_secret_key,
                config.alpaca_base_url,
                config.alpaca_secondary_api_key,
                config.alpaca_secondary_secret_key,
                config.alpaca_secondary_base_url,
            ),
        }
        st.session_state["portfolio_snapshot"] = None
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

trade_tab, portfolio_tab = st.tabs(
    ["Recommendations & Alpaca Trading", "Current Holdings, P&L & Trade Status"]
)

analysis: Dict[str, Any] | None = st.session_state.get("trading_analysis")

with trade_tab:
    if not analysis:
        st.info("Run an analysis to generate student recommendations.")
    else:
        analysis_owner = st.session_state.get("trading_analysis_owner", {})
        approval_context_matches = (
            analysis_owner.get("lookback_days") == int(lookback_days)
            and analysis_owner.get("timeframe") == timeframe
            and analysis_owner.get("credential_fingerprint")
            == credential_fingerprint(
                config.alpaca_api_key,
                config.alpaca_secret_key,
                config.alpaca_base_url,
                config.alpaca_secondary_api_key,
                config.alpaca_secondary_secret_key,
                config.alpaca_secondary_base_url,
            )
        )
        if not approval_context_matches:
            st.warning(
                "Analysis settings or credentials changed since this analysis. "
                "Run a new analysis before placing orders."
            )

        budgets = analysis.get("student_budgets")
        if not budgets:
            st.warning("This is an older analysis result. Run a new group analysis to refresh it.")
        else:
            buckets = get_bucket_state()
            current_prices = {
                str(position.get("symbol", "")).upper(): (
                    abs(float(position.get("market_value", 0.0) or 0.0))
                    / abs(float(position.get("qty", 0.0) or 0.0))
                )
                for position in analysis.get("positions", [])
                if float(position.get("qty", 0.0) or 0.0) != 0
            }
            current_prices.update(
                {
                    str(score["symbol"]).upper(): float(score["price"])
                    for score in analysis.get("scores", [])
                    if score.get("price") is not None
                }
            )
            bucket_performance = calculate_bucket_performance(buckets, current_prices)
            overall_profit = sum(
                item["total_profit"] for item in bucket_performance.values()
            )
            overall_budget = sum(
                item["initial_budget"] for item in buckets.values()
            )
            overall_return = overall_profit / overall_budget if overall_budget else 0.0
            bucket_cash_available = sum(
                float(bucket["cash"])
                - sum(
                    float(order["reserved_cash"])
                    for order in bucket["pending_orders"]
                    if order["side"] == "buy"
                )
                for bucket in buckets.values()
            )
            bucket_capital_deployed = sum(
                item["market_value"] for item in bucket_performance.values()
            ) + sum(
                float(order["reserved_cash"])
                for bucket in buckets.values()
                for order in bucket["pending_orders"]
                if order["side"] == "buy"
            )

            st.subheader("Group trading summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Combined student budgets", f"${overall_budget:,.2f}")
            col2.metric(
                "Bucket capital deployed",
                f"${bucket_capital_deployed:,.2f}",
                help="Virtual holdings plus cash committed to pending student buy orders.",
            )
            col3.metric(
                "Available student bucket cash",
                f"${bucket_cash_available:,.2f}",
            )
            st.caption(
                f"Recommendation universe: 60 rotating S&P 500 stocks for the two-day "
                f"period starting {analysis.get('rotation_period', 'today')} UTC; "
                "current holdings are also scored."
            )
            overdrawn_buckets = [
                (student, -float(bucket["cash"]))
                for student, bucket in buckets.items()
                if float(bucket["cash"]) < -1e-6
            ]
            if overdrawn_buckets:
                details = ", ".join(
                    f"{student}: ${amount:,.2f}"
                    for student, amount in overdrawn_buckets
                )
                st.warning(
                    "A completed Alpaca market-order fill exceeded virtual bucket "
                    f"cash ({details}). The fill has been recorded; new buys for "
                    "those buckets remain unavailable until their cash is positive."
                )
            st.metric(
                "Overall bucket profit / loss",
                f"${overall_profit:,.2f}",
                delta=f"{overall_return:.2%} vs. start",
                delta_color="normal",
            )

            st.subheader("Alpaca Trading — select recommended stocks to buy")
            st.caption(
                "Select only the recommended stocks you want to buy. Score thresholds are "
                "bypassed for these candidates, but all orders remain subject to paper-account "
                "buying power and per-student position limits."
            )
            recommendations = analysis.get("recommendations", {})
            previous_execution = st.session_state.get("trading_execution")
            proposal_by_bucket_symbol = {
                (str(proposal.get("student", "")), str(proposal.get("symbol", "")).upper()): proposal
                for proposal in analysis.get("proposals", [])
                if str(proposal.get("side", "")).lower() == "buy"
            }
            selected_keys: list[tuple[str, str]] = []
            for student in STUDENTS:
                picks = recommendations.get(student, [])
                with st.expander(
                    f"{student} — {len(picks)} recommendations",
                    expanded=True,
                ):
                    if not picks:
                        st.info("No available recommendations for this bucket.")
                    for item in picks:
                        symbol = str(item["symbol"]).upper()
                        proposal = proposal_by_bucket_symbol.get((student, symbol))
                        price_text = (
                            f"est. ${float(proposal['estimated_value']):,.2f}"
                            if proposal
                            else "currently unavailable under portfolio or cash limits"
                        )
                        label = (
                            f"Buy {symbol} — score {float(item['final_score']):.1f}, "
                            f"{item['decision']}, {price_text}"
                        )
                        checkbox_key = (
                            f"buy_{analysis['created_at']}_{student}_{symbol}"
                        )
                        selected = st.checkbox(
                            label,
                            key=checkbox_key,
                            disabled=(
                                previous_execution is not None
                                or proposal is None
                            ),
                        )
                        if selected and proposal is not None:
                            selected_keys.append((student, symbol))

            sell_proposals = [
                proposal
                for proposal in analysis.get("proposals", [])
                if str(proposal.get("side", "")).lower() == "sell"
            ]
            selected_sell_keys: list[tuple[str, str, str]] = []
            st.subheader("Sell/exit signals for current holdings")
            st.caption(
                "A sell is proposed when the score is SELL, price reaches 2 ATR below "
                "average entry, or price reaches 3 ATR above average entry. Select exits "
                "individually; nothing is sold without approval."
            )
            if not sell_proposals:
                st.info("No current holdings triggered a sell/exit signal in this analysis.")
            for proposal in sell_proposals:
                student = str(proposal.get("student", "Unassigned"))
                symbol = str(proposal.get("symbol", "")).upper()
                account_name = str(proposal.get("account_name", "primary"))
                selected = st.checkbox(
                    f"Sell {symbol} ({proposal['qty']} shares) — {student}, "
                    f"{account_name}; {proposal['reason']}",
                    key=(
                        f"sell_{analysis['created_at']}_{student}_"
                        f"{account_name}_{symbol}"
                    ),
                    disabled=previous_execution is not None,
                )
                if selected:
                    selected_sell_keys.append((student, symbol, account_name))

            st.subheader("Submit selected paper trades")
            st.caption(
                f"Up to {MAX_GROUP_BUYS_PER_STUDENT} new buys per student per run. "
                "Existing holdings, open orders, live account buying power, and student "
                "position caps are checked again before submission. No cash reserve is kept."
            )
            enable_paper_trading = st.checkbox(
                "Enable Alpaca paper-order submission",
                value=False,
                key=f"enable_paper_{analysis['created_at']}",
            )
            selected_proposals = select_buy_proposals(
                analysis.get("proposals", []),
                selected_keys,
            )
            selected_proposals.extend(
                select_sell_proposals(
                    analysis.get("proposals", []),
                    selected_sell_keys,
                )
            )
            st.write(
                f"Selected: **{len(selected_proposals)}** paper trade(s)"
            )
            approve_selected = st.checkbox(
                "I reviewed and approve only the selected paper buys and sells.",
                key=f"approve_selected_{analysis['created_at']}",
                disabled=not enable_paper_trading or not approval_context_matches,
            )
            if st.button(
                "Submit selected paper trades",
                type="primary",
                disabled=(
                    not enable_paper_trading
                    or not approve_selected
                    or not selected_proposals
                    or previous_execution is not None
                    or not approval_context_matches
                ),
                key=f"submit_selected_{analysis['created_at']}",
            ):
                try:
                    selected_analysis = dict(analysis)
                    selected_analysis["proposals"] = selected_proposals
                    results = execute_approved_orders(
                        selected_analysis,
                        api_key=config.alpaca_api_key,
                        secret_key=config.alpaca_secret_key,
                        approved=True,
                        base_url=config.alpaca_base_url,
                        secondary_api_key=config.alpaca_secondary_api_key,
                        secondary_secret_key=config.alpaca_secondary_secret_key,
                        secondary_base_url=config.alpaca_secondary_base_url,
                    )
                    st.session_state["trading_execution"] = results
                    st.success("Selected paper orders were submitted or rejected; see results below.")
                except Exception as exc:
                    st.error(f"Paper-order submission failed: {exc}")

            execution = st.session_state.get("trading_execution")
            if execution is not None:
                st.subheader("Latest execution results")
                st.dataframe(execution, width="stretch")

            st.subheader("Deterministic scores")
            score_rows = [
                {
                    "Symbol": item["symbol"],
                    "Technical": item["technical_score"],
                    "ML": item["ml_score"],
                    "News": item["news_score"],
                    "Risk": item["risk_score"],
                    "Final": item["final_score"],
                    "Decision": item["decision"],
                    "Price": item["price"],
                    "News status": item["news_status"],
                    "Student bucket": next(
                        (
                            student
                            for student, symbols in analysis["student_watchlists"].items()
                            if item["symbol"] in symbols
                        ),
                        "",
                    ),
                }
                for item in analysis.get("scores", [])
            ]
            st.dataframe(score_rows, width="stretch")
            if analysis.get("errors"):
                st.warning("Some symbols had data or analysis issues.")
                st.json(analysis["errors"])

            run_stamp = analysis["created_at"].replace(":", "").replace("-", "")
            report_path = Path("reports") / f"analysis_group_{run_stamp}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# Paper trading analysis\n\n"
                "- Portfolio: shared group account\n"
                "- Per-student virtual budget: $100,000\n"
                f"- Run time: {analysis['created_at']}\n\n"
                "## Scores and proposals\n\n"
                f"```json\n{json.dumps({'scores': analysis['scores'], 'proposals': analysis['proposals'], 'errors': analysis['errors']}, indent=2, default=str)}\n```\n",
                encoding="utf-8",
            )
            st.caption(f"Analysis report: {report_path}")

with portfolio_tab:
    st.subheader("Current Alpaca paper-account holdings and trades")
    st.caption(
        "Refresh to fetch current Alpaca account positions and recent order statuses. "
        "Virtual bucket P&L is calculated separately from the shared broker account."
    )
    if not credentials_configured:
        st.error("Configure Alpaca paper credentials in .env to load current portfolio data.")
    elif st.button("Refresh holdings and trade status", key="refresh_portfolio"):
        try:
            st.session_state["portfolio_snapshot"] = fetch_portfolio_dashboard(
                config.alpaca_api_key,
                config.alpaca_secret_key,
                base_url=config.alpaca_base_url,
                secondary_api_key=config.alpaca_secondary_api_key,
                secondary_secret_key=config.alpaca_secondary_secret_key,
                secondary_base_url=config.alpaca_secondary_base_url,
            )
        except Exception as exc:
            st.error(f"Could not refresh Alpaca portfolio: {exc}")

    snapshot = st.session_state.get("portfolio_snapshot")
    if snapshot:
        st.caption(f"Last refreshed: {snapshot['refreshed_at']}")
        positions = snapshot["positions"]
        orders = snapshot["orders"]
        open_orders = snapshot["open_orders"]

        total_unrealized_pl = sum(
            float(position.get("unrealized_pl", 0.0) or 0.0)
            for position in positions
        )
        total_unrealized_pct = (
            sum(
                float(position.get("cost_basis", 0.0) or 0.0)
                for position in positions
            )
        )
        total_unrealized_pct = (
            total_unrealized_pl / total_unrealized_pct
            if total_unrealized_pct
            else 0.0
        )
        col1, col2 = st.columns(2)
        col1.metric(
            "Broker unrealized P/L",
            f"${total_unrealized_pl:,.2f}",
            delta=f"{total_unrealized_pct:.2%}",
        )
        col2.metric("Open orders", len(open_orders))

        st.markdown("#### Alpaca holdings")
        position_rows = [
            {
                "Paper account": position.get("_account_name", "primary"),
                "Symbol": position.get("symbol"),
                "Qty": position.get("qty"),
                "Avg entry": position.get("avg_entry_price"),
                "Current price": position.get("current_price"),
                "Market value": position.get("market_value"),
                "Unrealized P/L": position.get("unrealized_pl"),
                "Unrealized P/L %": (
                    100.0 * float(position.get("unrealized_plpc", 0.0) or 0.0)
                ),
            }
            for position in positions
        ]
        if position_rows:
            st.dataframe(position_rows, width="stretch")
        else:
            st.info("The configured Alpaca paper accounts currently have no positions.")

        buckets = snapshot["buckets"]
        prices = {
            str(position.get("symbol", "")).upper(): float(
                position.get("current_price", 0.0) or 0.0
            )
            for position in positions
            if float(position.get("current_price", 0.0) or 0.0) > 0
        }
        bucket_performance = calculate_bucket_performance(buckets, prices)
        st.markdown("#### Student virtual bucket P&L")
        bucket_rows = []
        virtual_holding_rows = []
        for student, bucket in buckets.items():
            performance = bucket_performance[student]
            bucket_rows.append(
                {
                    "Student": student,
                    "Virtual cash": round(performance["cash"], 2),
                    "Market value": round(performance["market_value"], 2),
                    "Total value": round(performance["total_value"], 2),
                    "Realized P/L": round(performance["realized_profit"], 2),
                    "Unrealized P/L": round(performance["unrealized_profit"], 2),
                    "Total P/L": round(performance["total_profit"], 2),
                    "Holdings": ", ".join(
                        f"{holding['symbol']} ({holding['qty']:g})"
                        for holding in bucket["holdings"]
                    )
                    or "None",
                }
            )
            for holding in bucket["holdings"]:
                symbol = str(holding["symbol"]).upper()
                quantity = float(holding["qty"])
                average_price = float(holding["average_price"])
                current_price = float(prices.get(symbol, average_price))
                virtual_holding_rows.append(
                    {
                        "Student": student,
                        "Paper account": holding.get("account_name", "primary"),
                        "Symbol": symbol,
                        "Qty": quantity,
                        "Average cost": average_price,
                        "Current price": current_price,
                        "Market value": quantity * current_price,
                        "Unrealized P/L": quantity * (current_price - average_price),
                    }
                )
        st.dataframe(bucket_rows, width="stretch")
        st.markdown("#### Student-attributed holdings")
        if virtual_holding_rows:
            st.dataframe(virtual_holding_rows, width="stretch")
        else:
            st.info("No shares are currently attributed to student buckets.")

        st.markdown("#### Alpaca order status")
        order_rows = [
            {
                "Paper account": order.get("_account_name", "primary"),
                "Order ID": order.get("id"),
                "Symbol": order.get("symbol"),
                "Side": order.get("side"),
                "Qty": order.get("qty"),
                "Filled qty": order.get("filled_qty"),
                "Status": order.get("status"),
                "Filled avg price": order.get("filled_avg_price"),
                "Submitted": order.get("submitted_at"),
                "Updated": order.get("updated_at"),
            }
            for order in orders[:100]
        ]
        if order_rows:
            st.dataframe(order_rows, width="stretch")
        else:
            st.info("No recent Alpaca orders were returned.")
    else:
        st.info("Click **Refresh holdings and trade status** to load live Alpaca data.")

st.caption(
    "All submitted orders are paper orders sent to Alpaca's paper endpoint. "
    "Review your selected symbols and quantities before approving."
)
