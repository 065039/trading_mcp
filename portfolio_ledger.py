from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

from trading_engine import STUDENTS, STUDENT_BUDGET


DB_PATH = Path("logs") / "portfolio_ledger.sqlite3"
logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS student_buckets (
            student TEXT PRIMARY KEY,
            initial_budget REAL NOT NULL,
            cash REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bucket_positions (
            student TEXT NOT NULL,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            average_price REAL NOT NULL,
            account_name TEXT NOT NULL DEFAULT 'primary',
            PRIMARY KEY (student, symbol),
            FOREIGN KEY (student) REFERENCES student_buckets(student)
        );
        CREATE TABLE IF NOT EXISTS bucket_orders (
            order_id TEXT PRIMARY KEY,
            student TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            requested_qty REAL NOT NULL,
            reference_price REAL NOT NULL,
            reserved_cash REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            account_name TEXT NOT NULL DEFAULT 'primary',
            FOREIGN KEY (student) REFERENCES student_buckets(student)
        );
        """
    )
    for table in ("bucket_positions", "bucket_orders"):
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if "account_name" not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN account_name TEXT NOT NULL DEFAULT 'primary'"
            )
    for student in STUDENTS:
        connection.execute(
            """
            INSERT OR IGNORE INTO student_buckets (student, initial_budget, cash)
            VALUES (?, ?, ?)
            """,
            (student, STUDENT_BUDGET, STUDENT_BUDGET),
        )
    connection.commit()
    return connection


def get_bucket_state() -> Dict[str, Any]:
    with closing(_connect()) as connection:
        buckets = {}
        for row in connection.execute(
            "SELECT student, initial_budget, cash FROM student_buckets ORDER BY student"
        ):
            holdings = [
                dict(position)
                for position in connection.execute(
                    """
                    SELECT symbol, qty, average_price, account_name
                    FROM bucket_positions
                    WHERE student = ? AND qty > 0
                    ORDER BY symbol
                    """,
                    (row["student"],),
                )
            ]
            pending_orders = [
                dict(order)
                for order in connection.execute(
                    """
                    SELECT order_id, symbol, side, requested_qty, reference_price, reserved_cash, status, account_name
                    FROM bucket_orders
                    WHERE student = ? AND status = 'pending'
                    ORDER BY symbol
                    """,
                    (row["student"],),
                )
            ]
            buckets[row["student"]] = {
                "initial_budget": float(row["initial_budget"]),
                "cash": float(row["cash"]),
                "holdings": holdings,
                "pending_orders": pending_orders,
            }
        return buckets


def get_bucket_position(student: str, symbol: str) -> Dict[str, Any] | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT qty, average_price, account_name
            FROM bucket_positions
            WHERE student = ? AND symbol = ?
            """,
            (student, symbol.upper()),
        ).fetchone()
    if row is None:
        return None
    return {
        "qty": float(row["qty"]),
        "average_price": float(row["average_price"]),
        "account_name": str(row["account_name"]),
    }


def get_pending_buy_reservations(account_name: str | None = None) -> float:
    with closing(_connect()) as connection:
        if account_name is None:
            row = connection.execute(
                "SELECT COALESCE(SUM(reserved_cash), 0) AS total FROM bucket_orders WHERE status = 'pending' AND side = 'buy'"
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(reserved_cash), 0) AS total
                FROM bucket_orders
                WHERE status = 'pending' AND side = 'buy' AND account_name = ?
                """,
                (account_name,),
            ).fetchone()
        return float(row["total"])


def get_pending_sell_qty(student: str, symbol: str) -> float:
    with closing(_connect()) as connection:
        row = connection.execute(
            """
            SELECT COALESCE(SUM(requested_qty), 0) AS total
            FROM bucket_orders
            WHERE student = ? AND symbol = ? AND side = 'sell' AND status = 'pending'
            """,
            (student, symbol.upper()),
        ).fetchone()
        return float(row["total"])


def record_pending_order(
    *,
    order_id: str,
    student: str,
    symbol: str,
    side: str,
    qty: float,
    reference_price: float,
    account_name: str = "primary",
) -> None:
    reserved_cash = qty * reference_price if side.lower() == "buy" else 0.0
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO bucket_orders (
                order_id, student, symbol, side, requested_qty,
                reference_price, reserved_cash, status, account_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                order_id,
                student,
                symbol.upper(),
                side.lower(),
                qty,
                reference_price,
                reserved_cash,
                account_name,
            ),
        )
        connection.commit()


def apply_order_update(order: Dict[str, Any]) -> None:
    order_id = str(order.get("id", ""))
    if not order_id:
        return
    status = str(order.get("status", "")).lower()
    filled_qty = float(order.get("filled_qty", 0.0) or 0.0)
    fill_price = float(order.get("filled_avg_price", 0.0) or 0.0)
    with closing(_connect()) as connection:
        record = connection.execute(
            "SELECT * FROM bucket_orders WHERE order_id = ? AND status = 'pending'",
            (order_id,),
        ).fetchone()
        if record is None:
            return
        terminal = status in {"filled", "canceled", "cancelled", "rejected", "expired"}
        if not terminal:
            return

        student = record["student"]
        symbol = record["symbol"]
        side = record["side"]
        quantity = min(float(record["requested_qty"]), max(0.0, filled_qty))

        if quantity > 0 and fill_price > 0 and side == "buy":
            bucket = connection.execute(
                "SELECT cash FROM student_buckets WHERE student = ?",
                (student,),
            ).fetchone()
            actual_cost = quantity * fill_price
            remaining_cash = float(bucket["cash"]) - actual_cost
            if remaining_cash < -1e-6:
                logger.warning(
                    "Alpaca fill %s cost $%.2f, exceeding %s's remaining virtual "
                    "bucket cash by $%.2f; recording the completed fill accurately.",
                    order_id,
                    actual_cost,
                    student,
                    abs(remaining_cash),
                )
            position = connection.execute(
                "SELECT qty, average_price FROM bucket_positions WHERE student = ? AND symbol = ?",
                (student, symbol),
            ).fetchone()
            old_qty = float(position["qty"]) if position else 0.0
            old_cost = old_qty * float(position["average_price"]) if position else 0.0
            new_qty = old_qty + quantity
            average_price = (old_cost + actual_cost) / new_qty
            connection.execute(
                """
                INSERT INTO bucket_positions (student, symbol, qty, average_price, account_name)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student, symbol) DO UPDATE SET
                    qty = excluded.qty,
                    average_price = excluded.average_price,
                    account_name = excluded.account_name
                """,
                (student, symbol, new_qty, average_price, record["account_name"]),
            )
            connection.execute(
                "UPDATE student_buckets SET cash = cash - ? WHERE student = ?",
                (actual_cost, student),
            )
        elif quantity > 0 and fill_price > 0 and side == "sell":
            position = connection.execute(
                "SELECT qty FROM bucket_positions WHERE student = ? AND symbol = ?",
                (student, symbol),
            ).fetchone()
            if position is None or float(position["qty"]) + 1e-6 < quantity:
                raise RuntimeError(
                    f"Cannot reconcile {order_id}: sold shares exceed {student}'s virtual holding."
                )
            connection.execute(
                "UPDATE bucket_positions SET qty = qty - ? WHERE student = ? AND symbol = ?",
                (quantity, student, symbol),
            )
            connection.execute(
                "UPDATE student_buckets SET cash = cash + ? WHERE student = ?",
                (quantity * fill_price, student),
            )

        connection.execute(
            "UPDATE bucket_orders SET status = ? WHERE order_id = ?",
            (status if terminal else "pending", order_id),
        )
        connection.commit()


def get_open_bucket_symbol_pairs() -> set[tuple[str, str]]:
    with closing(_connect()) as connection:
        rows = connection.execute(
            "SELECT student, symbol FROM bucket_orders WHERE status = 'pending'"
        ).fetchall()
        return {(str(row["student"]), str(row["symbol"]).upper()) for row in rows}


def virtual_position_value(bucket: Dict[str, Any], prices: Dict[str, float]) -> float:
    return sum(
        float(holding["qty"]) * float(prices.get(holding["symbol"], holding["average_price"]))
        for holding in bucket["holdings"]
    )


def calculate_bucket_performance(
    buckets: Dict[str, Any],
    prices: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    performance = {}
    for student, bucket in buckets.items():
        initial_budget = float(bucket["initial_budget"])
        cash = float(bucket["cash"])
        cost_basis = sum(
            float(holding["qty"]) * float(holding["average_price"])
            for holding in bucket["holdings"]
        )
        market_value = virtual_position_value(bucket, prices)
        total_value = cash + market_value
        performance[student] = {
            "cash": cash,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "total_value": total_value,
            "realized_profit": cash + cost_basis - initial_budget,
            "unrealized_profit": market_value - cost_basis,
            "total_profit": total_value - initial_budget,
        }
    return performance
