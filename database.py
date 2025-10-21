import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import config

class ExpenseDatabase:
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                payment_method TEXT NOT NULL,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_anomaly BOOLEAN DEFAULT 0,
                anomaly_score REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                monthly_income REAL,
                savings_goal REAL,
                risk_tolerance TEXT,
                notification_enabled BOOLEAN DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                forecast_date DATE NOT NULL,
                predicted_amount REAL NOT NULL,
                category TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def add_expense(self, user_id: int, amount: float, category: str,
                    payment_method: str, description: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expenses (user_id, amount, category, payment_method, description)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, amount, category, payment_method, description))
        expense_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return expense_id

    def get_user_expenses(self, user_id: int, days: int = 90) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM expenses
            WHERE user_id = ?
              AND timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn, params=(user_id, days))
        conn.close()
        return df

    def update_anomaly_status(self, expense_id: int, is_anomaly: bool, score: float):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE expenses
               SET is_anomaly = ?, anomaly_score = ?
             WHERE id = ?
        """, (is_anomaly, score, expense_id))
        conn.commit()
        conn.close()

    def save_forecast(self, user_id: int, forecasts: List[Dict]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for f in forecasts:
            cursor.execute("""
                INSERT INTO forecasts (user_id, forecast_date, predicted_amount, category)
                VALUES (?, ?, ?, ?)
            """, (user_id, f["date"], f["amount"], f.get("category", "total")))
        conn.commit()
        conn.close()

    def get_user_preferences(self, user_id: int) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            columns = ["user_id", "monthly_income", "savings_goal", "risk_tolerance", "notification_enabled"]
            return dict(zip(columns, row))
        return None

    def set_user_preferences(self, user_id: int, monthly_income: float = None,
                             savings_goal: float = None, risk_tolerance: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (user_id, monthly_income, savings_goal, risk_tolerance)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                monthly_income   = COALESCE(?, monthly_income),
                savings_goal     = COALESCE(?, savings_goal),
                risk_tolerance   = COALESCE(?, risk_tolerance)
        """, (user_id, monthly_income, savings_goal, risk_tolerance,
              monthly_income, savings_goal, risk_tolerance))
        conn.commit()
        conn.close()
