import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import config
from database import ExpenseDatabase

class AnomalyDetector:
    def __init__(self, contamination: float = config.ANOMALY_CONTAMINATION):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.db = ExpenseDatabase()

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day

        cat_enc = pd.get_dummies(df["category"], prefix="cat")
        pay_enc = pd.get_dummies(df["payment_method"], prefix="pay")

        feats = pd.concat(
            [df[["amount", "hour", "day_of_week", "day_of_month"]], cat_enc, pay_enc],
            axis=1
        )
        feats = feats.fillna(0.0)
        return feats.values

    def detect_anomalies(self, user_id: int) -> pd.DataFrame:
        df = self.db.get_user_expenses(user_id, days=90)
        if len(df) < 10:
            return pd.DataFrame()

        X = self.prepare_features(df)
        Xs = self.scaler.fit_transform(X)
        pred = self.model.fit_predict(Xs)
        scores = self.model.score_samples(Xs)

        df["is_anomaly"] = pred == -1
        df["anomaly_score"] = -scores  # higher = more anomalous

        for _, row in df.iterrows():
            self.db.update_anomaly_status(
                int(row["id"]),
                bool(row["is_anomaly"]),
                float(row["anomaly_score"])
            )
        return df[df["is_anomaly"]]

    def check_new_expense(self, user_id: int, expense_id: int) -> Tuple[bool, float, str]:
        df = self.db.get_user_expenses(user_id, days=90)
        if len(df) < 10:
            return False, 0.0, "Not enough data for anomaly detection"

        new_df = df[df["id"] == expense_id]
        if new_df.empty:
            return False, 0.0, "Expense not found"

        hist = df[df["id"] != expense_id]
        X_hist = self.prepare_features(hist)
        X_new = self.prepare_features(new_df)

        Xs_hist = self.scaler.fit_transform(X_hist)
        Xs_new = self.scaler.transform(X_new)

        self.model.fit(Xs_hist)
        pred = self.model.predict(Xs_new)[0]
        score = -self.model.score_samples(Xs_new)[0]
        is_anom = pred == -1

        self.db.update_anomaly_status(expense_id, bool(is_anom), float(score))
        explanation = self._explain(new_df.iloc[0], hist)
        return bool(is_anom), float(score), explanation

    def _explain(self, expense: pd.Series, hist: pd.DataFrame) -> str:
        amt = float(expense["amount"])
        cat = str(expense["category"])
        cat_hist = hist[hist["category"] == cat]
        if len(cat_hist) >= 3:
            avg = cat_hist["amount"].mean()
            std = cat_hist["amount"].std(ddof=0)
            if amt > avg + 2 * std:
                return f"This {cat} expense (₹{amt:.2f}) is significantly higher than your average (₹{avg:.2f})"
            if amt < avg - 2 * std:
                return f"This {cat} expense (₹{amt:.2f}) is unusually low for this category"
        return "This expense pattern is unusual based on your history"
