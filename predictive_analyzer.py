import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from prophet import Prophet
import config
from database import ExpenseDatabase

class PredictiveAnalyzer:
    def __init__(self):
        self.db = ExpenseDatabase()
        self.forecast_days = config.FORECAST_DAYS
        self.min_data_points = config.MIN_DATA_POINTS

    def _daily_series(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d["date"] = d["timestamp"].dt.date
        daily = d.groupby("date")["amount"].sum().reset_index()
        daily.columns = ["ds", "y"]
        return daily

    def forecast_expenses_prophet(self, user_id: int) -> Dict:
        df = self.db.get_user_expenses(user_id, days=90)
        if len(df) < self.min_data_points:
            return {"success": False, "message": "Not enough data for forecasting", "forecasts": []}

        ts = self._daily_series(df)
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(ts)
        future = model.make_future_dataframe(periods=self.forecast_days)
        fcst = model.predict(future)
        future_fcst = fcst[fcst["ds"] > ts["ds"].max()]

        out = []
        for _, r in future_fcst.iterrows():
            out.append({
                "date": r["ds"].strftime("%Y-%m-%d"),
                "amount": max(0.0, float(r["yhat"])),
                "lower_bound": max(0.0, float(r["yhat_lower"])),
                "upper_bound": max(0.0, float(r["yhat_upper"]))
            })

        self.db.save_forecast(user_id, out)
        return {
            "success": True,
            "forecasts": out,
            "total_predicted": sum(f["amount"] for f in out),
            "model": "Prophet"
        }

    def forecast_by_category(self, user_id: int) -> Dict:
        df = self.db.get_user_expenses(user_id, days=90)
        if len(df) < self.min_data_points:
            return {"success": False, "message": "Not enough data"}

        cats = df["category"].unique().tolist()
        summary = {}
        for c in cats:
            cdf = df[df["category"] == c]
            if len(cdf) < 10:
                continue
            ts = self._daily_series(cdf)
            avg = float(ts["y"].mean())
            std = float(ts["y"].std(ddof=0))
            summary[c] = {
                "predicted_total": avg * self.forecast_days,
                "daily_average": avg,
                "volatility": std
            }

        return {"success": True, "category_forecasts": summary,
                "total_predicted": sum(v["predicted_total"] for v in summary.values())}

    def detect_budget_overrun(self, user_id: int) -> Optional[Dict]:
        prefs = self.db.get_user_preferences(user_id)
        if not prefs or not prefs.get("monthly_income"):
            return None

        # Current month spend to date
        df = self.db.get_user_expenses(user_id, days=30)
        current_total = float(df["amount"].sum()) if not df.empty else 0.0

        # Project end of month using simple average
        today = datetime.now()
        days_passed = max(1, today.day)
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        remaining = (last_day - today).days
        daily_avg = current_total / days_passed
        projected = current_total + daily_avg * remaining

        budget_limit = float(prefs["monthly_income"]) * 0.5  # simple 50% cap
        return {
            "current_spending": current_total,
            "projected_total": projected,
            "budget_limit": budget_limit,
            "will_exceed": projected > budget_limit,
            "excess_amount": max(0.0, projected - budget_limit),
            "days_remaining": remaining
        }
