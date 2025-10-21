from typing import List, Dict
from pydantic import BaseModel, Field
from openai import OpenAI
import config
from database import ExpenseDatabase

class SavingsRecommendation(BaseModel):
    strategy: str
    description: str
    potential_savings: float
    priority: str
    action_steps: List[str]

class InvestmentRecommendation(BaseModel):
    investment_type: str
    rationale: str
    risk_level: str
    expected_return: str
    minimum_amount: float
    suitability: str

class FinancialAdvice(BaseModel):
    summary: str
    savings_recommendations: List[SavingsRecommendation]
    investment_recommendations: List[InvestmentRecommendation]
    warnings: List[str]
    monthly_action_plan: str

class AIFinancialAdvisor:
    def __init__(self):
        self.db = ExpenseDatabase()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def get_spending_summary(self, user_id: int) -> Dict:
        df = self.db.get_user_expenses(user_id, days=90)
        prefs = self.db.get_user_preferences(user_id) or {}
        total = float(df["amount"].sum()) if not df.empty else 0.0
        monthly_avg = total / 3 if total > 0 else 0.0
        by_cat = df.groupby("category")["amount"].sum().to_dict() if not df.empty else {}
        cat_pct = {k: (v / total * 100.0) for k, v in by_cat.items()} if total > 0 else {}
        by_pay = df.groupby("payment_method")["amount"].sum().to_dict() if not df.empty else {}
        anomalies = int((df["is_anomaly"] == True).sum()) if ("is_anomaly" in df) else 0
        return {
            "total_spending_90days": total,
            "monthly_average": monthly_avg,
            "category_spending": by_cat,
            "category_percentages": cat_pct,
            "payment_breakdown": by_pay,
            "anomalies_detected": anomalies,
            "user_preferences": prefs
        }

    def generate_savings_advice(self, user_id: int) -> Dict:
        s = self.get_spending_summary(user_id)
        ctx = "Spending by category:\n" + "\n".join(
            f"- {k}: {v:.1f}% (₹{s['category_spending'][k]:.2f})"
            for k, v in s["category_percentages"].items()
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial advisor for Indian users."
                    " Provide actionable savings recommendations with rupee amounts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Monthly average spend: ₹{s['monthly_average']:.2f}\n"
                    f"Income: ₹{s['user_preferences'].get('monthly_income', 'NA')}\n"
                    f"Savings goal: ₹{s['user_preferences'].get('savings_goal', 'NA')}\n"
                    f"Risk tolerance: {s['user_preferences'].get('risk_tolerance', 'NA')}\n\n"
                    f"{ctx}\n\n"
                    "Give 3-5 strategies with estimated monthly savings and action steps."
                )
            }
        ]
        resp = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=FinancialAdvice
        )
        return resp.choices[0].message.parsed.model_dump()

    def generate_investment_advice(self, user_id: int) -> Dict:
        s = self.get_spending_summary(user_id)
        income = float(s["user_preferences"].get("monthly_income", 0) or 0)
        available = max(0.0, income - s["monthly_average"])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Indian investment advisor."
                    " Recommend PPF, EPF, NPS, FDs, equity/mutual fund SIPs, etc., with risk and returns."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Available to invest monthly: ₹{available:.2f}\n"
                    f"Risk tolerance: {s['user_preferences'].get('risk_tolerance', 'moderate')}\n"
                    "Provide 3-5 options, each with expected return, risk, minimum amount, and suitability."
                )
            }
        ]
        resp = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=FinancialAdvice
        )
        return resp.choices[0].message.parsed.model_dump()

    def get_quick_tip(self, user_id: int, extra: str = "") -> str:
        s = self.get_spending_summary(user_id)
        top_cat = max(s["category_percentages"], key=s["category_percentages"].get) if s["category_percentages"] else "general"
        prompt = (
            f"Monthly avg: ₹{s['monthly_average']:.2f}\n"
            f"Top category: {top_cat}\n{extra}\n"
            "Give 1 practical money-saving tip in 2 sentences."
        )
        tip = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, concise financial coach."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return tip.choices[0].message.content
