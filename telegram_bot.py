import logging
from typing import Dict
from pydantic import BaseModel, Field
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

import config
from database import ExpenseDatabase
from anomaly_detector import AnomalyDetector
from predictive_analyzer import PredictiveAnalyzer
from ai_financial_advisor import AIFinancialAdvisor

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

db = ExpenseDatabase()
anom = AnomalyDetector()
pred = PredictiveAnalyzer()
advisor = AIFinancialAdvisor()
oai = OpenAI(api_key=config.OPENAI_API_KEY)

class ExpenseData(BaseModel):
    amount: float = Field(description="Amount in rupees")
    category: str = Field(description="Category like food, groceries, rent, etc.")
    payment_method: str = Field(description="cash, UPI, credit card, debit card")
    description: str = Field(description="Short description")

def parse_expense(msg: str) -> Dict:
    resp = oai.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract expense fields from natural language for Indian users; "
                    "recognize UPI/Paytm/PhonePe/credit/debit/cash and common categories."
                )
            },
            {"role": "user", "content": f"Parse this expense: {msg}"}
        ],
        response_format=ExpenseData
    )
    return resp.choices[0].message.parsed.model_dump()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Welcome to SpendWell Smart Expense Tracker!\n\n"
        "Send expenses naturally (e.g., 'Spent ₹500 on dinner via UPI').\n\n"
        "Commands:\n"
        "/stats, /forecast, /advice, /invest, /setincome <amount>\n"
    )
    await update.message.reply_text(txt)

async def add_expense(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text
    await update.message.reply_text("Processing your expense...")
    try:
        data = parse_expense(text)
    except Exception as e:
        logger.error(f"Parse error: {e}")
        await update.message.reply_text(
            "Sorry, couldn't understand that, try: 'Spent ₹500 on groceries via UPI'"
        )
        return

    exp_id = db.add_expense(
        user_id=user_id,
        amount=float(data["amount"]),
        category=data["category"],
        payment_method=data["payment_method"],
        description=data["description"]
    )

    is_anom, score, explanation = anom.check_new_expense(user_id, exp_id)
    msg = (
        f"Recorded:\n"
        f"Amount: ₹{float(data['amount']):.2f}\n"
        f"Category: {data['category']}\n"
        f"Payment: {data['payment_method']}\n"
        f"Note: {data['description']}"
    )
    if is_anom:
        msg += f"\n\nANOMALY DETECTED (Score: {score:.2f})\n{explanation}"
        tip = advisor.get_quick_tip(user_id, f"Unusual expense: {explanation}")
        msg += f"\n\nTip: {tip}"
    await update.message.reply_text(msg)

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    df = db.get_user_expenses(user_id, days=30)
    if df.empty:
        await update.message.reply_text("No expenses yet, try sending one now!")
        return
    total = float(df["amount"].sum())
    avg = total / 30.0
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    lines = [f"Total: ₹{total:.2f}", f"Daily Avg: ₹{avg:.2f}", "By Category:"]
    for c, a in by_cat.head(5).items():
        pct = a / total * 100 if total > 0 else 0
        lines.append(f"- {c}: ₹{a:.2f} ({pct:.1f}%)")
    await update.message.reply_text("\n".join(lines))

async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("Computing forecast...")
    fc = pred.forecast_expenses_prophet(user_id)
    if not fc["success"]:
        await update.message.reply_text(fc.get("message", "Not enough data"))
        return
    total = fc["total_predicted"]
    wk = sum(x["amount"] for x in fc["forecasts"][:7])
    msg = (
        f"Expense Forecast (30 days)\n"
        f"Predicted Total: ₹{total:.2f}\n"
        f"Next Week: ₹{wk:.2f}\n"
        f"Daily Avg: ₹{total/30:.2f}"
    )
    budget = pred.detect_budget_overrun(user_id)
    if budget and budget["will_exceed"]:
        msg += (
            f"\n\nBudget Alert: projected exceed by ₹{budget['excess_amount']:.2f} "
            f"with {budget['days_remaining']} days remaining"
        )
    await update.message.reply_text(msg)

async def advice_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("Generating savings recommendations...")
    try:
        adv = advisor.generate_savings_advice(user_id)
        header = adv.get("summary", "Savings Opportunities")
        body = []
        for i, r in enumerate(adv.get("savings_recommendations", [])[:3], 1):
            body.append(
                f"{i}. {r['strategy']} (₹{r['potential_savings']:.0f}/mo)\n- {r['description']}"
            )
        await update.message.reply_text(header + "\n\n" + "\n\n".join(body))
    except Exception as e:
        logger.error(f"Advice error: {e}")
        await update.message.reply_text("Error generating advice, try later.")

async def invest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text("Preparing investment recommendations...")
    try:
        adv = advisor.generate_investment_advice(user_id)
        header = adv.get("summary", "Investments")
        body = []
        for i, r in enumerate(adv.get("investment_recommendations", [])[:3], 1):
            body.append(
                f"{i}. {r['investment_type']} ({r['risk_level']} risk)\n"
                f"   Return: {r['expected_return']}, Min: ₹{r['minimum_amount']:.0f}\n"
                f"   {r['suitability']}"
            )
        await update.message.reply_text(header + "\n\n" + "\n\n".join(body))
    except Exception as e:
        logger.error(f"Invest error: {e}")
        await update.message.reply_text("Error generating advice, try later.")

async def setincome_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /setincome <amount>")
        return
    try:
        income = float(context.args[0])
        db.set_user_preferences(user_id, monthly_income=income)
        await update.message.reply_text(f"Monthly income set to ₹{income:.2f}")
    except ValueError:
        await update.message.reply_text("Please provide a valid number")

def main():
    app = ApplicationBuilder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("forecast", forecast_cmd))
    app.add_handler(CommandHandler("advice", advice_cmd))
    app.add_handler(CommandHandler("invest", invest_cmd))
    app.add_handler(CommandHandler("setincome", setincome_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, add_expense))
    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
