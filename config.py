import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database
DB_PATH = "expenses.db"

# Anomaly Detection Settings
ANOMALY_CONTAMINATION = 0.1  # Expected proportion of anomalies (10%)
ANOMALY_THRESHOLD = 0.5      # Not used directly by IsolationForest; keep for messaging

# Predictive Analysis Settings
FORECAST_DAYS = 30
MIN_DATA_POINTS = 30

# Notification Settings (optional scheduling if you add cron/scheduler)
DAILY_SUMMARY_TIME = "20:00"
WEEKLY_REPORT_DAY = "Sunday"
