
# SpendWell Bot

A Python-based, Telegram-first personal finance assistant that logs expenses in natural language, detects spending anomalies, forecasts future outflows, and delivers AI-driven savings and investment recommendations tailored for the Indian context.

## Overview

SpendWell Bot enhances traditional expense tracking by replacing rigid forms with conversational flows, then layering machine learning for anomaly detection, forecasting, and structured AI financial advice. It's designed to grow with your data - start with basic tracking and unlock advanced features as your financial history develops.

## Key Features

- **Natural Language Expense Tracking**: Telegram chat interface for quick, natural-language expense entry with robust structured parsing
- **Real-time Anomaly Detection**: Isolation Forest algorithm with human-readable explanations based on your historical patterns
- **30-Day Forecasting**: Prophet/ARIMA forecasting with budget overrun warnings for proactive control
- **AI Financial Advisory**: Savings strategies and Indian-market investment recommendations via structured JSON outputs
- **Modular Architecture**: Start with tracking and add ML/advisory features as your dataset grows

## Tech Stack

- **Bot Framework**: `python-telegram-bot` v22 for async handlers and production-ready patterns
- **NLP + Advisory**: OpenAI GPT‑4o with Structured Outputs for schema-validated JSON responses
- **Anomaly Detection**: `scikit-learn` IsolationForest with temporal and categorical features
- **Forecasting**: Prophet for seasonal patterns and ARIMA for autoregressive structure
- **Data & Utilities**: SQLite, pandas, numpy, dotenv for local-first persistence

## Quick Start

### Prerequisites

- Python 3.10+
- Telegram account with bot token from [BotFather](https://t.me/botfather)
- OpenAI API key for parsing and advisory modules

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spendwell-bot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
TELEGRAM_BOT_TOKEN=your_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run the bot:
```bash
python telegram_bot.py
```

### First Use

1. Start chatting with your bot in Telegram
2. Use `/start` to see help message
3. Send "Spent ₹500 on lunch via UPI" to add your first expense

## Configuration

Edit `config.py` to customize:

- `ANOMALY_CONTAMINATION`: Adjust sensitivity for anomaly alerts (default: 0.1)
- `FORECAST_DAYS`: Planning window for forecasts (default: 30)
- Data thresholds and model parameters

## Available Commands

- `/start` - Introduction and usage tips
- `/stats` - 30-day totals, daily average, and top category breakdowns
- `/forecast` - Next 30-day forecast with budget exceed alerts
- `/advice` - Savings strategies with ₹ estimates and prioritized actions
- `/invest` - Indian investment options with risk levels and expected returns
- `/setincome <amount>` - Set monthly income for budget personalization

## Data Model

### Expenses Table
- `id`, `user_id`, `amount`, `category`, `payment_method`
- `description`, `timestamp`, `is_anomaly`, `anomaly_score`

### User Preferences
- `user_id`, `monthly_income`, `savings_goal`, `risk_tolerance`, `notification_enabled`

### Forecasts
- `id`, `user_id`, `forecast_date`, `predicted_amount`, `category`, `created_at`

## Anomaly Detection

**Model**: Isolation Forest algorithm that isolates outliers based on shorter path lengths in random partitioning

**Features**: 
- Amount, hour, day-of-week, day-of-month
- Category one-hot encoding
- Payment method one-hot encoding

**Output**: Real-time flag with explanation comparing to your category mean and variance

## Forecasting

**Prophet**: Handles daily/weekly seasonality, trend changepoints, and gaps in spending data

**ARIMA**: Complementary model for short-horizon autoregressive baselines

**Budget Alerts**: Compares projected end-of-month spend to income-derived envelopes

## AI Advisory

**Structured Outputs**: Pydantic schemas ensure consistent JSON responses with required fields

**Savings Recommendations**: 3-5 targeted suggestions with estimated monthly ₹ savings

**Indian Investments**: PPF, SIPs, FDs with rationale, risk levels, expected returns, and minimum amounts

## Security & Privacy

- Store secrets in `.env` (excluded from version control)
- Regular key rotation recommended
- Local SQLite storage by default
- Minimal personal data in AI prompts
- Follow least-privilege principles

## Deployment

### Development
```bash
python telegram_bot.py  # Uses polling
```

### Production
- Migrate to webhooks behind HTTPS for cloud deployment
- Containerize with slim Python image
- Supply environment variables at runtime

## Roadmap

- [ ] Scheduled daily/weekly reports and proactive notifications
- [ ] Receipt OCR ingestion and web dashboard
- [ ] Multi-currency support and bank integrations
- [ ] Enhanced visualization and goal tracking

## Development Time

Core implementation (bot, parsing, storage, baseline analytics): **4-6 hours**  
Full feature set with ML modules: **Additional 2-3 hours**

---

**Note**: Prophet may require system toolchains on some platforms. Check [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) if you encounter build issues.
```

This README provides a comprehensive overview while maintaining the conversational, practical tone from your original documentation. It's structured for easy navigation and includes all the essential information for users to get started quickly.
