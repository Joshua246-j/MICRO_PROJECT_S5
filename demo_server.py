"""
StockAI Terminal — Complete Demo Edition Server
Serves the full-featured Stock-rag research terminal UI with deterministic simulated market data,
forecasting projections, financial news, and interactive simulated AI chat.
Zero Ollama, PyTorch, FAISS, or heavy ML dependencies. Portable for Docker, Minikube, and Render.
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("stockai-demo")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FIXTURES_FILE = BASE_DIR / "demo_data" / "stocks.json"

DEMO_DISCLAIMER = "DEMO MODE — SIMULATED MARKET DATA — NOT LIVE PRICES"

# Load deterministic demonstration fixtures
try:
    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        STOCKS_DATA = json.load(f)
    logger.info("Loaded %d demo stock fixtures from %s", len(STOCKS_DATA), FIXTURES_FILE)
except Exception as e:
    logger.error("Failed to load demo fixtures: %s", e)
    STOCKS_DATA = {}

app = FastAPI(
    title="StockAI Terminal — Demo Edition",
    description="Lightweight simulated financial research terminal for DevOps micro-project demonstration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory settings state
demo_settings: Dict[str, Any] = {
    "theme": "dark",
    "ai_model": "cloud-market-preview",
    "temperature": 0.2,
    "context_size": 4096,
    "citation_visibility": True,
    "max_output_tokens": 256,
    "top_p": 0.9,
    "top_k": 40,
    "ui_density": "default",
    "request_timeout": 30,
    "default_chat_mode": "general",
    "disclaimer": DEMO_DISCLAIMER,
}

# In-memory simulated news (Timely, institutional-grade headlines for US Tech & NSE Bluechips)
now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
simulated_news = [
    {
        "title": "NVIDIA AI Accelerator Order Book Surges to Multi-Year Record Amid Datacenter Expansion (Simulated)",
        "publisher": "Bloomberg (Demo)",
        "ticker": "NVDA",
        "market": "us",
        "category": "AI & Semiconductors",
        "link": "https://example.com/demo-news-nvda",
        "url": "https://example.com/demo-news-nvda",
        "published": "2026-10-27T05:30:00Z",
        "published_at": "2026-10-27T05:30:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Hyperscale cloud providers expand multi-gigawatt datacenter infrastructure commitments, boosting forward Blackwell architecture orders and margins.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Microsoft Azure Cloud Surpasses Consensus Projections on Copilot Enterprise Adoption (Simulated)",
        "publisher": "Financial Times (Demo)",
        "ticker": "MSFT",
        "market": "us",
        "category": "Enterprise Cloud",
        "link": "https://example.com/demo-news-msft",
        "url": "https://example.com/demo-news-msft",
        "published": "2026-10-27T05:00:00Z",
        "published_at": "2026-10-27T05:00:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Enterprise AI seat activations exceed 45 million across Global 2000 organizations, accelerating commercial cloud gross margin expansion.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Reliance Industries Outlines Multi-Year Retail & Jio 5G Ecosystem Monetization (Simulated)",
        "publisher": "Economic Times (Demo)",
        "ticker": "RELIANCE.NS",
        "market": "india",
        "category": "Energy & Retail",
        "link": "https://example.com/demo-news-reliance",
        "url": "https://example.com/demo-news-reliance",
        "published": "2026-10-27T04:30:00Z",
        "published_at": "2026-10-27T04:30:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Strategic expansion in omnichannel retail footprint and 5G subscriber ARPU gains support strong consolidated operating cash flows.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Apple Expands Services Margin and Edge Silicon Roadmap Ahead of Q4 Release (Simulated)",
        "publisher": "Wall Street Journal (Demo)",
        "ticker": "AAPL",
        "market": "us",
        "category": "Consumer Tech",
        "link": "https://example.com/demo-news-aapl",
        "url": "https://example.com/demo-news-aapl",
        "published": "2026-10-27T04:00:00Z",
        "published_at": "2026-10-27T04:00:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "App Store, iCloud storage, and device subscription services revenue reaches new all-time quarterly high with stable hardware replacement cycles.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Alphabet Demonstrates TPU v6 Efficiency and Gemini Ultra Enterprise Scaling (Simulated)",
        "publisher": "Reuters (Demo)",
        "ticker": "GOOGL",
        "market": "us",
        "category": "AI & Cloud",
        "link": "https://example.com/demo-news-googl",
        "url": "https://example.com/demo-news-googl",
        "published": "2026-10-27T03:30:00Z",
        "published_at": "2026-10-27T03:30:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Next-generation custom silicon yields 2.8x higher compute density per watt, expanding Google Cloud enterprise AI capture and search monetization.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "HDFC Bank Records Strong Deposit Inflows and Asset Quality Resilience in Q3 (Simulated)",
        "publisher": "Livemint (Demo)",
        "ticker": "HDFCBANK.NS",
        "market": "india",
        "category": "Banking & Credit",
        "link": "https://example.com/demo-news-hdfc",
        "url": "https://example.com/demo-news-hdfc",
        "published": "2026-10-27T03:00:00Z",
        "published_at": "2026-10-27T03:00:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Credit disbursement across retail and commercial loans exhibits steady double-digit growth with net non-performing assets at historic lows.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "TCS Wins Landmark Multi-Billion Dollar European Digital Transformation Contract (Simulated)",
        "publisher": "Business Standard (Demo)",
        "ticker": "TCS.NS",
        "market": "india",
        "category": "IT Services",
        "link": "https://example.com/demo-news-tcs",
        "url": "https://example.com/demo-news-tcs",
        "published": "2026-10-27T02:15:00Z",
        "published_at": "2026-10-27T02:15:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Tata Consultancy Services secures a 7-year strategic modernization engagement with a premier European financial and cloud institution.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Infosys Expands Generative AI Delivery Framework Topaz Across Global Clients (Simulated)",
        "publisher": "CNBC (Demo)",
        "ticker": "INFY.NS",
        "market": "india",
        "category": "IT Services",
        "link": "https://example.com/demo-news-infy",
        "url": "https://example.com/demo-news-infy",
        "published": "2026-10-27T01:45:00Z",
        "published_at": "2026-10-27T01:45:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Active pipeline of enterprise GenAI engagements surpasses 250 deployments, bolstering large deal Total Contract Value (TCV).",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Federal Reserve & Reserve Bank Maintain Steady Monetary Liquidity Stance (Simulated)",
        "publisher": "Bloomberg (Demo)",
        "ticker": "MACRO",
        "market": "us",
        "category": "Macroeconomics",
        "link": "https://example.com/demo-news-fed",
        "url": "https://example.com/demo-news-fed",
        "published": "2026-10-27T01:00:00Z",
        "published_at": "2026-10-27T01:00:00Z",
        "retrieved_at": "2026-10-27T05:45:00Z",
        "summary": "Central bank policymakers signal balanced inflation trajectories and steady liquidity facilities across global sovereign bond markets.",
        "disclaimer": DEMO_DISCLAIMER,
    },
]

# In-memory simulated chat sessions
demo_sessions: Dict[str, Dict[str, Any]] = {
    "demo-session-1": {
        "session_id": "demo-session-1",
        "title": "NVDA vs MSFT Enterprise AI Moats (Demo)",
        "created_at": "2026-10-27T05:15:00Z",
        "is_pinned": True,
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Compare NVIDIA vs Microsoft enterprise AI market share and valuation multiples.",
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": (
                    "**[StockAI Terminal — Neural Research Synthesis]**\n\n"
                    "Comparative breakdown for **NVIDIA (NVDA)** vs **Microsoft (MSFT)**:\n\n"
                    "1. **Valuation Multiples**:\n"
                    "   - **NVDA**: Price $141.50 | Forward P/E 34.2x | Market Cap $3.48T | 24h Vol 57.5M\n"
                    "   - **MSFT**: Price $432.80 | Forward P/E 32.6x | Market Cap $3.22T | 24h Vol 22.8M\n\n"
                    "2. **Competitive Moats**:\n"
                    "   - **Hardware/CUDA Layer**: NVIDIA maintains ~88% of datacenter accelerator capacity with CUDA developer lock-in.\n"
                    "   - **Application/Platform Layer**: Microsoft Azure and M365 Copilot dominate enterprise workflows with over 45M paid seats.\n\n"
                    "3. **Forecast Horizon (5D/20D)**:\n"
                    "   - NVDA models project +4.8% walk-forward return edge with 88% conformal directional confidence.\n"
                    "   - MSFT models project +2.4% steady return edge with 82% confidence.\n\n"
                    "> *Notice: Simulated demonstration response generated in DevOps demo mode without heavy PyTorch or Ollama dependencies.*"
                ),
            },
        ],
    },
    "demo-session-2": {
        "session_id": "demo-session-2",
        "title": "Reliance Retail & Jio 5G Valuation (Demo)",
        "created_at": "2026-10-27T04:45:00Z",
        "is_pinned": True,
        "messages": [
            {
                "id": "msg-3",
                "role": "user",
                "content": "Analyze Reliance Industries retail and telecoms growth drivers in India.",
            },
            {
                "id": "msg-4",
                "role": "assistant",
                "content": (
                    "**[StockAI Terminal — Neural Research Synthesis]**\n\n"
                    "Executive analysis for **Reliance Industries Ltd. (RELIANCE.NS)**:\n\n"
                    "- **Current Demo Price**: ₹2,985.40 (+0.84%)\n"
                    "- **Valuation**: Market Cap ₹20.2T | P/E 26.4x | 52W Range: ₹2,420 – ₹3,217\n"
                    "- **Key Catalysts**:\n"
                    "  - *Jio Infocomm*: 5G monetization ramp with 490M+ total subscribers and rising ARPU.\n"
                    "  - *Reliance Retail*: Pan-India grocery, consumer electronics, and e-commerce logistics scale.\n"
                    "  - *Clean Energy Giga-Complex*: Solar and green hydrogen capex transitioning traditional refining margins.\n\n"
                    "> *Notice: Deterministic prototype intelligence. Not live financial advice.*"
                ),
            },
        ],
    },
    "demo-session-3": {
        "session_id": "demo-session-3",
        "title": "ML Quantitative Uncertainty Audit (Demo)",
        "created_at": "2026-10-27T03:30:00Z",
        "is_pinned": False,
        "messages": [
            {
                "id": "msg-5",
                "role": "user",
                "content": "Explain how the point-in-time quantitative ML forecasts calculate uncertainty ranges.",
            },
            {
                "id": "msg-6",
                "role": "assistant",
                "content": (
                    "**[StockAI Terminal — Quantitative Model Audit]**\n\n"
                    "The terminal employs a zero-leakage, walk-forward conformal prediction methodology:\n\n"
                    "- **Residual-Based Bounds**: Empirical forecast error residuals from past non-overlapping validation folds establish the 80% coverage envelope.\n"
                    "- **Lookahead Prevention**: Point-in-time feature engineering ensures training sets only access data timestamped prior to forecast cutoffs.\n"
                    "- **Latency**: Evaluated in pure CPU NumPy routines with average execution time under 12ms."
                ),
            },
        ],
    },
}


# ---------------------------------------------------------
# Health and Diagnostics Endpoints
# ---------------------------------------------------------

@app.get("/api/health/live", tags=["Diagnostics"])
def liveness_probe():
    return {
        "application": "running",
        "status": "live",
        "mode": "demo",
        "version": "1.0.0",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/health/ready", tags=["Diagnostics"])
def readiness_probe():
    ready = bool(STOCKS_DATA)
    return {
        "application": "running" if ready else "degraded",
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "mode": "demo",
        "fixtures_count": len(STOCKS_DATA),
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/health", tags=["Diagnostics"])
def health_overview(include_external: bool = False):
    ready = bool(STOCKS_DATA)
    return {
        "application": "running" if ready else "degraded",
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "mode": "market-only",
        "cloud_preview": True,
        "database": "available",
        "ollama": "excluded (lightweight demo)",
        "model": "cloud-market-preview",
        "context_size": 4096,
        "database_stats": {"documents": 0, "chunks": 0},
        "model_available": True,
        "embedding_model": "not_applicable",
        "embedding_model_available": True,
        "document_retrieval": "market_only",
        "semantic_document_retrieval": "not_applicable",
        "keyword_document_retrieval": "not_applicable",
        "vector_database": "not_applicable",
        "vector_backend": "none",
        "web_search": {"status": "available", "service": "Simulated Demo News Provider"},
        "market_data": {
            "status": "available",
            "provider": "Deterministic Local Fixtures (Simulated)",
            "data_status": "SIMULATED_DEMO",
            "symbols": list(STOCKS_DATA.keys()),
        },
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Settings Endpoints
# ---------------------------------------------------------

@app.get("/api/settings", tags=["Settings"])
def get_settings():
    return demo_settings


@app.put("/api/settings", tags=["Settings"])
async def update_settings(req: Request):
    try:
        body = await req.json()
        demo_settings.update(body)
    except Exception:
        pass
    return demo_settings


# ---------------------------------------------------------
# Market & Stock Endpoints
# ---------------------------------------------------------

@app.get("/api/symbols/resolve/{symbol}", tags=["Market Data"])
def resolve_symbol(symbol: str):
    norm_symbol = symbol.strip().upper()
    resolved = norm_symbol if norm_symbol in STOCKS_DATA else "AAPL"
    return {
        "valid": True,
        "symbol": symbol,
        "resolved_symbol": resolved,
        "market": "india" if "NS" in resolved else "us",
        "exchange": "NSE" if "NS" in resolved else "NASDAQ",
        "name": STOCKS_DATA.get(resolved, {}).get("profile", {}).get("name", resolved),
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/stocks/{symbol}/quote", tags=["Market Data"])
def get_quote(symbol: str, refresh: bool = Query(False)):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    return stock_entry["quote"]


@app.get("/api/stocks/{symbol}/history", tags=["Market Data"])
def get_history(
    symbol: str,
    period: str = Query("1mo", pattern="^(1mo|3mo|6mo|1y|all)$"),
    interval: str = Query("1d", pattern="^(1d|1wk|1mo)$"),
    refresh: bool = Query(False),
    auto_adjust: bool = Query(True),
):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    all_history = stock_entry["history"]

    if period == "1mo":
        data = all_history[-22:] if len(all_history) >= 22 else all_history
    elif period == "3mo":
        data = all_history[-65:] if len(all_history) >= 65 else all_history
    else:
        data = all_history

    return {
        "success": True,
        "symbol": norm_symbol,
        "period": period,
        "interval": interval,
        "total_bars": len(data),
        "data": data,
        "source": stock_entry["quote"].get("source", "Simulated Demonstration Fixture"),
        "data_status": "SIMULATED_DEMO",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/stocks/{symbol}", tags=["Market Data"])
def get_profile(symbol: str):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    return stock_entry["profile"]


@app.get("/api/stocks/{symbol}/fundamentals", tags=["Market Data"])
@app.get("/api/stocks/{symbol}/financials", tags=["Market Data"])
def get_fundamentals(symbol: str, refresh: bool = Query(False)):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    quote = stock_entry["quote"]
    return {
        "success": True,
        "symbol": norm_symbol,
        "source": "Deterministic Demo Fixtures",
        "data_status": "SIMULATED_DEMO",
        "retrieved_at": "2026-10-26T15:30:00Z",
        "disclaimer": DEMO_DISCLAIMER,
        "metrics": {
            "Market Cap": quote.get("market_cap", "3.45T"),
            "P/E Ratio (TTM)": str(quote.get("pe_ratio", 33.8)),
            "52-Week High": str(quote.get("week_52_high", 237.23)),
            "52-Week Low": str(quote.get("week_52_low", 164.08)),
            "Previous Close": str(quote.get("previous_close", 221.30)),
            "Day Range": f"{quote.get('day_low', 221.50)} - {quote.get('day_high', 225.80)}",
            "Volume (24H)": f"{quote.get('volume', 48250000):,}",
            "Currency": quote.get("currency", "USD"),
        },
    }


@app.get("/api/stocks/{symbol}/technical", tags=["Market Data"])
def get_technical(
    symbol: str,
    period: str = Query("6mo"),
    interval: str = Query("1d"),
    refresh: bool = Query(False),
    auto_adjust: bool = Query(True),
):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    price = stock_entry["quote"]["price"]
    return {
        "success": True,
        "symbol": norm_symbol,
        "period": period,
        "interval": interval,
        "source": "Deterministic Technical Analytics",
        "data_status": "SIMULATED_DEMO",
        "last_bar_at": "2026-10-23",
        "retrieved_at": "2026-10-26T15:30:00Z",
        "disclaimer": DEMO_DISCLAIMER,
        "metrics": {
            "RSI (14)": 56.8,
            "MACD (12,26,9)": 1.45,
            "Signal Line": 1.18,
            "SMA 20": round(price * 0.985, 2),
            "SMA 50": round(price * 0.965, 2),
            "SMA 200": round(price * 0.920, 2),
            "Technical Sentiment": "NEUTRAL_BULLISH",
        },
        "scenario_estimate": {
            "horizon_bars": 30,
            "horizon_trading_days": 30,
            "bar_interval": "1d",
            "central_estimate": round(price * 1.025, 2),
            "lower_80_percent": round(price * 0.96, 2),
            "upper_80_percent": round(price * 1.09, 2),
            "persistence_baseline_holdout_mae": 1.85,
            "limitations": "Simulated demonstration statistics. Not live predictive investment advice.",
        },
    }


@app.get("/api/stocks/{symbol}/news", tags=["Market Data"])
def get_stock_news(symbol: str, refresh: bool = Query(False)):
    norm_symbol = symbol.strip().upper()
    return {
        "symbol": norm_symbol,
        "success": True,
        "news": simulated_news,
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/news/feed", tags=["News"])
def get_news_feed(limit: int = 20, refresh: bool = False):
    return {
        "success": True,
        "news": simulated_news,
        "retrieved_at": "2026-10-26T15:00:00Z",
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Forecasting Projections (Full UI Alignment)
# ---------------------------------------------------------

@app.get("/api/forecast/featured/summary", tags=["Forecasting"])
def featured_forecast_summary(horizon: str = Query("5")):
    items = []
    try:
        digits = "".join(c for c in str(horizon) if c.isdigit())
        h_val = int(digits) if digits else 5
    except Exception:
        h_val = 5

    # Ticker-specific baseline expected return for 5D
    base_momentum = {
        "NVDA": {"ret_5d": 4.82, "prob": 0.88, "bias": "bullish"},
        "RELIANCE.NS": {"ret_5d": 3.15, "prob": 0.85, "bias": "bullish"},
        "MSFT": {"ret_5d": 2.44, "prob": 0.82, "bias": "bullish"},
        "HDFCBANK.NS": {"ret_5d": 2.12, "prob": 0.81, "bias": "bullish"},
        "GOOGL": {"ret_5d": 1.95, "prob": 0.78, "bias": "bullish"},
        "AAPL": {"ret_5d": 1.80, "prob": 0.79, "bias": "bullish"},
        "INFY.NS": {"ret_5d": 1.45, "prob": 0.76, "bias": "bullish"},
        "TCS.NS": {"ret_5d": 1.20, "prob": 0.74, "bias": "bullish"},
    }

    scale = 0.25 if h_val <= 1 else (1.0 if h_val <= 5 else (2.4 if h_val <= 20 else 3.2))

    for sym in STOCKS_DATA.keys():
        entry = STOCKS_DATA.get(sym)
        if entry:
            q = entry["quote"]
            p = float(q["price"])
            curr = q.get("currency", "USD")
            meta = base_momentum.get(sym, {"ret_5d": 2.0, "prob": 0.80, "bias": "bullish"})
            
            exp_ret = round(meta["ret_5d"] * scale, 2)
            target_p = round(p * (1.0 + exp_ret / 100.0), 2)
            
            # 5-point sparkline points from near current price to target price
            step = (target_p - p) / 4.0
            spark = [
                round(p * 0.998, 2),
                round(p + step * 0.9, 2),
                round(p + step * 2.1, 2),
                round(p + step * 3.2, 2),
                target_p
            ]

            items.append({
                "symbol": sym,
                "name": q.get("company_name", sym),
                "current_price": p,
                "forecast_mean": target_p,
                "expected_price": target_p,
                "expected_return_pct": exp_ret,
                "expected_return_percent": exp_ret,
                "directional_probability": meta["prob"],
                "direction": meta["bias"],
                "confidence": "HIGH (SIMULATED)" if meta["prob"] >= 0.80 else "MODERATE (SIMULATED)",
                "sparkline_points": spark,
                "last_bar_at": "2026-10-26",
                "currency": curr,
                "model": "Simulated Walk-Forward Ensemble",
                "horizon": str(h_val),
                "disclaimer": DEMO_DISCLAIMER,
            })
    return {
        "success": True,
        "status": "SIMULATED_DEMO",
        "items": items,
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/forecast/{symbol}", tags=["Forecasting"])
def get_forecast(
    symbol: str,
    horizon: str = Query("30d"),
    model: str = Query("auto"),
    confidence: str = Query("0.80"),
    refresh: bool = Query(False),
):
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol) or STOCKS_DATA.get("AAPL")
    price = stock_entry["quote"]["price"]
    currency = stock_entry["quote"].get("currency", "USD")

    # Generate realistic projected trajectory path (15 sessions into future)
    start_date = datetime.date(2026, 10, 26)
    curr = start_date
    projection_path = []
    for i in range(15):
        while curr.weekday() >= 5:
            curr += datetime.timedelta(days=1)
        p_step = round(price * (1 + 0.0022 * (i + 1)), 2)
        projection_path.append({
            "date": curr.isoformat(),
            "price": p_step,
            "upper": round(p_step * 1.035, 2),
            "lower": round(p_step * 0.965, 2),
        })
        curr += datetime.timedelta(days=1)

    target_price = projection_path[-1]["price"]
    lower_bound = projection_path[-1]["lower"]
    upper_bound = projection_path[-1]["upper"]
    expected_ret = round(((target_price - price) / price) * 100, 2)

    return {
        "success": True,
        "symbol": norm_symbol,
        "currency": currency,
        "current_price": price,
        "last_bar_at": "2026-10-23",
        "horizon": 30,
        "selected_model": "historical_drift",
        "model_name": "Historical Drift (Simulated)",
        "directional_label": "Bullish Edge",
        "directional_probability": 0.84,
        "interval": {
            "expected_price": target_price,
            "lower_bound_price": lower_bound,
            "upper_bound_price": upper_bound,
            "expected_return_percent": expected_ret,
            "confidence_percent": 80,
        },
        "chart_data": {
            "target_date": projection_path[-1]["date"],
            "projection_path": projection_path,
        },
        "validation": {
            "mae_percent": 1.85,
            "beats_naive_baseline": True,
            "improvement_vs_naive_percent": 14.2,
        },
        "system_health": {"latency_ms": 1.2},
        "hardware": {
            "status": "ready",
            "cpu_cores": 4,
            "cpu_percent": 12,
            "ram_available_gb": 4.8,
            "gpu_available": False,
        },
        "freshness": "FRESH",
        "disagreement": False,
        "model_metadata": {
            "historical_drift": {
                "short_name": "Historical Drift",
                "name": "Empirical Baseline Drift Model (Simulated)",
                "description": "Deterministic baseline drift model for lightweight demonstration.",
            }
        },
        "third_party_forecasts": {
            "analyst_consensus": {
                "available": True,
                "reliability_badge": "External Consensus",
                "implied_return_percent": 8.4,
                "target_price": round(price * 1.084, 2),
                "rating": "Moderate Buy",
                "target_high": round(price * 1.20, 2),
                "target_low": round(price * 0.95, 2),
            }
        },
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/forecast/system/health", tags=["Forecasting"])
def forecast_system_health():
    return {
        "status": "healthy",
        "mode": "demo",
        "models_loaded": 1,
        "active_runtime": "FastAPI Lightweight Demo",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.post("/api/forecast/system/revalidate", tags=["Forecasting"])
def forecast_system_revalidate():
    return {"status": "revalidated", "mode": "demo"}


# ---------------------------------------------------------
# Interactive Simulated AI Chat Endpoints
# ---------------------------------------------------------

@app.get("/api/chat/sessions", tags=["Chat"])
def get_chat_sessions(limit: int = 50):
    sessions_list = list(demo_sessions.values())
    return {"sessions": sessions_list}


@app.get("/api/chat/sessions/{session_id}", tags=["Chat"])
def get_chat_session(session_id: str):
    sess = demo_sessions.get(session_id)
    if not sess:
        return {
            "session_id": session_id,
            "title": "New Conversation (Demo)",
            "messages": [],
            "document_ids": [],
            "dataset_id": None,
        }
    return sess


@app.put("/api/chat/sessions/{session_id}/pin", tags=["Chat"])
async def pin_chat_session(session_id: str, req: Request):
    body = await req.json()
    if session_id in demo_sessions:
        demo_sessions[session_id]["is_pinned"] = body.get("is_pinned", True)
    return {"status": "ok"}


@app.put("/api/chat/sessions/{session_id}/rename", tags=["Chat"])
async def rename_chat_session(session_id: str, req: Request):
    body = await req.json()
    if session_id in demo_sessions:
        demo_sessions[session_id]["title"] = body.get("title", "Renamed Chat")
    return {"status": "ok"}


@app.delete("/api/chat/sessions/{session_id}", tags=["Chat"])
def delete_chat_session(session_id: str):
    demo_sessions.pop(session_id, None)
    return {"status": "deleted"}


@app.post("/api/chat/sessions/{session_id}/branch", tags=["Chat"])
def branch_chat_session(session_id: str):
    new_id = f"branch-{uuid.uuid4().hex[:6]}"
    demo_sessions[new_id] = {
        "session_id": new_id,
        "title": "Branched Conversation (Demo)",
        "created_at": datetime.datetime.now().isoformat(),
        "parent_session_id": session_id,
        "messages": [],
    }
    return {"session_id": new_id, "status": "ok"}


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(req: Request):
    body = await req.json()
    user_msg = body.get("message", "Market analysis query")
    sym = body.get("symbol", "AAPL")
    session_id = body.get("session_id") or f"session-{uuid.uuid4().hex[:6]}"

    # Fetch symbol details if available
    stock = STOCKS_DATA.get(sym.upper()) or STOCKS_DATA.get("AAPL")
    q = stock["quote"]

    simulated_reply = (
        f"**[StockAI Terminal — Interactive Demo Response]**\n\n"
        f"Here is the simulated equity overview for **{q['company_name']} ({q['symbol']})**:\n\n"
        f"- **Current Demo Price**: {q['currency']} {q['price']:.2f} ({'+' if q['change'] >= 0 else ''}{q['change_percent']:.2f}%)\n"
        f"- **Day Trading Range**: {q['currency']} {q['day_low']:.2f} – {q['day_high']:.2f}\n"
        f"- **Trading Volume**: {q['volume']:,} shares\n"
        f"- **Technical Indicator**: RSI (14) = 56.8 | MACD = +1.45 (Neutral-Bullish Bias)\n"
        f"- **Analyst Target (Simulated)**: {q['currency']} {round(q['price'] * 1.084, 2)} (Moderate Buy consensus)\n\n"
        f"> **Notice**: *You are interacting with the StockAI Terminal Demo Edition. "
        f"This response is generated by a lightweight simulated engine (<60MB container footprint) "
        f"without requiring local Ollama LLM downloads or GPU acceleration.*"
    )

    async def event_generator():
        # Stream chunks simulating AI tokens
        words = simulated_reply.split(" ")
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i : i + 3]) + " "
            yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
            await asyncio.sleep(0.04)

        done_payload = {
            "response": simulated_reply,
            "session_id": session_id,
            "user_message_id": f"msg-u-{uuid.uuid4().hex[:4]}",
            "assistant_message_id": f"msg-a-{uuid.uuid4().hex[:4]}",
        }
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat", tags=["Chat"])
@app.post("/api/chat", tags=["Chat"])
def standard_chat_notice():
    return {
        "status": "disabled_in_demo",
        "response": "Interactive streaming chat is active at /api/chat/stream.",
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Models & Document Handlers
# ---------------------------------------------------------

@app.get("/api/models", tags=["Chat"])
def get_models():
    return {
        "available": True,
        "models": ["cloud-market-preview", "demo-edition"],
        "embedding_models": ["not-applicable"],
        "status": "market_cloud_preview",
    }


@app.get("/api/documents", tags=["Documents"])
def get_documents():
    return {
        "documents": [],
        "total": 0,
        "status": "disabled_in_demo",
        "message": "Document storage and RAG are disabled in this lightweight demonstration.",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
def delete_document(doc_id: str):
    return {"status": "ok"}


@app.post("/api/forecast/custom/evaluate", tags=["Forecasting"])
async def evaluate_custom_forecast(req: Request):
    body = await req.json()
    sym = body.get("symbol", "AAPL").upper()
    return {
        "success": True,
        "symbol": sym,
        "improvement_vs_baseline_percent": 14.8,
        "custom_metrics": {
            "mae_percent": 1.72,
            "test_sample_size": 30,
            "directional_accuracy": 0.82,
        },
        "baseline_metrics": {
            "mae_percent": 2.02,
        },
        "status": "SIMULATED_DEMO",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.post("/api/crawl", tags=["Documents"])
async def crawl_url(req: Request):
    body = await req.json()
    url = body.get("url", "https://example.com")
    ticker = body.get("ticker", "DEMO")
    return {
        "success": True,
        "title": f"Simulated Research Dossier for {ticker}",
        "url": url,
        "chunks_indexed": 3,
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.post("/api/upload/{upload_type}", tags=["Documents"])
async def upload_document(upload_type: str):
    doc_id = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) % 100000
    return {
        "success": True,
        "document_id": doc_id,
        "type": upload_type,
        "message": f"Demo {upload_type} registered successfully.",
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Database Clean & Maintenance
# ---------------------------------------------------------

@app.get("/api/database/clean", tags=["Database"])
@app.post("/api/database/clean", tags=["Database"])
@app.get("/api/db/clean", tags=["Database"])
@app.post("/api/db/clean", tags=["Database"])
def clean_database():
    """
    Cleans demo database state, clears non-default chat sessions,
    and resets in-memory audit logs to a clean slate.
    """
    count_before = len(demo_sessions)
    demo_sessions.clear()
    return {
        "status": "success",
        "action": "database_cleaned",
        "message": "Database and in-memory chat session audit logs cleaned successfully.",
        "sessions_cleared": count_before,
        "current_sessions": 0,
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Static File & UI Shell Serving
# ---------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
@app.get("/markets", include_in_schema=False)
@app.get("/news", include_in_schema=False)
@app.get("/watchlist", include_in_schema=False)
@app.get("/compare", include_in_schema=False)
@app.get("/ai", include_in_schema=False)
@app.get("/stock/{symbol}", include_in_schema=False)
def serve_terminal_shell(symbol: Optional[str] = None):
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        content={
            "application": "StockAI Terminal — Demo Edition",
            "status": "live",
            "disclaimer": DEMO_DISCLAIMER,
        }
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting StockAI Terminal Demo Server on %s:%d", host, port)
    uvicorn.run("demo_server:app", host=host, port=port, log_level="info")
