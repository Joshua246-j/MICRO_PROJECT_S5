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

# In-memory simulated news
simulated_news = [
    {
        "title": "Tech Markets Rally on Robust Enterprise AI and Cloud Demand (Simulated)",
        "publisher": "Financial Times (Demo)",
        "link": "https://example.com/demo-news-1",
        "published": "2026-10-26T14:00:00Z",
        "retrieved_at": "2026-10-26T15:00:00Z",
        "summary": "Simulated market report: Tech equities demonstrate strength as enterprise cloud spending accelerates.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Global Semiconductor Sector Sees Continued Momentum in Q4 (Simulated)",
        "publisher": "Reuters (Demo)",
        "link": "https://example.com/demo-news-2",
        "published": "2026-10-26T12:30:00Z",
        "retrieved_at": "2026-10-26T15:00:00Z",
        "summary": "Simulated report: Chipmakers report steady order backlogs and expansion across global manufacturing hubs.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Federal Reserve & Reserve Bank Maintain Steady Monetary Stance (Simulated)",
        "publisher": "Bloomberg (Demo)",
        "link": "https://example.com/demo-news-3",
        "published": "2026-10-26T10:15:00Z",
        "retrieved_at": "2026-10-26T15:00:00Z",
        "summary": "Simulated analysis: Central bank liquidity operations remain stable across US and Asian equity sessions.",
        "disclaimer": DEMO_DISCLAIMER,
    },
    {
        "title": "Cloud Infrastructure Spending Projected to Outpace Inflation Rates (Simulated)",
        "publisher": "Wall Street Journal (Demo)",
        "link": "https://example.com/demo-news-4",
        "published": "2026-10-26T08:45:00Z",
        "retrieved_at": "2026-10-26T15:00:00Z",
        "summary": "Simulated report: Multi-cloud deployments and localized inference architectures drive sustained demand.",
        "disclaimer": DEMO_DISCLAIMER,
    },
]

# In-memory simulated chat sessions
demo_sessions: Dict[str, Dict[str, Any]] = {
    "demo-session-1": {
        "session_id": "demo-session-1",
        "title": "AAPL Technical & Price Action Review (Demo)",
        "created_at": "2026-10-26T10:00:00Z",
        "is_pinned": True,
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "What is the current technical setup and price action for AAPL?",
            },
            {
                "id": "msg-2",
                "role": "assistant",
                "content": (
                    "**[StockAI Terminal — Simulated Response]**\n\n"
                    "Apple Inc. (**AAPL**) is currently trading at **$208.55** (+0.53%) on simulated demo data.\n\n"
                    "- **RSI (14)**: 56.8 (Neutral-to-Bullish momentum)\n"
                    "- **MACD**: Positive crossover with a 0.27 histogram expansion\n"
                    "- **Support & Resistance**: Immediate floor at $202.10 with technical ceiling near $225.80\n"
                    "- **Volume**: 48.2M shares transacted\n\n"
                    "> *Notice: This is a deterministic simulation in DevOps demo mode without Ollama or external LLM models.*"
                ),
            },
        ],
    },
    "demo-session-2": {
        "session_id": "demo-session-2",
        "title": "Global Semiconductor Outlook (Demo)",
        "created_at": "2026-10-26T11:00:00Z",
        "is_pinned": False,
        "messages": [
            {
                "id": "msg-3",
                "role": "user",
                "content": "How are semiconductor valuation multiples holding up across the sector?",
            },
            {
                "id": "msg-4",
                "role": "assistant",
                "content": (
                    "**[StockAI Terminal — Simulated Response]**\n\n"
                    "Leading semiconductor equities (e.g. **NVDA** at $120.28) show robust revenue multiple support, driven by datacenter accelerator demand and sovereign AI investments.\n\n"
                    "Valuation metrics remain elevated with forward P/E at 52.4x, supported by deterministic 90-day volume expansion.\n\n"
                    "> *Notice: Simulated demonstration data. No live financial advice.*"
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
def featured_forecast_summary(horizon: str = Query("30d")):
    items = []
    for sym in STOCKS_DATA.keys():
        entry = STOCKS_DATA.get(sym)
        if entry:
            q = entry["quote"]
            p = q["price"]
            items.append({
                "symbol": sym,
                "name": q.get("company_name", sym),
                "current_price": p,
                "forecast_mean": round(p * 1.025, 2),
                "expected_return_pct": 2.50,
                "direction": "bullish",
                "confidence": "HIGH (SIMULATED)",
                "model": "Simulated Demo Ensemble",
                "horizon": horizon,
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
