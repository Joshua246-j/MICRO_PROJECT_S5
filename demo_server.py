"""
StockAI Terminal — Complete Demo Edition Server
Serves the exact full-featured UI with deterministic simulated market data fixtures.
Eliminates all heavy ML, LLM (Ollama), and database dependencies for portable Docker,
Kubernetes (Minikube), and Render cloud hosting.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    "theme": "light",
    "ai_model": "cloud-market-preview",
    "temperature": 0.2,
    "context_size": 4096,
    "citation_visibility": True,
    "max_output_tokens": 256,
    "top_p": 0.9,
    "top_k": 40,
    "ui_density": "default",
    "request_timeout": 30,
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
]


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
    is_valid = norm_symbol in STOCKS_DATA or any(k.startswith(norm_symbol) for k in STOCKS_DATA)
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
# Forecasting Showcase Endpoints
# ---------------------------------------------------------

@app.get("/api/forecast/featured/summary", tags=["Forecasting"])
def featured_forecast_summary(horizon: str = Query("30d")):
    items = []
    for sym in ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TCS.NS"]:
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
    return {
        "success": True,
        "symbol": norm_symbol,
        "model": "historical_drift",
        "selected_model": "historical_drift",
        "target_price": round(price * 1.025, 2),
        "expected_return": "+2.50%",
        "direction": "bullish",
        "probability": "84%",
        "mae": "1.85",
        "confidence": confidence,
        "horizon": horizon,
        "lower_bound": round(price * 0.96, 2),
        "upper_bound": round(price * 1.09, 2),
        "data_status": "SIMULATED_DEMO",
        "disclaimer": DEMO_DISCLAIMER,
        "disagreement": False,
        "model_metadata": {
            "historical_drift": {
                "name": "Historical Drift (Simulated Demo)",
                "description": "Deterministic baseline drift model for lightweight demonstration.",
            }
        },
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
# Chat & RAG Document Fallback Endpoints
# ---------------------------------------------------------

@app.get("/api/chat/sessions", tags=["Chat"])
def get_chat_sessions():
    return {"sessions": []}


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


@app.get("/api/chat", tags=["Chat"])
@app.post("/api/chat", tags=["Chat"])
def chat_response():
    return {
        "response": (
            "Demo Notice: Local Ollama LLM and heavy document RAG models are omitted "
            "from this micro-project container to keep RAM usage under 60MB and ensure fast portability."
        ),
        "status": "disabled_in_demo",
        "model": "cloud-market-preview",
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Static File & UI Serving
# ---------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
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
