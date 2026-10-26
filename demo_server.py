"""
StockAI Terminal — Lightweight Demo Edition Backend Server
Designed for DevOps evaluation (Docker, Kubernetes/Minikube, and Render).
Uses zero heavy ML/LLM/database dependencies (No Ollama, FAISS, PyTorch, or Pandas).
Provides deterministic simulated market data fixtures and fast health probes.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("stockai-demo")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FIXTURES_FILE = BASE_DIR / "demo_data" / "stocks.json"

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
    description="Lightweight micro-project demo server with simulated market data fixtures for DevOps testing.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

DEMO_DISCLAIMER = "DEMO MODE — SIMULATED MARKET DATA — NOT LIVE PRICES"


# ---------------------------------------------------------
# Health and Diagnostics Endpoints (Crucial for K8s Probes)
# ---------------------------------------------------------

@app.get("/api/health/live", tags=["Diagnostics"])
def liveness_probe():
    """Lightweight Kubernetes / Render liveness probe."""
    return {
        "application": "running",
        "status": "live",
        "mode": "demo",
        "version": "1.0.0",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/health/ready", tags=["Diagnostics"])
def readiness_probe():
    """Kubernetes readiness probe verifying fixtures are loaded and server is ready."""
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
def health_overview():
    """Comprehensive health diagnostic overview for DevOps reporting."""
    ready = bool(STOCKS_DATA)
    return {
        "application": "running" if ready else "degraded",
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "mode": "demo",
        "cloud_preview": True,
        "database": "available (in-memory fixtures)",
        "ollama": "excluded (lightweight demo)",
        "model": "none (demo mode)",
        "market_data": {
            "status": "available",
            "provider": "Deterministic Local Fixtures",
            "data_status": "SIMULATED_DEMO",
            "symbols": list(STOCKS_DATA.keys()),
        },
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Market & Stock Endpoints (Deterministic Fixtures)
# ---------------------------------------------------------

@app.get("/api/stocks/{symbol}/quote", tags=["Market Data"])
def get_quote(symbol: str, refresh: bool = Query(False)):
    """Return deterministic simulated stock quote."""
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol)
    if not stock_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in demo fixtures. Available symbols: {list(STOCKS_DATA.keys())}",
        )
    return stock_entry["quote"]


@app.get("/api/stocks/{symbol}/history", tags=["Market Data"])
def get_history(
    symbol: str,
    period: str = Query("1mo", pattern="^(1mo|3mo|6mo|1y|all)$"),
    interval: str = Query("1d", pattern="^(1d|1wk|1mo)$"),
    refresh: bool = Query(False),
    auto_adjust: bool = Query(True),
):
    """Return deterministic simulated historical price bars (OHLCV)."""
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol)
    if not stock_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found in demo fixtures. Available symbols: {list(STOCKS_DATA.keys())}",
        )

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
    """Return company profile from demo fixture."""
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol)
    if not stock_entry:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found.")
    return stock_entry["profile"]


@app.get("/api/stocks/{symbol}/technical", tags=["Market Data"])
def get_technical(symbol: str, period: str = Query("6mo")):
    """Return deterministic technical indicators."""
    norm_symbol = symbol.strip().upper()
    stock_entry = STOCKS_DATA.get(norm_symbol)
    if not stock_entry:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found.")

    price = stock_entry["quote"]["price"]
    return {
        "success": True,
        "symbol": norm_symbol,
        "period": period,
        "rsi_14": 56.8,
        "macd": {"macd": 1.45, "signal": 1.18, "histogram": 0.27},
        "sma_20": round(price * 0.985, 2),
        "sma_50": round(price * 0.965, 2),
        "sma_200": round(price * 0.920, 2),
        "sentiment": "NEUTRAL_BULLISH",
        "data_status": "SIMULATED_DEMO",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/symbols/resolve/{symbol}", tags=["Market Data"])
def resolve_symbol(symbol: str):
    """Resolve symbol and market."""
    norm_symbol = symbol.strip().upper()
    is_valid = norm_symbol in STOCKS_DATA
    return {
        "valid": is_valid,
        "symbol": symbol,
        "resolved_symbol": norm_symbol if is_valid else symbol,
        "market": "india" if "NS" in norm_symbol else "us",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/settings", tags=["Configuration"])
def get_settings():
    """Return lightweight demo runtime settings."""
    return {
        "ai_model": "none (demo mode)",
        "context_size": 0,
        "mode": "demo",
        "ollama_enabled": False,
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Excluded Heavy Feature Notice Endpoints
# ---------------------------------------------------------

@app.get("/api/chat", tags=["Notice"])
@app.post("/api/chat", tags=["Notice"])
def chat_demo_notice():
    return {
        "status": "disabled_in_demo",
        "message": "AI Chat and Ollama LLM are disabled in this lightweight DevOps demonstration to keep container images <100MB and portable.",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/documents", tags=["Notice"])
def documents_demo_notice():
    return {
        "status": "disabled_in_demo",
        "documents": [],
        "message": "RAG document indexing and FAISS vector store are excluded from this demo edition.",
        "disclaimer": DEMO_DISCLAIMER,
    }


@app.get("/api/forecast/{symbol}", tags=["Notice"])
def forecast_demo_notice(symbol: str):
    return {
        "status": "simulated_demo",
        "symbol": symbol,
        "forecast": "Probabilistic ML forecasting engines are omitted from the demo container.",
        "disclaimer": DEMO_DISCLAIMER,
    }


# ---------------------------------------------------------
# Web UI Root Serving
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the lightweight demo dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        content={
            "application": "StockAI Terminal — Demo Edition",
            "status": "live",
            "message": "index.html not found, please ensure static/index.html exists.",
            "disclaimer": DEMO_DISCLAIMER,
        }
    )


# Terminal route aliases
@app.get("/markets", include_in_schema=False)
@app.get("/watchlist", include_in_schema=False)
@app.get("/compare", include_in_schema=False)
@app.get("/stock/{symbol}", include_in_schema=False)
def serve_routes(symbol: Optional[str] = None):
    return serve_dashboard()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting StockAI Demo Server on %s:%d", host, port)
    uvicorn.run("demo_server:app", host=host, port=port, log_level="info")
