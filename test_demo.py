"""
Automated QA Verification Suite for StockAI Terminal Demo Edition.
Tests all required endpoints, schema integrity, OHLCV candle mathematical constraints,
static asset delivery, and safety isolation (absence of heavy ML/LLM runtimes).
"""

import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure import works from current directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_server import app, DEMO_DISCLAIMER

client = TestClient(app)

PASSED = 0
FAILED = 0


def log_test(name: str, passed: bool, details: str = ""):
    global PASSED, FAILED
    if passed:
        PASSED += 1
        print(f" [PASS] {name} {details}")
    else:
        FAILED += 1
        print(f" [FAIL] {name} - ERROR: {details}")


def run_all_tests():
    print("=" * 60)
    print(" STOCKAI TERMINAL DEMO EDITION — TEST SUITE")
    print("=" * 60)

    # 1. Root page serves HTTP 200
    res = client.get("/")
    log_test("Root Page (/)", res.status_code == 200, f"Status: {res.status_code}")
    log_test("Root Page Demo Disclosure", "DEMO MODE — SIMULATED MARKET DATA" in res.text)

    # 2. Health Liveness & Readiness Probes
    live_res = client.get("/api/health/live")
    log_test("/api/health/live (HTTP 200)", live_res.status_code == 200)
    live_json = live_res.json()
    log_test("/api/health/live status=='live'", live_json.get("status") == "live")

    ready_res = client.get("/api/health/ready")
    log_test("/api/health/ready (HTTP 200)", ready_res.status_code == 200)
    ready_json = ready_res.json()
    log_test("/api/health/ready ready==True", ready_json.get("ready") is True)

    health_res = client.get("/api/health")
    log_test("/api/health overview", health_res.status_code == 200 and health_res.json().get("ollama") == "excluded (lightweight demo)")

    # 3. Stock Quotes for All 8 Dashboard Symbols
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    for sym in symbols:
        q_res = client.get(f"/api/stocks/{sym}/quote")
        passed = (q_res.status_code == 200)
        q_json = q_res.json() if passed else {}
        log_test(f"Quote for {sym}", passed, f"Price: {q_json.get('price')} {q_json.get('currency')}")
        
        # Verify schema
        required_fields = ["symbol", "price", "change", "change_percent", "volume", "market_cap", "disclaimer", "data_status"]
        schema_ok = all(k in q_json for k in required_fields)
        log_test(f"Quote Schema {sym}", schema_ok)
        log_test(f"Quote Disclaimer {sym}", q_json.get("disclaimer") == DEMO_DISCLAIMER)

    # 4. Historical OHLCV Bars Verification
    for sym in symbols:
        h_res = client.get(f"/api/stocks/{sym}/history?period=1mo&interval=1d")
        passed = (h_res.status_code == 200)
        h_json = h_res.json() if passed else {}
        bars = h_json.get("data", [])
        log_test(f"History for {sym} (1mo)", passed and len(bars) >= 20, f"Bars: {len(bars)}")

        # Mathematical and data integrity validation
        all_candles_valid = True
        candle_err = ""
        for b in bars:
            o, h, l, c, v = b.get("Open"), b.get("High"), b.get("Low"), b.get("Close"), b.get("Volume")
            if not (h >= o and h >= c):
                all_candles_valid = False
                candle_err = f"High {h} < max(Open {o}, Close {c})"
                break
            if not (l <= o and l <= c):
                all_candles_valid = False
                candle_err = f"Low {l} > min(Open {o}, Close {c})"
                break
            if not (h >= l):
                all_candles_valid = False
                candle_err = f"High {h} < Low {l}"
                break
            if not (isinstance(v, int) and v >= 0):
                all_candles_valid = False
                candle_err = f"Volume {v} not non-negative int"
                break

        log_test(f"OHLCV Constraints {sym}", all_candles_valid, candle_err)

    # 5. Static Assets Verification
    css_res = client.get("/static/tailwind.css")
    log_test("Static Asset: tailwind.css", css_res.status_code == 200)
    index_css_res = client.get("/static/index.css")
    log_test("Static Asset: index.css", index_css_res.status_code == 200)
    app_js_res = client.get("/static/app.js")
    log_test("Static Asset: app.js", app_js_res.status_code == 200)
    charts_res = client.get("/static/vendor/lightweight-charts.js")
    log_test("Static Asset: lightweight-charts.js", charts_res.status_code == 200)
    lucide_res = client.get("/static/vendor/lucide.js")
    log_test("Static Asset: lucide.js", lucide_res.status_code == 200)

    # 6. Safety Exclusions & Custom Endpoints
    chat_res = client.get("/api/chat")
    log_test("Chat Disabled Notice", chat_res.status_code == 200 and chat_res.json().get("status") == "disabled_in_demo")
    docs_res = client.get("/api/documents")
    log_test("Docs RAG Disabled Notice", docs_res.status_code == 200 and docs_res.json().get("status") == "disabled_in_demo")
    custom_res = client.post("/api/forecast/custom/evaluate", json={"symbol": "AAPL"})
    log_test("/api/forecast/custom/evaluate", custom_res.status_code == 200 and custom_res.json().get("success") is True)
    db_clean_res = client.get("/api/database/clean")
    log_test("/api/database/clean", db_clean_res.status_code == 200 and db_clean_res.json().get("action") == "database_cleaned")


    print("=" * 60)
    print(f" TOTAL TESTS: {PASSED + FAILED} | PASSED: {PASSED} | FAILED: {FAILED}")
    print("=" * 60)

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
