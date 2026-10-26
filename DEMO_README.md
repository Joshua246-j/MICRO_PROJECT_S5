# MICRO_PROJECT_S5 — StockAI Terminal Demo Edition

> **College Evaluation & DevOps Demonstration Prototype**  
> **Notice:** **DEMO MODE — SIMULATED MARKET DATA — NOT LIVE PRICES.**  
> Built as a portable, lightweight micro-project container runtime (<60MB RAM footprint). Heavy ML/LLM dependencies (Ollama, FAISS, PyTorch) are omitted to enable instant startup and seamless execution inside Docker, Kubernetes (Minikube), and Render Cloud Hosting.

---

## 1. Project Overview & Architecture

StockAI Terminal Demo Edition is a full-featured, responsive financial research dashboard and backend designed for DevOps evaluation.

```
+-------------------------------------------------------------+
|                      Client Browser                         |
|  - HTML5, Tailwind CSS, Lucide Icons, TradingView Charts   |
|  - Watchlist (AAPL, MSFT, NVDA, RELIANCE.NS, TCS.NS)        |
|  - Real-time Interactive Candlestick / Area Chart          |
|  - Prominent Simulated Data Disclosures                     |
+------------------------------+------------------------------+
                               | HTTP (Port 8000 / 8003)
                               v
+-------------------------------------------------------------+
|                 FastAPI / Uvicorn Server                    |
|  - Liveness & Readiness Probes: /api/health/live            |
|  - Deterministic Stock Quotes: /api/stocks/{symbol}/quote   |
|  - Validated OHLCV Price Bars: /api/stocks/{symbol}/history |
|  - Swagger Interactive API Docs: /docs                      |
|  - Deterministic JSON Fixtures (Memory Cached)              |
+-------------------------------------------------------------+
                               |
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
 [ Local Docker ]      [ Minikube K8s ]         [ Render Cloud ]
 (compose.demo.yaml)   (k8s-demo / k8s)         (render.yaml)
```

---

## 2. Key Features

- **Professional Financial UI:** Apple-inspired light/dark theme, responsive mobile & desktop navigation.
- **TradingView Lightweight Charts:** Interactive candlestick and area charts with volume histogram, hover crosshair tooltips, and time period selectors (1M, 3M, 6M, 1Y).
- **Deterministic Market Fixtures:** 100% mathematically consistent OHLCV candles (`High >= max(Open, Close)`, `Low <= min(Open, Close)`, non-negative volumes).
- **DevOps Ready:**
  - Non-root user (`UID 10001`) container security.
  - Health check probes (`/api/health/live`) responding in <5ms.
  - Native Kubernetes manifests with Kustomize support (`namespace`, `deployment`, `service`).
  - Render Blueprint specification for free-tier cloud hosting.

---

## 3. Quick Start & Teacher Demonstration

### A. Run Locally with Python

```powershell
pip install -r requirements.txt
python main.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser.  
Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger API documentation.

---

### B. Docker Demonstration

```powershell
# 1. Build the lightweight container image
docker build -t stockai-demo:1.0.0 .

# 2. Run with Docker Compose
docker compose up -d

# 3. Verify container status and health
docker compose ps
docker compose logs --tail=20

# 4. Test endpoints
curl http://localhost:8000/api/health/live
curl http://localhost:8000/api/stocks/AAPL/quote
curl "http://localhost:8000/api/stocks/AAPL/history?period=1mo&interval=1d"

# 5. Clean up
docker compose down
```

---

### C. Kubernetes (Minikube) Demonstration

```powershell
# 1. Ensure Minikube is running
minikube status

# 2. Load the locally built image into Minikube (no Docker Hub push required)
minikube image load stockai-demo:1.0.0

# 3. Apply Kubernetes manifests
kubectl apply -k k8s/

# 4. Wait for pod rollout (Reaches 1/1 Ready)
kubectl rollout status deployment/stockai-demo -n stockai-demo --timeout=120s
kubectl get pods -n stockai-demo -o wide

# 5. Port-forward to access the application
kubectl port-forward -n stockai-demo svc/stockai-demo 8003:8000

# 6. Verify via browser
# Open: http://127.0.0.1:8003
```

---

### D. Render Cloud Deployment

1. Create a free account at [render.com](https://render.com).
2. Click **New +** -> **Blueprint**.
3. Connect your repository: `https://github.com/Joshua246-j/MICRO_PROJECT_S5`.
4. Render automatically detects `render.yaml` and deploys the web service on branch `main`.
5. Access your public live terminal URL!

---

## 4. Automated QA Test Suite

To run all 37 automated tests:

```powershell
python test_demo.py
```

Results verify:
- Root page returns HTTP 200 with proper demo disclosure.
- Liveness probe returns `status: live`.
- Quotes and OHLCV bars satisfy all financial mathematical invariants.
- Heavy ML and Ollama subsystems are strictly excluded.
