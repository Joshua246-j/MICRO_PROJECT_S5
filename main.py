"""
StockAI Terminal Entry Point
Micro-Project S5 Demo Edition
"""

import os
import uvicorn
from demo_server import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, log_level="info")
