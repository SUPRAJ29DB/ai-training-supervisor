"""
FastAPI main application.
Mounts all route modules and configures middleware.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import jobs, metrics, experiments, reports
from database.db import init_db
from utils.config_loader import load_config
from utils.logger import setup_logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config(ROOT / "config.yaml")
    setup_logging(
        log_level=config.get("app", {}).get("log_level", "INFO"),
        log_dir=ROOT / config.get("app", {}).get("log_dir", "logs"),
        audit_log=ROOT / config.get("app", {}).get("audit_log", "logs/audit.log"),
    )
    init_db(config["database"]["url"])
    logger.info("FastAPI server started.")
    yield
    logger.info("FastAPI server shutting down.")


app = FastAPI(
    title="AI Training Supervisor API",
    description="REST API for managing and monitoring AI/ML training jobs.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit dashboard to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(jobs.router,        prefix="/api/v1/jobs",        tags=["Jobs"])
app.include_router(metrics.router,     prefix="/api/v1/metrics",     tags=["Metrics"])
app.include_router(experiments.router, prefix="/api/v1/experiments", tags=["Experiments"])
app.include_router(reports.router,     prefix="/api/v1/reports",     tags=["Reports"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "AI Training Supervisor"}
