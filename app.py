"""
AI Training Supervisor - Main Application Entry Point
Starts the FastAPI backend and launches the Streamlit dashboard.
"""

import sys
import os
import subprocess
import threading
import time
import logging
import argparse
from pathlib import Path

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from database.db import init_db
from supervisor.orchestrator import Orchestrator
from utils.config_loader import load_config
from utils.logger import setup_logging


def start_api(host: str, port: int) -> None:
    """Launch the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="warning",
    )


def start_dashboard(port: int) -> None:
    """Launch the Streamlit dashboard in a subprocess."""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "dashboard" / "home.py"),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
    ]
    subprocess.Popen(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Training Supervisor")
    parser.add_argument("--api-only", action="store_true", help="Start only the API server")
    parser.add_argument("--dashboard-only", action="store_true", help="Start only the dashboard")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching the dashboard")
    args = parser.parse_args()

    # ── Load config & logging ──────────────────────────────────────
    config = load_config(ROOT / "config.yaml")
    setup_logging(
        log_level=config.get("app", {}).get("log_level", "INFO"),
        log_dir=ROOT / config.get("app", {}).get("log_dir", "logs"),
        audit_log=ROOT / config.get("app", {}).get("audit_log", "logs/audit.log"),
    )
    logger = logging.getLogger(__name__)
    logger.info("AI Training Supervisor starting up …")

    # ── Initialise database ────────────────────────────────────────
    init_db(config["database"]["url"])
    logger.info("Database initialised.")

    # ── Start orchestrator (singleton) ─────────────────────────────
    orchestrator = Orchestrator(config=config)
    orchestrator.start()
    logger.info("Orchestrator started.")

    api_cfg = config.get("api", {})
    api_host = os.getenv("API_HOST", api_cfg.get("host", "127.0.0.1"))
    api_port = int(os.getenv("API_PORT", api_cfg.get("port", 8000)))
    dash_port = int(
        os.getenv("DASHBOARD_PORT", config.get("dashboard", {}).get("port", 8501))
    )

    if args.api_only:
        logger.info(f"API server: http://{api_host}:{api_port}")
        start_api(api_host, api_port)
        return

    if args.dashboard_only:
        logger.info(f"Dashboard: http://127.0.0.1:{dash_port}")
        start_dashboard(dash_port)
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return

    # ── Default: start both ────────────────────────────────────────
    api_thread = threading.Thread(
        target=start_api, args=(api_host, api_port), daemon=True
    )
    api_thread.start()
    logger.info(f"API server running at http://{api_host}:{api_port}")

    if not args.no_dashboard:
        time.sleep(1)                       # Let the API bind its port first
        start_dashboard(dash_port)
        logger.info(f"Dashboard available at http://127.0.0.1:{dash_port}")

    print(f"\n{'='*60}")
    print(f"  AI Training Supervisor is running!")
    print(f"  API  : http://{api_host}:{api_port}/docs")
    print(f"  Dash : http://127.0.0.1:{dash_port}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Stopping orchestrator …")
        orchestrator.stop()
        logger.info("Supervisor stopped gracefully.")


if __name__ == "__main__":
    main()
