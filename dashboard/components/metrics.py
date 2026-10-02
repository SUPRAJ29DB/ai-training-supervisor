"""
Metric display components for the Streamlit dashboard.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def metric_cards(metrics: dict[str, Any]) -> None:
    """Display a row of metric cards."""
    if not metrics:
        st.info("No metrics available.")
        return
    cols = st.columns(len(metrics))
    for col, (key, value) in zip(cols, metrics.items()):
        with col:
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            st.metric(label=key.replace("_", " ").title(), value=formatted)


def job_status_badge(status: str) -> str:
    """Return a coloured emoji for a job status string."""
    badges = {
        "pending":    "⏳ Pending",
        "queued":     "📋 Queued",
        "running":    "🟢 Running",
        "paused":     "⏸ Paused",
        "completed":  "✅ Completed",
        "failed":     "❌ Failed",
        "stopped":    "🛑 Stopped",
        "recovering": "🔄 Recovering",
    }
    return badges.get(status.lower(), status)


def progress_bar(current: int, total: int, label: str = "Progress") -> None:
    if total > 0:
        pct = current / total
        st.progress(pct, text=f"{label}: Epoch {current}/{total} ({pct*100:.1f}%)")
    else:
        st.progress(0.0, text=f"{label}: Waiting …")
