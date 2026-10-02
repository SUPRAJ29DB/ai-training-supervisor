"""
Reusable Streamlit chart components for the dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def loss_curve(metrics: list[dict[str, Any]], title: str = "Training Curves") -> None:
    """Render a training/validation loss curve."""
    if not metrics:
        st.info("No metrics available yet.")
        return
    df = pd.DataFrame(metrics)
    df = df.rename(columns={"train_loss": "Train Loss", "val_loss": "Val Loss"})
    cols = [c for c in ["Train Loss", "Val Loss"] if c in df.columns]
    if not cols:
        st.info("No loss data to display.")
        return
    st.line_chart(df.set_index("epoch")[cols], use_container_width=True)


def accuracy_curve(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return
    df = pd.DataFrame(metrics)
    if "accuracy" not in df.columns:
        return
    df = df.rename(columns={"accuracy": "Accuracy"})
    st.line_chart(df.set_index("epoch")[["Accuracy"]], use_container_width=True)


def resource_gauges(snap: dict[str, float]) -> None:
    """Render 4 resource usage progress bars."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cpu = snap.get("cpu_percent", 0)
        st.metric("CPU", f"{cpu:.1f}%")
        st.progress(min(cpu / 100, 1.0))
    with col2:
        used = snap.get("ram_used_gb", 0)
        total = snap.get("ram_total_gb", 1)
        st.metric("RAM", f"{used:.1f} / {total:.1f} GB")
        st.progress(min(used / max(total, 1), 1.0))
    with col3:
        gpu = snap.get("gpu_utilization", 0)
        st.metric("GPU", f"{gpu:.1f}%")
        st.progress(min(gpu / 100, 1.0))
    with col4:
        vram_used = snap.get("vram_used_gb", 0)
        vram_total = snap.get("vram_total_gb", 6)
        st.metric("VRAM", f"{vram_used:.1f} / {vram_total:.1f} GB")
        st.progress(min(vram_used / max(vram_total, 1), 1.0))


def experiment_comparison_table(experiments: list[dict[str, Any]]) -> None:
    if not experiments:
        st.info("No experiments to compare.")
        return
    rows = []
    for e in experiments:
        rows.append({
            "ID":           e.get("id"),
            "Name":         e.get("name"),
            "Framework":    e.get("framework"),
            "Best Acc":     f"{e.get('best_accuracy', 0) or 0:.4f}",
            "Best Val Loss":f"{e.get('best_val_loss', 0) or 0:.4f}",
            "Status":       e.get("status"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
