"""
Alert display components for the Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st


def show_alerts(alerts: list[str]) -> None:
    for alert in alerts:
        st.warning(f"⚠️  {alert}")


def show_recovery_history(recoveries: list[dict]) -> None:
    if not recoveries:
        st.info("No recovery attempts recorded.")
        return
    for r in recoveries:
        status = r.get("status", "unknown")
        icon = "✅" if status == "success" else "❌"
        with st.expander(f"{icon} Attempt #{r.get('attempt_number', '?')} — {r.get('error_type', 'unknown')}"):
            st.write(f"**Action taken:** {r.get('action_taken', 'N/A')}")
            st.write(f"**Diagnosis:** {r.get('diagnosis', 'N/A')}")
            st.write(f"**Outcome:** {r.get('outcome_notes', 'N/A')}")
            st.write(f"**Time:** {r.get('created_at', 'N/A')}")
