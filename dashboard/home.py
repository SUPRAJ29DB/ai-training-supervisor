"""
AI Training Supervisor – Main Streamlit Dashboard
Implements all 8 pages using a sidebar navigation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.components.charts import resource_gauges, loss_curve, experiment_comparison_table
from dashboard.components.metrics import job_status_badge, metric_cards, progress_bar
from dashboard.components.alerts import show_alerts, show_recovery_history

API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="AI Training Supervisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem; }
    
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    
    .status-running  { color: #4ade80; font-weight: 600; }
    .status-failed   { color: #f87171; font-weight: 600; }
    .status-pending  { color: #facc15; font-weight: 600; }
    .status-complete { color: #60a5fa; font-weight: 600; }
    
    .sidebar .sidebar-content { background: rgba(15,12,41,0.95); }
    
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.8rem;
    }
    
    .stProgress > div > div { background: linear-gradient(90deg, #667eea, #764ba2); }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path: str, data: dict = None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_available() -> bool:
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar navigation ────────────────────────────────────────────────────────

PAGES = [
    "🏠 Overview",
    "⚙️ Training Jobs",
    "📊 Monitoring",
    "🔄 Recovery",
    "📈 Evaluation",
    "🧪 Experiments",
    "🤖 AI Advisor",
    "⚙ Settings",
]

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);'>
        <span style='font-size: 2.5rem;'>🤖</span><br>
        <span style='color: #667eea; font-weight: 700; font-size: 1rem;'>AI Training Supervisor</span><br>
        <span style='color: rgba(255,255,255,0.5); font-size: 0.75rem;'>v1.0.0</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    selected = st.radio("Navigation", PAGES, label_visibility="hidden")

    st.markdown("---")
    api_ok = api_available()
    if api_ok:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline")
    st.caption(f"API: {API_BASE}")


# ════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ════════════════════════════════════════════════════════════════════

if "Overview" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>🏠 System Overview</h1>
        <p>Real-time status of your AI training environment</p>
    </div>
    """, unsafe_allow_html=True)

    # Resource metrics
    st.subheader("💻 System Resources")
    snap = api_get("/metrics/resources", default={})
    if snap:
        resource_gauges(snap)
    else:
        st.warning("Resource monitor not available. Start the API server.")

    st.divider()

    # Active jobs
    st.subheader("⚡ Active Jobs")
    jobs = api_get("/jobs/", default=[])
    active = [j for j in jobs if j.get("status") in ("running", "recovering", "paused")]

    if not active:
        st.info("No active training jobs. Go to **Training Jobs** to start one.")
    else:
        for job in active:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{job['name']}** `{job['framework']}`")
                    progress_bar(job.get("current_epoch", 0), job.get("total_epochs", 1))
                with col2:
                    st.write(job_status_badge(job["status"]))
                with col3:
                    if st.button("⏹ Stop", key=f"stop_{job['id']}"):
                        api_post(f"/jobs/{job['id']}/action", {"action": "stop"})
                        st.rerun()

    st.divider()

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    all_jobs = api_get("/jobs/", default=[])
    with col1:
        st.metric("Total Jobs",      len(all_jobs))
    with col2:
        st.metric("Completed",       sum(1 for j in all_jobs if j.get("status") == "completed"))
    with col3:
        st.metric("Failed",          sum(1 for j in all_jobs if j.get("status") == "failed"))
    with col4:
        experiments = api_get("/experiments/", default=[])
        st.metric("Experiments",     len(experiments))

    if st.button("🔄 Refresh"):
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# PAGE 2: Training Jobs
# ════════════════════════════════════════════════════════════════════

elif "Training Jobs" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>⚙️ Training Jobs</h1>
        <p>Create, start, pause, resume, and stop training jobs</p>
    </div>
    """, unsafe_allow_html=True)

    tab_list, tab_create = st.tabs(["📋 Job List", "➕ Create Job"])

    with tab_list:
        jobs = api_get("/jobs/", default=[])
        if not jobs:
            st.info("No jobs yet. Create one in the **Create Job** tab.")
        else:
            for job in jobs:
                with st.expander(
                    f"{'🟢' if job['status']=='running' else '⏸' if job['status']=='paused' else '⏳'} "
                    f"Job #{job['id']}: {job['name']} — {job_status_badge(job['status'])}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Framework:** {job.get('framework', 'N/A')}")
                        st.write(f"**Model type:** {job.get('model_type', 'N/A')}")
                        st.write(f"**Created:** {job.get('created_at', 'N/A')}")
                        if job.get("error_message"):
                            st.error(f"Error: {job['error_message']}")
                    with col2:
                        progress_bar(job.get("current_epoch", 0), job.get("total_epochs", 1))

                    # Controls
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("▶ Start",  key=f"start_{job['id']}", disabled=job["status"] not in ("pending","queued")):
                            api_post(f"/jobs/{job['id']}/action", {"action": "start"})
                            st.rerun()
                    with c2:
                        if st.button("⏸ Pause",  key=f"pause_{job['id']}", disabled=job["status"] != "running"):
                            api_post(f"/jobs/{job['id']}/action", {"action": "pause"})
                            st.rerun()
                    with c3:
                        if st.button("▶ Resume", key=f"resume_{job['id']}", disabled=job["status"] != "paused"):
                            api_post(f"/jobs/{job['id']}/action", {"action": "resume"})
                            st.rerun()
                    with c4:
                        if st.button("⏹ Stop",   key=f"stop2_{job['id']}", disabled=job["status"] in ("completed","failed","stopped")):
                            api_post(f"/jobs/{job['id']}/action", {"action": "stop"})
                            st.rerun()

    with tab_create:
        st.subheader("Create New Training Job")
        with st.form("create_job_form"):
            name        = st.text_input("Job Name *", placeholder="My CIFAR-10 Experiment")
            description = st.text_area("Description")
            framework   = st.selectbox("Framework", ["pytorch", "sklearn", "tensorflow", "custom"])
            model_type  = st.selectbox("Model Type", ["classifier", "regressor", "detector", "custom"])
            priority    = st.slider("Priority", 0, 10, 0)

            st.markdown("**Training Configuration**")
            col1, col2, col3 = st.columns(3)
            with col1:
                script_path = st.text_input("Script Path", value="training/examples/sample_classifier.py")
            with col2:
                epochs      = st.number_input("Epochs", 1, 1000, 20)
            with col3:
                batch_size  = st.number_input("Batch Size", 1, 512, 32)
            lr = st.number_input("Learning Rate", 1e-6, 1.0, 1e-3, format="%.6f")

            submitted = st.form_submit_button("🚀 Create Job", type="primary")
            if submitted and name:
                result = api_post("/jobs/", {
                    "name": name,
                    "description": description,
                    "framework": framework,
                    "model_type": model_type,
                    "priority": priority,
                    "config": {
                        "script_path": script_path,
                        "working_dir": ".",
                        "batch_size":  batch_size,
                        "num_epochs":  epochs,
                        "learning_rate": lr,
                        "args": ["--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr)],
                    },
                })
                if result:
                    st.success(f"✅ Job created with ID={result['id']}. Go to Job List to start it.")


# ════════════════════════════════════════════════════════════════════
# PAGE 3: Monitoring
# ════════════════════════════════════════════════════════════════════

elif "Monitoring" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Live Monitoring</h1>
        <p>Real-time training metrics and system resources</p>
    </div>
    """, unsafe_allow_html=True)

    jobs = api_get("/jobs/", default=[])
    running = [j for j in jobs if j.get("status") == "running"]
    job_options = {f"Job #{j['id']}: {j['name']}": j['id'] for j in jobs}

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_job_label = st.selectbox("Select Job", list(job_options.keys()) or ["No jobs"])
    with col2:
        auto_refresh = st.toggle("Auto-refresh (5s)", value=False)

    if job_options and selected_job_label in job_options:
        job_id = job_options[selected_job_label]
        metrics = api_get(f"/metrics/job/{job_id}", default=[])

        if metrics:
            latest = metrics[-1]
            st.subheader("Latest Metrics")
            metric_cards({
                "Train Loss": latest.get("train_loss", 0),
                "Val Loss":   latest.get("val_loss", 0),
                "Accuracy":   latest.get("accuracy", 0),
                "Epoch":      latest.get("epoch", 0),
            })

            st.subheader("Loss Curves")
            loss_curve(metrics)

            st.subheader("Accuracy")
            from dashboard.components.charts import accuracy_curve
            accuracy_curve(metrics)
        else:
            st.info("No metrics yet. Job may not have started training.")

    st.subheader("💻 System Resources")
    snap = api_get("/metrics/resources", default={})
    if snap:
        resource_gauges(snap)

    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# PAGE 4: Recovery
# ════════════════════════════════════════════════════════════════════

elif "Recovery" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>🔄 Recovery Dashboard</h1>
        <p>Error details, recovery actions, and history</p>
    </div>
    """, unsafe_allow_html=True)

    jobs = api_get("/jobs/", default=[])
    failed = [j for j in jobs if j.get("status") in ("failed", "recovering")]

    if not failed:
        st.success("✅ No failed jobs. System is healthy.")
    else:
        for job in failed:
            st.error(f"❌ **Job #{job['id']}: {job['name']}** — {job.get('error_message', 'Unknown error')}")

    st.divider()
    st.subheader("Recovery History")
    job_options = {f"Job #{j['id']}: {j['name']}": j['id'] for j in jobs}
    if job_options:
        sel = st.selectbox("Select Job", list(job_options.keys()))
        if sel:
            # Recovery attempts would be fetched via API
            st.info("Recovery history will appear here as jobs recover.")


# ════════════════════════════════════════════════════════════════════
# PAGE 5: Evaluation
# ════════════════════════════════════════════════════════════════════

elif "Evaluation" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Model Evaluation</h1>
        <p>Task-specific metrics, predictions, and evaluation reports</p>
    </div>
    """, unsafe_allow_html=True)

    experiments = api_get("/experiments/", default=[])
    exp_options = {f"Exp #{e['id']}: {e['name']}": e['id'] for e in experiments}

    if not exp_options:
        st.info("No experiments found. Create a job first.")
    else:
        sel = st.selectbox("Select Experiment", list(exp_options.keys()))
        if sel:
            exp_id = exp_options[sel]
            eval_result = api_get(f"/reports/experiment/{exp_id}")
            if eval_result:
                st.subheader("Evaluation Metrics")
                metric_cards(eval_result.get("metrics", {}))
                st.write(f"**Split:** {eval_result.get('dataset_split')}")
                st.write(f"**Report:** `{eval_result.get('report_path')}`")
            else:
                st.info("No evaluation results yet. Complete training first.")


# ════════════════════════════════════════════════════════════════════
# PAGE 6: Experiments
# ════════════════════════════════════════════════════════════════════

elif "Experiments" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>🧪 Experiment Tracking</h1>
        <p>Compare experiments, configurations, and results</p>
    </div>
    """, unsafe_allow_html=True)

    experiments = api_get("/experiments/", default=[])

    st.subheader("Experiment Comparison")
    experiment_comparison_table(experiments)

    st.divider()
    st.subheader("Create Experiment Record")
    with st.form("create_exp_form"):
        name        = st.text_input("Experiment Name *")
        description = st.text_area("Description")
        framework   = st.selectbox("Framework", ["pytorch", "sklearn", "tensorflow"])
        model_type  = st.text_input("Model Type", value="classifier")
        dataset_ref = st.text_input("Dataset Reference", placeholder="data/datasets/train.csv")
        tags_str    = st.text_input("Tags (comma-separated)", placeholder="baseline,v1")

        submitted = st.form_submit_button("Save Experiment", type="primary")
        if submitted and name:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            result = api_post("/experiments/", {
                "name": name, "description": description,
                "framework": framework, "model_type": model_type,
                "dataset_ref": dataset_ref, "tags": tags,
                "hyperparams": {},
            })
            if result:
                st.success(f"Experiment created: ID={result['id']}")


# ════════════════════════════════════════════════════════════════════
# PAGE 7: AI Advisor
# ════════════════════════════════════════════════════════════════════

elif "AI Advisor" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 AI Improvement Advisor</h1>
        <p>Evidence-based recommendations for improving training performance</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "ℹ️ The advisor analyses your training metrics and suggests experiments. "
        "**Recommendations are proposals only** – no changes are made without your approval."
    )

    experiments = api_get("/experiments/", default=[])
    exp_options = {f"Exp #{e['id']}: {e['name']}": e['id'] for e in experiments}

    if exp_options:
        sel = st.selectbox("Select Experiment to Analyse", list(exp_options.keys()))
        if st.button("🔍 Analyse & Generate Recommendations", type="primary"):
            exp_id = exp_options[sel]
            result = api_post(f"/experiments/{exp_id}/analyse")
            if result:
                st.success(f"Generated {result.get('recommendations_created', 0)} recommendation(s).")

        st.divider()
        st.subheader("Pending Recommendations")
        if sel and sel in exp_options:
            exp_id = exp_options[sel]
            recs = api_get(f"/experiments/{exp_id}/recommendations", default=[])
            pending = [r for r in recs if r.get("status") == "pending"]
            if not pending:
                st.info("No pending recommendations.")
            else:
                for rec in pending:
                    with st.expander(f"💡 {rec['title']}"):
                        st.write(f"**Source:** {rec.get('source', 'N/A')}")
                        st.write(f"**Description:** {rec.get('description', '')}")
                        st.write(f"**Rationale:** {rec.get('rationale', '')}")
                        if rec.get("proposed_config"):
                            st.json(rec["proposed_config"])
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Approve", key=f"approve_{rec['id']}"):
                                requests.patch(
                                    f"{API_BASE}/experiments/recommendations/{rec['id']}",
                                    json={"status": "approved"},
                                )
                                st.rerun()
                        with col2:
                            if st.button("❌ Reject", key=f"reject_{rec['id']}"):
                                requests.patch(
                                    f"{API_BASE}/experiments/recommendations/{rec['id']}",
                                    json={"status": "rejected"},
                                )
                                st.rerun()
    else:
        st.info("No experiments yet.")


# ════════════════════════════════════════════════════════════════════
# PAGE 8: Settings
# ════════════════════════════════════════════════════════════════════

elif "Settings" in selected:
    st.markdown("""
    <div class='main-header'>
        <h1>⚙ Settings</h1>
        <p>Configure resource limits, retry policies, and notifications</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Settings are managed via `config.yaml`. Restart the application to apply changes.")

    config_path = ROOT / "config.yaml"
    if config_path.exists():
        with st.expander("📄 View config.yaml", expanded=True):
            st.code(config_path.read_text(), language="yaml")
        if st.button("Open config.yaml in editor"):
            st.info(f"Edit the file at: `{config_path}`")
    else:
        st.error("config.yaml not found!")

    st.divider()
    st.subheader("🔌 LLM Connection (Raspberry Pi)")
    col1, col2 = st.columns(2)
    with col1:
        pi_host = st.text_input("Raspberry Pi IP", value="192.168.1.100")
        pi_port = st.number_input("Ollama Port", value=11434)
    with col2:
        if st.button("Test Connection"):
            try:
                r = requests.get(f"http://{pi_host}:{pi_port}/api/tags", timeout=3)
                if r.status_code == 200:
                    st.success("✅ Connected to Ollama!")
                else:
                    st.error(f"Connection failed: HTTP {r.status_code}")
            except Exception as e:
                st.error(f"Cannot reach {pi_host}:{pi_port} — {e}")
