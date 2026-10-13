import os

for key, value in st.secrets.items():
    os.environ.setdefault(key, str(value))import os
import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(page_title="LLM Security Benchmark UI", layout="wide", page_icon="🛡️")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")
api_base_url = st.sidebar.text_input("API Base URL", value=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
api_token = st.sidebar.text_input("API Token", value=os.getenv("API_TOKEN", ""), type="password")

# Helper for authenticated requests
def api_request(method: str, endpoint: str, **kwargs):
    url = f"{api_base_url.rstrip('/')}{endpoint}"
    headers = kwargs.pop("headers", {})
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    headers["Content-Type"] = "application/json"
    
    try:
        response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error {e.response.status_code}: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

# --- Main App ---
st.title("🛡️ LLM Security Benchmark Dashboard")

# Check backend health
health = api_request("GET", "/health/ready")
if not health or health.get("status") != "ready":
    st.warning("⚠️ Backend is not ready or unreachable. Check your API URL and ensure the FastAPI server is running.")
    st.stop()

st.success("✅ Backend Connected")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Datasets", "🚀 Benchmark Runs", "📈 Results & Reports"])

# ==========================================
# TAB 1: DATASETS
# ==========================================
with tab1:
    st.header("Dataset Management")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Generate New Dataset")
        with st.form("generate_dataset_form"):
            version = st.text_input("Version Name (e.g., v1.0-draft)", "v1.0-draft")
            fields = st.multiselect(
                "Select Fields", 
                ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"], 
                default=["F1", "F3"]
            )
            submitted = st.form_submit_button("Generate Dataset (Qwen-Max)")
            
            if submitted:
                if not fields:
                    st.error("Please select at least one field.")
                else:
                    with st.spinner("Generating dataset... This may take a minute."):
                        payload = {"version": version, "fields": fields}
                        result = api_request("POST", "/api/v1/datasets/generate", json=payload)
                        if result:
                            st.success(f"Dataset '{version}' generated successfully!")
                            st.rerun()

    with col2:
        st.subheader("Existing Datasets")
        datasets = api_request("GET", "/api/v1/datasets?limit=10")
        if datasets and "items" in datasets:
            for ds in datasets["items"]:
                status_color = "🟢" if ds["status"] == "published" else "🟡"
                st.markdown(f"**{status_color} {ds['version']}** ({ds['status']}) - {ds['total_cases']} cases")
                
                if ds["status"] == "draft":
                    if st.button(f"Publish {ds['version']}", key=f"pub_{ds['id']}"):
                        res = api_request("POST", f"/api/v1/datasets/{ds['id']}/publish")
                        if res:
                            st.success("Published!")
                            st.rerun()
        else:
            st.info("No datasets found.")

# ==========================================
# TAB 2: BENCHMARK RUNS
# ==========================================
with tab2:
    st.header("Execute Benchmark Runs")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Create New Run")
        # Fetch published datasets for selection
        datasets = api_request("GET", "/api/v1/datasets?status=published&limit=10")
        dataset_versions = [ds["version"] for ds in datasets.get("items", [])] if datasets else []
        
        with st.form("create_run_form"):
            ds_version = st.selectbox("Dataset Version", dataset_versions)
            fields = st.multiselect("Fields to Test", ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"], default=["F1", "F3"])
            provider = st.selectbox("Target Provider", ["qwen", "gemini"])
            model = st.text_input("Target Model", value="qwen-max" if provider == "qwen" else "gemini-1.5-pro")
            concurrency = st.slider("Concurrency", 1, 16, 4)
            
            submitted = st.form_submit_button("Start Benchmark Run")
            
            if submitted:
                if not ds_version:
                    st.error("Please select a published dataset version.")
                else:
                    payload = {
                        "dataset_version": ds_version,
                        "selected_fields": fields,
                        "target_provider": provider,
                        "target_model": model,
                        "concurrency": concurrency
                    }
                    with st.spinner("Enqueuing run..."):
                        result = api_request("POST", "/api/v1/runs", json=payload)
                        if result:
                            st.success(f"Run {result['id']} started successfully!")
                            st.rerun()

    with col2:
        st.subheader("Active & Recent Runs")
        runs = api_request("GET", "/api/v1/runs?limit=10")
        if runs and "items" in runs:
            for run in runs["items"]:
                status_emoji = {"pending": "⏳", "running": "🔄", "completed": "✅", "failed": "❌", "cancelled": "⛔"}.get(run["status"], "❓")
                st.markdown(f"**{status_emoji} Run:** `{run['id'][:8]}...` | **Status:** `{run['status']}` | **Target:** `{run['target_provider']}/{run['target_model']}`")
                
                if run["status"] == "running":
                    if st.button("Cancel", key=f"cancel_{run['id']}"):
                        api_request("POST", f"/api/v1/runs/{run['id']}/cancel")
                        st.rerun()
        else:
            st.info("No runs found.")

# ==========================================
# TAB 3: RESULTS & REPORTS
# ==========================================
with tab3:
    st.header("Run Results & AI Reports")
    
    runs = api_request("GET", "/api/v1/runs?limit=10")
    if not runs or not runs.get("items"):
        st.info("No runs available to inspect.")
        st.stop()
        
    run_options = {f"{r['id'][:8]}... ({r['status']}) - {r['dataset_version']}": r["id"] for r in runs["items"]}
    selected_label = st.selectbox("Select a Run to Inspect", list(run_options.keys()))
    selected_run_id = run_options[selected_label]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Run Summary")
        summary = api_request("GET", f"/api/v1/runs/{selected_run_id}/summary")
        if summary:
            st.metric("Overall Score", f"{summary['overall_score']:.2%}" if summary['overall_score'] else "N/A")
            st.metric("Critical Failures", summary['critical_failures'])
            st.metric("Completed Tests", f"{summary['completed_tests']} / {summary['planned_tests']}")
            
            if summary['incomplete_fields']:
                st.warning(f"Incomplete fields: {', '.join(summary['incomplete_fields'])}")
                
            st.markdown("#### Field Breakdown")
            for field in summary['fields']:
                score_val = f"{field['score']['value']:.2%}" if field['score']['value'] is not None else "N/A"
                st.markdown(f"- **{field['field_id']} ({field['name']})**: Score {score_val} | Passed: {field['passed']} | Failed: {field['failed']}")

    with col2:
        st.subheader("AI Evaluation Report")
        if st.button("Generate / Refresh Report"):
            with st.spinner("Asking Gemini to analyze results..."):
                report = api_request("GET", f"/api/v1/runs/{selected_run_id}/report")
                if report:
                    st.session_state[f"report_{selected_run_id}"] = report["report_text"]
                    st.session_state[f"report_time_{selected_run_id}"] = report["generated_at"]
        
        report_text = st.session_state.get(f"report_{selected_run_id}")
        if report_text:
            report_time = st.session_state.get(f"report_time_{selected_run_id}", "")
            st.caption(f"Generated at: {report_time}")
            st.markdown(report_text)
        else:
            st.info("Click 'Generate / Refresh Report' to get the AI security audit.") 
