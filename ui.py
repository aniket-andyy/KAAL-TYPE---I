
import os
import streamlit as st
import requests

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="LLM Security Benchmark",
    page_icon="🛡️",
    layout="wide",
)

# Load Streamlit Cloud Secrets into environment variables.
# Existing environment variables take precedence.
for key, value in st.secrets.items():
    if not isinstance(value, (dict, list)):
        os.environ.setdefault(key, str(value))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

FIELDS = [f"F{i}" for i in range(1, 9)]

FIELD_NAMES = {
    "F1": "Attack Success Evaluation",
    "F2": "Prompt-Injection Testing",
    "F3": "Sensitive-Information Disclosure",
    "F4": "Unauthorized Actions / Excessive Agency",
    "F5": "Adaptive Attack Generation",
    "F6": "Vulnerability Memory / Recurrence",
    "F7": "Blast-Radius / Impact Assessment",
    "F8": "Defense Effectiveness Evaluation",
}

# =========================================================
# API CLIENT
# =========================================================

st.sidebar.title("⚙️ Configuration")

api_base_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value=API_BASE_URL,
    help="Use your deployed backend's HTTPS URL on Streamlit Cloud.",
).strip().rstrip("/")

api_token = st.sidebar.text_input(
    "API Token",
    value=os.getenv("API_TOKEN", ""),
    type="password",
)

def api_request(method, endpoint, **kwargs):
    """Send an authenticated request to the FastAPI backend."""
    if not api_base_url:
        st.error("Configure your FastAPI backend URL.")
        return None

    headers = kwargs.pop("headers", {}).copy()

    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        response = requests.request(
            method=method,
            url=f"{api_base_url}{endpoint}",
            headers=headers,
            timeout=60,
            **kwargs,
        )

        response.raise_for_status()

        if response.status_code == 204 or not response.content:
            return {}

        return response.json()

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        st.error(f"Backend HTTP error ({status}): {body}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the backend. Check the backend URL, "
            "deployment status, and network access."
        )

    except requests.exceptions.Timeout:
        st.error("The backend request timed out.")

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

    except ValueError:
        st.error("The backend returned an invalid JSON response.")

    return None


def get_items(data):
    """Accept common list response formats."""
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        items = data.get("items", data.get("results", []))
        return items if isinstance(items, list) else []

    return []


def format_score(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2%}"
    except (ValueError, TypeError):
        return str(value)


def show_error_details(data):
    if isinstance(data, dict) and data.get("detail"):
        st.error(str(data["detail"]))


# =========================================================
# HEADER AND BACKEND HEALTH
# =========================================================

st.title("🛡️ LLM Security Benchmark")
st.caption(
    "Evaluate model security across eight benchmark fields, "
    "inspect test results, and generate audit reports."
)

if not api_token:
    st.warning("Enter your API token in the sidebar if your backend requires one.")

health = api_request("GET", "/health/ready")

if not isinstance(health, dict) or health.get("status") != "ready":
    st.warning(
        "Backend is not ready. Verify the backend URL, authentication, "
        "and that the /health/ready endpoint exists."
    )
    st.stop()

st.success("FastAPI backend connected and ready.")

# =========================================================
# NAVIGATION
# =========================================================

tab_datasets, tab_runs, tab_results = st.tabs(
    ["📚 Datasets", "🚀 Benchmark Runs", "📊 Results & Reports"]
)

# =========================================================
# TAB 1: DATASETS
# =========================================================

with tab_datasets:
    st.header("Dataset Management")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Generate Dataset")

        with st.form("generate_dataset_form"):
            version = st.text_input(
                "Dataset version",
                value="v1.0-draft",
            )

            selected_fields = st.multiselect(
                "Benchmark fields",
                options=FIELDS,
                default=["F1", "F3"],
                format_func=lambda field: f"{field} — {FIELD_NAMES[field]}",
            )

            generate = st.form_submit_button(
                "Generate Dataset (Qwen)",
                use_container_width=True,
            )

        if generate:
            if not version.strip():
                st.error("Enter a dataset version.")
            elif not selected_fields:
                st.error("Select at least one benchmark field.")
            else:
                payload = {
                    "version": version.strip(),
                    "fields": selected_fields,
                }

                with st.spinner("Requesting dataset generation..."):
                    result = api_request(
                        "POST",
                        "/api/v1/datasets/generate",
                        json=payload,
                    )

                if result is not None:
                    st.success("Dataset generation request completed.")
                    st.json(result)

    with right:
        st.subheader("Existing Datasets")

        if st.button("🔄 Refresh Datasets", key="refresh_datasets"):
            st.rerun()

        dataset_data = api_request(
            "GET",
            "/api/v1/datasets",
            params={"limit": 50},
        )

        datasets = get_items(dataset_data)

        if datasets:
            for dataset in datasets:
                if not isinstance(dataset, dict):
                    continue

                dataset_id = dataset.get("id", "")
                dataset_version = dataset.get("version", "Unknown")
                status = dataset.get("status", "unknown")
                total_cases = dataset.get("total_cases", 0)

                with st.container(border=True):
                    st.markdown(f"**{dataset_version}**")
                    st.write(f"Status: `{status}`")
                    st.write(f"Test cases: {total_cases}")

                    if status == "draft" and dataset_id:
                        if st.button(
                            f"Publish {dataset_version}",
                            key=f"publish_{dataset_id}",
                        ):
                            with st.spinner("Publishing dataset..."):
                                result = api_request(
                                    "POST",
                                    f"/api/v1/datasets/{dataset_id}/publish",
                                )

                            if result is not None:
                                st.success("Publish request completed.")
                                st.rerun()
        elif dataset_data is not None:
            st.info("No datasets found.")

# =========================================================
# TAB 2: BENCHMARK RUNS
# =========================================================

with tab_runs:
    st.header("Execute Benchmark")

    dataset_data = api_request(
        "GET",
        "/api/v1/datasets",
        params={"status": "published", "limit": 50},
    )
    published_datasets = [
        dataset
        for dataset in get_items(dataset_data)
        if isinstance(dataset, dict) and dataset.get("status") == "published"
    ]

    if not published_datasets:
        st.info("Publish a dataset before starting a benchmark run.")
    else:
        dataset_options = {
            f"{item.get('version', 'Unknown')} ({item.get('total_cases', 0)} cases)": item
            for item in published_datasets
        }

        left, right = st.columns([1, 2])

        with left:
            st.subheader("New Benchmark Run")

            with st.form("create_run_form"):
                selected_dataset_label = st.selectbox(
                    "Published dataset",
                    list(dataset_options.keys()),
                )

                selected_fields = st.multiselect(
                    "Fields to test",
                    options=FIELDS,
                    default=["F1", "F3"],
                    format_func=lambda field: f"{field} — {FIELD_NAMES[field]}",
                )

                provider = st.selectbox(
                    "Target provider",
                    ["qwen", "gemini"],
                )

                default_model = (
                    "qwen-max"
                    if provider == "qwen"
                    else os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
                )

                model = st.text_input(
                    "Target model",
                    value=default_model,
                    help="Enter a model ID supported by your provider and backend.",
                )

                concurrency = st.slider(
                    "Concurrency",
                    min_value=1,
                    max_value=8,
                    value=4,
                )

                start_run = st.form_submit_button(
                    "Start Benchmark Run",
                    use_container_width=True,
                )

            if start_run:
                selected_dataset = dataset_options[selected_dataset_label]

                if not selected_fields:
                    st.error("Select at least one field.")
                elif not model.strip():
                    st.error("Enter a target model.")
                else:
                    payload = {
                        "dataset_version": selected_dataset.get("version"),
                        "selected_fields": selected_fields,
                        "target_provider": provider,
                        "target_model": model.strip(),
                        "concurrency": concurrency,
                    }

                    with st.spinner("Submitting benchmark run..."):
                        result = api_request(
                            "POST",
                            "/api/v1/runs",
                            json=payload,
                        )

                    if result is not None:
                        run_id = result.get("id")
                        if run_id:
                            st.success(f"Run submitted: {run_id}")
                        else:
                            st.success("Run request completed.")
                        st.json(result)

        with right:
            st.subheader("Recent Runs")

            if st.button("🔄 Refresh Runs", key="refresh_runs"):
                st.rerun()

            runs_data = api_request(
                "GET",
                "/api/v1/runs",
                params={"limit": 20},
            )
            runs = get_items(runs_data)

            if runs:
                for run in runs:
                    if not isinstance(run, dict):
                        continue

                    run_id = str(run.get("id", "unknown"))
                    status = run.get("status", "unknown")
                    provider_name = run.get("target_provider", "unknown")
                    model_name = run.get("target_model", "unknown")

                    with st.container(border=True):
                        st.markdown(f"**Run:** `{run_id}`")
                        st.write(f"Status: `{status}`")
                        st.write(f"Target: `{provider_name}/{model_name}`")

                        if status in ("pending", "running"):
                            if st.button(
                                "Cancel Run",
                                key=f"cancel_{run_id}",
                            ):
                                result = api_request(
                                    "POST",
                                    f"/api/v1/runs/{run_id}/cancel",
                                )
                                if result is not None:
                                    st.success("Cancellation request submitted.")
                                    st.rerun()
            elif runs_data is not None:
                st.info("No benchmark runs found.")

# =========================================================
# TAB 3: RESULTS AND REPORTS
# =========================================================

with tab_results:
    st.header("Results & AI Security Reports")

    runs_data = api_request(
        "GET",
        "/api/v1/runs",
        params={"limit": 50},
    )
    runs = [
        run for run in get_items(runs_data)
        if isinstance(run, dict) and run.get("id")
    ]

    if not runs:
        st.info("Start a benchmark run before viewing results.")
    else:
        run_options = {
            (
                f"{str(run['id'])[:8]} — "
                f"{run.get('status', 'unknown')} — "
                f"{run.get('dataset_version', 'dataset')}"
            ): str(run["id"])
            for run in runs
        }

        selected_label = st.selectbox(
            "Select a run",
            list(run_options.keys()),
        )
        selected_run_id = run_options[selected_label]

        if st.button("🔄 Refresh Results"):
            st.rerun()

        left, right = st.columns(2)

        with left:
            st.subheader("Run Summary")

            summary = api_request(
                "GET",
                f"/api/v1/runs/{selected_run_id}/summary",
            )

            if isinstance(summary, dict):
                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Overall Score",
                    format_score(summary.get("overall_score")),
                )
                metric2.metric(
                    "Critical Failures",
                    summary.get("critical_failures", "N/A"),
                )
                metric3.metric(
                    "Completed Tests",
                    (
                        f"{summary.get('completed_tests', 0)} / "
                        f"{summary.get('planned_tests', 0)}"
                    ),
                )

                incomplete = summary.get("incomplete_fields", [])
                if incomplete:
                    st.warning(
                        "Incomplete fields: " + ", ".join(incomplete)
                    )

                st.markdown("#### Field Breakdown")

                for field in summary.get("fields", []):
                    if not isinstance(field, dict):
                        continue

                    field_id = field.get("field_id", "?")
                    field_name = field.get(
                        "name",
                        FIELD_NAMES.get(field_id, "Unknown"),
                    )
                    score_info = field.get("score") or {}
                    score_value = (
                        score_info.get("value")
                        if isinstance(score_info, dict)
                        else score_info
                    )

                    st.markdown(
                        f"- **{field_id}: {field_name}** — "
                        f"Score: {format_score(score_value)} | "
                        f"Passed: {field.get('passed', 0)} | "
                        f"Failed: {field.get('failed', 0)}"
                    )
            elif summary is not None:
                st.json(summary)

        with right:
            st.subheader("AI Evaluation Report")

            if st.button("Generate / Refresh Report"):
                with st.spinner("Requesting report from the backend..."):
                    report = api_request(
                        "GET",
                        f"/api/v1/runs/{selected_run_id}/report",
                    )

                if isinstance(report, dict):
                    st.session_state[f"report_{selected_run_id}"] = report

            report = st.session_state.get(f"report_{selected_run_id}")

            if isinstance(report, dict):
                report_text = report.get(
                    "report_text",
                    report.get("report", ""),
                )

                if report.get("generated_at"):
                    st.caption(f"Generated: {report['generated_at']}")

                if report_text:
                    st.markdown(report_text)
                    st.download_button(
                        "Download Report (.md)",
                        data=report_text,
                        file_name=f"security_report_{selected_run_id}.md",
                        mime="text/markdown",
                    )
                else:
                    st.json(report)
            else:
                st.info(
                    "Generate a report to inspect the security evaluation."
                )

        st.divider()
        st.subheader("Run Results")

        if st.button("Load Individual Test Results"):
            results = api_request(
                "GET",
                f"/api/v1/runs/{selected_run_id}/results",
            )

            if results is not None:
                st.session_state[f"results_{selected_run_id}"] = results

        results = st.session_state.get(f"results_{selected_run_id}")

        if results is not None:
            result_items = get_items(results)

            if result_items:
                st.dataframe(result_items, use_container_width=True)
                st.download_button(
                    "Download Results (JSON)",
                    data=__import__("json").dumps(
                        results,
                        indent=2,
                        default=str,
                    ),
                    file_name=f"benchmark_results_{selected_run_id}.json",
                    mime="application/json",
                )
            else:
                st.json(results)

st.caption(
    "LLM Security Benchmark • Verify findings in a controlled test "
    "environment before drawing security conclusions."
)
