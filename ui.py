
import os
import json
import streamlit as st
import pandas as pd

from app import (
    run_benchmark,
    load_dataset,
    FIELD_DEFINITIONS,
    DEFAULT_TARGET_MODEL,
    GEMINI_JUDGE_MODEL,
    results_to_csv,
    score_results,
)

st.set_page_config(
    page_title="KAAL Security Benchmark",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ KAAL")
st.caption("LLM Security Benchmark — Phase I")

# ---------------- Configuration ----------------

st.header("Model Configuration")

provider = st.selectbox(
    "Target Provider",
    ["qwen", "gemini"],
)

default_model = (
    DEFAULT_TARGET_MODEL if provider == "qwen"
    else "gemini-2.5-flash"
)

model = st.text_input("Target Model", value=default_model)

target_key = st.text_input(
    "Target API Key",
    type="password",
    help="Your selected provider's API key.",
)

judge_key = st.text_input(
    "Gemini Judge API Key",
    type="password",
    help="Used to evaluate the target model's responses.",
)

judge_model = st.text_input(
    "Judge Model",
    value=GEMINI_JUDGE_MODEL,
)

system_prompt = st.text_area(
    "Target System Prompt (optional)",
    placeholder="Enter a system prompt for the model under test...",
)

# ---------------- Test Configuration ----------------

st.header("Test Configuration")

selected_fields = st.multiselect(
    "Select Security Test Fields",
    options=list(FIELD_DEFINITIONS.keys()),
    default=["F1"],
    format_func=lambda x: f"{x} — {FIELD_DEFINITIONS[x]}",
)

tests_per_field = st.number_input(
    "Tests per field",
    min_value=1,
    max_value=500,
    value=1,
    step=1,
    help="Start with 1 test to verify your configuration.",
)

# ---------------- Run Benchmark ----------------

if st.button("🚀 Run Benchmark", type="primary", use_container_width=True):

    if not selected_fields:
        st.error("Select at least one test field.")

    elif not target_key.strip() or not judge_key.strip():
        st.error("Enter both API keys.")

    elif not model.strip() or not judge_model.strip():
        st.error("Enter both model names.")

    else:
        # The existing app.py reads API keys from environment variables.
        # Keys are supplied at runtime and are not written to a file.
        if provider == "qwen":
            os.environ["QWEN_API_KEY"] = target_key.strip()
        else:
            os.environ["GEMINI_API_KEY"] = target_key.strip()

        # The judge always uses Gemini.
        os.environ["GEMINI_API_KEY"] = judge_key.strip()

        all_results = []
        reports = []

        progress = st.progress(0)
        status = st.empty()

        total_fields = len(selected_fields)

        try:
            for index, field_id in enumerate(selected_fields):
                status.info(
                    f"Running {field_id}: "
                    f"{FIELD_DEFINITIONS[field_id]}"
                )

                # Load the selected dataset.
                dataset = load_dataset(
                    field_id,
                    limit=int(tests_per_field),
                )

                report = run_benchmark(
                    dataset,
                    target_provider=provider,
                    target_model=model.strip(),
                    judge_model=judge_model.strip(),
                    target_system_prompt=system_prompt,
                )

                reports.append(report)
                all_results.extend(report.get("results", []))

                progress.progress((index + 1) / total_fields)

            status.success("Benchmark completed.")

            # ---------------- Results ----------------

            st.header("Results")

            summary = score_results(all_results)

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Tests", summary["total_tests"])
            col2.metric("Evaluated", summary["evaluated_tests"])
            col3.metric("Passed", summary["passed"])
            col4.metric(
                "Critical Failures",
                summary["critical_failure_count"],
            )

            pass_rate = summary["pass_rate_pct"]
            st.metric(
                "Pass Rate",
                f"{pass_rate:.2f}%" if pass_rate is not None else "N/A",
            )

            st.subheader("Field-wise Results")

            field_rows = []
            for report in reports:
                field_summary = report["summary"]["by_field"]
                for fid, details in field_summary.items():
                    field_rows.append({
                        "Field": fid,
                        "Name": details["field_name"],
                        "Tests": details["total"],
                        "Passed": details["passed"],
                        "Failed": details["failed"],
                        "Errors": details["errors"],
                        "Pass Rate (%)": details["pass_rate_pct"],
                    })

            if field_rows:
                st.dataframe(
                    pd.DataFrame(field_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader("Individual Test Results")

            if all_results:
                df = pd.DataFrame(all_results)
                display_columns = [
                    col for col in [
                        "test_id",
                        "field_id",
                        "status",
                        "passed",
                        "attack_succeeded",
                        "critical_failure",
                        "confidence",
                        "reason",
                    ]
                    if col in df.columns
                ]
                st.dataframe(
                    df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Inspect full model responses"):
                    st.json(all_results)

                csv_data = results_to_csv({"results": all_results})

                st.download_button(
                    "⬇️ Download Results (CSV)",
                    data=csv_data,
                    file_name="kaal_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.download_button(
                    "⬇️ Download Full Report (JSON)",
                    data=json.dumps(reports, indent=2, ensure_ascii=False),
                    file_name="kaal_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

        except Exception as exc:
            status.error("Benchmark stopped because an error occurred.")
            st.exception(exc)

st.divider()
st.caption(
    "KAAL Phase I • F1–F8 LLM security evaluation • "
    "Use only systems you are authorized to test."
)
