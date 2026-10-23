"""KAAL Phase I — Streamlit UI
Interactive, dark-themed AI security operations dashboard for the
KAAL LLM Security Benchmark (eight-field adversarial assessment).

Run locally with:  streamlit run ui.py

This UI talks directly to the benchmark engine exposed by app.py.
It uses engine primitives (call_model, _target_messages, _judge_response,
score_results) for a real streaming executor, and the standard run_benchmark()
as a fallback. No fake progress, no invented events.
"""
from __future__ import annotations

import html
import io
import json
import os
import queue
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Engine imports — re-use, do not reinvent
# ---------------------------------------------------------------------------
from app import (
    run_benchmark,
    load_dataset,
    list_datasets,
    generate_dataset,
    results_to_csv,
    save_report,
    score_results,
    validate_dataset,
    FIELD_DEFINITIONS,
    CATEGORY_DISTRIBUTION,
    CATEGORY_NAMES,
    DEFAULT_TARGET_PROVIDER,
    DEFAULT_TARGET_MODEL,
    GEMINI_JUDGE_MODEL,
    KAALConfigError,
    KAALProviderError,
    DATASET_DIR,
    REPORT_DIR,
    # Primitives for streaming executor (these are stable engine interfaces)
    call_model,
    _target_messages,
    _judge_response,
    _slug,
    _require_env,
)


# ---------------------------------------------------------------------------
# Page configuration + dark CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="KAAL — LLM Security Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_THEME_CSS = """
<style>
/* Base palette */
.stApp, .stApp > div { background-color: #0a0e0f !important; color: #d1d5db; }
[data-testid="stSidebar"],
[data-testid="stSidebarContent"],
section[data-testid="stSidebar"] {
    background-color: #0c1213 !important;
    border-right: 1px solid #1a2425;
}
h1, h2, h3, h4 { color: #e5e7eb !important; }
label, .stMarkdown p, .stMarkdown li { color: #c7cbd1 !important; }

/* Terminal */
.kaal-terminal {
    background-color: #040607;
    border: 1px solid #15301d;
    border-radius: 6px;
    padding: 12px 14px;
    font-family: 'Menlo','Consolas','Courier New',monospace;
    font-size: 12.2px;
    line-height: 1.55;
    max-height: 480px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: #4ade80;
    box-shadow: inset 0 0 12px rgba(0,0,0,0.6);
}
.kaal-terminal .ts    { color: #5b6470; }
.kaal-terminal .info  { color: #4ade80; }
.kaal-terminal .warn  { color: #fbbf24; }
.kaal-terminal .error { color: #f87171; }
.kaal-terminal .stage { color: #60a5fa; }
.kaal-terminal .head  { color: #c4b5fd; font-weight: 700; }
.kaal-terminal .dim   { color: #6b7280; }

/* Status badges */
.badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 700; font-family: monospace;
    letter-spacing: 0.3px;
}
.b-idle    { background:#1f2937; color:#9ca3af; }
.b-run     { background:#1e3a8a; color:#93c5fd; animation: pulse 1.4s infinite; }
.b-pass    { background:#14532d; color:#86efac; }
.b-fail    { background:#7f1d1d; color:#fca5a5; }
.b-error   { background:#78350f; color:#fde68a; }
.b-skip    { background:#374151; color:#9ca3af; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }

/* Field cards */
.field-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 10px; margin-bottom: 6px;
}
.field-card {
    background: #0f1617; border: 1px solid #1e2829;
    border-radius: 6px; padding: 10px 12px;
    transition: border-color .15s;
}
.field-card.running { border-color: #3b82f6; }
.field-card.passed  { border-color: #22c55e; }
.field-card.failed  { border-color: #ef4444; }
.field-card.error   { border-color: #f59e0b; }
.field-card .fid    { color: #4ade80; font-family: monospace; font-weight: 700; }
.field-card .fname  { color: #e5e7eb; font-size: 12.5px; margin-top: 2px; }
.field-card .fmeta  { color: #9ca3af; font-size: 11px; margin-top: 6px; font-family: monospace; }

/* Metric cards */
.kmetric {
    background: #0f1617; border-left: 3px solid #4ade80;
    padding: 10px 14px; border-radius: 0 4px 4px 0;
}
.kmetric.warn { border-left-color: #fbbf24; }
.kmetric.err  { border-left-color: #f87171; }
.kmetric.crit { border-left-color: #f43f5e; }
.kmetric .val { font-size: 22px; font-weight: 700; color: #f9fafb; }
.kmetric .lbl { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; }

/* Sidebar tweaks */
[data-testid="stSidebar"] .stButton > button {
    width: 100%; background:#0f1617; border:1px solid #1f2937; color:#d1d5db;
    border-radius: 4px; padding: 8px 10px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background:#111a1b; border-color:#4ade80;
}

/* Status bar */
.status-bar {
    display:flex; gap:16px; padding:8px 14px; border-radius:4px;
    background:#0d1314; border:1px solid #1a2425; font-size:12px;
    font-family: monospace;
}
.status-bar .ok   { color:#4ade80; }
.status-bar .warn { color:#fbbf24; }
.status-bar .err  { color:#f87171; }
.status-bar .lbl  { color:#6b7280; margin-right:4px; }

/* Hide default footer */
footer { visibility: hidden; }
</style>
"""
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state — safe initialization
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults = {
        # Run state
        "run_status": "idle",           # idle | running | cancelling | done | error
        "run_id": None,
        "started_at": None,
        "finished_at": None,
        # Live UI
        "logs": [],                     # bounded list of (ts, level, text) tuples
        "progress": {
            "completed": 0, "total": 0, "failed": 0, "errored": 0,
            "current_field": None, "current_test_id": None, "current_op": None,
        },
        "test_rows": [],                # per-test summary rows
        "field_status": {f: "idle" for f in FIELD_DEFINITIONS},
        # Results
        "results": None,                # full report dict after run
        "cached_datasets": None,        # list of dataset metadata
        # Threading
        "_stop_event": None,            # threading.Event
        "_thread": None,
        "_event_queue": None,
        # Pending generation review
        "generated_draft": None,        # dict (awaiting user approval to save)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session_state()

# Module-level stop event (survives Streamlit reruns within a single process)
_STOP_EVENT: threading.Event = threading.Event()


# ---------------------------------------------------------------------------
# Logging + event dispatch
# ---------------------------------------------------------------------------
MAX_LOG_LINES = 800

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _append_log(level: str, message: str) -> None:
    """Append a log line to session state (bounded)."""
    entry = (_ts(), level, str(message))
    logs = st.session_state["logs"]
    logs.append(entry)
    if len(logs) > MAX_LOG_LINES:
        del logs[: len(logs) - MAX_LOG_LINES]

def _safe_log(level: str, message: str) -> None:
    try:
        _append_log(level, message)
    except Exception:
        pass  # Never let logging crash the UI


# ---------------------------------------------------------------------------
# Streaming benchmark executor — uses engine primitives directly
# ---------------------------------------------------------------------------
def streaming_benchmark(
    *,
    datasets: list[dict],
    target_provider: str,
    target_model: str,
    judge_model: str,
    target_system_prompt: str,
    stop_event: threading.Event,
    emit: Callable[[dict], None],
) -> dict:
    """
    Streaming executor. Uses app.py primitives (call_model, _target_messages,
    _judge_response, score_results) directly. Emits real events via `emit`.
    """
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    emit({"type": "run_start", "run_id": run_id, "started_at": started})

    all_results: list[dict] = []
    total_tests = sum(len(d["tests"]) for d in datasets)
    completed = 0
    emit({"type": "progress", "total": total_tests, "completed": 0, "failed": 0, "errored": 0})

    for dataset in datasets:
        if stop_event.is_set():
            break
        field_id = dataset["field_id"]
        field_name = dataset["field_name"]
        st.session_state["field_status"][field_id] = "running"
        emit({"type": "field_start", "field_id": field_id, "field_name": field_name})

        for test in dataset["tests"]:
            if stop_event.is_set():
                break
            test_with_field = dict(test, field_id=field_id)
            emit({
                "type": "test_start",
                "test_id": test["id"],
                "field_id": field_id,
                "category": test.get("category"),
                "severity": test.get("severity"),
            })

            result: dict = {
                "test_id": test["id"],
                "field_id": field_id,
                "category": test.get("category"),
                "severity": test.get("severity"),
                "prompt": test.get("prompt", ""),
                "expected_behavior": test.get("expected_behavior", ""),
                "pass_criteria": test.get("pass_criteria", ""),
            }

            try:
                emit({"type": "op", "stage": "target_request", "test_id": test["id"],
                      "provider": target_provider, "model": target_model})
                messages = _target_messages(test_with_field, target_system_prompt)
                response = call_model(
                    target_provider, messages,
                    model=target_model, temperature=0.0,
                )
                result["target_response"] = response

                emit({"type": "op", "stage": "judge_evaluation", "test_id": test["id"],
                      "provider": "gemini", "model": judge_model})
                evaluation = _judge_response(test_with_field, response, model=judge_model)
                evaluation["status"] = "evaluated"
                result.update(evaluation)

                emit({
                    "type": "test_end",
                    "test_id": test["id"],
                    "field_id": field_id,
                    "status": "pass" if evaluation.get("passed") else "fail",
                    "passed": evaluation.get("passed"),
                    "critical": evaluation.get("critical_failure"),
                })

            except (KAALConfigError, KAALProviderError) as exc:
                result.update({
                    "status": "error",
                    "passed": None,
                    "attack_succeeded": None,
                    "critical_failure": False,
                    "confidence": 0.0,
                    "reason": f"{type(exc).__name__}: {str(exc)[:600]}",
                    "evidence": "",
                    "failure_type": "provider_error",
                    "target_response": "",
                })
                emit({"type": "test_end", "test_id": test["id"], "field_id": field_id,
                      "status": "error", "error": str(exc)[:200]})
            except Exception as exc:
                result.update({
                    "status": "error",
                    "passed": None,
                    "attack_succeeded": None,
                    "critical_failure": False,
                    "confidence": 0.0,
                    "reason": f"{type(exc).__name__}: {str(exc)[:600]}",
                    "evidence": "",
                    "failure_type": "unexpected_error",
                    "target_response": "",
                })
                emit({"type": "test_end", "test_id": test["id"], "field_id": field_id,
                      "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})

            all_results.append(result)
            completed += 1
            is_fail = (result.get("status") == "fail")
            is_err = (result.get("status") == "error")
            emit({
                "type": "progress",
                "total": total_tests,
                "completed": completed,
                "failed": sum(1 for r in all_results if r.get("status") == "fail"),
                "errored": sum(1 for r in all_results if r.get("status") == "error"),
            })

        # Mark field outcome
        field_results = [r for r in all_results if r.get("field_id") == field_id]
        errs = sum(1 for r in field_results if r.get("status") == "error")
        fails = sum(1 for r in field_results if r.get("status") == "fail")
        passes = sum(1 for r in field_results if r.get("passed") is True)
        if stop_event.is_set():
            st.session_state["field_status"][field_id] = "skipped"
        elif errs == len(field_results):
            st.session_state["field_status"][field_id] = "error"
        elif fails > 0:
            st.session_state["field_status"][field_id] = "failed"
        elif passes > 0:
            st.session_state["field_status"][field_id] = "passed"
        else:
            st.session_state["field_status"][field_id] = "idle"
        emit({"type": "field_end", "field_id": field_id})

    # Mark any fields that were never touched as skipped (if cancelled)
    if stop_event.is_set():
        for f, s in st.session_state["field_status"].items():
            if s in ("idle", "running"):
                st.session_state["field_status"][f] = "skipped"

    finished = datetime.now(timezone.utc).isoformat()
    try:
        summary = score_results(all_results)
    except Exception as exc:
        summary = {"error": f"Scoring failed: {exc}", "total_tests": len(all_results)}

    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "datasets": [{
            "field_id": d["field_id"],
            "field_name": d["field_name"],
            "version": d.get("version", "1.0"),
            "source_file": d.get("source_file"),
            "test_count": len(d["tests"]),
        } for d in datasets],
        "target": {"provider": target_provider, "model": target_model},
        "judge": {"provider": "gemini", "model": judge_model},
        "summary": summary,
        "results": all_results,
        "cancelled": stop_event.is_set(),
    }
    emit({"type": "run_end", "report": report, "cancelled": stop_event.is_set()})
    return report


def _benchmark_worker(
    datasets, target_provider, target_model, judge_model, target_system_prompt
) -> None:
    """Background worker — emits events into the session queue."""
    q: queue.Queue = st.session_state["_event_queue"]
    stop_event: threading.Event = st.session_state["_stop_event"]

    def emit(evt: dict) -> None:
        try:
            q.put(evt)
        except Exception:
            pass

    try:
        emit({"type": "log", "level": "head", "msg": "KAAL BENCHMARK INITIALIZED"})
        emit({"type": "log", "level": "info", "msg": "Loading benchmark configuration..."})
        emit({"type": "log", "level": "info",
              "msg": f"Target: {target_provider.upper()} / {target_model or '(default)'}"})
        emit({"type": "log", "level": "info",
              "msg": f"Judge: GEMINI / {judge_model}"})

        # Validate API readiness
        emit({"type": "log", "level": "info", "msg": "Validating provider credentials..."})
        if target_provider == "qwen":
            _require_env("QWEN_API_KEY")
        elif target_provider == "gemini":
            _require_env("GEMINI_API_KEY")
        _require_env("GEMINI_API_KEY")  # judge always needs Gemini
        emit({"type": "log", "level": "info", "msg": "Credentials verified (redacted)."})

        for ds in datasets:
            emit({"type": "log", "level": "info",
                  "msg": f"Loading dataset: {ds['field_id']} ({len(ds['tests'])} tests)"})

        streaming_benchmark(
            datasets=datasets,
            target_provider=target_provider,
            target_model=target_model,
            judge_model=judge_model,
            target_system_prompt=target_system_prompt,
            stop_event=stop_event,
            emit=emit,
        )
    except KAALConfigError as exc:
        emit({"type": "log", "level": "error", "msg": f"Configuration error: {exc}"})
        emit({"type": "fatal_error", "msg": str(exc)})
    except Exception as exc:
        tb = traceback.format_exc()
        emit({"type": "log", "level": "error",
              "msg": f"Unexpected error: {type(exc).__name__}: {exc}"})
        emit({"type": "fatal_error", "msg": f"{type(exc).__name__}: {exc}\n{tb[-500:]}"})


def _start_benchmark(
    selected_fields: list[str],
    tests_per_field: int,
    target_provider: str,
    target_model: str,
    judge_model: str,
    target_system_prompt: str,
) -> None:
    """Validate, load datasets, and launch worker thread."""
    if st.session_state["run_status"] in ("running",):
        st.warning("A benchmark is already running.")
        return
    if not selected_fields:
        st.error("Select at least one benchmark field (F1–F8).")
        return

    _safe_log("info", "Preparing benchmark run...")

    # Load datasets
    datasets = []
    for field_id in selected_fields:
        try:
            ds = load_dataset(field_id, limit=tests_per_field)
            datasets.append(ds)
        except FileNotFoundError as exc:
            st.error(f"Dataset for {field_id} not found: {exc}")
            return
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Dataset for {field_id} is invalid: {exc}")
            return

    if not datasets:
        st.error("No datasets could be loaded.")
        return

    total = sum(len(d["tests"]) for d in datasets)
    if total > 200:
        st.warning(f"About to execute {total} tests. This may take significant time.")

    # Reset state
    st.session_state["run_status"] = "running"
    st.session_state["logs"] = []
    st.session_state["test_rows"] = []
    st.session_state["results"] = None
    st.session_state["field_status"] = {f: "idle" for f in FIELD_DEFINITIONS}
    st.session_state["progress"] = {
        "completed": 0, "total": total, "failed": 0, "errored": 0,
        "current_field": None, "current_test_id": None, "current_op": None,
    }
    st.session_state["started_at"] = datetime.now(timezone.utc).isoformat()
    st.session_state["finished_at"] = None
    st.session_state["run_id"] = None
    st.session_state["_stop_event"] = _STOP_EVENT
    _STOP_EVENT.clear()
    st.session_state["_event_queue"] = queue.Queue()

    thread = threading.Thread(
        target=_benchmark_worker,
        args=(datasets, target_provider, target_model, judge_model, target_system_prompt),
        daemon=True,
        name="kaal-benchmark-worker",
    )
    st.session_state["_thread"] = thread
    thread.start()


def _drain_events() -> None:
    """Pull pending events from the queue into session state."""
    q: Optional[queue.Queue] = st.session_state.get("_event_queue")
    if q is None:
        return
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            break
        _handle_event(evt)


def _handle_event(evt: dict) -> None:
    t = evt.get("type")
    if t == "log":
        level = evt.get("level", "info")
        msg = evt.get("msg", "")
        _append_log(level, msg)

    elif t == "run_start":
        st.session_state["run_id"] = evt.get("run_id")
        st.session_state["started_at"] = evt.get("started_at")
        _append_log("head", f"Run ID: {evt.get('run_id')}")

    elif t == "field_start":
        fid = evt.get("field_id")
        fname = evt.get("field_name")
        st.session_state["field_status"][fid] = "running"
        st.session_state["progress"]["current_field"] = fid
        _append_log("stage", f"[{fid}] === {fname} ===")

    elif t == "field_end":
        fid = evt.get("field_id")
        _append_log("stage", f"[{fid}] Field complete.")

    elif t == "test_start":
        tid = evt.get("test_id")
        fid = evt.get("field_id")
        cat = evt.get("category") or "-"
        sev = evt.get("severity") or "-"
        st.session_state["progress"]["current_test_id"] = tid
        st.session_state["test_rows"].append({
            "field": fid, "test_id": tid, "category": cat, "severity": sev,
            "stage": "starting", "target": "-", "judge": "-",
            "status": "running", "duration": "-",
            "_started_ts": time.time(),
        })
        _append_log("info", f"[{fid}] Test {tid} | Preparing test case")

    elif t == "op":
        stage = evt.get("stage")
        tid = evt.get("test_id")
        prov = evt.get("provider", "").upper()
        mod = evt.get("model", "")
        # Update last row for this test
        rows = st.session_state["test_rows"]
        if rows and rows[-1]["test_id"] == tid:
            rows[-1]["stage"] = stage
            if stage == "target_request":
                rows[-1]["target"] = f"{prov} / {mod}"
            elif stage == "judge_evaluation":
                rows[-1]["judge"] = f"{prov} / {mod}"
        st.session_state["progress"]["current_op"] = stage
        _append_log("info", f"{prov} | {mod} | {stage}")

    elif t == "test_end":
        tid = evt.get("test_id")
        status = evt.get("status")
        fid = evt.get("field_id")
        rows = st.session_state["test_rows"]
        for row in reversed(rows):
            if row["test_id"] == tid:
                row["status"] = status
                row["stage"] = "done"
                elapsed = time.time() - row.pop("_started_ts", time.time())
                row["duration"] = f"{elapsed:.2f}s"
                break
        level = "info" if status == "pass" else ("warn" if status == "fail" else "error")
        _append_log(level, f"[{fid}] Test {tid} | {status.upper()}")

    elif t == "progress":
        st.session_state["progress"].update({
            "completed": evt.get("completed", 0),
            "total": evt.get("total", st.session_state["progress"]["total"]),
            "failed": evt.get("failed", st.session_state["progress"]["failed"]),
            "errored": evt.get("errored", st.session_state["progress"]["errored"]),
        })

    elif t == "run_end":
        st.session_state["run_status"] = "done"
        st.session_state["finished_at"] = datetime.now(timezone.utc).isoformat()
        st.session_state["results"] = evt.get("report")
        _append_log("head", f"Run complete. Cancelled: {evt.get('cancelled')}")
        _append_log("info", f"Report saved to session. Run ID: {st.session_state['run_id']}")
        # Persist locally too
        try:
            path = save_report(evt["report"])
            _append_log("info", f"Report written to disk: {path}")
        except Exception as exc:
            _append_log("warn", f"Could not save report to disk: {exc}")

    elif t == "fatal_error":
        st.session_state["run_status"] = "error"
        st.session_state["finished_at"] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset discovery helper
# ---------------------------------------------------------------------------
def _get_available_datasets() -> list[dict]:
    cached = st.session_state.get("cached_datasets")
    if cached is not None:
        return cached
    items = list_datasets()
    st.session_state["cached_datasets"] = items
    return items


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
def render_sidebar() -> dict:
    st.sidebar.title("🛡️ KAAL Control")
    st.sidebar.caption("Phase I — Configuration")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Target Model")

    # Target provider
    provider_opts = ["qwen", "gemini"]
    target_provider = st.sidebar.selectbox(
        "Target provider",
        provider_opts,
        index=0 if DEFAULT_TARGET_PROVIDER == "qwen" else 1,
        key="cfg_provider",
    )

    # Default models per provider
    if target_provider == "qwen":
        target_model = st.sidebar.text_input(
            "Target model", value=DEFAULT_TARGET_MODEL, key="cfg_target_model"
        )
    else:
        target_model = st.sidebar.text_input(
            "Target model", value=GEMINI_JUDGE_MODEL, key="cfg_target_model"
        )

    st.sidebar.subheader("Judge")
    judge_model = st.sidebar.text_input(
        "Judge model (Gemini)", value=GEMINI_JUDGE_MODEL, key="cfg_judge_model"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Benchmark Fields")

    all_fields = list(FIELD_DEFINITIONS.keys())
    datasets = _get_available_datasets()
    available_fields = {d["field_id"] for d in datasets if not d.get("invalid")}
    if not available_fields:
        st.sidebar.warning("⚠️ No valid datasets found under datasets/. Add F1–F8 JSON files.")
    else:
        st.sidebar.caption(f"{len(available_fields)} of 8 fields available locally")

    selected_fields: list[str] = []
    for f in all_fields:
        is_avail = f in available_fields
        default = is_avail
        checked = st.sidebar.checkbox(
            f"{f} — {FIELD_DEFINITIONS[f]}",
            value=default,
            disabled=not is_avail,
            key=f"cfg_field_{f}",
        )
        if checked and is_avail:
            selected_fields.append(f)

    # Tests per field
    max_per_field = 50
    # Find smallest dataset size among selected
    if selected_fields:
        size_by_field = {d["field_id"]: d["test_count"] for d in datasets if not d.get("invalid")}
        min_size = min((size_by_field.get(f, 0) for f in selected_fields), default=0)
        max_per_field = max(1, min_size)
    tests_per_field = st.sidebar.slider(
        "Tests per field",
        min_value=1,
        max_value=max(1, max_per_field),
        value=min(10, max(1, max_per_field)),
        key="cfg_tests_per_field",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Prompt (optional)")
    target_system_prompt = st.sidebar.text_area(
        "Target system prompt",
        value="",
        height=80,
        key="cfg_sysprompt",
        help="Optional system prompt sent to the target model for every test.",
    )

    st.sidebar.markdown("---")

    # Action buttons
    btn_disabled = st.session_state["run_status"] in ("running",)
    run_col, stop_col = st.sidebar.columns(2)
    with run_col:
        run_clicked = st.button(
            "▶ Start Benchmark",
            type="primary",
            use_container_width=True,
            disabled=btn_disabled or not selected_fields,
        )
    with stop_col:
        stop_clicked = st.button(
            "■ Stop",
            use_container_width=True,
            disabled=st.session_state["run_status"] != "running",
        )

    if run_clicked:
        _start_benchmark(
            selected_fields=selected_fields,
            tests_per_field=tests_per_field,
            target_provider=target_provider,
            target_model=target_model,
            judge_model=judge_model,
            target_system_prompt=target_system_prompt,
        )
    if stop_clicked:
        _STOP_EVENT.set()
        st.session_state["run_status"] = "cancelling"
        _safe_log("warn", "Cancellation requested. Finishing current test...")

    # Dataset generation
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Generate New Dataset"):
        gen_field = st.selectbox("Field", list(FIELD_DEFINITIONS.keys()), key="gen_field")
        gen_count = st.slider("Cases", 5, 50, value=50, step=5, key="gen_count")
        if st.button("Generate with Qwen", key="gen_btn", use_container_width=True):
            with st.spinner(f"Generating {gen_field} dataset ({gen_count} cases)..."):
                try:
                    new_ds = generate_dataset(gen_field, cases_per_field=gen_count)
                    st.session_state["generated_draft"] = new_ds
                    st.session_state["cached_datasets"] = None  # invalidate cache
                    st.success(f"Generated {len(new_ds['tests'])} cases for {gen_field}. Review below.")
                except (KAALConfigError, KAALProviderError) as exc:
                    st.error(f"Generation failed: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected generation error: {exc}")

    return {
        "provider": target_provider,
        "target_model": target_model,
        "judge_model": judge_model,
        "selected_fields": selected_fields,
        "tests_per_field": tests_per_field,
        "target_system_prompt": target_system_prompt,
    }


# ---------------------------------------------------------------------------
# Header + system status
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        "<h1 style='margin-bottom:2px'>🛡️ KAAL <span style='color:#4ade80;font-size:18px'>Phase I</span></h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "LLM Security Benchmark — Eight-field adversarial assessment of target model boundaries."
    )

    datasets = _get_available_datasets()
    valid = [d for d in datasets if not d.get("invalid")]
    invalid = [d for d in datasets if d.get("invalid")]
    total_tests = sum(d["test_count"] for d in valid)

    # API readiness
    qwen_ok = bool(os.getenv("QWEN_API_KEY", "").strip())
    gemini_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())

    run_status = st.session_state["run_status"]
    status_label = {
        "idle": ("IDLE", "b-idle"),
        "running": ("RUNNING", "b-run"),
        "cancelling": ("CANCELLING", "b-run"),
        "done": ("COMPLETE", "b-pass"),
        "error": ("ERROR", "b-error"),
    }.get(run_status, ("IDLE", "b-idle"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='kmetric'>"
            f"<div class='lbl'>System Status</div>"
            f"<div class='val'><span class='badge {status_label[1]}'>{status_label[0]}</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        ds_count = len(valid)
        cls = "warn" if ds_count < 8 else ""
        st.markdown(
            f"<div class='kmetric {cls}'>"
            f"<div class='lbl'>Fields Available</div>"
            f"<div class='val'>{ds_count} / 8</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='kmetric'>"
            f"<div class='lbl'>Total Test Cases</div>"
            f"<div class='val'>{total_tests}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        api_html = (
            f"<span class='lbl'>QWEN</span> "
            f"<span class='{'ok' if qwen_ok else 'err'}'>{'●' if qwen_ok else '●'}</span>  "
            f"<span class='lbl'>GEMINI</span> "
            f"<span class='{'ok' if gemini_ok else 'err'}'>{'●' if gemini_ok else '●'}</span>"
        )
        st.markdown(
            f"<div class='kmetric'>"
            f"<div class='lbl'>API Keys Loaded</div>"
            f"<div class='val' style='font-size:14px;font-family:monospace'>{api_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if not qwen_ok or not gemini_ok:
        missing = []
        if not qwen_ok: missing.append("QWEN_API_KEY")
        if not gemini_ok: missing.append("GEMINI_API_KEY")
        st.error(
            f"⚠️ Missing environment variable(s): {', '.join(missing)}. "
            f"Copy `.env.example` to `.env` and fill in your API keys before running a benchmark."
        )
    if invalid:
        st.warning(f"⚠️ {len(invalid)} invalid dataset file(s) found under datasets/.")


# ---------------------------------------------------------------------------
# Benchmark overview (F1-F8 cards)
# ---------------------------------------------------------------------------
def render_field_overview() -> None:
    st.markdown("### Benchmark Fields")
    datasets = _get_available_datasets()
    ds_map = {d["field_id"]: d for d in datasets if not d.get("invalid")}
    field_status = st.session_state["field_status"]

    html_cards = ["<div class='field-grid'>"]
    for fid in FIELD_DEFINITIONS:
        fname = FIELD_DEFINITIONS[fid]
        status = field_status.get(fid, "idle")
        cls_map = {"running": "running", "passed": "passed", "failed": "failed",
                   "error": "error", "skipped": "", "idle": ""}
        cls = cls_map.get(status, "")
        badge_cls = {"idle": "b-idle", "running": "b-run", "passed": "b-pass",
                     "failed": "b-fail", "error": "b-error", "skipped": "b-skip"}[status]
        info = ds_map.get(fid)
        count_txt = f"{info['test_count']} tests" if info else "no dataset"
        version_txt = f"v{info['version']}" if info else "—"
        html_cards.append(
            f"<div class='field-card {cls}'>"
            f"<div class='fid'>{fid}</div>"
            f"<div class='fname'>{html.escape(fname)}</div>"
            f"<div class='fmeta'>Tests: {count_txt} · {version_txt}</div>"
            f"<div style='margin-top:6px'><span class='badge {badge_cls}'>{status.upper()}</span></div>"
            f"</div>"
        )
    html_cards.append("</div>")
    st.markdown("".join(html_cards), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live execution panel (progress + terminal + table)
# ---------------------------------------------------------------------------
def render_live_panel() -> None:
    st.markdown("### Live Execution")

    prog = st.session_state["progress"]
    total = prog["total"]
    completed = prog["completed"]
    failed = prog["failed"]
    errored = prog["errored"]
    pct = int(100 * completed / total) if total else 0

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.progress(pct / 100 if total else 0)
    with c2:
        st.markdown(f"<div class='kmetric'><div class='lbl'>Progress</div><div class='val'>{pct}%</div></div>", unsafe_allow_html=True)
    with c3:
        elapsed_txt = "—"
        if st.session_state["started_at"]:
            try:
                start_dt = datetime.fromisoformat(st.session_state["started_at"])
                delta = (datetime.now(timezone.utc) - start_dt).total_seconds()
                elapsed_txt = f"{int(delta//60)}m {int(delta%60)}s"
            except Exception:
                elapsed_txt = "—"
        st.markdown(f"<div class='kmetric'><div class='lbl'>Elapsed</div><div class='val' style='font-size:16px'>{elapsed_txt}</div></div>", unsafe_allow_html=True)

    meta_cols = st.columns(5)
    meta_cols[0].metric("Completed", completed)
    meta_cols[1].metric("Remaining", max(0, total - completed))
    meta_cols[2].metric("Failed (security)", failed)
    meta_cols[3].metric("Errors (infra)", errored)
    meta_cols[4].metric("Current", prog.get("current_test_id") or "—")

    # Terminal + table side-by-side
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### 🖥️ Event Terminal")
        terminal = st.empty()
        lines_html = []
        for ts, level, msg in st.session_state["logs"]:
            cls = {"info": "info", "warn": "warn", "error": "error",
                   "stage": "stage", "head": "head"}.get(level, "dim")
            safe_msg = html.escape(str(msg)).replace("\n", "<br>")
            lines_html.append(
                f"<span class='ts'>[{html.escape(ts)}]</span> "
                f"<span class='{cls}'>{safe_msg}</span>"
            )
        terminal.markdown(
            f"<div class='kaal-terminal'>{''.join(lines_html) or '<span class=\"dim\">Awaiting events...</span>'}</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### 📋 Per-Test Activity")
        rows = st.session_state["test_rows"]
        if rows:
            # Build DataFrame from last N rows
            display_rows = rows[-50:]
            df = pd.DataFrame([{
                "Field": r["field"],
                "Test ID": r["test_id"],
                "Category": r["category"],
                "Stage": r["stage"],
                "Target": r["target"],
                "Judge": r["judge"],
                "Status": r["status"],
                "Duration": r["duration"],
            } for r in display_rows])
            st.dataframe(df, use_container_width=True, height=380, hide_index=True)
        else:
            st.info("No tests executed yet. Configure fields in the sidebar and click **Start Benchmark**.")


# ---------------------------------------------------------------------------
# Results + analytics
# ---------------------------------------------------------------------------
def render_results() -> None:
    report: Optional[dict] = st.session_state.get("results")
    if report is None:
        st.info("Run a benchmark to see security evaluation results here.")
        return

    st.markdown("### Results & Security Evaluation")

    summary = report.get("summary", {})
    if "error" in summary:
        st.error(f"Scoring error: {summary['error']}")
        return

    total = summary.get("total_tests", 0)
    evaluated = summary.get("evaluated_tests", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    errors = summary.get("execution_or_judge_errors", 0)
    crit = summary.get("critical_failure_count", 0)
    pass_rate = summary.get("pass_rate_pct")
    avg_conf = summary.get("average_judge_confidence")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f"<div class='kmetric'><div class='lbl'>Total Tests</div><div class='val'>{total}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kmetric'><div class='lbl'>Evaluated</div><div class='val'>{evaluated}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kmetric'><div class='lbl'>Passed</div><div class='val'>{passed}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kmetric warn'><div class='lbl'>Failed (security)</div><div class='val'>{failed}</div></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='kmetric crit'><div class='lbl'>Critical</div><div class='val'>{crit}</div></div>", unsafe_allow_html=True)
    k6.markdown(f"<div class='kmetric err'><div class='lbl'>Infra Errors</div><div class='val'>{errors}</div></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Pass Rate", f"{pass_rate:.1f}%" if pass_rate is not None else "n/a")
    m2.metric("Avg Judge Confidence", f"{avg_conf:.2f}" if avg_conf is not None else "n/a")
    m3.metric("Attack Success Rate",
              f"{100 - pass_rate:.1f}%" if pass_rate is not None and evaluated > 0 else "n/a",
              help="Of evaluated tests, percentage where target did NOT meet stated security criteria.")

    # Per-field chart + table
    by_field = summary.get("by_field", {})
    if by_field:
        st.markdown("#### Per-Field Breakdown")
        field_rows = []
        for fid, data in sorted(by_field.items()):
            field_rows.append({
                "Field": f"{fid} — {data.get('field_name', '')}",
                "Total": data.get("total", 0),
                "Evaluated": data.get("evaluated", 0),
                "Passed": data.get("passed", 0),
                "Failed": data.get("failed", 0),
                "Errors": data.get("errors", 0),
                "Pass %": data.get("pass_rate_pct"),
                "Critical": data.get("critical_failures", 0),
            })
        chart_df = pd.DataFrame(field_rows).set_index("Field")[["Passed", "Failed", "Errors"]]
        st.bar_chart(chart_df)
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True, hide_index=True)

    # Critical failures list
    critical_list = summary.get("critical_failures", [])
    if critical_list:
        st.markdown("#### 🚨 Critical Security Failures")
        for item in critical_list:
            with st.expander(f"⚠️ {item.get('field_id')} / {item.get('test_id')} — severity: {item.get('severity')}"):
                st.write(f"**Reason:** {item.get('reason')}")
                st.write(f"**Evidence:** {item.get('evidence')}")

    # Test-level inspector
    st.markdown("#### 🔍 Individual Test Inspector")
    results_list = report.get("results", [])
    if results_list:
        test_labels = [f"{r['field_id']}/{r['test_id']}" for r in results_list]
        sel = st.selectbox("Select test", test_labels, key="test_inspector")
        idx = test_labels.index(sel)
        r = results_list[idx]

        status_cls = {"pass": "b-pass", "fail": "b-fail", "error": "b-error"}.get(
            "pass" if r.get("passed") is True else ("fail" if r.get("passed") is False else "error"),
            "b-idle",
        )
        st.markdown(
            f"<span class='badge {status_cls}'>{'PASS' if r.get('passed') is True else ('FAIL' if r.get('passed') is False else 'ERROR')}</span>",
            unsafe_allow_html=True,
        )

        i1, i2, i3 = st.columns(3)
        i1.markdown(f"**Field:** {r.get('field_id')}")
        i2.markdown(f"**Category:** {r.get('category', '—')}")
        i3.markdown(f"**Severity:** {r.get('severity', '—')}")

        with st.expander("📝 Test Prompt"):
            st.code(r.get("prompt", "(empty)"))
        with st.expander("🎯 Expected Behavior & Pass Criteria"):
            st.write(f"**Expected:** {r.get('expected_behavior', '—')}")
            st.write(f"**Pass criteria:** {r.get('pass_criteria', '—')}")
        with st.expander("🤖 Target Response (untrusted)"):
            st.text(r.get("target_response", "(empty)"))
        with st.expander("⚖️ Judge Evaluation"):
            j1, j2, j3 = st.columns(3)
            j1.metric("Passed", r.get("passed"))
            j2.metric("Attack Succeeded", r.get("attack_succeeded"))
            j3.metric("Critical", r.get("critical_failure"))
            st.write(f"**Confidence:** {r.get('confidence')}")
            st.write(f"**Reason:** {r.get('reason')}")
            st.write(f"**Evidence:** {r.get('evidence')}")
            st.write(f"**Failure type:** {r.get('failure_type')}")
        if r.get("status") == "error":
            with st.expander("❌ Error Details"):
                st.write(r.get("reason"))

    # Exports
    st.markdown("#### 📥 Export Report")
    try:
        json_bytes = json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        st.download_button(
            "⬇️ Download Full JSON Report",
            data=json_bytes,
            file_name=f"kaal_{report.get('run_id','run')}.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"Could not serialize JSON report: {exc}")

    try:
        csv_text = results_to_csv(report)
        st.download_button(
            "⬇️ Download CSV Results",
            data=csv_text.encode("utf-8"),
            file_name=f"kaal_{report.get('run_id','run')}.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Could not generate CSV: {exc}")

    try:
        # Concise summary report
        summary_txt = []
        summary_txt.append(f"KAAL BENCHMARK SUMMARY")
        summary_txt.append(f"Run ID: {report.get('run_id')}")
        summary_txt.append(f"Started: {report.get('started_at')}")
        summary_txt.append(f"Finished: {report.get('finished_at')}")
        summary_txt.append(f"Target: {report.get('target')}")
        summary_txt.append(f"Judge: {report.get('judge')}")
        summary_txt.append(f"Datasets: {report.get('datasets')}")
        summary_txt.append("")
        summary_txt.append("AGGREGATE METRICS:")
        for k, v in summary.items():
            if k not in ("by_field", "by_category", "critical_failures"):
                summary_txt.append(f"  {k}: {v}")
        st.download_button(
            "⬇️ Download Summary TXT",
            data="\n".join(summary_txt).encode("utf-8"),
            file_name=f"kaal_{report.get('run_id','run')}_summary.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"Could not generate summary: {exc}")


# ---------------------------------------------------------------------------
# Draft dataset approval (for generation)
# ---------------------------------------------------------------------------
def render_generation_draft() -> None:
    draft: Optional[dict] = st.session_state.get("generated_draft")
    if draft is None:
        return
    st.markdown("---")
    st.markdown("### 🧪 Generated Dataset — Awaiting Approval")
    st.info(
        f"Qwen generated **{len(draft.get('tests', []))}** cases for "
        f"**{draft.get('field_id')}** — {draft.get('field_name')}. "
        f"Review before saving to datasets/."
    )

    tests = draft.get("tests", [])
    for i, t in enumerate(tests[:10]):  # show first 10 as preview
        with st.expander(f"{t.get('id')} — {t.get('category')} / {t.get('severity')}"):
            st.code(t.get("prompt", ""))
            st.write(f"**Expected:** {t.get('expected_behavior', '')}")
            st.write(f"**Pass criteria:** {t.get('pass_criteria', '')}")
    if len(tests) > 10:
        st.caption(f"+ {len(tests) - 10} more cases (saved in full on approval)")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve & Save to datasets/", type="primary"):
            try:
                dest = DATASET_DIR
                dest.mkdir(parents=True, exist_ok=True)
                filename = f"{draft['field_id']}_{_slug(draft['field_name'])}.json"
                path = dest / filename
                path.write_text(json.dumps(draft, indent=2, ensure_ascii=False), encoding="utf-8")
                st.success(f"Saved to {path}")
                st.session_state["generated_draft"] = None
                st.session_state["cached_datasets"] = None
            except Exception as exc:
                st.error(f"Failed to save: {exc}")
    with c2:
        if st.button("❌ Discard"):
            st.session_state["generated_draft"] = None


# ---------------------------------------------------------------------------
# Main render orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    # Drain background events first
    _drain_events()

    cfg = render_sidebar()

    render_header()
    render_field_overview()
    render_live_panel()
    render_generation_draft()
    render_results()

    # Auto-rerun while benchmark is running, so UI updates live
    if st.session_state["run_status"] in ("running", "cancelling"):
        time.sleep(0.35)
        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        st.error(f"UI error: {type(exc).__name__}: {exc}")
        st.code(traceback.format_exc())
