"""Streamlit dashboard to visualize scraping statistics."""

import json
import os
from pathlib import Path

import streamlit as st
import psutil
import requests

PROGRESS_FILE = Path("logs/progress.json")
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
PROM_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
LOG_FILE = Path(os.environ.get("LOG_FILE", "logs/scraper.log"))


def load_progress():
    try:
        resp = requests.get(f"{API_BASE}/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        if PROGRESS_FILE.exists():
            try:
                with PROGRESS_FILE.open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def load_dataset_info():
    try:
        resp = requests.get(f"{API_BASE}/dataset/summary", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def tail_logs(lines: int = 20) -> str:
    if LOG_FILE.exists():
        data = LOG_FILE.read_text(encoding="utf-8").splitlines()
        return "\n".join(data[-lines:])
    return ""


def enqueue_tasks(tasks: list[dict]) -> bool:
    try:
        resp = requests.post(f"{API_BASE}/queue/add", json=tasks, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def clear_tasks() -> bool:
    try:
        resp = requests.post(f"{API_BASE}/queue/clear", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def load_metrics():
    metrics = {
        "success": 0,
        "error": 0,
        "block": 0,
        "pages": 0,
        "failures": 0,
        "avg_time": 0,
        "retries": 0,
        "avg_session": 0,
        "completeness": 0,
        "diversity": 0,
        "lang_cov": 0,
        "domain_cov": 0,
        "bias": 0,
    }
    try:
        for name, key in {
            "success": "scrape_success_total",
            "error": "scrape_error_total",
            "block": "scrape_block_total",
            "pages": "pages_scraped_total",
            "failures": "requests_failed_total",
            "retries": "request_retries_total",
            "completeness": "dataset_completeness_ratio",
            "diversity": "dataset_topic_diversity",
            "lang_cov": "dataset_language_coverage",
            "domain_cov": "dataset_domain_coverage",
            "bias": "dataset_bias_detected",
        }.items():
            resp = requests.get(
                f"{PROM_URL}/api/v1/query",
                params={"query": key},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("data", {}).get("result"):
                metrics[name] = float(data["data"]["result"][0]["value"][1])

        # Calculate average processing time
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "page_processing_seconds_sum"},
            timeout=5,
        )
        resp.raise_for_status()
        sum_data = resp.json()
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "page_processing_seconds_count"},
            timeout=5,
        )
        resp.raise_for_status()
        count_data = resp.json()
        if sum_data.get("data", {}).get("result") and count_data.get("data", {}).get(
            "result"
        ):
            total = float(sum_data["data"]["result"][0]["value"][1])
            count = float(count_data["data"]["result"][0]["value"][1])
            if count:
                metrics["avg_time"] = total / count

        # Average session duration
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "scrape_session_seconds_sum"},
            timeout=5,
        )
        resp.raise_for_status()
        sess_sum = resp.json()
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": "scrape_session_seconds_count"},
            timeout=5,
        )
        resp.raise_for_status()
        sess_count = resp.json()
        if sess_sum.get("data", {}).get("result") and sess_count.get("data", {}).get(
            "result"
        ):
            total = float(sess_sum["data"]["result"][0]["value"][1])
            count = float(sess_count["data"]["result"][0]["value"][1])
            if count:
                metrics["avg_session"] = total / count
    except Exception:
        pass
    return metrics


def language_coverage(langs: list[str]) -> float:
    """Return ratio of languages to configured ones."""
    try:
        from scraper_wiki import Config

        return len(set(langs)) / len(Config.LANGUAGES) if Config.LANGUAGES else 0.0
    except Exception:
        return 0.0


def domain_coverage(categories: list[str]) -> float:
    """Return ratio of categories present to configured categories."""
    try:
        from scraper_wiki import Config

        return (
            len(set(categories)) / len(Config.CATEGORIES) if Config.CATEGORIES else 0.0
        )
    except Exception:
        return 0.0


def detect_bias(langs: list[str], categories: list[str]) -> list[str]:
    """Return list of bias alert messages."""
    alerts: list[str] = []
    if language_coverage(langs) <= 0.5:
        alerts.append("Low language coverage")
    if domain_coverage(categories) <= 0.5:
        alerts.append("Low domain coverage")
    return alerts


def main():
    st.title("Scraper Progress Dashboard")

    st.sidebar.header("Queue Management")
    task_input = st.sidebar.text_area("URLs or titles", height=100)
    if st.sidebar.button("Start/Enqueue"):
        tasks = []
        for line in task_input.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("http"):
                tasks.append({"url": line})
            else:
                tasks.append({"title": line, "lang": "en"})
        if tasks and enqueue_tasks(tasks):
            st.sidebar.success("Tasks queued")
        elif tasks:
            st.sidebar.error("Failed to enqueue tasks")
    if st.sidebar.button("Stop/Clear Queue"):
        if clear_tasks():
            st.sidebar.success("Queue cleared")
        else:
            st.sidebar.error("Failed to clear queue")

    progress = load_progress()
    metrics = load_metrics()
    pages_processed = progress.get("pages_processed", 0)
    clusters = progress.get("clusters", [])
    topics = progress.get("topics", [])
    languages = progress.get("languages", [])
    categories = progress.get("categories", [])

    lang_cov = language_coverage(languages)
    dom_cov = domain_coverage(categories)
    bias_alerts = detect_bias(languages, categories)

    st.metric("Pages processed", pages_processed)
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    st.metric("CPU usage", f"{cpu}%")
    st.metric("RAM usage", f"{ram}%")
    st.metric("Avg processing time (s)", round(metrics["avg_time"], 2))
    st.metric("Avg session time (s)", round(metrics["avg_session"], 2))

    st.metric("Scrape success", metrics["success"])
    st.metric("Scrape errors", metrics["error"])
    st.metric("Scrape blocks", metrics["block"])
    st.metric("Request retries", metrics["retries"])
    st.metric("Dataset completeness", round(metrics["completeness"], 2))
    st.metric("Topic diversity", round(metrics["diversity"], 2))
    st.metric("Language coverage", round(lang_cov, 2))
    st.metric("Domain coverage", round(dom_cov, 2))
    st.metric("Bias detected", metrics["bias"])
    st.metric("Duplicates removed", progress.get("duplicates_removed", 0))
    st.metric("Invalid records", progress.get("invalid_records", 0))

    st.subheader("Clusters")
    st.write(clusters)

    st.subheader("Topics")
    st.write(topics)

    st.subheader("Languages")
    st.write(languages)
    if bias_alerts:
        st.warning(" | ".join(bias_alerts))

    info = load_dataset_info()
    if info:
        st.subheader("Dataset Summary")
        st.json(info)

    st.subheader("Recent Logs")
    st.text(tail_logs(20))


if __name__ == "__main__":
    main()
