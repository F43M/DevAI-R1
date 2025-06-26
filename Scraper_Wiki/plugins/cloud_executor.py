"""Execute plugin steps using AWS Lambda or Google Cloud Functions."""

from __future__ import annotations

import json
from typing import Any, Dict, List
import logging

from scraper_wiki import log_error

logger = logging.getLogger(__name__)
import requests


def invoke_lambda(
    function_name: str, payload: Dict[str, Any], region: str = "us-east-1"
) -> Dict[str, Any]:
    """Invoke an AWS Lambda function with ``payload``.

    Parameters
    ----------
    function_name:
        Name of the Lambda function.
    payload:
        Data sent as the invocation payload.
    region:
        AWS region where the function is deployed.

    Returns
    -------
    dict
        JSON response from the Lambda function.
    """
    try:  # pragma: no cover - optional dependency
        import boto3  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("boto3 required for AWS Lambda") from exc
    client = boto3.client("lambda", region_name=region)
    resp = client.invoke(
        FunctionName=function_name, Payload=json.dumps(payload).encode()
    )
    data = resp.get("Payload")
    if data:
        return json.loads(data.read())
    return {}


def invoke_cloud_function(function_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call a Google Cloud Function via HTTP."""
    resp = requests.post(function_url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_plugin_lambda(
    plugin_name: str,
    items: List[Dict[str, Any]],
    function_name: str,
    region: str = "us-east-1",
) -> List[Dict[str, Any]]:
    """Execute ``parse_item`` for ``items`` remotely via AWS Lambda."""
    payload = {"plugin": plugin_name, "items": items}
    result = invoke_lambda(function_name, payload, region=region)
    return result.get("records", [])


def run_plugin_gcf(
    plugin_name: str, items: List[Dict[str, Any]], function_url: str
) -> List[Dict[str, Any]]:
    """Execute ``parse_item`` for ``items`` on Google Cloud Functions."""
    payload = {"plugin": plugin_name, "items": items}
    result = invoke_cloud_function(function_url, payload)
    return result.get("records", [])


__all__ = [
    "invoke_lambda",
    "invoke_cloud_function",
    "run_plugin_lambda",
    "run_plugin_gcf",
]
