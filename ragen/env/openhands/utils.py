import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def safe_format_template(template: str, values: Dict[str, Any]) -> str:
    """Format templates without KeyError on missing placeholders."""
    return template.format_map(_SafeDict(values))


def to_plain_dict(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-compatible values."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): to_plain_dict(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_plain_dict(v) for v in obj]

    if is_dataclass(obj):
        return to_plain_dict(asdict(obj))

    if hasattr(obj, "model_dump"):
        try:
            return to_plain_dict(obj.model_dump(mode="json"))
        except TypeError:
            return to_plain_dict(obj.model_dump())

    if hasattr(obj, "dict"):
        try:
            return to_plain_dict(obj.dict())
        except Exception:
            pass

    return str(obj)


def event_to_jsonable(event: Any) -> Dict[str, Any]:
    """Serialize one event into a JSONable dict."""
    payload = to_plain_dict(event)
    if isinstance(payload, dict):
        payload.setdefault("event_type", type(event).__name__)
        return payload
    return {"event_type": type(event).__name__, "payload": payload}


def _dig_number(data: Any, keys: list[str]) -> float:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return 0.0
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return 0.0


def _extract_single_usage(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract usage totals from a single metrics dict."""
    prompt_tokens = _dig_number(metrics, ["accumulated_token_usage", "prompt_tokens"])
    completion_tokens = _dig_number(metrics, ["accumulated_token_usage", "completion_tokens"])

    if prompt_tokens == 0.0 and completion_tokens == 0.0:
        token_usages = metrics.get("token_usages", [])
        if isinstance(token_usages, list):
            for item in token_usages:
                if not isinstance(item, dict):
                    continue
                try:
                    prompt_tokens += float(item.get("prompt_tokens", 0.0) or 0.0)
                    completion_tokens += float(item.get("completion_tokens", 0.0) or 0.0)
                except Exception:
                    continue

    total_tokens = _dig_number(metrics, ["accumulated_token_usage", "total_tokens"])
    if total_tokens <= 0.0:
        total_tokens = prompt_tokens + completion_tokens

    cost_usd = _dig_number(metrics, ["accumulated_cost"])
    if cost_usd <= 0.0:
        for key in ("total_cost", "cost", "spend"):
            try:
                cost_usd = max(cost_usd, float(metrics.get(key, 0.0) or 0.0))
            except Exception:
                continue

    return {
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
        "total_tokens": float(total_tokens),
        "cost_usd": float(cost_usd),
    }


def extract_usage_breakdown(metrics_obj: Any) -> Dict[str, Any]:
    """
    Extract totals and per-usage-id metrics when available.

    Returns:
      {
        "totals": {...},
        "by_usage_id": {"agent": {...}, ...}
      }
    """
    metrics = to_plain_dict(metrics_obj)
    if not isinstance(metrics, dict):
        metrics = {}

    root_totals = _extract_single_usage(metrics)
    usage_to_metrics = metrics.get("usage_to_metrics", {})
    by_usage_id: Dict[str, Dict[str, float]] = {}

    if isinstance(usage_to_metrics, dict):
        for usage_id, usage_metrics in usage_to_metrics.items():
            if isinstance(usage_metrics, dict):
                by_usage_id[str(usage_id)] = _extract_single_usage(usage_metrics)

    if by_usage_id:
        agg = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
        }
        for usage in by_usage_id.values():
            for key in agg:
                agg[key] += float(usage.get(key, 0.0) or 0.0)
        totals = agg
    else:
        totals = root_totals

    return {
        "totals": totals,
        "by_usage_id": by_usage_id,
    }


def extract_usage_totals(metrics_obj: Any) -> Dict[str, float]:
    """Extract cumulative token/cost totals from OpenHands metric objects/dicts."""
    return extract_usage_breakdown(metrics_obj)["totals"]


def compute_usage_delta(previous: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    """Compute non-negative deltas between two cumulative usage snapshots."""
    delta: Dict[str, float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"):
        prev_val = float(previous.get(key, 0.0) or 0.0)
        cur_val = float(current.get(key, 0.0) or 0.0)
        delta[key] = float(max(0.0, cur_val - prev_val))
    return delta


def detect_finish_action(events: list[Any]) -> bool:
    """Best-effort detection of finish action from event objects."""
    for event in reversed(events):
        event_type = type(event).__name__.lower()
        if "action" not in event_type:
            continue

        action_obj = getattr(event, "action", None)
        action_type = type(action_obj).__name__.lower() if action_obj is not None else ""
        if "finishaction" in action_type or action_type in {"finish", "finish_action"}:
            return True

        event_payload = event_to_jsonable(event)
        candidates = [
            event_payload.get("action"),
            event_payload.get("action_type"),
            event_payload.get("name"),
            event_payload.get("tool"),
            event_payload.get("command"),
            event_payload.get("action_name"),
        ]
        for candidate in candidates:
            candidate_str = str(candidate).strip().lower()
            if candidate_str in {"finish", "finish_action", "agent_finish_action"}:
                return True
            if candidate_str.endswith("finishaction"):
                return True

        payload_action = event_payload.get("action")
        if isinstance(payload_action, dict):
            nested_name = str(
                payload_action.get("name")
                or payload_action.get("action_type")
                or payload_action.get("type")
                or ""
            ).lower()
            if nested_name in {"finish", "finish_action", "agent_finish_action"} or nested_name.endswith("finishaction"):
                return True

        if "finishaction" in event_type:
            return True

    return False


def summarize_events(events: list[Any], limit: int = 5) -> str:
    """Short textual summary for a sequence of events."""
    if not events:
        return "No new OpenHands events in this turn."

    lines = []
    for event in events[-limit:]:
        payload = event_to_jsonable(event)
        event_type = payload.get("event_type", type(event).__name__)
        message = payload.get("message") or payload.get("content") or ""
        message = str(message).replace("\n", " ").strip()
        if message:
            message = message[:160]
            lines.append(f"- {event_type}: {message}")
        else:
            lines.append(f"- {event_type}")

    if len(events) > limit:
        lines.insert(0, f"({len(events)} events; showing last {limit})")

    return "\n".join(lines)


def extract_event_diagnostics(events: list[Any]) -> Dict[str, str]:
    """Extract coarse tool/error diagnostics from event payloads."""
    diagnostics = {
        "last_tool_status": "unknown",
        "last_error": "none",
    }

    for event in reversed(events):
        payload = event_to_jsonable(event)
        lowered = json.dumps(payload).lower()

        if "error" in lowered or "exception" in lowered or "traceback" in lowered:
            diagnostics["last_tool_status"] = "error"
            diagnostics["last_error"] = str(payload)[:240]
            return diagnostics

        if "exit_code" in payload:
            try:
                exit_code = int(payload["exit_code"])
                if exit_code == 0:
                    diagnostics["last_tool_status"] = "ok"
                else:
                    diagnostics["last_tool_status"] = "error"
                    diagnostics["last_error"] = str(payload)[:240]
                return diagnostics
            except Exception:
                pass

        if "tool" in lowered and diagnostics["last_tool_status"] == "unknown":
            diagnostics["last_tool_status"] = "ok"

    return diagnostics
