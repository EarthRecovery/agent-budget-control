from ragen.env.openhands.utils import (
    compute_usage_delta,
    detect_finish_action,
    extract_usage_breakdown,
    extract_usage_totals,
    safe_format_template,
)


class _PlainAction:
    pass


class _FinishAction:
    pass


class _ActionEvent:
    def __init__(self, action=None, payload=None):
        self.action = action
        self._payload = payload or {}

    def model_dump(self, mode="json"):
        return dict(self._payload)


class _OtherEvent:
    pass


def test_safe_format_template_missing_key_is_empty():
    rendered = safe_format_template("Hello {name} {missing}", {"name": "OpenHands"})
    assert rendered == "Hello OpenHands "


def test_extract_usage_totals_from_accumulated_fields():
    metrics = {
        "accumulated_token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
        },
        "accumulated_cost": 0.42,
    }
    totals = extract_usage_totals(metrics)
    assert totals["prompt_tokens"] == 100.0
    assert totals["completion_tokens"] == 40.0
    assert totals["total_tokens"] == 140.0
    assert totals["cost_usd"] == 0.42


def test_extract_usage_breakdown_sums_usage_to_metrics():
    metrics = {
        "usage_to_metrics": {
            "agent": {
                "accumulated_token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "accumulated_cost": 0.03,
            },
            "condenser": {
                "accumulated_token_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "total_tokens": 5,
                },
                "accumulated_cost": 0.01,
            },
        }
    }
    breakdown = extract_usage_breakdown(metrics)
    totals = breakdown["totals"]
    by_usage = breakdown["by_usage_id"]

    assert by_usage["agent"]["total_tokens"] == 15.0
    assert by_usage["condenser"]["total_tokens"] == 5.0
    assert totals["prompt_tokens"] == 14.0
    assert totals["completion_tokens"] == 6.0
    assert totals["total_tokens"] == 20.0
    assert totals["cost_usd"] == 0.04


def test_compute_usage_delta_non_negative():
    prev = {
        "prompt_tokens": 100.0,
        "completion_tokens": 50.0,
        "total_tokens": 150.0,
        "cost_usd": 0.3,
    }
    cur = {
        "prompt_tokens": 120.0,
        "completion_tokens": 70.0,
        "total_tokens": 190.0,
        "cost_usd": 0.5,
    }
    delta = compute_usage_delta(prev, cur)
    assert delta == {
        "prompt_tokens": 20.0,
        "completion_tokens": 20.0,
        "total_tokens": 40.0,
        "cost_usd": 0.2,
    }

    # Backwards values should clamp to zero.
    neg_delta = compute_usage_delta(cur, prev)
    assert neg_delta == {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
    }


def test_detect_finish_action_via_action_type():
    events = [_ActionEvent(action=_FinishAction())]
    assert detect_finish_action(events) is True


def test_detect_finish_action_via_payload_name():
    events = [_ActionEvent(action=_PlainAction(), payload={"action_type": "finish"})]
    assert detect_finish_action(events) is True


def test_detect_finish_action_ignores_non_action_events():
    events = [_OtherEvent()]
    assert detect_finish_action(events) is False

