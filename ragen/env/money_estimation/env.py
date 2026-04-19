import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.token_estimation.env import (
    _extract_tag_text,
    _normalize_message,
    _resolve_turn_assistant_content,
    _resolve_turn_user_content,
    _safe_int,
)
from ragen.utils import all_seed

from .config import MoneyEstimationEnvConfig


_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"


@dataclass
class MoneyEstimationSample:
    sample_id: str
    rollout_index: int
    env_id: Optional[int]
    absolute_env_id: Optional[int]
    turn_idx: int
    total_turns: int
    source_system: str
    input_messages: List[Dict[str, str]]
    target_output: str
    completed_turns: int
    completed_day: int
    completed_weeks: int
    relative_progress: float
    current_cash_usd: int
    target_cash_usd: int
    current_time_weeks: int
    current_warehouse_item_weeks: int
    current_cost_usd: int
    budget_time_weeks: int
    budget_warehouse_item_weeks: int
    budget_cost_usd: int
    consumption_history: List[Dict[str, int]]
    actual_can_finish: bool
    actual_remaining_turn: int
    actual_remaining_time_weeks: Optional[int]
    actual_remaining_warehouse_item_weeks: Optional[int]
    actual_remaining_cost_usd: Optional[int]
    actual_total_time_weeks: int
    actual_total_warehouse_item_weeks: int
    actual_total_cost_usd: int
    rollout_success: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_to_int(value: Any) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(round(float(numeric)))


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _extract_day(text: Any) -> Optional[int]:
    if text is None:
        return None
    match = re.search(r"Day\s+(\d+)", str(text))
    if not match:
        return None
    return int(match.group(1))


def _extract_cash(text: Any) -> Optional[int]:
    if text is None:
        return None
    match = re.search(r"Cash:\s*\$([0-9,]+(?:\.\d+)?)", str(text))
    if not match:
        return None
    numeric = match.group(1).replace(",", "")
    return _round_to_int(numeric)


def _resolve_financial_metric(turn: Dict[str, Any], key: str) -> Optional[int]:
    financials = turn.get("financials")
    if not isinstance(financials, dict):
        return None
    return _round_to_int(financials.get(key))


def _resolve_rollout_success(rollout: Dict[str, Any], turns: List[Dict[str, Any]]) -> bool:
    top_level = _coerce_bool(rollout.get("success"))
    if top_level is not None:
        return top_level
    turn_values = [_coerce_bool(turn.get("success")) for turn in turns]
    return any(value is True for value in turn_values)


def _resolve_source_system(turns: List[Dict[str, Any]]) -> str:
    if not turns:
        return ""
    first_messages = [_normalize_message(msg) for msg in list(turns[0].get("messages") or [])]
    if first_messages and first_messages[0].get("role") == "system":
        return first_messages[0].get("content", "")
    return ""


def _resolve_completed_day(
    rollout: Dict[str, Any],
    turns: List[Dict[str, Any]],
    completed_turns: int,
) -> int:
    next_turn = turns[completed_turns]
    parsed_day = _extract_day(_resolve_turn_user_content(next_turn))
    if parsed_day is not None:
        return parsed_day
    initial_day = _extract_day(rollout.get("initial_state")) or 0
    return initial_day + (14 * int(completed_turns))


def _resolve_total_weeks(rollout: Dict[str, Any], turns: List[Dict[str, Any]]) -> int:
    final_day = _extract_day(rollout.get("final_state"))
    if final_day is None:
        initial_day = _extract_day(rollout.get("initial_state")) or 0
        final_day = initial_day + (14 * len(turns))
    return int(round(float(final_day) / 7.0))


def _resolve_final_cash(rollout: Dict[str, Any], turns: List[Dict[str, Any]]) -> Optional[int]:
    if turns:
        from_financials = _resolve_financial_metric(turns[-1], "cash")
        if from_financials is not None:
            return from_financials
    return _extract_cash(rollout.get("final_state"))


def _resolve_budget(total_value: int, absolute_value: Optional[float], ratio: float) -> int:
    if absolute_value is not None:
        return int(round(float(absolute_value)))
    return int(round(float(total_value) * float(ratio)))


def _resolve_target_cash(
    final_cash_usd: int,
    *,
    absolute_value: Optional[float],
    ratio: float,
    mode: str,
    half_reachable_target_cash_usd: Optional[int],
) -> int:
    if str(mode) == "half_reachable":
        if half_reachable_target_cash_usd is None:
            raise ValueError("half_reachable target cash mode requires per-rollout target cash.")
        return int(half_reachable_target_cash_usd)
    return _resolve_budget(
        total_value=final_cash_usd,
        absolute_value=absolute_value,
        ratio=ratio,
    )


def _sample_half_reachable_target_cash(
    final_cash_usd: int,
    *,
    reachable: bool,
    rng: random.Random,
) -> int:
    if reachable:
        sampled_ratio = rng.uniform(0.92, 0.99)
        target_cash_usd = int(round(float(final_cash_usd) * sampled_ratio))
        return min(int(final_cash_usd), target_cash_usd)

    sampled_ratio = rng.uniform(1.01, 1.10)
    target_cash_usd = int(round(float(final_cash_usd) * sampled_ratio))
    if target_cash_usd <= int(final_cash_usd):
        target_cash_usd = int(final_cash_usd) + max(
            1,
            int(round(abs(float(final_cash_usd)) * 0.05)),
        )
    return target_cash_usd


def _normalize_interval(lower: float, upper: float) -> Tuple[float, float]:
    if lower <= upper:
        return float(lower), float(upper)
    return float(upper), float(lower)


def _parse_labeled_interval(answer_text: str, labels: List[str]) -> Optional[Tuple[float, float]]:
    for label in labels:
        pattern = rf"{label}\s*[:：]\s*\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\]"
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return _normalize_interval(float(match.group(1)), float(match.group(2)))
    return None


def _parse_answer_intervals(answer_text: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    if answer_text is None:
        return None

    time_interval = _parse_labeled_interval(
        answer_text,
        labels=[r"time_weeks", r"time"],
    )
    warehouse_interval = _parse_labeled_interval(
        answer_text,
        labels=[r"warehouse_item_weeks", r"warehouse"],
    )
    cost_interval = _parse_labeled_interval(
        answer_text,
        labels=[r"cumulative_cost_usd", r"cumulative_cost", r"cost"],
    )

    if time_interval and warehouse_interval and cost_interval:
        return {
            "time_weeks": time_interval,
            "warehouse_item_weeks": warehouse_interval,
            "cumulative_cost_usd": cost_interval,
        }

    bracketed = re.findall(
        rf"\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\]",
        answer_text,
        flags=re.IGNORECASE,
    )
    if len(bracketed) < 3:
        return None
    fallback_intervals = [
        _normalize_interval(float(low), float(high))
        for low, high in bracketed[:3]
    ]
    return {
        "time_weeks": fallback_intervals[0],
        "warehouse_item_weeks": fallback_intervals[1],
        "cumulative_cost_usd": fallback_intervals[2],
    }


def _interval_contains(interval: Optional[Tuple[float, float]], actual: Optional[int]) -> Optional[bool]:
    if interval is None or actual is None:
        return None
    return float(interval[0]) <= float(actual) <= float(interval[1])


def _interval_width(interval: Optional[Tuple[float, float]]) -> Optional[float]:
    if interval is None:
        return None
    return float(interval[1]) - float(interval[0])


def _interval_midpoint_abs_error(interval: Optional[Tuple[float, float]], actual: Optional[int]) -> Optional[float]:
    if interval is None or actual is None:
        return None
    midpoint = (float(interval[0]) + float(interval[1])) / 2.0
    return abs(midpoint - float(actual))


class MoneyEstimationEnv(BaseLanguageBasedEnv):
    def __init__(self, config: MoneyEstimationEnvConfig):
        super(MoneyEstimationEnv, self).__init__()
        self.config = config
        self.rollouts = self._load_rollouts(self.config.input_path)
        self.samples = self._flatten_rollouts(self.rollouts)
        self.current_sample: Optional[MoneyEstimationSample] = None
        self.current_index: Optional[int] = None
        self.render_cache: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def _load_rollouts(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Money estimation input file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(
                f"Money estimation input must be a list or dict, got {type(payload).__name__}."
            )
        return payload

    def _build_half_reachable_target_cash_map(
        self,
        prepared_rollouts: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        rng = random.Random(int(self.config.target_cash_half_reachable_seed))
        rollout_indices = [int(entry["rollout_index"]) for entry in prepared_rollouts]
        shuffled_rollout_indices = list(rollout_indices)
        rng.shuffle(shuffled_rollout_indices)
        unreachable_rollouts = set(
            shuffled_rollout_indices[: len(shuffled_rollout_indices) // 2]
        )

        target_cash_by_rollout: Dict[int, int] = {}
        for entry in prepared_rollouts:
            rollout_index = int(entry["rollout_index"])
            target_cash_by_rollout[rollout_index] = _sample_half_reachable_target_cash(
                int(entry["final_cash_usd"]),
                reachable=rollout_index not in unreachable_rollouts,
                rng=rng,
            )
        return target_cash_by_rollout

    def _flatten_rollouts(self, rollouts: List[Dict[str, Any]]) -> List[MoneyEstimationSample]:
        prepared_rollouts: List[Dict[str, Any]] = []
        for rollout_index, rollout in enumerate(rollouts):
            turns = sorted(
                [
                    turn
                    for turn in list(rollout.get("turns") or [])
                    if _resolve_turn_assistant_content(turn)
                ],
                key=lambda turn: int(turn.get("turn_idx", 0) or 0),
            )
            if len(turns) < 2:
                continue

            final_cash_usd = _resolve_final_cash(rollout, turns)
            actual_total_warehouse_item_weeks = _resolve_financial_metric(turns[-1], "cumulative_inventory_weeks")
            actual_total_cost_usd = _resolve_financial_metric(turns[-1], "cumulative_cost")
            actual_total_time_weeks = _resolve_total_weeks(rollout, turns)
            if (
                final_cash_usd is None
                or actual_total_warehouse_item_weeks is None
                or actual_total_cost_usd is None
            ):
                continue

            prepared_rollouts.append(
                {
                    "rollout_index": rollout_index,
                    "rollout": rollout,
                    "turns": turns,
                    "final_cash_usd": int(final_cash_usd),
                    "actual_total_warehouse_item_weeks": int(actual_total_warehouse_item_weeks),
                    "actual_total_cost_usd": int(actual_total_cost_usd),
                    "actual_total_time_weeks": int(actual_total_time_weeks),
                }
            )

        target_cash_by_rollout: Dict[int, int] = {}
        if str(self.config.target_cash_mode) == "half_reachable":
            target_cash_by_rollout = self._build_half_reachable_target_cash_map(
                prepared_rollouts
            )

        samples: List[MoneyEstimationSample] = []
        for prepared_rollout in prepared_rollouts:
            rollout_index = int(prepared_rollout["rollout_index"])
            rollout = prepared_rollout["rollout"]
            turns = prepared_rollout["turns"]
            final_cash_usd = int(prepared_rollout["final_cash_usd"])
            actual_total_warehouse_item_weeks = int(
                prepared_rollout["actual_total_warehouse_item_weeks"]
            )
            actual_total_cost_usd = int(prepared_rollout["actual_total_cost_usd"])
            actual_total_time_weeks = int(prepared_rollout["actual_total_time_weeks"])

            target_cash_usd = _resolve_target_cash(
                final_cash_usd=final_cash_usd,
                absolute_value=self.config.target_cash_usd,
                ratio=self.config.target_cash_ratio,
                mode=str(self.config.target_cash_mode),
                half_reachable_target_cash_usd=target_cash_by_rollout.get(rollout_index),
            )
            budget_time_weeks = _resolve_budget(
                total_value=actual_total_time_weeks,
                absolute_value=self.config.time_budget_weeks,
                ratio=self.config.time_budget_ratio,
            )
            budget_warehouse_item_weeks = _resolve_budget(
                total_value=actual_total_warehouse_item_weeks,
                absolute_value=self.config.warehouse_budget_item_weeks,
                ratio=self.config.warehouse_budget_ratio,
            )
            budget_cost_usd = _resolve_budget(
                total_value=actual_total_cost_usd,
                absolute_value=self.config.cost_budget_usd,
                ratio=self.config.cost_budget_ratio,
            )

            rollout_success = _resolve_rollout_success(rollout, turns)
            actual_can_finish = bool(
                rollout_success
                and actual_total_time_weeks <= budget_time_weeks
                and actual_total_warehouse_item_weeks <= budget_warehouse_item_weeks
                and actual_total_cost_usd <= budget_cost_usd
                and final_cash_usd >= target_cash_usd
            )

            source_system = _resolve_source_system(turns)
            env_id = _safe_int(rollout.get("env_id"))
            absolute_env_id = _safe_int(rollout.get("absolute_env_id"))

            initial_day = _extract_day(rollout.get("initial_state")) or 0
            cumulative_warehouse_so_far = 0
            cumulative_cost_so_far = 0
            consumption_history: List[Dict[str, int]] = []
            for turn_offset, turn in enumerate(turns, start=1):
                current_warehouse = _resolve_financial_metric(turn, "cumulative_inventory_weeks")
                current_cost = _resolve_financial_metric(turn, "cumulative_cost")
                if current_warehouse is None or current_cost is None:
                    consumption_history.append(
                        {
                            "start_day": initial_day + ((turn_offset - 1) * 14),
                            "end_day": initial_day + (turn_offset * 14),
                            "time_weeks": 2,
                            "warehouse_item_weeks": 0,
                            "cost_usd": 0,
                        }
                    )
                    continue
                delta_warehouse = max(0, current_warehouse - cumulative_warehouse_so_far)
                delta_cost = max(0, current_cost - cumulative_cost_so_far)
                cumulative_warehouse_so_far = current_warehouse
                cumulative_cost_so_far = current_cost
                consumption_history.append(
                    {
                        "start_day": initial_day + ((turn_offset - 1) * 14),
                        "end_day": initial_day + (turn_offset * 14),
                        "time_weeks": 2,
                        "warehouse_item_weeks": delta_warehouse,
                        "cost_usd": delta_cost,
                    }
                )

            for completed_turns in range(1, len(turns)):
                current_turn = turns[completed_turns - 1]
                next_turn = turns[completed_turns]
                current_cash_usd = _resolve_financial_metric(current_turn, "cash")
                if current_cash_usd is None:
                    current_cash_usd = _extract_cash(_resolve_turn_user_content(next_turn))
                current_warehouse_item_weeks = _resolve_financial_metric(
                    current_turn, "cumulative_inventory_weeks"
                )
                current_cost_usd = _resolve_financial_metric(current_turn, "cumulative_cost")
                if (
                    current_cash_usd is None
                    or current_warehouse_item_weeks is None
                    or current_cost_usd is None
                ):
                    continue

                completed_day = _resolve_completed_day(rollout, turns, completed_turns)
                completed_weeks = int(round(float(completed_day) / 7.0))

                history_messages: List[Dict[str, str]] = []
                for prior_turn in turns[:completed_turns]:
                    history_messages.append(
                        {
                            "role": "user",
                            "content": _resolve_turn_user_content(prior_turn),
                        }
                    )
                    history_messages.append(
                        {
                            "role": "assistant",
                            "content": _resolve_turn_assistant_content(prior_turn),
                        }
                    )

                target_output = _resolve_turn_assistant_content(next_turn)
                actual_remaining_turn = max(0, len(turns) - completed_turns)
                relative_progress = float(completed_turns) / float(len(turns))

                samples.append(
                    MoneyEstimationSample(
                        sample_id=f"rollout-{rollout_index}-turn-{completed_turns}",
                        rollout_index=rollout_index,
                        env_id=env_id,
                        absolute_env_id=absolute_env_id,
                        turn_idx=completed_turns,
                        total_turns=len(turns),
                        source_system=source_system,
                        input_messages=history_messages,
                        target_output=target_output,
                        completed_turns=completed_turns,
                        completed_day=completed_day,
                        completed_weeks=completed_weeks,
                        relative_progress=relative_progress,
                        current_cash_usd=current_cash_usd,
                        target_cash_usd=target_cash_usd,
                        current_time_weeks=completed_weeks,
                        current_warehouse_item_weeks=current_warehouse_item_weeks,
                        current_cost_usd=current_cost_usd,
                        budget_time_weeks=budget_time_weeks,
                        budget_warehouse_item_weeks=budget_warehouse_item_weeks,
                        budget_cost_usd=budget_cost_usd,
                        consumption_history=[
                            dict(entry) for entry in consumption_history[:completed_turns]
                        ],
                        actual_can_finish=actual_can_finish,
                        actual_remaining_turn=actual_remaining_turn,
                        actual_remaining_time_weeks=(
                            max(0, actual_total_time_weeks - completed_weeks)
                            if actual_can_finish
                            else None
                        ),
                        actual_remaining_warehouse_item_weeks=(
                            max(0, actual_total_warehouse_item_weeks - current_warehouse_item_weeks)
                            if actual_can_finish
                            else None
                        ),
                        actual_remaining_cost_usd=(
                            max(0, actual_total_cost_usd - current_cost_usd)
                            if actual_can_finish
                            else None
                        ),
                        actual_total_time_weeks=actual_total_time_weeks,
                        actual_total_warehouse_item_weeks=actual_total_warehouse_item_weeks,
                        actual_total_cost_usd=actual_total_cost_usd,
                        rollout_success=rollout_success,
                    )
                )
                if self.config.max_instances is not None and len(samples) >= int(self.config.max_instances):
                    return samples
        return samples

    def _format_input_messages(self, messages: List[Dict[str, str]]) -> str:
        blocks = []
        for idx, message in enumerate(messages, start=1):
            role = message.get("role", "").upper() or "UNKNOWN"
            content = message.get("content", "")
            blocks.append(f"[{idx}] {role}\n{content}")
        return "\n\n".join(blocks)

    def _build_resource_consumption_text(self, sample: MoneyEstimationSample) -> str:
        if not sample.consumption_history:
            return "None yet."
        parts = []
        for entry in sample.consumption_history:
            parts.append(
                "Day {start_day}: time +{time_weeks} weeks, warehouse +{warehouse_item_weeks} item-weeks, "
                "cost +{cost_usd} USD".format(**entry)
            )
        return "; ".join(parts)

    def _render_api_messages(self, messages: List[Dict[str, str]]) -> str:
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def build_user_prompt(self, sample: MoneyEstimationSample) -> str:
        return self.config.user_prompt_template.format(
            completed_weeks=int(sample.completed_weeks),
            completed_turns=int(sample.completed_turns),
            completed_day=int(sample.completed_day),
            total_turns=int(sample.total_turns),
            relative_progress=float(sample.relative_progress),
            relative_progress_text=f"{float(sample.relative_progress):.2f}",
            current_cash_usd=int(sample.current_cash_usd),
            target_cash_usd=int(sample.target_cash_usd),
            current_time_weeks=int(sample.current_time_weeks),
            current_warehouse_item_weeks=int(sample.current_warehouse_item_weeks),
            current_cost_usd=int(sample.current_cost_usd),
            budget_time_weeks=int(sample.budget_time_weeks),
            budget_warehouse_item_weeks=int(sample.budget_warehouse_item_weeks),
            budget_cost_usd=int(sample.budget_cost_usd),
            resource_consumption_text=self._build_resource_consumption_text(sample),
        ).strip()

    def build_system_prompt(self) -> str:
        return self.config.system_prompt_template.strip()

    def build_api_messages(self, sample: Optional[MoneyEstimationSample] = None) -> List[Dict[str, str]]:
        sample = sample or self.current_sample
        if sample is None:
            raise ValueError("No active money estimation sample. Call reset() first or pass a sample.")
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        messages.extend(_normalize_message(message) for message in sample.input_messages)
        messages.append({"role": "user", "content": self.build_user_prompt(sample)})
        return messages

    def get_sample(self, index: int) -> MoneyEstimationSample:
        return self.samples[int(index)]

    def export_temp_pairs(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = []
        for sample in self.samples:
            api_messages = self.build_api_messages(sample)
            payload.append(
                {
                    "sample_id": sample.sample_id,
                    "rollout_index": sample.rollout_index,
                    "turn_idx": sample.turn_idx,
                    "input_messages": api_messages,
                    "rollout_history_messages": sample.input_messages,
                    "input_text": self._render_api_messages(api_messages),
                    "estimation_user_prompt": self.build_user_prompt(sample),
                    "output": sample.target_output,
                    "source_system": sample.source_system,
                    "completed_turns": sample.completed_turns,
                    "completed_weeks": sample.completed_weeks,
                    "completed_day": sample.completed_day,
                    "total_turns": sample.total_turns,
                    "relative_progress": sample.relative_progress,
                    "current_cash_usd": sample.current_cash_usd,
                    "target_cash_usd": sample.target_cash_usd,
                    "current_time_weeks": sample.current_time_weeks,
                    "current_warehouse_item_weeks": sample.current_warehouse_item_weeks,
                    "current_cost_usd": sample.current_cost_usd,
                    "budget_time_weeks": sample.budget_time_weeks,
                    "budget_warehouse_item_weeks": sample.budget_warehouse_item_weeks,
                    "budget_cost_usd": sample.budget_cost_usd,
                    "consumption_history": sample.consumption_history,
                    "actual_can_finish": sample.actual_can_finish,
                    "actual_remaining_time_weeks": sample.actual_remaining_time_weeks,
                    "actual_remaining_warehouse_item_weeks": sample.actual_remaining_warehouse_item_weeks,
                    "actual_remaining_cost_usd": sample.actual_remaining_cost_usd,
                    "actual_total_time_weeks": sample.actual_total_time_weeks,
                    "actual_total_warehouse_item_weeks": sample.actual_total_warehouse_item_weeks,
                    "actual_total_cost_usd": sample.actual_total_cost_usd,
                }
            )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path

    def reset(
        self,
        seed: Optional[int] = None,
        mode: Optional[str] = None,
        index: Optional[int] = None,
    ) -> str:
        if not self.samples:
            raise ValueError("No money estimation samples were loaded.")
        if index is None:
            with all_seed(seed):
                index = random.randint(0, len(self.samples) - 1)
        self.current_index = int(index)
        self.current_sample = self.samples[self.current_index]
        self.last_result = None
        self.render_cache = self._render_api_messages(self.build_api_messages(self.current_sample))
        return self.render_cache

    def parse_prediction(self, action: str) -> Dict[str, Any]:
        response = str(action or "").strip()
        answer_text = _extract_tag_text(response, "answer")
        is_impossible = (
            answer_text is not None and answer_text.strip().lower() == "impossible"
        )
        intervals = None if is_impossible else _parse_answer_intervals(answer_text)
        can_finish = None
        if is_impossible:
            can_finish = False
        elif intervals is not None:
            can_finish = True
        return {
            "raw_response": response,
            "think": _extract_tag_text(response, "think"),
            "answer_raw": answer_text,
            "can_finish": can_finish,
            "intervals": intervals,
            "is_impossible": is_impossible,
        }

    def evaluate_prediction(
        self,
        sample: MoneyEstimationSample,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        intervals = prediction.get("intervals") or {}
        time_interval = intervals.get("time_weeks")
        warehouse_interval = intervals.get("warehouse_item_weeks")
        cost_interval = intervals.get("cumulative_cost_usd")

        can_finish_correct = (
            prediction.get("can_finish") == sample.actual_can_finish
            if prediction.get("can_finish") is not None
            else None
        )
        time_contains_actual = _interval_contains(time_interval, sample.actual_remaining_time_weeks)
        warehouse_contains_actual = _interval_contains(
            warehouse_interval,
            sample.actual_remaining_warehouse_item_weeks,
        )
        cost_contains_actual = _interval_contains(cost_interval, sample.actual_remaining_cost_usd)

        reward_terms: List[float] = []
        if can_finish_correct is not None:
            reward_terms.append(1.0 if can_finish_correct else 0.0)
        if sample.actual_can_finish and prediction.get("can_finish") is True:
            for contains_actual in [time_contains_actual, warehouse_contains_actual, cost_contains_actual]:
                if contains_actual is not None:
                    reward_terms.append(1.0 if contains_actual else 0.0)

        return {
            "can_finish_correct": can_finish_correct,
            "time_interval_contains_actual": time_contains_actual,
            "warehouse_interval_contains_actual": warehouse_contains_actual,
            "cost_interval_contains_actual": cost_contains_actual,
            "time_interval_width_weeks": _interval_width(time_interval),
            "warehouse_interval_width_item_weeks": _interval_width(warehouse_interval),
            "cost_interval_width_usd": _interval_width(cost_interval),
            "time_midpoint_abs_error_weeks": _interval_midpoint_abs_error(
                time_interval,
                sample.actual_remaining_time_weeks,
            ),
            "warehouse_midpoint_abs_error_item_weeks": _interval_midpoint_abs_error(
                warehouse_interval,
                sample.actual_remaining_warehouse_item_weeks,
            ),
            "cost_midpoint_abs_error_usd": _interval_midpoint_abs_error(
                cost_interval,
                sample.actual_remaining_cost_usd,
            ),
            "all_intervals_cover_actual": (
                bool(time_contains_actual and warehouse_contains_actual and cost_contains_actual)
                if sample.actual_can_finish and prediction.get("can_finish") is True
                else None
            ),
            "reward": (sum(reward_terms) / len(reward_terms)) if reward_terms else 0.0,
        }

    def step(self, action: str):
        if self.current_sample is None:
            raise ValueError("No active money estimation sample. Call reset() first.")
        prediction = self.parse_prediction(action)
        metrics = self.evaluate_prediction(self.current_sample, prediction)
        result = {
            "sample": self.current_sample.to_dict(),
            "prediction": prediction,
            "metrics": metrics,
        }
        self.last_result = result
        self.render_cache = "Evaluation complete."
        return self.render_cache, float(metrics["reward"]), True, result

    def render(self, mode: str = "text") -> Any:
        return self.render_cache

    def close(self) -> None:
        self.current_sample = None
        self.current_index = None
        self.render_cache = None
        self.last_result = None
