import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed

from .config import TokenEstimationEnvConfig


_TRUE_VALUES = {"1", "true", "yes", "y"}
_FALSE_VALUES = {"0", "false", "no", "n"}


@dataclass
class TokenEstimationSample:
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
    relative_progress: float
    completed_turn_token_usage: List[int]
    completed_turn_token_usage_details: List[Dict[str, Optional[int]]]
    actual_tokens_used_so_far: int
    actual_can_finish: bool
    actual_remaining_turn: int
    actual_remaining_total_tokens: Optional[int]
    rollout_success: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_turn_total_tokens(turn: Dict[str, Any]) -> Optional[int]:
    total_tokens = _safe_int(turn.get("api_total_tokens"))
    if total_tokens is None:
        total_tokens = _safe_int(turn.get("total_tokens"))
    if total_tokens is not None:
        return total_tokens

    input_tokens = _safe_int(turn.get("api_input_tokens"))
    if input_tokens is None:
        input_tokens = _safe_int(turn.get("input_tokens"))
    output_tokens = _safe_int(turn.get("api_output_tokens"))
    if output_tokens is None:
        output_tokens = _safe_int(turn.get("output_tokens"))
    if input_tokens is not None or output_tokens is not None:
        return int(input_tokens or 0) + int(output_tokens or 0)

    return _safe_int(turn.get("actual_token"))


def _resolve_turn_input_tokens(turn: Dict[str, Any]) -> Optional[int]:
    value = _safe_int(turn.get("api_input_tokens"))
    if value is None:
        value = _safe_int(turn.get("input_tokens"))
    return value


def _resolve_turn_output_tokens(turn: Dict[str, Any]) -> Optional[int]:
    value = _safe_int(turn.get("api_output_tokens"))
    if value is None:
        value = _safe_int(turn.get("output_tokens"))
    if value is None:
        value = _safe_int(turn.get("actual_token"))
    return value


def _extract_last_user_message(messages: List[Dict[str, Any]]) -> Optional[str]:
    for message in reversed(messages):
        if str(message.get("role", "") or "").strip() == "user":
            return str(message.get("content", "") or "")
    return None


def _resolve_turn_user_content(turn: Dict[str, Any]) -> str:
    user_prompt = turn.get("user_prompt")
    if user_prompt is not None:
        return str(user_prompt)
    messages = list(turn.get("messages") or [])
    extracted = _extract_last_user_message(messages)
    return "" if extracted is None else extracted


def _resolve_turn_assistant_content(turn: Dict[str, Any]) -> str:
    parsed = str(turn.get("parsed_response") or "").strip()
    if parsed:
        return parsed
    return str(turn.get("raw_response") or "").strip()


def _is_context_token_truncated_turn(turn: Dict[str, Any]) -> bool:
    if bool(turn.get("context_token_truncated")):
        return True
    info = turn.get("info")
    return isinstance(info, dict) and bool(info.get("context_token_truncated"))


def _turn_has_no_executed_actions(turn: Dict[str, Any]) -> bool:
    actions = turn.get("actions")
    if isinstance(actions, list):
        return len(actions) == 0
    if actions not in (None, ""):
        return False
    action_names = turn.get("action_names")
    if isinstance(action_names, list):
        return len(action_names) == 0
    return False


def _is_implicit_context_token_truncated_turn(
    turn: Dict[str, Any],
    *,
    max_context_window_tokens: Optional[int],
) -> bool:
    token_limit = _safe_int(max_context_window_tokens)
    total_tokens = _resolve_turn_total_tokens(turn)
    if token_limit is None or total_tokens is None:
        return False
    if int(total_tokens) <= int(token_limit):
        return False
    if bool(turn.get("success")):
        return False
    if not _has_turn_assistant_content(turn):
        return False
    return _turn_has_no_executed_actions(turn)


def _has_direct_api_usage(turn: Dict[str, Any]) -> bool:
    interactions = turn.get("api_interactions")
    return isinstance(interactions, list) and len(interactions) > 0


def _has_turn_assistant_content(turn: Dict[str, Any]) -> bool:
    return bool(_resolve_turn_assistant_content(turn))


def _resolve_turn_token_usage_detail(turn: Dict[str, Any]) -> Dict[str, Optional[int]]:
    input_tokens = _resolve_turn_input_tokens(turn)
    output_tokens = _resolve_turn_output_tokens(turn)
    total_tokens = _resolve_turn_total_tokens(turn)

    if total_tokens is not None:
        if input_tokens is None and output_tokens is not None and int(output_tokens) <= int(total_tokens):
            input_tokens = int(total_tokens) - int(output_tokens)
        if output_tokens is None and input_tokens is not None and int(input_tokens) <= int(total_tokens):
            output_tokens = int(total_tokens) - int(input_tokens)

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _usage_details_look_cumulative(
    details: List[Dict[str, Optional[int]]],
) -> bool:
    comparable = 0
    cumulative_votes = 0
    prev_total: Optional[int] = None
    for detail in details:
        current_total = detail.get("total_tokens")
        current_input = detail.get("input_tokens")
        if prev_total is not None and current_input is not None:
            comparable += 1
            if int(current_input) >= int(prev_total):
                cumulative_votes += 1
        if current_total is not None:
            prev_total = int(current_total)
    if comparable == 0:
        return False
    return cumulative_votes * 2 >= comparable


def _normalize_turn_usage_details(
    raw_details: List[Dict[str, Optional[int]]],
    *,
    assume_cumulative: Optional[bool] = None,
) -> Tuple[List[Dict[str, Optional[int]]], List[Optional[int]], str]:
    if not raw_details:
        return [], [], "delta"

    use_cumulative = _usage_details_look_cumulative(raw_details) if assume_cumulative is None else bool(assume_cumulative)

    if not use_cumulative:
        cumulative_totals: List[Optional[int]] = []
        running_total = 0
        for detail in raw_details:
            total_tokens = detail.get("total_tokens")
            if total_tokens is None and (
                detail.get("input_tokens") is not None or detail.get("output_tokens") is not None
            ):
                total_tokens = int(detail.get("input_tokens") or 0) + int(detail.get("output_tokens") or 0)
            if total_tokens is None:
                cumulative_totals.append(None)
                continue
            running_total += int(total_tokens)
            cumulative_totals.append(running_total)
        return [dict(detail) for detail in raw_details], cumulative_totals, "delta"

    normalized_details: List[Dict[str, Optional[int]]] = []
    cumulative_totals: List[Optional[int]] = []
    prev_total = 0
    prev_total_known = True
    for detail in raw_details:
        current_input = detail.get("input_tokens")
        current_output = detail.get("output_tokens")
        current_total = detail.get("total_tokens")

        if current_total is None and (
            current_input is not None or current_output is not None
        ):
            current_total = int(current_input or 0) + int(current_output or 0)

        if current_total is None:
            cumulative_totals.append(None)
            normalized_details.append(
                {
                    "input_tokens": current_input,
                    "output_tokens": current_output,
                    "total_tokens": current_total,
                }
            )
            prev_total_known = False
            continue

        current_total_int = int(current_total)
        cumulative_totals.append(current_total_int)

        input_delta: Optional[int]
        if current_input is None:
            input_delta = None
        elif prev_total_known:
            input_delta = max(0, int(current_input) - int(prev_total))
        else:
            input_delta = int(current_input)

        output_delta = None if current_output is None else int(current_output)

        if prev_total_known:
            total_delta = max(0, current_total_int - int(prev_total))
        else:
            total_delta = current_total_int

        normalized_total = total_delta
        if input_delta is None and output_delta is not None:
            input_delta = max(0, total_delta - int(output_delta))

        normalized_details.append(
            {
                "input_tokens": input_delta,
                "output_tokens": output_delta,
                "total_tokens": normalized_total,
            }
        )
        prev_total = current_total_int
        prev_total_known = True

    return normalized_details, cumulative_totals, "cumulative"


def _normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    role = str(message.get("role", "") or "").strip()
    content = str(message.get("content", "") or "")
    return {"role": role, "content": content}


def _extract_tag_text(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _parse_range(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    lower, upper = numbers[0], numbers[1]
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


class TokenEstimationEnv(BaseLanguageBasedEnv):
    def __init__(self, config: TokenEstimationEnvConfig):
        super(TokenEstimationEnv, self).__init__()
        self.config = config
        self.rollouts = self._load_rollouts(self.config.input_path)
        self.samples = self._flatten_rollouts(self.rollouts)
        self.current_sample: Optional[TokenEstimationSample] = None
        self.current_index: Optional[int] = None
        self.render_cache: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None

    def _load_rollouts(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Token estimation input file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(
                f"Token estimation input must be a list or dict, got {type(payload).__name__}."
            )
        return payload

    def _flatten_rollouts(self, rollouts: List[Dict[str, Any]]) -> List[TokenEstimationSample]:
        samples: List[TokenEstimationSample] = []
        for rollout_index, rollout in enumerate(rollouts):
            sorted_raw_turns = sorted(
                list(rollout.get("turns") or []),
                key=lambda turn: int(turn.get("turn_idx", 0) or 0),
            )
            if not sorted_raw_turns:
                continue
            raw_turns: List[Dict[str, Any]] = []
            for turn in sorted_raw_turns:
                if _is_context_token_truncated_turn(turn) or _is_implicit_context_token_truncated_turn(
                    turn,
                    max_context_window_tokens=self.config.max_context_window_tokens,
                ):
                    break
                raw_turns.append(turn)
            if not raw_turns:
                continue
            turns = [turn for turn in raw_turns if _has_turn_assistant_content(turn)]
            if not turns:
                continue

            rollout_success = any(bool(turn.get("success")) for turn in turns)
            raw_per_turn_token_usage_details = [
                _resolve_turn_token_usage_detail(turn) for turn in turns
            ]
            # These rollout files store per-request token usage for progressively
            # longer prefixes:
            #   turn 1 -> tokens for {turn1}
            #   turn 2 -> tokens for {turn1, turn2}
            #   turn 3 -> tokens for {turn1, turn2, turn3}
            #
            # Direct API usage therefore does not imply "delta" accounting. Let
            # the normalizer infer whether the sequence is cumulative from the
            # observed token pattern instead of forcing delta mode.
            assume_cumulative = None
            per_turn_token_usage_details, cumulative_turn_totals, _ = (
                _normalize_turn_usage_details(
                    raw_per_turn_token_usage_details,
                    assume_cumulative=assume_cumulative,
                )
            )
            per_turn_total_tokens = list(cumulative_turn_totals)
            total_rollout_tokens = (
                None
                if any(token_value is None for token_value in per_turn_total_tokens)
                else int(per_turn_total_tokens[-1])
            )
            first_turn_messages = [
                _normalize_message(msg) for msg in list(turns[0].get("messages") or [])
            ]
            source_system = ""
            if first_turn_messages and first_turn_messages[0].get("role") == "system":
                source_system = first_turn_messages[0].get("content", "")

            env_id = _safe_int(rollout.get("env_id"))
            absolute_env_id = _safe_int(rollout.get("absolute_env_id"))

            for completed_turns in range(1, len(turns)):
                next_turn = turns[completed_turns]
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
                relative_progress = (
                    float(completed_turns) / float(len(turns))
                    if turns
                    else 0.0
                )
                completed_turn_token_usage = [
                    int(detail.get("total_tokens") or 0)
                    for detail in per_turn_token_usage_details[:completed_turns]
                ]
                completed_turn_token_usage_details = [
                    dict(detail) for detail in per_turn_token_usage_details[:completed_turns]
                ]
                current_total_tokens = per_turn_total_tokens[completed_turns - 1]
                actual_tokens_used_so_far = (
                    sum(completed_turn_token_usage)
                    if current_total_tokens is None
                    else int(current_total_tokens)
                )
                actual_remaining_total_tokens = (
                    None
                    if total_rollout_tokens is None or current_total_tokens is None
                    else max(0, int(total_rollout_tokens) - int(current_total_tokens))
                )
                actual_can_finish = (
                    rollout_success
                    and total_rollout_tokens is not None
                    and int(total_rollout_tokens) <= int(self.config.max_context_window_tokens)
                )

                samples.append(
                    TokenEstimationSample(
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
                        relative_progress=relative_progress,
                        completed_turn_token_usage=completed_turn_token_usage,
                        completed_turn_token_usage_details=completed_turn_token_usage_details,
                        actual_tokens_used_so_far=actual_tokens_used_so_far,
                        actual_can_finish=actual_can_finish,
                        actual_remaining_turn=actual_remaining_turn,
                        actual_remaining_total_tokens=actual_remaining_total_tokens,
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

    def _build_turn_token_usage_text(self, sample: TokenEstimationSample) -> str:
        if not sample.completed_turn_token_usage_details:
            return "None yet."
        turn_token_usage_parts = []
        for idx, detail in enumerate(sample.completed_turn_token_usage_details, start=1):
            input_tokens = detail.get("input_tokens")
            output_tokens = detail.get("output_tokens")
            total_tokens = detail.get("total_tokens")
            input_text = (
                f"{int(input_tokens)} tokens"
                if input_tokens is not None
                else "unknown tokens"
            )
            output_text = (
                f"{int(output_tokens)} tokens"
                if output_tokens is not None
                else "unknown tokens"
            )
            if total_tokens is None:
                turn_token_usage_parts.append(
                    f"Turn {idx}: input {input_text}, output {output_text}"
                )
            else:
                turn_token_usage_parts.append(
                    f"Turn {idx}: input {input_text}, output {output_text}, total {int(total_tokens)} tokens"
                )
        return "; ".join(turn_token_usage_parts)

    def _render_api_messages(self, messages: List[Dict[str, str]]) -> str:
        return json.dumps(messages, ensure_ascii=False, indent=2)

    def build_user_prompt(self, sample: TokenEstimationSample) -> str:
        source_system = sample.source_system if self.config.include_source_system else ""
        turn_token_usage_text = self._build_turn_token_usage_text(sample)
        history_json = json.dumps(sample.input_messages, ensure_ascii=False, indent=2)
        return self.config.user_prompt_template.format(
            source_system=source_system,
            input_messages_text=self._format_input_messages(sample.input_messages),
            input_messages_json=json.dumps(sample.input_messages, ensure_ascii=False, indent=2),
            history_text=self._format_input_messages(sample.input_messages),
            history_json=history_json,
            turn_idx=int(sample.turn_idx),
            total_turns=int(sample.total_turns),
            completed_turns=int(sample.completed_turns),
            relative_progress=float(sample.relative_progress),
            relative_progress_text=f"{float(sample.relative_progress):.2f}",
            turn_token_usage_text=turn_token_usage_text,
            max_context_window_tokens=int(self.config.max_context_window_tokens),
        ).strip()

    def build_system_prompt(self) -> str:
        return self.config.system_prompt_template.format(
            max_context_window_tokens=int(self.config.max_context_window_tokens),
        ).strip()

    def build_api_messages(self, sample: Optional[TokenEstimationSample] = None) -> List[Dict[str, str]]:
        sample = sample or self.current_sample
        if sample is None:
            raise ValueError("No active token estimation sample. Call reset() first or pass a sample.")
        messages: List[Dict[str, str]] = []
        source_system = sample.source_system.strip() if self.config.include_source_system else ""
        fallback_system = self.build_system_prompt()
        if source_system:
            messages.append({"role": "system", "content": source_system})
        elif fallback_system:
            messages.append({"role": "system", "content": fallback_system})
        messages.extend(_normalize_message(message) for message in sample.input_messages)
        messages.append({"role": "user", "content": self.build_user_prompt(sample)})
        return messages

    def get_sample(self, index: int) -> TokenEstimationSample:
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
                    "total_turns": sample.total_turns,
                    "relative_progress": sample.relative_progress,
                    "completed_turn_token_usage": sample.completed_turn_token_usage,
                    "completed_turn_token_usage_details": sample.completed_turn_token_usage_details,
                    "actual_tokens_used_so_far": sample.actual_tokens_used_so_far,
                    "actual_can_finish": sample.actual_can_finish,
                    "actual_remaining_total_tokens": sample.actual_remaining_total_tokens,
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
            raise ValueError("No token estimation samples were loaded.")
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
        remaining_token_interval = None if is_impossible else _parse_range(answer_text)
        can_finish = None
        if is_impossible:
            can_finish = False
        elif remaining_token_interval is not None:
            can_finish = True
        return {
            "raw_response": response,
            "think": _extract_tag_text(response, "think"),
            "answer_raw": answer_text,
            "can_finish": can_finish,
            "remaining_token_interval": remaining_token_interval,
            "is_impossible": is_impossible,
        }

    def evaluate_prediction(
        self,
        sample: TokenEstimationSample,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        remaining_token_interval = prediction.get("remaining_token_interval")
        interval_width = (
            int(remaining_token_interval[1]) - int(remaining_token_interval[0])
            if remaining_token_interval is not None
            else None
        )
        remaining_token_interval_contains_actual = (
            (
                int(remaining_token_interval[0]) <= int(sample.actual_remaining_total_tokens) <= int(remaining_token_interval[1])
            )
            if remaining_token_interval is not None and sample.actual_remaining_total_tokens is not None
            else None
        )
        can_finish_correct = (
            prediction.get("can_finish") == sample.actual_can_finish
            if prediction.get("can_finish") is not None
            else None
        )

        reward_terms = []
        if can_finish_correct is not None:
            reward_terms.append(1.0 if can_finish_correct else 0.0)
        if remaining_token_interval is not None and sample.actual_can_finish:
            reward_terms.append(1.0 if remaining_token_interval_contains_actual else 0.0)

        return {
            "can_finish_correct": can_finish_correct,
            "remaining_token_interval_contains_actual": remaining_token_interval_contains_actual,
            "remaining_token_interval_width": interval_width,
            "reward": (sum(reward_terms) / len(reward_terms)) if reward_terms else 0.0,
        }

    def step(self, action: str):
        if self.current_sample is None:
            raise ValueError("No active token estimation sample. Call reset() first.")
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
