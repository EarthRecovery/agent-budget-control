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
    completed_turn_token_usage: List[int]
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


def _last_known_token_value(values: List[Optional[int]]) -> Optional[int]:
    for value in reversed(values):
        if value is not None:
            return int(value)
    return None


def _compute_token_growths(values: List[Optional[int]]) -> List[int]:
    growths: List[int] = []
    last_known_total = 0
    for value in values:
        if value is None:
            growths.append(0)
            continue
        current_total = int(value)
        growths.append(max(0, current_total - last_known_total))
        last_known_total = current_total
    return growths


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
            turns = list(rollout.get("turns") or [])
            if not turns:
                continue
            rollout_success = any(bool(turn.get("success")) for turn in turns)
            per_turn_actual_tokens = [_resolve_turn_total_tokens(turn) for turn in turns]
            per_turn_token_growths = _compute_token_growths(per_turn_actual_tokens)
            final_turn_total_tokens = _last_known_token_value(per_turn_actual_tokens)
            for turn_offset, turn in enumerate(turns):
                messages = [_normalize_message(msg) for msg in list(turn.get("messages") or [])]
                if not messages:
                    continue

                source_system = ""
                input_messages = messages
                if messages[0].get("role") == "system":
                    source_system = messages[0].get("content", "")
                    input_messages = messages[1:]

                target_output = (
                    str(turn.get("parsed_response") or "").strip()
                    or str(turn.get("raw_response") or "").strip()
                )
                actual_remaining_turn = _safe_int(turn.get("actual_remaining_turn"))
                if actual_remaining_turn is None:
                    actual_remaining_turn = len(turns) - turn_offset
                completed_turn_token_usage = [
                    int(token_value)
                    for token_value in per_turn_token_growths[:turn_offset]
                ]
                # Context-window estimation is anchored to the finishing turn's
                # token footprint, not the cumulative spend across all turns.
                actual_tokens_used_so_far = sum(completed_turn_token_usage)
                actual_remaining_total_tokens = (
                    None
                    if final_turn_total_tokens is None
                    else max(0, int(final_turn_total_tokens) - int(actual_tokens_used_so_far))
                )
                actual_can_finish = (
                    rollout_success
                    and final_turn_total_tokens is not None
                    and int(final_turn_total_tokens)
                    <= int(self.config.max_context_window_tokens)
                )
                env_id = _safe_int(rollout.get("env_id"))
                absolute_env_id = _safe_int(rollout.get("absolute_env_id"))

                samples.append(
                    TokenEstimationSample(
                        sample_id=f"rollout-{rollout_index}-turn-{turn_offset + 1}",
                        rollout_index=rollout_index,
                        env_id=env_id,
                        absolute_env_id=absolute_env_id,
                        turn_idx=turn_offset + 1,
                        total_turns=len(turns),
                        source_system=source_system,
                        input_messages=input_messages,
                        target_output=target_output,
                        completed_turns=turn_offset,
                        completed_turn_token_usage=completed_turn_token_usage,
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

    def build_user_prompt(self, sample: TokenEstimationSample) -> str:
        source_system = sample.source_system if self.config.include_source_system else ""
        if sample.completed_turn_token_usage:
            turn_token_usage_text = "\n".join(
                f"Turn {idx}: {token_usage} tokens"
                for idx, token_usage in enumerate(sample.completed_turn_token_usage, start=1)
            )
        else:
            turn_token_usage_text = "None yet."
        return self.config.user_prompt_template.format(
            source_system=source_system,
            input_messages_text=self._format_input_messages(sample.input_messages),
            input_messages_json=json.dumps(sample.input_messages, ensure_ascii=False, indent=2),
            turn_idx=int(sample.turn_idx),
            total_turns=int(sample.total_turns),
            completed_turns=int(sample.completed_turns),
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
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(sample)},
        ]

    def get_sample(self, index: int) -> TokenEstimationSample:
        return self.samples[int(index)]

    def export_temp_pairs(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = []
        for sample in self.samples:
            payload.append(
                {
                    "sample_id": sample.sample_id,
                    "rollout_index": sample.rollout_index,
                    "turn_idx": sample.turn_idx,
                    "input_messages": sample.input_messages,
                    "input_text": self.build_user_prompt(sample),
                    "output": sample.target_output,
                    "source_system": sample.source_system,
                    "completed_turns": sample.completed_turns,
                    "completed_turn_token_usage": sample.completed_turn_token_usage,
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
        self.render_cache = self.build_user_prompt(self.current_sample)
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
