from typing import Any, Dict, List, Optional, Tuple

import copy
import json
import logging
import os
import re

import numpy as np
from tensordict import TensorDict


class CtxManagerWrapper:
    """
    Intercept and optionally reassemble contexts before they are sent to vLLM.

    Override `reassemble_messages` to customize the message list per turn.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.enabled = getattr(self.config.agent_proxy, "enable_ctx_wrapper", True)
        self.ctx_log = getattr(self.config.agent_proxy, "ctx_log", False)
        self._ctx_log_path = self._init_ctx_log_path()
        self.turn_idx = 0
        self.mode: Optional[str] = None
        self.state: Dict[str, Any] = {}
        self._validate_eval_estimation_mode()
        self._estimation_log_path = self._init_estimation_log_path()
        self._estimation_records: Dict[int, Dict[str, Any]] = {}
        self._pending_turn_records: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _agent_proxy_get(self, key: str, default: Any = None) -> Any:
        agent_cfg = getattr(self.config, "agent_proxy", None)
        if agent_cfg is None:
            return default
        if hasattr(agent_cfg, "get"):
            value = agent_cfg.get(key, None)
            if value is None:
                value = agent_cfg.get(key.replace("-", "_"), None)
            return default if value is None else value
        return getattr(agent_cfg, key.replace("-", "_"), default)

    def _get_eval_estimation_mode(self) -> Optional[str]:
        single_enabled = bool(self._agent_proxy_get("eval-estimation-single", False))
        multi_enabled = bool(self._agent_proxy_get("eval-estimation-multi", False))
        if single_enabled and multi_enabled:
            raise ValueError(
                "agent_proxy.eval-estimation-single and "
                "agent_proxy.eval-estimation-multi cannot both be True."
            )
        if multi_enabled:
            return "multi"
        if single_enabled:
            return "single"
        return None

    def _validate_eval_estimation_mode(self) -> None:
        self._get_eval_estimation_mode()

    def _eval_estimation_enabled(self) -> bool:
        return self._get_eval_estimation_mode() is not None

    def _resolve_log_dir(self) -> str:
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None and getattr(trainer_cfg, "local_log_dir", None):
            return str(trainer_cfg.local_log_dir)
        output_cfg = getattr(self.config, "output", None)
        if output_cfg is not None and getattr(output_cfg, "dir", None):
            return str(output_cfg.dir)
        return "logs"

    def _resolve_run_name(self) -> str:
        trainer_cfg = getattr(self.config, "trainer", None)
        if trainer_cfg is not None and getattr(trainer_cfg, "experiment_name", None):
            return str(trainer_cfg.experiment_name)
        output_cfg = getattr(self.config, "output", None)
        if output_cfg is not None and getattr(output_cfg, "filename", None):
            return os.path.splitext(str(output_cfg.filename))[0]
        model_cfg = getattr(self.config, "model_config", None)
        if model_cfg is not None and getattr(model_cfg, "model_name", None):
            return str(model_cfg.model_name)
        return "experiment"

    def _init_ctx_log_path(self) -> Optional[str]:
        if not self.ctx_log:
            return None
        log_dir = self._resolve_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"{self._resolve_run_name()}.log")

    def _init_estimation_log_path(self) -> Optional[str]:
        if not self._eval_estimation_enabled():
            return None
        log_dir = self._resolve_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(
            log_dir,
            f"{self._resolve_run_name()}_eval_estimation_dialogues.json",
        )

    def get_estimation_log_path(self) -> Optional[str]:
        return self._estimation_log_path

    def _write_ctx_log(self, payload: Dict[str, Any]) -> None:
        if not self.ctx_log or not self._ctx_log_path:
            return
        record = dict(payload)
        record.setdefault("turn_idx", self.turn_idx)
        record.setdefault("mode", self.mode)
        with open(self._ctx_log_path, "a", encoding="utf-8") as log_f:
            log_f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _resolve_estimation_slice_info(self) -> Tuple[int, int]:
        es_cfg = getattr(self.config, "es_manager", None)
        if es_cfg is None:
            return 0, 1
        val_cfg = getattr(es_cfg, "val", None)
        if val_cfg is None:
            return 0, 1
        start_group_index = int(getattr(val_cfg, "start_group_index", 0) or 0)
        group_size = int(getattr(val_cfg, "group_size", 1) or 1)
        return start_group_index, group_size

    def _build_estimation_payload(self) -> List[Dict[str, Any]]:
        payload = []
        start_group_index, group_size = self._resolve_estimation_slice_info()
        for env_id in sorted(self._estimation_records):
            record = self._estimation_records[env_id]
            turns = sorted(record.get("turns", []), key=lambda item: item.get("turn_idx", 0))
            api_interaction_count = self._sum_optional_ints(
                [turn.get("api_interaction_count") for turn in turns]
            )
            api_input_tokens = self._sum_optional_ints(
                [turn.get("api_input_tokens") for turn in turns]
            )
            api_output_tokens = self._sum_optional_ints(
                [turn.get("api_output_tokens") for turn in turns]
            )
            api_total_tokens = self._sum_optional_ints(
                [turn.get("api_total_tokens") for turn in turns]
            )
            group_id = record.get("group_id")
            absolute_group_id = (
                start_group_index + int(group_id)
                if group_id is not None
                else None
            )
            absolute_env_id = start_group_index * group_size + int(record.get("env_id", env_id))
            payload.append(
                {
                    "env_id": record.get("env_id"),
                    "group_id": record.get("group_id"),
                    "absolute_env_id": absolute_env_id,
                    "absolute_group_id": absolute_group_id,
                    "start_group_index": start_group_index,
                    "uid": record.get("uid"),
                    "tag": record.get("tag"),
                    "mode": record.get("mode"),
                    "budget_turn": record.get("budget_turn"),
                    "budget_token": record.get("budget_token"),
                    "total_turns": record.get("total_turns"),
                    "initial_state": record.get("initial_state"),
                    "final_state": record.get("final_state"),
                    "api_interaction_count": api_interaction_count,
                    "api_input_tokens": api_input_tokens,
                    "api_output_tokens": api_output_tokens,
                    "api_total_tokens": api_total_tokens,
                    "turns": turns,
                }
            )
        return payload

    def _load_existing_estimation_payload(self) -> List[Dict[str, Any]]:
        if not self._estimation_log_path or not os.path.exists(self._estimation_log_path):
            return []
        try:
            with open(self._estimation_log_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(
                "Failed to load existing estimation log %s: %s. Overwriting with current payload.",
                self._estimation_log_path,
                exc,
            )
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        logging.warning(
            "Unexpected estimation log payload type %s in %s. Overwriting with current payload.",
            type(payload).__name__,
            self._estimation_log_path,
        )
        return []

    def _write_estimation_log(self) -> None:
        if not self._estimation_log_path:
            return
        payload = self._load_existing_estimation_payload()
        payload.extend(self._build_estimation_payload())
        with open(self._estimation_log_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def begin_rollout(self) -> None:
        self._pending_turn_records = {}
        self._estimation_records = {}

    def finalize_rollout(self, rollout_states: List[Dict[str, Any]]) -> None:
        if not self._eval_estimation_enabled():
            return

        eval_mode = self._get_eval_estimation_mode()
        for rollout_state in rollout_states:
            env_id = int(rollout_state.get("env_id"))
            group_id = rollout_state.get("group_id")
            uid = rollout_state.get("uid")
            env_record = self._ensure_env_record(env_id, group_id=group_id, uid=uid)
            env_record["tag"] = rollout_state.get("tag")
            env_record["budget_turn"] = rollout_state.get("budget_turn")
            env_record["budget_token"] = rollout_state.get("budget_token")

            history = rollout_state.get("history", []) or []
            reward_turns = [turn for turn in history if "reward" in turn]
            env_record["total_turns"] = len(reward_turns)
            if history:
                env_record["initial_state"] = history[0].get("state")
                env_record["final_state"] = history[-1].get("state")

            for turn_idx, turn in enumerate(reward_turns, start=1):
                turn_record = self._ensure_turn_record(env_record, turn_idx)
                generation_error = turn.get("llm_error") or turn_record.get("generation_error")
                if generation_error:
                    llm_raw_response = str(
                        turn_record.get("raw_response")
                        or turn_record.get("raw_generation")
                        or turn.get("llm_raw_response")
                        or ""
                    )
                else:
                    llm_raw_response = str(
                        turn.get("llm_raw_response")
                        or turn_record.get("raw_response")
                        or turn_record.get("raw_generation")
                        or ""
                    )
                estimate_token = self._extract_token_estimate(llm_raw_response)
                actual_token = max(0, int(turn.get("token_count", 0) or 0))

                turn_record["raw_response"] = llm_raw_response
                turn_record["parsed_response"] = "" if generation_error else turn.get("llm_response")
                turn_record["actions"] = list(turn.get("actions", []) or [])
                turn_record["reward"] = turn.get("reward")
                turn_record["success"] = bool(turn.get("info", {}).get("success", False))
                turn_record["estimate_token"] = estimate_token
                turn_record["actual_token"] = actual_token
                turn_record["generation_error"] = generation_error
                turn_record["generation_error_type"] = (
                    turn.get("llm_error_type") or turn_record.get("generation_error_type")
                )
                turn_record["generation_error_code"] = (
                    turn.get("llm_error_code") or turn_record.get("generation_error_code")
                )
                turn_record["generation_error_status_code"] = (
                    turn.get("llm_error_status_code") or turn_record.get("generation_error_status_code")
                )
                turn_record["generation_retryable"] = (
                    turn.get("llm_error_retryable") if turn.get("llm_error_retryable") is not None
                    else turn_record.get("generation_retryable")
                )
                turn_record["generation_success"] = generation_error is None

                if eval_mode == "multi":
                    estimate_remaining_turn = self._extract_turn_estimate(llm_raw_response)
                    actual_remaining_turn = len(reward_turns) - turn_idx + 1
                    turn_record["estimate_remaining_turn"] = estimate_remaining_turn
                    turn_record["actual_remaining_turn"] = actual_remaining_turn

        self._pending_turn_records = {}
        self._write_estimation_log()

    def set_state(self, turn_idx: int, mode: Optional[str] = None, **kwargs: Any) -> None:
        self.turn_idx = turn_idx
        if mode is not None:
            self.mode = mode
        self.state.update(kwargs)

    def reassemble_messages(self, messages_list: List[List[Dict]]) -> List[List[Dict]]:
        return messages_list

    def _to_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _sum_optional_ints(self, values: List[Any]) -> Optional[int]:
        ints = []
        for value in values:
            if value is None:
                continue
            try:
                ints.append(int(value))
            except (TypeError, ValueError):
                continue
        if not ints:
            return None
        return sum(ints)

    def _sanitize_api_interactions(self, interactions: Any) -> List[Dict[str, Any]]:
        sanitized = []
        for interaction in self._to_list(interactions):
            if isinstance(interaction, dict):
                sanitized.append(copy.deepcopy(interaction))
        return sanitized

    def _summarize_api_interactions(
        self,
        interactions: List[Dict[str, Any]],
    ) -> Dict[str, Optional[int]]:
        return {
            "api_interaction_count": len(interactions),
            "api_input_tokens": self._sum_optional_ints(
                [item.get("input_tokens") for item in interactions]
            ),
            "api_output_tokens": self._sum_optional_ints(
                [item.get("output_tokens") for item in interactions]
            ),
            "api_total_tokens": self._sum_optional_ints(
                [item.get("total_tokens") for item in interactions]
            ),
        }

    def _ensure_env_record(
        self,
        env_id: int,
        group_id: Optional[int] = None,
        uid: Optional[Any] = None,
    ) -> Dict[str, Any]:
        env_id = int(env_id)
        if env_id not in self._estimation_records:
            self._estimation_records[env_id] = {
                "env_id": env_id,
                "group_id": int(group_id) if group_id is not None else None,
                "uid": uid,
                "mode": self._get_eval_estimation_mode(),
                "turns": [],
            }
        record = self._estimation_records[env_id]
        if group_id is not None:
            record["group_id"] = int(group_id)
        if uid is not None:
            record["uid"] = uid
        return record

    def _ensure_turn_record(
        self,
        env_record: Dict[str, Any],
        turn_idx: int,
    ) -> Dict[str, Any]:
        for turn in env_record["turns"]:
            if int(turn.get("turn_idx", -1)) == int(turn_idx):
                return turn
        turn_record = {
            "turn_idx": int(turn_idx),
            "mode": self.mode,
            "questions": self._build_eval_estimation_questions(),
        }
        env_record["turns"].append(turn_record)
        return turn_record

    def _build_eval_estimation_questions(self) -> List[str]:
        eval_mode = self._get_eval_estimation_mode()
        if eval_mode == "single":
            return ["How many tokens do you expect to need to answer the question?"]
        if eval_mode == "multi":
            return [
                "How many turns do you expect are still needed to finish?",
                "How many tokens do you expect to use in this turn?",
            ]
        return []

    def _extract_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def _record_estimation_inputs(
        self,
        non_tensor_batch: Dict[str, Any],
        messages_list: List[List[Dict[str, Any]]],
        texts: List[str],
        generation_suffix: str,
    ) -> None:
        if not self._eval_estimation_enabled():
            return

        env_ids = self._to_list(non_tensor_batch.get("env_ids"))
        group_ids = self._to_list(non_tensor_batch.get("group_ids"))
        uid_list = self._to_list(non_tensor_batch.get("uid"))
        turn_idx = int(self.turn_idx + 1)

        for idx, (env_id, messages, prompt_text) in enumerate(zip(env_ids, messages_list, texts)):
            group_id = group_ids[idx] if idx < len(group_ids) else None
            uid = uid_list[idx] if idx < len(uid_list) else None
            env_record = self._ensure_env_record(env_id, group_id=group_id, uid=uid)
            turn_record = self._ensure_turn_record(env_record, turn_idx)
            turn_record["mode"] = self.mode
            turn_record["questions"] = self._build_eval_estimation_questions()
            turn_record["messages"] = copy.deepcopy(messages)
            turn_record["user_prompt"] = self._extract_last_user_message(messages)
            turn_record["prompt_text"] = prompt_text
            turn_record["generation_suffix"] = generation_suffix
            self._pending_turn_records[(int(env_id), turn_idx)] = turn_record

    def _decorate_response_for_estimation(self, raw_response: str) -> str:
        eval_mode = self._get_eval_estimation_mode()
        if eval_mode in {"single", "multi"}:
            return self._ensure_leading_tag(raw_response, "budget-thinking")
        return raw_response

    def _ensure_leading_tag(self, raw_response: str, tag_name: str) -> str:
        text = str(raw_response)
        stripped = text.lstrip()
        opening_tag = f"<{tag_name}>"
        if stripped.startswith(opening_tag):
            return text
        return f"{opening_tag}{text}"

    def _count_tokens(self, raw_response: str) -> int:
        try:
            return len(self.tokenizer.encode(raw_response, add_special_tokens=False))
        except Exception:
            return len(str(raw_response).split())

    def _get_chat_template_kwargs(self) -> Dict:
        qwen_enable_thinking = getattr(self.config.agent_proxy, "qwen_enable_thinking", None)
        if qwen_enable_thinking is None:
            return {}
        return {"enable_thinking": bool(qwen_enable_thinking)}

    def _extract_tagged_int(
        self,
        text: str,
        tag_name: str,
    ) -> Optional[int]:
        if not text:
            return None
        match_iter = list(
            re.finditer(
                rf"<{tag_name}>\s*([+-]?\d+)\s*</{tag_name}>",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        )
        if not match_iter:
            return None

        # New multi-turn eval format places estimation tags after </budget-thinking>
        # and before the first <think> or <answer>. Keep accepting the legacy
        # prefix-before-budget format as a fallback so older logs still parse.
        budget_close_match = re.search(r"</budget-thinking>", text, re.IGNORECASE)
        if budget_close_match:
            budget_close_pos = budget_close_match.end()
            trailing_boundary_match = re.search(
                r"<think>|<answer>",
                text[budget_close_pos:],
                re.IGNORECASE,
            )
            trailing_boundary_pos = (
                budget_close_pos + trailing_boundary_match.start()
                if trailing_boundary_match
                else len(text)
            )
            for match in match_iter:
                if budget_close_pos <= match.start() < trailing_boundary_pos:
                    break
            else:
                match = None
        else:
            match = None

        if match is None:
            budget_open_match = re.search(r"<budget-thinking>", text, re.IGNORECASE)
            if budget_open_match:
                budget_open_pos = budget_open_match.end()
                trailing_boundary_match = re.search(
                    r"<think>|<answer>",
                    text[budget_open_pos:],
                    re.IGNORECASE,
                )
                trailing_boundary_pos = (
                    budget_open_pos + trailing_boundary_match.start()
                    if trailing_boundary_match
                    else len(text)
                )
                for candidate in match_iter:
                    if budget_open_pos <= candidate.start() < trailing_boundary_pos:
                        match = candidate
                        break

        if match is None:
            boundary_match = re.search(r"<budget-thinking>|<think>|<answer>", text, re.IGNORECASE)
            boundary_pos = boundary_match.start() if boundary_match else len(text)
            for candidate in match_iter:
                if candidate.start() < boundary_pos:
                    match = candidate
                    break
        if match is None:
            return None
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            return None
        return max(0, value)

    def _extract_token_estimate(self, text: str) -> Optional[int]:
        return self._extract_tagged_int(text, "token_estimation")

    def _extract_turn_estimate(self, text: str) -> Optional[int]:
        return self._extract_tagged_int(text, "turn_estimation")

    def _record_estimation_outputs(self, lm_outputs) -> None:
        if not self._eval_estimation_enabled():
            return
        if getattr(lm_outputs, "non_tensor_batch", None) is None:
            return

        response_texts = lm_outputs.non_tensor_batch.get("response_texts")
        env_ids = lm_outputs.non_tensor_batch.get("env_ids")
        if response_texts is None or env_ids is None:
            return

        response_texts = self._to_list(response_texts)
        env_ids = self._to_list(env_ids)
        response_errors = lm_outputs.non_tensor_batch.get("response_errors")
        api_interactions = lm_outputs.non_tensor_batch.get("api_interactions")
        if response_errors is None:
            response_errors = [None] * len(response_texts)
        else:
            response_errors = self._to_list(response_errors)
        if api_interactions is None:
            api_interactions = [[] for _ in response_texts]
        else:
            api_interactions = self._to_list(api_interactions)
        turn_idx = int(self.turn_idx + 1)

        for idx, (env_id, raw_generation) in enumerate(zip(env_ids, response_texts)):
            env_id_int = int(env_id)
            turn_record = self._pending_turn_records.get((env_id_int, turn_idx))
            if turn_record is None:
                env_record = self._ensure_env_record(env_id_int)
                turn_record = self._ensure_turn_record(env_record, turn_idx)
            response_error = response_errors[idx] if idx < len(response_errors) else None
            interactions = self._sanitize_api_interactions(
                api_interactions[idx] if idx < len(api_interactions) else []
            )
            raw_generation = str(raw_generation)
            full_response = (
                raw_generation
                if response_error is not None
                else self._decorate_response_for_estimation(raw_generation)
            )
            turn_record["raw_generation"] = raw_generation
            turn_record["raw_response"] = full_response
            turn_record["api_interactions"] = interactions
            turn_record.update(self._summarize_api_interactions(interactions))
            turn_record["estimate_token"] = (
                None if response_error is not None else self._extract_token_estimate(full_response)
            )
            turn_record["actual_token"] = 0 if response_error is not None else self._count_tokens(raw_generation)
            turn_record["generation_error"] = None if response_error is None else response_error.get("error")
            turn_record["generation_error_type"] = None if response_error is None else response_error.get("error_type")
            turn_record["generation_error_code"] = None if response_error is None else response_error.get("error_code")
            turn_record["generation_error_status_code"] = None if response_error is None else response_error.get("status_code")
            turn_record["generation_retryable"] = None if response_error is None else response_error.get("retryable")
            turn_record["generation_success"] = response_error is None
            if self._get_eval_estimation_mode() == "multi":
                estimate_remaining_turn = (
                    None if response_error is not None else self._extract_turn_estimate(full_response)
                )
                turn_record["estimate_remaining_turn"] = estimate_remaining_turn

    def _inject_budget_prompt(
        self,
        messages_list: List[List[Dict]],
        budget_turns: List[Optional[int]],
    ) -> None:
        current_turn = self.turn_idx + 1
        for messages, budget_turn in zip(messages_list, budget_turns):
            if budget_turn is None:
                continue
            turns_left = max(0, int(budget_turn) - current_turn)
            note = (
                f"You are expected to answer in turn {int(budget_turn)}, and there are "
                f"{turns_left} turn(s) left. You will be penalized if you generate "
                "answer later than this."
            )
            if not messages:
                continue
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if "You are expected to answer in turn" in content:
                        break
                    msg["content"] = content + ("\n" if content else "") + note
                    break

    def _append_note_to_last_user_message(
        self,
        messages_list: List[List[Dict]],
        note: str,
        marker: str,
    ) -> None:
        for messages in messages_list:
            if not messages:
                continue
            for msg in reversed(messages):
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if marker in content:
                    break
                msg["content"] = content + ("\n" if content else "") + note
                break

    def _inject_eval_estimation_single_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        note = (
            "Before your final answer, first answer this question: "
            "\"How many tokens do you expect to need to answer the question?\"\n"
            "Fill <token_estimation> with exactly one integer in the required response format."
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="How many tokens do you expect to need to answer the question?",
        )

    def _inject_eval_estimation_multi_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        note = (
            "Before your final answer, first answer these two questions in order:\n"
            "1. How many turns do you expect are still needed to finish? "
            "The count should include the current turn.\n"
            "2. How many tokens do you expect to use in this turn?\n"
            "Fill <turn_estimation> and <token_estimation> with integers in the required response format."
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="How many turns do you expect are still needed to finish?",
        )

    def _inject_token_estimation_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        note = (
            "Before your reasoning, first output token estimation as the first segment of your response. "
            "Write only one integer wrapped by <token_estimation> and </token_estimation> AFTER <budget-thinking>. "
            "Then continue with normal format. Example: "
            "<budget-thinking>...</budget-thinking><token_estimation>256</token_estimation>"
            "<think>...</think><answer>...</answer>."
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="<token_estimation>",
        )

    def _inject_turn_estimation_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        benchmark_cfg = getattr(self.config.agent_proxy, "benchmark_factors", None)
        benchmark_enabled = bool(getattr(benchmark_cfg, "enabled", False)) if benchmark_cfg is not None else False
        max_turn = int(self.state.get("max_turn", getattr(self.config.agent_proxy, "max_turn", 1)) or 1)
        if (not benchmark_enabled) or max_turn <= 1:
            return

        note = (
            "estimate how many turns you will expect to finish this task. "
            "After reasoning, output one integer wrapped by <turn_estimation> and </turn_estimation>. "
            "Example: <budget-thinking>...</budget-thinking><turn_estimation>3</turn_estimation>"
            "<think>...</think><answer>...</answer>."
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="estimate how many turns you will expect to finish this task.",
        )

    def _inject_benchmark_turn_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        benchmark_cfg = getattr(self.config.agent_proxy, "benchmark_factors", None)
        enabled = bool(getattr(benchmark_cfg, "enabled", False)) if benchmark_cfg is not None else False
        mode = str(getattr(benchmark_cfg, "mode", "turn")).strip().lower() if benchmark_cfg is not None else "turn"
        if (not enabled) or mode != "turn":
            return

        low_bound = int(getattr(benchmark_cfg, "low_bound", 0))
        high_bound = int(getattr(benchmark_cfg, "high_bound", -1))
        max_turn = int(self.state.get("max_turn", getattr(self.config.agent_proxy, "max_turn", 1)) or 1)
        current_turn = int(self.turn_idx + 1)
        used_turns = int(self.turn_idx)
        turns_left = max(0, max_turn - used_turns)

        if high_bound >= 0:
            target_msg = (
                f"You should finish this task between {low_bound} and {high_bound} turns. "
                f"You have already used {used_turns} turn(s), and you can still use up to {turns_left} turn(s)."
            )
        else:
            target_msg = (
                f"You should finish this task in at least {low_bound} turns. "
                f"You have already used {used_turns} turn(s), and you can still use up to {turns_left} turn(s)."
            )

        note = f"[Turn Budget Guidance] Current turn: {current_turn}. {target_msg}"
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="[Turn Budget Guidance]",
        )

    def _build_texts(
        self,
        messages_list: List[List[Dict]],
        add_generation_prompt: bool,
        generation_suffix: str,
    ) -> List[str]:
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **self._get_chat_template_kwargs(),
            )
            for messages in messages_list
        ]
        if add_generation_prompt and generation_suffix:
            texts = [text + generation_suffix for text in texts]
        return texts

    def _apply_max_length(
        self,
        messages: List[Dict],
        add_generation_prompt: bool,
    ) -> List[Dict]:
        max_model_len = getattr(self.config.actor_rollout_ref.rollout, "max_model_len", None)
        if max_model_len is None:
            return messages
        response_length = int(getattr(self.config.actor_rollout_ref.rollout, "response_length", 0) or 0)
        model_cfg = getattr(self.config, "model_config", None)
        prompt_token_margin = int(getattr(model_cfg, "prompt_token_margin", 0) or 0)
        max_length = max(1, int(max_model_len) - response_length - prompt_token_margin)

        full_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self._get_chat_template_kwargs(),
        )
        token_len = len(self.tokenizer(full_text, add_special_tokens=False)["input_ids"])

        if token_len <= max_length:
            return messages

        system_msg = messages[0]
        conversation = messages[1:]

        while token_len > max_length and len(conversation) > 2:
            if len(conversation) >= 3:
                if (
                    conversation[0]["role"] == "user"
                    and conversation[1]["role"] == "assistant"
                    and len(conversation) > 2
                    and conversation[2]["role"] == "user"
                    and "Reward" in conversation[2].get("content", "")
                ):
                    conversation = conversation[3:]
                elif (
                    conversation[0]["role"] == "user"
                    and conversation[1]["role"] == "assistant"
                ):
                    conversation = conversation[2:]
                else:
                    conversation = conversation[1:]
            else:
                break

            truncated = [system_msg] + conversation
            full_text = self.tokenizer.apply_chat_template(
                truncated,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **self._get_chat_template_kwargs(),
            )
            token_len = len(self.tokenizer(full_text, add_special_tokens=False)["input_ids"])

        if token_len > max_length:
            logging.warning(
                f"Cannot truncate prompt to {max_length} tokens (current: {token_len}). "
                f"Configured total context budget={int(max_model_len)}, response_length={response_length}, "
                f"prompt_token_margin={prompt_token_margin}. Single turn may exceed max length."
            )

        return [system_msg] + conversation

    def _tokenize(self, texts: List[str]):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=False,
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        return input_ids, attention_mask, position_ids

    def intercept(
        self,
        lm_inputs,
        add_generation_prompt: bool,
        generation_suffix: str,
    ):
        if not self.enabled:
            return lm_inputs

        messages_arr = lm_inputs.non_tensor_batch.get("messages_list")
        if messages_arr is None:
            return lm_inputs

        messages_list = (
            messages_arr.tolist() if hasattr(messages_arr, "tolist") else list(messages_arr)
        )
        messages_list = self.reassemble_messages(messages_list)
        budget_turns = lm_inputs.non_tensor_batch.get("budget_turns")
        if budget_turns is not None:
            budget_turns = (
                budget_turns.tolist()
                if hasattr(budget_turns, "tolist")
                else list(budget_turns)
            )
            self._inject_budget_prompt(messages_list, budget_turns)

        eval_mode = self._get_eval_estimation_mode()
        if eval_mode == "single":
            self._inject_eval_estimation_single_prompt(messages_list)
        elif eval_mode == "multi":
            self._inject_eval_estimation_multi_prompt(messages_list)
        else:
            if bool(getattr(self.config.agent_proxy, "token_estimation", False)):
                self._inject_token_estimation_prompt(messages_list)
            self._inject_turn_estimation_prompt(messages_list)

        self._inject_benchmark_turn_prompt(messages_list)

        messages_list = [
            self._apply_max_length(messages, add_generation_prompt)
            for messages in messages_list
        ]

        texts = self._build_texts(messages_list, add_generation_prompt, generation_suffix)
        self._record_estimation_inputs(
            non_tensor_batch=lm_inputs.non_tensor_batch,
            messages_list=messages_list,
            texts=texts,
            generation_suffix=generation_suffix,
        )
        self._write_ctx_log(
            {
                "type": "input",
                "messages_list": messages_list,
                "texts": texts,
                "add_generation_prompt": add_generation_prompt,
                "generation_suffix": generation_suffix,
            }
        )
        input_ids, attention_mask, position_ids = self._tokenize(texts)

        lm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": input_ids[:, 1:],
            },
            batch_size=input_ids.shape[0],
        )
        lm_inputs.non_tensor_batch["messages_list"] = np.array(messages_list, dtype=object)
        return lm_inputs

    def log_outputs(self, lm_outputs) -> None:
        response_texts = None
        if getattr(lm_outputs, "non_tensor_batch", None):
            response_texts = lm_outputs.non_tensor_batch.get("response_texts")
            if hasattr(response_texts, "tolist"):
                response_texts = response_texts.tolist()
        if response_texts is None:
            return

        self._record_estimation_outputs(lm_outputs)
        self._write_ctx_log(
            {
                "type": "output",
                "response_texts": response_texts,
            }
        )
