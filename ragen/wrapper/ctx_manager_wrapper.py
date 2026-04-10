from typing import Any, Dict, List, Optional, Tuple

import copy
import json
import logging
import os
import re

import numpy as np
from tensordict import TensorDict

from ragen.llm_agent.eval_config import (
    resolve_eval_adaptation_turn_config,
    resolve_eval_compliance_mode,
    resolve_eval_compliance_turn_budget_change,
    resolve_eval_compliance_turn_mutation_config,
    resolve_eval_compliance_turn_mutation_turn,
    resolve_eval_estimation_mode,
    resolve_toolcall_action_point_cap,
)


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
        self._validate_eval_modes()
        self._toolcall_action_point_cap = resolve_toolcall_action_point_cap(self.config)
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
        return resolve_eval_estimation_mode(self.config)

    def _eval_adaptation_turn_enabled(self) -> bool:
        return self._get_eval_estimation_mode() == "adaptation_turn"

    def _get_eval_adaptation_turn_config(self) -> Optional[Tuple[int, int, int]]:
        return resolve_eval_adaptation_turn_config(self.config)

    def _get_eval_compliance_mode(self) -> Optional[str]:
        return resolve_eval_compliance_mode(self.config)

    def _eval_compliance_token_enabled(self) -> bool:
        return self._get_eval_compliance_mode() == "token"

    def _eval_compliance_turn_enabled(self) -> bool:
        return self._get_eval_compliance_mode() == "turn"

    def _eval_compliance_toolcall_enabled(self) -> bool:
        return self._get_eval_compliance_mode() == "toolcall"

    def _eval_compliance_enabled(self) -> bool:
        return self._get_eval_compliance_mode() is not None

    def _normalize_eval_compliance_scope(
        self,
        raw_scope: Any,
        config_key: str,
    ) -> List[int]:
        if raw_scope is None:
            return []
        if hasattr(raw_scope, "tolist"):
            raw_scope = raw_scope.tolist()
        if isinstance(raw_scope, (str, bytes)):
            values = [raw_scope]
        else:
            try:
                values = list(raw_scope)
            except TypeError:
                values = [raw_scope]

        scope = []
        for idx, value in enumerate(values):
            try:
                scope.append(max(0, int(value)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"agent_proxy.{config_key} must contain integers. "
                    f"Invalid value at index {idx}: {value!r}."
                ) from exc
        return scope

    def _get_eval_compliance_token_scope(self) -> List[int]:
        raw_scope = self._agent_proxy_get("eval_compliance_token_scope", [])
        return self._normalize_eval_compliance_scope(
            raw_scope,
            config_key="eval_compliance_token_scope",
        )

    def _get_eval_compliance_turn_scope(self) -> List[int]:
        raw_scope = self._agent_proxy_get("eval_compliance_turn_scope", [])
        return self._normalize_eval_compliance_scope(
            raw_scope,
            config_key="eval_compliance_turn_scope",
        )

    def _get_eval_compliance_turn_mutation_turn(self) -> Optional[int]:
        return resolve_eval_compliance_turn_mutation_turn(self.config)

    def _get_eval_compliance_turn_budget_change(self) -> List[int]:
        return resolve_eval_compliance_turn_budget_change(self.config)

    def _get_eval_compliance_turn_mutation_config(self) -> Optional[Tuple[int, List[int]]]:
        return resolve_eval_compliance_turn_mutation_config(self.config)

    def _get_eval_compliance_toolcall_scope(self) -> List[int]:
        raw_scope = self._agent_proxy_get("eval_compliance_toolcall_scope", [])
        return self._normalize_eval_compliance_scope(
            raw_scope,
            config_key="eval_compliance_toolcall_scope",
        )

    def _get_eval_adaptation_turn_limit(
        self,
        current_turn: Optional[int] = None,
    ) -> Optional[int]:
        adaptation_cfg = self._get_eval_adaptation_turn_config()
        if adaptation_cfg is None:
            return None
        mutation_turn, budget_before, budget_after = adaptation_cfg
        resolved_turn = int(self.turn_idx + 1) if current_turn is None else int(current_turn)
        if resolved_turn <= int(mutation_turn):
            return int(budget_before)
        return int(budget_after)

    def _get_active_eval_log_mode(self) -> Optional[str]:
        eval_estimation_mode = self._get_eval_estimation_mode()
        if eval_estimation_mode is not None:
            return eval_estimation_mode
        compliance_mode = self._get_eval_compliance_mode()
        if compliance_mode == "token":
            return "compliance_token"
        if compliance_mode == "turn":
            return "compliance_turn"
        if compliance_mode == "toolcall":
            return "compliance_toolcall"
        return None

    def _validate_eval_modes(self) -> None:
        eval_estimation_mode = self._get_eval_estimation_mode()
        compliance_mode = self._get_eval_compliance_mode()
        if eval_estimation_mode is not None and compliance_mode is not None:
            raise ValueError(
                "agent_proxy.eval_compliance_token, agent_proxy.eval_compliance_turn, or agent_proxy.eval_compliance_toolcall cannot be enabled together with "
                "agent_proxy.eval-estimation-single, agent_proxy.eval-estimation-multi, agent_proxy.eval-estimation-toolcall, or agent_proxy.eval_adaptation_turn."
            )
        if eval_estimation_mode == "adaptation_turn":
            self._get_eval_adaptation_turn_config()
        if compliance_mode == "token" and not self._get_eval_compliance_token_scope():
            raise ValueError(
                "agent_proxy.eval_compliance_token is enabled, but "
                "agent_proxy.eval_compliance_token_scope is empty."
            )
        if compliance_mode == "turn":
            turn_scope = self._get_eval_compliance_turn_scope()
            turn_mutation_cfg = self._get_eval_compliance_turn_mutation_config()
            if turn_scope and turn_mutation_cfg is not None:
                raise ValueError(
                    "agent_proxy.eval_compliance_turn_scope cannot be used together with "
                    "agent_proxy.eval_compliance_turn_mutation_turn / "
                    "agent_proxy.eval_compliance_turn_budget_change."
                )
            if not turn_scope and turn_mutation_cfg is None:
                raise ValueError(
                    "agent_proxy.eval_compliance_turn is enabled, but neither "
                    "agent_proxy.eval_compliance_turn_scope nor the mutation-based turn compliance "
                    "configuration is set."
                )
        if compliance_mode == "toolcall" and not self._get_eval_compliance_toolcall_scope():
            raise ValueError(
                "agent_proxy.eval_compliance_toolcall is enabled, but "
                "agent_proxy.eval_compliance_toolcall_scope is empty."
            )
        if eval_estimation_mode == "toolcall":
            resolve_toolcall_action_point_cap(self.config)
        if compliance_mode == "toolcall":
            resolve_toolcall_action_point_cap(self.config)

    def _eval_estimation_enabled(self) -> bool:
        return self._get_eval_estimation_mode() is not None

    def _eval_logging_enabled(self) -> bool:
        return self._get_active_eval_log_mode() is not None

    def _output_filename_configured(self) -> bool:
        output_cfg = getattr(self.config, "output", None)
        if output_cfg is None:
            return False
        return bool(getattr(output_cfg, "filename", None))

    def _dialogue_logging_enabled(self) -> bool:
        return self._eval_logging_enabled() or self._output_filename_configured()

    def _get_dialogue_log_mode(self) -> Optional[str]:
        active_mode = self._get_active_eval_log_mode()
        if active_mode is not None:
            return active_mode
        if self._output_filename_configured():
            return "dialogue"
        return None

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
        if not self._dialogue_logging_enabled():
            return None
        log_dir = self._resolve_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        mode = self._get_active_eval_log_mode()
        suffix = (
            "_eval_compliance_dialogues.json"
            if mode in {"compliance_token", "compliance_turn", "compliance_toolcall"}
            else "_eval_estimation_dialogues.json"
        )
        return os.path.join(
            log_dir,
            f"{self._resolve_run_name()}{suffix}",
        )

    def get_estimation_log_path(self) -> Optional[str]:
        return self._estimation_log_path

    def get_eval_log_key(self) -> Optional[str]:
        mode = self._get_active_eval_log_mode()
        if mode in {"single", "multi", "toolcall", "adaptation_turn"}:
            return "eval_estimation_json_path"
        if mode in {"compliance_token", "compliance_turn", "compliance_toolcall"}:
            return "eval_compliance_json_path"
        return None

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
                    "eval_compliance_token_scope": record.get("eval_compliance_token_scope"),
                    "compliance_token_limit": record.get("compliance_token_limit"),
                    "eval_compliance_turn_scope": record.get("eval_compliance_turn_scope"),
                    "compliance_turn_limit": record.get("compliance_turn_limit"),
                    "eval_compliance_turn_mutation_turn": record.get("eval_compliance_turn_mutation_turn"),
                    "eval_compliance_turn_budget_change": record.get("eval_compliance_turn_budget_change"),
                    "eval_adaptation_turn_scope": record.get("eval_adaptation_turn_scope"),
                    "adaptation_turn_mutation_turn": record.get("adaptation_turn_mutation_turn"),
                    "adaptation_turn_budget_change": record.get("adaptation_turn_budget_change"),
                    "adaptation_turn_limit": record.get("adaptation_turn_limit"),
                    "within_adaptation_turn_limit": record.get("within_adaptation_turn_limit"),
                    "adaptation_turn_limit_delta": record.get("adaptation_turn_limit_delta"),
                    "success_within_adaptation_turn_limit": record.get("success_within_adaptation_turn_limit"),
                    "eval_compliance_toolcall_scope": record.get("eval_compliance_toolcall_scope"),
                    "compliance_toolcall_limit": record.get("compliance_toolcall_limit"),
                    "max_action_points": record.get("max_action_points"),
                    "budget_turn": record.get("budget_turn"),
                    "budget_token": record.get("budget_token"),
                    "budget_toolcall": record.get("budget_toolcall"),
                    "total_turns": record.get("total_turns"),
                    "within_turn_limit": record.get("within_turn_limit"),
                    "turn_limit_delta": record.get("turn_limit_delta"),
                    "success_within_turn_limit": record.get("success_within_turn_limit"),
                    "total_action_points_used": record.get("total_action_points_used"),
                    "total_toolcalls_used": record.get("total_toolcalls_used"),
                    "within_toolcall_limit": record.get("within_toolcall_limit"),
                    "toolcall_limit_delta": record.get("toolcall_limit_delta"),
                    "success_within_toolcall_limit": record.get("success_within_toolcall_limit"),
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
        if not self._dialogue_logging_enabled():
            return

        eval_mode = self._get_eval_estimation_mode()
        compliance_token_scope = self._get_eval_compliance_token_scope()
        compliance_turn_scope = self._get_eval_compliance_turn_scope()
        compliance_toolcall_scope = self._get_eval_compliance_toolcall_scope()
        adaptation_turn_cfg = self._get_eval_adaptation_turn_config()
        toolcall_cap = self._get_toolcall_action_point_cap()
        for rollout_state in rollout_states:
            env_id = int(rollout_state.get("env_id"))
            group_id = rollout_state.get("group_id")
            uid = rollout_state.get("uid")
            env_record = self._ensure_env_record(env_id, group_id=group_id, uid=uid)
            mixed_toolcall_budget = rollout_state.get("budget_toolcall")
            env_toolcall_cap = (
                int(mixed_toolcall_budget)
                if mixed_toolcall_budget is not None
                else (int(toolcall_cap) if toolcall_cap is not None else None)
            )
            compliance_token_limit = (
                self._get_eval_compliance_token_limit_for_env(
                    env_id=env_id,
                    group_id=None if group_id is None else int(group_id),
                )
                if self._eval_compliance_token_enabled()
                else None
            )
            compliance_turn_limit = (
                self._get_eval_compliance_turn_limit_for_env(
                    env_id=env_id,
                    group_id=None if group_id is None else int(group_id),
                    current_turn=None,
                )
                if self._eval_compliance_turn_enabled()
                else None
            )
            compliance_toolcall_limit = (
                self._get_eval_compliance_toolcall_limit_for_env(
                    env_id=env_id,
                    group_id=None if group_id is None else int(group_id),
                )
                if self._eval_compliance_toolcall_enabled()
                else None
            )
            if compliance_token_scope:
                env_record["eval_compliance_token_scope"] = list(compliance_token_scope)
            if compliance_token_limit is not None:
                env_record["compliance_token_limit"] = compliance_token_limit
            if compliance_turn_scope:
                env_record["eval_compliance_turn_scope"] = list(compliance_turn_scope)
            turn_mutation_cfg = self._get_eval_compliance_turn_mutation_config()
            if turn_mutation_cfg is not None:
                mutation_turn, budget_change = turn_mutation_cfg
                env_record["eval_compliance_turn_mutation_turn"] = int(mutation_turn)
                env_record["eval_compliance_turn_budget_change"] = list(budget_change)
            if adaptation_turn_cfg is not None:
                mutation_turn, budget_before, budget_after = adaptation_turn_cfg
                env_record["eval_adaptation_turn_scope"] = [
                    int(mutation_turn),
                    int(budget_before),
                    int(budget_after),
                ]
                env_record["adaptation_turn_mutation_turn"] = int(mutation_turn)
                env_record["adaptation_turn_budget_change"] = [
                    int(budget_before),
                    int(budget_after),
                ]
            if compliance_turn_limit is not None:
                env_record["compliance_turn_limit"] = compliance_turn_limit
            if compliance_toolcall_scope:
                env_record["eval_compliance_toolcall_scope"] = list(compliance_toolcall_scope)
            if compliance_toolcall_limit is not None:
                env_record["compliance_toolcall_limit"] = compliance_toolcall_limit
            if env_toolcall_cap is not None:
                env_record["max_action_points"] = int(env_toolcall_cap)
            env_record["tag"] = rollout_state.get("tag")
            env_record["budget_turn"] = rollout_state.get("budget_turn")
            env_record["budget_token"] = rollout_state.get("budget_token")
            env_record["budget_toolcall"] = mixed_toolcall_budget

            history = rollout_state.get("history", []) or []
            reward_turns = [turn for turn in history if "reward" in turn]
            reward_turn_action_points = [
                self._resolve_actual_action_points_used(turn)
                for turn in reward_turns
            ]
            reward_turn_toolcalls = [
                self._resolve_toolcalls_used(turn)
                for turn in reward_turns
            ]
            env_record["total_turns"] = len(reward_turns)
            total_action_points_used = int(sum(reward_turn_action_points))
            env_record["total_action_points_used"] = total_action_points_used
            env_record["total_toolcalls_used"] = int(sum(reward_turn_toolcalls))
            if history:
                env_record["initial_state"] = history[0].get("state")
                env_record["final_state"] = history[-1].get("state")

            rollout_success = False
            final_reward = None
            final_reward_source = None
            final_rollout_reward = None
            final_goal_predicate_ratio_reward = None
            last_goal_predicate_ratio_reward = None
            last_goal_predicates_satisfied = None
            last_goal_predicates_total = None
            for turn_idx, turn in enumerate(reward_turns, start=1):
                turn_record = self._ensure_turn_record(env_record, turn_idx)
                turn_info = (turn.get("info", {}) or {}).copy()
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
                actual_token = max(0, int(turn.get("token_count", 0) or 0))
                actual_action_points = reward_turn_action_points[turn_idx - 1]
                toolcalls_used = reward_turn_toolcalls[turn_idx - 1]
                action_points_used_before_turn = int(sum(reward_turn_action_points[: turn_idx - 1]))
                cumulative_action_points_used = int(sum(reward_turn_action_points[:turn_idx]))
                budget_remaining_before_turn = (
                    None
                    if env_toolcall_cap is None
                    else int(env_toolcall_cap) - action_points_used_before_turn
                )
                budget_remaining_after_turn = (
                    None
                    if env_toolcall_cap is None
                    else int(env_toolcall_cap) - cumulative_action_points_used
                )
                actual_remaining_action_points_to_finish = (
                    self._clip_action_point_value(
                        sum(reward_turn_action_points[turn_idx - 1:])
                    )
                    or 0
                )
                rollout_reward = turn.get("reward")
                goal_ratio_reward = turn_info.get("goal_predicate_ratio_reward")
                if goal_ratio_reward is not None:
                    last_goal_predicate_ratio_reward = float(goal_ratio_reward)
                effective_goal_ratio_reward = last_goal_predicate_ratio_reward
                if "goal_predicates_satisfied" in turn_info:
                    last_goal_predicates_satisfied = int(
                        turn_info.get("goal_predicates_satisfied", 0) or 0
                    )
                if "goal_predicates_total" in turn_info:
                    last_goal_predicates_total = int(
                        turn_info.get("goal_predicates_total", 0) or 0
                    )

                turn_record["raw_response"] = llm_raw_response
                turn_record["parsed_response"] = "" if generation_error else turn.get("llm_response")
                turn_actions = list(turn.get("actions", []) or [])
                turn_record["actions"] = turn_actions
                turn_record["action_names"] = turn_actions
                turn_record["toolcalls_used"] = toolcalls_used
                turn_record["reward"] = (
                    effective_goal_ratio_reward
                    if effective_goal_ratio_reward is not None
                    else rollout_reward
                )
                final_rollout_reward = rollout_reward
                if effective_goal_ratio_reward is not None:
                    final_goal_predicate_ratio_reward = effective_goal_ratio_reward
                    turn_record["rollout_reward"] = rollout_reward
                    turn_record["reward_source"] = (
                        "goal_predicate_ratio"
                        if goal_ratio_reward is not None
                        else "goal_predicate_ratio_carry_forward"
                    )
                turn_record["success"] = bool(turn_info.get("success", False))
                rollout_success = rollout_success or bool(turn_record["success"])
                turn_record["actual_token"] = actual_token
                if last_goal_predicates_satisfied is not None:
                    turn_record["goal_predicates_satisfied"] = last_goal_predicates_satisfied
                if last_goal_predicates_total is not None:
                    turn_record["goal_predicates_total"] = last_goal_predicates_total
                if effective_goal_ratio_reward is not None:
                    turn_record["goal_predicate_ratio_reward"] = effective_goal_ratio_reward
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
                if mixed_toolcall_budget is not None:
                    turn_record["budget_toolcall"] = int(mixed_toolcall_budget)

                if eval_mode == "single":
                    turn_record["estimate_token"] = self._extract_token_estimate(llm_raw_response)
                elif eval_mode in {"multi", "adaptation_turn"}:
                    turn_record["estimate_token"] = self._extract_token_estimate(llm_raw_response)
                    estimate_remaining_turn = self._extract_turn_estimate(llm_raw_response)
                    actual_remaining_turn = len(reward_turns) - turn_idx + 1
                    turn_record["estimate_remaining_turn"] = estimate_remaining_turn
                    turn_record["actual_remaining_turn"] = actual_remaining_turn
                elif eval_mode == "toolcall":
                    turn_record["max_action_points"] = (
                        int(env_toolcall_cap) if env_toolcall_cap is not None else None
                    )
                    turn_record["estimate_remaining_action_points"] = (
                        self._extract_remaining_action_points_estimate(llm_raw_response)
                    )
                    turn_record["estimate_remaining_action_points_to_finish"] = (
                        turn_record["estimate_remaining_action_points"]
                    )
                    turn_record["actual_remaining_action_points"] = (
                        actual_remaining_action_points_to_finish
                    )
                    turn_record["actual_remaining_action_points_to_finish"] = (
                        actual_remaining_action_points_to_finish
                    )
                    turn_record["estimate_action_points"] = (
                        self._extract_action_points_estimate(llm_raw_response)
                    )
                    turn_record["actual_action_points"] = actual_action_points
                    turn_record["action_points_used_before_turn"] = action_points_used_before_turn
                    turn_record["cumulative_action_points_used"] = cumulative_action_points_used
                    turn_record["budget_remaining_before_turn"] = budget_remaining_before_turn
                    turn_record["budget_remaining_after_turn"] = budget_remaining_after_turn
                self._set_estimation_accuracy_fields(
                    turn_record,
                    estimate_key="estimate_token",
                    actual_key="actual_token",
                )
                self._set_estimation_accuracy_fields(
                    turn_record,
                    estimate_key="estimate_remaining_turn",
                    actual_key="actual_remaining_turn",
                )
                self._set_estimation_accuracy_fields(
                    turn_record,
                    estimate_key="estimate_action_points",
                    actual_key="actual_action_points",
                )
                self._set_estimation_accuracy_fields(
                    turn_record,
                    estimate_key="estimate_remaining_action_points",
                    actual_key="actual_remaining_action_points",
                )

                if self._eval_compliance_token_enabled():
                    compliance_token_limit = (
                        turn_record.get("compliance_token_limit")
                        if turn_record.get("compliance_token_limit") is not None
                        else env_record.get("compliance_token_limit")
                    )
                    has_answer = bool(generation_error is None and str(llm_raw_response).strip())
                    within_token_limit = (
                        None
                        if compliance_token_limit is None
                        else actual_token <= int(compliance_token_limit)
                    )
                    turn_record["compliance_token_limit"] = compliance_token_limit
                    turn_record["has_answer"] = has_answer
                    turn_record["within_token_limit"] = within_token_limit
                    turn_record["answered_within_token_limit"] = bool(
                        has_answer and within_token_limit is True
                    )
                    turn_record["token_limit_delta"] = (
                        None
                        if compliance_token_limit is None
                        else actual_token - int(compliance_token_limit)
                    )

                if self._eval_compliance_turn_enabled():
                    compliance_turn_limit = (
                        self._get_eval_compliance_turn_limit_for_env(
                            env_id=env_id,
                            group_id=None if group_id is None else int(group_id),
                            current_turn=turn_idx,
                        )
                    )
                    turn_budget_distance = (
                        None
                        if compliance_turn_limit is None
                        else int(compliance_turn_limit) - int(turn_idx)
                    )
                    within_turn_limit_so_far = (
                        None
                        if compliance_turn_limit is None
                        else int(turn_idx) <= int(compliance_turn_limit)
                    )
                    exceeded_turn_limit = (
                        None
                        if compliance_turn_limit is None
                        else int(turn_idx) > int(compliance_turn_limit)
                    )
                    turn_record["compliance_turn_limit"] = compliance_turn_limit
                    turn_record["current_turn"] = int(turn_idx)
                    turn_record["turn_budget_distance"] = turn_budget_distance
                    turn_record["within_turn_limit_so_far"] = within_turn_limit_so_far
                    turn_record["exceeded_turn_limit"] = exceeded_turn_limit
                    if compliance_turn_limit is not None:
                        turn_record["compliance_instruction"] = self._build_eval_compliance_turn_note(
                            int(compliance_turn_limit),
                            int(turn_idx),
                        )

                if self._eval_adaptation_turn_enabled():
                    adaptation_turn_limit = self._get_eval_adaptation_turn_limit(
                        current_turn=turn_idx,
                    )
                    turn_budget_distance = (
                        None
                        if adaptation_turn_limit is None
                        else int(adaptation_turn_limit) - int(turn_idx)
                    )
                    within_adaptation_turn_limit_so_far = (
                        None
                        if adaptation_turn_limit is None
                        else int(turn_idx) <= int(adaptation_turn_limit)
                    )
                    exceeded_adaptation_turn_limit = (
                        None
                        if adaptation_turn_limit is None
                        else int(turn_idx) > int(adaptation_turn_limit)
                    )
                    turn_record["adaptation_turn_limit"] = adaptation_turn_limit
                    turn_record["current_turn"] = int(turn_idx)
                    turn_record["turn_budget_distance"] = turn_budget_distance
                    turn_record["within_adaptation_turn_limit_so_far"] = (
                        within_adaptation_turn_limit_so_far
                    )
                    turn_record["exceeded_adaptation_turn_limit"] = (
                        exceeded_adaptation_turn_limit
                    )
                    if adaptation_turn_limit is not None:
                        turn_record["adaptation_instruction"] = self._build_eval_adaptation_turn_note(
                            int(adaptation_turn_limit),
                            int(turn_idx),
                        )

                if self._eval_compliance_toolcall_enabled():
                    compliance_toolcall_limit = (
                        turn_record.get("compliance_toolcall_limit")
                        if turn_record.get("compliance_toolcall_limit") is not None
                        else env_record.get("compliance_toolcall_limit")
                    )
                    remaining_before_turn = (
                        None
                        if compliance_toolcall_limit is None
                        else int(compliance_toolcall_limit) - action_points_used_before_turn
                    )
                    remaining_after_turn = (
                        None
                        if compliance_toolcall_limit is None
                        else int(compliance_toolcall_limit) - cumulative_action_points_used
                    )
                    within_toolcall_limit_so_far = (
                        None
                        if compliance_toolcall_limit is None
                        else cumulative_action_points_used <= int(compliance_toolcall_limit)
                    )
                    exceeded_toolcall_limit = (
                        None
                        if compliance_toolcall_limit is None
                        else cumulative_action_points_used > int(compliance_toolcall_limit)
                    )
                    turn_record["compliance_toolcall_limit"] = compliance_toolcall_limit
                    turn_record["action_points_used_before_turn"] = action_points_used_before_turn
                    turn_record["remaining_action_points_before_turn"] = remaining_before_turn
                    turn_record["actual_action_points"] = actual_action_points
                    turn_record["cumulative_action_points_used"] = cumulative_action_points_used
                    turn_record["remaining_action_points_after_turn"] = remaining_after_turn
                    turn_record["within_toolcall_limit_so_far"] = within_toolcall_limit_so_far
                    turn_record["exceeded_toolcall_limit"] = exceeded_toolcall_limit
                    if compliance_toolcall_limit is not None:
                        turn_record["compliance_instruction"] = self._build_eval_compliance_toolcall_note(
                            int(compliance_toolcall_limit),
                            action_points_used_before_turn,
                        )

            if final_goal_predicate_ratio_reward is not None:
                final_reward = final_goal_predicate_ratio_reward
                final_reward_source = "goal_predicate_ratio"
            else:
                final_reward = final_rollout_reward
                final_reward_source = "rollout" if final_rollout_reward is not None else None

            env_record["success"] = bool(rollout_success)
            env_record["final_reward"] = final_reward
            env_record["final_reward_source"] = final_reward_source
            env_record["final_rollout_reward"] = final_rollout_reward
            if final_goal_predicate_ratio_reward is not None:
                env_record["final_goal_predicate_ratio_reward"] = (
                    final_goal_predicate_ratio_reward
                )

            if self._eval_compliance_turn_enabled():
                env_turn_limit = self._get_eval_compliance_turn_limit_for_env(
                    env_id=env_id,
                    group_id=None if group_id is None else int(group_id),
                    current_turn=int(env_record["total_turns"]),
                )
                env_record["compliance_turn_limit"] = env_turn_limit
                within_turn_limit = (
                    None
                    if env_turn_limit is None
                    else int(env_record["total_turns"]) <= int(env_turn_limit)
                )
                env_record["within_turn_limit"] = within_turn_limit
                env_record["turn_limit_delta"] = (
                    None
                    if env_turn_limit is None
                    else int(env_record["total_turns"]) - int(env_turn_limit)
                )
                env_record["success_within_turn_limit"] = bool(
                    rollout_success and within_turn_limit is True
                )

            if self._eval_adaptation_turn_enabled():
                adaptation_turn_limit = self._get_eval_adaptation_turn_limit(
                    current_turn=int(env_record["total_turns"]),
                )
                env_record["adaptation_turn_limit"] = adaptation_turn_limit
                within_adaptation_turn_limit = (
                    None
                    if adaptation_turn_limit is None
                    else int(env_record["total_turns"]) <= int(adaptation_turn_limit)
                )
                env_record["within_adaptation_turn_limit"] = within_adaptation_turn_limit
                env_record["adaptation_turn_limit_delta"] = (
                    None
                    if adaptation_turn_limit is None
                    else int(env_record["total_turns"]) - int(adaptation_turn_limit)
                )
                env_record["success_within_adaptation_turn_limit"] = bool(
                    rollout_success and within_adaptation_turn_limit is True
                )

            if self._eval_compliance_toolcall_enabled():
                env_toolcall_limit = env_record.get("compliance_toolcall_limit")
                within_toolcall_limit = (
                    None
                    if env_toolcall_limit is None
                    else int(total_action_points_used) <= int(env_toolcall_limit)
                )
                env_record["within_toolcall_limit"] = within_toolcall_limit
                env_record["toolcall_limit_delta"] = (
                    None
                    if env_toolcall_limit is None
                    else int(total_action_points_used) - int(env_toolcall_limit)
                )
                env_record["success_within_toolcall_limit"] = bool(
                    rollout_success and within_toolcall_limit is True
                )

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

    def _set_estimation_accuracy_fields(
        self,
        turn_record: Dict[str, Any],
        *,
        estimate_key: str,
        actual_key: str,
    ) -> None:
        diff_key = f"{estimate_key}_diff"
        abs_error_key = f"{estimate_key}_abs_error"
        exact_match_key = f"{estimate_key}_exact_match"
        estimate_value = turn_record.get(estimate_key)
        actual_value = turn_record.get(actual_key)
        if estimate_value is None or actual_value is None:
            turn_record[diff_key] = None
            turn_record[abs_error_key] = None
            turn_record[exact_match_key] = None
            return
        try:
            diff = int(estimate_value) - int(actual_value)
        except (TypeError, ValueError):
            turn_record[diff_key] = None
            turn_record[abs_error_key] = None
            turn_record[exact_match_key] = None
            return
        turn_record[diff_key] = diff
        turn_record[abs_error_key] = abs(diff)
        turn_record[exact_match_key] = diff == 0

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
                "mode": self._get_dialogue_log_mode(),
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
        if eval_mode in {"multi", "adaptation_turn"}:
            return [
                "How many turns do you expect are still needed to finish?",
                "How many tokens do you expect to use in this turn?",
            ]
        if eval_mode == "toolcall":
            return [
                "How many additional action points do you expect are still needed to finish from this turn onward?",
                "How many action points do you expect to use in this turn?",
            ]
        if self._eval_compliance_enabled():
            return []
        return []

    def _extract_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    def _get_eval_compliance_limit_for_env(
        self,
        scope: List[int],
        env_id: int,
        group_id: Optional[int] = None,
    ) -> Optional[int]:
        if not scope:
            return None
        effective_group_size = 1
        es_cfg = getattr(self.config, "es_manager", None)
        val_cfg = getattr(es_cfg, "val", None) if es_cfg is not None else None
        if val_cfg is not None and getattr(val_cfg, "group_size", None) is not None:
            effective_group_size = max(1, int(val_cfg.group_size))

        scope_len = len(scope)
        if effective_group_size >= scope_len and effective_group_size % scope_len == 0:
            base_group_size = max(1, effective_group_size // scope_len)
            if group_id is None:
                position_within_group = int(env_id) % effective_group_size
            else:
                position_within_group = int(env_id) - int(group_id) * effective_group_size
            compliance_idx = min(
                scope_len - 1,
                max(0, int(position_within_group) // base_group_size),
            )
            return int(scope[compliance_idx])

        return int(scope[int(env_id) % scope_len])

    def _get_eval_compliance_token_limit_for_env(
        self,
        env_id: int,
        group_id: Optional[int] = None,
    ) -> Optional[int]:
        scope = self._get_eval_compliance_token_scope()
        return self._get_eval_compliance_limit_for_env(scope, env_id, group_id)

    def _get_eval_compliance_turn_limit_for_env(
        self,
        env_id: int,
        group_id: Optional[int] = None,
        current_turn: Optional[int] = None,
    ) -> Optional[int]:
        mutation_cfg = self._get_eval_compliance_turn_mutation_config()
        if mutation_cfg is not None:
            mutation_turn, budget_change = mutation_cfg
            resolved_turn = int(self.turn_idx + 1) if current_turn is None else int(current_turn)
            if resolved_turn <= int(mutation_turn):
                return int(budget_change[0])
            return int(budget_change[1])
        scope = self._get_eval_compliance_turn_scope()
        return self._get_eval_compliance_limit_for_env(scope, env_id, group_id)

    def _get_eval_compliance_toolcall_limit_for_env(
        self,
        env_id: int,
        group_id: Optional[int] = None,
    ) -> Optional[int]:
        scope = self._get_eval_compliance_toolcall_scope()
        return self._get_eval_compliance_limit_for_env(scope, env_id, group_id)

    def _get_toolcall_action_point_cap(self) -> Optional[int]:
        return self._toolcall_action_point_cap

    def _clip_action_point_value(self, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        capped_value = max(0, int(value))
        max_action_points = self._get_toolcall_action_point_cap()
        if max_action_points is not None:
            capped_value = min(capped_value, int(max_action_points))
        return capped_value

    def _extract_action_points_estimate(self, text: str) -> Optional[int]:
        value = self._extract_tagged_int(text, "action_points_estimation")
        return self._clip_action_point_value(value)

    def _extract_remaining_action_points_estimate(self, text: str) -> Optional[int]:
        value = self._extract_tagged_int(text, "remaining_action_points_estimation")
        return self._clip_action_point_value(value)

    def _resolve_actual_action_points_used(self, turn: Dict[str, Any]) -> int:
        value = turn.get("action_points_used")
        if value is None:
            value = turn.get("info", {}).get("budget_action_cost", 0)
        try:
            return self._clip_action_point_value(int(value)) or 0
        except (TypeError, ValueError):
            return 0

    def _resolve_toolcalls_used(self, turn: Dict[str, Any]) -> int:
        actions = turn.get("actions")
        if isinstance(actions, (list, tuple)):
            return len(actions)
        if actions is None:
            return 0
        return 1 if str(actions).strip() else 0

    def _record_estimation_inputs(
        self,
        non_tensor_batch: Dict[str, Any],
        messages_list: List[List[Dict[str, Any]]],
        texts: List[str],
        generation_suffix: str,
    ) -> None:
        if not self._dialogue_logging_enabled():
            return

        env_ids = self._to_list(non_tensor_batch.get("env_ids"))
        group_ids = self._to_list(non_tensor_batch.get("group_ids"))
        uid_list = self._to_list(non_tensor_batch.get("uid"))
        action_points_used_so_far = self._to_list(non_tensor_batch.get("action_points_used_so_far"))
        budget_toolcalls = self._to_list(non_tensor_batch.get("budget_toolcalls"))
        turn_idx = int(self.turn_idx + 1)
        toolcall_cap = self._get_toolcall_action_point_cap()
        adaptation_turn_cfg = self._get_eval_adaptation_turn_config()

        for idx, (env_id, messages, prompt_text) in enumerate(zip(env_ids, messages_list, texts)):
            group_id = group_ids[idx] if idx < len(group_ids) else None
            uid = uid_list[idx] if idx < len(uid_list) else None
            compliance_token_limit = self._get_eval_compliance_token_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
            )
            compliance_turn_limit = self._get_eval_compliance_turn_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
                current_turn=turn_idx,
            )
            compliance_toolcall_limit = self._get_eval_compliance_toolcall_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
            )
            action_points_used_before_turn = (
                int(action_points_used_so_far[idx])
                if idx < len(action_points_used_so_far) and action_points_used_so_far[idx] is not None
                else 0
            )
            budget_toolcall = (
                int(budget_toolcalls[idx])
                if idx < len(budget_toolcalls) and budget_toolcalls[idx] is not None
                else None
            )
            effective_toolcall_cap = (
                int(budget_toolcall)
                if budget_toolcall is not None
                else (int(toolcall_cap) if toolcall_cap is not None else None)
            )
            env_record = self._ensure_env_record(env_id, group_id=group_id, uid=uid)
            if budget_toolcall is not None:
                env_record["budget_toolcall"] = int(budget_toolcall)
            if effective_toolcall_cap is not None:
                env_record["max_action_points"] = int(effective_toolcall_cap)
            if compliance_token_limit is not None:
                env_record["compliance_token_limit"] = compliance_token_limit
                env_record["eval_compliance_token_scope"] = list(
                    self._get_eval_compliance_token_scope()
                )
            if compliance_turn_limit is not None:
                env_record["compliance_turn_limit"] = compliance_turn_limit
                env_record["eval_compliance_turn_scope"] = list(
                    self._get_eval_compliance_turn_scope()
                )
            turn_mutation_cfg = self._get_eval_compliance_turn_mutation_config()
            if turn_mutation_cfg is not None:
                mutation_turn, budget_change = turn_mutation_cfg
                env_record["eval_compliance_turn_mutation_turn"] = int(mutation_turn)
                env_record["eval_compliance_turn_budget_change"] = list(budget_change)
            if compliance_toolcall_limit is not None:
                env_record["compliance_toolcall_limit"] = compliance_toolcall_limit
                env_record["eval_compliance_toolcall_scope"] = list(
                    self._get_eval_compliance_toolcall_scope()
                )
            if adaptation_turn_cfg is not None:
                mutation_turn, budget_before, budget_after = adaptation_turn_cfg
                adaptation_turn_limit = self._get_eval_adaptation_turn_limit(current_turn=turn_idx)
                env_record["eval_adaptation_turn_scope"] = [
                    int(mutation_turn),
                    int(budget_before),
                    int(budget_after),
                ]
                env_record["adaptation_turn_mutation_turn"] = int(mutation_turn)
                env_record["adaptation_turn_budget_change"] = [
                    int(budget_before),
                    int(budget_after),
                ]
                env_record["adaptation_turn_limit"] = adaptation_turn_limit
            turn_record = self._ensure_turn_record(env_record, turn_idx)
            turn_record["mode"] = self.mode
            turn_record["questions"] = self._build_eval_estimation_questions()
            turn_record["messages"] = copy.deepcopy(messages)
            turn_record["user_prompt"] = self._extract_last_user_message(messages)
            turn_record["prompt_text"] = prompt_text
            turn_record["generation_suffix"] = generation_suffix
            if compliance_token_limit is not None:
                turn_record["compliance_token_limit"] = compliance_token_limit
                turn_record["compliance_instruction"] = (
                    f"You must complete your answer within {compliance_token_limit} tokens. You must write something in think part. Please allocate your reasoning tokens carefully to improve the precision of your response."
                )
            if compliance_turn_limit is not None:
                turn_record["compliance_turn_limit"] = compliance_turn_limit
                turn_record["current_turn"] = turn_idx
                turn_record["turn_budget_distance"] = int(compliance_turn_limit) - int(turn_idx)
                turn_record["within_turn_limit_so_far"] = int(turn_idx) <= int(compliance_turn_limit)
                turn_record["exceeded_turn_limit"] = int(turn_idx) > int(compliance_turn_limit)
                turn_record["compliance_instruction"] = self._build_eval_compliance_turn_note(
                    int(compliance_turn_limit),
                    int(turn_idx),
                )
            if compliance_toolcall_limit is not None:
                remaining_action_points_before_turn = int(compliance_toolcall_limit) - int(
                    action_points_used_before_turn
                )
                turn_record["compliance_toolcall_limit"] = compliance_toolcall_limit
                turn_record["action_points_used_before_turn"] = action_points_used_before_turn
                turn_record["remaining_action_points_before_turn"] = remaining_action_points_before_turn
                turn_record["within_toolcall_limit_so_far"] = (
                    int(action_points_used_before_turn) <= int(compliance_toolcall_limit)
                )
                turn_record["exceeded_toolcall_limit"] = (
                    int(action_points_used_before_turn) > int(compliance_toolcall_limit)
                )
                turn_record["compliance_instruction"] = self._build_eval_compliance_toolcall_note(
                    int(compliance_toolcall_limit),
                    int(action_points_used_before_turn),
                )
            if adaptation_turn_cfg is not None:
                adaptation_turn_limit = self._get_eval_adaptation_turn_limit(current_turn=turn_idx)
                turn_record["adaptation_turn_limit"] = adaptation_turn_limit
                turn_record["current_turn"] = turn_idx
                turn_record["turn_budget_distance"] = (
                    None
                    if adaptation_turn_limit is None
                    else int(adaptation_turn_limit) - int(turn_idx)
                )
                turn_record["within_adaptation_turn_limit_so_far"] = (
                    None
                    if adaptation_turn_limit is None
                    else int(turn_idx) <= int(adaptation_turn_limit)
                )
                turn_record["exceeded_adaptation_turn_limit"] = (
                    None
                    if adaptation_turn_limit is None
                    else int(turn_idx) > int(adaptation_turn_limit)
                )
                if adaptation_turn_limit is not None:
                    turn_record["adaptation_instruction"] = self._build_eval_adaptation_turn_note(
                        int(adaptation_turn_limit),
                        int(turn_idx),
                    )
            if budget_toolcall is not None:
                turn_record["budget_toolcall"] = int(budget_toolcall)
            if effective_toolcall_cap is not None:
                turn_record["max_action_points"] = int(effective_toolcall_cap)
            self._pending_turn_records[(int(env_id), turn_idx)] = turn_record

    def _decorate_response_for_estimation(self, raw_response: str) -> str:
        eval_mode = self._get_eval_estimation_mode()
        if eval_mode in {"single", "multi", "toolcall", "adaptation_turn"}:
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
        if not self._dialogue_logging_enabled():
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
        eval_mode = self._get_eval_estimation_mode()

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
            turn_record["actual_token"] = 0 if response_error is not None else self._count_tokens(raw_generation)
            turn_record["generation_error"] = None if response_error is None else response_error.get("error")
            turn_record["generation_error_type"] = None if response_error is None else response_error.get("error_type")
            turn_record["generation_error_code"] = None if response_error is None else response_error.get("error_code")
            turn_record["generation_error_status_code"] = None if response_error is None else response_error.get("status_code")
            turn_record["generation_retryable"] = None if response_error is None else response_error.get("retryable")
            turn_record["generation_success"] = response_error is None
            if eval_mode == "single":
                turn_record["estimate_token"] = (
                    None if response_error is not None else self._extract_token_estimate(full_response)
                )
            elif eval_mode in {"multi", "adaptation_turn"}:
                turn_record["estimate_token"] = (
                    None if response_error is not None else self._extract_token_estimate(full_response)
                )
                estimate_remaining_turn = (
                    None if response_error is not None else self._extract_turn_estimate(full_response)
                )
                turn_record["estimate_remaining_turn"] = estimate_remaining_turn
            elif eval_mode == "toolcall":
                turn_record["estimate_action_points"] = (
                    None if response_error is not None else self._extract_action_points_estimate(full_response)
                )
                estimate_remaining_action_points = (
                    None
                    if response_error is not None
                    else self._extract_remaining_action_points_estimate(full_response)
                )
                turn_record["estimate_remaining_action_points"] = estimate_remaining_action_points
                turn_record["estimate_remaining_action_points_to_finish"] = (
                    estimate_remaining_action_points
                )
            self._set_estimation_accuracy_fields(
                turn_record,
                estimate_key="estimate_token",
                actual_key="actual_token",
            )
            self._set_estimation_accuracy_fields(
                turn_record,
                estimate_key="estimate_remaining_turn",
                actual_key="actual_remaining_turn",
            )
            self._set_estimation_accuracy_fields(
                turn_record,
                estimate_key="estimate_action_points",
                actual_key="actual_action_points",
            )
            self._set_estimation_accuracy_fields(
                turn_record,
                estimate_key="estimate_remaining_action_points",
                actual_key="actual_remaining_action_points",
            )

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

    def _build_mixed_toolcall_budget_note(
        self,
        budget_toolcall: int,
        action_points_used_so_far: int,
    ) -> str:
        remaining_action_points = int(budget_toolcall) - int(action_points_used_so_far)
        if remaining_action_points > 0:
            status = (
                f"There are {remaining_action_points} action point(s) left within this budget."
            )
        elif remaining_action_points == 0:
            status = (
                "There are 0 action points left within this budget. Any additional tool call will exceed it."
            )
        else:
            status = (
                f"You have already exceeded this budget by {abs(remaining_action_points)} action point(s)."
            )
        return (
            "[Toolcall Budget Guidance] "
            f"You are expected to finish within {int(budget_toolcall)} action points total. "
            f"Action points used so far: {int(action_points_used_so_far)}. "
            f"{status} You will be penalized if you finish after this budget."
        )

    def _inject_mixed_toolcall_budget_prompt(
        self,
        messages_list: List[List[Dict]],
        budget_toolcalls: List[Optional[int]],
        action_points_used_so_far: Optional[List[Any]] = None,
    ) -> None:
        if action_points_used_so_far is None:
            action_points_used_so_far = [0] * len(messages_list)
        for idx, (messages, budget_toolcall) in enumerate(zip(messages_list, budget_toolcalls)):
            if budget_toolcall is None:
                continue
            used_so_far = (
                int(action_points_used_so_far[idx])
                if idx < len(action_points_used_so_far) and action_points_used_so_far[idx] is not None
                else 0
            )
            note = self._build_mixed_toolcall_budget_note(
                int(budget_toolcall),
                int(used_so_far),
            )
            self._append_note_to_last_user_message(
                [messages],
                note=note,
                marker="[Toolcall Budget Guidance]",
            )

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

    def _build_eval_adaptation_turn_note(
        self,
        adaptation_turn_limit: int,
        current_turn: int,
    ) -> str:
        if self._get_eval_adaptation_turn_config() is None:
            return ""
        turn_budget_distance = int(adaptation_turn_limit) - int(current_turn)
        max_turn = int(
            self.state.get(
                "max_turn",
                getattr(self.config.agent_proxy, "max_turn", 1),
            )
            or 1
        )
        if turn_budget_distance > 0:
            status = f"You are {turn_budget_distance} turn(s) away from the suggested budget."
        elif turn_budget_distance == 0:
            status = "This is the last suggested budgeted turn."
        else:
            status = (
                f"You have already exceeded the suggested budget by "
                f"{abs(turn_budget_distance)} turn(s). You may continue until max_turn {int(max_turn)}."
            )
        return (
            "[Turn Budget Adaptation] "
            f"Suggested budget turn: {int(adaptation_turn_limit)}. "
            f"Current turn: {int(current_turn)}. "
            f"Turns used so far: {int(current_turn)}. "
            f"Hard stop: max_turn {int(max_turn)}. "
            "This is guidance only, not a hard cutoff. "
            f"{status}"
        )

    def _inject_eval_adaptation_turn_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        current_turn = int(self.turn_idx + 1)
        adaptation_turn_limit = self._get_eval_adaptation_turn_limit(current_turn=current_turn)
        if adaptation_turn_limit is None:
            return
        note = self._build_eval_adaptation_turn_note(
            int(adaptation_turn_limit),
            current_turn,
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="[Turn Budget Adaptation]",
        )

    def _inject_eval_estimation_toolcall_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        max_action_points = self._get_toolcall_action_point_cap()
        clipped_range_note = ""
        if max_action_points is not None:
            clipped_range_note = (
                f" Each integer must be between 0 and {int(max_action_points)} inclusive."
            )
        note = (
            "Before your final answer, first answer these two questions in order:\n"
            "1. How many additional action points do you expect are still needed to finish from this turn onward? "
            "Count the action points you expect to spend starting in this turn and continuing until the task is complete. "
            "This is not the same as the current budget remaining shown in the state.\n"
            "2. How many action points do you expect to use in this turn?\n"
            "Fill <remaining_action_points_estimation> and <action_points_estimation> with integers "
            "in the required response format."
            f"{clipped_range_note}"
        )
        self._append_note_to_last_user_message(
            messages_list,
            note=note,
            marker="How many additional action points do you expect are still needed to finish from this turn onward?",
        )

    def _build_eval_compliance_turn_note(
        self,
        compliance_turn_limit: int,
        current_turn: int,
    ) -> str:
        turn_budget_distance = int(compliance_turn_limit) - int(current_turn)
        mutation_cfg = self._get_eval_compliance_turn_mutation_config()
        if turn_budget_distance > 0:
            status = f"You are {turn_budget_distance} turn(s) away from this budget."
        elif turn_budget_distance == 0:
            status = "This is the last budgeted turn."
        else:
            status = (
                f"You have already exceeded this budget by "
                f"{abs(turn_budget_distance)} turn(s)."
            )
        mutation_note = ""
        if mutation_cfg is not None:
            mutation_turn, budget_change = mutation_cfg
            mutation_note = (
                f" Mutation turn: {int(mutation_turn)}. "
                f"Budget schedule: {int(budget_change[0])} before or at the mutation turn, "
                f"{int(budget_change[1])} after the mutation turn."
            )
        return (
            "[Turn Budget Compliance] "
            f"Budget turn: {int(compliance_turn_limit)}. "
            f"Current turn: {int(current_turn)}."
            f"{mutation_note} "
            f"{status}"
        )

    def _build_eval_compliance_toolcall_note(
        self,
        compliance_toolcall_limit: int,
        action_points_used_so_far: int,
    ) -> str:
        remaining_action_points = int(compliance_toolcall_limit) - int(action_points_used_so_far)
        if remaining_action_points > 0:
            status = (
                f"You can still use {remaining_action_points} action point(s) within this budget."
            )
        elif remaining_action_points == 0:
            status = (
                "You have 0 action points remaining in this budget. Any additional tool call will exceed it."
            )
        else:
            status = (
                f"You have already exceeded this budget by {abs(remaining_action_points)} action point(s)."
            )
        return (
            "[Toolcall Budget Compliance] "
            f"You must finish this task within {int(compliance_toolcall_limit)} action points. "
            f"Action points used so far: {int(action_points_used_so_far)}. "
            f"{status}"
        )

    def _inject_eval_compliance_token_prompt(
        self,
        messages_list: List[List[Dict]],
        env_ids: List[Any],
        group_ids: Optional[List[Any]] = None,
    ) -> None:
        if group_ids is None:
            group_ids = [None] * len(messages_list)
        for messages, env_id, group_id in zip(messages_list, env_ids, group_ids):
            token_limit = self._get_eval_compliance_token_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
            )
            if token_limit is None:
                continue
            note = f"You must finish your answer in {token_limit} tokens. Include sufficient context in the answer section to improve accuracy."
            self._append_note_to_last_user_message(
                [messages],
                note=note,
                marker=note,
            )

    def _inject_eval_compliance_turn_prompt(
        self,
        messages_list: List[List[Dict]],
        env_ids: List[Any],
        group_ids: Optional[List[Any]] = None,
    ) -> None:
        if group_ids is None:
            group_ids = [None] * len(messages_list)
        current_turn = int(self.turn_idx + 1)
        for messages, env_id, group_id in zip(messages_list, env_ids, group_ids):
            turn_limit = self._get_eval_compliance_turn_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
                current_turn=current_turn,
            )
            if turn_limit is None:
                continue
            note = self._build_eval_compliance_turn_note(
                int(turn_limit),
                current_turn,
            )
            self._append_note_to_last_user_message(
                [messages],
                note=note,
                marker="[Turn Budget Compliance]",
            )

    def _inject_eval_compliance_toolcall_prompt(
        self,
        messages_list: List[List[Dict]],
        env_ids: List[Any],
        group_ids: Optional[List[Any]] = None,
        action_points_used_so_far: Optional[List[Any]] = None,
    ) -> None:
        if group_ids is None:
            group_ids = [None] * len(messages_list)
        if action_points_used_so_far is None:
            action_points_used_so_far = [0] * len(messages_list)
        for idx, (messages, env_id, group_id) in enumerate(zip(messages_list, env_ids, group_ids)):
            toolcall_limit = self._get_eval_compliance_toolcall_limit_for_env(
                env_id=int(env_id),
                group_id=None if group_id is None else int(group_id),
            )
            if toolcall_limit is None:
                continue
            used_so_far = (
                int(action_points_used_so_far[idx])
                if idx < len(action_points_used_so_far) and action_points_used_so_far[idx] is not None
                else 0
            )
            note = self._build_eval_compliance_toolcall_note(
                int(toolcall_limit),
                int(used_so_far),
            )
            self._append_note_to_last_user_message(
                [messages],
                note=note,
                marker="[Toolcall Budget Compliance]",
            )

    def _inject_token_estimation_prompt(
        self,
        messages_list: List[List[Dict]],
    ) -> None:
        note = (
            "Before your reasoning, first output token estimation as the first segment of your response. "
            "Write only one integer wrapped by <token_estimation> and </token_estimation> "
            "before your normal answer tags. "
            "Then continue with normal format. Example: "
            "<token_estimation>256</token_estimation><think>...</think><answer>...</answer>."
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
            "Example: <turn_estimation>3</turn_estimation><think>...</think><answer>...</answer>."
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
        env_ids = self._to_list(lm_inputs.non_tensor_batch.get("env_ids"))
        group_ids = self._to_list(lm_inputs.non_tensor_batch.get("group_ids"))
        action_points_used_so_far = self._to_list(
            lm_inputs.non_tensor_batch.get("action_points_used_so_far")
        )
        budget_turns = lm_inputs.non_tensor_batch.get("budget_turns")
        budget_toolcalls = lm_inputs.non_tensor_batch.get("budget_toolcalls")
        if budget_turns is not None:
            budget_turns = (
                budget_turns.tolist()
                if hasattr(budget_turns, "tolist")
                else list(budget_turns)
            )
            self._inject_budget_prompt(messages_list, budget_turns)
        if budget_toolcalls is not None:
            budget_toolcalls = (
                budget_toolcalls.tolist()
                if hasattr(budget_toolcalls, "tolist")
                else list(budget_toolcalls)
            )
            self._inject_mixed_toolcall_budget_prompt(
                messages_list,
                budget_toolcalls,
                action_points_used_so_far,
            )

        eval_mode = self._get_eval_estimation_mode()
        if eval_mode == "single":
            self._inject_eval_estimation_single_prompt(messages_list)
        elif eval_mode == "multi":
            self._inject_eval_estimation_multi_prompt(messages_list)
        elif eval_mode == "adaptation_turn":
            self._inject_eval_estimation_multi_prompt(messages_list)
            self._inject_eval_adaptation_turn_prompt(messages_list)
        elif eval_mode == "toolcall":
            self._inject_eval_estimation_toolcall_prompt(messages_list)
        else:
            if bool(getattr(self.config.agent_proxy, "token_estimation", False)):
                self._inject_token_estimation_prompt(messages_list)
            self._inject_turn_estimation_prompt(messages_list)
        if self._eval_compliance_token_enabled():
            self._inject_eval_compliance_token_prompt(messages_list, env_ids, group_ids)
        if self._eval_compliance_turn_enabled():
            self._inject_eval_compliance_turn_prompt(messages_list, env_ids, group_ids)
        if self._eval_compliance_toolcall_enabled():
            self._inject_eval_compliance_toolcall_prompt(
                messages_list,
                env_ids,
                group_ids,
                action_points_used_so_far,
            )

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
