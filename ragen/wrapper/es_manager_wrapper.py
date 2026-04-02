from typing import Any, Dict, List, Optional, Tuple

import random
import re


class EsManagerWrapper:
    """
    Intercept and optionally modify env_outputs before returning to the agent.

    Override `reassemble_env_outputs` to customize the states.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(self.config.agent_proxy, "enable_es_wrapper", True)
        self.debug_reward_flow = bool(getattr(self.config.agent_proxy, "debug_reward_flow", False))
        self.turn_idx = 0
        self.mode: Optional[str] = None
        self.state: Dict[str, Any] = {}
        self._printed_budget_range = False

    def _debug_reward(self, message: str) -> None:
        if self.debug_reward_flow:
            print(f"[reward-debug][es_wrapper][{self.mode}][turn={self.turn_idx}] {message}")

    def set_state(self, turn_idx: int, mode: Optional[str] = None, **kwargs: Any) -> None:
        self.turn_idx = turn_idx
        if mode is not None:
            self.mode = mode
        self.state.update(kwargs)

    def reassemble_env_outputs(self, env_outputs: List[Dict]) -> List[Dict]:
        return env_outputs

    @staticmethod
    def _cal_budget_reward(reward, current_turn, budget_turn, tau=1.0, use_hard=False):
        import math

        # Hard cutoff version
        if use_hard:
            if current_turn > budget_turn:
                return 0
            # reward is unchanged before B, zero after B
            return reward

        # Soft scaling function S(t; B, tau)
        def scaling(t, B, tau=tau):
            # Flexible budget-controlled tanh reward
            return 0.5 * (1 - math.tanh((t - B) / tau))

        scaled_reward = reward * scaling(current_turn, budget_turn)
        return scaled_reward

    def _apply_budget_curve(self, env_output: Dict) -> None:
        budget_turn = env_output.get("budget_turn")
        if budget_turn is None:
            self._debug_reward(f"turn_curve_skip env_id={env_output.get('env_id')} reason=no_budget_turn")
            return
        budget_cfg = getattr(self.config.agent_proxy, "mixed_turn_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            self._debug_reward(f"turn_curve_skip env_id={env_output.get('env_id')} reason=turn_budget_disabled")
            return
        reward_curve = getattr(budget_cfg, "reward_curve", None)
        tau = getattr(reward_curve, "tau", 1.0) if reward_curve is not None else 1.0
        use_hard = getattr(reward_curve, "use_hard", False) if reward_curve is not None else False
        self._debug_reward(
            f"turn_curve_start env_id={env_output.get('env_id')}, budget_turn={budget_turn}, tau={tau}, use_hard={use_hard}"
        )

        history = env_output.get("history", [])
        turn_idx = 0
        for turn in history:
            if "reward" not in turn:
                continue
            turn_idx += 1
            if "origin_reward" not in turn:
                turn["origin_reward"] = float(turn["reward"])
            scaled = self._cal_budget_reward(
                turn["origin_reward"],
                current_turn=turn_idx,
                budget_turn=int(budget_turn),
                tau=tau,
                use_hard=use_hard,
            )
            self._debug_reward(
                f"turn_curve env_id={env_output.get('env_id')}, turn_idx={turn_idx}, success={turn.get('info', {}).get('success', None)}, "
                f"origin_reward={float(turn['origin_reward']):.4f}, scaled_reward={float(scaled):.4f}"
            )
            turn["reward"] = float(scaled)
            turn["true_reward"] = float(turn["origin_reward"])

    def _apply_token_budget_curve(self, env_output: Dict) -> None:
        budget_token = env_output.get("budget_token")
        if budget_token is None:
            self._debug_reward(f"token_curve_skip env_id={env_output.get('env_id')} reason=no_budget_token")
            return
        budget_cfg = getattr(self.config.agent_proxy, "mixed_token_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            self._debug_reward(f"token_curve_skip env_id={env_output.get('env_id')} reason=token_budget_disabled")
            return
        reward_curve = getattr(budget_cfg, "reward_curve", None)
        tau = getattr(reward_curve, "tau", 1.0) if reward_curve is not None else 1.0
        use_hard = getattr(reward_curve, "use_hard", False) if reward_curve is not None else False
        self._debug_reward(
            f"token_curve_start env_id={env_output.get('env_id')}, budget_token={budget_token}, tau={tau}, use_hard={use_hard}"
        )

        history = env_output.get("history", [])
        token_used = 0
        for turn in history:
            if "reward" not in turn:
                continue
            token_used += int(turn.get("token_count", 0))
            if "origin_reward" not in turn:
                turn["origin_reward"] = float(turn["reward"])
            scaled = self._cal_budget_reward(
                turn["origin_reward"],
                current_turn=token_used,
                budget_turn=int(budget_token),
                tau=tau,
                use_hard=use_hard,
            )

            self._debug_reward(
                f"token_curve env_id={env_output.get('env_id')}, token_count={turn.get('token_count', 0)}, "
                f"token_used={token_used}, budget_token={budget_token}, success={turn.get('info', {}).get('success', None)}, "
                f"origin_reward={float(turn['origin_reward']):.4f}, scaled_reward={float(scaled):.4f}"
            )
            turn["reward"] = float(scaled)
            turn["true_reward"] = float(turn["origin_reward"])

    def _resolve_budget_range(self) -> Optional[Tuple[int, int]]:
        budget_cfg = getattr(self.config.agent_proxy, "mixed_turn_budget", None)
        if budget_cfg is None:
            return None
        if not getattr(budget_cfg, "enabled", False):
            return None
        if not getattr(budget_cfg, "mixed_budget", False):
            return None
        raw_range = getattr(budget_cfg, "mixed_budget_range", None)
        if raw_range is None:
            max_turn = int(getattr(self.config.agent_proxy, "max_turn", 0) or 0)
            return (0, max_turn)
        else:
            low, high = int(raw_range[0]), int(raw_range[1])
            if low > high:
                low, high = high, low
            return (low, high)
        return None

    @staticmethod
    def _extract_tagged_int_from_text(text: str, tag_name: str) -> Optional[int]:
        if not text:
            return None
        matches = list(
            re.finditer(
                rf"<{tag_name}>\s*([+-]?\d+)\s*</{tag_name}>",
                text,
                re.IGNORECASE | re.DOTALL,
            )
        )
        if not matches:
            return None

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
            for match in matches:
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
                for candidate in matches:
                    if budget_open_pos <= candidate.start() < trailing_boundary_pos:
                        match = candidate
                        break

        if match is None:
            boundary_match = re.search(r"<budget-thinking>|<think>|<answer>", text, re.IGNORECASE)
            boundary_pos = boundary_match.start() if boundary_match else len(text)
            for candidate in matches:
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

    @staticmethod
    def _extract_estimate_token_value(turn: Dict) -> Optional[int]:
        text = str(turn.get("llm_response", "") or "")
        if not text:
            text = str(turn.get("llm_raw_response", "") or "")
        return EsManagerWrapper._extract_tagged_int_from_text(text, "token_estimation")

    @staticmethod
    def _extract_turn_estimate_value(turn: Dict) -> Optional[int]:
        text = str(turn.get("llm_response", "") or "")
        if not text:
            text = str(turn.get("llm_raw_response", "") or "")
        return EsManagerWrapper._extract_tagged_int_from_text(text, "turn_estimation")

    def _apply_token_estimation_adjustment(self, env_output: Dict) -> None:
        token_estimation_enabled = bool(getattr(self.config.agent_proxy, "token_estimation", False))
        benchmark_factors = env_output.get("benchmark_factors") or self.compute_benchmark_factors(env_output)
        compliance_factor = float(benchmark_factors.get("compliance_factor", 1.0))
        reward_cfg = getattr(self.config.agent_proxy, "token_estimation_reward", None)
        estimate_bonus_coef = (
            float(getattr(reward_cfg, "estimate_bonus", 0.2)) if reward_cfg is not None else 0.2
        )

        history = env_output.get("history", [])
        rollout_turn_idx = 0
        for turn in history:
            if "reward" not in turn:
                continue
            rollout_turn_idx += 1
            success = bool(turn.get("info", {}).get("success", False))
            actual_tokens = max(1, int(turn.get("token_count", 0) or 0))
            estimate_tokens = self._extract_estimate_token_value(turn)
            turn["estimate_token"] = estimate_tokens
            turn["actual_token"] = actual_tokens
            turn["reward_before_estimation_adjustment"] = float(turn["reward"])

            if not success:
                # Hard rule: unsuccessful turns always have zero reward.
                turn["estimate_success"] = bool(estimate_tokens is not None)
                turn["estimate_token_diff"] = None
                turn["estimate_token_error_ratio"] = None
                turn["estimate_token_accuracy"] = 0.0
                turn["reward"] = 0.0
                self._debug_reward(
                    f"token_estimation_turn env_id={env_output.get('env_id')}, rollout_turn={rollout_turn_idx}, "
                    f"success=False -> reward=0.0000"
                )
                continue

            # success=True path:
            # base reward is compliance-aware, and estimate_bonus is added only when tag is present.
            base_reward = 1.0 * compliance_factor
            has_estimate_tag = token_estimation_enabled and (estimate_tokens is not None)
            estimate_bonus = estimate_bonus_coef if has_estimate_tag else 0.0
            adjusted_reward = base_reward + estimate_bonus
            turn["reward"] = float(adjusted_reward)

            estimation_info = self.compute_estimation_factor(
                estimate_tokens=estimate_tokens,
                actual_tokens=actual_tokens,
            )

            if estimate_tokens is None:
                turn["estimate_success"] = False
                turn["estimate_token_diff"] = None
                turn["estimate_token_error_ratio"] = None
                turn["estimate_token_accuracy"] = None
            else:
                turn["estimate_success"] = True
                turn["estimate_token_diff"] = estimation_info["diff"]
                turn["estimate_token_error_ratio"] = estimation_info["error_ratio"]
                turn["estimate_token_accuracy"] = estimation_info["factor"]

            self._debug_reward(
                f"token_estimation_turn env_id={env_output.get('env_id')}, rollout_turn={rollout_turn_idx}, "
                f"estimate_tokens={estimate_tokens}, actual_tokens={actual_tokens}, "
                f"compliance_factor={compliance_factor:.4f}, estimate_bonus={estimate_bonus:.4f}, "
                f"reward_before={float(turn['reward_before_estimation_adjustment']):.4f}, "
                f"reward_after={float(adjusted_reward):.4f}"
            )

    def _apply_turn_level_compliance_adjustment(self, env_output: Dict) -> None:
        factors = env_output.get("benchmark_factors") or {}
        if "compliance_factor" not in factors:
            return
        compliance_factor = float(factors.get("compliance_factor", 1.0))
        history = env_output.get("history", [])
        rollout_turn_idx = 0
        for turn in history:
            if "reward" not in turn:
                continue
            rollout_turn_idx += 1
            if "origin_reward" not in turn:
                turn["origin_reward"] = float(turn["reward"])
            origin_reward = float(turn["origin_reward"])
            adjusted_reward = origin_reward + compliance_factor
            turn["reward_before_compliance_adjustment"] = float(turn["reward"])
            turn["reward"] = float(adjusted_reward)
            self._debug_reward(
                f"turn_compliance_turn env_id={env_output.get('env_id')}, rollout_turn={rollout_turn_idx}, "
                f"origin_reward={origin_reward:.4f}, compliance_factor={compliance_factor:.4f}, "
                f"reward_after={adjusted_reward:.4f}"
            )

    def compute_compliance_factor(self, budget_used: float, low_bound: int, high_bound: int, penalty_coef: float):
        if budget_used < low_bound:
            out_of_bound = float(low_bound - budget_used)
        elif high_bound >= 0 and budget_used > high_bound:
            out_of_bound = float(budget_used - high_bound)
        else:
            out_of_bound = 0.0
        penalty = penalty_coef * out_of_bound
        compliance_factor = max(0.0, 1.0 - penalty)

        compliance_is_out_of_bound = budget_used < low_bound or budget_used > high_bound
        compliance_out_of_bound_range = 0
        if budget_used < low_bound:
            compliance_out_of_bound_range = low_bound - budget_used
        elif high_bound >= 0 and budget_used > high_bound:
            compliance_out_of_bound_range = budget_used - high_bound
        return (compliance_factor, compliance_is_out_of_bound, compliance_out_of_bound_range)

    def compute_estimation_factor(
        self, estimate_tokens: Optional[int], actual_tokens: Optional[int]
    ) -> Dict[str, Optional[float]]:
        if estimate_tokens is None or actual_tokens is None:
            return {"factor": 0.0, "diff": None, "error_ratio": None}

        actual = max(1, int(actual_tokens))
        estimate = max(0, int(estimate_tokens))
        diff = int(estimate - actual)
        error_ratio = float(abs(diff) / float(actual))
        factor = float(max(0.0, 1.0 - error_ratio))
        return {"factor": factor, "diff": diff, "error_ratio": error_ratio}

    @staticmethod
    def _normalize_bounds(low_bound: int, high_bound: int) -> Tuple[int, int]:
        if high_bound >= 0 and low_bound > high_bound:
            return high_bound, low_bound
        return low_bound, high_bound

    def compute_benchmark_factors(self, env_output: Dict) -> Dict[str, float]:
        is_last_turn = 1.0 if bool(env_output.get("turn_done", False)) else 0.0
        if not is_last_turn:
            return {}
        cfg = getattr(self.config.agent_proxy, "benchmark_factors", None)
        enabled = bool(getattr(cfg, "enabled", False)) if cfg is not None else False
        if not enabled:
            return {}

        mode = str(getattr(cfg, "mode", "turn")).strip().lower()
        if mode not in {"token", "turn", "tool-call"}:
            raise ValueError(f"Invalid benchmark_factors mode: {mode}")

        low_bound = int(getattr(cfg, "low_bound", 0))
        high_bound = int(getattr(cfg, "high_bound", -1))
        penalty_coef = float(getattr(cfg, "penalty_coef", 0.0))
        enable_adaptation = bool(getattr(cfg, "enable_adaptation", False))
        low_bound, high_bound = self._normalize_bounds(low_bound, high_bound)

        history = env_output.get("history", [])
        reward_turns = [turn for turn in history if "reward" in turn]
        turn_used = len(reward_turns)
        token_used = sum(int(turn.get("token_count", 0) or 0) for turn in reward_turns)
        # tool_call_used = sum(len(turn.get("actions", []) or []) for turn in reward_turns)
        

        if mode == "token":
            used = float(token_used)
        # elif mode == "tool-call":
        #     used = float(tool_call_used)
        else:
            used = float(turn_used)

        if enable_adaptation:
            happen_low = int(getattr(cfg, "adaptation_happened_low_bound", 0))
            happen_high = int(getattr(cfg, "adaptation_happened_high_bound", 0))
            happen_low, happen_high = self._normalize_bounds(happen_low, happen_high)
            happen_low = max(1, happen_low)
            happen_high = max(1, happen_high)
            adaptation_turn = random.randint(happen_low, happen_high)
            adaptation_turn = min(max(1, adaptation_turn), max(1, turn_used))

            adaptation_low = int(getattr(cfg, "adaptation_low_bound", low_bound))
            adaptation_high = int(getattr(cfg, "adaptation_high_bound", high_bound))
            adaptation_low, adaptation_high = self._normalize_bounds(adaptation_low, adaptation_high)

            turn_estimates: List[Optional[int]] = [self._extract_turn_estimate_value(turn) for turn in reward_turns]
            true_turn_used = max(1, int(turn_used))
            pre_compliance, post_compliance = [], []
            pre_estimation, post_estimation = [], []
            pre_count, post_count = 0, 0
            pre_missing, post_missing = 0, 0

            for turn_idx_1based in range(1, true_turn_used + 1):
                if turn_idx_1based <= adaptation_turn:
                    c_factor, _, _ = self.compute_compliance_factor(
                        budget_used=float(turn_idx_1based),
                        low_bound=low_bound,
                        high_bound=high_bound,
                        penalty_coef=penalty_coef,
                    )
                else:
                    c_factor, _, _ = self.compute_compliance_factor(
                        budget_used=float(turn_idx_1based),
                        low_bound=adaptation_low,
                        high_bound=adaptation_high,
                        penalty_coef=penalty_coef,
                    )

                estimate_value = turn_estimates[turn_idx_1based - 1]
                e_factor = float(self.compute_estimation_factor(estimate_value, true_turn_used)["factor"] or 0.0)

                if turn_idx_1based <= adaptation_turn:
                    pre_compliance.append(float(c_factor))
                    pre_estimation.append(float(e_factor))
                    pre_count += 1
                    if estimate_value is None:
                        pre_missing += 1
                else:
                    post_compliance.append(float(c_factor))
                    post_estimation.append(float(e_factor))
                    post_count += 1
                    if estimate_value is None:
                        post_missing += 1

            def _safe_mean(values: List[float]) -> float:
                return float(sum(values) / len(values)) if values else 0.0

            factors = {
                "mode": mode,
                "used": float(used),
                "enable_adaptation": 1.0,
                "adaptation_turn": float(adaptation_turn),
                "pre_low_bound": float(low_bound),
                "pre_high_bound": float(high_bound),
                "post_low_bound": float(adaptation_low),
                "post_high_bound": float(adaptation_high),
                "pre_segment_turn_count": float(pre_count),
                "post_segment_turn_count": float(post_count),
                "pre_compliance_factor_mean": _safe_mean(pre_compliance),
                "post_compliance_factor_mean": _safe_mean(post_compliance),
                "pre_estimation_factor_mean": _safe_mean(pre_estimation),
                "post_estimation_factor_mean": _safe_mean(post_estimation),
                "pre_estimate_missing_count": float(pre_missing),
                "post_estimate_missing_count": float(post_missing),
            }
            env_output["benchmark_factors"] = factors
            return factors

        compliance_factor, compliance_is_out_of_bound, compliance_out_of_bound_range = self.compute_compliance_factor(
            budget_used=used,
            low_bound=low_bound,
            high_bound=high_bound,
            penalty_coef=penalty_coef,
        )

        turn_estimates = []
        for turn in reward_turns:
            estimate = self._extract_turn_estimate_value(turn)
            if estimate is not None:
                turn_estimates.append(float(estimate))
        turn_estimate_count = len(turn_estimates)
        turn_estimate_missing_count = len(reward_turns) - turn_estimate_count
        true_turn_used = float(turn_used)
        if turn_estimates:
            turn_estimate_mean = float(sum(turn_estimates) / float(turn_estimate_count))
            turn_estimate_mean_abs_error = float(
                sum(abs(estimate - true_turn_used) for estimate in turn_estimates) / float(turn_estimate_count)
            )
            turn_estimate_mean_signed_error = float(
                sum((estimate - true_turn_used) for estimate in turn_estimates) / float(turn_estimate_count)
            )
        else:
            turn_estimate_mean = 0.0
            turn_estimate_mean_abs_error = 0.0
            turn_estimate_mean_signed_error = 0.0
        
        factors = {
            "mode": mode,
            "used": float(used),
            "low_bound": float(low_bound),
            "high_bound": float(high_bound),
            "compliance_is_out_of_bound": float(compliance_is_out_of_bound),
            "compliance_out_of_bound_range": float(compliance_out_of_bound_range),
            "compliance_factor": float(compliance_factor),
            "true_turn_used": float(true_turn_used),
            "turn_estimate_count": float(turn_estimate_count),
            "turn_estimate_missing_count": float(turn_estimate_missing_count),
            "turn_estimate_mean": float(turn_estimate_mean),
            "turn_estimate_mean_abs_error": float(turn_estimate_mean_abs_error),
            "turn_estimate_mean_signed_error": float(turn_estimate_mean_signed_error),
        }
        env_output["benchmark_factors"] = factors
        return factors

    def _update_reward_sums(self, env_output: Dict) -> None:
        history = env_output.get("history", [])
        reward_sum = sum(float(turn.get("reward", 0.0)) for turn in history)
        env_output["reward_sum"] = float(reward_sum)
        self._debug_reward(
            f"reward_sum env_id={env_output.get('env_id')}, reward_sum={float(reward_sum):.4f}, "
            f"n_reward_turns={sum(1 for turn in history if 'reward' in turn)}"
        )

        budget_cfg = getattr(self.config.agent_proxy, "mixed_turn_budget", None)
        budget_enabled = bool(getattr(budget_cfg, "enabled", False)) if budget_cfg else False
        has_origin = any("origin_reward" in turn for turn in history)
        if budget_enabled or has_origin:
            origin_sum = sum(float(turn.get("origin_reward", 0.0)) for turn in history)
            env_output["origin_reward_sum"] = float(origin_sum)
            self._debug_reward(
                f"origin_reward_sum env_id={env_output.get('env_id')}, origin_reward_sum={float(origin_sum):.4f}"
            )

    def intercept(self, env_outputs: List[Dict]) -> List[Dict]:

        if not self.enabled:
            self._debug_reward("intercept_bypass reason=enable_es_wrapper=False")
            return env_outputs
        self._debug_reward(f"intercept_inputs={len(env_outputs)}")
        for env_output in env_outputs:
            history = env_output.get("history", [])
            last_turn = history[-2] if len(history) >= 2 else {}
            self._debug_reward(
                f"pre_env env_id={env_output.get('env_id')}, budget_turn={env_output.get('budget_turn')}, "
                f"budget_token={env_output.get('budget_token')}, last_reward={last_turn.get('reward', None)}, "
                f"last_success={last_turn.get('info', {}).get('success', None)}"
            )
            if env_output.get("budget_turn") is not None:
                self._apply_budget_curve(env_output)
            if env_output.get("budget_token") is not None:
                self._apply_token_budget_curve(env_output)
            self.compute_benchmark_factors(env_output)
            benchmark_cfg = getattr(self.config.agent_proxy, "benchmark_factors", None)
            benchmark_enabled = bool(getattr(benchmark_cfg, "enabled", False)) if benchmark_cfg is not None else False
            benchmark_mode = str(getattr(benchmark_cfg, "mode", "")).strip().lower() if benchmark_cfg is not None else ""
            if benchmark_enabled and benchmark_mode == "turn":
                self._apply_turn_level_compliance_adjustment(env_output)
            else:
                self._apply_token_estimation_adjustment(env_output)
            self._update_reward_sums(env_output)
            self._debug_reward(
                f"post_env env_id={env_output.get('env_id')}, reward_sum={env_output.get('reward_sum', None)}, "
                f"origin_reward_sum={env_output.get('origin_reward_sum', None)}"
            )
        return self.reassemble_env_outputs(env_outputs)
