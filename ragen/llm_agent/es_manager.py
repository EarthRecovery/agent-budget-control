"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np
import logging

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.wrapper.es_manager_wrapper import EsManagerWrapper
from ragen.utils import register_resolvers
register_resolvers()

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.debug_reward_flow = bool(getattr(self.sys_config.agent_proxy, "debug_reward_flow", False))
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        self.start_group_index = int(getattr(self.config, "start_group_index", 0) or 0)
        if self.start_group_index < 0:
            raise ValueError(f"start_group_index must be >= 0, got {self.start_group_index}")
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self._init_envs()
        self.es_wrapper = EsManagerWrapper(config)
        self._turn_idx = 0
        self.rollout_cache = None
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._executors_shutdown = False
        self._register_parallel_executors()
        atexit.register(self._shutdown_executors)

    def _debug_reward(self, message: str) -> None:
        if self.debug_reward_flow:
            print(f"[reward-debug][es_manager][{self.mode}][turn={self._turn_idx}] {message}")

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                raw_env_cfg = cfg_template.env_config or {}
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                if raw_env_cfg.get("action_lookup") is None and "action_lookup" in raw_env_cfg:
                    env_config.action_lookup = None
                budget_turn = self._init_budget_turn()
                budget_token = self._init_budget_token()
                budget_toolcall = self._init_budget_toolcall()
                budget_toolcall = self._apply_mixed_toolcall_budget(
                    tag=tag,
                    env_class=env_class,
                    env_config=env_config,
                    budget_toolcall=budget_toolcall,
                )
                env_obj = REGISTERED_ENVS[env_class](env_config)
                parallel_friendly = bool(getattr(cfg_template, "parallel_friendly", False))
                max_workers = int(getattr(cfg_template, "max_workers", 1) or 1)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj,
                        'parallel_friendly': parallel_friendly, 'max_workers': max_workers, 'budget_turn': budget_turn,
                        'budget_token': budget_token, 'budget_toolcall': budget_toolcall}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def _init_budget_turn(self) -> Optional[int]:
        budget_cfg = getattr(self.sys_config.agent_proxy, "mixed_turn_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            return None
        if not getattr(budget_cfg, "mixed_budget", False):
            return None
        raw_range = getattr(budget_cfg, "mixed_budget_range", None)
        if raw_range is None:
            max_turn = int(getattr(self.sys_config.agent_proxy, "max_turn", 0) or 0)
            low, high = 0, max_turn
        else:
            low, high = int(raw_range[0]), int(raw_range[1])
            if low > high:
                low, high = high, low
        return random.randint(low, high)

    def _init_budget_token(self) -> Optional[int]:
        budget_cfg = getattr(self.sys_config.agent_proxy, "mixed_token_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            return None
        if not getattr(budget_cfg, "mixed_budget", False):
            return None
        raw_range = getattr(budget_cfg, "mixed_budget_range", None)
        if raw_range is None:
            rollout_cfg = getattr(self.sys_config, "actor_rollout_ref", None)
            rollout = getattr(rollout_cfg, "rollout", None) if rollout_cfg is not None else None
            max_tokens = int(getattr(rollout, "response_length", 0) or 0)
            low, high = 0, max_tokens
        else:
            low, high = int(raw_range[0]), int(raw_range[1])
            if low > high:
                low, high = high, low
        return random.randint(low, high)

    def _init_budget_toolcall(self) -> Optional[int]:
        budget_cfg = getattr(self.sys_config.agent_proxy, "mixed_toolcall_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            return None
        if not getattr(budget_cfg, "mixed_budget", False):
            return None
        raw_range = getattr(budget_cfg, "mixed_budget_range", None)
        if raw_range is None:
            low, high = 0, 0
        else:
            low, high = int(raw_range[0]), int(raw_range[1])
            if low > high:
                low, high = high, low
        return random.randint(low, high)

    def _apply_mixed_toolcall_budget(
        self,
        *,
        tag: str,
        env_class: str,
        env_config: Any,
        budget_toolcall: Optional[int],
    ) -> Optional[int]:
        budget_cfg = getattr(self.sys_config.agent_proxy, "mixed_toolcall_budget", None)
        if budget_cfg is None or not getattr(budget_cfg, "enabled", False):
            return None
        if env_class != "robotouille":
            raise ValueError(
                "agent_proxy.mixed_toolcall_budget can only be enabled for robotouille environments. "
                f"Got tag={tag!r}, env_type={env_class!r}."
            )
        if budget_toolcall is None:
            return None
        env_config.enable_action_budget = True
        env_config.max_action_points = int(budget_toolcall)
        return int(budget_toolcall)

    def _register_parallel_executors(self):
        tag_seen: Dict[str, dict] = {}
        for entry in self.envs:
            tag = entry["tag"]
            cfg = {
                "parallel_friendly": entry.get("parallel_friendly", False),
                "max_workers": entry.get("max_workers", 1),
            }
            if tag in tag_seen:
                assert tag_seen[tag] == cfg, f"Inconsistent config for tag {tag}: {tag_seen[tag]} vs {cfg}"
            else:
                tag_seen[tag] = cfg

        for tag, cfg in tag_seen.items():
            parallel_friendly = cfg.get('parallel_friendly', False)
            max_workers = cfg.get('max_workers', 1)
            if parallel_friendly and max_workers > 1:
                self._executors[tag] = ThreadPoolExecutor(max_workers=max_workers)

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [
            {
                "env_id": entry['env_id'],
                "history": [],
                "group_id": entry['group_id'],
                "tag": entry['tag'],
                "penalty": 0,
                "budget_turn": entry.get("budget_turn"),
                "budget_token": entry.get("budget_token"),
                "budget_toolcall": entry.get("budget_toolcall"),
                "turn_done": False,
            }
            for entry in envs
        ]

        # reset all environments
        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.start_group_index + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                base_seed = 123 if self.base_seed is None else self.base_seed
                seed = base_seed + self.start_group_index
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - (self.base_seed + self.start_group_index) + 1
        seeds = _expand_seed(seed)

        def _reset_single(entry, single_seed):
            entry['env'].reset(seed=single_seed, mode=self.mode)
            entry['status'] = EnvStatus(seed=single_seed)
            return entry['env_id'], self._handle_mm_state(entry['env'].render())

        reset_results = {}
        tag2entries: Dict[str, List[tuple]] = {}
        for single_seed, entry in zip(seeds, envs):
            tag2entries.setdefault(entry['tag'], []).append((entry, single_seed))

        for tag, items in tag2entries.items():
            parallel_friendly = items[0][0].get('parallel_friendly', False)
            max_workers = items[0][0].get('max_workers', 1)
            executor = self._executors.get(tag) if parallel_friendly and max_workers > 1 else None
            if executor is None or len(items) == 1:
                for entry, single_seed in items:
                    env_id, next_state = _reset_single(entry, single_seed)
                    reset_results[env_id] = next_state
            else:
                future_map = {
                    executor.submit(_reset_single, entry, single_seed): entry
                    for entry, single_seed in items
                }
                for future, entry in future_map.items():
                    env_id, next_state = future.result()
                    reset_results[env_id] = next_state

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = reset_results[env['env_id']]
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        self._turn_idx = 0
        self.es_wrapper.set_state(
            turn_idx=self._turn_idx,
            mode=self.mode,
            n_inputs=len(rollout_cache),
            is_reset=True,
        )
        rollout_cache = self.es_wrapper.intercept(rollout_cache)
        self._turn_idx = 1
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            action_points_used = 0
            budget_max = None
            budget_remaining = None
            budget_enabled = False
            for action in actions:
                _, reward, done, info = env.step(action)
                self._debug_reward(
                    f"env.step action={action!r}, reward={float(reward):.4f}, done={bool(done)}, success={info.get('success', None)}"
                )
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if info.get("budget_enabled", False):
                    budget_enabled = True
                    if info.get("budget_max") is not None:
                        budget_max = int(info.get("budget_max"))
                    if info.get("budget_remaining") is not None:
                        budget_remaining = int(info.get("budget_remaining"))
                    if info.get("action_is_effective", True):
                        action_points_used += max(0, int(info.get("budget_action_cost", 0) or 0))
                if done:
                    turn_done = True
                    break
            return (
                acc_reward,
                turn_info,
                turn_done,
                executed_actions,
                action_points_used,
                budget_enabled,
                budget_max,
                budget_remaining,
            )

        def _log_env_state(
            status,
            history,
            cur_obs,
            max_actions_per_traj,
            executed_actions,
            all_actions,
            acc_reward,
            turn_done,
            turn_info,
            env_input,
            action_points_used,
            budget_enabled,
            budget_max,
            budget_remaining,
        ):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response'],
                'token_count': env_input.get('response_tokens', 0),
                'llm_error': env_input.get('llm_error'),
                'llm_error_type': env_input.get('llm_error_type'),
                'llm_error_code': env_input.get('llm_error_code'),
                'llm_error_status_code': env_input.get('llm_error_status_code'),
                'llm_error_retryable': env_input.get('llm_error_retryable'),
                'action_points_used': int(action_points_used),
                'budget_enabled': bool(budget_enabled),
                'budget_max': budget_max,
                'budget_remaining': budget_remaining,
            })
            self._debug_reward(
                f"history_update env_id={env_input['env_id']}, executed_actions={executed_actions}, "
                f"acc_reward={float(acc_reward):.4f}, turn_done={turn_done}, success={turn_info.get('success', None)}, "
                f"token_count={env_input.get('response_tokens', 0)}, action_points_used={int(action_points_used)}"
            )
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []
        wrapper_inputs = []
        active_env_ids = []

        def _process_env_input(env_input: Dict) -> Dict:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            (
                acc_reward,
                turn_info,
                turn_done,
                executed_actions,
                action_points_used,
                budget_enabled,
                budget_max,
                budget_remaining,
            ) = _execute_actions(env, valid_actions[:actions_left_before])
            no_manager_action = len(valid_actions) == 0
            penalty_delta = 0.0
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                penalty_delta = self.sys_config.es_manager.format_penalty
            if no_manager_action:
                turn_info = dict(turn_info)
                turn_info['manager_invalid_action'] = True

            status, history = _log_env_state(
                entry['status'],
                self.rollout_cache[env_id]['history'],
                entry['env'].render(),
                entry['max_actions_per_traj'],
                executed_actions,
                valid_actions,
                acc_reward,
                turn_done,
                turn_info,
                env_input,
                action_points_used,
                budget_enabled,
                budget_max,
                budget_remaining,
            )
            if no_manager_action and history:
                history[-1]['manager_invalid_action'] = True
            if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
                status.truncated = True
                status.terminated = True
                turn_done = True
            self._debug_reward(
                f"post_process env_id={env_id}, input_actions={env_input['actions']}, valid_actions={valid_actions}, "
                f"no_manager_action={no_manager_action}, penalty_delta={penalty_delta}, turn_done={turn_done}, "
                f"status_terminated={status.terminated}, status_truncated={status.truncated}"
            )

            return {
                'env_id': env_id,
                'status': status,
                'history': history,
                'turn_done': turn_done,
                'penalty_delta': penalty_delta,
            }

        results: List[Optional[Dict]] = [None] * len(all_env_inputs)
        tag2items: Dict[str, List[tuple]] = {}
        for idx, env_input in enumerate(all_env_inputs):
            entry = envs[env_input['env_id']]
            tag2items.setdefault(entry['tag'], []).append((idx, env_input))

        for tag, items in tag2items.items():
            sample_entry = envs[items[0][1]['env_id']]
            parallel_friendly = sample_entry.get('parallel_friendly', False)
            max_workers = sample_entry.get('max_workers', 1)
            executor = self._executors.get(tag) if parallel_friendly and max_workers > 1 else None
            if executor is None or len(items) == 1:
                for idx, env_input in items:
                    results[idx] = _process_env_input(env_input)
            else:
                futures = {executor.submit(_process_env_input, env_input): idx for idx, env_input in items}
                for future, idx in futures.items():
                    results[idx] = future.result()

        for result in results:
            if result is None:
                continue
            env_id = result['env_id']
            entry = envs[env_id]
            if result['penalty_delta']:
                self.rollout_cache[env_id]["penalty"] += result['penalty_delta']
            self.rollout_cache[env_id]['history'] = result['history']
            self.rollout_cache[env_id]['turn_done'] = bool(result['turn_done'])
            entry['status'] = result['status']
            last_turn = result['history'][-2] if len(result['history']) >= 2 else {}
            self._debug_reward(
                f"cache_commit env_id={env_id}, turn_done={result['turn_done']}, "
                f"last_reward={float(last_turn.get('reward', 0.0)):.4f}, "
                f"last_success={last_turn.get('info', {}).get('success', None)}"
            )
            wrapper_inputs.append(self.rollout_cache[env_id])
            if not result['turn_done']:
                active_env_ids.append(env_id)
            else:
                self._debug_reward(f"filtered_from_env_outputs env_id={env_id} because turn_done=True")

        self.es_wrapper.set_state(
            turn_idx=self._turn_idx,
            mode=self.mode,
            n_inputs=len(all_env_inputs),
        )
        self._debug_reward(
            f"before_wrapper active_env_outputs={len(active_env_ids)}, wrapper_inputs={len(wrapper_inputs)}"
        )
        wrapped_outputs = self.es_wrapper.intercept(wrapper_inputs)
        wrapped_by_env_id = {output["env_id"]: output for output in wrapped_outputs}
        for env_id, wrapped in wrapped_by_env_id.items():
            self.rollout_cache[env_id] = wrapped
        env_outputs = [wrapped_by_env_id[env_id] for env_id in active_env_ids if env_id in wrapped_by_env_id]
        self._debug_reward(f"after_wrapper active_env_outputs={len(env_outputs)}")
        self._turn_idx += 1

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            estimation_turns = [
                turn for turn in cache.get('history', [])
                if 'estimate_success' in turn
            ]
            if estimation_turns:
                success_flags = [1.0 if bool(turn.get('estimate_success', False)) else 0.0 for turn in estimation_turns]
                missing_flags = [1.0 if (not bool(turn.get('estimate_success', False))) else 0.0 for turn in estimation_turns]
                abs_token_errors = [
                    float(abs(turn.get('estimate_token_diff', 0.0)))
                    for turn in estimation_turns
                    if turn.get('estimate_success', False) and turn.get('estimate_token_diff') is not None
                ]
                env_metric['token_estimation_success_rate'] = float(np.mean(success_flags))
                env_metric['token_estimation_missing_tag_rate'] = float(np.mean(missing_flags))
                mean_abs_error = float(np.mean(abs_token_errors)) if abs_token_errors else 0.0
                env_metric['token_estimation_mean_abs_error'] = mean_abs_error
                env_metric['token_estimation_mean_abs_error_ratio'] = mean_abs_error

            try:
                import numpy as _np
                if hasattr(entry['env'], 'grid'):
                    env_metric['max_tile'] = int(_np.max(entry['env'].grid))
            except Exception:
                pass  

            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    try:
                        custom_metric[k].append(float(v))
                    except (ValueError, TypeError):
                        logging.warning(
                            "Skipping non-numeric metric '%s' with value %r for env %s.",
                            k, v, entry['tag']
                        )
            try:
                if 'raw_reward' in custom_metric:
                    env_metric['episodic_return'] = float(np.sum(custom_metric['raw_reward']))
            except Exception:
                pass    

            for k, v in custom_metric.items():
                # TODO: Move TURN_LVL_METRICS into the environment
                if "webshop" not in cache['tag'].lower() or ("webshop" in cache['tag'].lower() and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation
                else:
                    env_metric['traj_sum/' + k] = np.sum(v)
            try:
                if 'score' in custom_metric and len(custom_metric['score']) > 0:
                    env_metric['final_score'] = float(custom_metric['score'][-1])
            except Exception:
                pass

            if "reward_sum" in cache:
                env_metric["reward_sum"] = float(cache["reward_sum"])
            if "origin_reward_sum" in cache:
                env_metric["origin_reward_sum"] = float(cache["origin_reward_sum"])
            benchmark_factors = cache.get("benchmark_factors", {})
            for key, value in benchmark_factors.items():
                if isinstance(value, (int, float, bool, np.number)):
                    env_metric[f"benchmark/{key}"] = float(value)
            if self.debug_reward_flow:
                tracked_rewards = [float(turn.get("reward", 0.0)) for turn in cache.get("history", []) if "reward" in turn]
                tracked_success = [turn.get("info", {}).get("success", None) for turn in cache.get("history", []) if "reward" in turn]
                self._debug_reward(
                    f"rollout_state env_id={entry['env_id']}, rewards={tracked_rewards}, success_flags={tracked_success}, "
                    f"metric_success={env_metric.get('success', None)}"
                )

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        # calculate pass@k where k is the group size
        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        for (tag, gid), succ_list in group_success.items():
            pass_success = float(any(succ_list))
            for entry, cache in zip(envs, rollout_cache):
                if entry['tag'] == tag and entry['group_id'] == gid:
                    cache['metrics'][f"{tag}/pass@{self.group_size}"] = pass_success
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        elif action_lookup == {}:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()
        self._shutdown_executors()

    def _shutdown_executors(self):
        if getattr(self, "_executors_shutdown", False):
            return
        self._executors_shutdown = True
        executors = getattr(self, "_executors", None)
        if not executors:
            return
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        self._shutdown_executors()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
