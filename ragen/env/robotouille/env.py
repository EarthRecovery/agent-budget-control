import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from ragen.env.base import BaseDiscreteActionEnv
from ragen.utils import all_seed

from .config import RobotouilleEnvConfig


def _ensure_robotouille_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    robotouille_root = os.path.join(repo_root, "external", "robotouille")
    if robotouille_root not in sys.path:
        sys.path.insert(0, robotouille_root)


class RobotouilleEnv(BaseDiscreteActionEnv):
    FINISH_ACTION = "Finish task"

    def __init__(self, config: RobotouilleEnvConfig = None):
        self.config = config or RobotouilleEnvConfig()
        self._env = None
        self._last_obs = ""
        self._last_valid_actions: List[str] = []
        self._last_action_names: Dict[str, str] = {}
        self._last_goal_count = 0
        self._best_goal_count = 0
        self._action_count = 0
        self._test_reward_idx = 0
        self._budget_remaining: Optional[int] = None
        self._action_costs = self._load_action_costs()
        BaseDiscreteActionEnv.__init__(self)

    def _load_action_costs(self) -> Dict[str, int]:
        cost_path = os.path.join(os.path.dirname(__file__), "action_costs.json")
        with open(cost_path, "r", encoding="utf-8") as cost_file:
            raw_costs = json.load(cost_file)

        action_costs: Dict[str, int] = {}
        for action_name, cost in raw_costs.items():
            normalized_name = str(action_name).strip().lower()
            cost_value = int(cost)
            if cost_value < 0:
                raise ValueError(f"Action cost for '{action_name}' must be non-negative.")
            action_costs[normalized_name] = cost_value
        return action_costs

    def _reset_budget(self) -> None:
        if self.config.enable_action_budget:
            self._budget_remaining = int(self.config.max_action_points)
        else:
            self._budget_remaining = None

    def _format_budget_status(self) -> str:
        if not self.config.enable_action_budget:
            return ""
        remaining = 0 if self._budget_remaining is None else int(self._budget_remaining)
        return (
            f"Action Budget: enabled\n"
            f"Action Points Remaining: {remaining}/{int(self.config.max_action_points)}"
        )

    def _format_valid_action_costs(self) -> str:
        if not self.config.enable_action_budget or not self._last_valid_actions:
            return ""

        lines = ["Valid Action Point Costs:"]
        for action in self._last_valid_actions:
            action_name = self._get_action_name(action)
            action_cost = self._get_action_cost(action_name)
            point_label = "point" if action_cost == 1 else "points"
            lines.append(f"- {action}: {action_cost} action {point_label}")
        return "\n".join(lines)

    def _attach_budget_status(self, obs: str) -> str:
        if not self.config.enable_action_budget:
            return obs
        budget_status = self._format_budget_status()
        action_costs = self._format_valid_action_costs()
        sections = [section for section in (obs, budget_status, action_costs) if section]
        return "\n\n".join(sections)

    def _get_action_name(self, chosen_action: str) -> str:
        mapped_name = self._last_action_names.get(chosen_action)
        if mapped_name:
            return mapped_name
        action_prefix = chosen_action.split(" ", 1)[0].strip().lower()
        return action_prefix.replace("'", "")

    def _get_action_cost(self, action_name: str) -> int:
        return int(self._action_costs.get(action_name, 1))

    def _build_budget_info(
        self,
        *,
        action_name: Optional[str] = None,
        action_cost: int = 0,
        budget_exhausted: bool = False,
    ) -> Dict:
        if not self.config.enable_action_budget:
            return {
                "budget_enabled": False,
                "budget_max": None,
                "budget_remaining": None,
                "budget_action_name": action_name,
                "budget_action_cost": action_cost,
                "budget_exhausted": False,
            }
        return {
            "budget_enabled": True,
            "budget_max": int(self.config.max_action_points),
            "budget_remaining": 0 if self._budget_remaining is None else int(self._budget_remaining),
            "budget_action_name": action_name,
            "budget_action_cost": int(action_cost),
            "budget_exhausted": bool(budget_exhausted),
        }

    def _build_goal_progress_info(self) -> Dict:
        satisfied = int(self._count_goal_predicates())
        total = int(self._count_total_goal_predicates())
        ratio = 0.0 if total <= 0 else float(satisfied) / float(total)
        return {
            "goal_predicates_satisfied": satisfied,
            "goal_predicates_total": total,
            "goal_predicate_ratio_reward": float(ratio),
        }

    def _init_env(self):
        _ensure_robotouille_on_path()
        from robotouille.robotouille_env import create_robotouille_env

        self._env = create_robotouille_env(
            self.config.env_name,
            self.config.seed,
            self.config.noisy_randomization,
            enable_rendering=False,
        )

    def _extract_valid_actions(self):
        if self._env is None:
            self._last_valid_actions = [self.FINISH_ACTION]
            self._last_action_names = {self.FINISH_ACTION: "finish"}
            return
        valid_actions, str_valid_actions = self._env.current_state.get_valid_actions_and_str()
        if len(str_valid_actions) > self.config.max_action_space:
            valid_actions = valid_actions[: self.config.max_action_space]
            str_valid_actions = str_valid_actions[: self.config.max_action_space]
        self._last_valid_actions = list(str_valid_actions)
        self._last_action_names = {
            action_str: action_def.name.strip().lower()
            for (action_def, _), action_str in zip(valid_actions, str_valid_actions)
        }
        if self.FINISH_ACTION not in self._last_valid_actions:
            self._last_valid_actions.append(self.FINISH_ACTION)
        self._last_action_names[self.FINISH_ACTION] = "finish"

    def _is_finish_action(self, action: Optional[str]) -> bool:
        return isinstance(action, str) and action.strip() == self.FINISH_ACTION

    def _is_goal_satisfied(self) -> bool:
        total = int(self._count_total_goal_predicates())
        if total <= 0:
            return False
        return int(self._count_goal_predicates()) >= total

    def reset(self, seed=None, mode=None):
        if seed is not None:
            self.config.seed = seed
        if self._env is None:
            self._init_env()
        with all_seed(self.config.seed):
            obs, _ = self._env.reset()
        self._last_obs = obs
        self._extract_valid_actions()
        self._last_goal_count = self._count_goal_predicates()
        self._best_goal_count = self._last_goal_count
        total_goal_count = self._count_total_goal_predicates()
        print(
            f"[DEBUG] Initial goal predicates satisfied: {self._last_goal_count}",
            flush=True,
        )
        print(f"[DEBUG] total goal number: {total_goal_count}", flush=True)
        self._action_count = 0
        self._test_reward_idx = 0
        self._reset_budget()
        return self.render()

    def step(self, action) -> Tuple[str, float, bool, Dict]:
        if self._env is None:
            self._init_env()

        action_is_valid = False
        chosen_action = None
        if isinstance(action, int) and 0 <= action < len(self._last_valid_actions):
            chosen_action = self._last_valid_actions[action]
            action_is_valid = True
        elif isinstance(action, str):
            stripped_action = action.strip()
            if stripped_action in self._last_valid_actions:
                chosen_action = stripped_action
                action_is_valid = True

        if chosen_action is None:
            self._last_obs = self._last_obs or ""
            print("INVALID action | reward=0.0", flush=True)
            info = {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
            }
            info.update(self._build_budget_info())
            info.update(self._build_goal_progress_info())
            return self.render(), 0.0, False, info

        action_name = self._get_action_name(chosen_action)
        action_cost = self._get_action_cost(action_name)

        if self._is_finish_action(chosen_action):
            self._action_count += 1
            success = self._is_goal_satisfied()
            info = {
                "action_is_valid": True,
                "action_is_effective": True,
                "success": success,
                "terminated_by_agent": True,
            }
            info.update(
                self._build_budget_info(
                    action_name=action_name,
                    action_cost=action_cost,
                    budget_exhausted=False,
                )
            )
            info.update(self._build_goal_progress_info())
            print(
                f"Action {self._action_count}: {chosen_action} | reward=0.0 | "
                f"action_cost={action_cost} | budget_remaining={info.get('budget_remaining')} | "
                f"success={success}",
                flush=True,
            )
            return self.render(), 0.0, True, info

        if self.config.enable_action_budget:
            remaining = 0 if self._budget_remaining is None else int(self._budget_remaining)
            if remaining < action_cost:
                print(
                    f"BUDGET exhausted before action: {chosen_action} | "
                    f"remaining={remaining}, required={action_cost}",
                    flush=True,
                )
                info = {
                    "action_is_valid": True,
                    "action_is_effective": False,
                    "success": False,
                }
                info.update(
                    self._build_budget_info(
                        action_name=action_name,
                        action_cost=action_cost,
                        budget_exhausted=True,
                    )
                )
                info.update(self._build_goal_progress_info())
                return self.render(), 0.0, True, info

        prev_best_goal_count = self._best_goal_count
        self._action_count += 1
        obs, _, success_done, info = self._env.step(chosen_action)
        self._last_obs = obs
        if self.config.enable_action_budget and self._budget_remaining is not None:
            self._budget_remaining = max(0, self._budget_remaining - action_cost)
        self._extract_valid_actions()
        self._last_goal_count = self._count_goal_predicates()
        if self._last_goal_count > self._best_goal_count:
            self._best_goal_count = self._last_goal_count

        reward = float(max(0, self._best_goal_count - prev_best_goal_count))
        test_rewards = getattr(self.config, "test_rewards", None)
        if test_rewards is not None and self._test_reward_idx < len(test_rewards):
            info = info or {}
            info["origin_reward"] = reward
            reward = float(test_rewards[self._test_reward_idx])
        self._test_reward_idx += 1
        budget_exhausted = bool(
            self.config.enable_action_budget
            and self._budget_remaining is not None
            and self._budget_remaining <= 0
        )
        done = bool(success_done or budget_exhausted)
        info = (info or {}).copy()
        info.update(
            {
                "action_is_valid": action_is_valid,
                "action_is_effective": True,
                "success": bool(success_done),
            }
        )
        info.update(
            self._build_budget_info(
                action_name=action_name,
                action_cost=action_cost,
                budget_exhausted=budget_exhausted and not bool(success_done),
            )
        )
        info.update(self._build_goal_progress_info())
        print(
            f"Action {self._action_count}: {chosen_action} | reward={reward} | "
            f"action_cost={action_cost} | budget_remaining={info.get('budget_remaining')}",
            flush=True,
        )
        return self.render(), float(reward), bool(done), info

    def render(self, mode: str = "text") -> str:
        if mode != "text":
            return self._last_obs
        return self._attach_budget_status(self._last_obs)

    def get_all_actions(self) -> List[int]:
        return list(range(len(self._last_valid_actions)))

    def close(self):
        self._env = None

    def _count_total_goal_predicates(self) -> int:
        if self._env is None or not hasattr(self._env, "current_state"):
            return 0
        state = self._env.current_state
        goal_sets = getattr(state, "goal", [])
        if not goal_sets:
            return 0
        return max(len(goal_set) for goal_set in goal_sets)

    def _count_goal_predicates(self) -> int:
        if self._env is None or not hasattr(self._env, "current_state"):
            return 0
        state = self._env.current_state
        max_count = 0
        for goal_set in getattr(state, "goal", []):
            satisfied = sum(1 for goal in goal_set if state.get_predicate_value(goal))
            if satisfied > max_count:
                max_count = satisfied
        return max_count
