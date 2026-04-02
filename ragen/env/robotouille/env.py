import os
import sys
from typing import Dict, List, Tuple

from ragen.env.base import BaseDiscreteActionEnv
from ragen.utils import all_seed

from .config import RobotouilleEnvConfig


def _ensure_robotouille_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    robotouille_root = os.path.join(repo_root, "external", "robotouille")
    if robotouille_root not in sys.path:
        sys.path.insert(0, robotouille_root)


class RobotouilleEnv(BaseDiscreteActionEnv):
    def __init__(self, config: RobotouilleEnvConfig = None):
        self.config = config or RobotouilleEnvConfig()
        self._env = None
        self._last_obs = ""
        self._last_valid_actions: List[str] = []
        self._last_goal_count = 0
        self._best_goal_count = 0
        self._action_count = 0
        self._test_reward_idx = 0
        BaseDiscreteActionEnv.__init__(self)

    def _init_env(self):
        _ensure_robotouille_on_path()
        from robotouille.robotouille_env import create_robotouille_env

        self._env = create_robotouille_env(
            self.config.env_name,
            self.config.seed,
            self.config.noisy_randomization,
        )

    def _extract_valid_actions(self):
        if self._env is None:
            self._last_valid_actions = []
            return
        _, str_valid_actions = self._env.current_state.get_valid_actions_and_str()
        if len(str_valid_actions) > self.config.max_action_space:
            str_valid_actions = str_valid_actions[: self.config.max_action_space]
        self._last_valid_actions = list(str_valid_actions)

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
            return self.render(), 0.0, False, {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
            }

        prev_best_goal_count = self._best_goal_count
        self._action_count += 1
        obs, _, done, info = self._env.step(chosen_action)
        self._last_obs = obs
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
        info = (info or {}).copy()
        info.update(
            {
                "action_is_valid": action_is_valid,
                "action_is_effective": True,
                "success": bool(done),
            }
        )
        print(
            f"Action {self._action_count}: {chosen_action} | reward={reward}",
            flush=True,
        )
        return self.render(), float(reward), bool(done), info

    def render(self, mode: str = "text") -> str:
        if mode != "text":
            return self._last_obs
        return self._last_obs

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
