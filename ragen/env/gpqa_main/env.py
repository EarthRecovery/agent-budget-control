import os
import random
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset

from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed

from .config import GPQAMainEnvConfig


class GPQAMainEnv(BaseLanguageBasedEnv):
    def __init__(self, config: Optional[GPQAMainEnvConfig] = None):
        super(GPQAMainEnv, self).__init__()
        self.config = config if config is not None else GPQAMainEnvConfig()
        self.dataset = load_dataset(
            self.config.dataset_path,
            self.config.dataset_config,
            cache_dir=self.config.cache_dir,
        )
        self.render_mode = self.config.render_mode

        self.current_question: Optional[str] = None
        self.current_answer: Optional[str] = None
        self.step_num = 0
        self.render_cache: Optional[str] = None
        self.debug_reward_flow = (
            bool(getattr(self.config, "debug_reward_flow", False))
            or os.environ.get("RAGEN_DEBUG_REWARD_FLOW", "0") == "1"
        )

    def _format_question(self, item: Dict[str, Any]) -> Tuple[str, str]:
        question = item.get("question", item.get("Question", ""))
        choices = item.get("choices", item.get("options", None))
        answer = item.get("answer", item.get("correct_answer", ""))

        if not choices and "Correct Answer" in item:
            choices = [
                item.get("Correct Answer", ""),
                item.get("Incorrect Answer 1", ""),
                item.get("Incorrect Answer 2", ""),
                item.get("Incorrect Answer 3", ""),
            ]
            answer = "A"

        if isinstance(choices, dict):
            ordered = [f"{key}. {value}" for key, value in choices.items()]
            choices_text = "\n".join(ordered)
            prompt = f"Question: {question}\nChoices:\n{choices_text}"
        elif isinstance(choices, list):
            labeled = [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
            choices_text = "\n".join(labeled)
            prompt = f"Question: {question}\nChoices:\n{choices_text}"
        else:
            prompt = f"Question: {question}"

        return prompt, str(answer).strip()

    def reset(self, seed: Optional[int] = None, mode: Optional[str] = None) -> Any:
        split = self.config.split or next(iter(self.dataset.keys()))
        dataset_split = self.dataset[split]
        with all_seed(seed):
            index = random.randint(0, len(dataset_split) - 1)
        item = dataset_split[index]
        self.current_question, self.current_answer = self._format_question(item)
        self.step_num = 0
        self.render_cache = self.current_question
        return self.render_cache

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        action = action.strip()
        is_valid = action != ""
        is_correct = is_valid and (action.lower() == (self.current_answer or "").lower())
        reward = 1.0 if is_correct else 0.0
        observation = "Correct!" if is_correct else "Incorrect."
        done = True if is_correct else (self.step_num + 1) >= self.config.max_steps
        self.step_num += 1
        info = {
            "action_is_effective": is_correct,
            "action_is_valid": is_valid,
            "success": is_correct,
        }
        if self.debug_reward_flow:
            print(
                "[reward-debug][gpqa_env] "
                f"step={self.step_num}, action={action!r}, expected={self.current_answer!r}, "
                f"is_valid={is_valid}, is_correct={is_correct}, reward={reward:.4f}, "
                f"done={done}, success={info['success']}"
            )
        self.render_cache = observation
        return observation, reward, done, info

    def render(self, mode: Optional[str] = None) -> Any:
        return self.render_cache

    def close(self) -> None:
        self.render_cache = None
