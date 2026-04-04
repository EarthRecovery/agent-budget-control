from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RobotouilleEnvConfig:
    env_name: str = "synchronous/0_cheese_sandwich"
    max_steps: int = 100
    seed: Optional[int] = None
    noisy_randomization: bool = False
    render_mode: str = "text"
    max_action_space: int = 256
    enable_action_budget: bool = False
    max_action_points: int = 10
    action_lookup: Optional[Dict[int, str]] = field(default_factory=dict)
    test_rewards: Optional[List[float]] = None

    def __post_init__(self):
        if self.max_action_points < 0:
            raise ValueError("max_action_points must be non-negative.")
        if not self.action_lookup:
            self.action_lookup = {
                idx: f"Action[{idx}]"
                for idx in range(self.max_action_space)
            }
