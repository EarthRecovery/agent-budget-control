from dataclasses import dataclass
from typing import Optional


@dataclass
class GPQAMainEnvConfig:
    dataset_path: str = "idavidrein/gpqa"
    dataset_config: str = "gpqa_main"
    cache_dir: Optional[str] = "./data"
    split: str = "train"
    render_mode: str = "text"
    max_steps: int = 1
    invalid_action_score: float = 0.0
