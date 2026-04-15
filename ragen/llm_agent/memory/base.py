from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class BaseMemory(ABC):
    """
    Base memory class for managing per-environment history in single-turn mode.
    Different environments can have different history formatting strategies.
    """

    @abstractmethod
    def reset(self, batch_size: int):
        """Reset memory for a new batch of environments."""
        pass

    @abstractmethod
    def store(self, record: Dict[str, List[Any]]):
        """Store a new record (one step of history) for each environment."""
        pass

    @abstractmethod
    def build_user_content(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
        history_start: int,
        turn_offset: int,
        include_warning: bool,
        format_prompt: str,
        length_prompt: str,
        max_actions_per_turn: int,
        no_budget_prompt: bool,
    ) -> str:
        """
        Build user content for single-turn format.

        Args:
            env_output: Environment output dict containing env_id, etc.
            history: List of turn dicts with 'state', 'llm_response', 'reward', etc.
            turn_idx: Current turn index in history
            history_start: Start index for history window
            turn_offset: Offset for actual turn number display
            include_warning: Whether to include invalid action warning
            format_prompt: Format instruction string (e.g., "<think>...</think><answer>...</answer>")
            length_prompt: Length instruction string
            max_actions_per_turn: Maximum number of actions the model may output in a single turn
            no_budget_prompt: Whether to suppress turn-count and action-budget text in the prompt

        Returns:
            Formatted user content string
        """
        pass
