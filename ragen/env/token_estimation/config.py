from dataclasses import dataclass, field
from typing import Optional


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an evaluation agent. Based on the provided rollout context, estimate:
1. Whether the rollout can still finish successfully within {max_context_window_tokens} total tokens (input + output).
2. If yes, how many additional tokens (input + output) are still needed from the next turn onward."""


DEFAULT_USER_PROMPT_TEMPLATE = """
[Original system]
{source_system}

[Completed history]
{history_json}

Based on the provided rollout context:
1. You have completed {completed_turns} turns.
2. Each completed turn used: {turn_token_usage_text}
3. You must finish the full interaction within {max_context_window_tokens} total tokens (input + output).

Estimate:
1. Whether the rollout can finish successfully within the total token budget.
2. If yes, how many additional tokens (input + output) are still needed starting from the next turn. Return a tight interval [est_low, est_high].
3. If no, answer "impossible".

Output exactly one of the following:
<think>[YOUR THINKING]</think><answer>[est_low, est_high]</answer>
or
<think>[YOUR THINKING]</think><answer>impossible</answer>"""

@dataclass
class TokenEstimationEnvConfig:
    input_path: str
    max_context_window_tokens: int = 81920
    max_instances: Optional[int] = None
    include_source_system: bool = True
    system_prompt_template: str = field(default=DEFAULT_SYSTEM_PROMPT_TEMPLATE)
    user_prompt_template: str = field(default=DEFAULT_USER_PROMPT_TEMPLATE)
