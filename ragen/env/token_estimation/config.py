from dataclasses import dataclass, field
from typing import Optional


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an evaluation agent. Based on the provided rollout context, estimate:
1. Whether this rollout can still finish successfully while keeping the finishing turn within {max_context_window_tokens} total tokens (input + output).
2. If yes, by how many total tokens (input + output) the rollout's total-token footprint still needs to grow, relative to the most recently completed turn (or 0 if no turn has been completed yet), in order to reach the finishing turn. This growth includes this turn's output tokens and only the net-new portion of this turn's input tokens relative to the most recently completed turn."""


DEFAULT_USER_PROMPT_TEMPLATE = """
[Original system]
{source_system}

[Visible history]
{input_messages_text}

You have completed {completed_turns} turns. Each turn, your token consumption is: 
{turn_token_usage_text}

The finishing-turn context-window budget is {max_context_window_tokens} total tokens (input + output). Based on the current state, estimate by how many total tokens (input + output) the rollout's total-token footprint still needs to grow, relative to the most recently completed turn (or 0 if no turn has been completed yet), in order to reach the finishing turn. This growth includes this turn's output tokens and only the net-new portion of this turn's input tokens relative to the most recently completed turn. If the finishing turn would exceed this budget, answer "impossible".

Output exactly one of the following:
<think>[YOUR THINKING]</think><answer>[Est_low, Est_high]</answer>
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
