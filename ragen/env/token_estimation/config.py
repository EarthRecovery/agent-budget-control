from dataclasses import dataclass, field
from typing import Optional


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an evaluation agent. Based on the provided rollout context, estimate:
1. Whether the rollout can still finish successfully within {max_context_window_tokens} total tokens (input + output).
2. If yes, how many additional tokens (input + output) are still needed from the next turn onward."""


DEFAULT_USER_PROMPT_TEMPLATE = """
Based on the provided rollout context, you are provided below information:
1. You have completed {completed_turns} turns.
2. Each turn, your token consumption is {turn_token_usage_text}.
3. You need to finish the task within {max_context_window_tokens} tokens.

Now, estimate:
1. Whether you can finish the task successfully within {max_context_window_tokens} total tokens (input + output).
2. If yes, how many additional tokens (input + output) are still needed to finish the task, starting from the next turn. Return an estimation interval: at least est_low tokens and at most est_high tokens.
3. If no, answer "impossible".
4. You should try your best to estimate whether the task can finish within budget (most important). If you think the task can finish within budget, your interval should be as tight as possible while still covering the true remaining token budget.

Example:
For a three-turn interaction, suppose only Turn 1 has been completed.
The full interaction is:
Turn 1: input X1 tokens, output Y1 tokens;
Turn 2: input X2 tokens, output Y2 tokens;
Turn 3: input X3 tokens, output Y3 tokens.
You will receive:
turn_token_usage_text: Turn 1: input X1 tokens, output Y1 tokens
You should estimate:
X2 + Y2 + X3 + Y3

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
