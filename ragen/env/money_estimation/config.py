from dataclasses import dataclass, field
from typing import Optional


DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are an evaluation agent for historical warehouse-management rollouts.
Judge whether the rollout can still finish successfully within all remaining budgets.
If it can, estimate the remaining resource usage from the next turn onward.
Follow the required output format exactly."""


DEFAULT_USER_PROMPT_TEMPLATE = """
Based on the provided warehouse rollout context, you are given the following information:
1. You have completed {completed_weeks} weeks in {completed_turns} turns.
2. Current cumulative usage so far:
   - time_weeks: {current_time_weeks}
   - warehouse_item_weeks: {current_warehouse_item_weeks}
   - cumulative_cost_usd: {current_cost_usd}
3. Current cash is {current_cash_usd} USD. To count as finished, final cash must reach at least {target_cash_usd} USD.
4. Historical resource consumption by completed step is:
{resource_consumption_text}
5. The rollout must finish within all three budgets:
   - time_weeks <= {budget_time_weeks}
   - warehouse_item_weeks <= {budget_warehouse_item_weeks}
   - cumulative_cost_usd <= {budget_cost_usd}

Now, estimate:
1. Whether the rollout can still finish successfully within all three budgets while also reaching the target cash.
2. If yes, how much additional usage is still needed from the next turn onward. Return one interval for each metric.
3. If no, answer "impossible".
4. Prioritize the can-finish judgment over interval tightness. If you think the rollout can finish within budget, make each interval as tight as possible while still covering the true remaining value.

Output exactly one of the following:
<think>[YOUR THINKING]</think><answer>time_weeks:[est_low, est_high], warehouse_item_weeks:[est_low, est_high], cumulative_cost_usd:[est_low, est_high]</answer>
or
<think>[YOUR THINKING]</think><answer>impossible</answer>"""


@dataclass
class MoneyEstimationEnvConfig:
    input_path: str
    max_instances: Optional[int] = None
    include_source_system: bool = True
    system_prompt_template: str = field(default=DEFAULT_SYSTEM_PROMPT_TEMPLATE)
    user_prompt_template: str = field(default=DEFAULT_USER_PROMPT_TEMPLATE)
    target_cash_usd: Optional[float] = None
    target_cash_mode: str = "ratio"
    target_cash_ratio: float = 1.0
    target_cash_half_reachable_seed: int = 42
    time_budget_weeks: Optional[float] = None
    time_budget_ratio: float = 1.0
    warehouse_budget_item_weeks: Optional[float] = None
    warehouse_budget_ratio: float = 1.0
    cost_budget_usd: Optional[float] = None
    cost_budget_ratio: float = 1.0
