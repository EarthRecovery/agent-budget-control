# Robotouille Action Budget

This directory adds an optional action-point budget on top of the existing Robotouille wrapper in RAGEN.

## Config

The Robotouille environment config now exposes two budget-related fields in [config.py](/home/bba1908/agent-budget-control/ragen/env/robotouille/config.py):

- `enable_action_budget`: whether the action-point system is enabled. Default is `False`.
- `max_action_points`: the maximum number of action points available after each `reset()`. Default is `10`.

## Behavior

- When `enable_action_budget=False`, Robotouille behaves exactly as before.
- When `enable_action_budget=True`, each valid Robotouille action consumes action points according to [action_costs.json](/home/bba1908/agent-budget-control/ragen/env/robotouille/action_costs.json).
- The remaining budget is appended to the rendered text observation as:
  - `Action Budget: enabled`
  - `Action Points Remaining: current/max`
- The rendered text observation also appends a `Valid Action Point Costs:` section that lists each currently valid action together with its per-action-point cost.
- The `info` dictionary returned by `step()` also includes:
  - `budget_enabled`
  - `budget_max`
  - `budget_remaining`
  - `budget_action_name`
  - `budget_action_cost`
  - `budget_exhausted`

## Budget Exhaustion Rule

- If the next action cost is larger than the remaining action points, the action is not executed.
- In that case the environment returns `done=True`, `reward=0.0`, and `budget_exhausted=True`.
- Otherwise the action executes first, then its cost is deducted from the remaining budget.

## Action Cost Table

Default per-action costs are stored in [action_costs.json](/home/bba1908/agent-budget-control/ragen/env/robotouille/action_costs.json). The JSON is keyed by Robotouille's internal action names such as `move`, `stack`, `cook`, and `wait`.

You can tune the budget system by editing:

- `max_action_points` to change the total available budget.
- `action_costs.json` to change the cost of individual actions.
