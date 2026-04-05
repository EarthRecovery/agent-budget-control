from ragen.env.robotouille import RobotouilleEnv, RobotouilleEnvConfig


def test_render_includes_valid_action_costs_when_budget_enabled():
    env = RobotouilleEnv(
        RobotouilleEnvConfig(enable_action_budget=True, max_action_points=6)
    )
    env._last_obs = "Observation: test state"
    env._budget_remaining = 4
    env._last_valid_actions = [
        "move robot1 to station1",
        "cook patty1 on stove1",
    ]
    env._last_action_names = {
        "move robot1 to station1": "move",
        "cook patty1 on stove1": "cook",
    }

    rendered = env.render()

    assert "Action Budget: enabled" in rendered
    assert "Action Points Remaining: 4/6" in rendered
    assert "Valid Action Point Costs:" in rendered
    assert "- move robot1 to station1: 1 action point" in rendered
    assert "- cook patty1 on stove1: 2 action points" in rendered


def test_render_omits_action_costs_when_budget_disabled():
    env = RobotouilleEnv(RobotouilleEnvConfig(enable_action_budget=False))
    env._last_obs = "Observation: test state"
    env._last_valid_actions = ["move robot1 to station1"]
    env._last_action_names = {"move robot1 to station1": "move"}

    rendered = env.render()

    assert rendered == "Observation: test state"


def test_build_goal_progress_info_reports_ratio():
    env = RobotouilleEnv(RobotouilleEnvConfig(enable_action_budget=False))
    env._count_goal_predicates = lambda: 2
    env._count_total_goal_predicates = lambda: 5

    info = env._build_goal_progress_info()

    assert info == {
        "goal_predicates_satisfied": 2,
        "goal_predicates_total": 5,
        "goal_predicate_ratio_reward": 0.4,
    }