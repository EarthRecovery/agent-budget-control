from ragen.env.robotouille import RobotouilleEnv, RobotouilleEnvConfig


class _DummyActionDef:
    def __init__(self, name):
        self.name = name


class _DummyState:
    def __init__(self, valid_actions):
        self._valid_actions = valid_actions

    def get_valid_actions_and_str(self):
        valid_action_defs = [item[0] for item in self._valid_actions]
        valid_action_strs = [item[1] for item in self._valid_actions]
        return valid_action_defs, valid_action_strs


class _DummyRobotouilleEnv:
    def __init__(self, valid_actions):
        self.current_state = _DummyState(valid_actions)


def test_robotouille_config_keeps_action_lookup_optional():
    config = RobotouilleEnvConfig()

    assert config.action_lookup is None


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


def test_extract_valid_actions_always_appends_finish_action():
    env = RobotouilleEnv(RobotouilleEnvConfig(enable_action_budget=False, max_action_space=1))
    env._env = _DummyRobotouilleEnv(
        [
            ((_DummyActionDef("move"), None), "move robot1 to station1"),
            ((_DummyActionDef("cook"), None), "cook patty1 on stove1"),
        ]
    )

    env._extract_valid_actions()

    assert env._last_valid_actions == [
        "move robot1 to station1",
        env.FINISH_ACTION,
    ]
    assert env._last_action_names[env.FINISH_ACTION] == "finish"


def test_finish_action_ends_episode_early_without_success():
    env = RobotouilleEnv(
        RobotouilleEnvConfig(enable_action_budget=True, max_action_points=6)
    )
    env._env = object()
    env._last_obs = "Observation: test state"
    env._budget_remaining = 4
    env._last_valid_actions = [env.FINISH_ACTION]
    env._last_action_names = {env.FINISH_ACTION: "finish"}
    env._count_goal_predicates = lambda: 2
    env._count_total_goal_predicates = lambda: 5

    obs, reward, done, info = env.step(env.FINISH_ACTION)

    assert obs == env.render()
    assert reward == 0.0
    assert done is True
    assert info["success"] is False
    assert info["terminated_by_agent"] is True
    assert info["budget_action_name"] == "finish"
    assert info["budget_action_cost"] == 0
    assert info["budget_remaining"] == 4


def test_finish_action_marks_success_when_goal_is_complete():
    env = RobotouilleEnv(RobotouilleEnvConfig(enable_action_budget=False))
    env._env = object()
    env._last_obs = "Observation: solved state"
    env._last_valid_actions = [env.FINISH_ACTION]
    env._last_action_names = {env.FINISH_ACTION: "finish"}
    env._count_goal_predicates = lambda: 3
    env._count_total_goal_predicates = lambda: 3

    _, reward, done, info = env.step(env.FINISH_ACTION)

    assert reward == 0.0
    assert done is True
    assert info["success"] is True
    assert info["terminated_by_agent"] is True
