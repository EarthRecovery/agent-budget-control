from omegaconf import OmegaConf

from ragen.wrapper.es_manager_wrapper import EsManagerWrapper


def test_extract_estimation_values_accept_budget_then_estimates_order():
    turn = {
        "llm_raw_response": (
            "<budget-thinking>Budget first</budget-thinking>"
            "<turn_estimation>4</turn_estimation>"
            "<token_estimation>128</token_estimation>"
            "<think>Reason</think><answer>Up</answer>"
        )
    }

    assert EsManagerWrapper._extract_turn_estimate_value(turn) == 4
    assert EsManagerWrapper._extract_estimate_token_value(turn) == 128


def test_extract_estimation_values_still_accept_legacy_prefix_order():
    turn = {
        "llm_raw_response": (
            "<turn_estimation>2</turn_estimation>"
            "<token_estimation>32</token_estimation>"
            "<budget-thinking>Legacy order</budget-thinking>"
            "<think>Reason</think><answer>Down</answer>"
        )
    }

    assert EsManagerWrapper._extract_turn_estimate_value(turn) == 2
    assert EsManagerWrapper._extract_estimate_token_value(turn) == 32


def test_mixed_toolcall_budget_curve_uses_cumulative_action_points():
    wrapper = EsManagerWrapper(
        OmegaConf.create(
            {
                "agent_proxy": {
                    "mixed_toolcall_budget": {
                        "enabled": True,
                        "mixed_budget": True,
                        "reward_curve": {
                            "use_hard": False,
                            "tau": 1.0,
                        },
                    }
                }
            }
        )
    )
    env_output = {
        "env_id": 0,
        "budget_toolcall": 3,
        "history": [
            {"reward": 1.0, "action_points_used": 1, "info": {"success": False}},
            {"reward": 1.0, "action_points_used": 3, "info": {"success": True}},
        ],
    }

    wrapper._apply_toolcall_budget_curve(env_output)

    expected_first = EsManagerWrapper._cal_budget_reward(1.0, current_turn=1, budget_turn=3)
    expected_second = EsManagerWrapper._cal_budget_reward(1.0, current_turn=4, budget_turn=3)
    assert env_output["history"][0]["origin_reward"] == 1.0
    assert env_output["history"][1]["origin_reward"] == 1.0
    assert env_output["history"][0]["reward"] == expected_first
    assert env_output["history"][1]["reward"] == expected_second
    assert env_output["history"][0]["reward"] > env_output["history"][1]["reward"]
