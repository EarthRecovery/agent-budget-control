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
