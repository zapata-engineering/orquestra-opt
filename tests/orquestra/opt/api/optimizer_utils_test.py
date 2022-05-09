################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np

from orquestra.opt.api import FunctionWithGradient
from orquestra.opt.api.optimizer import construct_history_info
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.opt.history import recorder


def sum_x_squared(x):
    return sum(x**2.0)


class TestConstructHistoryInfo:
    def test_history_is_empty_if_keep_value_history_is_false(self):
        cost_function = recorder(sum_x_squared)

        cost_function(np.array([1, 2, 3]))

        history_info = construct_history_info(cost_function, False)

        assert not history_info["history"]
        assert "gradient_history" not in history_info

    def test_history_info_contains_only_history_for_function_without_gradient(self):
        cost_function = recorder(sum_x_squared)

        cost_function(np.array([1, 2, 3]))

        history_info = construct_history_info(cost_function, True)

        assert len(history_info["history"]) == 1
        assert "gradient_history" not in history_info

        history_entry = history_info["history"][0]
        assert history_entry.call_number == 0
        np.testing.assert_array_equal(history_entry.params, [1, 2, 3])
        assert history_entry.value == cost_function(np.array([1, 2, 3]))

    def test_history_info_contains_gradient_history_for_function_with_gradient(self):
        cost_function = recorder(
            FunctionWithGradient(
                sum_x_squared, finite_differences_gradient(sum_x_squared)
            )
        )

        cost_function(np.array([1, 2, 3]))
        cost_function.gradient(np.array([0, -1, 1]))

        history_info = construct_history_info(cost_function, True)

        assert len(history_info["history"]) == 1
        assert len(history_info["gradient_history"]) == 1

        history_entry = history_info["history"][0]
        assert history_entry.call_number == 0
        np.testing.assert_array_equal(history_entry.params, [1, 2, 3])
        assert history_entry.value == cost_function(np.array([1, 2, 3]))

        history_entry = history_info["gradient_history"][0]
        assert history_entry.call_number == 0
        np.testing.assert_array_equal(history_entry.params, [0, -1, 1])
        np.testing.assert_array_equal(
            history_entry.value, cost_function.gradient(np.array([0, -1, 1]))
        )
