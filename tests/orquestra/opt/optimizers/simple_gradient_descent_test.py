################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest

from orquestra.opt.api import function_with_gradient
from orquestra.opt.api.optimizer_test import (
    _validate_changing_keep_history_does_not_change_results,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function,
)
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.opt.optimizers.simple_gradient_descent import SimpleGradientDescent

SIMPLE_GRADIENT_DESCENT_CONTRACTS = [
    _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function,
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_changing_keep_history_does_not_change_results,
]
# The following contracts are expected to fail, and thus omitted from this list:
# _validate_optimizer_succeeds_with_optimizing_rosenbrock_function: This test fails
# since the gradient of the rosenbrock function is too sensitive when using finite
# differences
# _validate_optimizer_succeeds_on_cost_function_without_gradient: This test fails since
#  SimpleGradientDescent requires cost_function to have gradient method.


@pytest.fixture(
    params=[
        {"learning_rate": 0.1, "number_of_iterations": 100},
        {"learning_rate": 0.15, "number_of_iterations": 100},
        {"learning_rate": 0.215242, "number_of_iterations": 100},
        {"learning_rate": 0.99, "number_of_iterations": 1000},
    ]
)
def optimizer(request):
    return SimpleGradientDescent(**request.param)


class TestSimpleGradientDescent:
    @pytest.mark.parametrize("contract", SIMPLE_GRADIENT_DESCENT_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)

    @pytest.fixture
    def sum_x_squared(self):
        def _sum_x_squared(x):
            return sum(x**2)

        return function_with_gradient(
            _sum_x_squared, finite_differences_gradient(_sum_x_squared)
        )

    def test_fails_to_initialize_when_number_of_iterations_is_negative(self):
        with pytest.raises(AssertionError):
            SimpleGradientDescent(0.1, -1)

    def test_fails_to_minimize_when_cost_function_does_not_have_gradient_method(
        self, optimizer
    ):
        def cost_function(x):
            return sum(x)

        with pytest.raises(AssertionError):
            optimizer.minimize(cost_function, np.array([0, 0]))

    def test_history_contains_function_evaluations(self, optimizer, sum_x_squared):
        results = optimizer.minimize(sum_x_squared, np.array([1, 0]), keep_history=True)

        assert len(results.history) == optimizer.number_of_iterations
