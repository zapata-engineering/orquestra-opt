################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""Test case prototypes that can be used in other projects.

Note that this file won't be executed on its own by pytest.
You need to define your own test cases that inherit from the ones defined here.
"""


from typing import Callable

import numpy as np
import pytest

from ..gradients import finite_differences_gradient
from ..history.recorder import recorder
from .cost_function import CostFunction
from .functions import FunctionWithGradient
from .optimizer import NestedOptimizer

MANDATORY_OPTIMIZATION_RESULT_FIELDS = (
    "nfev",
    "nit",
    "opt_value",
    "opt_params",
    "history",
)


def sum_x_squared(x: np.ndarray) -> float:
    return sum(x**2.0)


def rosenbrock_function(x: np.ndarray) -> float:
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


"""Contracts for optimizers tests.

Basic usage pattern:
    from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS

    class TestMyOptimizer():
        # Test all contracts
        @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
        def test_optimizer_satisfies_contracts(self, contract):
            optimizer = ...
            assert contract(optimizer)

        def test_some_new_feature(self, optimizer): # new test
            ....

Notice that the `optimizer` can be made into a fixture and parametrized if you wish to
perform tests for various configurations of your optimizer.
"""


def _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function(
    optimizer, accuracy: float = 1e-5
):
    cost_function = FunctionWithGradient(
        sum_x_squared, finite_differences_gradient(sum_x_squared)
    )

    results = optimizer.minimize(cost_function, initial_params=np.array([1, -1]))

    return (
        results.opt_value == pytest.approx(0, abs=accuracy)
        and results.opt_params == pytest.approx(np.zeros(2), abs=accuracy * 10)
        and all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)
    )


def _validate_optimizer_succeeds_with_optimizing_rosenbrock_function(
    optimizer, accuracy: float = 1e-4
):
    cost_function = FunctionWithGradient(
        rosenbrock_function, finite_differences_gradient(rosenbrock_function)
    )

    results = optimizer.minimize(cost_function, initial_params=np.array([0, 0]))

    return (
        results.opt_value == pytest.approx(0, abs=accuracy)
        and results.opt_params == pytest.approx(np.ones(2), abs=accuracy * 10)
        and all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)
    )


def _validate_optimizer_succeeds_on_cost_function_without_gradient(
    optimizer, accuracy: float = 1e-5
):
    cost_function = sum_x_squared

    results = optimizer.minimize(cost_function, initial_params=np.array([1, -1]))
    return (
        results.opt_value == pytest.approx(0, abs=accuracy)
        and results.opt_params == pytest.approx(np.zeros(2), abs=accuracy * 10)
        and all(field in results for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)
        and "gradient_history" not in results
    )


def _validate_optimizer_records_history_if_keep_history_is_true(optimizer):
    # To check that history is recorded correctly, we wrap cost_function
    # with a recorder. Optimizer should wrap it a second time and
    # therefore we can compare two histories to see if they agree.
    cost_function = recorder(
        FunctionWithGradient(sum_x_squared, finite_differences_gradient(sum_x_squared))
    )

    result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)
    if len(result.history) != len(cost_function.history):
        return False

    for result_history_entry, cost_function_history_entry in zip(
        result.history, cost_function.history
    ):
        if not (
            result_history_entry.call_number == cost_function_history_entry.call_number
            and np.allclose(
                result_history_entry.params, cost_function_history_entry.params
            )
            and np.isclose(
                result_history_entry.value, cost_function_history_entry.value
            )
        ):
            return False
    return True


def _validate_gradients_history_is_recorded_if_keep_history_is_true(optimizer):
    # To check that history is recorded correctly, we wrap cost_function
    # with a recorder. Optimizer should wrap it a second time and
    # therefore we can compare two histories to see if they agree.
    cost_function = recorder(
        FunctionWithGradient(sum_x_squared, finite_differences_gradient(sum_x_squared))
    )

    result = optimizer.minimize(cost_function, np.array([-1, 1]), keep_history=True)
    if "gradient_history" not in result:
        return False
    if len(result.gradient_history) != len(cost_function.gradient.history):
        return False

    for result_history_entry, cost_function_history_entry in zip(
        result.gradient_history, cost_function.gradient.history
    ):
        if result_history_entry.call_number != cost_function_history_entry.call_number:
            return False
        if not np.allclose(
            result_history_entry.params, cost_function_history_entry.params
        ):
            return False
        if not np.allclose(
            result_history_entry.value, cost_function_history_entry.value
        ):
            return False

    return True


def _validate_optimizer_does_not_record_history_if_keep_history_is_false(optimizer):
    cost_function = FunctionWithGradient(
        sum_x_squared, finite_differences_gradient(sum_x_squared)
    )
    result = optimizer.minimize(cost_function, np.array([-2, 0.5]), keep_history=False)

    return result.history == []


def _validate_optimizer_does_not_record_history_by_default(optimizer):
    cost_function = FunctionWithGradient(
        sum_x_squared, finite_differences_gradient(sum_x_squared)
    )
    result = optimizer.minimize(cost_function, np.array([-2, 0.5]))

    return result.history == []


def _validate_changing_keep_history_does_not_change_results(
    optimizer, accuracy: float = 1e-5
):
    cost_function = FunctionWithGradient(
        sum_x_squared, finite_differences_gradient(sum_x_squared)
    )

    results_with_history = optimizer.minimize(
        cost_function, initial_params=np.array([1, -1]), keep_history=True
    )
    results_without_history = optimizer.minimize(
        cost_function, initial_params=np.array([1, -1]), keep_history=False
    )

    return results_with_history.opt_value == pytest.approx(
        results_without_history.opt_value, abs=accuracy
    ) and results_without_history.opt_params == pytest.approx(
        results_without_history.opt_params, abs=accuracy * 10
    )


OPTIMIZER_CONTRACTS = [
    _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function,
    _validate_optimizer_succeeds_with_optimizing_rosenbrock_function,
    _validate_optimizer_succeeds_on_cost_function_without_gradient,
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_changing_keep_history_does_not_change_results,
]


"""Contracts for instances of the NestedOptimizer base class that can be used in
other projects.

Usage:

    .. code:: python

       from orquestra.opt.api.optimizer_test import NESTED_OPTIMIZER_CONTRACTS

       @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
       def test_nestedoptimizer_contract(contract):
           optimizer = MockNestedOptimizer(inner_optimizer=MockOptimizer(), n_iters=5)
           cost_function_factory = ...
           initial_params = ...
           assert contract(optimizer, cost_function_factory, initial_params)
"""


def _validate_nested_optimizer_records_history_if_keep_history_is_true(
    optimizer: NestedOptimizer,
    cost_function_factory: Callable[..., CostFunction],
    initial_params: np.ndarray,
):
    result = optimizer.minimize(
        cost_function_factory, initial_params, keep_history=True
    )
    return len(result.history) != 0


def _validate_nested_optimizer_records_gradient_history_if_keep_history_is_true(
    optimizer: NestedOptimizer,
    cost_function_factory: Callable[..., CostFunction],
    initial_params: np.ndarray,
):
    def cost_function_with_gradients_factory(*args, **kwargs):
        cost_function = cost_function_factory(*args, **kwargs)
        return FunctionWithGradient(
            cost_function, finite_differences_gradient(cost_function)
        )

    result = optimizer.minimize(
        cost_function_with_gradients_factory, initial_params, keep_history=True
    )
    return hasattr(result, "gradient_history")


def _validate_nested_optimizer_does_not_record_history_if_keep_history_is_false(
    optimizer: NestedOptimizer,
    cost_function_factory: Callable[..., CostFunction],
    initial_params: np.ndarray,
):
    result = optimizer.minimize(
        cost_function_factory, initial_params, keep_history=False
    )
    return len(result.history) == 0


def _validate_nested_optimizer_does_not_record_history_by_default(
    optimizer: NestedOptimizer,
    cost_function_factory: Callable[..., CostFunction],
    initial_params: np.ndarray,
):

    result = optimizer.minimize(cost_function_factory, initial_params)
    return result.history == []


def _validate_nested_optimizer_returns_all_the_mandatory_fields_in_results(
    optimizer: NestedOptimizer,
    cost_function_factory: Callable[..., CostFunction],
    initial_params: np.ndarray,
):
    result = optimizer.minimize(cost_function_factory, initial_params)
    return all(field in result for field in MANDATORY_OPTIMIZATION_RESULT_FIELDS)


NESTED_OPTIMIZER_CONTRACTS = [
    _validate_nested_optimizer_records_history_if_keep_history_is_true,
    _validate_nested_optimizer_records_gradient_history_if_keep_history_is_true,
    _validate_nested_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_nested_optimizer_does_not_record_history_by_default,
    _validate_nested_optimizer_returns_all_the_mandatory_fields_in_results,
]
