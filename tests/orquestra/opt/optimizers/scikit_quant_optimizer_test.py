################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest

from orquestra.opt.api.optimizer_test import (
    _validate_changing_keep_history_does_not_change_results,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_optimizer_succeeds_on_cost_function_without_gradient,
    _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function,
    sum_x_squared,
)
from orquestra.opt.optimizers.scikit_quant_optimizer import ScikitQuantOptimizer

# Scikit-Quant optimizers dont work well for Rosenbrock function, so we omit the
# Rosenbrock contract from our list.
SCIKIT_QUANT_CONTTRACTS = [
    _validate_optimizer_succeeds_on_cost_function_without_gradient,
    _validate_optimizer_succeeds_with_optimizing_sum_of_squares_function,
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_changing_keep_history_does_not_change_results,
]


@pytest.fixture(
    params=[
        {
            "method": "imfil",
            "bounds": np.array([[-4.05, 4.05], [-4.05, 4.05]], dtype=float),
            "budget": 10000,
        },
        {
            "method": "snobfit",
            "bounds": np.array([[-4.05, 4.05], [-4.05, 4.05]], dtype=float),
            "budget": 10000,
        },
        {
            "method": "pybobyqa",
            "bounds": np.array([[-2.05, 2.05], [-2.05, 2.05]], dtype=float),
        },  # passed with the bounds defined
    ]
)
def optimizer(request):
    return ScikitQuantOptimizer(**request.param)


class TestScikitQuantOptimizer:
    @pytest.mark.parametrize("contract", SCIKIT_QUANT_CONTTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)

    @pytest.mark.parametrize("number_of_params", [2, 3, 100])
    def test_bounds_are_defined_correctly_if_not_provided(self, number_of_params):

        initial_params = np.zeros(number_of_params)

        optimizer = ScikitQuantOptimizer(method="imfil")

        _ = optimizer.minimize(sum_x_squared, initial_params)

        assert len(optimizer.bounds) == number_of_params
        np.testing.assert_array_equal(optimizer.bounds[0], np.array([-1000, 1000]))

    def test_fails_if_bounds_dont_match_params(self, optimizer):
        initial_params = np.ones(5)
        optimizer.bounds = np.array([[-1000, 1000], [-1000, 1000]])
        with pytest.raises(ValueError):
            _ = optimizer.minimize(sum_x_squared, initial_params)

    def test_length_1_input_fails(self, optimizer):
        initial_params = np.ones(1)
        with pytest.raises(ValueError):
            _ = optimizer.minimize(sum_x_squared, initial_params)
