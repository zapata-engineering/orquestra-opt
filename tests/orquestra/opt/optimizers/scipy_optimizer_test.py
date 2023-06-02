################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest

from orquestra.opt.api import FunctionWithGradient
from orquestra.opt.api.optimizer_test import (
    OPTIMIZER_CONTRACTS,
    rosenbrock_function,
    sum_x_squared,
)
from orquestra.opt.gradients import finite_differences_gradient
from orquestra.opt.optimizers.scipy_optimizer import ScipyOptimizer


@pytest.fixture(
    params=[
        {"method": "BFGS"},
        {"method": "L-BFGS-B"},
        {"method": "Nelder-Mead"},
        {"method": "SLSQP"},
        {"method": "Powell"},
        {"method": "COBYLA", "options": {"maxiter": 50000, "tol": 1e-7}},
    ]
)
def optimizer(request):
    return ScipyOptimizer(**request.param)


@pytest.fixture(
    params=[
        {"method": "L-BFGS-B"},
        {"method": "Nelder-Mead"},
        {"method": "SLSQP"},
    ]
)
def optimizer_with_bounds(request):
    bounds = [(2, 3), (2, 3)]
    return ScipyOptimizer(bounds=bounds, **request.param)


class TestScipyOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)

    def test_optimizers_work_with_bounds_provided(self, optimizer_with_bounds):
        # Given
        cost_function = FunctionWithGradient(
            sum_x_squared, finite_differences_gradient(sum_x_squared)
        )

        initial_params = np.array([2.5, 2.5])
        target_params = np.array([2, 2])
        target_value = 8

        # When
        results = optimizer_with_bounds.minimize(
            cost_function, initial_params=initial_params
        )

        # Then
        assert results.opt_value == pytest.approx(target_value, abs=1e-3)
        assert results.opt_params == pytest.approx(target_params, abs=1e-3)

    def test_SLSQP_with_equality_constraints(self):
        # Given
        cost_function = FunctionWithGradient(
            rosenbrock_function, finite_differences_gradient(rosenbrock_function)
        )
        constraint_cost_function = sum_x_squared

        constraints = ({"type": "eq", "fun": constraint_cost_function},)
        optimizer = ScipyOptimizer(method="SLSQP", constraints=constraints)
        initial_params = np.array([1, 1])
        target_params = np.array([0, 0])
        target_value = 1

        # When
        results = optimizer.minimize(cost_function, initial_params=initial_params)

        # Then
        assert results.opt_value == pytest.approx(target_value, abs=1e-3)
        assert results.opt_params == pytest.approx(target_params, abs=1e-3)

    def test_SLSQP_with_inequality_constraints(self):
        # Given
        cost_function = FunctionWithGradient(
            rosenbrock_function, finite_differences_gradient(rosenbrock_function)
        )
        constraints = ({"type": "ineq", "fun": lambda x: x[0] + x[1] - 3},)
        optimizer = ScipyOptimizer(method="SLSQP")
        initial_params = np.array([0, 0])

        # When
        results_without_constraints = optimizer.minimize(
            cost_function, initial_params=initial_params
        )
        optimizer.constraints = constraints
        results_with_constraints = optimizer.minimize(
            cost_function, initial_params=initial_params
        )

        # Then
        assert results_without_constraints.opt_value == pytest.approx(
            results_with_constraints.opt_value, abs=1e-1
        )
        assert results_with_constraints.opt_params.sum() >= 3

    # https://github.com/scipy/scipy/issues/17673 reports a function and that is
    # minimised with an initial point that has a given cost function value which is
    # lower than the optimised value by the Powell optimizer. This test is to ensure
    # that our solution over scipy is not affected by this issue.
    @pytest.mark.parametrize("method", ["Powell", "L-BFGS-B", "Nelder-Mead", "SLSQP"])
    def test_Langermann_function(self, method):
        # Definition of the function
        langer_c = np.array([6, 1, 4, 4, 8])
        langer_A = np.array(
            [[4, 6, 3, 5], [8, 7, 9, 9], [2, 7, 8, 8], [9, 2, 6, 9], [5, 4, 1, 4]]
        )

        def langermann(parameters: np.ndarray) -> float:
            # This function is bounded in the [0, 10] box
            sum_ = np.sum(np.subtract(langer_A, parameters) ** 2, axis=1)
            vec = np.exp(-1 / np.pi * sum_) * np.cos(np.pi * sum_)
            result = np.dot(langer_c, vec)
            return result

        # Initial point
        x0 = np.array([4.65116802, 4.42985893, 1.74720157, 4.29727392])
        y0 = langermann(x0)

        optimiser = ScipyOptimizer(method=method, bounds=[(0, 10)] * 4)
        result = optimiser.minimize(langermann, x0, keep_history=True)
        history_costs = [hel.value for hel in result.history]
        assert result.opt_value <= y0
        assert result.opt_value == min(history_costs)
