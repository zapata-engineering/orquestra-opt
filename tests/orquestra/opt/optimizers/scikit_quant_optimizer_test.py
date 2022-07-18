import numpy as np
import pytest

from orquestra.opt.api.optimizer_test import OptimizerTests
from orquestra.opt.optimizers.scikit_quant_optimizer import ScikitQuantOptimizer


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


class TestScikitQuantOptimizer(OptimizerTests):
    def test_optimizer_succeeds_with_optimizing_rosenbrock_function(
        self, optimizer, rosenbrock_function, keep_history
    ):

        pytest.xfail("Scikit-Quant optimizers dont work well for Rosenbrock function")

    @pytest.mark.parametrize("number_of_params", [2, 3, 100])
    def test_bounds_are_defined_correctly_if_not_provided(
        self, number_of_params, sum_x_squared
    ):

        initial_params = np.zeros(number_of_params)

        optimizer = ScikitQuantOptimizer(method="imfil")

        _ = optimizer.minimize(sum_x_squared, initial_params)

        assert len(optimizer.bounds) == number_of_params
        np.testing.assert_array_equal(optimizer.bounds[0], np.array([-1000, 1000]))

    def test_fails_if_bounds_dont_match_params(self, optimizer, sum_x_squared):
        initial_params = np.ones(5)
        optimizer.bounds = np.array([[-1000, 1000], [-1000, 1000]])
        with pytest.raises(ValueError):
            _ = optimizer.minimize(sum_x_squared, initial_params)

    def test_length_1_input_fails(self, optimizer, sum_x_squared):
        initial_params = np.ones(1)
        with pytest.raises(ValueError):
            _ = optimizer.minimize(sum_x_squared, initial_params)
