################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import numpy as np
import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.search_points_optimizer import SearchPointsOptimizer


@pytest.fixture(
    params=[
        [
            np.array([0, 0]),
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([1.3, 0]),
            np.array([0.000001, -1.3]),
        ],
        np.array(
            [
                np.array([0, 0]),
                np.array([1, 1]),
                np.array([0, 1]),
                np.array([1.3, 0]),
                np.array([0.000001, -1.3]),
            ]
        ),
        [
            np.array([1, 1]),
            np.array([0, 0]),
        ],
    ]
)
def parameter_values_list(request):
    return request.param


class TestSearchPointsOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)

    @pytest.fixture()
    def optimizer(self, parameter_values_list):
        return SearchPointsOptimizer(parameter_values_list=parameter_values_list)

    def test_assertion_raised_when_no_points_passed(self):
        with pytest.raises(AssertionError):
            SearchPointsOptimizer(parameter_values_list=[])
