import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers import ScipyLeastSquares


@pytest.fixture(params=[{"bounds": (-2.0, 2.0), "n_optimization_steps": 10000}])
def optimizer(request):
    return ScipyLeastSquares(**request.param)


class TestLeastSquaresOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)
