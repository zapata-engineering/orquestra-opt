import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.tensor_train_optimizer import TensorTrainOptimizer


@pytest.fixture(
    params=[
        {
            "n_grid_points": 25,
            "n_evaluations": 10000,
            "bounds": (-2.0, 2.0),
            "maximum_tensor_train_rank": 4,
            "n_rounds": 3,
            "maximum_number_of_candidates": 3,
            "random_seed": 1,
        }
    ]
)
def optimizer(request):
    return TensorTrainOptimizer(**request.param)


class TestTensorTrainOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)
