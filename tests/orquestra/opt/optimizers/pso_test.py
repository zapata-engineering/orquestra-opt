import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.pso import PSOOptimizer


@pytest.fixture(
    params=[
        {
            "swarm_size": 10,
            "bounds": (-2.0, 2.0),
            "inertia": 1.0,
            "affinity_towards_best_particle_position": 1.0,
            "affinity_towards_best_swarm_position": 1.0,
            "max_iterations": 100,
            "learning_rate": 1.0,
        }
    ]
)
def optimizer(request):
    return PSOOptimizer(**request.param)


class TestPSOOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)
