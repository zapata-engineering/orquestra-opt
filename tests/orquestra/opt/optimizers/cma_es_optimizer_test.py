################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.mock_objects import mock_cost_function
from orquestra.opt.optimizers.cma_es_optimizer import CMAESOptimizer


@pytest.fixture(scope="function")
def optimizer():
    return CMAESOptimizer(sigma_0=0.1)


class TestCMAESOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)

    def test_cmaes_specific_fields(self):
        results = CMAESOptimizer(
            sigma_0=0.1, options={"maxfevals": 99, "popsize": 5}
        ).minimize(mock_cost_function, initial_params=[0, 0], keep_history=True)

        assert "cma_xfavorite" in results
        assert isinstance(results["cma_xfavorite"], list)
        assert len(results["history"]) == 100
