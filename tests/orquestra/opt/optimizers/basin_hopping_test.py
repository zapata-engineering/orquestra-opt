################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.basin_hopping import BasinHoppingOptimizer


@pytest.fixture(
    params=[
        {
            "niter": 3,
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "BFGS",
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "L-BFGS-B",
                "options": {
                    "ftol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "Nelder-Mead",
                "options": {
                    "fatol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "SLSQP",
                "options": {
                    "ftol": 1e-7,
                },
            },
        },
        {
            "niter": 3,
            "minimizer_kwargs": {
                "method": "COBYLA",
                "options": {"maxiter": 1000000, "tol": 1e-7},
            },
        },
    ]
)
def optimizer(request):
    return BasinHoppingOptimizer(**request.param)


class TestBasinHoppingOptimizer:
    @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
    def test_optimizer_satisfies_contracts(self, contract, optimizer):
        assert contract(optimizer)
