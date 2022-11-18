import pytest

from orquestra.opt.api.optimizer_test import OPTIMIZER_CONTRACTS
from orquestra.opt.optimizers.pso import PSOOptimizer, StarTopology
import numpy as np


def test_pso():
    pso = PSOOptimizer(
        swarm_size=10,
        dimensions=2,
        bounds=[(-1, 1), (-1, 1)],
        inertia=0.4,
        affinity_towards_best_particle_position=0.4,
        affinity_towards_best_swarm_position=0.4,
        patience=10,
        delta=1e-10,
        max_iterations=100,
        max_fevals=1000,
        learning_rate=1.0,
        velocity_bounds=[(-1, 1), (-1, 1)],
        topology_constructor=StarTopology,
        seed=42,
    )
    result = pso.minimize(
        lambda parameters: np.sum(np.square(parameters)),
        pso.get_initial_params(),
        keep_history=True,
    )
    print(result)


# @pytest.fixture(
#     params=[
#         {
#             "niter": 3,
#         },
#         {
#             "niter": 3,
#             "minimizer_kwargs": {
#                 "method": "BFGS",
#             },
#         },
#         {
#             "niter": 3,
#             "minimizer_kwargs": {
#                 "method": "L-BFGS-B",
#                 "options": {
#                     "ftol": 1e-7,
#                 },
#             },
#         },
#         {
#             "niter": 3,
#             "minimizer_kwargs": {
#                 "method": "Nelder-Mead",
#                 "options": {
#                     "fatol": 1e-7,
#                 },
#             },
#         },
#         {
#             "niter": 3,
#             "minimizer_kwargs": {
#                 "method": "SLSQP",
#                 "options": {
#                     "ftol": 1e-7,
#                 },
#             },
#         },
#         {
#             "niter": 3,
#             "minimizer_kwargs": {
#                 "method": "COBYLA",
#                 "options": {"maxiter": 1000000, "tol": 1e-7},
#             },
#         },
#     ]
# )
# def optimizer(request):
#     return PSOOptimizer(**request.param)
# class TestBasinHoppingOptimizer:
#     @pytest.mark.parametrize("contract", OPTIMIZER_CONTRACTS)
#     def test_optimizer_satisfies_contracts(self, contract, optimizer):
#         assert contract(optimizer)
