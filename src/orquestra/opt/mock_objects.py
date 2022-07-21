################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import random
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from .api import (
    CostFunction,
    NestedOptimizer,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from .api.optimizer import extend_histories
from .history.recorder import RecorderFactory
from .history.recorder import recorder as _recorder


class MockOptimizer(Optimizer):
    def _minimize(
        self, cost_function, initial_params: np.ndarray, keep_history: bool = False
    ):
        if keep_history:
            cost_function = _recorder(cost_function)
        new_parameters = initial_params
        for i in range(len(initial_params)):
            new_parameters[i] += random.random()
        new_parameters = np.array(new_parameters)
        opt_value = cost_function(new_parameters)
        return optimization_result(
            opt_value=opt_value,
            opt_params=new_parameters,
            nit=1,
            nfev=1,
            **construct_history_info(cost_function, keep_history),
        )


def mock_cost_function(parameters: np.ndarray):
    return np.sum(parameters**2)


class MockNestedOptimizer(NestedOptimizer):
    """
    As most mock objects this implementation does not make much sense in itself,
    however it's an example of how a NestedOptimizer could be implemented.

    """

    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner_optimizer

    @property
    def recorder(self) -> RecorderFactory:
        return self._recorder

    def __init__(
        self,
        inner_optimizer: Optimizer,
        n_iters: int,
        recorder: RecorderFactory = _recorder,
    ):
        self._inner_optimizer = inner_optimizer
        self.n_iters = n_iters
        self._recorder = recorder

    def _minimize(
        self,
        cost_function_factory: Callable[[int], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        histories: Dict[str, List] = defaultdict(list)
        histories["history"] = []
        nfev = 0
        current_params = initial_params
        for i in range(self.n_iters):
            if i != 0:
                # Increase the length of params every iteration
                # and repeats optimization with the longer params vector.
                current_params = np.append(current_params, 1)

            # Cost function changes with every iteration of NestedOptimizer
            # because it's dependent on iteration number
            cost_function = cost_function_factory(i)
            if keep_history:
                cost_function = self.recorder(cost_function)
            opt_result = self.inner_optimizer.minimize(cost_function, initial_params)
            nfev += opt_result.nfev
            current_params = opt_result.opt_params
            if keep_history:
                histories = extend_histories(cost_function, histories)  # type: ignore
        return optimization_result(
            opt_value=opt_result.opt_value,
            opt_params=current_params,
            nit=self.n_iters,
            nfev=nfev,
            **histories,
        )
