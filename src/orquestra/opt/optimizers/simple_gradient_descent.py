################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import copy
from typing import Callable, Optional, Union

import numpy as np
from scipy.optimize import OptimizeResult

from ..api import (
    CallableWithGradient,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from ..gradients import finite_differences_gradient
from ..history.recorder import RecorderFactory
from ..history.recorder import recorder as _recorder


class SimpleGradientDescent(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        number_of_iterations: int,
        patience: Optional[int] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Args:
            learning_rate: learning rate.
            number_of_iterations: number of gradient descent iterations.
            patience: number of iterations to wait before early stopping.
            recorder: recorder object which defines how to store
                the optimization history.
        """
        super().__init__(recorder=recorder)
        self.learning_rate = learning_rate

        assert number_of_iterations > 0
        self.number_of_iterations = number_of_iterations
        self.patience = patience

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        """
        Finds the parameters which minimize given cost function, by trying all
        the parameters from the provided list of points.

        Note:
            This optimizer does not require evaluation of the cost function,
            but relies only on gradient evaluation. This means, that if we want to
            keep track of values of the cost functions for each iteration, we
            need to perform extra evaluations. Therefore using `keep_history=True`
            will add extra evaluations that are not necessary for
            the optimization process itself.

        Args:
            cost_function: object representing cost function we want to minimize
            inital_params: initial parameters for the cost function
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded. Using this will increase runtime,
                see note.

        """
        # So that mypy does not complain about missing attributes:
        assert hasattr(cost_function, "gradient")
        current_parameters = copy.deepcopy(initial_params)
        if self.patience is not None:
            best_value = np.inf
            best_iteration = 0
        for iteration in range(self.number_of_iterations):
            gradients = cost_function.gradient(current_parameters)
            current_parameters = current_parameters - (self.learning_rate * gradients)
            if keep_history:
                final_value = cost_function(current_parameters)
            if self.patience is not None:
                if keep_history:
                    current_value = final_value
                else:
                    current_value = cost_function(current_parameters)
                improvement = best_value - current_value
                if improvement > 1e-8:
                    best_value = current_value
                    best_iteration = iteration
                elif iteration - best_iteration >= self.patience:
                    break

        if not keep_history:
            final_value = cost_function(current_parameters)

        return optimization_result(
            opt_value=final_value,
            opt_params=current_parameters,
            nit=iteration + 1,
            nfev=None,
            **construct_history_info(cost_function, keep_history),  # type: ignore
        )

    def _preprocess_cost_function(
        self, cost_function: Union[CallableWithGradient, Callable]
    ) -> CallableWithGradient:
        if not isinstance(cost_function, CallableWithGradient):
            gradient_fn = finite_differences_gradient(cost_function)

            class WrappedCostFunction:
                def __init__(self, cost_function):
                    self.cost_function = cost_function

                def __call__(self, params: np.ndarray) -> float:
                    return self.cost_function(params)

                def gradient(self, params: np.ndarray) -> np.ndarray:
                    return gradient_fn(params)

            cost_function = WrappedCostFunction(cost_function=cost_function)
        return cost_function
