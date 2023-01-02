################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import scipy

from ..api import (
    CallableWithGradient,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from ..history.recorder import RecorderFactory
from ..history.recorder import recorder as _recorder


class _CostFunctionWithBestValue:
    def __init__(self, cost_function: Union[CallableWithGradient, Callable]) -> None:
        self.cost_function = cost_function
        # Inherit all attributes from the cost function
        for attr in dir(cost_function):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(cost_function, attr))
        self.best_value = np.inf
        self.best_params = np.empty(1)
        self.best_params.fill(np.nan)

    def __call__(self, params: np.ndarray) -> float:
        value = self.cost_function(params)
        if value < self.best_value:
            self.best_value = value
            self.best_params = params
        return value


class ScipyOptimizer(Optimizer):
    def __init__(
        self,
        method: str,
        constraints: Optional[Tuple[Dict[str, Callable]]] = None,
        bounds: Union[
            scipy.optimize.Bounds,
            Sequence[Tuple[float, float]],
            None,
        ] = None,
        options: Optional[Dict] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Integration with scipy optimizers. Documentation for this module is minimal,
        please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
            method: defines the optimization method
            constraints: list of constraints in the scipy compatible format.
            bounds: bounds for the parameters in the scipy compatible format.
            options: dictionary with additional options for the optimizer.
            recorder: recorder object which defines how to store
                the optimization history.

        """  # noqa: E501
        super().__init__(recorder=recorder)
        self.method = method
        if options is None:
            options = {}
        self.options = options
        self.constraints = [] if constraints is None else constraints
        self.bounds = bounds

    def _preprocess_cost_function(
        self, cost_function: Union[CallableWithGradient, Callable]
    ) -> _CostFunctionWithBestValue:
        return _CostFunctionWithBestValue(cost_function)

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        """
        Minimizes given cost function using functions from scipy.minimize.

        Args:
            cost_function: python method which takes numpy.ndarray as input
            initial_params: initial parameters to be used for optimization
            keep_history: flag indicating whether history of cost function
                evaluations should be recorded.

        """
        assert isinstance(cost_function, _CostFunctionWithBestValue)
        jacobian = None
        if isinstance(cost_function, CallableWithGradient) and callable(
            getattr(cost_function, "gradient")
        ):
            jacobian = cost_function.gradient

        result = scipy.optimize.minimize(
            cost_function,
            initial_params,
            method=self.method,
            options=self.options,
            constraints=self.constraints,
            bounds=self.bounds,
            jac=jacobian,
        )
        opt_value = cost_function.best_value
        opt_params = cost_function.best_params

        nit = result.get("nit", None)
        nfev = result.get("nfev", None)

        return optimization_result(
            opt_value=opt_value,
            opt_params=opt_params,
            nit=nit,
            nfev=nfev,
            **construct_history_info(cost_function, keep_history)  # type: ignore
        )
