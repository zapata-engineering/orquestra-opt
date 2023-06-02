################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import warnings
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
from .pso.continuous_pso_optimizer import _get_bounds_like_array


class _CostFunctionWithBestValueType(type):
    def __instancecheck__(cls, obj: object) -> bool:
        if callable(obj) and hasattr(obj, "best_value") and hasattr(obj, "best_params"):
            return True
        return False


class _CostFunctionWithBestValue(metaclass=_CostFunctionWithBestValueType):
    best_value: float = np.inf
    best_params: np.ndarray = np.empty(1)

    def __init__(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        constraints: Optional[Tuple[Dict[str, Callable]]],
        bounds: Union[
            scipy.optimize.Bounds,
            Sequence[Tuple[float, float]],
            None,
        ] = None,
    ) -> None:
        self.cost_function = cost_function
        # Inherit all attributes from the cost function
        for attr in dir(cost_function):
            if not attr.startswith("__"):
                if attr in [
                    "best_value",
                    "best_params",
                    "_are_params_bounded",
                    "_are_params_constrained",
                ]:
                    warnings.warn(
                        f"Attribute {attr} of the cost function is being overwritten "
                        "by the optimizer.",
                        UserWarning,
                    )
                setattr(self, attr, getattr(cost_function, attr))
        self.best_params.fill(np.nan)
        self.constraints = constraints
        self.bounds = _get_bounds_like_array(bounds) if bounds is not None else None

    def __call__(self, parameters: np.ndarray) -> float:
        value = self.cost_function(parameters)
        if (
            value < self.best_value
            and self._are_params_bounded(parameters)
            and self._are_params_constrained(parameters)
        ):
            self.best_value = value
            self.best_params = parameters
        return value

    def _are_params_bounded(self, params: np.ndarray) -> bool:
        if self.bounds:
            return bool(
                np.all(
                    np.bitwise_and(params >= self.bounds[0], params <= self.bounds[1])
                )
            )
        return True

    def _are_params_constrained(self, params: np.ndarray) -> bool:
        if self.constraints:
            for constraint in self.constraints:
                constraint_args = constraint.get("args", ())
                assert isinstance(
                    constraint_args, tuple
                ), "If you pass args to the constraint, they must be a tuple."
                if constraint["type"] == "eq":
                    if not np.isclose(constraint["fun"](params, *constraint_args), 0):
                        return False
                elif constraint["type"] == "ineq":
                    if constraint["fun"](params, *constraint_args) < 0:
                        return False
        return True


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
        store_best_parameters: bool = True,
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
            store_best_parameters: whether to wrap the cost function to store the best
                parameters the optimizer has seen so far.

        """  # noqa: E501
        super().__init__(recorder=recorder)
        self.method = method
        if options is None:
            options = {}
        self.options = options
        self.constraints = constraints
        self.bounds = bounds
        self.store_best_parameters = store_best_parameters

    def _preprocess_cost_function(
        self, cost_function: Union[CallableWithGradient, Callable]
    ) -> Union[CallableWithGradient, Callable, _CostFunctionWithBestValue]:
        if self.store_best_parameters:
            return _CostFunctionWithBestValue(
                cost_function, self.constraints, self.bounds
            )
        else:
            return cost_function

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
        if isinstance(cost_function, _CostFunctionWithBestValue):
            opt_value = cost_function.best_value
            opt_params = cost_function.best_params
        else:
            opt_value = result.fun
            opt_params = result.x

        nit = result.get("nit", None)
        nfev = result.get("nfev", None)

        return optimization_result(
            opt_value=opt_value,
            opt_params=opt_params,
            nit=nit,
            nfev=nfev,
            **construct_history_info(cost_function, keep_history),  # type: ignore
        )
