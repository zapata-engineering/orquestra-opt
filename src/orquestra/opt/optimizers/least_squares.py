import typing

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, least_squares

from orquestra.opt.api import (
    Bounds,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from orquestra.opt.history.recorder import RecorderFactory
from orquestra.opt.history.recorder import recorder as _recorder


class ScipyLeastSquares(Optimizer):
    def __init__(
        self,
        bounds: Bounds,
        n_optimization_steps: int,
        ftol: float = 1e-15,
        xtol: float = 1e-15,
        gtol: float = 1e-15,
        method: typing.Literal["trf", "dogbox", "lm"] = "dogbox",
        diff_step: typing.Optional[npt.ArrayLike] = None,
        recorder: RecorderFactory = _recorder,
    ):
        """
        A wrapper around scipy's least_squares optimizer.

        Args:
            bounds: A Bounds object, which can be a ScipyBounds object, a
                sequence of tuples, one tuple being the bonds per parameter, or a tuple
                of two floats, which are the lower and upper bounds for all parameters.
            n_optimization_steps (int): Maximum number of function evaluations
            ftol: Tolerance for termination by the change of the cost function. Defaults
                to 1e-15.
            xtol: Tolerance for termination by the change of the solution vector.
                Defaults to 1e-15.
            gtol: Tolerance for termination by the norm of the gradient. Defaults to
            1e-15.
            method: Algorithm to perform minimisation. Defaults to "dogbox".
            diff_step: Determines the relative step size for the finite difference
                approximation of the Jacobian. The actual step is computed as
                `x * diff_step`. If None (default), then diff_step is taken to be a
                conventional “optimal” power of machine epsilon for the finite
                difference scheme used (see scipy's documentation). Defaults to 1.
            recorder: Recorder factory for keeping history of calls to the objective
                function.
        """
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.method = method
        self.bounds = bounds
        self.diff_step = diff_step
        self.n_optimization_steps = n_optimization_steps
        super().__init__(recorder)

    def _minimize(
        self, cost_function, initial_params: npt.NDArray, keep_history: bool = False
    ) -> OptimizeResult:
        result = least_squares(
            cost_function,
            initial_params,
            bounds=self.bounds,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            max_nfev=self.n_optimization_steps,
            method=self.method,
            diff_step=self.diff_step,
        )
        opt_value = result.fun
        if isinstance(result.x, np.ndarray):
            opt_value = opt_value.item()
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
