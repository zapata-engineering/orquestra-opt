from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import OptimizeResult
from ttopt import TTOpt

from orquestra.opt.api.functions import CallableWithGradient
from orquestra.opt.optimizers.pso.continuous_pso_optimizer import (
    Bounds,  # TODO: where should these Bounds live?
)
from orquestra.opt.optimizers.pso.continuous_pso_optimizer import _get_bounds_like_array

from ..api import CostFunction, Optimizer, construct_history_info, optimization_result
from ..history.recorder import RecorderFactory
from ..history.recorder import recorder as _recorder


def _get_n_evaluations_per_candidate(
    evaluation_budget: int, n_rounds: int, n_candidates_per_round: int
) -> List[int]:
    """
    Computes how many evaluations correspond to each candidate in the fairest possible
    way given an evaluation budget. The first round always includes only one candidate,
    which is the initial optimisation. The remaining rounds include
    `n_candidates_per_round` candidates.

    Args:
        evaluation_budget: Total number of evaluations available for all the
            candidates.
        n_rounds: Number of rounds for the optimization.
        n_candidates_per_round: Number of candidates to evaluate in each round.

    Returns:
        A list of length (n_rounds - 1) * n_candidates_per_round + 1, with the number of
        evaluations for each candidate.
    """
    total_candidates = (n_rounds - 1) * n_candidates_per_round + 1
    result = [
        int(x)
        for x in np.diff(
            np.round(np.linspace(0, evaluation_budget, total_candidates + 1))
        )
    ]
    if 0 in result:
        raise ValueError(
            f"Cannot allocate {evaluation_budget} evaluations to {total_candidates}"
            "candidates, because at least one candidate would get 0 evaluations. "
            f"Assigned valuations are {result}"
        )
    return result


def _get_tighter_bounds(
    candidate: np.ndarray,
    bounds: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a candidate and the bounds it lives in, returns a tighter bounds for the
    candidate based on how close it is to the boundary of the hyperbox defined by the
    bounds, and based on the scale of the bounds (i.e. the distance between the lower
    and upper bounds).

    Args:
        candidate: An array of parameters.
        bounds: Either a tuple of floats (lower_bound, upper_bound) or a tuple of arrays
            (lower_bound, upper_bound), where lower_bound and upper_bound are arrays of
            the same shape as candidate.

    Returns:
        A tuple of arrays (lower_bound, upper_bound), where lower_bound and upper_bound
        are arrays of the same shape as candidate.
    """
    lower_bound, upper_bound = bounds
    dist_to_lower_bound = candidate - lower_bound
    dist_to_upper_bound = upper_bound - candidate
    bounds_scale = (upper_bound - lower_bound) / 4
    if isinstance(bounds_scale, float):
        bounds_scale = np.ones_like(candidate) * bounds_scale
    closest_distances = np.max(
        np.vstack((dist_to_lower_bound, dist_to_upper_bound)), axis=0
    )
    distances = np.min(np.vstack((closest_distances, bounds_scale)), axis=0)
    return candidate - distances, candidate + distances


class _CostFunctionWithBestCandidates:
    def __init__(
        self,
        cost_function: CostFunction,
        n_candidates: int,
        bounds: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    ):
        """
        A wrapper for the cost function, which will keep information about the best
        candidates found so far.

        Args:
            cost_function: A cost function.
            n_candidates: Number of candidates to keep track of.
            bounds: Either a tuple of floats (lower_bound, upper_bound) or a tuple of
            arrays (lower_bound, upper_bound), where lower_bound and upper_bound are
            arrays of the same shape.
        """
        self.cost_function = cost_function
        # self should have access to cost_function attributes:
        for attr in dir(cost_function):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(cost_function, attr))
        self.candidates: List[dict] = [{}] * n_candidates
        self.candidates_hashes = [0] * n_candidates
        self.bounds = bounds
        self.nfev = 0
        self.candidate_index = 0
        self.first_exploration = True

    def __call__(self, parameters: np.ndarray) -> float:
        cost = self.cost_function(parameters)
        self.nfev += 1
        if self.first_exploration:
            for i, candidate in enumerate(self.candidates):
                if not candidate or cost <= candidate["cost"]:
                    parameters_hash = hash(parameters.tobytes())
                    if parameters_hash in self.candidates_hashes:
                        self.candidates[i]["bounds"] = _get_tighter_bounds(
                            parameters, self.bounds
                        )
                    else:
                        self.candidates.insert(
                            i,
                            {
                                "cost": cost,
                                "parameters": parameters,
                                "bounds": _get_tighter_bounds(parameters, self.bounds),
                            },
                        )
                        self.candidates.pop()
                        self.candidates_hashes.insert(i, parameters_hash)
                        self.candidates_hashes.pop()
                    break
            return cost
        candidate = self.candidates[self.candidate_index]
        if not candidate or cost <= candidate["cost"]:
            parameters_hash = hash(parameters.tobytes())
            if parameters_hash == self.candidates_hashes[self.candidate_index]:
                self.candidates[self.candidate_index]["bounds"] = _get_tighter_bounds(
                    parameters, self.bounds
                )
                return cost
            self.candidates[self.candidate_index] = {
                "cost": cost,
                "parameters": parameters,
                "bounds": _get_tighter_bounds(parameters, self.bounds),
            }
            self.candidates_hashes[self.candidate_index] = parameters_hash
        return cost

    def gather_best_candidates(self) -> List[dict]:
        """
        Creates a list of the best candidates found so far.
        """
        return [c for c in self.candidates if "parameters" in c.keys()]


class TensorTrainOptimizer(Optimizer):
    def __init__(
        self,
        n_grid_points: Union[int, Sequence[int]],
        n_evaluations: int,
        bounds: Bounds,
        maximum_tensor_train_rank: int = 4,
        n_rounds: int = 1,
        maximum_number_of_candidates: int = 5,
        recorder: RecorderFactory = _recorder,
    ):
        """
        Constructor of a TensorTrainOptimizer. This optimizer uses a tensor-train
        representation of the cost function to find the minimum of the cost function.
        It uses the `ttopt` library to do so. This optimizer finds the minimum of the
        cost function at points given in a grid. This class extends this concept by
        being able to pick the best candidate solutions in the grid, and then one
        simply iterates over those candidates on finer grids centered around each
        candidate to replace each candidate with a better one found in the finer grid.

        Args:
            n_grid_points: Number of grid points to use in each dimension.
            n_evaluations: Cost function evaluations budget.
            bounds: Lower and upper bounds for each parameter.
            maximum_tensor_train_rank: Maximum bond dimension for the tensor train.
            n_rounds: Number of optimisation rounds. If 1, then only the initial grid
                is used. If 2 or more, then the best candidates found in the initial
                grid are used to create finer grids. Each of these finer grids will
                get finer and finer depending on the requested number of rounds.
            maximum_number_of_candidates: Number of candidates to keep track of.
            recorder Recorder factory for keeping history of calls to the objective
                function.
        """
        super().__init__(recorder=recorder)
        self.n_grid_points = n_grid_points
        self.n_evaluations = n_evaluations
        self.bounds = _get_bounds_like_array(bounds)
        self.maximum_tensor_train_rank = maximum_tensor_train_rank
        self.n_rounds = n_rounds
        self.maximum_number_of_candidates = maximum_number_of_candidates
        self.evaluations_per_candidate = _get_n_evaluations_per_candidate(
            n_evaluations, n_rounds, maximum_number_of_candidates
        )

    def _preprocess_cost_function(
        self, cost_function: CostFunction
    ) -> _CostFunctionWithBestCandidates:
        """
        Wraps the cost function in a CostFunctionWithBestCandidates.
        """
        return _CostFunctionWithBestCandidates(
            cost_function, self.maximum_number_of_candidates, self.bounds
        )

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        assert isinstance(cost_function, _CostFunctionWithBestCandidates)
        n_dim = initial_params.size
        evaluations_per_candidate_iterator = iter(self.evaluations_per_candidate)
        for i in range(self.n_rounds):
            best_candidates = cost_function.gather_best_candidates()
            if len(best_candidates) == 0:
                ttopt = TTOpt(
                    f=cost_function,
                    d=n_dim,
                    a=self.bounds[0],
                    b=self.bounds[1],
                    n=self.n_grid_points,
                    evals=next(evaluations_per_candidate_iterator),
                    is_vect=False,
                    is_func=True,
                )
                ttopt.minimize(self.maximum_tensor_train_rank)
            else:
                cost_function.first_exploration = False
                for idx, candidate in enumerate(best_candidates):
                    cost_function.candidate_index = idx
                    cost_function.bounds = candidate["bounds"]
                    ttopt = TTOpt(
                        f=cost_function,
                        d=n_dim,
                        a=candidate["bounds"][0],
                        b=candidate["bounds"][1],
                        n=self.n_grid_points,
                        evals=next(evaluations_per_candidate_iterator),
                        is_vect=False,
                        is_func=True,
                    )
                    ttopt.minimize(self.maximum_tensor_train_rank)
        best_candidate = cost_function.gather_best_candidates()[0]
        return optimization_result(
            opt_value=best_candidate["cost"],
            opt_params=best_candidate["parameters"],
            nit=None,
            nfev=cost_function.nfev,
            **construct_history_info(cost_function, keep_history),  # type: ignore
        )
