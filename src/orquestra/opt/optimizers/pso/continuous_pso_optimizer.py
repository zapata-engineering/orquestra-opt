from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy
from scipy.optimize import OptimizeResult

from orquestra.opt.api import (
    CostFunction,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from orquestra.opt.history.recorder import RecorderFactory, recorder as _recorder
from orquestra.opt.optimizers.pso.topologies import StarTopology, SwarmTopology

Bounds = Union[
    scipy.optimize.Bounds, Sequence[Tuple[float, float]], Tuple[float, float]
]


def _get_bounds_like_array(bounds: Bounds) -> np.ndarray:
    if isinstance(bounds, scipy.optimize.Bounds):
        return bounds.lb, bounds.ub
    else:
        return np.array(bounds).T


class PSOOptimizer(Optimizer):
    def __init__(
        self,
        swarm_size: int,
        dimensions: int,
        bounds: Bounds,
        inertia: float,
        affinity_towards_best_particle_position: float,
        affinity_towards_best_swarm_position: float,
        patience: Optional[int] = None,
        delta: float = 1e-10,
        max_iterations: Optional[int] = None,
        max_fevals: Optional[int] = None,
        learning_rate: float = 1.0,
        velocity_bounds: Optional[Bounds] = None,
        topology_constructor: Callable[[int], SwarmTopology] = StarTopology,
        seed: Optional[int] = None,
        recorder: RecorderFactory = _recorder,
    ):
        assert 0 < learning_rate <= 1.0, "Learning rate must be in (0, 1]."
        assert 0 < inertia <= 1, "Inertia must be in (0, 1]"
        assert (
            affinity_towards_best_particle_position >= 0
        ), "Affinity towards best particle position must be non-negative"
        assert (
            affinity_towards_best_swarm_position >= 0
        ), "Affinity towards best swarm position must be non-negative"

        # Convergence control:
        assert (
            patience is not None or max_iterations is not None or max_fevals is not None
        ), "At least one of patience, max_iterations or max_fevals must be specified to stop PSO"
        super().__init__(recorder=recorder)

        self.patience = patience
        self.delta = delta
        self.max_iterations = max_iterations
        self.max_fevals = max_fevals

        # Other attributes:
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.random_number_generator = np.random.default_rng(seed)
        self.bounds = _get_bounds_like_array(bounds)
        self.scale = self.bounds[1] - self.bounds[0]
        self.shift = self.bounds[0]
        self.positions = self.get_initial_params()
        self.best_positions = self.positions.copy()
        self.function_at_best_positions = np.ones(swarm_size, dtype=float) * np.infty
        self.topology = topology_constructor(dimensions)
        self.inertia = inertia
        self.affinity_towards_best_particle_position = (
            affinity_towards_best_particle_position
        )
        self.affinity_towards_best_swarm_position = affinity_towards_best_swarm_position
        self.learning_rate = learning_rate
        self.velocities = self.random_number_generator.uniform(
            0, 1, (swarm_size, dimensions)
        )
        if velocity_bounds is None:
            self.velocity_bounds = None
            self.velocities = 0.5 * (self.scale * self.velocities - self.shift)
        else:
            self.velocity_bounds = _get_bounds_like_array(velocity_bounds)
            scale = self.velocity_bounds[1] - self.velocity_bounds[0]
            shift = self.velocity_bounds[0]
            self.velocities = scale * self.velocities + shift

    def get_initial_params(self) -> np.ndarray:
        return (
            self.random_number_generator.uniform(
                0, self.scale, (self.swarm_size, self.dimensions)
            )
            + self.shift
        )

    def update_positions(self):
        inertia_term = self.inertia * self.velocities
        # The affinity terms are quite random, as they change for every dimension, every particle and every
        # call to this function. Maybe we could have levels of randomness here.
        affinity_towards_best_particle_position_term = (
            self.random_number_generator.uniform(
                0,
                self.affinity_towards_best_particle_position,
                (self.swarm_size, self.dimensions),
            )
            * np.subtract(self.best_positions, self.positions)
        )
        affinity_towards_best_swarm_position_term = (
            self.random_number_generator.uniform(
                0,
                self.affinity_towards_best_swarm_position,
                (self.swarm_size, self.dimensions),
            )
            * np.subtract(self.topology.best_swarm_position, self.positions)
        )
        self.velocities[:] = (
            inertia_term
            + affinity_towards_best_particle_position_term
            + affinity_towards_best_swarm_position_term
        )

        if self.velocity_bounds is not None:
            self.velocities[:] = np.clip(
                self.velocities, self.velocity_bounds[0], self.velocity_bounds[1]
            )
        self.positions[:] = self.positions + self.learning_rate * self.velocities

    def _minimize(
        self,
        cost_function: CostFunction,
        initial_params: Optional[np.ndarray] = None,
        keep_history: bool = False,
    ) -> OptimizeResult:
        if initial_params is not None:
            assert self.positions.shape == initial_params.shape
            self.positions = initial_params
        n_iterations_since_last_improvement = 0
        best_swarm_value_checkpoint = np.infty
        iterations = 0
        fevals = 0
        while True:
            # Evaluate the cost function at the current positions:
            for particle_index in range(self.swarm_size):
                value = cost_function(self.positions[particle_index])
                fevals += 1
                if self.function_at_best_positions[particle_index] > value:
                    self.function_at_best_positions[particle_index] = value
                    self.best_positions[particle_index] = self.positions[particle_index]
            self.topology.update_best(
                self.best_positions, self.function_at_best_positions
            )
            iterations += 1
            if self.patience is not None:
                if (
                    best_swarm_value_checkpoint - self.topology.best_swarm_value
                    < self.delta
                ):
                    n_iterations_since_last_improvement += 1
                else:
                    n_iterations_since_last_improvement = 0
                    best_swarm_value_checkpoint = self.topology.best_swarm_value
                if n_iterations_since_last_improvement >= self.patience:
                    break
            if self.max_iterations is not None:
                if iterations >= self.max_iterations:
                    break
            if self.max_fevals is not None:
                if fevals >= self.max_fevals:
                    break
        return optimization_result(
            opt_value=self.topology.best_swarm_value,
            opt_params=self.topology.best_swarm_position,
            nit=iterations,
            nfev=fevals,
            **construct_history_info(cost_function, keep_history)  # type: ignore
        )
