from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import Bounds as ScipyBounds
from scipy.optimize import OptimizeResult

from orquestra.opt.api import (
    CallableWithGradient,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from orquestra.opt.history.recorder import RecorderFactory
from orquestra.opt.history.recorder import recorder as _recorder
from orquestra.opt.optimizers.pso.topologies import StarTopology, SwarmTopology

Bounds = Union[ScipyBounds, Sequence[Tuple[float, float]], Tuple[float, float]]


def _get_bounds_like_array(
    bounds: Bounds,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """
    Casts a Bounds object to an object with two entries: the first being the lower
    bounds and the second being the upper bounds.

    Args:
        bounds: A Bounds object, which can be a ScipyBounds object, a sequence of
            tuples, one tuple being the bonds per parameter, or a tuple of two
            floats, which are the lower and upper bounds for all parameters.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[float, float]]: Lower and
            upper bounds.
    """
    if isinstance(bounds, ScipyBounds):
        return bounds.lb, bounds.ub
    else:
        _bounds = np.array(bounds).T
        return _bounds[0], _bounds[1]


class PSOOptimizer(Optimizer):
    def __init__(
        self,
        swarm_size: int,
        bounds: Bounds,
        inertia: float = 0.88,
        affinity_towards_best_particle_position: float = 0.6,
        affinity_towards_best_swarm_position: float = 0.26,
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
        """
        Constructor of the Particle Swarm Optimizer for continuous variables.
        When you call `.minimize`, the initial positions of the particles can be
        set using the method `get_initial_params`, which by default samples
        uniformly from the parameter space.

        Args:
            swarm_size: Number of particles in the swarm.
            bounds: A Bounds object, which can be a
                scipy.optimize.Bounds object, a sequence of tuples, one tuple
                being the bonds per parameter, or a tuple of two floats, which
                are the lower and upper bounds for all parameters.
            inertia: Velocities of the particles keep a fraction of
                their previous value, which is the inertia.
            affinity_towards_best_particle_position: Dictates the scale of the
                affinity of a particle towards the best position it has
                experienced in the past. The larger this is, the more the
                particle will be drawn towards the best position it has seen.
            affinity_towards_best_swarm_position: Dictates the scale
                of the affinity of a particle towards the best position the
                swarm has experienced in the past. The larger this is, the
                more the particle will be drawn towards the best position the
                swarm has seen.
            patience: Number of iterations that the optimizer will wait before
                stopping, if an improvement of `delta` has not been made in
                the previous `patience` iterations, by default None. If `None`,
                either `max_iterations` or `max_fevals` must be set.
            delta: Minimum improvement that the optimizer must experience in
                `patience` steps for it not to stop. Valid only if `patience`
                is not None.
            max_iterations: Maximum number of updates of the whole swarm. If
                None, either `max_fevals` or `patience` must be set for the
                optimizer to stop.
            max_fevals: Maximum number of function evaluations. If None,
                either `max_iterations` or `patience` must be set for the
                optimizer to stop.
            learning_rate: Velocities will be updated proportionally to the
            learning rate. Modifying this parameter is equivalent to
                multiplying `inertia`,
                `affinity_towards_best_particle_position` and
                `affinity_towards_best_swarm_position` altogether.
            velocity_bounds: Velocity bounds which can avoid the velocity to
                explode. If None, no bounds are imposed.
            topology_constructor: Class that receives the number of dimensions
                in the initialiser and creates a SwarmTopology.
            seed: Random seed for the numpy random number generator.
            recorder: Recorder factory for keeping history of calls to the
                objective function.
        """
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
        ), (
            "At least one of patience, max_iterations or max_fevals "
            "must be specified to stop PSO"
        )
        super().__init__(recorder=recorder)

        self.patience = patience
        self.delta = delta
        self.max_iterations = max_iterations
        self.max_fevals = max_fevals

        # Other attributes:
        self.swarm_size = swarm_size
        self.random_number_generator = np.random.default_rng(seed)
        self.bounds = _get_bounds_like_array(bounds)
        self.scale = self.bounds[1] - self.bounds[0]
        self.shift = self.bounds[0]
        self.function_at_best_positions = np.ones(swarm_size, dtype=float) * np.inf
        self.inertia = inertia
        self.topology_constructor = topology_constructor
        self.affinity_towards_best_particle_position = (
            affinity_towards_best_particle_position
        )
        self.affinity_towards_best_swarm_position = affinity_towards_best_swarm_position
        self.learning_rate = learning_rate
        self.velocity_bounds = (
            _get_bounds_like_array(velocity_bounds) if velocity_bounds else None
        )

    def _get_initial_velocities(self, dimensions: int) -> np.ndarray:
        """
        Initialises velocities for the particles in the swarm for a given number of
        dimensions.

        Args:
            dimensions: Number of dimensions of the problem.

        Returns:
            np.ndarray: (N, dimensions) array, where N is the swarm size.
        """
        velocities = self.random_number_generator.uniform(
            0, 1, (self.swarm_size, dimensions)
        )
        if self.velocity_bounds is None:
            velocities = 0.5 * (self.scale * velocities - self.shift)
        else:
            scale = self.velocity_bounds[1] - self.velocity_bounds[0]
            shift = self.velocity_bounds[0]
            velocities = scale * velocities + shift
        return velocities

    def get_initial_params(self, dimensions: int) -> np.ndarray:
        """
        Uniformly samples the parameter space to initialise the particles in the swarm.


        Args:
            dimensions: Number of dimensions of the problem.

        Returns:
            np.ndarray: (N, dimensions) array, where N is the swarm size.
        """
        return (
            self.random_number_generator.uniform(
                0, self.scale, (self.swarm_size, dimensions)
            )
            + self.shift
        )

    def _update_positions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        best_positions: np.ndarray,
        topology: SwarmTopology,
    ) -> None:
        """
        In-place modification of the positions and velocities of the
        particles in the swarm.

        Args:
            positions: (N, D) array of the current positions of the particles in the
                swarm.
            velocities: (N, D) array of the current velocities of the particles in
                the swarm.
            best_positions: (N, D) array of the best positions the particles in the
                swarm that have been seen.
            topology: Topology object that dictates the best swarm position.
        """
        dimensions = positions.shape[-1]
        inertia_term = self.inertia * velocities
        # The affinity terms are quite random, as they change for every dimension,
        # every particle and every call to this function. Maybe we could have levels
        # of randomness here.
        affinity_towards_best_particle_position_term = (
            self.random_number_generator.uniform(
                0,
                self.affinity_towards_best_particle_position,
                (self.swarm_size, dimensions),
            )
            * np.subtract(best_positions, positions)
        )
        affinity_towards_best_swarm_position_term = (
            self.random_number_generator.uniform(
                0,
                self.affinity_towards_best_swarm_position,
                (self.swarm_size, dimensions),
            )
            * np.subtract(topology.best_swarm_position, positions)
        )
        velocities[:] = (
            inertia_term
            + affinity_towards_best_particle_position_term
            + affinity_towards_best_swarm_position_term
        )

        if self.velocity_bounds is not None:
            velocities[:] = np.clip(
                velocities, self.velocity_bounds[0], self.velocity_bounds[1]
            )
        positions[:] = positions + self.learning_rate * velocities

    def _minimize(
        self,
        cost_function: Union[CallableWithGradient, Callable],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        dimensions = initial_params.shape[-1]
        if initial_params.ndim == 2:
            assert initial_params.shape[0] == self.swarm_size, (
                "If you provide an (N, D) array as initial parameters for the swarm, "
                "N must equal the size of the swarm."
            )
            self.positions = initial_params
        else:
            self.positions = self.get_initial_params(dimensions)
            # Set first particle in the swarm to the initial parameters:
            self.positions[0, :] = initial_params
        self.best_positions = self.positions.copy()
        self.topology = self.topology_constructor(dimensions)
        self.velocities = self._get_initial_velocities(dimensions)

        n_iterations_since_last_improvement = 0
        best_swarm_value_checkpoint = np.inf
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
            # Update the positions:
            self._update_positions(
                self.positions, self.velocities, self.best_positions, self.topology
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
