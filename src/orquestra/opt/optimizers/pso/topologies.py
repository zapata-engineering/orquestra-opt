import abc

import numpy as np


class SwarmTopology(abc.ABC):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.best_swarm_value = np.infty
        self.best_swarm_position = np.empty(dimensions, dtype=float)

    @abc.abstractmethod
    def update_best(self, positions, function_at_positions):
        raise NotImplementedError


class StarTopology(SwarmTopology):
    def update_best(self, positions, function_at_positions):
        new_min_value = np.min(function_at_positions)
        if new_min_value < self.best_swarm_value:
            self.best_swarm_value = new_min_value
            self.best_swarm_position[:] = positions[np.argmin(function_at_positions)]
