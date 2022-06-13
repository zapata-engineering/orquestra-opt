################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from .basin_hopping import BasinHoppingOptimizer

# we are using try/except to avoid errors when optional requirements are not installed
try:
    from .cma_es_optimizer import CMAESOptimizer
except ModuleNotFoundError:
    pass
try:
    from .qiskit_optimizer import QiskitOptimizer
except ModuleNotFoundError:
    pass
try:
    from .scikit_quant_optimizer import ScikitQuantOptimizer
except ModuleNotFoundError:
    pass
from .scipy_optimizer import ScipyOptimizer
from .search_points_optimizer import SearchPointsOptimizer
from .simple_gradient_descent import SimpleGradientDescent
