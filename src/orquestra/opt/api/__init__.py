################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Sequence, Tuple, Union

from scipy.optimize import Bounds as ScipyBounds

from .cost_function import CostFunction
from .functions import (
    CallableStoringArtifacts,
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
    FunctionWithGradient,
    FunctionWithGradientStoringArtifacts,
    StoreArtifact,
    function_with_gradient,
    has_store_artifact_param,
)
from .optimizer import (
    NestedOptimizer,
    Optimizer,
    construct_history_info,
    optimization_result,
)
from .save_conditions import SaveCondition, always, every_nth

Bounds = Union[ScipyBounds, Sequence[Tuple[float, float]], Tuple[float, float]]
