from typing import Callable, Union

from .recorder import (
    ArtifactRecorder,
    ArtifactRecorderWithGradient,
    SimpleRecorder,
    SimpleRecorderWithGradient,
    recorder,
)

AnyRecorder = Union[SimpleRecorder, ArtifactRecorder]
RecorderFactory = Callable[[Callable], AnyRecorder]
