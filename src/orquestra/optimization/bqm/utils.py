import numpy as np
from typing import Union, Tuple
from dimod import BinaryQuadraticModel


def evaluate_bitstring_for_qubo(
    bitstring: Union[str, np.ndarray, Tuple[int, ...]], qubo: BinaryQuadraticModel
):
    """Returns the energy associated with given bitstring for a specific qubo.

    Args:
        bitstring: string/array of zeros and ones representing solution to a qubo .
        qubo: qubo that we want to evaluate for.

    Returns:
        float: energy associated with a bistring
    """
    return qubo.energy({i: bit for i, bit in enumerate(map(int, bitstring))})
