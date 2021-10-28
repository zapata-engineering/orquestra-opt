from typing import Optional

import dimod
import numpy as np
from dimod import BinaryQuadraticModel, SampleSet
from openfermion import IsingOperator
from zquantum.core.measurement import Measurements


def convert_qubo_to_openfermion_ising(qubo: BinaryQuadraticModel) -> IsingOperator:
    """Converts dimod BinaryQuadraticModel to OpenFermion IsingOperator object.

    The resulting Openfermion IsingOperator has the following property: for every
    bitstring, its expected value is the same as the energy of the original QUBO.
    In order to ensure this, we had to add a minus sign for the coefficients
    of the linear terms coming from dimod conversion.
    For more context about conventions used please refer to note in
    `convert_measurements_to_sampleset` docstring.

    Args:
        qubo: Object we want to convert

    Returns:
        IsingOperator: IsingOperator representation of the input qubo.

    """
    linear_coeffs, quadratic_coeffs, offset = qubo.to_ising()

    list_of_ising_strings = [f"{offset}[]"]

    for i, value in linear_coeffs.items():
        list_of_ising_strings.append(f"{-value}[Z{i}]")

    for (i, j), value in quadratic_coeffs.items():
        list_of_ising_strings.append(f"{value}[Z{i} Z{j}]")

    ising_string = " + ".join(list_of_ising_strings)
    return IsingOperator(ising_string)


def convert_openfermion_ising_to_qubo(operator: IsingOperator) -> BinaryQuadraticModel:
    """
    Converts dimod Openfermion IsingOperator to BinaryQuadraticModel object.
    The resulting QUBO has the following property:
    For every bitstring, its energy is the same as the expected value of the original
    Ising Hamiltonian. For more context about conventions used please refer to note in
    `convert_measurements_to_sampleset` docstring.

    Note:
        The conversion might not be 100% accurate due to performing floating point
        operations during conversion between Ising and QUBO models.

    Args:
        operator: IsingOperator we want to convert
    Returns:
        qubo: BinaryQuadraticModel representation of the input operator

    """

    if not isinstance(operator, IsingOperator):
        raise TypeError(
            f"Input is of type: {type(operator)}. Only Ising Operators are supported."
        )
    offset = 0
    linear_terms = {}
    quadratic_terms = {}
    for term, coeff in operator.terms.items():
        if len(term) == 0:
            offset = coeff
        if len(term) == 1:
            linear_terms[term[0][0]] = -coeff
        if len(term) == 2:
            quadratic_terms[(term[0][0], term[1][0])] = coeff
        if len(term) > 2:
            raise ValueError(
                "Ising to QUBO conversion works only for quadratic Ising models."
            )

    dimod_ising = BinaryQuadraticModel(
        linear_terms, quadratic_terms, offset, vartype="SPIN"
    )
    return dimod_ising.change_vartype("BINARY", inplace=False)


def convert_sampleset_to_measurements(
    sampleset: SampleSet,
    change_bitstring_convention: bool = False,
) -> Measurements:
    """
    Converts dimod SampleSet to zquantum.core Measurements.
    Works only for the sampleset with "BINARY" vartype and variables being range of
    integers starting from 0.

    Note:
        Since Measurements doesn't hold information about the energy of the samples,
        this conversion is lossy. For more explanation regarding
        change_bitstring_convention please read docs of
        `convert_measurements_to_sampleset`.

    Args:
        sampleset: SampleSet we want to convert
        change_bitstring_convention: whether to flip the bits in bitstrings to, depends
        on the convention one is using (see note).

    """
    if sampleset.vartype != dimod.BINARY:
        raise TypeError("Sampleset needs to have vartype BINARY")
    for i in range(max(sampleset.variables)):
        if sampleset.variables[i] != i:
            raise ValueError(
                "Variables of sampleset need to be ordered list of integers"
            )

    bitstrings = [
        tuple(int(change_bitstring_convention != sample[i]) for i in range(len(sample)))
        for sample in sampleset.samples()
    ]
    return Measurements(bitstrings)


def convert_measurements_to_sampleset(
    measurements: Measurements,
    bqm: Optional[BinaryQuadraticModel] = None,
    change_bitstring_convention: bool = False,
) -> SampleSet:
    """
    Converts dimod SampleSet to zquantum.core Measurements.
    If no bqm is specified, the vartype of the SampleSet will be "BINARY" and the
    energies will be NaN. If bqm is specified, its vartype will be preserved and
    the energy values will be calculated.

    Note:
        The convention commonly used in quantum computing is that 0 in a bitstring
        represents eigenvalue 1 of an Ising Hamiltonian,and 1 represents eigenvalue -1.
        However, there is another convention, used in dimod, where the mapping is
        0 -> -1 and 1 -> 1 instead. Therefore if we try to use the bitstrings coming
        from solving the problem framed in one convention to evaluate energy for
        problem state in the second one, the results will be incorrect. This can be
        fixed with changing the value of `change_bitstring_convention` flag. Currently
        we changed the conventions appropriately in `convert_qubo_to_openfermion_ising`
        and `convert_openfermion_ising_to_qubo`, however, this still might be an issue
        in cases where qubo/Ising representation is created using different tools.
        It's hard to envision a specific example at the time of writing, but experience
        shows that it needs to be handled with caution.
    Args:
        measurements: Measurements object to be converted
        bqm: if provided, SampleSet will include energy values for each sample.
        change_bitstring_convention: whether to flip the bits in bitstrings to, depends
            on the convention one is using (see note).
    Returns:
        SampleSet object
    """
    bitstrings = [
        tuple(int(change_bitstring_convention != bit) for bit in bitstring)
        for bitstring in measurements.bitstrings
    ]

    if not bqm:
        return SampleSet.from_samples(
            bitstrings, "BINARY", [np.nan for _ in bitstrings]
        )
    if bqm.vartype != dimod.BINARY:
        raise TypeError("BQM needs to have vartype BINARY")

    return SampleSet.from_samples_bqm(bitstrings, bqm)
