import numpy as np
from openfermion import IsingOperator
import dimod
from dimod import BinaryQuadraticModel, SampleSet
from typing import Optional

from zquantum.core.measurement import Measurements


def convert_qubo_to_openfermion_ising(qubo: BinaryQuadraticModel) -> IsingOperator:
    """Converts dimod BinaryQuadraticModel to OpenFermion IsingOperator object.

    Args:
        qubo: Object we want to convert

    Returns:
        IsingOperator: IsingOperator representation of the input qubo.

    """
    linear_coeffs, quadratic_coeffs, offset = qubo.to_ising()

    list_of_ising_strings = [f"{offset}[]"]

    for i, value in linear_coeffs.items():
        list_of_ising_strings.append(f"{value}[Z{i}]")

    for (i, j), value in quadratic_coeffs.items():
        list_of_ising_strings.append(f"{value}[Z{i} Z{j}]")

    ising_string = " + ".join(list_of_ising_strings)
    return IsingOperator(ising_string)


def convert_openfermion_ising_to_qubo(operator: IsingOperator) -> BinaryQuadraticModel:
    """
    Converts dimod BinaryQuadraticModel to OpenFermion IsingOperator object.
    NOTE: The conversion might not be 100% accurate due to performing floating point operations during conversion between Ising and QUBO models.

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
            linear_terms[term[0][0]] = coeff
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


def convert_sampleset_to_measurements(sampleset: SampleSet) -> Measurements:
    """
    Converts dimod SampleSet to zquantum.core Measurements.
    Works only for the sampleset with "BINARY" vartype and variables being range of integers starting from 0.

    Note:
        Since Measurements doesn't hold information about the energy of the samples, this conversion is lossy.

    Args:
        sampleset: SampleSet we want to convert
    Returns:
        Measurements object

    """
    if sampleset.vartype != dimod.BINARY:
        raise TypeError("Sampleset needs to have vartype BINARY")
    for i in range(max(sampleset.variables)):
        if sampleset.variables[i] != i:
            raise ValueError(
                "Variables of sampleset need to be ordered list of integers"
            )

    bitstrings = [
        tuple(sample[i] for i in range(len(sample))) for sample in sampleset.samples()
    ]
    return Measurements(bitstrings)


def convert_measurements_to_sampleset(
    measurements: Measurements, bqm: Optional[BinaryQuadraticModel] = None
) -> SampleSet:
    """
    Converts dimod SampleSet to zquantum.core Measurements.
    If no bqm is specified, the vartype of the SampleSet will be "BINARY" and the energies will be NaN.
    If bqm is specified, its vartype will be preserved and the energy values will be calculated.

    Args:
        measurements: Measurements object to be converted
        bqm: if provided, SampleSet will include energy values for each sample.
    Returns:
        SampleSet object
    """
    if not bqm:
        return SampleSet.from_samples(
            measurements.bitstrings, "BINARY", [np.nan for _ in measurements.bitstrings]
        )
    if bqm.vartype != dimod.BINARY:
        raise TypeError("BQM needs to have vartype BINARY")
    return SampleSet.from_samples_bqm(measurements.bitstrings, bqm)