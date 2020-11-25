from openfermion import IsingOperator
from dimod import BinaryQuadraticModel


def convert_qubo_to_openfermion_ising(qubo: BinaryQuadraticModel):
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


def convert_openfermion_ising_to_qubo(operator: IsingOperator):
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
