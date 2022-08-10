################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################

import numpy as np
from orquestra.quantum.operators import PauliSum, PauliTerm


def get_random_ising_hamiltonian(
    number_of_qubits: int, number_of_terms: int, max_number_of_qubits_per_term: int
) -> PauliSum:
    """Generates a random Hamiltonian for a given number of qubits and terms with
    weights between -1 and 1.

    NOTE: Due to randomness, we cannot ensure that the returned hamiltonian has an
        operation on every qubit.

    Args:
        number_of_qubits: The number of qubits in the Hamiltonian. Should be >= 2.
        max_number_qubits_per_term: The maximum number of qubits for each term in the
            hamiltonian. Should be <= number_of_qubits.
    """
    hamiltonian = PauliSum()

    # Add terms with random qubits
    for _ in range(number_of_terms):
        num_qubits_in_term = np.random.randint(1, max_number_of_qubits_per_term + 1)
        qubits = np.random.choice(
            range(number_of_qubits), num_qubits_in_term, replace=False
        )
        qubits.sort()
        hamiltonian += PauliTerm.from_iterable([("Z", q) for q in qubits])

    # Add constant term with a random coefficient
    hamiltonian += PauliTerm("I0", np.random.rand() * 2 - 1)

    return hamiltonian
