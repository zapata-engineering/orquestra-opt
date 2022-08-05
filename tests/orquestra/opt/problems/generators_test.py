################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest

from orquestra.opt.problems import get_random_ising_hamiltonian


class TestGetRandomIsingHamiltonian:
    @pytest.mark.parametrize("num_terms", [2, 6])
    @pytest.mark.parametrize("num_qubits", [2, 5, 7])
    def test_num_qubits_and_num_terms_is_correct(self, num_qubits, num_terms):
        # Given
        if num_qubits >= 5:
            max_number_of_qubits_per_term = 4
        else:
            max_number_of_qubits_per_term = num_qubits

        # When
        hamiltonian = get_random_ising_hamiltonian(
            num_qubits, num_terms, max_number_of_qubits_per_term
        )

        # Then
        # Some qubits may not be included due to randomness, thus the generated number
        # of qubits must be less than or equal to `num_qubits`
        assert max(hamiltonian.qubits) <= num_qubits
        generated_num_terms = len(hamiltonian.terms) - 1

        # If two of the randomly generated terms have the same qubits that are operated
        # on, then the two terms will be combined. Therefore, the generated number of
        # terms may be less than `num_terms`
        assert generated_num_terms <= num_terms

    @pytest.mark.parametrize("max_num_qubits_per_term", [2, 4])
    def test_random_hamiltonian_max_num_qubits_per_term(self, max_num_qubits_per_term):
        # Given
        num_qubits = 5
        num_terms = 3

        # When
        hamiltonian = get_random_ising_hamiltonian(
            num_qubits, num_terms, max_num_qubits_per_term
        )

        # Then
        for term in hamiltonian.terms:
            # Each term must have at most `max_num_qubits_per_term` operators (qubits)
            assert len(term.operations) <= max_num_qubits_per_term
