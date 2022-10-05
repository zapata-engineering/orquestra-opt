################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import dimod
import numpy as np
import pytest
from orquestra.quantum.operators import PauliTerm

from orquestra.opt.bqm.conversions import (
    convert_measurements_to_sampleset,
    convert_paulisum_to_qubo,
    convert_qubo_to_paulisum,
    convert_sampleset_to_measurements,
)

from orquestra.quantum.measurements import Measurements  # isort: skip


def test_qubo_conversion_with_binary_fractions():
    qubo = dimod.BinaryQuadraticModel(
        {0: 1, 1: 2, 2: 3},
        {(1, 2): 0.5, (1, 0): -0.25, (0, 2): 2.125},
        -1,
        vartype=dimod.BINARY,
    )
    ising = convert_qubo_to_paulisum(qubo)
    new_qubo = convert_paulisum_to_qubo(ising)
    assert qubo == new_qubo


def test_qubo_conversion_with_non_binary_fractions():
    qubo = dimod.BinaryQuadraticModel(
        {0: 1.01, 1: -2.03, 2: 3},
        {(1, 2): 0.51, (1, 0): -0.9, (0, 2): 2.125},
        -1,
        vartype=dimod.BINARY,
    )
    ising = convert_qubo_to_paulisum(qubo)
    new_qubo = convert_paulisum_to_qubo(ising)

    assert len(qubo.linear) == len(new_qubo.linear)
    assert len(qubo.quadratic) == len(new_qubo.quadratic)

    assert np.isclose(qubo.offset, new_qubo.offset)
    assert qubo.vartype == new_qubo.vartype

    for key in qubo.linear.keys():
        assert np.isclose(qubo.linear[key], new_qubo.linear[key])

    for key in qubo.quadratic.keys():
        assert np.isclose(qubo.quadratic[key], new_qubo.quadratic[key])


def test_converted_ising_evaluates_to_the_same_energy_as_original_qubo():
    qubo = dimod.BinaryQuadraticModel(
        {0: 1, 1: 2, 2: 3},
        {
            (0, 1): 1,
            (0, 2): 0.5,
            (1, 2): 0.5,
        },
        -1,
        vartype=dimod.BINARY,
    )
    all_solutions = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    ising = convert_qubo_to_paulisum(qubo)
    for solution in all_solutions:
        qubo_energy = qubo.energy(solution)
        ising_energy = np.sum(
            Measurements([tuple(solution)]).get_expectation_values(ising).values
        )
        assert qubo_energy == ising_energy


def test_converted_qubo_evaluates_to_the_same_energy_as_original_ising():
    ising = (
        PauliTerm("I0", 2.5)
        + PauliTerm("Z0")
        + PauliTerm({0: "Z", 1: "Z"}, 2)
        + PauliTerm({0: "Z", 2: "Z"}, 0.5)
        + PauliTerm("Z1")
        + PauliTerm("0.75*Z1*Z2")
        - PauliTerm("Z2")
    )

    all_solutions = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    qubo = convert_paulisum_to_qubo(ising)
    for solution in all_solutions:
        qubo_energy = qubo.energy(solution)
        ising_energy = np.sum(
            Measurements([tuple(solution)]).get_expectation_values(ising).values
        )
        assert qubo_energy == ising_energy


def test_convert_sampleset_to_measurements():
    bitstrings = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
    ]
    energies = [0 for i in range(len(bitstrings))]
    sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.BINARY, energies)
    target_measurements = Measurements(bitstrings)
    converted_measurements = convert_sampleset_to_measurements(sampleset)

    assert converted_measurements.bitstrings == target_measurements.bitstrings


def test_convert_sampleset_to_measurements_with_change_bitstring_convention():
    bitstrings = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
    ]

    target_bitstrings = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 0, 1),
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
    ]
    change_bitstring_convention = True
    energies = [0 for i in range(len(bitstrings))]
    sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.BINARY, energies)
    target_measurements = Measurements(target_bitstrings)
    converted_measurements = convert_sampleset_to_measurements(
        sampleset, change_bitstring_convention=change_bitstring_convention
    )

    assert converted_measurements.bitstrings == target_measurements.bitstrings


def test_convert_sampleset_to_measurements_fails_for_non_binary_vartype():
    bitstrings = [
        (0, 0, 0),
    ]
    energies = [0 for _ in range(len(bitstrings))]
    sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.SPIN, energies)
    with pytest.raises(Exception):
        _ = convert_sampleset_to_measurements(sampleset)


def test_convert_sampleset_to_measurements_fails_for_non_int_variables():
    bitstrings = [
        (0, 0, 0),
    ]
    energies = [0 for _ in range(len(bitstrings))]
    sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.SPIN, energies)
    sampleset = sampleset.relabel_variables({0: 0.0, 1: 0.1, 2: 0.2})
    with pytest.raises(Exception):
        _ = convert_sampleset_to_measurements(sampleset)


def test_convert_sampleset_to_measurements_fails_for_variables_from_wrong_range():
    bitstrings = [
        (0, 0, 0),
    ]
    energies = [0 for _ in range(len(bitstrings))]
    sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.SPIN, energies)
    sampleset = sampleset.relabel_variables({0: 1, 1: 2, 2: 3})
    with pytest.raises(Exception):
        _ = convert_sampleset_to_measurements(sampleset)


def test_convert_measurements_to_sampleset_without_qubo():
    bitstrings = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
    ]
    measurements = Measurements(bitstrings)

    target_sampleset = dimod.SampleSet.from_samples(
        bitstrings, dimod.BINARY, [np.nan for _ in bitstrings]
    )
    converted_sampleset = convert_measurements_to_sampleset(measurements)

    # Since energies should be np.nans, using "==" will result in error
    for (target_record, converted_record) in zip(
        target_sampleset.record, converted_sampleset.record
    ):
        for target_element, converted_element in zip(target_record, converted_record):
            np.testing.assert_equal(target_element, converted_element)

    assert converted_sampleset.vartype == target_sampleset.vartype


def test_convert_measurements_to_sampleset_with_change_bitstring_convention():
    bitstrings = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
    ]
    target_bitstrings = [
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (0, 1, 1),
        (0, 0, 1),
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
    ]
    change_bitstring_convention = True
    measurements = Measurements(bitstrings)
    target_sampleset = dimod.SampleSet.from_samples(
        target_bitstrings, dimod.BINARY, [np.nan for _ in bitstrings]
    )
    converted_sampleset = convert_measurements_to_sampleset(
        measurements, change_bitstring_convention=change_bitstring_convention
    )

    # Since energies should be np.nans, using "==" will result in error
    for (target_record, converted_record) in zip(
        target_sampleset.record, converted_sampleset.record
    ):
        for target_element, converted_element in zip(target_record, converted_record):
            np.testing.assert_equal(target_element, converted_element)

    assert converted_sampleset.vartype == target_sampleset.vartype


def test_convert_measurements_to_sampleset_with_qubo():
    bitstrings = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 0, 1),
        (0, 0, 1),
    ]
    qubo = dimod.BinaryQuadraticModel(
        {0: 1, 1: 2, 2: 3},
        {(1, 2): 0.5, (1, 0): -0.25, (0, 2): 2.125},
        0,
        vartype=dimod.BINARY,
    )
    energies = [0, 3, 2, 5.5, 1, 2.75, 8.375, 6.125, 3]
    measurements = Measurements(bitstrings)

    target_sampleset = dimod.SampleSet.from_samples(bitstrings, dimod.BINARY, energies)
    converted_sampleset = convert_measurements_to_sampleset(measurements, qubo)
    assert target_sampleset == converted_sampleset
