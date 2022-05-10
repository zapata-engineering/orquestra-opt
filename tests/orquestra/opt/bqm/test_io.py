################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from io import StringIO

import dimod
import numpy as np
import pytest

from orquestra.opt.bqm.io import (
    bqm_from_serializable,
    bqm_to_serializable,
    load_qubo,
    load_sampleset,
    save_qubo,
    save_sampleset,
)


class TestConvertingBQMToSerializable:
    def test_all_linear_coefficients_are_stored(self):
        bqm = dimod.BinaryQuadraticModel(
            {0: 1, 1: 2, 2: 3},
            {(1, 2): 0.5, (1, 0): 0.7, (0, 2): 0.9},
            -10,
            vartype=dimod.BINARY,
        )

        serializable = bqm_to_serializable(bqm)

        assert set(serializable["linear"]) == {(0, 1), (1, 2), (2, 3)}

    def test_all_quadratic_coefficients_are_stored(self):
        bqm = dimod.BinaryQuadraticModel(
            {0: 0.5, 2: -2.0, 3: 3},
            {(2, 1): 0.5, (1, 0): 0.4, (0, 3): -0.1},
            -5,
            vartype=dimod.BINARY,
        )

        serializable = bqm_to_serializable(bqm)

        assert set(serializable["quadratic"]) == {
            (1, 2, 0.5),
            (0, 1, 0.4),
            (0, 3, -0.1),
        }

    @pytest.mark.parametrize("offset", [-5, 10, 0])
    def test_offset_is_stored(self, offset):
        bqm = dimod.BinaryQuadraticModel(
            {0: 0.5, 2: -2.0, 3: 3},
            {(2, 1): 0.5, (1, 0): 0.4, (0, 3): -0.1},
            offset,
            vartype=dimod.BINARY,
        )

        serializable = bqm_to_serializable(bqm)

        assert serializable["offset"] == offset

    @pytest.mark.parametrize(
        "vartype, expected_output_vartype",
        [
            ("SPIN", "SPIN"),
            ("BINARY", "BINARY"),
            (dimod.BINARY, "BINARY"),
            (dimod.SPIN, "SPIN"),
        ],
    )
    def test_vartype_is_stored(self, vartype, expected_output_vartype):
        bqm = dimod.BinaryQuadraticModel(
            {0: 0.5, 2: -2.0, 3: 3},
            {(2, 1): 0.5, (1, 0): 0.4, (0, 3): -0.1},
            vartype=vartype,
        )

        serializable = bqm_to_serializable(bqm)

        assert serializable["vartype"] == expected_output_vartype


class TestConstructingBQMFromSerializable:
    def test_all_linear_coefficients_are_loaded(self):
        bqm_dict = {
            "linear": [(0, 2.0), (2, 0.5), (1, -1.0)],
            "quadratic": [(0, 1, 1.2), (1, 2, 4.0)],
            "offset": 0.5,
            "vartype": "SPIN",
        }

        bqm = bqm_from_serializable(bqm_dict)
        assert bqm.linear == {0: 2.0, 2: 0.5, 1: -1.0}

    def test_all_quadratic_coefficients_are_loaded(self):
        bqm_dict = {
            "linear": [(0, 1.0), (1, 2.0), (2, 0.5)],
            "quadratic": [(0, 1, 2.1), (1, 2, 4.0), (1, 3, -1.0)],
            "offset": 0.1,
            "vartype": "BINARY",
        }

        bqm = bqm_from_serializable(bqm_dict)

        assert bqm.quadratic == {(0, 1): 2.1, (1, 2): 4.0, (1, 3): -1.0}

    @pytest.mark.parametrize("offset", [0.1, 2, -3.5])
    def test_offset_is_loaded(self, offset):
        bqm_dict = {
            "linear": [(0, 1.0), (1, 2.0), (2, 0.5)],
            "quadratic": [(0, 1, 2.1), (1, 2, 4.0), (1, 3, -1.0)],
            "offset": offset,
            "vartype": "BINARY",
        }

        bqm = bqm_from_serializable(bqm_dict)

        assert bqm.offset == offset

    @pytest.mark.parametrize(
        "vartype, expected_bqm_vartype",
        [("SPIN", dimod.SPIN), ("BINARY", dimod.BINARY)],
    )
    def test_vartype_is_set_correctly(self, vartype, expected_bqm_vartype):
        bqm_dict = {
            "linear": [(0, 1.0), (2, 2.0), (1, 0.5)],
            "quadratic": [(0, 1, 1), (1, 2, 4.0), (1, 3, 1e-2)],
            "offset": 0.5,
            "vartype": vartype,
        }

        bqm = bqm_from_serializable(bqm_dict)

        assert bqm.vartype == expected_bqm_vartype


def test_loading_saved_qubo_gives_the_same_qubo():
    qubo = dimod.BinaryQuadraticModel(
        {0: 0.5, 2: -2.0, 3: 3},
        {(2, 1): 0.5, (1, 0): 0.4, (0, 3): -0.1},
        -5,
        vartype="BINARY",
    )

    output_file = StringIO()

    save_qubo(qubo, output_file)
    # Move to the beginning of the file
    output_file.seek(0)
    new_qubo = load_qubo(output_file)

    assert qubo == new_qubo


def test_loading_qubo_with_complex_variables():
    qubo = dimod.BinaryQuadraticModel(
        {(0, 0): 1, (1, 0): -1, (2, 0): 0.5},
        {((0, 0), (1, 0)): 0.5, ((1, 0), (2, 0)): 1.5},
        42,
        vartype="SPIN",
    )

    output_file = StringIO()

    save_qubo(qubo, output_file)
    # Move to the beginning of the file
    output_file.seek(0)
    new_qubo = load_qubo(output_file)

    assert qubo == new_qubo


def test_loading_saved_sampleset_gives_the_same_sampleset():
    sampleset = dimod.SampleSet.from_samples(np.ones(5, dtype="int8"), "BINARY", 0)

    output_file = StringIO()

    save_sampleset(sampleset, output_file)
    # Move to the beginning of the file
    output_file.seek(0)
    new_sampleset = load_sampleset(output_file)

    assert sampleset == new_sampleset
