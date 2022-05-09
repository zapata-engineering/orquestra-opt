################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import json
from typing import Any, Dict

import dimod
from orquestra.quantum.typing import DumpTarget, LoadSource
from orquestra.quantum.utils import ensure_open


def bqm_to_serializable(bqm: dimod.BinaryQuadraticModel) -> Dict[str, Any]:
    """Convert binary quadratic model to a serializable dictionary.

    Args:
        bqm: binary quadratic model to convert

    Returns:
        Dictionary with the following keys:
        - linear: list of pairs (i, a_i), where a_i is linear coefficient
          corresponding to variable i,
        - quadratic: list of triples (i, j, b_ij) where b_ij is quadratic
          coefficient corresponding to variables i < j,
        - offset: offset taken from the model.
        - vartype: field determining whether variables in the model are
          from the set {-1, 1} ("SPIN") or the set {0, 1} ("BINARY").

    Notes:
        A list of tuples was chosen instead of dictionary because dictionary
        keyed with tuples is not JSON-serializable.
    """
    return {
        "linear": [(label, coef) for label, coef in bqm.linear.items()],
        "quadratic": [
            (*sorted((label_1, label_2)), coef)
            for (label_1, label_2), coef in bqm.quadratic.items()
        ],
        "offset": bqm.offset,
        "vartype": bqm.vartype.name,
    }


def bqm_from_serializable(serializable: Dict[str, Any]) -> dimod.BinaryQuadraticModel:
    """Create a binary quadratic model from serializable dictionary.

    Args:
        serializable: dictionary representing BQM. The expected format
        of this dictionary as the same as the output format of
        `bqm_to_serializable`.

    Returns:
        Binary quadratic model converted from the input dictionary.
    """

    def ensure_hashable(i):
        return tuple(i) if isinstance(i, list) else i

    return dimod.BinaryQuadraticModel(
        {ensure_hashable(i): coef for i, coef in serializable["linear"]},
        {
            (ensure_hashable(i), ensure_hashable(j)): coef
            for i, j, coef in serializable["quadratic"]
        },
        serializable["offset"],
        vartype=serializable["vartype"],
    )


def load_qubo(input_file: LoadSource):

    with ensure_open(input_file, "r") as f:
        qubo_dict = json.load(f)

    return bqm_from_serializable(qubo_dict)


def save_qubo(qubo, output_file: DumpTarget):
    qubo_dict = bqm_to_serializable(qubo)

    with ensure_open(output_file, "w") as f:
        json.dump(qubo_dict, f)


def save_sampleset(sampleset, output_file: DumpTarget):
    sampleset_dict = sampleset.to_serializable()

    with ensure_open(output_file, "w") as f:
        json.dump(sampleset_dict, f)


def load_sampleset(input_file: LoadSource):

    with ensure_open(input_file, "r") as f:
        sampleset_dict = json.load(f)

    return dimod.SampleSet.from_serializable(sampleset_dict)
