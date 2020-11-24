from dimod import BinaryQuadraticModel
from zquantum.core.utils import SCHEMA_VERSION
from io import TextIOBase
import json


def load_qubo(input_file):
    if isinstance(input_file, TextIOBase):
        qubo_dict = json.load(input_file)
    else:
        with open(input_file, 'r') as f:
            qubo_dict = json.load(f)

    del qubo_dict["schema"]
    return BinaryQuadraticModel.from_serializable(qubo_dict)


def save_qubo(qubo, output_file):
    dict_qubo = qubo.to_serializable()
    dict_qubo["schema"] = SCHEMA_VERSION + "-qubo"

    if isinstance(output_file, TextIOBase):
        json.dump(dict_qubo, output_file)
    else:
        with open(output_file, 'w') as f:
            json.dump(dict_qubo, f)
