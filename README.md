# orquestra-opt

## What is it?

`orquestra-opt` is a library with core functionalities for optimizing cost functions developed by [Zapata](https://www.zapatacomputing.com) for our [Orquestra](https://www.zapatacomputing.com/orquestra/) platform.

`orquestra-opt` provides:

- interfaces for implementing ansatzes including qaoa and qcbm.
- optimizers and cost functions tailored to opt
- misc functions such as grouping, qaoa interpolation, and estimators

## Installation

Even though it's intended to be used with Orquestra, `orquestra-opt` can be also used as a standalone Python module.
For a basic install, you just need to run `pip install -e .` from the main directory.
If you need to make use of optimizers from qiskit or solve qubo problems, then you should instead run `pip install -e '.[qiskit]'` or `pip install -e '.[qubo]'` respectively. If you need both, you should run `pip install -e '.[all]'`

## Usage

Here's an example of how to use methods from `orquestra-opt` to solve a simple maximum cut problem.

```python
from orquestra.opt.problems import MaxCut
import networkx as nx

def orquestra_opt_example_function()
   graph = nx.complete_graph(4)
   value, solutions = MaxCut().solve_by_exhaustive_search(graph)
   return solutions
```

## Development and Contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).