# orquestra-opt

## What is it?

`orquestra-opt` is a core library of the scientific code for [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing variational quantum algorithms.

`orquestra-opt` provides:

- interfaces for implementing ansatzes including qaoa and qcbm.
- optimizers and cost functions tailored to opt
- misc functions such as grouping, qaoa interpolation, and estimators

## Usage

### Workflows

Here's an example of how to use methods from `orquestra-opt` to run a workflow. This workflow solves a simple maximal independent set problem.

```python
from orquestra.opt.problems import MaxCut
import networkx as nx

@sdk.task(
    source_import=sdk.GitImport(repo_url="git@github.com:my_username/my_repository.git", git_ref="main"),
    dependency_imports=[sdk.GitImport(repo_url="git@github.com:zapatacomputing/orquestra-opt.git", git_ref="main")]
)
def orquestra_opt_example_task()
   node_ids=range(2)
   edges=[(0, 1)]
   
   graph = nx.Graph()
   graph.add_nodes_from(node_ids)
   graph.add_edges_from(edges)

   value, solutions = MaxCut().solve_by_exhaustive_search(graph)

   return solutions


@sdk.workflow()
def orquestra_opt_example_workflow():
    solutions = orquestra_opt_example_task()
    return [solutions]
```

### Python

Even though it's intended to be used with Orquestra, `orquestra-opt` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory.

Here's an example of how to use methods from `orquestra-opt` to solve a simple maximal independent set problem.

```python
from orquestra.opt.problems import MaxCut
import networkx as nx

def orquestra_opt_example_function()
   node_ids=range(2)
   edges=[(0, 1)]
   
   graph = nx.Graph()
   graph.add_nodes_from(node_ids)
   graph.add_edges_from(edges)

   value, solutions = MaxCut().solve_by_exhaustive_search(graph)

   return solutions
```

## Development and Contribution

You can find the development guidelines in the [`orquestra-quantum` repository](https://github.com/zapatacomputing/orquestra-quantum).