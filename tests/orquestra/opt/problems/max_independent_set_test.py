################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import copy

import networkx as nx
import pytest
from orquestra.quantum.operators import PauliTerm

from orquestra.opt.problems import MaxIndependentSet

from ._helpers import graph_node_index, make_graph

MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=range(2), edges=[(0, 1)]),
        [PauliTerm("I0", -0.5), PauliTerm({0: "Z", 1: "Z"}, 0.5)],
    ),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        [
            PauliTerm("I0", -0.5),
            PauliTerm("Z0", -0.5),
            PauliTerm({0: "Z", 1: "Z"}, 0.5),
            PauliTerm({0: "Z", 2: "Z"}, 0.5),
        ],
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        [
            PauliTerm("I0", -0.5),
            PauliTerm("Z0", -1),
            PauliTerm({0: "Z", 1: "Z"}, 0.5),
            PauliTerm({0: "Z", 2: "Z"}, 0.5),
            PauliTerm({0: "Z", 3: "Z"}, 0.5),
        ],
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [
            PauliTerm("I0", -1),
            PauliTerm("Z1", -0.5),
            PauliTerm({0: "Z", 1: "Z"}, 0.5),
            PauliTerm({1: "Z", 2: "Z"}, 0.5),
            PauliTerm({3: "Z", 4: "Z"}, 0.5),
        ],
    ),
]

NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS = [
    (
        make_graph(node_ids=[4, 2], edges=[(2, 4)]),
        [
            PauliTerm("I0", -0.5),
            PauliTerm({0: "Z", 1: "Z"}, 0.5),
        ],
    ),
    (
        make_graph(node_ids="CBA", edges=[("C", "B"), ("C", "A")]),
        [
            PauliTerm("I0", -0.5),
            PauliTerm("Z0", -0.5),
            PauliTerm({0: "Z", 1: "Z"}, 0.5),
            PauliTerm({0: "Z", 2: "Z"}, 0.5),
        ],
    ),
]

GRAPH_EXAMPLES = [
    *[graph for graph, _ in MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    *[graph for graph, _ in NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS],
    make_graph(
        node_ids=range(10),
        edges=[
            (0, 2),
            (0, 3),
            (1, 2),
            (4, 5),
            (0, 8),
        ],
    ),
    make_graph(
        node_ids=["foo", "bar", "baz"],
        edges=[
            ("foo", "baz"),
            ("bar", "baz"),
        ],
    ),
]

GRAPH_SOLUTION_COST_LIST = [
    (make_graph(node_ids=range(2), edges=[(0, 1)]), (0, 0), 0),
    (make_graph(node_ids=range(2), edges=[(0, 1)]), (0, 1), -1),
    (
        make_graph(
            node_ids=range(4), edges=[(0, 1, 1), (0, 2, 2), (0, 3, 3)], use_weights=True
        ),
        (1, 0, 0, 0),
        -1,
    ),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), (0, 0, 1, 1), -2),
    (make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]), (0, 1, 1, 1), -3),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        (1, 1, 1, 1, 1),
        1,
    ),
]

GRAPH_BEST_SOLUTIONS_COST_LIST = [
    (make_graph(node_ids=range(2), edges=[(0, 1)]), [(0, 1), (1, 0)], -1),
    (
        make_graph(node_ids=range(3), edges=[(0, 1), (0, 2)]),
        [(0, 1, 1)],
        -2,
    ),
    (
        make_graph(node_ids=range(4), edges=[(0, 1), (0, 2), (0, 3)]),
        [
            (0, 1, 1, 1),
        ],
        -3,
    ),
    (
        make_graph(node_ids=range(5), edges=[(0, 1), (1, 2), (3, 4)]),
        [(1, 0, 1, 0, 1), (1, 0, 1, 1, 0)],
        -3,
    ),
]


class TestGetMaxIndependentSetHamiltonian:
    @pytest.mark.parametrize(
        "graph,terms",
        [
            *MONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
            *NONMONOTONIC_GRAPH_OPERATOR_TERM_PAIRS,
        ],
    )
    def test_returns_expected_terms(self, graph, terms):
        pauli_sum = MaxIndependentSet().get_hamiltonian(graph)
        assert set(pauli_sum.terms) == set(terms)

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_5_weight_on_edge_terms(self, graph: nx.Graph):
        pauli_sum = MaxIndependentSet().get_hamiltonian(graph)

        for vertex_id1, vertex_id2 in graph.edges:
            qubit_index1 = graph_node_index(graph, vertex_id1)
            qubit_index2 = graph_node_index(graph, vertex_id2)
            edge_term = [
                term
                for term in pauli_sum.terms
                if term.qubits == {qubit_index1, qubit_index2}
            ][0]
            assert edge_term.coefficient == 0.5

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_mod_5_weight_on_vertex_terms(self, graph: nx.Graph):
        pauli_sum = MaxIndependentSet().get_hamiltonian(graph)

        for vertex in graph.nodes:
            qubit_index = graph_node_index(graph, vertex)
            vertex_term = next(
                filter(lambda term: term.qubits == {qubit_index}, pauli_sum.terms), None
            )
            if vertex_term is None:
                # There is no term with only an operator on this vertex qubit
                coefficient = 0 + 0j
            else:
                coefficient = vertex_term.coefficient

            assert coefficient.real % 0.5 == 0
            assert coefficient.real <= 0.5
            assert coefficient.imag == 0

    @pytest.mark.parametrize("graph", GRAPH_EXAMPLES)
    def test_has_correct_constant_term(self, graph: nx.Graph):
        expected_constant_term = 0.0

        pauli_sum = MaxIndependentSet().get_hamiltonian(graph)
        for _ in graph.edges:
            expected_constant_term += 1 / 2

        expected_constant_term -= len(graph.nodes) / 2

        constant_term = [term for term in pauli_sum.terms if term.is_constant][0]
        assert constant_term.coefficient == expected_constant_term


class TestEvaluateMaxIndependentSetSolution:
    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_stable_set_solution(self, graph, solution, target_value):
        value = MaxIndependentSet().evaluate_solution(solution, graph)
        assert value == target_value

    @pytest.mark.parametrize("graph,solution,target_value", [*GRAPH_SOLUTION_COST_LIST])
    def test_evaluate_stable_set_solution_with_invalid_input(
        self, graph, solution, target_value
    ):
        too_long_solution = solution + (1,)
        too_short_solution = solution[:-1]
        invalid_value_solution = copy.copy(solution)
        invalid_value_solution = list(invalid_value_solution)
        invalid_value_solution[0] = -1
        invalid_value_solution = tuple(invalid_value_solution)
        invalid_solutions = [
            too_long_solution,
            too_short_solution,
            invalid_value_solution,
        ]
        for invalid_solution in invalid_solutions:
            with pytest.raises(ValueError):
                _ = MaxIndependentSet().evaluate_solution(invalid_solution, graph)


class TestSolveMaxIndependentSetByExhaustiveSearch:
    @pytest.mark.parametrize(
        "graph,target_solutions,target_value", [*GRAPH_BEST_SOLUTIONS_COST_LIST]
    )
    def test_solve_stable_set_by_exhaustive_search(
        self, graph, target_solutions, target_value
    ):
        value, solutions = MaxIndependentSet().solve_by_exhaustive_search(graph)
        assert set(solutions) == set(target_solutions)
        assert value == target_value
