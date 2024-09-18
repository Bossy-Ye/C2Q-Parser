import ast
import unittest

import networkx
import networkx as nx

from src.graph import Graph
from src.parser import Parser, CodeVisitor
from src.reducer import *

class MyTestCase(unittest.TestCase):
    def setUp(self):
        """
        NB, snippets defined withing triple quotes() can not work somehow...
        :return:
        """
        self.mul_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)"
        self.matrix_define = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)\nmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nnx.add_edges_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2 , 3, matrix)"
        self.sub_snippet = "def a(p, q):\n    return p - q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.parser = Parser(model_path="../saved_models")
        self.tsp_snippet = "def a(cost_matrix):\n    n = len(cost_matrix)\n    visited = [0]\n    total_cost = 0\n    current = 0\n    while len(visited) < n:\n        next_city = min([city for city in range(n) if city not in visited], key=lambda city: cost_matrix[current][city])\n        total_cost += cost_matrix[current][next_city]\n        visited.append(next_city)\n        current = next_city\n    total_cost += cost_matrix[visited[-1]][0]\n    return total_cost, visited\n\n# Input data\ncost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\ncost, route = a(cost_matrix)\nprint(cost, route)"
        self.code_visitor = CodeVisitor()
        self.clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input data\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
        self.maxCut_snippet_adj = "import networkx as nx\n\n" \
                                  "def adjacency_matrix_to_edges(matrix):\n" \
                                  "    edges = []\n" \
                                  "    for i in range(len(matrix)):\n" \
                                  "        for j in range(i + 1, len(matrix[i])):  # Use j = i + 1 to avoid duplicating edges\n" \
                                  "            if matrix[i][j] != 0:\n" \
                                  "                edges.append((i, j))  # Add edge (i, j) to the edge list\n" \
                                  "    return edges\n\n" \
                                  "def simple_cut_strategy(edges, n):\n" \
                                  "    A, B = set(), set()\n" \
                                  "    for node in range(n):\n" \
                                  "        if len(A) < len(B):\n" \
                                  "            A.add(node)\n" \
                                  "        else:\n" \
                                  "            B.add(node)\n" \
                                  "    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n" \
                                  "# Adjacency matrix as input\n" \
                                  "adjacency_matrix = [\n" \
                                  "    [0, 1, 1, 1, 1, 1],\n" \
                                  "    [1, 0, 0, 1, 0, 0],\n" \
                                  "    [1, 0, 0, 1, 0, 0],\n" \
                                  "    [1, 1, 1, 0, 1, 1],\n" \
                                  "    [1, 0, 0, 1, 0, 1],\n" \
                                  "    [1, 0, 0, 1, 1, 0]\n" \
                                  "]\n\n" \
                                  "# Convert adjacency matrix to edge list\n" \
                                  "edges = adjacency_matrix_to_edges(adjacency_matrix)\n\n" \
                                  "# Use simple_cut_strategy with the edge list and number of nodes\n" \
                                  "cut_value, A, B = simple_cut_strategy(edges, len(adjacency_matrix))\n\n" \
                                  "print(f\"Cut Value: {cut_value}\")\n" \
                                  "print(f\"Set A: {A}\")\n" \
                                  "print(f\"Set B: {B}\")\n\n" \
                                  "# Visualization of the graph\n" \
                                  "G = nx.Graph()\n" \
                                  "G.add_edges_from(edges)\n" \
                                  "pos = nx.spring_layout(G)\n" \
                                  "nx.draw(G, pos, with_labels=True, node_color=\"lightblue\", node_size=500, font_size=15)\n" \
                                  "nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): 1 for u, v in edges})  # Assuming unweighted edges\n" \
                                  "plt.show()\n"

    def test_something(self):
        problem_type, data = self.parser.parse(self.mul_snippet)
        self.assertEqual(problem_type, 'MUL')  # add assertion here

    def test_max_cut(self):
        problem_type, data = self.parser.parse(self.maxCut_snippet_adj)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MaxCut')  # add assertion here
        self.assertIsInstance(data.graph, networkx.Graph)
        data.visualize()
        self.assertEqual(networkx.is_weighted(data.graph), True)

    def test_tsp_snippet(self):
        problem_type, data = self.parser.parse(self.tsp_snippet)
        self.assertEqual(problem_type, 'TSP')
        data.visualize()

    def test_mis(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MIS')  # add assertion here
        self.assertIsInstance(data.graph, networkx.Graph)
        data.visualize()
        self.assertEqual(networkx.is_weighted(data.graph), True)

    def test_mul(self):
        problem_type, data = self.parser.parse(self.mul_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MUL')

    def test_sub(self):
        problem_type, data = self.parser.parse(self.sub_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'SUB')

    def test_codeVisitor(self):
        tree = ast.parse(self.mul_snippet)
        self.code_visitor.visit(tree)
        print(self.code_visitor.get_extracted_data())
        print(self.code_visitor.function_calls)

    def test_clique_snippet(self):
        problem_type, data = self.parser.parse(self.clique_snippet)
        print(problem_type, data)
        cnf = clique_to_sat(data.graph, 3)
        data.visualize()
        sat = sat_to_3sat(cnf)
        print(f'clauses before conversion: {len(cnf.clauses)}')
        print(f'clauses after conversion: {len(sat.clauses)}')
        print(cnf.clauses)

    def test_clique(self):
        self.assertEqual(True,True)

    def test_graph_init(self):
        # Example 1: Using a distance matrix
        distance_matrix = [
            [0, 2, 3, 0],
            [2, 0, 4, 6],
            [3, 4, 0, 5],
            [0, 6, 5, 0]
        ]
        graph_matrix = Graph(input_data=distance_matrix)
        graph_matrix.visualize()

        # Example 2: Using a list of edges with weights
        edges_with_weights = [(0, 1, 2), (0, 2, 3), (1, 2, 4), (1, 3, 6), (2, 3, 5)]
        graph_edges_with_weights = Graph(input_data=edges_with_weights)
        graph_edges_with_weights.visualize()

        # Example 3: Using a list of edges without weights (default weight of 1 will be assigned)
        edges_without_weights = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        graph_edges_without_weights = Graph(input_data=edges_without_weights)
        graph_edges_without_weights.visualize()

        # Example 4: Generate a random graph
        random_graph = Graph.random_graph(num_nodes=5, edge_prob=0.5, weighted=True)
        random_graph.visualize()

        # Example 5: Invalid input
        invalid_input = {"a": 1, "b": 2}  # Not a valid format
        try:
            graph_invalid = Graph(input_data=invalid_input)
        except ValueError as e:
            print(f"Error: {e}")

        self.assertEqual(True, False)

    def test_2_3sat(self):
        # Example usage:
        cnf = [[1, 2, 3, 4, -5, 6], [-1, -2, -3, -4], [-1, -2, 3, 4], [-1, -2, 3, -4], [1, 2], [-1, -2], [1, 3]]
        converted_cnf = sat_to_3sat(cnf)

        print("Converted 3-SAT CNF:", converted_cnf)

        # Find all solutions for the original and converted CNF
        original_solutions = solve_all_cnf_solutions(cnf)
        print(len(original_solutions))
        converted_solutions = solve_all_cnf_solutions(converted_cnf)
        print(len(converted_solutions))

        print("Original CNF solutions:", original_solutions)
        print("Converted CNF solutions:", converted_solutions)


if __name__ == '__main__':
    unittest.main()
