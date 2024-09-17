import ast
import unittest

from src.parser import Parser, CodeVisitor


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.mul_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = 8, 8\nresult = a(p, q)\nprint(result)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)"
        self.matrix_define = "matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nnx.add_edges_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2 , 3, matrix)"
        self.parser = Parser(model_path="../saved_models")
        self.code_visitor = CodeVisitor()

    def test_something(self):
        problem_type, data = self.parser.parse(self.mul_snippet)
        self.assertEqual(problem_type, 'MUL')  # add assertion here

    def test_codeVisitor(self):
        tree = ast.parse(self.matrix_define)
        self.code_visitor.visit(tree)
        print(self.code_visitor.get_extracted_data())
        print(self.code_visitor.function_calls)


if __name__ == '__main__':
    unittest.main()
