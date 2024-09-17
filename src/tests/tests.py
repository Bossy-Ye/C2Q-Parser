import unittest
from src.parser import Parser


class MyTestCase(unittest.TestCase):
    def __init__(self):
        super().__init__()
        self.code = """
            import networkx as nx
            def maximal_independent_set(G):
                return nx.maximal_independent_set(G)
            """
        self.parser = Parser(model_path="src/saved_models")

    def test_something(self):

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
