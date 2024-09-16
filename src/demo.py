# Python code
import itertools
import networkx as nx
def maxcut_bruteforce(G):
    max_cut_value = 0
    best_partition = None
    nodes = list(G.nodes())
    for i in range(1, len(nodes)):
        for cut in itertools.combinations(nodes, i):
            partition_A = set(cut)
            partition_B = set(nodes) - partition_A
            cut_value = sum(1 for u, v in G.edges() if (u in partition_A and v in partition_B) or (u in partition_B and v in partition_A))
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                best_partition = (partition_A, partition_B)
                return max_cut_value, best_partition

                G = nx.Graph()
                G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
                max_cut_value, best_partition = maxcut_bruteforce(G)
                print(f"Maximum cut value: {max_cut_value}")
                print(f"Best partition: {best_partition}")