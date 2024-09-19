import networkx as nx
from pysat.formula import CNF
from pysat.solvers import Solver
from collections import defaultdict


def clique_to_sat(graph: nx.Graph, k: int) -> CNF:
    """
    Converts the k-Clique problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.
        k (int): The size of the clique to find.

    Returns:
        CNF: The SAT formula in CNF representing the k-Clique problem.
    """
    cnf = CNF()
    n = len(graph.nodes)

    # Variables: x_iv where i is the position in the clique and v is the vertex
    var = lambda i, v: i * n + v + 1  # Create unique variables (i is 0-based, v is 0-based)

    # Clause 1: Each position in the clique must be occupied by exactly one vertex
    # k + \frac{kn(n+1)}{2} clauses
    for i in range(k):
        # At least one vertex occupies the i-th position in the clique
        cnf.append([var(i, v) for v in range(n)])
        # No more than one vertex occupies the i-th position in the clique
        for v in range(n):
            for u in range(v + 1, n):
                cnf.append([-var(i, v), -var(i, u)])

    # Clause 2: No vertex can occupy more than one position in the clique
    # \frac{kn(k-1)}{2} clauses
    for v in range(n):
        for i in range(k):
            for j in range(i + 1, k):
                cnf.append([-var(i, v), -var(j, v)])

    # Clause 3: All vertices in the clique must be connected by an edge
    # \(k^2-k\)\(\frac{n^2-n}{2}-|E|\) clauses
    for i in range(k):
        for j in range(i + 1, k):
            for v in range(n):
                for u in range(n):
                    if v != u and not graph.has_edge(v, u):
                        cnf.append([-var(i, v), -var(j, u)])

    return cnf


def tseytin_or_to_cnf(a, b, aux_var):
    """
    Applies Tseytin transformation for the expression aux_var = (a OR b)
    reference: https://en.wikipedia.org/wiki/Tseytin_transformation
    Returns the equivalent CNF clauses for the transformation.
    """
    return [
        [-aux_var, a, b],  # (aux_var -> (a OR b)) <=> (¬aux_var OR a OR b)
        [aux_var, -a],  # (¬a -> aux_var) <=> (aux_var OR ¬a)
        [aux_var, -b]  # (¬b -> aux_var) <=> (aux_var OR ¬b)
    ]


def solve_all_cnf_solutions(cnf_formula):
    """
    Finds all solutions to the CNF formula using a SAT solver.

    Args:
        cnf_formula: The CNF formula to solve.
    Returns:
        list: A list of all satisfying assignments, where each assignment is a list of literals.
    """
    solutions = []
    with Solver(bootstrap_with=cnf_formula) as solver:
        for model in solver.enum_models():
            solutions.append(model)
    return solutions


def sat_to_3sat(cnf):
    """
    Converts a CNF with more than three literals per clause into a 3-SAT CNF
    using Tseytin transformation.
    reference: https://en.wikipedia.org/wiki/Tseytin_transformation
    Parameters:
        cnf (CNF): The CNF formula object (from pysat.formula.CNF) to be converted.

    Returns:
        CNF: The converted 3-SAT CNF formula object.
    """
    new_clauses = []
    aux_var_counter = max(abs(lit) for clause in cnf for lit in clause) + 1  # Start for auxiliary variables

    for clause in cnf:
        while len(clause) > 3:
            first_literal = clause[0]
            second_literal = clause[1]

            # Introduce a new auxiliary variable
            aux_var = aux_var_counter
            aux_var_counter += 1

            # Apply Tseytin transformation for (first_literal OR second_literal) = aux_var
            new_clauses += tseytin_or_to_cnf(first_literal, second_literal, aux_var)

            # Now create a new clause starting with the auxiliary variable and the rest of the literals
            clause = [aux_var] + clause[2:]

        # After the loop, the clause has three or fewer literals, so append it as-is
        new_clauses.append(clause)

    return CNF(from_clauses=new_clauses)


def cnf_to_qubo(cnf, method="Chancellor"):
    """
    Converts a 3-SAT CNF formula to a QUBO formula based on the given rules.
    Args:
        cnf (CNF): The CNF formula object from the pysat library.

    Returns:
        dict: A dictionary representing the QUBO matrix, with keys as tuples (i, j)
              and values as the coefficients for the QUBO objective function.
    """
    Q = defaultdict(float)  # To store the QUBO coefficients

    for clause in cnf.clauses:
        if len(clause) == 3:
            x1, x2, x3 = clause
            # QUBO formula for (x1 ∨ x2 ∨ x3)
            Q[(abs(x1), abs(x1))] += -1 if x1 > 0 else 1
            Q[(abs(x2), abs(x2))] += -1 if x2 > 0 else 1
            Q[(abs(x3), abs(x3))] += -1 if x3 > 0 else 1
            Q[(abs(x1), abs(x2))] += 1 if x1 > 0 and x2 > 0 else -1
            Q[(abs(x1), abs(x3))] += 1 if x1 > 0 and x3 > 0 else -1
            Q[(abs(x2), abs(x3))] += 1 if x2 > 0 and x3 > 0 else -1
            Q[(abs(x1), abs(x2), abs(x3))] += -1 if x1 > 0 and x2 > 0 and x3 > 0 else 1
        elif len(clause) == 2:
            x1, x2 = clause
            # QUBO formula for (x1 ∨ x2)
            Q[(abs(x1), abs(x1))] += -1 if x1 > 0 else 1
            Q[(abs(x2), abs(x2))] += -1 if x2 > 0 else 1
            Q[(abs(x1), abs(x2))] += 1 if x1 > 0 and x2 > 0 else -1
        elif len(clause) == 1:
            x1 = clause[0]
            # QUBO formula for (x1)
            Q[(abs(x1), abs(x1))] += -1 if x1 > 0 else 1

    return dict(Q)


def qubo_to_ising(cnf):
    return 1

