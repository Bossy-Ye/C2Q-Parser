"This Chancellor 3sat2qubo class takes a reference from https://arxiv.org/pdf/2305.02659"
import numpy as np
from itertools import product
from pysat.formula import CNF
from pysat.solvers import Solver


class Chancellor:

    def __init__(self, formula, V):
        # sort the formula (i.e. all negative literals are at the back of the clause)
        self.formula = [sorted(c, reverse=True) for c in formula]
        self.V = V
        self.Q = {}

    # new values are added to the QUBO-Matrix Q via this monitor
    def add(self, x, y, value):
        x = np.abs(x) - 1
        y = np.abs(y) - 1
        if x > y:
            x, y = y, x
        if (x, y) in self.Q.keys():
            self.Q[(x, y)] += value
        else:
            self.Q[(x, y)] = value

    # this function creates the QUBO-Matrix Q
    def fillQ(self):
        for i, c in enumerate(self.formula):
            if len(c) == 3:
                if list(np.sign(c)) == [1, 1, 1]:
                    self.add(c[0], c[0], -48)
                    self.add(c[0], c[1], 24)
                    self.add(c[0], c[2], 24)
                    self.add(c[0], self.V + i + 1, 40)
                    self.add(c[1], c[1], -48)
                    self.add(c[1], c[2], 24)
                    self.add(c[1], self.V + i + 1, 40)
                    self.add(c[2], c[2], -48)
                    self.add(c[2], self.V + i + 1, 40)
                    self.add(self.V + i + 1, self.V + i + 1, -64)
                elif list(np.sign(c)) == [1, 1, -1]:
                    self.add(c[0], c[0], -40)
                    self.add(c[0], c[1], 24)
                    self.add(c[0], c[2], 16)
                    self.add(c[0], self.V + i + 1, 40)
                    self.add(c[1], c[1], -40)
                    self.add(c[1], c[2], 16)
                    self.add(c[1], self.V + i + 1, 40)
                    self.add(c[2], c[2], -32)
                    self.add(c[2], self.V + i + 1, 40)
                    self.add(self.V + i + 1, self.V + i + 1, -56)
                elif list(np.sign(c)) == [1, -1, -1]:
                    self.add(c[0], c[0], -40)
                    self.add(c[0], c[1], 16)
                    self.add(c[0], c[2], 16)
                    self.add(c[0], self.V + i + 1, 40)
                    self.add(c[1], c[1], -40)
                    self.add(c[1], c[2], 24)
                    self.add(c[1], self.V + i + 1, 40)
                    self.add(c[2], c[2], -40)
                    self.add(c[2], self.V + i + 1, 40)
                    self.add(self.V + i + 1, self.V + i + 1, -64)
                else:
                    self.add(c[0], c[0], -40)
                    self.add(c[0], c[1], 24)
                    self.add(c[0], c[2], 24)
                    self.add(c[0], self.V + i + 1, 40)
                    self.add(c[1], c[1], -40)
                    self.add(c[1], c[2], 24)
                    self.add(c[1], self.V + i + 1, 40)
                    self.add(c[2], c[2], -40)
                    self.add(c[2], self.V + i + 1, 40)
                    self.add(self.V + i + 1, self.V + i + 1, -56)
            elif len(c) == 2:
                # Handling clauses with 2 literals (x1 ∨ x2)
                self.add(c[0], c[0], -1 if c[0] > 0 else 24)
                self.add(c[1], c[1], -1 if c[1] > 0 else 24)
                self.add(c[0], c[1], 1 if c[0] > 0 and c[1] > 0 else -24)
            elif len(c) == 1:
                # Handling clauses with 1 literal (x1)
                self.add(c[0], c[0], -24 if c[0] > 0 else 24)

    def solveQ(self):
        self.fillQ()
        # Get the number of variables from the maximum key index in Q
        num_vars = max(max(x, y) for x, y in self.Q.keys()) + 1

        # Generate all possible binary assignments (0 or 1) for the variables
        best_assignment = None
        best_energy = float('inf')  # Set initial best energy to infinity

        # Iterate over all possible assignments
        for assignment in product([0, 1], repeat=num_vars):
            energy = 0
            # Calculate the energy for the current assignment
            for (x, y), value in self.Q.items():
                energy += value * assignment[x] * assignment[y]
            # Update the best assignment if this one has lower energy
            if energy < best_energy:
                best_energy = energy
                best_assignment = assignment

        # Print the best assignment and its corresponding energy
        print(f"Best assignment: {best_assignment}")
        print(f"Minimum energy: {best_energy}")
        return best_assignment, best_energy

    def visualizeQ(self):
        """# Visualize the Q in a matrix format"""



