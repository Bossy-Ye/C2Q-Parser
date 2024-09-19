"""
This Chancellor 3sat2qubo class takes a reference from https://arxiv.org/pdf/2305.02659
part of the codes are from https://github.com/JonasNuesslein/3SAT-with-QUBO/ with modifications
to satisfy c2|q>'s needs.
"""
import numpy as np
from itertools import product

import seaborn as sns
from matplotlib import pyplot as plt
from pysat.formula import CNF
from pysat.solvers import Solver


class Chancellor:
    def __init__(self, formula: CNF):
        # sort the formula (i.e. all negative literals are at the back of the clause)
        self.formula = formula.clauses
        self.formula = [sorted(c, reverse=True) for c in self.formula]
        self.V = formula.nv
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
                self.add(c[0], c[0], -40)
                self.add(c[1], c[1], -40)
                self.add(c[0], c[1], 23)
            elif len(c) == 1:
                # Handling clauses with 1 literal (x1)
                self.add(c[0], c[0], -40)

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
        """Visualize the QUBO matrix in a matrix format."""
        # Get the number of variables from the maximum key index in Q
        num_vars = max(max(x, y) for x, y in self.Q.keys()) + 1

        # Create a zero matrix of size num_vars x num_vars
        Q_matrix = np.zeros((num_vars, num_vars))

        # Fill the matrix with values from the QUBO dictionary
        for (x, y), value in self.Q.items():
            Q_matrix[x, y] = value

        # Plot the matrix using seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(Q_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
        plt.title("QUBO Matrix Visualization")
        plt.xlabel("Variable Index")
        plt.ylabel("Variable Index")
        plt.show()


class Nuesslein1:

    def __init__(self, formula: CNF, V):
        # sort the formula (i.e. all negative literals are at the back of the clause)
        self.formula = formula.clauses
        self.formula = [sorted(c, reverse=True) for c in self.formula]
        self.L = []
        for i in range(V):
            self.L.append(i + 1)
            self.L.append(-(i + 1))
        self.V = V
        self.Q = {}

    # new values are added to the QUBO-Matrix Q via this monitor
    def add(self, x, y, value):
        if x > y:
            x, y = y, x
        if (x, y) in self.Q.keys():
            self.Q[(x, y)] += value
        else:
            self.Q[(x, y)] = value

    def R1(self, x):
        n = 0
        for c in self.formula:
            if x in c:
                n += 1
        return n

    def R2(self, x, y):
        n = 0
        for c in self.formula:
            if x in c and y in c:
                n += 1
        return n

    # this function creates the QUBO-Matrix Q
    def fillQ(self):
        for i in range(2 * self.V + len(self.formula)):
            for j in range(2 * self.V + len(self.formula)):
                if i > j:
                    continue
                if i == j and j < 2 * self.V:
                    self.add(i, j, -self.R1(self.L[i]))
                elif i == j and j >= 2 * self.V:
                    self.add(i, j, 2)
                elif j < 2 * self.V and j - i == 1 and i % 2 == 0:
                    self.add(i, j, len(self.formula) + 1)
                elif i < 2 * self.V and j < 2 * self.V:
                    self.add(i, j, self.R2(self.L[i], self.L[j]))
                elif j >= 2 * self.V and i < 2 * self.V and self.L[i] in self.formula[j - 2 * self.V]:
                    self.add(i, j, -1)
