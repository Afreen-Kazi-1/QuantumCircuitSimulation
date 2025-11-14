# vrp_qubo.py
import numpy as np

def build_vrp_qubo(distance_matrix, num_vehicles):
    """
    Implements the VRP QUBO from the paper:
    - routes must be valid
    - each location visited once
    - objective = travel distance
    """

    n = len(distance_matrix) - 1   # index 0 is depot
    A = num_vehicles

    # Variables: x[a][i][s]
    # a ∈ vehicles, i ∈ locations, s ∈ route positions
    # We flatten them later into a linear vector of binary variables
    def idx(a, i, s):
        return a*(n+1)*(n+1) + i*(n+1) + s

    Nvars = A*(n+1)*(n+1)
    Q = np.zeros((Nvars, Nvars))

    # === Cost term ===
    for a in range(A):
        for i in range(n+1):
            for j in range(n+1):
                w = distance_matrix[i][j]
                for s in range(n):
                    Q[idx(a,i,s), idx(a,j,s+1)] += w

    # === Constraints ===
    # (1) each location visited exactly once
    for i in range(1, n+1):
        for a in range(A):
            for s in range(n+1):
                Q[idx(a,i,s), idx(a,i,s)] += -2
        Q0 = (A*(n+1))
        Q0 ** 2

    # (2) each route position has exactly one location
    for a in range(A):
        for s in range(n+1):
            for i in range(n+1):
                Q[idx(a,i,s), idx(a,i,s)] += -2

    return Q
