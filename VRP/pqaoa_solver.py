import numpy as np
import time
from multiprocessing import Pool
from qaoa_solver import solve_qaoa

# Slice the QUBO into diagonal blocks
def slice_qubo(Q, r):
    n = Q.shape[0]
    size = n // r
    slices = []

    for i in range(r):
        s = Q[i * size:(i + 1) * size, i * size:(i + 1) * size]
        slices.append(s)

    return slices

# Wrapper for multiprocessing
def solve_slice(args):
    Qslice, p = args
    print("Slice qubits:", len(Qslice))
    return solve_qaoa(Qslice, p)

# Parallel QAOA
def solve_pqaoa(Q, r=2, p=1):
    start = time.time()

    slices = slice_qubo(Q, r)

    with Pool(r) as pool:
        energies = pool.map(solve_slice, [(s, p) for s in slices])

    total_energy = sum(energies)

    end = time.time()
    print(f"   pQAOA execution time: {end - start:.4f} seconds")

    return total_energy
