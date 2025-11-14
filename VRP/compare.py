import time
from classical_solver import solve_vrp_classical
from qaoa_solver import solve_qaoa
from pqaoa_solver import solve_pqaoa
from vrp_qubo import build_vrp_qubo
import numpy as np

def create_random_vrp(n=4):
    coords = np.random.rand(n, 2)*100
    D = [[np.linalg.norm(coords[i]-coords[j]) for j in range(n)] for i in range(n)]
    return D

def run():
    D = create_random_vrp(6)       # VRP instance
    Q = build_vrp_qubo(D, num_vehicles=2)

    print("====== Classical Solver ======")
    t0 = time.time()
    c = solve_vrp_classical(D, num_vehicles=2)
    # print("Classical:", c)
    print("   Classical time:", time.time() - t0, "seconds\n")

    print("====== Standard QAOA ======")
    q = solve_qaoa(Q, p=1)
    print("QAOA:", q, "\n")

    print("====== Parallel QAOA (pQAOA) ======")
    pq = solve_pqaoa(Q, r=4, p=1)
    print("pQAOA:", pq, "\n")

if __name__ == "__main__":
    run()
