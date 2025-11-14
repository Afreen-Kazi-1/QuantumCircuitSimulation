import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import COBYLA


# Build one QAOA layer
def qaoa_layer(circuit, Q, beta, gamma):
    n = Q.shape[0]

    # Cost Hamiltonian (RZZ)
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i][j] != 0:
                circuit.rzz(2 * gamma * Q[i][j], i, j)

    
    for i in range(n):
        circuit.rx(2 * beta, i)


# Solve QUBO using QAOA
def solve_qaoa(Q, p=1):
    start = time.time()

    n = Q.shape[0]
    print(f"Number of qubits: {n}")  

    beta_params = [Parameter(f"beta_{i}") for i in range(p)]
    gamma_params = [Parameter(f"gamma_{i}") for i in range(p)]

    qc = QuantumCircuit(n)
    qc.h(range(n))

    # Add QAOA layers
    for i in range(p):
        qaoa_layer(qc, Q, beta_params[i], gamma_params[i])

    qc.measure_all()

    # Aer simulator with MPS backend
    simulator = AerSimulator(method="matrix_product_state")

    # Cost function for optimizer
    def objective(x):
        bind_dict = {}
        for i in range(p):
            bind_dict[beta_params[i]] = x[i]
            bind_dict[gamma_params[i]] = x[i + p]

        # Run circuit
        result = simulator.run(
            qc.bind_parameters(bind_dict),
            shots=2000
        ).result()

        counts = result.get_counts()

        energy = 0
        for bitstring, freq in counts.items():
            xbin = np.array([int(b) for b in reversed(bitstring)])
            energy += freq * (xbin @ Q @ xbin)

        return energy / 2000

    optimizer = COBYLA(maxiter=30)

    result = optimizer.minimize(
        fun=objective,
        x0=np.random.rand(2 * p)
    )

    end = time.time()
    print(f"   QAOA execution time: {end - start:.4f} seconds")

    return result.fun  # best energy
