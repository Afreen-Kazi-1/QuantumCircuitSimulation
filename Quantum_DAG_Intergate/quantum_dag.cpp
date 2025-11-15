// quantum_dag.cpp
// Compile: mpic++ -fopenmp -O2 quantum_dag.cpp -o quantum_dag
// Run: mpirun -np <P> ./quantum_dag

#include <mpi.h>
#include <omp.h>

#include <complex>
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <cassert>

using cd = std::complex<double>;
using namespace std;

// ---------- Gate representation (single- or two-qubit) ----------
struct Gate {
    // For simplicity support:
    // type = "single" or "two"
    string type;
    int q0; // target (for single) or control (for two)
    int q1; // target (for two), -1 for single
    // single-qubit 2x2 matrix: g[0][0], g[0][1], g[1][0], g[1][1]
    cd g00, g01, g10, g11;
};

struct Task {
    int id;
    Gate gate;
    set<int> blocksTouched;
    vector<int> deps;    // ids of tasks it depends on
    vector<int> children;
    int indeg = 0;
};

// ---------- Utilities ----------
inline bool bit_set(int x, int b) { return (x >> b) & 1; }

// Partition helper: block index for basis index
int block_of_index(int idx, int blockSize) {
    return idx / blockSize;
}

// build which blocks a gate touches by brute-force scanning indices
set<int> blocks_touched_by_gate(const Gate &g, int nQubits, int nStates, int blockSize) {
    set<int> blk;
    if (g.type == "single") {
        int q = g.q0;
        int stride = 1 << q;
        for (int i = 0; i < nStates; ++i) {
            int j = i ^ stride;
            blk.insert(block_of_index(i, blockSize));
            blk.insert(block_of_index(j, blockSize));
        }
    } else { // two-qubit
        int q0 = g.q0, q1 = g.q1;
        int stride0 = 1 << q0;
        int stride1 = 1 << q1;
        for (int i = 0; i < nStates; ++i) {
            int i00 = i;
            int i01 = i ^ stride1;
            int i10 = i ^ stride0;
            int i11 = i ^ stride0 ^ stride1;
            blk.insert(block_of_index(i00, blockSize));
            blk.insert(block_of_index(i01, blockSize));
            blk.insert(block_of_index(i10, blockSize));
            blk.insert(block_of_index(i11, blockSize));
        }
    }
    return blk;
}

// apply a single-qubit gate to produce deltas (delta_re, delta_im)
// state_re/im are the current amplitudes
// this function computes new amplitudes for the pairs touched by the gate and
// accumulates (new - old) into delta_re/ delta_im
void apply_single_qubit_gate_delta(const Gate &g,
                                   int nQubits, int nStates,
                                   const double *state_re, const double *state_im,
                                   double *delta_re, double *delta_im)
{
    int q = g.q0;
    int stride = 1 << q;

    // iterate pairs (i with bit q = 0, j = i|stride)
    // Parallelize index loop with OpenMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nStates; ++i) {
        if (bit_set(i, q)) continue; // only process pair once
        int j = i | stride;

        double ai_re = state_re[i], ai_im = state_im[i];
        double aj_re = state_re[j], aj_im = state_im[j];

        // compute new amplitudes: new_i = g00*ai + g01*aj
        cd new_i = g.g00 * cd(ai_re, ai_im) + g.g01 * cd(aj_re, aj_im);
        cd new_j = g.g10 * cd(ai_re, ai_im) + g.g11 * cd(aj_re, aj_im);

        // delta = new - old
        cd di = new_i - cd(ai_re, ai_im);
        cd dj = new_j - cd(aj_re, aj_im);

        // atomic additions for the delta arrays (split real/imag)
        #pragma omp atomic
        delta_re[i] += di.real();
        #pragma omp atomic
        delta_im[i] += di.imag();

        #pragma omp atomic
        delta_re[j] += dj.real();
        #pragma omp atomic
        delta_im[j] += dj.imag();
    }
}

// apply two-qubit gate (we'll assume generic 4x4 unitary encoded via combination)
// For demonstration we'll implement controlled-NOT (CNOT) where gate matrices are simple,
// but to stay general we compute exact mapping for control-target pairs by enumerating 4 states.
void apply_two_qubit_gate_delta(const Gate &g,
                                int nQubits, int nStates,
                                const double *state_re, const double *state_im,
                                double *delta_re, double *delta_im)
{
    int qc = g.q0, qt = g.q1;
    // We'll process groups of 4 basis states that differ in bits qc and qt.
    int low = min(qc, qt), high = max(qc, qt);
    int stride_low = 1 << low;
    int stride_high = 1 << high;

    // iterate i over states where both bits low/high are zero to avoid duplicates
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nStates; ++i) {
        // ensure we only handle one representative per 4-block:
        if (bit_set(i, low) || bit_set(i, high)) continue;
        int i00 = i;
        int i01 = i ^ stride_high;     // high bit = 1
        int i10 = i ^ stride_low;      // low bit = 1
        int i11 = i ^ stride_low ^ stride_high;

        cd a00(state_re[i00], state_im[i00]);
        cd a01(state_re[i01], state_im[i01]);
        cd a10(state_re[i10], state_im[i10]);
        cd a11(state_re[i11], state_im[i11]);

        // For a generic 4x4, we'd have:
        // [b00 b01 b02 b03; b10 ...] * [a00 a01 a10 a11]^T
        // But our Gate struct only has 2x2 fields; so we will support a small set:
        // If it's a CNOT encoded in a particular way, we can implement it.
        // For demonstration we'll implement CNOT (control=q0, target=q1)
        // by swapping amplitudes where control=1 and flipping target.
        // We'll detect CNOT by recognizing gate matrix as identity for control=0 and swap for control=1.
        // Simpler: if g.g00 == 1 and others indicate CNOT:
        bool isCNOT = false;
        // Heuristic: for CNOT, behavior: |c t> -> |c, t ^ c>
        // We'll detect if g behaves like: a00->a00, a01->a01, a10->a11, a11->a10
        // (i.e., swap the a10 and a11)
        // In practice for clarity, user can create known two-qubit gates by setting type "CNOT".
        // We'll thus check type specially instead of relying on numeric fields.
        // But since Gate.type here might be "two", we will treat two-qubit as CNOT for demo.
        isCNOT = true;

        if (isCNOT) {
            // new amplitudes:
            cd b00 = a00;
            cd b01 = a01;
            cd b10 = a11; // swap
            cd b11 = a10;

            cd d00 = b00 - a00;
            cd d01 = b01 - a01;
            cd d10 = b10 - a10;
            cd d11 = b11 - a11;

            #pragma omp atomic
            delta_re[i00] += d00.real();
            #pragma omp atomic
            delta_im[i00] += d00.imag();

            #pragma omp atomic
            delta_re[i01] += d01.real();
            #pragma omp atomic
            delta_im[i01] += d01.imag();

            #pragma omp atomic
            delta_re[i10] += d10.real();
            #pragma omp atomic
            delta_im[i10] += d10.imag();

            #pragma omp atomic
            delta_re[i11] += d11.real();
            #pragma omp atomic
            delta_im[i11] += d11.imag();
        }
    }
}


// ---------- DAG build: tasks dependent if they touch same block ----------
vector<Task> build_tasks_and_dag(int nQubits, const vector<Gate> &gates, int nBlocks) {
    int nStates = 1 << nQubits;
    int blockSize = nStates / nBlocks;
    vector<Task> tasks;
    tasks.reserve(gates.size());
    for (size_t i = 0; i < gates.size(); ++i) {
        Task t;
        t.id = (int)i;
        t.gate = gates[i];
        t.blocksTouched = blocks_touched_by_gate(gates[i], nQubits, nStates, blockSize);
        tasks.push_back(move(t));
    }

    // Build dependencies: if two tasks share any block, add an edge between them (we'll make an undirected
    // overlap imply dependency; to have an acyclic graph we simply order tasks by id and only create directed edges
    // from lower-id to higher-id when they overlap)
    for (size_t i = 0; i < tasks.size(); ++i) {
        for (size_t j = i + 1; j < tasks.size(); ++j) {
            // overlap?
            bool overlap = false;
            for (int b : tasks[i].blocksTouched) {
                if (tasks[j].blocksTouched.count(b)) { overlap = true; break; }
            }
            if (overlap) {
                // direct edge i -> j (enforce ordering to keep DAG)
                tasks[i].children.push_back(tasks[j].id);
                tasks[j].deps.push_back(tasks[i].id);
                tasks[j].indeg++;
            }
        }
    }
    return tasks;
}

// Topological execution with MPI + OpenMP
void execute_dag_mpi_openmp(vector<Task> &tasks,
                            int nQubits,
                            vector<cd> &state,
                            int nBlocks,
                            int rank, int size)
{
    int nStates = 1 << nQubits;
    int blockSize = nStates / nBlocks;
    int T = tasks.size();

    // map id->index in tasks vector (here ids are 0..T-1)
    vector<int> indeg(T);
    for (int i = 0; i < T; ++i) indeg[i] = tasks[i].indeg;

    // ownership: simple round-robin by task id across MPI ranks
    auto owned_by = [&](int tid) { return tid % size; };

    // arrays for deltas
    vector<double> state_re(nStates), state_im(nStates);
    for (int i = 0; i < nStates; ++i) { state_re[i] = state[i].real(); state_im[i] = state[i].imag(); }

    vector<double> delta_re(nStates), delta_im(nStates), global_delta_re(nStates), global_delta_im(nStates);

    // Kahn's algorithm loop: process ready tasks in batches (levels)
    queue<int> q;
    for (int i = 0; i < T; ++i) if (indeg[i] == 0) q.push(i);

    int processed = 0;
    while (!q.empty()) {
        // collect all currently ready tasks into a level (to allow parallel execution)
        vector<int> level;
        int qsz = q.size();
        for (int i = 0; i < qsz; ++i) {
            level.push_back(q.front()); q.pop();
        }

        // zero deltas
        fill(delta_re.begin(), delta_re.end(), 0.0);
        fill(delta_im.begin(), delta_im.end(), 0.0);

        // Each rank executes the tasks in 'level' that it owns.
        // For each owned task, compute its delta and accumulate into delta_re/delta_im.
        // We parallelize inside gate application routines using OpenMP.
        for (int tid : level) {
            if (owned_by(tid) != rank) continue;
            Gate &g = tasks[tid].gate;
            if (g.type == "single") {
                apply_single_qubit_gate_delta(g, nQubits, nStates, state_re.data(), state_im.data(),
                                              delta_re.data(), delta_im.data());
            } else {
                apply_two_qubit_gate_delta(g, nQubits, nStates, state_re.data(), state_im.data(),
                                           delta_re.data(), delta_im.data());
            }
        }

        // Reduce deltas across ranks (sum)
        MPI_Allreduce(delta_re.data(), global_delta_re.data(), nStates, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(delta_im.data(), global_delta_im.data(), nStates, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update state with global deltas
        for (int i = 0; i < nStates; ++i) {
            state_re[i] += global_delta_re[i];
            state_im[i] += global_delta_im[i];
        }

        // mark tasks in this level as processed (everyone must agree)
        for (int tid : level) {
            processed++;
            for (int ch : tasks[tid].children) {
                indeg[ch]--;
                if (indeg[ch] == 0) q.push(ch);
            }
        }

        // Important: broadcast updated state arrays to local state vector variable for next round.
        // (All ranks already have updated state_re/state_im because we updated after MPI_Allreduce)
    }

    // write back into `state`
    for (int i = 0; i < nStates; ++i) state[i] = cd(state_re[i], state_im[i]);

    if (rank == 0) {
        cout << "[Rank 0] All tasks processed: " << processed << " tasks.\n";
    }
}

// ---------- helper to create common gates ----------
Gate make_hadamard_on(int q) {
    Gate g;
    g.type = "single";
    g.q0 = q; g.q1 = -1;
    double invs = 1.0 / sqrt(2.0);
    g.g00 = cd(invs,0); g.g01 = cd(invs,0);
    g.g10 = cd(invs,0); g.g11 = cd(-invs,0);
    return g;
}
Gate make_x_on(int q) {
    Gate g; g.type="single"; g.q0=q; g.q1=-1;
    g.g00=cd(0,0); g.g01=cd(1,0); g.g10=cd(1,0); g.g11=cd(0,0);
    return g;
}
Gate make_cnot(int qc, int qt) {
    Gate g; g.type="two"; g.q0=qc; g.q1=qt;
    // not using numeric 4x4 in this demo; apply_two_qubit_gate_delta treats type "two" as CNOT
    return g;
}

// ---------- Demo main ----------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem size
    int nQubits = 4;              // adjust for testing (2^n states)
    int nStates = 1 << nQubits;
    int nBlocks = min(4, nStates); // number of contiguous blocks (for partitioning)

    if (rank == 0) {
        cout << "MPI ranks: " << size << ", nQubits=" << nQubits << ", nStates=" << nStates << ", nBlocks=" << nBlocks << "\n";
    }

    // initial state |0...0>
    vector<cd> state(nStates, cd(0.0,0.0));
    state[0] = cd(1.0, 0.0);

    // define a sequence of gates (tasks)
    vector<Gate> gates;
    gates.push_back(make_hadamard_on(0));   // H on qubit 0
    gates.push_back(make_hadamard_on(1));   // H on qubit 1
    gates.push_back(make_cnot(0,2));        // CNOT (0 -> 2)
    gates.push_back(make_x_on(3));          // X on qubit 3
    gates.push_back(make_cnot(1,3));        // CNOT (1 -> 3)
    gates.push_back(make_hadamard_on(2));   // H on qubit 2

    // Build tasks and DAG
    vector<Task> tasks = build_tasks_and_dag(nQubits, gates, nBlocks);
    if (rank == 0) cout << "Built " << tasks.size() << " tasks.\n";

    // Execute DAG with MPI+OpenMP
    execute_dag_mpi_openmp(tasks, nQubits, state, nBlocks, rank, size);

    // Final state output (only rank 0 prints)
    if (rank == 0) {
        cout << "Final state amplitudes:\n";
        for (int i = 0; i < nStates; ++i) {
            cout << i << ": " << state[i] << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
