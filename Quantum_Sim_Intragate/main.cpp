#include "quantum.hpp"
#include <iomanip>
#include <array>
#include <complex>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <cassert>
#include <cmath>
#include <chrono>


// --- Basic gates (2x2)
inline std::array<cpx,4> hadamard_matrix() {
    double s = 1.0 / SQRT2;
    return { cpx(s,0), cpx(s,0), cpx(s,0), cpx(-s,0) };
}
inline std::array<cpx,4> pauliX_matrix() {
    return { cpx(0,0), cpx(1,0), cpx(1,0), cpx(0,0) };
}

// Helper: check if bit k of global index is 1
inline int bit_of(uint64_t idx, uint32_t k) {
    return (idx >> k) & 1ULL;
}

void apply_single_qubit_distributed(DistState &ds, const std::array<cpx,4> &mat, uint32_t target, MPI_Comm comm = MPI_COMM_WORLD) {
    uint64_t local_n = ds.local_size;
    vcpx new_local(local_n);

    uint32_t k = target;
    // Fast path: if partner of each local index stays on same rank -> no communication
    bool all_local = true;
    uint64_t mask = 1ULL << k;
    uint64_t start = ds.local_offset;
    uint64_t end = ds.local_offset + local_n - 1;
    // compute partner range for start and end
    uint64_t p1 = start ^ mask;
    uint64_t p2 = end ^ mask;
    // If partner range lies within same rank chunk bounds (i.e., [local_offset, local_offset+local_n-1]) then safe.
    if (!((p1 >= ds.local_offset && p1 <= ds.local_offset+local_n-1) &&
          (p2 >= ds.local_offset && p2 <= ds.local_offset+local_n-1))) {
        all_local = false;
    }

    if (all_local) {
        // purely local pairing
        #pragma omp parallel for schedule(static)
        for (int64_t li = 0; li < (int64_t)local_n; ++li) {
            uint64_t gi = ds.global_index(li);
            uint64_t partner_g = gi ^ mask;
            uint64_t partner_l = partner_g - ds.local_offset;
            int row = bit_of(gi, k); 
            cpx a0 = ds.state[ (bit_of(gi,k) ? partner_l : li) ];
            if (bit_of(gi,k) == 0) {
                cpx s0 = mat[0] * ds.state[li] + mat[1] * ds.state[partner_l];
                new_local[li] = s0;
            } else {
                cpx s1 = mat[2] * ds.state[partner_l] + mat[3] * ds.state[li];
                new_local[li] = s1;
            }
        }
    } else {
        vcpx full_state;
        if (ds.world_rank == 0) {
            // root collects full state into full_state
        }
        // Use MPI_Allgather to collect everyone's local chunk into full vector on all ranks
        full_state.resize(ds.global_size);
        // MPI_Allgather expects contiguous arrays of POD; complex<double> is contiguous.
        MPI_Allgather(ds.state.data(), (int)local_n * (int)sizeof(cpx), MPI_BYTE,
                      full_state.data(), (int)local_n * (int)sizeof(cpx), MPI_BYTE, comm);
        // Now compute
        #pragma omp parallel for schedule(static)
        for (int64_t li = 0; li < (int64_t)local_n; ++li) {
            uint64_t gi = ds.global_index(li);
            uint64_t partner_g = gi ^ mask;
            cpx old_g = full_state[gi];
            cpx old_p = full_state[partner_g];
            if (bit_of(gi,k) == 0) {
                new_local[li] = mat[0] * old_g + mat[1] * old_p;
            } else {
                new_local[li] = mat[2] * old_p + mat[3] * old_g;
            }
        }
    }

    // swap in new data
    ds.state.swap(new_local);
}

// Apply CNOT with control and target (control != target)
void apply_cnot_distributed(DistState &ds, uint32_t control, uint32_t target, MPI_Comm comm = MPI_COMM_WORLD) {
    uint64_t local_n = ds.local_size;
    uint64_t mask_t = 1ULL << target;
    uint64_t mask_c = 1ULL << control;

    // For simplicity use Allgather path when partner remote; try fast local path first
    bool all_local = true;
    uint64_t start = ds.local_offset;
    uint64_t end = ds.local_offset + local_n - 1;
    uint64_t p1 = start ^ mask_t;
    uint64_t p2 = end ^ mask_t;
    if (!((p1 >= ds.local_offset && p1 <= ds.local_offset+local_n-1) &&
          (p2 >= ds.local_offset && p2 <= ds.local_offset+local_n-1))) {
        all_local = false;
    }

    vcpx new_local(local_n);
    if (all_local) {
        #pragma omp parallel for schedule(static)
        for (int64_t li = 0; li < (int64_t)local_n; ++li) {
            uint64_t gi = ds.global_index(li);
            if (bit_of(gi, control) == 1) {
                uint64_t partner_g = gi ^ mask_t;
                uint64_t partner_l = partner_g - ds.local_offset;
                // swap amplitudes
                new_local[li] = ds.state[partner_l];
            } else {
                new_local[li] = ds.state[li];
            }
        }
    } else {
        // fallback allgather (simple)
        vcpx full_state(ds.global_size);
        MPI_Allgather(ds.state.data(), (int)local_n * (int)sizeof(cpx), MPI_BYTE,
                      full_state.data(), (int)local_n * (int)sizeof(cpx), MPI_BYTE, comm);
        #pragma omp parallel for schedule(static)
        for (int64_t li = 0; li < (int64_t)local_n; ++li) {
            uint64_t gi = ds.global_index(li);
            if (bit_of(gi, control) == 1) {
                uint64_t partner_g = gi ^ mask_t;
                new_local[li] = full_state[partner_g];
            } else {
                new_local[li] = full_state[gi];
            }
        }
    }
    ds.state.swap(new_local);
}

uint64_t measure_all(DistState &ds, MPI_Comm comm = MPI_COMM_WORLD) {
    double local_sum = 0.0;
    for (uint64_t i = 0; i < ds.local_size; ++i) local_sum += std::norm(ds.state[i]);
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    // normalize if necessary
    if (std::abs(global_sum - 1.0) > 1e-8) {
        // normalize
        double norm = std::sqrt(global_sum);
        for (uint64_t i = 0; i < ds.local_size; ++i) ds.state[i] /= norm;
    }

    // compute prefix sums on root: simplest approach - gather all probs to rank 0, sample, broadcast result
    std::vector<double> all_probs;
    if (ds.world_rank == 0) all_probs.resize(ds.global_size);
    // gather probabilities (as doubles)
    std::vector<double> local_probs(ds.local_size);
    for (uint64_t i = 0; i < ds.local_size; ++i) local_probs[i] = std::norm(ds.state[i]);
    MPI_Gather(local_probs.data(), (int)ds.local_size, MPI_DOUBLE,
               (ds.world_rank==0 ? all_probs.data() : nullptr), (int)ds.local_size, MPI_DOUBLE, 0, comm);

    uint64_t measured = 0;
    if (ds.world_rank == 0) {
        // sample
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_real_distribution<double> dis(0.0,1.0);
        double r = dis(gen);
        double acc = 0.0;
        for (uint64_t i = 0; i < ds.global_size; ++i) {
            acc += all_probs[i];
            if (r <= acc) { measured = i; break; }
        }
    }
    // broadcast measured outcome
    MPI_Bcast(&measured, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);

    // collapse state: each rank keeps only amplitude corresponding to measured; optionally normalize
    for (uint64_t i = 0; i < ds.local_size; ++i) ds.state[i] = cpx(0.0,0.0);
    if (measured >= ds.local_offset && measured < ds.local_offset + ds.local_size) {
        uint64_t li = measured - ds.local_offset;
        ds.state[li] = cpx(1.0,0.0);
    }
    return measured;
}

// Utility: print small state (gather to root)
void print_state(DistState &ds, MPI_Comm comm = MPI_COMM_WORLD) {
    std::vector<cpx> full;
    if (ds.world_rank == 0) full.resize(ds.global_size);
    MPI_Gather(ds.state.data(), (int)ds.local_size * (int)sizeof(cpx), MPI_BYTE,
               (ds.world_rank==0 ? full.data() : nullptr), (int)ds.local_size * (int)sizeof(cpx), MPI_BYTE, 0, comm);
    if (ds.world_rank == 0) {
        std::cout<<std::fixed<<std::setprecision(6);
        for (uint64_t i = 0; i < ds.global_size; ++i) {
            auto &v = full[i];
            std::cout<<"|"<<i<<"> : "<<v<<"  ("<<std::norm(v)<<")\n";
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int provided;
    MPI_Query_thread(&provided);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (argc < 2) {
        if (rank==0) std::cout<<"Usage: mpirun -n <P> ./qsim <n_qubits>\nExample: mpirun -n 2 ./qsim 3\n";
        MPI_Finalize();
        return 0;
    }
    uint32_t n = (uint32_t)atoi(argv[1]);
    DistState ds(n, comm);
    double t0 = MPI_Wtime();
    ds.init_zero();
    double t1 = MPI_Wtime();

    if (rank==0) std::cout<<"Initialized "<<n<<" qubits (global size "<<ds.global_size<<"), procs="<<nprocs<<"\n";

    // Example 1: Bell state on 2 qubits (qubit 0 = LSB)
    if (n >= 2) {
        if (rank==0) std::cout<<"=== Creating Bell state (H on qubit 0, CNOT(0->1)) ===\n";
        // Apply H on qubit 0
        apply_single_qubit_distributed(ds, hadamard_matrix(), 0, comm);
        // Apply CNOT with control=0, target=1
        apply_cnot_distributed(ds, 0, 1, comm);

        // print
        if (rank==0) std::cout<<"State after Bell creation:\n";
        print_state(ds, comm);

        // measure
        uint64_t m = measure_all(ds, comm);
        if (rank==0) std::cout<<"Measurement collapsed to: "<<m<<"\n";
        MPI_Barrier(comm);
    } else {
        if (rank==0) std::cout<<"Need at least 2 qubits for Bell example.\n";
    }

    if (n >= 3) {
    if (rank==0) std::cout<<"=== Creating GHZ state (H0 -> CNOT(0,1) -> CNOT(1,2) -> CNOT(2,3)) ===\n";

    apply_single_qubit_distributed(ds, hadamard_matrix(), 0, comm); // H on q0
    apply_cnot_distributed(ds, 0, 1, comm); // CNOT(control=0,target=1)
    apply_cnot_distributed(ds, 1, 2, comm); // CNOT(control=1,target=2)
    apply_cnot_distributed(ds, 2, 3, comm);

    if (rank==0) std::cout<<"State after GHZ creation:\n";
    print_state(ds, comm);

    uint64_t m = measure_all(ds, comm);
    if (rank==0) std::cout<<"Measurement collapsed to: "<<m<<"\n";
    MPI_Barrier(comm);
    }

    double t_end = MPI_Wtime();
    if (rank==0) std::cout<<"Total elapsed: "<<(t_end - t0)<<" s. Init: "<<(t1-t0)<<" s\n";

    MPI_Finalize();
    return 0;
}
