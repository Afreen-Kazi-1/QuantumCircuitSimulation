#pragma once
#include <mpi.h>
#include <omp.h>
#include <complex>
#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <cmath>
#include <chrono>

using cpx = std::complex<double>;
using vcpx = std::vector<cpx>;
static const double SQRT2 = std::sqrt(2.0);

struct DistState {
    int world_rank;
    int world_size;
    uint32_t nqubits;        // total qubits
    uint64_t global_size;    // 2^nqubits
    uint64_t local_size;     // chunk per rank
    uint64_t local_offset;   // start index (global) of this rank
    vcpx state;              // local chunk
   
    DistState(uint32_t n, MPI_Comm comm = MPI_COMM_WORLD) : nqubits(n) {
        MPI_Comm_rank(comm, &world_rank);
        MPI_Comm_size(comm, &world_size);
        global_size = 1ULL << nqubits;
        assert(global_size % world_size == 0 && "For simplicity, require (2^n) divisible by nprocs");
        local_size = global_size / world_size;
        local_offset = local_size * (uint64_t)world_rank;
        state.assign(local_size, cpx(0.0, 0.0));
    }

    void init_zero() {
        for (uint64_t i = 0; i < local_size; ++i) state[i] = cpx(0.0,0.0);
        if (local_offset == 0) state[0] = cpx(1.0,0.0);
    }

    // initialize basis state |index>
    void init_basis(uint64_t basis_index) {
        for (uint64_t i = 0; i < local_size; ++i) state[i] = cpx(0.0,0.0);
        if (basis_index >= global_size) { if(world_rank==0) std::cerr<<"basis index out of range\n"; return; }
        if (basis_index >= local_offset && basis_index < local_offset + local_size) {
            state[basis_index - local_offset] = cpx(1.0,0.0);
        }
    }

    // global index of a local element
    inline uint64_t global_index(uint64_t local_idx) const { return local_offset + local_idx; }
};