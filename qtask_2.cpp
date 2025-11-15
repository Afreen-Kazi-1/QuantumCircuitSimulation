// qtask_mpi_openmp_correct.cpp
// MPI + OpenMP demonstration of qTask-style incremental update (correct quantum evolution).
//
// Compile:
//   mpicxx -std=c++17 -O3 -fopenmp -o qtask_mpi_openmp_correct qtask_mpi_openmp_correct.cpp
//
// Run example:
//   mpirun -np 4 ./qtask_mpi_openmp_correct
//
// Note: This implementation replicates the full state vector on every rank
//       and distributes computation of partitions (work) across ranks.
//       This simplifies communication and ensures correctness for demonstration.

#include <mpi.h>
#include <omp.h>

#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <set>
#include <map>
#include <cmath>

using cplx = std::complex<double>;
static const double SQRT2 = std::sqrt(2.0);

// ---------- Basic helpers ----------
inline size_t pow2(int n) { return size_t(1) << n; }
inline int bit_of(size_t x, int b) { return (x >> b) & 1; }

// ---------- Gate & Circuit ----------
enum GateType
{
    H_GATE,
    CNOT_GATE
};

struct Gate
{
    GateType type;
    std::vector<int> qubits; // for H: size 1; for CNOT: [control, target]
    std::string name;
    Gate(GateType t = {}, std::vector<int> q = {}, std::string nm = "") : type(t), qubits(q), name(nm) {}
};

struct Circuit
{
    int nqubits;
    std::vector<std::vector<Gate>> nets; // nets (levels) containing parallel gates
    Circuit(int n = 0) : nqubits(n) {}
    void add_net() { nets.emplace_back(); }
    void insert_gate(int net_idx, const Gate &g)
    {
        assert(net_idx >= 0 && net_idx <= (int)nets.size() - 1);
        nets[net_idx].push_back(g);
    }
    void remove_gate_by_name(int net_idx, const std::string &name)
    {
        if (net_idx < 0 || net_idx >= (int)nets.size())
            return;
        auto &v = nets[net_idx];
        v.erase(std::remove_if(v.begin(), v.end(), [&](const Gate &G)
                               { return G.name == name; }),
                v.end());
    }
};

// ---------- Partition description ----------
struct Partition
{
    size_t start_idx; // inclusive
    size_t length;
    int owner_rank;
    int id;
    Partition() : start_idx(0), length(0), owner_rank(0), id(-1) {}
    Partition(size_t s, size_t l, int o, int i) : start_idx(s), length(l), owner_rank(o), id(i) {}
};

// ---------- Simulation context ----------
struct SimContext
{
    int rank, nranks;
    int nqubits;
    size_t N;          // 2^nqubits
    size_t block_size; // B (amplitudes per block)
    std::vector<Partition> partitions;
    // partition successor graph
    std::vector<std::set<int>> succ; // succ[p] = set of partition indices that depend on p
};

// ---------- Build partitions ----------
void build_partitions(SimContext &ctx)
{
    ctx.partitions.clear();
    size_t idx = 0;
    int pid = 0;
    while (idx < ctx.N)
    {
        size_t len = std::min(ctx.block_size, ctx.N - idx);
        int owner = pid % ctx.nranks; // round-robin partition ownership for demo
        ctx.partitions.emplace_back(idx, len, owner, pid);
        idx += len;
        pid++;
    }
    ctx.succ.assign(ctx.partitions.size(), {});
}

// ---------- Utility: find partition index containing a global amplitude index ----------
int find_partition_for_index(const SimContext &ctx, size_t idx)
{
    // linear search OK for demo small N; for production use binary search
    for (size_t p = 0; p < ctx.partitions.size(); ++p)
    {
        if (idx >= ctx.partitions[p].start_idx && idx < ctx.partitions[p].start_idx + ctx.partitions[p].length)
            return (int)p;
    }
    return -1;
}

// ---------- Which partitions does a gate touch? ----------
// We conservatively mark a partition as touched if any amplitude index in the partition
// participates in the gate's action (i.e., index or its pair for H, or index and partner for CNOT).
// Accurate partitions touched by a gate (drop-in replacement).
// - For H gates: we conservatively mark ALL partitions as touched (H mixes pairs across the whole vector).
// - For CNOT gates: we mark a partition as touched iff
//     (A) it contains some index i with control-bit == 1 (amplitude for i moves), OR
//     (B) it contains some index j that is the target-image of an index with control==1
//         (i.e., there exists i with control==1 s.t. j == (i ^ (1<<target))).
//
// This function iterates only over indices inside each partition (O(N)), which is fine for small examples.
// For large-scale usage you should replace iteration with bit-pattern arithmetic or interval checks.
std::vector<int> partitions_touched_by_gate(const SimContext &ctx, const Gate &g)
{
    std::vector<int> out;

    // H gate: affects every amplitude (mixes pairs i <-> i^(1<<q)),
    // so it touches every partition (this matches qTask behavior for superposition nets).
    if (g.type == H_GATE)
    {
        out.reserve(ctx.partitions.size());
        for (int pid = 0; pid < (int)ctx.partitions.size(); ++pid)
            out.push_back(pid);
        return out;
    }

    // CNOT gate: be precise
    if (g.type == CNOT_GATE)
    {
        int control = g.qubits[0];
        int target = g.qubits[1];
        const size_t N = ctx.N;

        for (int pid = 0; pid < (int)ctx.partitions.size(); ++pid)
        {
            const Partition &p = ctx.partitions[pid];
            bool touch = false;

            size_t start = p.start_idx;
            size_t end_ex = p.start_idx + p.length; // exclusive

            // iterate indices in this partition
            for (size_t idx = start; idx < end_ex; ++idx)
            {
                // condition A: this index has control bit == 1 (so this amplitude will move)
                if (((idx >> control) & 1ULL) == 1ULL)
                {
                    touch = true;
                    break;
                }
                // condition B: this index could be the destination of a moved amplitude:
                // there exists some i (with control==1) such that idx == (i ^ (1<<target))
                // so test preimage = idx ^ (1<<target). If preimage has control==1, then idx will receive amplitude.
                size_t pre = idx ^ (1ULL << target);
                if (pre < N && (((pre >> control) & 1ULL) == 1ULL))
                {
                    touch = true;
                    break;
                }
            }

            if (touch)
                out.push_back(pid);
        }
        return out;
    }

    // Fallback conservative: mark all partitions (should not reach for our supported gates)
    out.reserve(ctx.partitions.size());
    for (int pid = 0; pid < (int)ctx.partitions.size(); ++pid)
        out.push_back(pid);
    return out;
}

// ---------- Build partition-level dependency graph across nets ----------
// For each net in left-to-right order, we collect partitions touched. For net i and j>i,
// if any partition in net i intersects partition in net j, then net j partitions depend on net i partitions.
// For simplicity we connect all partitions touched in previous nets to current net partitions.
void build_dependencies(SimContext &ctx, const Circuit &ckt)
{
    int P = (int)ctx.partitions.size();
    for (int p = 0; p < P; ++p)
        ctx.succ[p].clear();

    std::vector<std::set<int>> touched_per_net(ckt.nets.size());
    for (int netid = 0; netid < (int)ckt.nets.size(); ++netid)
    {
        std::set<int> unionTouched;
        for (const auto &g : ckt.nets[netid])
        {
            auto v = partitions_touched_by_gate(ctx, g);
            unionTouched.insert(v.begin(), v.end());
        }
        touched_per_net[netid] = unionTouched;
    }
    // connect nets in sequence: any partition touched in earlier nets is predecessor of partitions touched later
    std::set<int> reached;
    for (int netid = 0; netid < (int)ckt.nets.size(); ++netid)
    {
        auto &cur = touched_per_net[netid];
        for (int prev : reached)
        {
            for (int cpart : cur)
                ctx.succ[prev].insert(cpart);
        }
        // add cur into reached (transitive)
        for (int c : cur)
            reached.insert(c);
    }
}

// ---------- DFS to collect reachable successors from frontiers ----------
void collect_reachable_partitions(const SimContext &ctx, const std::vector<int> &frontiers, std::set<int> &out)
{
    std::vector<char> visited(ctx.partitions.size(), 0);
    std::vector<int> stack;
    for (int f : frontiers)
    {
        if (!visited[f])
        {
            visited[f] = 1;
            stack.push_back(f);
            out.insert(f);
        }
    }
    while (!stack.empty())
    {
        int v = stack.back();
        stack.pop_back();
        for (int s : ctx.succ[v])
        {
            if (!visited[s])
            {
                visited[s] = 1;
                out.insert(s);
                stack.push_back(s);
            }
        }
    }
}

// ---------- Apply one gate to global state, but only compute results for indices in 'compute_mask' (set of partition indices) ----------
// For simplicity and correctness we keep a full copy of the state on each rank (state vector).
// Each rank computes updates for indices it owns in compute_mask and then we combine via MPI_Allreduce (sum).
void apply_gate_to_state_distributed(SimContext &ctx, const Gate &g, std::vector<cplx> &state, const std::set<int> &compute_partitions)
{
    size_t N = ctx.N;
    int rank = ctx.rank;

    // local_update array: zeros except for indices we compute (owned and in compute_partitions)
    std::vector<cplx> local_update(N);
    std::fill(local_update.begin(), local_update.end(), cplx(0.0, 0.0));

    if (g.type == H_GATE)
    {
        int q = g.qubits[0];
// H on qubit q: for each index i, pair j = i ^ (1<<q)
// new_i = (old_i + old_j)/sqrt2 ; new_j = (old_i - old_j)/sqrt2
// We'll compute new values for indices owned by this rank AND whose partition is in compute_partitions
#pragma omp parallel for schedule(static)
        for (int pid = 0; pid < (int)ctx.partitions.size(); ++pid)
        {
            const Partition &p = ctx.partitions[pid];
            if (p.owner_rank != rank)
                continue;
            if (!compute_partitions.count(pid))
                continue;
            for (size_t off = 0; off < p.length; ++off)
            {
                size_t i = p.start_idx + off;
                size_t j = i ^ (1ULL << q);
                // use global state (replicated)
                cplx ai = state[i];
                cplx aj = state[j];
                cplx ni = (ai + aj) / SQRT2;
                local_update[i] = ni;
            }
        }
    }
    else if (g.type == CNOT_GATE)
    {
        int control = g.qubits[0];
        int target = g.qubits[1];
// CNOT: if control bit == 1, flip target bit at index -> amplitude positions swap.
// Formally: permutation of indices. For amplitude at i:
// if control(i)==0: amplitude stays at i
// if control(i)==1: amplitude moves to i' = i ^ (1<<target)
// To perform update: for each index i we want to compute new amplitude for index i.
// Equivalently, new_state[i] = old_state[i'] if control(i')==1 and i = i' ^ (1<<target) etc.
// Simpler: we compute mapping by iterating indices in owned partitions.
#pragma omp parallel for schedule(static)
        for (int pid = 0; pid < (int)ctx.partitions.size(); ++pid)
        {
            const Partition &p = ctx.partitions[pid];
            if (p.owner_rank != rank)
                continue;
            if (!compute_partitions.count(pid))
                continue;
            for (size_t off = 0; off < p.length; ++off)
            {
                size_t i = p.start_idx + off;
                int c = bit_of(i, control);
                if (c == 0)
                {
                    // amplitude stays at i
                    local_update[i] = state[i];
                }
                else
                {
                    // amplitude moves to i_target = i ^ (1<<target)
                    size_t it = i ^ (1ULL << target);
                    // The amplitude at it after the CNOT equals the amplitude at index it ^ (1<<target) ??? Let's reason:
                    // We want final amplitude vector after applying gate to input 'state':
                    // For each basis |b>, the amplitude at basis 'k' after gate = amplitude at basis 'preimage' such that gate(preimage) = k.
                    // For CNOT, gate maps index x to y = x if control(x)==0 else x ^ (1<<target).
                    // So the mapping is a bijection; we can compute new_state[y] = old_state[x].
                    // Our approach: compute new amplitude at index i: find x such that gate(x)=i.
                    // Solve: if control(x)==0 and x==i -> x=i
                    //        else if control(x)==1 and (x ^ (1<<target)) == i -> x = i ^ (1<<target) and control(x)==1
                    // So new[i] = old[i] if control(i)==0, else new[i] = old[i ^ (1<<target)] if control(i ^ (1<<target))==1.
                    int control_i = bit_of(i, control);
                    if (control_i == 0)
                    {
                        local_update[i] = state[i];
                    }
                    else
                    {
                        // control_i == 1 but this case handled above; safe fallback:
                        local_update[i] = state[i ^ (1ULL << target)];
                    }
                }
            }
        }
    }

    // Combine local updates from all ranks into the next global state (we assume ranks compute disjoint indices or sum will be correct).
    // We use MPI_Allreduce with MPI_SUM on complex numbers represented as two doubles.
    // Pack local_update into double buffer [2*N]
    std::vector<double> sendbuf(2 * N);
    for (size_t i = 0; i < N; ++i)
    {
        sendbuf[2 * i] = local_update[i].real();
        sendbuf[2 * i + 1] = local_update[i].imag();
    }
    std::vector<double> recvbuf(2 * N);
    MPI_Allreduce(sendbuf.data(), recvbuf.data(), (int)(2 * N), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Unpack into state
    for (size_t i = 0; i < N; ++i)
    {
        state[i] = cplx(recvbuf[2 * i], recvbuf[2 * i + 1]);
    }
}

// ---------- High-level update_state: full or incremental ----------
// If full_sim==true: recompute nets 0..end on all partitions (we compute every net in order).
// If full_sim==false: compute only nets that touch affected partitions. Starting nets are chosen conservatively: we scan nets in order and if net touches any affected partition, we execute it (and only nets after that which touch affected partitions).
void update_state(SimContext &ctx, Circuit &ckt, std::vector<cplx> &state,
                  const std::vector<int> &inserted_partitions, const std::vector<int> &removed_partitions,
                  bool full_sim)
{
    // Build dependency graph to determine successors
    build_dependencies(ctx, ckt);

    std::vector<int> front;
    for (int p : inserted_partitions)
        front.push_back(p);
    // For removed partitions, frontiers are successors of those partitions
    for (int rem : removed_partitions)
    {
        for (int s : ctx.succ[rem])
            front.push_back(s);
    }
    std::set<int> affected_partitions;
    if (full_sim)
    {
        for (int p = 0; p < (int)ctx.partitions.size(); ++p)
            affected_partitions.insert(p);
    }
    else
    {
        collect_reachable_partitions(ctx, front, affected_partitions);
    }

    // Now process nets in order; only apply a net if it touches any affected partition.
    for (int netid = 0; netid < (int)ckt.nets.size(); ++netid)
    {
        // collect partitions touched by this net
        std::set<int> net_parts;
        for (const auto &g : ckt.nets[netid])
        {
            auto parts = partitions_touched_by_gate(ctx, g);
            for (int p : parts)
                net_parts.insert(p);
        }
        // Check intersection with affected_partitions
        bool intersect = false;
        for (int p : net_parts)
        {
            if (affected_partitions.count(p))
            {
                intersect = true;
                break;
            }
        }
        if (!intersect)
            continue; // skip entire net
        // for each gate in net, apply to state but compute only for partitions in affected_partitions
        for (const auto &g : ckt.nets[netid])
        {
            // compute mask of partitions for which to compute this gate: intersection of net_parts, affected_partitions, and owner's partitions
            std::set<int> compute_parts;
            auto parts = partitions_touched_by_gate(ctx, g);
            for (int p : parts)
                if (affected_partitions.count(p))
                    compute_parts.insert(p);
            // apply gate (distributed)
            apply_gate_to_state_distributed(ctx, g, state, compute_parts);
        }
    }
}

// ---------- Initialize full state vector (|0...0>) on each rank ----------
void init_state(std::vector<cplx> &state, size_t N)
{
    state.assign(N, cplx(0.0, 0.0));
    state[0] = cplx(1.0, 0.0);
}

// ---------- Helper to print full state on rank 0 ----------
void print_full_state_if_rank0(int rank, const std::vector<cplx> &state)
{
    if (rank != 0)
        return;
    std::cout << "Full state amplitudes (index: amplitude):\n";
    for (size_t i = 0; i < state.size(); ++i)
    {
        std::cout << i << ": (" << state[i].real() << "," << state[i].imag() << ")\n";
    }
}

// ---------- Build the 5-qubit example circuit ----------
Circuit build_paper_5q_circuit()
{
    Circuit ckt(5);
    // create 5 nets (net0 .. net4)
    for (int i = 0; i < 5; ++i)
        ckt.add_net();
    // net0: five Hadamards on q4..q0 (order not important)
    ckt.insert_gate(0, Gate(H_GATE, {4}, "G1"));
    ckt.insert_gate(0, Gate(H_GATE, {3}, "G2"));
    ckt.insert_gate(0, Gate(H_GATE, {2}, "G3"));
    ckt.insert_gate(0, Gate(H_GATE, {1}, "G4"));
    ckt.insert_gate(0, Gate(H_GATE, {0}, "G5"));
    // following nets: CNOTs G6..G9
    ckt.insert_gate(1, Gate(CNOT_GATE, {3, 4}, "G6"));
    ckt.insert_gate(2, Gate(CNOT_GATE, {1, 4}, "G7"));
    ckt.insert_gate(3, Gate(CNOT_GATE, {2, 3}, "G8"));
    ckt.insert_gate(4, Gate(CNOT_GATE, {0, 2}, "G9"));
    return ckt;
}

// ---------- Main ----------
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    SimContext ctx;
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.nranks);

    ctx.nqubits = 5;
    ctx.N = pow2(ctx.nqubits);
    ctx.block_size = 4; // as paper example uses B=4 in illustration (tunable)

    build_partitions(ctx);

    if (ctx.rank == 0)
    {
        std::cout << "MPI ranks: " << ctx.nranks << ", partitions: " << ctx.partitions.size() << ", block_size: " << ctx.block_size << "\n";
    }

    // build the example circuit
    Circuit ckt = build_paper_5q_circuit();

    // init state
    std::vector<cplx> state;
    init_state(state, ctx.N);

    // Full simulation (initial)
    if (ctx.rank == 0)
        std::cout << "== Performing full simulation (initial) ==\n";
    // For full sim, pass all partitions as inserted (conservative) and none removed
    std::vector<int> all_parts;
    for (int p = 0; p < (int)ctx.partitions.size(); ++p)
        all_parts.push_back(p);
    update_state(ctx, ckt, state, all_parts, {}, true);

    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank == 0)
        print_full_state_if_rank0(ctx.rank, state);

    // Now modify circuit: remove G8 (net 3), insert G10 in same net (CNOT 1->2)
    if (ctx.rank == 0)
        std::cout << "== Modify circuit: remove G8 and insert G10 ==\n";
    // We'll compute which partitions are affected by G8 and G10 using partitions_touched_by_gate
    Gate G8(CNOT_GATE, {2, 3}, "G8");
    Gate G10(CNOT_GATE, {1, 2}, "G10");
    auto rem_parts_vec = partitions_touched_by_gate(ctx, G8);
    auto ins_parts_vec = partitions_touched_by_gate(ctx, G10);
    // Remove G8
    ckt.remove_gate_by_name(3, "G8");
    // Insert G10 into net 3
    ckt.insert_gate(3, G10); // push to net 3 (after removal)
    // Convert rem/ins partitions lists to vectors
    if (ctx.rank == 0)
    {
        std::cout << "Removed gate G8 affected partitions: ";
        for (int p : rem_parts_vec)
            std::cout << p << " ";
        std::cout << "\nInserted gate G10 touches partitions: ";
        for (int p : ins_parts_vec)
            std::cout << p << " ";
        std::cout << "\n";
    }

    // Run incremental update with frontiers: inserted partitions and successors of removed partitions
    update_state(ctx, ckt, state, ins_parts_vec, rem_parts_vec, false);

    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx.rank == 0)
        std::cout << "== After incremental update ==\n";
    if (ctx.rank == 0)
        print_full_state_if_rank0(ctx.rank, state);

    MPI_Finalize();
    return 0;
}
