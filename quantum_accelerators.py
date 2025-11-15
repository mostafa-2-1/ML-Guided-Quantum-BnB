
# start hereee 
#!/usr/bin/env python3
"""
quantum_accelerators.py

Updated: Quantum-Assisted Beam Search TSP hybrid pipeline.

Features:
- Beam-search integration to produce a FULL TSP tour (0 -> ... -> 0).
- Grover-based min-finder over top-k candidate lists (Qiskit).
- Optional Montanaro-style feasibility check (practical enumeration + Grover) for small subtrees.
- Works on large pruned graphs: quantum calls restricted to small local subproblems.
- CLI flags, JSON+Markdown outputs, timing and resource logging.



"""

import argparse
import json
import math
import os
import sys
import time
from copy import deepcopy
from typing import List, Tuple, Dict, Any
from qiskit_aer import AerSimulator
import networkx as nx
import numpy as np

# Qiskit imports 
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate

from qiskit.quantum_info import Operator
# --------------------------
# ---- Graph I/O / helpers
# --------------------------
def load_pruned_graph(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
    for key in ('num_nodes', 'edges', 'distance_matrix'):
        if key not in data:
            raise ValueError(f"Pruned graph JSON missing required key: {key}")
    return data

def save_json(obj: Any, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_text(text: str, path: str):
    with open(path, 'w') as f:
        f.write(text)

def check_connectivity(n_nodes: int, edges: List[List[int]]) -> bool:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges])
    return nx.is_connected(G)

def ensure_connected(n_nodes:int, edges:List[List[int]], D:List[List[float]]) -> List[List[int]]:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges])
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return edges
    new_edges = edges.copy()
    # greedy connect nearest component pairs
    while len(components) > 1:
        best_pair = None
        best_d = float('inf')
        for i, compA in enumerate(components):
            for compB in components[i+1:]:
                for a in compA:
                    for b in compB:
                        d = D[a][b]
                        if d < best_d:
                            best_d = d
                            best_pair = (a,b)
        new_edges.append([best_pair[0], best_pair[1]])
        G.add_edge(*best_pair)
        components = list(nx.connected_components(G))
    return new_edges

# -------------------------
# Qiskit: Grover helpers (diagonal oracle)
# -------------------------
def build_phase_oracle_for_marked_indices(marked_indices: List[int], index_qubits: int) -> QuantumCircuit:
    dim = 1 << max(1, index_qubits)
    diag = np.ones(dim, dtype=complex)
    for m in marked_indices:
        if m < dim:
            diag[m] = -1.0
    U = np.diag(diag)
    gate = UnitaryGate(U, label="OracleDiag")
    qc = QuantumCircuit(index_qubits, name='Oracle')
    qc.append(gate, list(range(index_qubits)))
    return qc

def diffusion_operator(index_qubits: int) -> QuantumCircuit:
    dim = 1 << max(1, index_qubits)
    J = np.ones((dim, dim)) / dim
    D = 2 * J - np.eye(dim)
    gate = UnitaryGate(D, label="Diffusion")
    qc = QuantumCircuit(index_qubits, name='Diffusion')
    qc.append(gate, list(range(index_qubits)))
    return qc

def grover_search_marked_indices(marked_indices: List[int],
                                 index_qubits: int,
                                 repetitions: int,
                                 backend=None,
                                 shots: int = 512,
                                 time_budget_remaining: float = None,
                                 start_time: float = None) -> Dict[str,Any]:
    """
    Run Grover on the small index space. We check the remaining time budget before heavy steps.
    """
    if index_qubits < 1:
        index_qubits = 1
    qc = QuantumCircuit(index_qubits, index_qubits)
    qc.h(range(index_qubits))
    oracle_qc = build_phase_oracle_for_marked_indices(marked_indices, index_qubits)
    diffusion_qc = diffusion_operator(index_qubits)
    oracle_gate = oracle_qc.to_gate()
    diffusion_gate = diffusion_qc.to_gate()
    for _ in range(repetitions):
        # time check
        if time_budget_remaining is not None and start_time is not None:
            if (time.time() - start_time) > time_budget_remaining:
                return {'error': 'time_budget_exceeded_before_iteration'}
        qc.append(oracle_gate, list(range(index_qubits)))
        qc.append(diffusion_gate, list(range(index_qubits)))
    qc.measure(range(index_qubits), range(index_qubits))
    backend = backend or AerSimulator(method='statevector')
    t0 = time.time()
    compiled = transpile(qc, backend=backend)
    t1 = time.time()
    # time check before running heavy simulation
    if time_budget_remaining is not None and start_time is not None:
        if (time.time() - start_time) > time_budget_remaining:
            return {'error': 'time_budget_exceeded_before_simulation', 'transpile_time_s': t1 - t0}
    job = backend.run(compiled, shots=shots)
    res = job.result()
    t2 = time.time()
    counts = res.get_counts()
    timing = {'transpile_time_s': t1 - t0, 'sim_time_s': t2 - t1, 'total_time_s': t2 - t0}
    return {'counts': counts, 'timing': timing, 'qc': qc, 'compiled_qc': compiled}

# -------------------------
# small helpers: top-k, subtree extraction
# -------------------------
def topk_successors_from_pruned_graph(pruned_graph: Dict[str,Any], node:int, k:int) -> List[Tuple[int,float]]:
    D = pruned_graph['distance_matrix']
    neighbors = set()
    for a,b in pruned_graph['edges']:
        if a==node:
            neighbors.add(b)
        elif b==node:
            neighbors.add(a)
    neighs = [(nb, D[node][nb]) for nb in neighbors]
    neighs.sort(key=lambda x: x[1])
    return neighs[:k]

def extract_local_subtree(pruned_graph: Dict[str,Any], root:int, max_nodes:int=12) -> Dict[str,Any]:
    n = pruned_graph['num_nodes']
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([tuple(e) for e in pruned_graph['edges']])
    visited = [root]
    queue = [root]
    while queue and len(visited) < max_nodes:
        cur = queue.pop(0)
        for nb in G.neighbors(cur):
            if nb not in visited:
                visited.append(nb)
                queue.append(nb)
            if len(visited) >= max_nodes:
                break
    nodes = visited[:max_nodes]
    mapping = {nodes[i]: i for i in range(len(nodes))}
    local_edges = []
    for a,b in pruned_graph['edges']:
        if a in mapping and b in mapping:
            local_edges.append((mapping[a], mapping[b]))
    D_full = pruned_graph['distance_matrix']
    D = [[D_full[orig_a][orig_b] for orig_b in nodes] for orig_a in nodes]
    return {'nodes': nodes, 'edges': local_edges, 'distance_matrix': D, 'orig_to_local': mapping, 'local_to_orig': nodes}

# -------------------------
# Montanaro practical enumeration (small subtree)
# -------------------------
def enumerate_partial_assignments(subtree: Dict[str,Any], max_length:int=None) -> List[Dict[str,Any]]:
    nodes = list(range(len(subtree['nodes'])))
    D = subtree['distance_matrix']
    from itertools import permutations
    s = len(nodes)
    max_len = s if max_length is None else min(s, max_length)
    assignments = []
    for r in range(1, max_len+1):
        for perm in permutations(nodes, r):
            feasible = True
            cost = 0.0
            for i in range(1, len(perm)):
                a,b = perm[i-1], perm[i]
                if (a,b) not in subtree['edges'] and (b,a) not in subtree['edges']:
                    feasible = False
                    break
                cost += D[a][b]
            if feasible:
                bitmask = 0
                for v in perm:
                    bitmask |= (1<<v)
                assignments.append({'bitmask': bitmask, 'last': perm[-1], 'path': list(perm), 'cost': cost})
    # deduplicate
    best = {}
    for a in assignments:
        key = (a['bitmask'], a['last'])
        if key not in best or a['cost'] < best[key]['cost']:
            best[key] = a
    return list(best.values())

def montanaro_practical_feasibility(subtree: Dict[str,Any], cost_bound: float, backend=None, start_time:float=None, time_budget_remaining:float=None) -> Dict[str,Any]:
    enumerated = enumerate_partial_assignments(subtree)
    if len(enumerated) == 0:
        return {'feasible': False, 'reason': 'no_enumerated'}
    marked = [i for i,a in enumerate(enumerated) if a['cost'] <= cost_bound]
    if len(marked) == 0:
        return {'feasible': False, 'reason': 'no_marked', 'num_enumerated': len(enumerated)}
    index_qubits = max(1, math.ceil(math.log2(len(enumerated))))
    N = 1 << index_qubits
    M = len(marked)
    R = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))
    # guard time budget
    res = grover_search_marked_indices(marked, index_qubits, R, backend=backend, shots=256, time_budget_remaining=time_budget_remaining, start_time=start_time)
    if 'error' in res:
        return {'feasible': False, 'reason': 'time_budget_exceeded_during_grover', 'details': res}
    counts = res['counts']
    measured = max(counts.items(), key=lambda x: x[1])[0]
    measured_int = int(measured, 2)
    assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
    return {'feasible': assgn is not None and assgn['cost'] <= cost_bound, 'measured_assignment': assgn, 'num_enumerated': len(enumerated), 'marked_count': len(marked), 'grover_counts': counts, 'timing': res['timing']}

# -------------------------
# Beam Search + Controlled Quantum usage
# -------------------------
def path_cost(path: List[int], Dmat: List[List[float]]) -> float:
    c = 0.0
    for i in range(1, len(path)):
        c += Dmat[path[i-1]][path[i]]
    return c

def close_tour_cost(path: List[int], Dmat: List[List[float]]) -> float:
    c = path_cost(path, Dmat)
    c += Dmat[path[-1]][path[0]]
    return c

def average_branching_factor(pruned_graph: Dict[str,Any]) -> float:
    n = pruned_graph['num_nodes']
    degs = [0]*n
    for a,b in pruned_graph['edges']:
        degs[a] += 1
        degs[b] += 1
    # average non-zero degree
    nz = [d for d in degs if d>0]
    return float(sum(nz))/len(nz) if len(nz)>0 else 0.0

def beam_search_quantum(pruned_graph: Dict[str,Any],
                        beam_size:int,
                        topk:int,
                        use_montanaro:bool,
                        force_montanaro:bool,
                        n_max_quantum_user:int,
                        max_runtime:float,
                        backend=None):
    backend = backend or AerSimulator(method='statevector')
    n = pruned_graph['num_nodes']
    D = pruned_graph['distance_matrix']

    # dynamic n_max_quantum selection 
    if n_max_quantum_user is not None:
        n_max_quantum = n_max_quantum_user
    else:
        if n <= 10:
            n_max_quantum = 10
        elif n <= 20:
            n_max_quantum = 6
        elif n <= 50:
            n_max_quantum = 4
        else:
            n_max_quantum = 0

    branch = average_branching_factor(pruned_graph)
    # determine if montanaro should be allowed
    montanaro_allowed = use_montanaro and (n_max_quantum >= 3)
    if not force_montanaro:
        # auto-disable for large graphs or high branching
        if n > 20 or branch > 4.0:
            montanaro_allowed = False

    # global runtime tracking
    start_time = time.time()
    deadline = start_time + max_runtime

    def time_left():
        return max(0.0, deadline - time.time())

    # quick progress logger
    last_log_time = time.time()

    # helper: feasible neighbors excluding visited; cap to topk
    def feasible_neighbors(node:int, visited_mask:int):
        neighs = topk_successors_from_pruned_graph(pruned_graph, node, topk)
        out = [(nb,dist) for (nb,dist) in neighs if not (visited_mask & (1<<nb))]
        return out

    beam = [{'path':[0], 'visited': (1<<0), 'cost': 0.0}]
    total_quantum_time = 0.0
    quantum_calls = 0
    logs = {'steps': []}

    # Hard safety caps 
    # disallow enumerations that exceed 2000 items
    ENUMERATION_SAFE_LIMIT = 3000

    for step in range(1, n):
        # check global time budget
        if time.time() > deadline:
            print(f"[TIMEOUT] Reached global time budget at step {step}. Returning best-so-far.")
            break

        # progress print
        if time.time() - last_log_time > 10.0:
            print(f"[PROGRESS] step {step}/{n-1}, beam_size={len(beam)}, time_left={time_left():.1f}s, montanaro_allowed={montanaro_allowed}")
            last_log_time = time.time()

        expansions = []
        step_record = {'step': step, 'in_beam': deepcopy(beam), 'expansions': []}
        for item in beam:
            # check time budget inside loop
            if time.time() > deadline:
                print("[TIMEOUT] budget exhausted while expanding beam.")
                break
            last_city = item['path'][-1]
            visited = item['visited']
            neighs = feasible_neighbors(last_city, visited)
            if len(neighs) == 0:
                # fallback: choose nearest unvisited classically
                remaining = [i for i in range(n) if not (visited & (1<<i))]
                if len(remaining) == 0:
                    continue
                rem_sorted = sorted(remaining, key=lambda x: D[last_city][x])
                neighs = [(rem_sorted[0], D[last_city][rem_sorted[0]])]

            candidates = [t[0] for t in neighs]
            dists = [t[1] for t in neighs]

            # limit candidate pool size to keep Grover cost small (practical cap)
            MAX_CANDIDATES_CAP = min(len(candidates), 64)  # safe cap
            candidates = candidates[:MAX_CANDIDATES_CAP]
            dists = dists[:MAX_CANDIDATES_CAP]

            # mark the classical minimum index to simulate predicate ((Grover picks it)
            min_idx = int(np.argmin(dists))
            marked = [min_idx]

            index_qubits = max(1, math.ceil(math.log2(len(candidates))))
            N = 1 << index_qubits
            M = max(1, len(marked))
            repetitions = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))

            # guard if not enough time left to execute this Grover call
            # allocate a small slice of remaining time to each call  (rough heuristic)
            per_call_budget = max(1.0, time_left() * 0.15)  # allow up to 15% of remaining time
            grover_res = grover_search_marked_indices(marked, index_qubits, repetitions, backend=backend, shots=256, time_budget_remaining=per_call_budget, start_time=start_time)
            if 'error' in grover_res:
                # fallback to classical pick to avoid stalling
                chosen_idx = min_idx
                grover_info = {'fallback': True, 'reason': grover_res.get('error')}
            else:
                quantum_calls += 1
                total_quantum_time += grover_res['timing']['total_time_s']
                counts = grover_res['counts']
                measured = max(counts.items(), key=lambda x: x[1])[0]
                chosen_idx = int(measured, 2)
                grover_info = {'counts': counts, 'timing': grover_res['timing']}
            # clamp chosen idx into candidate range
            if chosen_idx >= len(candidates):
                chosen_idx = min_idx
            chosen_candidate = candidates[chosen_idx]

            new_path = item['path'] + [chosen_candidate]
            new_cost = item['cost'] + D[last_city][chosen_candidate]
            new_visited = visited | (1<<chosen_candidate)

            expansion = {'from': item['path'], 'chosen': chosen_candidate, 'new_path': new_path, 'new_cost': new_cost, 'grover_info': grover_info}
            #  run Montanaro practical feasibility on a tiny subtree
            if montanaro_allowed:
                # ensure we have time and subtree small enough
                subtree = extract_local_subtree(pruned_graph, chosen_candidate, max_nodes=n_max_quantum)
                if len(subtree['nodes']) <= n_max_quantum:
                    # quick enumeration size check
                    # approximate enumeration count: sum_{r=1..s} P(s,r) , we simply skip if too large
                    s = len(subtree['nodes'])
                    # crude bound: s! (will be huge) so skip if s>9 by default to keep fast, but allow if user forced
                    if (not force_montanaro) and s > 9:
                        expansion['montanaro'] = {'skipped': True, 'reason': f'subtree_size_{s}_too_large_for_auto_montanaro'}
                    else:
                        # set cost bound heuristic: current cost + average_edge*(remaining)
                        remaining = n - len(new_path)
                        avg_edge = 0.0
                        if len(subtree['edges'])>0:
                            vals = [subtree['distance_matrix'][a][b] for a,b in subtree['edges']]
                            avg_edge = sum(vals)/len(vals)
                        estimated_extra = remaining * max(1.0, avg_edge)
                        cost_bound = new_cost + estimated_extra
                        # check enumeration size quickly
                        # produce enumerated and ensure not exceeding safe limit
                        enumerated = enumerate_partial_assignments(subtree)
                        if len(enumerated) > ENUMERATION_SAFE_LIMIT and not force_montanaro:
                            expansion['montanaro'] = {'skipped': True, 'reason': f'enumeration_too_large_{len(enumerated)}'}
                        else:
                            # run Montanaro practical with a small per-call budget
                            per_ma_budget = min( max(1.0, time_left()*0.10), 30.0 )  # 10% of remaining up to 30s
                            ma_start = time.time()
                            ma_res = montanaro_practical_feasibility(subtree, cost_bound, backend=backend, start_time=start_time, time_budget_remaining=per_ma_budget)
                            ma_end = time.time()
                            if 'timing' in ma_res:
                                total_quantum_time += ma_res['timing'].get('total_time_s', 0.0)
                            expansion['montanaro'] = {'result': ma_res, 'runtime_s': ma_end - ma_start}
                else:
                    expansion['montanaro'] = {'skipped': True, 'reason': 'subtree_larger_than_n_max_quantum'}
            step_record['expansions'].append(expansion)
            expansions.append({'path': new_path, 'visited': new_visited, 'cost': new_cost})
            # time guard after each expansion
            if time.time() > deadline:
                print("[TIMEOUT] budget exhausted mid-expansion.")
                break

        # sort expansions by cost  and keep top beam_size
        expansions.sort(key=lambda x: x['cost'])
        beam = expansions[:beam_size] if len(expansions)>0 else beam
        logs['steps'].append(step_record)

        #  if beam contains full-length tours break early
        if any(len(b['path']) == n for b in beam):
            break

    # build full tours from beam (fill greedily any missing nodes)
    full_tours = []
    for b in beam:
        cur = b['path'][:]
        visited = b['visited']
        while len(cur) < n:
            last = cur[-1]
            remaining = [i for i in range(n) if not (visited & (1<<i))]
            if not remaining:
                break
            # choose nearest remaining
            nextc = min(remaining, key=lambda x: D[last][x])
            cur.append(nextc)
            visited |= (1<<nextc)
        cost = close_tour_cost(cur, D)
        full_tours.append({'path': cur, 'cost': cost})
    best = min(full_tours, key=lambda x: x['cost']) if len(full_tours)>0 else None

    result = {
        'best_tour': best,
        'quantum_time_total_s': total_quantum_time,
        'quantum_calls': quantum_calls,
        'wall_time_elapsed_s': time.time() - start_time,
        'logs': logs,
        'montanaro_allowed': montanaro_allowed,
        'n_max_quantum_used': n_max_quantum
    }
    return result

# -------------------------
# CLI / main
# -------------------------
def default_output_paths(input_json_path: str):
    base = os.path.splitext(os.path.basename(input_json_path))[0]
    out_dir = "quantum_run_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    return {'json': os.path.join(out_dir, f"{base}_beamrun_{ts}.json"), 'md': os.path.join(out_dir, f"{base}_beamrun_{ts}.md")}

def pretty_md_report(summary: Dict[str,Any], input_info: Dict[str,Any]) -> str:
    md = []
    md.append("# Quantum-Assisted Beam Search Run")
    md.append("")
    md.append("## Input")
    for k,v in input_info.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Summary")
    for k,v in summary.items():
        if k != 'logs':
            md.append(f"- **{k}**: {v}")
    if summary.get('best_tour'):
        md.append("")
        md.append("### Best tour")
        md.append(f"- Path: {summary['best_tour']['path']}")
        md.append(f"- Cost: {summary['best_tour']['cost']}")
    return "\n".join(md)

def main_cli():
    parser = argparse.ArgumentParser(description="Quantum-Assisted Beam Search TSP with runtime caps.")
    parser.add_argument('pruned_graph', help='pruned graph JSON')
    parser.add_argument('--beam', type=int, default=3, help='beam size (default 3)')
    parser.add_argument('--k', type=int, default=16, help='top-k neighbors for Grover (default 16)')
    parser.add_argument('--use_montanaro', action='store_true', help='attempt to use Montanaro practical feasibility checks (may be auto-disabled)')
    parser.add_argument('--force_montanaro', action='store_true', help='force Montanaro even if autosafety would disable it (careful)')
    parser.add_argument('--n_max_quantum', type=int, default=None, help='max nodes for Montanaro enumeration (auto-chosen if omitted)')
    parser.add_argument('--max_runtime', type=int, default=300, help='global runtime budget in seconds (default 300)')
    parser.add_argument('--k_cap', type=int, default=64, help='hard cap on candidate pool passed to Grover (default 64)')
    args = parser.parse_args()

    # load graph
    pruned = load_pruned_graph(args.pruned_graph)
    n = pruned['num_nodes']
    edges = [list(e) for e in pruned['edges']]
    D = pruned['distance_matrix']

    # ensure connectivity
    if not check_connectivity(n, edges):
        print("[INFO] Graph not connected — repairing...")
        edges = ensure_connected(n, edges, D)
        pruned['edges'] = edges
        print("[INFO] Connectivity repaired.")

    # configure backend
    backend = AerSimulator(method='statevector')

    # run
    print(f"[START] Running beam search quantum on {args.pruned_graph} (n={n}), beam={args.beam}, k={args.k}")
    t0 = time.time()
    res = beam_search_quantum(pruned, beam_size=args.beam, topk=args.k, use_montanaro=args.use_montanaro, force_montanaro=args.force_montanaro, n_max_quantum_user=args.n_max_quantum, max_runtime=args.max_runtime, backend=backend)
    t1 = time.time()
    total_wall = t1 - t0
    res['wall_time_total_s'] = total_wall

    # output
    out_paths = default_output_paths(args.pruned_graph)
    out_json = {
        'input': {'path': args.pruned_graph, 'num_nodes': n, 'beam': args.beam, 'k': args.k, 'use_montanaro': args.use_montanaro},
        'result': res
    }
    save_json(out_json, out_paths['json'])
    md = pretty_md_report(res, out_json['input'])
    save_text(md, out_paths['md'])
    print(f"[DONE] Results saved: {out_paths['json']}, {out_paths['md']}")
    # print a summary
    if res.get('best_tour'):
        print(f" Best tour cost: {res['best_tour']['cost']}, path len: {len(res['best_tour']['path'])}")
    print(f" Quantum calls: {res.get('quantum_calls')}, quantum time total: {res.get('quantum_time_total_s'):.3f}s")
    print(f" Wall time total: {res.get('wall_time_total_s'):.3f}s")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main_cli())
    except KeyboardInterrupt:
        print("\n[ABORT] User Cancelled.")
        sys.exit(1)
