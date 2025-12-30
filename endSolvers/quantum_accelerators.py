
# """
# quantum_beam_concise.py - Compact quantum-assisted beam search with clean output
# """

# import argparse
# import json
# import math
# import os
# import sys
# import time
# import logging
# from typing import List, Tuple, Dict, Any

# import networkx as nx
# import numpy as np
# import random

# from qiskit_aer import AerSimulator
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

# # ---------------------------
# # Configuration
# # ---------------------------
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger("quantum_beam")

# # ---------------------------
# # I/O & Graph Utilities
# # ---------------------------
# def load_pruned_graph(path: str) -> Dict[str, Any]:
#     with open(path, 'r') as f:
#         return json.load(f)

# def save_json(obj: Any, path: str):
#     with open(path, 'w') as f:
#         json.dump(obj, f, indent=2)

# # ---------------------------
# # Quantum Circuit (Simplified)
# # ---------------------------
# def grover_search(marked_indices: List[int], index_qubits: int, repetitions: int, backend) -> Dict[str, Any]:
#     """Simple Grover search that returns timing information."""
#     if index_qubits < 1:
#         index_qubits = 1
    
#     qc = QuantumCircuit(index_qubits, index_qubits)
#     qc.h(range(index_qubits))
    
#     # Build oracle (diagonal unitary)
#     dim = 1 << index_qubits
#     diag = np.ones(dim, dtype=complex)
#     for m in marked_indices:
#         if 0 <= m < dim:
#             diag[m] = -1.0
#     oracle_matrix = np.diag(diag)
    
#     # Build diffusion
#     J = np.ones((dim, dim)) / dim
#     diffusion_matrix = 2 * J - np.eye(dim)
    
#     # Apply Grover iterations
#     for _ in range(repetitions):
#         qc.unitary(oracle_matrix, range(index_qubits), label='oracle')
#         qc.unitary(diffusion_matrix, range(index_qubits), label='diffusion')
    
#     qc.measure(range(index_qubits), range(index_qubits))
    
#     start = time.time()
#     try:
#         compiled = transpile(qc, backend=backend)
#         job = backend.run(compiled, shots=256)
#         result = job.result()
#         counts = result.get_counts()
#         elapsed = time.time() - start
#         return {'counts': counts, 'time': elapsed, 'success': True}
#     except Exception as e:
#         elapsed = time.time() - start
#         return {'error': str(e), 'time': elapsed, 'success': False}

# # ---------------------------
# # Core Beam Search Algorithm
# # ---------------------------
# def beam_search_quantum(
#     pruned_graph: Dict[str, Any],
#     beam_size: int = 4,
#     topk: int = 12,
#     restarts: int = 2,
#     max_runtime: float = 300,
#     enable_2opt: bool = True
# ) -> Dict[str, Any]:
#     """Main beam search algorithm."""
#     n = pruned_graph['num_nodes']
#     D = pruned_graph['distance_matrix']
#     backend = AerSimulator()
    
#     start_time = time.time()
#     deadline = start_time + max_runtime
    
#     def time_left(): return max(0.0, deadline - time.time())
    
#     best_tour = None
#     best_cost = float('inf')
#     quantum_time = 0.0
    
#     for restart in range(restarts):
#         if time.time() > deadline:
#             break
            
#         # Initialize beam
#         beam = [{'path': [0], 'visited': 1 << 0, 'cost': 0.0}]
        
#         for step in range(1, n):
#             if time.time() > deadline:
#                 break
                
#             expansions = []
            
#             for item in beam:
#                 last = item['path'][-1]
#                 visited = item['visited']
                
#                 # Get unvisited neighbors from pruned edges
#                 neighbors = []
#                 for a, b in pruned_graph['edges']:
#                     if a == last and not (visited & (1 << b)):
#                         neighbors.append((b, D[last][b]))
#                     elif b == last and not (visited & (1 << a)):
#                         neighbors.append((a, D[last][a]))
                
#                 # Take top-k
#                 neighbors.sort(key=lambda x: x[1])
#                 candidates = [n[0] for n in neighbors[:topk]]
                
#                 if not candidates:
#                     # Classical fallback
#                     remaining = [i for i in range(n) if not (visited & (1 << i))]
#                     if remaining:
#                         chosen = min(remaining, key=lambda x: D[last][x])
#                         expansions.append({
#                             'path': item['path'] + [chosen],
#                             'visited': visited | (1 << chosen),
#                             'cost': item['cost'] + D[last][chosen]
#                         })
#                     continue
                
#                 # Quantum selection
#                 dists = [D[last][c] for c in candidates]
#                 min_d = min(dists)
#                 slack = 1.10
                
#                 # Mark candidates within slack
#                 marked = [i for i, d in enumerate(dists) if d <= min_d * slack]
#                 marked = marked[:min(3, len(marked))]
                
#                 # Run Grover
#                 index_qubits = max(1, math.ceil(math.log2(len(candidates))))
#                 repetitions = max(1, int((math.pi / 4) * math.sqrt(2 ** index_qubits / max(1, len(marked)))))
#                 repetitions = min(repetitions, 3)  # Safety cap
                
#                 grover_start = time.time()
#                 grover_res = grover_search(marked, index_qubits, repetitions, backend)
#                 quantum_time += time.time() - grover_start
                
#                 # Process result
#                 if not grover_res.get('success', False):
#                     chosen_idx = min(range(len(candidates)), key=lambda x: dists[x])
#                 else:
#                     counts = grover_res.get('counts', {})
#                     if counts:
#                         measured = max(counts.items(), key=lambda x: x[1])[0]
#                         chosen_idx = int(measured, 2) % len(candidates)
#                     else:
#                         chosen_idx = min(range(len(candidates)), key=lambda x: dists[x])
                
#                 chosen = candidates[chosen_idx]
#                 expansions.append({
#                     'path': item['path'] + [chosen],
#                     'visited': visited | (1 << chosen),
#                     'cost': item['cost'] + D[last][chosen]
#                 })
            
#             # Prune to beam size
#             expansions.sort(key=lambda x: x['cost'])
#             beam = expansions[:beam_size]
            
#             # Early exit if complete
#             if any(len(b['path']) == n for b in beam):
#                 break
        
#         # Complete tours and find best for this restart
#         for b in beam:
#             path = b['path'][:]
#             visited = b['visited']
            
#             # Complete greedily
#             while len(path) < n:
#                 last = path[-1]
#                 remaining = [i for i in range(n) if not (visited & (1 << i))]
#                 if not remaining:
#                     break
#                 chosen = min(remaining, key=lambda x: D[last][x])
#                 path.append(chosen)
#                 visited |= (1 << chosen)
            
#             if len(path) == n:
#                 # Compute closed tour cost
#                 cost = sum(D[path[i-1]][path[i]] for i in range(1, n)) + D[path[-1]][path[0]]
                
#                 # 2-opt improvement
#                 if enable_2opt and time_left() > 2.0:
#                     improved = path[:]
#                     improved_cost = cost
                    
#                     # Simple 2-opt
#                     for i in range(1, n - 2):
#                         for j in range(i + 1, n):
#                             if time.time() > deadline:
#                                 break
#                             new_path = improved[:i] + improved[i:j][::-1] + improved[j:]
#                             new_cost = sum(D[new_path[k-1]][new_path[k]] for k in range(1, n)) + D[new_path[-1]][new_path[0]]
#                             if new_cost < improved_cost:
#                                 improved, improved_cost = new_path, new_cost
                    
#                     path, cost = improved, improved_cost
                
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_tour = path
    
#     wall_time = time.time() - start_time
#     classical_time = wall_time - quantum_time
    
#     return {
#         'tour': best_tour,
#         'cost': best_cost,
#         'runtime_sec': wall_time,
#         'quantum_runtime_sec': quantum_time,
#         'classical_runtime_sec': classical_time,
#         'restarts': restarts,
#         'beam_size': beam_size,
#         'topk': topk
#     }

# # ---------------------------
# # Main Function
# # ---------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Quantum Beam Search for TSP")
#     parser.add_argument('pruned_graph', help='Path to pruned graph JSON')
#     parser.add_argument('--beam', type=int, default=4, help='Beam size')
#     parser.add_argument('--topk', type=int, default=12, help='Top-k neighbors')
#     parser.add_argument('--restarts', type=int, default=2, help='Number of restarts')
#     parser.add_argument('--max_runtime', type=int, default=300, help='Max runtime in seconds')
#     parser.add_argument('--no_2opt', action='store_true', help='Disable 2-opt improvement')
#     parser.add_argument('--output_dir', type=str, default='quantum_results', help='Output directory')
    
#     args = parser.parse_args()
    
#     # Load graph
#     pruned = load_pruned_graph(args.pruned_graph)
#     n = pruned['num_nodes']
    
#     # Extract instance name
#     instance = os.path.splitext(os.path.basename(args.pruned_graph))[0]
    
#     logger.info(f"Starting quantum beam search on {instance} (n={n})")
#     logger.info(f"Beam: {args.beam}, TopK: {args.topk}, Restarts: {args.restarts}")
    
#     # Run search
#     result = beam_search_quantum(
#         pruned_graph=pruned,
#         beam_size=args.beam,
#         topk=args.topk,
#         restarts=args.restarts,
#         max_runtime=args.max_runtime,
#         enable_2opt=not args.no_2opt
#     )
    
#     # Prepare concise output
#     concise_result = {
#         "instance": instance,
#         "n": n,
#         "tour_cost": result['cost'],
#         "tour": result['tour'] + [result['tour'][0]] if result['tour'] else None,  # Close the tour
#         "runtime_sec": result['runtime_sec'],
#         "quantum_runtime_sec": result['quantum_runtime_sec'],
#         "classical_runtime_sec": result['classical_runtime_sec'],
#         "total_runtime_sec": result['runtime_sec']  # For backward compatibility
#     }
    
#     # Save results
#     os.makedirs(args.output_dir, exist_ok=True)
#     timestamp = int(time.time())
    
#     # Save concise JSON
#     concise_path = os.path.join(args.output_dir, f"{instance}_quantum_{timestamp}.json")
#     save_json(concise_result, concise_path)
    
#     # Also save detailed results if needed
#     detailed_path = os.path.join(args.output_dir, f"{instance}_quantum_detailed_{timestamp}.json")
#     save_json({
#         'instance': instance,
#         'n': n,
#         'config': {
#             'beam_size': args.beam,
#             'topk': args.topk,
#             'restarts': args.restarts,
#             'max_runtime': args.max_runtime,
#             'enable_2opt': not args.no_2opt
#         },
#         'result': result
#     }, detailed_path)
    
#     # Print summary
#     logger.info(f"Results saved to {concise_path}")
#     if result['tour']:
#         logger.info(f"Best tour cost: {result['cost']:.3f}")
#         logger.info(f"Total runtime: {result['runtime_sec']:.2f}s")
#         logger.info(f"Quantum runtime: {result['quantum_runtime_sec']:.2f}s")
#         logger.info(f"Classical runtime: {result['classical_runtime_sec']:.2f}s")
    
#     return 0

# if __name__ == "__main__":
#     sys.exit(main())






"""
quantumSolver_runner.py
Batch quantum-assisted beam search over all graphs (full, EdgeGNN-only, EdgeGNN+cascade)
"""

import os
import sys
import json
import time
import math
import random
import logging
import numpy as np

from typing import Dict, Any, List
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level from endSolvers
sys.path.insert(0, parent_dir)  # Now Python can find modules in root

from newTrain import parse_tsp
from preprocess_data import build_candidate_edges

# =========================
# Configuration (FIXED)
# =========================

GRAPH_DIR = "graphs_chosen"
EDGEGNN_ONLY = "savedGraphs_EdgeGNN_only"
CASCADE_ROOT = "savedGraphs"
RESULTS_DIR = "quantumSolver_results"

BEAM_SIZE = 5
TOPK = 11
RESTARTS = 4
SHOTS = 16
MAX_RUNTIME = 300
ENABLE_2OPT = True

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("quantum_beam")

# =========================
# Quantum Executor
# =========================

class SerialQuantumExecutor:
    def __init__(self, shots: int):
        self.backend = AerSimulator()
        self.shots = shots
        self.total_quantum_time = 0.0
        self.call_count = 0

    def run_grover(self, marked_indices, index_qubits, repetitions=1):
        if not marked_indices or index_qubits == 0:
            return marked_indices[0]

        self.call_count += 1
        qc = QuantumCircuit(index_qubits, index_qubits)
        qc.h(range(index_qubits))

        target = marked_indices[0]
        for i in range(index_qubits):
            if not (target >> i) & 1:
                qc.x(i)

        qc.h(index_qubits - 1)
        qc.mcx(list(range(index_qubits - 1)), index_qubits - 1)
        qc.h(index_qubits - 1)

        for i in range(index_qubits):
            if not (target >> i) & 1:
                qc.x(i)

        qc.measure(range(index_qubits), range(index_qubits))

        start = time.time()
        compiled = transpile(qc, self.backend, optimization_level=0)
        result = self.backend.run(compiled, shots=self.shots).result()
        counts = result.get_counts()
        self.total_quantum_time += time.time() - start

        measured = max(counts, key=counts.get)
        return marked_indices[int(measured, 2) % len(marked_indices)]

    def stats(self):
        return {
            "quantum_calls": self.call_count,
            "quantum_runtime_sec": self.total_quantum_time,
        }

# =========================
# Utilities
# =========================

def load_pruned_graph(json_path, coords):
    with open(json_path) as f:
        data = json.load(f)

    n = data["n"]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    for e in data["edges"]:
        u, v = e[0], e[1]
        w = e[2] if len(e) == 3 else np.linalg.norm(coords[u] - coords[v])
        D[u, v] = w
        D[v, u] = w

    return {
        "num_nodes": n,
        "distance_matrix": D,
        "edges": [(u, v) for u in range(n) for v in range(n) if D[u, v] < np.inf and u != v]
    }

def close_tour_cost(path, D):
    return sum(D[path[i-1]][path[i]] for i in range(len(path))) + D[path[-1]][path[0]]

# =========================
# Quantum Beam Search
# =========================

def quantum_beam_search(pruned_graph):
    n = pruned_graph["num_nodes"]
    D = pruned_graph["distance_matrix"]

    quantum = SerialQuantumExecutor(SHOTS)
    start_time = time.time()

    best_cost = float("inf")
    best_tour = None

    for _ in range(RESTARTS):
        if time.time() - start_time > MAX_RUNTIME:
            break

        start_node = random.randint(0, n - 1)
        beam = [{"path": [start_node], "visited": 1 << start_node, "cost": 0.0}]

        for step in range(1, n):
            expansions = []

            for b in beam:
                last = b["path"][-1]
                visited = b["visited"]

                neighbors = [i for i in range(n) if not (visited & (1 << i))]
                neighbors.sort(key=lambda x: D[last][x])
                neighbors = neighbors[:TOPK]

                if not neighbors:
                    continue

                chosen = neighbors[0]

                if step % 3 == 1 and len(neighbors) >= 3:
                    marked = list(range(min(2, len(neighbors))))
                    idx_qubits = max(1, math.ceil(math.log2(len(neighbors))))
                    chosen = neighbors[
                        quantum.run_grover(marked, idx_qubits)
                    ]

                expansions.append({
                    "path": b["path"] + [chosen],
                    "visited": visited | (1 << chosen),
                    "cost": b["cost"] + D[last][chosen]
                })

            expansions.sort(key=lambda x: x["cost"])
            beam = expansions[:BEAM_SIZE]

        for b in beam:
            if len(b["path"]) == n:
                cost = close_tour_cost(b["path"], D)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = b["path"]

    total_time = time.time() - start_time
    qstats = quantum.stats()

    return {
        "tour": best_tour + [best_tour[0]] if best_tour else None,
        "tour_cost": best_cost,
        "runtime_sec": total_time,
        "quantum_runtime_sec": qstats["quantum_runtime_sec"],
        "quantum_calls": qstats["quantum_calls"]
    }

# =========================
# Main Runner (BATCH)
# =========================

def main():
    factors = [d for d in os.listdir(CASCADE_ROOT) if d.startswith("factor_")]

    for fname in os.listdir(GRAPH_DIR):
        if not fname.endswith(".tsp"):
            continue

        tsp = parse_tsp(os.path.join(GRAPH_DIR, fname))
        coords = tsp["coords"]

        # --- FULL graph ---
        edges, D_full = build_candidate_edges(coords)
        pg_full = {"num_nodes": len(coords), "distance_matrix": D_full, "edges": edges}
        out_dir = os.path.join(RESULTS_DIR, "full")
        os.makedirs(out_dir, exist_ok=True)
        result = quantum_beam_search(pg_full)
        result.update({"graph": fname, "n": len(coords)})
        with open(os.path.join(out_dir, fname.replace(".tsp", ".json")), "w") as f:
            json.dump(result, f, indent=2)

        # --- EdgeGNN-only ---
        pruned = os.path.join(EDGEGNN_ONLY, fname.replace(".tsp", ".json"))
        if os.path.exists(pruned):
            pg_edge = load_pruned_graph(pruned, coords)
            out_dir = os.path.join(RESULTS_DIR, "edgeGNN_only")
            os.makedirs(out_dir, exist_ok=True)
            result = quantum_beam_search(pg_edge)
            result.update({"graph": fname, "n": len(coords)})
            with open(os.path.join(out_dir, fname.replace(".tsp", ".json")), "w") as f:
                json.dump(result, f, indent=2)

        # --- EdgeGNN + cascade ---
        for factor in factors:
            edgelist = os.path.join(CASCADE_ROOT, factor, fname.replace(".tsp", ".edgelist"))
            if not os.path.exists(edgelist):
                continue

            edges = []
            D_cascade = np.full((len(coords), len(coords)), np.inf)
            np.fill_diagonal(D_cascade, 0)

            with open(edgelist) as f:
                for line in f:
                    u, v, w = line.split()
                    u, v, w = int(u), int(v), float(w)
                    D_cascade[u, v] = w
                    D_cascade[v, u] = w
                    edges.append((u, v))

            pg_cascade = {"num_nodes": len(coords), "distance_matrix": D_cascade, "edges": edges}
            out_dir = os.path.join(RESULTS_DIR, "edgeGNN_cascade", factor)
            os.makedirs(out_dir, exist_ok=True)
            result = quantum_beam_search(pg_cascade)
            result.update({"graph": fname, "n": len(coords)})
            with open(os.path.join(out_dir, fname.replace(".tsp", ".json")), "w") as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

