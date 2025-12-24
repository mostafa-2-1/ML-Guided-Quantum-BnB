
# # start hereee 
# #!/usr/bin/env python3
# """
# quantum_accelerators.py

# Updated: Quantum-Assisted Beam Search TSP hybrid pipeline.

# Features:
# - Beam-search integration to produce a FULL TSP tour (0 -> ... -> 0).
# - Grover-based min-finder over top-k candidate lists (Qiskit).
# - Optional Montanaro-style feasibility check (practical enumeration + Grover) for small subtrees.
# - Works on large pruned graphs: quantum calls restricted to small local subproblems.
# - CLI flags, JSON+Markdown outputs, timing and resource logging.



# """
# #beam search with quantum algorithm code to find the optimal path, grovers algorithm is always applied, montanaro is optional
# import argparse
# import json
# import math
# import os
# import sys
# import time
# from copy import deepcopy
# from typing import List, Tuple, Dict, Any
# from qiskit_aer import AerSimulator
# import networkx as nx
# import numpy as np
# import random
# # Qiskit imports 
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# from qiskit.circuit.library import UnitaryGate

# from qiskit.quantum_info import Operator
# # --------------------------
# # ---- Graph I/O / helpers
# # --------------------------
# def load_pruned_graph(path: str) -> Dict[str, Any]:
#     with open(path, 'r') as f:
#         data = json.load(f)
#     for key in ('num_nodes', 'edges', 'distance_matrix'):
#         if key not in data:
#             raise ValueError(f"Pruned graph JSON missing required key: {key}")
#     return data

# def save_json(obj: Any, path: str):
#     with open(path, 'w') as f:
#         json.dump(obj, f, indent=2)

# def save_text(text: str, path: str):
#     with open(path, 'w') as f:
#         f.write(text)

# def check_connectivity(n_nodes: int, edges: List[List[int]]) -> bool:
#     G = nx.Graph()
#     G.add_nodes_from(range(n_nodes))
#     G.add_edges_from([tuple(e) for e in edges])
#     return nx.is_connected(G)

# def ensure_connected(n_nodes:int, edges:List[List[int]], D:List[List[float]]) -> List[List[int]]:
#     G = nx.Graph()
#     G.add_nodes_from(range(n_nodes))
#     G.add_edges_from([tuple(e) for e in edges])
#     components = list(nx.connected_components(G))
#     if len(components) == 1:
#         return edges
#     new_edges = edges.copy()
#     # greedy connect nearest component pairs
#     while len(components) > 1:
#         best_pair = None
#         best_d = float('inf')
#         for i, compA in enumerate(components):
#             for compB in components[i+1:]:
#                 for a in compA:
#                     for b in compB:
#                         d = D[a][b]
#                         if d < best_d:
#                             best_d = d
#                             best_pair = (a,b)
#         new_edges.append([best_pair[0], best_pair[1]])
#         G.add_edge(*best_pair)
#         components = list(nx.connected_components(G))
#     return new_edges

# # -------------------------
# # Qiskit: Grover helpers (diagonal oracle)
# # -------------------------
# def build_phase_oracle_for_marked_indices(marked_indices: List[int], index_qubits: int) -> QuantumCircuit:
#     """
#     Build diagonal oracle unitary with -1 at marked_indices.
#     Practical for index_qubits <= 10 (matrix up to 1024x1024).
#     """
#     dim = 1 << max(1, index_qubits)
#     diag = np.ones(dim, dtype=complex)
#     for m in marked_indices:
#         if 0 <= m < dim:
#             diag[m] = -1.0
#     U = np.diag(diag)
#     gate = UnitaryGate(U, label="OracleDiag")
#     qc = QuantumCircuit(index_qubits, name='Oracle')
#     qc.append(gate, list(range(index_qubits)))
#     return qc

# def diffusion_operator(index_qubits: int) -> QuantumCircuit:
#     dim = 1 << max(1, index_qubits)
#     J = np.ones((dim, dim)) / dim
#     D = 2 * J - np.eye(dim)
#     gate = UnitaryGate(D, label="Diffusion")
#     qc = QuantumCircuit(index_qubits, name='Diffusion')
#     qc.append(gate, list(range(index_qubits)))
#     return qc

# def grover_search_marked_indices(marked_indices: List[int],
#                                  index_qubits: int,
#                                  repetitions: int,
#                                  backend=None,
#                                  shots: int = 512,
#                                  time_budget_remaining: float = None,
#                                  start_time: float = None) -> Dict[str,Any]:
#     """
#     Run Grover on a small index space; checks time budget before expensive steps.
#     Returns {'counts','timing','qc','compiled_qc'} or {'error':...}
#     """
#     if index_qubits < 1:
#         index_qubits = 1
#     qc = QuantumCircuit(index_qubits, index_qubits)
#     qc.h(range(index_qubits))
#     oracle_qc = build_phase_oracle_for_marked_indices(marked_indices, index_qubits)
#     diffusion_qc = diffusion_operator(index_qubits)
#     oracle_gate = oracle_qc.to_gate()
#     diffusion_gate = diffusion_qc.to_gate()
#     for _ in range(repetitions):
#         if time_budget_remaining is not None and start_time is not None:
#             if (time.time() - start_time) > time_budget_remaining:
#                 return {'error': 'time_budget_exceeded_before_iteration'}
#         qc.append(oracle_gate, list(range(index_qubits)))
#         qc.append(diffusion_gate, list(range(index_qubits)))
#     qc.measure(range(index_qubits), range(index_qubits))
#     backend = backend or AerSimulator(method='statevector')
#     t0 = time.time()
#     try:
#         compiled = transpile(qc, backend=backend)
#     except Exception as e:
#         return {'error': f'transpile_failed: {e}'}
#     t1 = time.time()
#     if time_budget_remaining is not None and start_time is not None:
#         if (time.time() - start_time) > time_budget_remaining:
#             return {'error': 'time_budget_exceeded_before_simulation', 'transpile_time_s': t1-t0}
#     try:
#         job = backend.run(compiled, shots=shots)
#         res = job.result()
#     except Exception as e:
#         return {'error': f'simulation_failed: {e}', 'transpile_time_s': t1-t0}
#     t2 = time.time()
#     counts = res.get_counts()
#     timing = {'transpile_time_s': t1 - t0, 'sim_time_s': t2 - t1, 'total_time_s': t2 - t0}
#     return {'counts': counts, 'timing': timing, 'qc': qc, 'compiled_qc': compiled}

# # -------------------------
# # small helpers: top-k, subtree extraction
# # -------------------------
# def topk_successors_from_pruned_graph(pruned_graph: Dict[str,Any], node:int, k:int) -> List[Tuple[int,float]]:
#     D = pruned_graph['distance_matrix']
#     neighbors = set()
#     for a,b in pruned_graph['edges']:
#         if a==node:
#             neighbors.add(b)
#         elif b==node:
#             neighbors.add(a)
#     neighs = [(nb, D[node][nb]) for nb in neighbors]
#     neighs.sort(key=lambda x: x[1])
#     return neighs[:k]

# def extract_local_subtree(pruned_graph: Dict[str,Any], root:int, max_nodes:int=12) -> Dict[str,Any]:
#     n = pruned_graph['num_nodes']
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     G.add_edges_from([tuple(e) for e in pruned_graph['edges']])
#     visited = [root]
#     queue = [root]
#     while queue and len(visited) < max_nodes:
#         cur = queue.pop(0)
#         for nb in G.neighbors(cur):
#             if nb not in visited:
#                 visited.append(nb)
#                 queue.append(nb)
#             if len(visited) >= max_nodes:
#                 break
#     nodes = visited[:max_nodes]
#     mapping = {nodes[i]: i for i in range(len(nodes))}
#     local_edges = []
#     for a,b in pruned_graph['edges']:
#         if a in mapping and b in mapping:
#             local_edges.append((mapping[a], mapping[b]))
#     D_full = pruned_graph['distance_matrix']
#     D = [[D_full[orig_a][orig_b] for orig_b in nodes] for orig_a in nodes]
#     return {'nodes': nodes, 'edges': local_edges, 'distance_matrix': D, 'orig_to_local': mapping, 'local_to_orig': nodes}

# # -------------------------
# # Montanaro practical enumeration (small subtree)
# # -------------------------
# def enumerate_partial_assignments(subtree: Dict[str,Any], max_length:int=None) -> List[Dict[str,Any]]:
#     nodes = list(range(len(subtree['nodes'])))
#     D = subtree['distance_matrix']
#     from itertools import permutations
#     s = len(nodes)
#     max_len = s if max_length is None else min(s, max_length)
#     assignments = []
#     for r in range(1, max_len+1):
#         for perm in permutations(nodes, r):
#             feasible = True
#             cost = 0.0
#             for i in range(1, len(perm)):
#                 a,b = perm[i-1], perm[i]
#                 if (a,b) not in subtree['edges'] and (b,a) not in subtree['edges']:
#                     feasible = False
#                     break
#                 cost += D[a][b]
#             if feasible:
#                 bitmask = 0
#                 for v in perm:
#                     bitmask |= (1<<v)
#                 assignments.append({'bitmask': bitmask, 'last': perm[-1], 'path': list(perm), 'cost': cost})
#     # deduplicate
#     best = {}
#     for a in assignments:
#         key = (a['bitmask'], a['last'])
#         if key not in best or a['cost'] < best[key]['cost']:
#             best[key] = a
#     return list(best.values())

# def montanaro_practical_feasibility(subtree: Dict[str,Any], cost_bound: float, backend=None, start_time:float=None, time_budget_remaining:float=None) -> Dict[str,Any]:
#     enumerated = enumerate_partial_assignments(subtree)
#     if len(enumerated) == 0:
#         return {'feasible': False, 'reason': 'no_enumerated'}
#     marked = [i for i,a in enumerate(enumerated) if a['cost'] <= cost_bound]
#     if len(marked) == 0:
#         return {'feasible': False, 'reason': 'no_marked', 'num_enumerated': len(enumerated)}
#     index_qubits = max(1, math.ceil(math.log2(len(enumerated))))
#     N = 1 << index_qubits
#     M = len(marked)
#     R = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))
#     res = grover_search_marked_indices(marked, index_qubits, R, backend=backend, shots=256, time_budget_remaining=time_budget_remaining, start_time=start_time)
#     if 'error' in res:
#         return {'feasible': False, 'reason': 'time_budget_exceeded_or_error', 'details': res}
#     counts = res['counts']
#     measured = max(counts.items(), key=lambda x: x[1])[0]
#     measured_int = int(measured, 2)
#     assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
#     return {'feasible': assgn is not None and assgn['cost'] <= cost_bound, 'measured_assignment': assgn, 'num_enumerated': len(enumerated), 'marked_count': len(marked), 'grover_counts': counts, 'timing': res['timing']}

# # -------------------------
# # 2-opt local improvement
# # -------------------------
# def two_opt_improve(path: List[int], D: List[List[float]], time_budget: float = None) -> Tuple[List[int], float]:
#     """
#     Fast 2-opt improvement (first improvement). Returns improved path and its cost.
#     time_budget: seconds allowed for the improvement; if None, runs until no improvement.
#     """
#     start = time.time()
#     n = len(path)
#     best = path[:]
#     best_cost = close_tour_cost(best, D)
#     improved = True
#     while improved:
#         improved = False
#         for i in range(1, n-2):
#             for j in range(i+1, n):
#                 # do 2-opt swap between i..j-1
#                 new_path = best[:i] + best[i:j][::-1] + best[j:]
#                 new_cost = close_tour_cost(new_path, D)
#                 if new_cost < best_cost:
#                     best = new_path
#                     best_cost = new_cost
#                     improved = True
#                     break  # first improvement
#             if improved:
#                 break
#             if time_budget is not None and (time.time() - start) > time_budget:
#                 return best, best_cost
#     return best, best_cost

# # -------------------------
# # Path cost helpers
# # -------------------------
# def path_cost(path: List[int], Dmat: List[List[float]]) -> float:
#     c = 0.0
#     for i in range(1, len(path)):
#         c += Dmat[path[i-1]][path[i]]
#     return c

# def close_tour_cost(path: List[int], Dmat: List[List[float]]) -> float:
#     c = path_cost(path, Dmat)
#     c += Dmat[path[-1]][path[0]]
#     return c

# def average_branching_factor(pruned_graph: Dict[str,Any]) -> float:
#     n = pruned_graph['num_nodes']
#     degs = [0]*n
#     for a,b in pruned_graph['edges']:
#         degs[a] += 1
#         degs[b] += 1
#     nz = [d for d in degs if d>0]
#     return float(sum(nz))/len(nz) if len(nz)>0 else 0.0

# # -------------------------
# # Beam-search with new features
# # -------------------------
# def beam_search_quantum(pruned_graph: Dict[str,Any],
#                         beam_size:int,
#                         topk:int,
#                         mode:str,
#                         restarts:int,
#                         use_montanaro:bool,
#                         force_montanaro:bool,
#                         n_max_quantum_user:int,
#                         max_runtime:float,
#                         backend=None,
#                         k_cap:int = 64) -> Dict[str,Any]:
#     """
#     Runs beam-search with quantum-assisted candidate selection, 2-opt refinement, and restarts.
#     Returns a result dict with best tour found and diagnostics.
#     """
#     backend = backend or AerSimulator(method='statevector')
#     n = pruned_graph['num_nodes']
#     D = pruned_graph['distance_matrix']

#     # Mode automatic tuning
#     if mode == 'speed':
#         beam_size = beam_size if beam_size is not None else 3
#         topk = min(topk, 8)
#         enable_2opt = False
#         restarts = 1
#     elif mode == 'balanced':
#         beam_size = beam_size if beam_size is not None else 4
#         topk = min(topk, 12)
#         enable_2opt = True
#         restarts = restarts if restarts is not None else 1
#     else:  # accuracy
#         beam_size = beam_size if beam_size is not None else 8
#         topk = min(topk, 24)
#         enable_2opt = True
#         restarts = restarts if restarts is not None else 2

#     # dynamic n_max_quantum if not provided
#     if n_max_quantum_user is not None:
#         n_max_quantum = n_max_quantum_user
#     else:
#         if n <= 10:
#             n_max_quantum = 10
#         elif n <= 20:
#             n_max_quantum = 6
#         elif n <= 50:
#             n_max_quantum = 4
#         else:
#             n_max_quantum = 0

#     # decide montanaro allowed (auto-guard)
#     branch = average_branching_factor(pruned_graph)
#     montanaro_allowed = use_montanaro and (n_max_quantum >= 3)
#     if not force_montanaro:
#         if n > 20 or branch > 4.0:
#             montanaro_allowed = False

#     # time budget
#     start_time = time.time()
#     deadline = start_time + max_runtime
#     def time_left(): return max(0.0, deadline - time.time())

#     # safety caps
#     ENUMERATION_SAFE_LIMIT = 3000
#     BATCH_GROVER_MAX = 6  # number of candidate sets to batch (keeps mem ok)

#     best_global = {'path': None, 'cost': float('inf')}
#     best_runs = []

#     # deterministic seed for reproducibility unless multiple restarts
#     for restart_idx in range(restarts):
#         if time.time() > deadline:
#             break
#         # small random perturbation seed: permute order of neighbor ties
#         seed = int(time.time() * 1000) % (2**31 - 1)
#         random.seed(seed + restart_idx)

#         # initial beam
#         beam = [{'path':[0], 'visited': (1<<0), 'cost': 0.0}]
#         # ensure beam_min so it doesn't collapse to 1 too early
#         beam_min = max(2, min(beam_size, 3))

#         run_quantum_time = 0.0
#         quantum_calls = 0
#         logs = {'steps': []}

#         for step in range(1, n):
#             if time.time() > deadline:
#                 break
#             # progress print periodically
#             if step % 5 == 0:
#                 print(f"[RUN {restart_idx+1}/{restarts}] step {step}/{n-1}, beam={len(beam)}, time_left={time_left():.1f}s, montanaro_allowed={montanaro_allowed}")
#             expansions = []
#             step_record = {'step': step, 'in_beam': deepcopy(beam), 'expansions': []}

#             # We'll optionally batch up candidate sets to reduce repeated transpile overhead.
#             # Collect up to BATCH_GROVER_MAX candidate sets from beam then run them (if beneficial).
#             batches = []
#             for item in beam[:BATCH_GROVER_MAX]:
#                 last = item['path'][-1]
#                 visited = item['visited']
#                 neighs = topk_successors_from_pruned_graph(pruned_graph, last, topk)
#                 # filter visited
#                 candidates = [t[0] for t in neighs if not (visited & (1<<t[0]))]
#                 candidates = candidates[:k_cap]  # hard cap
#                 batches.append({'item': item, 'candidates': candidates, 'dists': [pruned_graph['distance_matrix'][last][c] for c in candidates]})

#             # Handle cases where beam is larger than batch cap: process remaining items individually later
#             remaining_items = beam[BATCH_GROVER_MAX:] if len(beam) > BATCH_GROVER_MAX else []

#             # Process batched candidate sets
#             for batch in batches:
#                 item = batch['item']
#                 candidates = batch['candidates']
#                 dists = batch['dists']
#                 if len(candidates) == 0:
#                     # fallback: choose nearest unvisited classically
#                     rem = [i for i in range(n) if not (item['visited'] & (1<<i))]
#                     if not rem: continue
#                     chosen = min(rem, key=lambda x: pruned_graph['distance_matrix'][item['path'][-1]][x])
#                     new_path = item['path'] + [chosen]
#                     new_cost = item['cost'] + pruned_graph['distance_matrix'][item['path'][-1]][chosen]
#                     expansions.append({'path': new_path, 'visited': item['visited'] | (1<<chosen), 'cost': new_cost})
#                     step_record['expansions'].append({'from': item['path'], 'chosen': chosen, 'grover_info': {'fallback': True}})
#                     continue

#                 # classical best index used to mark (simulate predicate)
#                 min_idx = int(np.argmin(dists))
#                 marked = [min_idx]
#                 index_qubits = max(1, math.ceil(math.log2(len(candidates))))
#                 N = 1 << index_qubits
#                 M = max(1, len(marked))
#                 repetitions = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))

#                 # time budget allocation
#                 per_call_budget = max(1.0, time_left() * 0.1)  # give 10% remaining to this call roughly
#                 grover_res = grover_search_marked_indices(marked, index_qubits, repetitions, backend=backend, shots=256, time_budget_remaining=per_call_budget, start_time=start_time)
#                 if 'error' in grover_res:
#                     chosen_idx = min_idx
#                     grover_info = {'fallback': True, 'reason': grover_res.get('error')}
#                 else:
#                     quantum_calls += 1
#                     run_quantum_time += grover_res['timing']['total_time_s']
#                     counts = grover_res['counts']
#                     measured = max(counts.items(), key=lambda x: x[1])[0]
#                     chosen_idx = int(measured, 2)
#                     grover_info = {'counts': counts, 'timing': grover_res['timing']}
#                 if chosen_idx >= len(candidates):
#                     chosen_idx = min_idx
#                 chosen = candidates[chosen_idx]
#                 new_path = item['path'] + [chosen]
#                 new_cost = item['cost'] + pruned_graph['distance_matrix'][item['path'][-1]][chosen]
#                 new_visited = item['visited'] | (1<<chosen)
#                 expansion = {'from': item['path'], 'chosen': chosen, 'new_path': new_path, 'new_cost': new_cost, 'grover_info': grover_info}
#                 # optional Montanaro feasibility (very guarded)
#                 if montanaro_allowed:
#                     subtree = extract_local_subtree(pruned_graph, chosen, max_nodes=n_max_quantum)
#                     s = len(subtree['nodes'])
#                     if not force_montanaro and s > 9:
#                         expansion['montanaro'] = {'skipped': True, 'reason': 'subtree_too_large_for_auto'}
#                     else:
#                         # cost bound heuristic
#                         remaining = n - len(new_path)
#                         avg_edge = 0.0
#                         if len(subtree['edges'])>0:
#                             vals = [subtree['distance_matrix'][a][b] for a,b in subtree['edges']]
#                             avg_edge = sum(vals)/len(vals)
#                         estimated_extra = remaining * max(1.0, avg_edge)
#                         cost_bound = new_cost + estimated_extra
#                         enumerated = enumerate_partial_assignments(subtree)
#                         if len(enumerated) > ENUMERATION_SAFE_LIMIT and not force_montanaro:
#                             expansion['montanaro'] = {'skipped': True, 'reason': 'enumeration_too_large'}
#                         else:
#                             ma_budget = min(max(1.0, time_left()*0.05), 20.0)
#                             ma_res = montanaro_practical_feasibility(subtree, cost_bound, backend=backend, start_time=start_time, time_budget_remaining=ma_budget)
#                             expansion['montanaro'] = {'result': ma_res}
#                             if 'timing' in ma_res:
#                                 run_quantum_time += ma_res['timing'].get('total_time_s', 0.0)
#                                 quantum_calls += 1
#                 expansions.append({'path': new_path, 'visited': new_visited, 'cost': new_cost})
#                 step_record['expansions'].append(expansion)

#             # Process remaining beam items individually (similar to above)
#             for item in remaining_items:
#                 if time.time() > deadline: break
#                 last = item['path'][-1]
#                 visited = item['visited']
#                 neighs = topk_successors_from_pruned_graph(pruned_graph, last, topk)
#                 candidates = [t[0] for t in neighs if not (visited & (1<<t[0]))]
#                 candidates = candidates[:k_cap]
#                 if len(candidates) == 0:
#                     rem = [i for i in range(n) if not (visited & (1<<i))]
#                     if not rem: continue
#                     chosen = min(rem, key=lambda x: pruned_graph['distance_matrix'][last][x])
#                     new_path = item['path'] + [chosen]
#                     new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
#                     expansions.append({'path': new_path, 'visited': item['visited'] | (1<<chosen), 'cost': new_cost})
#                     step_record['expansions'].append({'from': item['path'], 'chosen': chosen, 'grover_info': {'fallback': True}})
#                     continue
#                 min_idx = 0
#                 dists = [pruned_graph['distance_matrix'][last][c] for c in candidates]
#                 min_idx = int(np.argmin(dists))
#                 marked = [min_idx]
#                 index_qubits = max(1, math.ceil(math.log2(len(candidates))))
#                 N = 1 << index_qubits
#                 M = max(1, len(marked))
#                 repetitions = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))
#                 per_call_budget = max(1.0, time_left() * 0.1)
#                 grover_res = grover_search_marked_indices(marked, index_qubits, repetitions, backend=backend, shots=256, time_budget_remaining=per_call_budget, start_time=start_time)
#                 if 'error' in grover_res:
#                     chosen_idx = min_idx
#                     grover_info = {'fallback': True, 'reason': grover_res.get('error')}
#                 else:
#                     quantum_calls += 1
#                     run_quantum_time += grover_res['timing']['total_time_s']
#                     counts = grover_res['counts']
#                     measured = max(counts.items(), key=lambda x: x[1])[0]
#                     chosen_idx = int(measured, 2)
#                     grover_info = {'counts': counts, 'timing': grover_res['timing']}
#                 if chosen_idx >= len(candidates):
#                     chosen_idx = min_idx
#                 chosen = candidates[chosen_idx]
#                 new_path = item['path'] + [chosen]
#                 new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
#                 new_visited = item['visited'] | (1<<chosen)
#                 expansion = {'from': item['path'], 'chosen': chosen, 'new_path': new_path, 'new_cost': new_cost, 'grover_info': grover_info}
#                 expansions.append({'path': new_path, 'visited': new_visited, 'cost': new_cost})
#                 step_record['expansions'].append(expansion)
#                 if time.time() > deadline:
#                     break

#             expansions.sort(key=lambda x: x['cost'])
#             # beam diversification: keep at least beam_min items, and apply small perturbation occasionally
#             beam = expansions[:beam_size] if len(expansions) >= beam_min else expansions[:beam_min]
#             # perturbation: with small prob, replace worst beam item with slight mutation
#             if random.random() < 0.05 and len(beam) >= 2 and mode != 'speed':
#                 idx = random.randrange(1, len(beam))
#                 item = beam[idx]
#                 # mutate by swapping last two cities if possible
#                 if len(item['path']) >= 3:
#                     p = item['path'][:]
#                     p[-2], p[-1] = p[-1], p[-2]
#                     beam[idx] = {'path': p, 'visited': item['visited'], 'cost': path_cost(p, D)}
#             logs['steps'].append(step_record)
#             if time.time() > deadline:
#                 break
#             # early exit if beam contains full-length tours
#             if any(len(b['path']) == n for b in beam):
#                 break

#         # build full tours from beam (greedily fill)
#         full_tours = []
#         for b in beam:
#             cur = b['path'][:]
#             visited = b['visited']
#             while len(cur) < n:
#                 last = cur[-1]
#                 remaining = [i for i in range(n) if not (visited & (1<<i))]
#                 if not remaining:
#                     break
#                 nextc = min(remaining, key=lambda x: pruned_graph['distance_matrix'][last][x])
#                 cur.append(nextc)
#                 visited |= (1<<nextc)
#             cost = close_tour_cost(cur, D)
#             full_tours.append({'path': cur, 'cost': cost})
#         # pick best for this restart
#         best = min(full_tours, key=lambda x: x['cost']) if full_tours else None
#         # apply 2-opt if enabled and time remains
#         if enable_2opt and best is not None and time_left() > 2.0:
#             opt_budget = min(10.0, time_left() * 0.25)
#             improved_path, improved_cost = two_opt_improve(best['path'], D, time_budget=opt_budget)
#             best = {'path': improved_path, 'cost': improved_cost}
#         best_runs.append({'restart': restart_idx, 'best': best, 'quantum_time': run_quantum_time, 'quantum_calls': quantum_calls, 'logs': logs})
#         if best is not None and best['cost'] < best_global['cost']:
#             best_global = best
#         # restart time check
#         if time.time() > deadline:
#             break

#     result = {
#         'best_tour': best_global,
#         'runs': best_runs,
#         'mode': mode,
#         'beam': beam_size,
#         'topk': topk,
#         'restarts': restarts,
#         'n_max_quantum': n_max_quantum,
#         'montanaro_allowed': montanaro_allowed,
#         'wall_time_s': time.time() - start_time
#     }
#     return result

# # -------------------------
# # CLI / main
# # -------------------------
# def default_output_paths(input_json_path: str):
#     base = os.path.splitext(os.path.basename(input_json_path))[0]
#     out_dir = "quantum_run_outputs"
#     os.makedirs(out_dir, exist_ok=True)
#     ts = int(time.time())
#     return {'json': os.path.join(out_dir, f"{base}_beamrun_{ts}.json"), 'md': os.path.join(out_dir, f"{base}_beamrun_{ts}.md")}

# def parse_opt_tour(opt_path: str) -> List[int]:
#     """Parse a TSPLIB .opt.tour file and return 0-based node list"""
#     with open(opt_path, 'r') as f:
#         lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
#     # find TOUR_SECTION
#     if 'TOUR_SECTION' in lines:
#         idx = lines.index('TOUR_SECTION')
#         seq = []
#         for l in lines[idx+1:]:
#             if l == '-1' or l.upper() == 'EOF':
#                 break
#             try:
#                 seq.append(int(l)-1)  # convert to 0-based
#             except:
#                 continue
#         return seq
#     # fallback: parse all integers ignoring headers
#     seq = []
#     for l in lines:
#         try:
#             seq.append(int(l)-1)
#         except:
#             continue
#     return [x for x in seq if x >= 0]

# def pretty_md_report(result: Dict[str,Any], input_info: Dict[str,Any], opt_info: Dict[str,Any]=None) -> str:
#     md = []
#     md.append("# Quantum-Assisted Beam Search Report")
#     md.append("")
#     md.append("## Input")
#     for k,v in input_info.items():
#         md.append(f"- **{k}**: {v}")
#     md.append("")
#     md.append("## Summary")
#     best = result.get('best_tour')
#     if best and best['path'] is not None:
#         md.append(f"- Best tour cost: **{best['cost']}**")
#         md.append(f"- Path (length {len(best['path'])}): {best['path']}")
#     else:
#         md.append("- No tour found.")
#     md.append(f"- Wall time (s): {result.get('wall_time_s'):.2f}")
#     md.append(f"- Mode: {result.get('mode')}")
#     md.append(f"- Montanaro allowed: {result.get('montanaro_allowed')}")
#     if opt_info is not None:
#         md.append("")
#         md.append("## Optimality comparison")
#         md.append(f"- Provided opt tour length: {opt_info.get('opt_cost')}")
#         md.append(f"- Gap: {opt_info.get('gap_percent'):.2f}%")
#     return "\n".join(md)

# def main_cli():
#     parser = argparse.ArgumentParser(description="Quantum-Assisted Beam Search TSP (with 2-opt, beam diversification, restarts).")
#     parser.add_argument('pruned_graph', help='Path to pruned graph JSON')
#     parser.add_argument('--mode', choices=['speed','balanced','accuracy'], default='balanced', help='Operation mode (speed/balanced/accuracy)')
#     parser.add_argument('--beam', type=int, default=None, help='Beam size (mode-dependent default)')
#     parser.add_argument('--k', type=int, default=16, help='Top-k neighbors considered initially')
#     parser.add_argument('--restarts', type=int, default=1, help='Number of restarts (multi-start)')
#     parser.add_argument('--use_montanaro', action='store_true', help='Attempt to use Montanaro practical feasibility checks (auto-guarded)')
#     parser.add_argument('--force_montanaro', action='store_true', help='Force Montanaro even if autosafety would disable it (dangerous)')
#     parser.add_argument('--n_max_quantum', type=int, default=None, help='Max nodes allowed for Montanaro enumeration (auto-chosen if omitted)')
#     parser.add_argument('--max_runtime', type=int, default=300, help='Global runtime budget in seconds (default 300)')
#     parser.add_argument('--k_cap', type=int, default=64, help='Hard cap on candidate pool for Grover (default 64)')
#     parser.add_argument('--opt_tour_path', type=str, default=None, help='Optional .opt.tour file to compute gap vs provided optimum')
#     args = parser.parse_args()

#     pruned = load_pruned_graph(args.pruned_graph)
#     n = pruned['num_nodes']
#     edges = [list(e) for e in pruned['edges']]
#     D = pruned['distance_matrix']

#     # ensure connectivity
#     if not check_connectivity(n, edges):
#         print("[INFO] Graph not connected - repairing...")
#         edges = ensure_connected(n, edges, D)
#         pruned['edges'] = edges
#         print("[INFO] Connectivity repaired.")

#     backend = AerSimulator(method='statevector')

#     print(f"[START] Running quantum-assisted beam search on {args.pruned_graph} (n={n}), mode={args.mode}, beam={args.beam}, k={args.k}, restarts={args.restarts}")
#     t0 = time.time()
#     result = beam_search_quantum(pruned, beam_size=args.beam, topk=args.k, mode=args.mode, restarts=args.restarts, use_montanaro=args.use_montanaro, force_montanaro=args.force_montanaro, n_max_quantum_user=args.n_max_quantum, max_runtime=args.max_runtime, backend=backend, k_cap=args.k_cap)
#     t1 = time.time()
#     result['wall_time_total_s'] = t1 - t0

#     opt_info = None
#     if args.opt_tour_path:
#         try:
#             opt_path = parse_opt_tour(args.opt_tour_path)
#             opt_cost = close_tour_cost(opt_path, D)
#             best = result.get('best_tour')
#             if best and best['path'] is not None:
#                 gap = 100.0 * (best['cost'] - opt_cost) / opt_cost
#             else:
#                 gap = None
#             opt_info = {'opt_cost': opt_cost, 'gap_percent': gap, 'opt_path': opt_path}
#         except Exception as e:
#             print("[WARN] Failed to parse opt tour:", e)

#     out_paths = default_output_paths(args.pruned_graph)
#     out_json = {'input': {'path': args.pruned_graph, 'n': n, 'mode': args.mode, 'beam': args.beam, 'k': args.k, 'restarts': args.restarts}, 'result': result, 'opt_info': opt_info}
#     save_json(out_json, out_paths['json'])
#     md = pretty_md_report(result, out_json['input'], opt_info)
#     save_text(md, out_paths['md'])
#     print(f"[DONE] Results saved: {out_paths['json']}, {out_paths['md']}")
#     if result.get('best_tour') and result['best_tour']['path'] is not None:
#         print(f" Best tour cost: {result['best_tour']['cost']:.3f}, path len: {len(result['best_tour']['path'])}")
#     if opt_info is not None and opt_info.get('gap_percent') is not None:
#         print(f" Gap vs opt: {opt_info['gap_percent']:.2f}% (opt={opt_info['opt_cost']})")
#     print(f" Wall time total: {result['wall_time_total_s']:.2f}s")
#     return 0

# if __name__ == "__main__":
#     try:
#         sys.exit(main_cli())
#     except KeyboardInterrupt:
#         print("\n[ABORT] User interrupted.")
#         sys.exit(1)





# #!/usr/bin/env python3
# """
# quantum_accelerators_fixed.py

# Production-level refactor of quantum-assisted beam search:
# - native circuit Grover oracle (no dense unitaries)
# - Held-Karp DP for Montanaro enumeration
# - sampling vs statevector backend split and correct routing
# - beam diversity, multiple-candidate expansion, independent restarts
# - time budgets and safe enumeration caps
# """

# import argparse
# import json
# import math
# import os
# import sys
# import time
# import logging
# from copy import deepcopy
# from typing import List, Tuple, Dict, Any, Optional

# import networkx as nx
# import numpy as np
# import random

# from qiskit_aer import AerSimulator
# from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# from qiskit.circuit.library import MCXGate

# # ---------------------------
# # Logging configuration
# # ---------------------------
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger("quantum_beam")



# def compute_grover_reps(index_qubits: int, marked_count: int, n_max_quantum: int = None) -> int:
#     """
#     Robust Grover repetition estimator.
#     - index_qubits: number of index qubits (N = 2**index_qubits)
#     - marked_count: M (>=1)
#     - n_max_quantum: optional user cap for repetitions (or max nodes)
#     """
#     N = 1 << max(1, index_qubits)
#     M = max(1, int(marked_count))
#     # Classic Grover estimate
#     R = int(round((math.pi / 4.0) * math.sqrt(N / float(M))))
#     # lower bound to ensure some amplification (empirical)
#     lb = max(1, int(max(1, math.sqrt(N) / 3.0)))
#     R = max(R, lb)
#     # optional upper bound: don't let R explode; if user provided n_max_quantum, use it as cap
#     if n_max_quantum is not None and isinstance(n_max_quantum, (int, float)):
#         try:
#             cap = max(1, int(n_max_quantum))
#             R = min(R, cap)
#         except Exception:
#             pass
#     return R



# # ---------------------------
# # I/O & helpers
# # ---------------------------
# def load_pruned_graph(path: str) -> Dict[str, Any]:
#     with open(path, 'r') as f:
#         data = json.load(f)
#     for key in ('num_nodes', 'edges', 'distance_matrix'):
#         if key not in data:
#             raise ValueError(f"Pruned graph JSON missing required key: {key}")
#     return data

# def save_json(obj: Any, path: str):
#     with open(path, 'w') as f:
#         json.dump(obj, f, indent=2)

# def save_text(text: str, path: str):
#     with open(path, 'w') as f:
#         f.write(text)

# def check_connectivity(n_nodes: int, edges: List[List[int]]) -> bool:
#     G = nx.Graph()
#     G.add_nodes_from(range(n_nodes))
#     G.add_edges_from([tuple(e) for e in edges])
#     return nx.is_connected(G)

# def ensure_connected(n_nodes:int, edges:List[List[int]], D:List[List[float]]) -> List[List[int]]:
#     G = nx.Graph()
#     G.add_nodes_from(range(n_nodes))
#     G.add_edges_from([tuple(e) for e in edges])
#     components = list(nx.connected_components(G))
#     if len(components) == 1:
#         return edges
#     new_edges = edges.copy()
#     # greedy connect nearest component pairs
#     while len(components) > 1:
#         best_pair = None
#         best_d = float('inf')
#         for i, compA in enumerate(components):
#             for compB in components[i+1:]:
#                 for a in compA:
#                     for b in compB:
#                         d = D[a][b]
#                         if d < best_d:
#                             best_d = d
#                             best_pair = (a,b)
#         new_edges.append([best_pair[0], best_pair[1]])
#         G.add_edge(*best_pair)
#         components = list(nx.connected_components(G))
#     return new_edges

# # ---------------------------
# # Grover helpers (native circuits, no dense unitaries)
# # ---------------------------
# def _apply_multi_controlled_z(qc: QuantumCircuit, qubits: List[int]):
#     """
#     Apply a multi-controlled Z on the given qubits using MCX trick:
#     H on target -> MCX(controls, target) -> H on target
#     qubits: list indices in the qc (controls ... target)
#     """
#     if len(qubits) == 0:
#         return
#     if len(qubits) == 1:
#         qc.z(qubits[0])
#         return
#     target = qubits[-1]
#     controls = qubits[:-1]
#     # H on target to convert X-target to Z-target
#     qc.h(target)
#     if len(controls) == 1:
#         qc.cx(controls[0], target)
#     else:
#         mcx = MCXGate(len(controls))
#         qc.append(mcx, controls + [target])
#     qc.h(target)

# def build_phase_oracle_marked_indices(marked_indices: List[int], index_qubits: int) -> QuantumCircuit:
#     """
#     Build oracle that flips the phase for basis states enumerated by marked_indices.
#     Avoid large dense unitaries by constructing controlled-Z patterns
#     that flip for each marked bitstring.
#     """
#     qr = QuantumRegister(index_qubits, 'idx')
#     qc = QuantumCircuit(qr, name='oracle')
#     # For each marked basis state, flip qubits where bit is 0, apply multi-controlled Z, then undo flips.
#     for idx in marked_indices:
#         bitstr = format(idx, '0{}b'.format(index_qubits))[::-1]  # LSB-first mapping
#         zero_positions = [i for i, b in enumerate(bitstr) if b == '0']
#         for p in zero_positions:
#             qc.x(qr[p])
#         _apply_multi_controlled_z(qc, [qr[i] for i in range(index_qubits)])
#         for p in zero_positions:
#             qc.x(qr[p])
#     return qc

# def build_grover_diffusion(index_qubits: int) -> QuantumCircuit:
#     """Standard diffusion using H, X, multi-controlled Z, X, H."""
#     qr = QuantumRegister(index_qubits, 'idx')
#     qc = QuantumCircuit(qr, name='diffusion')
#     qc.h(qr)
#     qc.x(qr)
#     _apply_multi_controlled_z(qc, [qr[i] for i in range(index_qubits)])
#     qc.x(qr)
#     qc.h(qr)
#     return qc

# def grover_search_marked_indices_circuit(marked_indices: List[int],
#                                          index_qubits: int,
#                                          repetitions: int,
#                                          backend,
#                                          shots: int = 512,
#                                          start_time: float = None,
#                                          time_budget_remaining: float = None) -> Dict[str,Any]:
#     """
#     Build and run Grover using circuit-based oracles.

#     Behavior depends on backend:
#     - If backend.options.get('method') == 'statevector': run without measurement and return 'statevector'
#     - Else: build with measurement and return 'counts' (sampling)

#     Returns dict with either 'counts' or 'statevector', plus 'timing' and circuit objects.
#     """
#     if index_qubits < 1:
#         index_qubits = 1
#     # Time guard
#     if time_budget_remaining is not None and start_time is not None:
#         if (time.time() - start_time) > time_budget_remaining:
#             return {'error': 'time_budget_exceeded_before_build'}
#     qr = QuantumRegister(index_qubits, 'idx')
#     cr = ClassicalRegister(index_qubits, 'c')
#     qc = QuantumCircuit(qr, cr)
#     qc.h(qr)
#     oracle = build_phase_oracle_marked_indices(marked_indices, index_qubits)
#     diffusion = build_grover_diffusion(index_qubits)

#     # Append repetitions
#     for _ in range(repetitions):
#         qc.compose(oracle, qubits=qr, inplace=True)
#         qc.compose(diffusion, qubits=qr, inplace=True)
#         if time_budget_remaining is not None and start_time is not None:
#             if (time.time() - start_time) > time_budget_remaining:
#                 return {'error': 'time_budget_exceeded_during_build'}

#     # Decide whether to measure or return statevector based on backend
#     method = None
#     try:
#         method = getattr(backend.options, 'method', None) or backend._configuration.method
#     except Exception:
#         # backend might be a remote backend or have different structure; default to sampling
#         method = None

#     if method == 'statevector' or (hasattr(backend, 'name') and 'statevector' in str(method)):
#         # For statevector, build a circuit *without measurement* and run
#         qc_nomeas = qc.remove_final_measurements(inplace=False) if hasattr(qc, 'remove_final_measurements') else qc.copy().reverse_bits()
#         # NOTE: some versions of qiskit have different APIs; simpler approach: build qc2 without measurements
#         qc2 = QuantumCircuit(qr)
#         qc2.h(qr)
#         qc2.compose(oracle, qubits=qr, inplace=True)
#         for _ in range(repetitions - 1):
#             qc2.compose(diffusion, qubits=qr, inplace=True)
#             qc2.compose(oracle, qubits=qr, inplace=True)
#             qc2.compose(diffusion, qubits=qr, inplace=True)
#         # Attempt to transpile and run
#         t0 = time.time()
#         try:
#             compiled = transpile(qc2, backend=backend)
#         except Exception as e:
#             return {'error': f'transpile_failed: {e}'}
#         t1 = time.time()
#         if time_budget_remaining is not None and start_time is not None:
#             if (time.time() - start_time) > time_budget_remaining:
#                 return {'error': 'time_budget_exceeded_before_sim', 'transpile_time_s': t1-t0}
#         try:
#             job = backend.run(compiled)
#             res = job.result()
#         except Exception as e:
#             return {'error': f'simulation_failed: {e}', 'transpile_time_s': t1-t0}
#         t2 = time.time()
#         # extract statevector if available
#         statevector = None
#         try:
#             # result.get_statevector(compiled) works in many Aer versions
#             statevector = res.get_statevector(compiled)
#         except Exception:
#             # fallback: some result objects expose data differently
#             try:
#                 statevector = res.data(0).get('statevector', None)
#             except Exception:
#                 statevector = None
#         timing = {'transpile_time_s': t1-t0, 'sim_time_s': t2-t1, 'total_time_s': t2-t0}
#         if statevector is None:
#             # If statevector couldn't be retrieved, return counts by measuring instead as fallback
#             qc_meas = qc.copy()
#             qc_meas.measure(qr, cr)
#             t0b = time.time()
#             try:
#                 compiled2 = transpile(qc_meas, backend=backend)
#                 job2 = backend.run(compiled2, shots=shots)
#                 res2 = job2.result()
#                 counts = res2.get_counts()
#                 t2b = time.time()
#                 timing['fallback_counts_time_s'] = t2b - t0b
#                 timing['total_time_s'] += timing['fallback_counts_time_s']
#                 return {'counts': counts, 'timing': timing, 'qc': qc_meas, 'compiled_qc': compiled2}
#             except Exception as e:
#                 return {'error': f'statevector_unavailable_and_counts_failed: {e}', 'timing': timing}
#         return {'statevector': statevector, 'timing': timing, 'qc': qc2, 'compiled_qc': compiled}

#     else:
#         # Sampling mode: append measurement and run with shots
#         qc.measure(qr, cr)
#         t0 = time.time()
#         try:
#             compiled = transpile(qc, backend=backend)
#         except Exception as e:
#             return {'error': f'transpile_failed: {e}'}
#         t1 = time.time()
#         if time_budget_remaining is not None and start_time is not None:
#             if (time.time() - start_time) > time_budget_remaining:
#                 return {'error': 'time_budget_exceeded_before_sim', 'transpile_time_s': t1-t0}
#         try:
#             job = backend.run(compiled, shots=shots)
#             res = job.result()
#         except Exception as e:
#             return {'error': f'simulation_failed: {e}', 'transpile_time_s': t1-t0}
#         t2 = time.time()
#         counts = res.get_counts()
#         timing = {'transpile_time_s': t1-t0, 'sim_time_s': t2-t1, 'total_time_s': t2-t0}
#         return {'counts': counts, 'timing': timing, 'qc': qc, 'compiled_qc': compiled}

# # ---------------------------
# # Held-Karp DP enumeration (efficient)
# # ---------------------------
# def held_karp_enumeration(subtree: Dict[str,Any], max_mask_size: Optional[int] = None) -> List[Dict[str,Any]]:
#     """
#     DP to enumerate best (mask,last) assignments up to max_mask_size (or all nodes).
#     Returns list of {'bitmask','last','path','cost'}.
#     """
#     nodes = list(range(len(subtree['nodes'])))
#     n = len(nodes)
#     D = subtree['distance_matrix']
#     max_mask_size = n if max_mask_size is None else min(n, max_mask_size)

#     dp: Dict[Tuple[int,int], Tuple[float,int]] = {}
#     for i in range(n):
#         mask = 1 << i
#         dp[(mask, i)] = (0.0, -1)

#     # iterate increasing mask sizes
#     for size in range(2, max_mask_size + 1):
#         # generate masks with exact popcount=size
#         masks = [m for m in range(1, 1<<n) if bin(m).count("1") == size]
#         for mask in masks:
#             for j in range(n):
#                 if not (mask & (1<<j)):
#                     continue
#                 prev_mask = mask ^ (1<<j)
#                 best_cost = float('inf')
#                 best_prev = -1
#                 # iterate possible previous nodes
#                 for k in range(n):
#                     if k == j: continue
#                     if not (prev_mask & (1<<k)): continue
#                     if (prev_mask, k) not in dp: continue
#                     prev_cost = dp[(prev_mask, k)][0]
#                     cost = prev_cost + D[k][j]
#                     if cost < best_cost:
#                         best_cost = cost
#                         best_prev = k
#                 if best_prev >= 0:
#                     dp[(mask, j)] = (best_cost, best_prev)

#     # reconstruct list
#     assignments = []
#     for (mask, last), (cost, prev) in dp.items():
#         # reconstruct path
#         path = []
#         cur_mask = mask
#         cur_last = last
#         while cur_last != -1:
#             path.append(cur_last)
#             entry = dp.get((cur_mask, cur_last), None)
#             if entry is None:
#                 break
#             _, p = entry
#             cur_mask ^= (1<<cur_last)
#             cur_last = p
#         path = path[::-1]
#         assignments.append({'bitmask': mask, 'last': last, 'path': path, 'cost': cost})
#     # deduplicate best (mask,last)
#     best_map: Dict[Tuple[int,int], Dict[str,Any]] = {}
#     for a in assignments:
#         key = (a['bitmask'], a['last'])
#         if key not in best_map or a['cost'] < best_map[key]['cost']:
#             best_map[key] = a
#     return list(best_map.values())

# def montanaro_practical_feasibility_improved(subtree: Dict[str,Any],
#                                              cost_bound: float,
#                                              backend,
#                                              start_time: float = None,
#                                              time_budget_remaining: float = None,
#                                              enumeration_safe_limit: int = 10000) -> Dict[str,Any]:
#     """
#     Use Held-Karp DP to enumerate assignments and call Grover on those indices.
#     backend should be a **statevector** backend for amplitude-level info (preferred).
#     If backend is sampling-only, grover_search will fall back to counts.
#     """
#     t0 = time.time()
#     enumerated = held_karp_enumeration(subtree)
#     t1 = time.time()
#     if len(enumerated) == 0:
#         return {'feasible': False, 'reason': 'no_enumerated'}
#     marked = [i for i,a in enumerate(enumerated) if a['cost'] <= cost_bound]
#     if len(marked) == 0:
#         return {'feasible': False, 'reason': 'no_marked', 'num_enumerated': len(enumerated)}
#     if len(enumerated) > enumeration_safe_limit and (time_budget_remaining is None or time_budget_remaining < 1.0):
#         return {'feasible': False, 'reason': 'enumeration_too_large', 'num_enumerated': len(enumerated)}
#     index_qubits = max(1, math.ceil(math.log2(len(enumerated))))
#     N = 1 << index_qubits
#     M = len(marked)
#     R = max(1, int(math.floor((math.pi/4.0) * math.sqrt(N / M))))
#     grover_res = grover_search_marked_indices_circuit(marked, index_qubits, R, backend=backend, shots=256, start_time=start_time, time_budget_remaining=time_budget_remaining)
#     if 'error' in grover_res:
#         return {'feasible': False, 'reason': 'time_budget_exceeded_or_error', 'details': grover_res}
#     # If statevector result available, choose highest-probability basis states by amplitude magnitude squared
#     if 'statevector' in grover_res:
#         sv = grover_res['statevector']
#         probs = np.abs(np.array(sv))**2
#         measured_int = int(np.argmax(probs))
#         assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
#         return {'feasible': assgn is not None and assgn['cost'] <= cost_bound,
#                 'measured_assignment': assgn,
#                 'num_enumerated': len(enumerated),
#                 'marked_count': len(marked),
#                 'state_probs_top': float(np.max(probs)),
#                 'timing': grover_res['timing']}
#     else:
#         counts = grover_res.get('counts', {})
#         if not counts:
#             return {'feasible': False, 'reason': 'no_counts', 'details': grover_res}
#         measured = max(counts.items(), key=lambda x: x[1])[0]
#         measured_int = int(measured, 2)
#         assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
#         return {'feasible': assgn is not None and assgn['cost'] <= cost_bound,
#                 'measured_assignment': assgn,
#                 'num_enumerated': len(enumerated),
#                 'marked_count': len(marked),
#                 'grover_counts': counts,
#                 'timing': grover_res['timing']}

# # ---------------------------
# # Utility helpers (costs, subtree extraction)
# # ---------------------------
# def path_cost(path: List[int], Dmat: List[List[float]]) -> float:
#     c = 0.0
#     for i in range(1, len(path)):
#         c += Dmat[path[i-1]][path[i]]
#     return c

# def close_tour_cost(path: List[int], Dmat: List[List[float]]) -> float:
#     c = path_cost(path, Dmat)
#     c += Dmat[path[-1]][path[0]]
#     return c

# def average_branching_factor(pruned_graph: Dict[str,Any]) -> float:
#     n = pruned_graph['num_nodes']
#     degs = [0]*n
#     for a,b in pruned_graph['edges']:
#         degs[a] += 1
#         degs[b] += 1
#     nz = [d for d in degs if d>0]
#     return float(sum(nz))/len(nz) if len(nz)>0 else 0.0

# def topk_successors_from_pruned_graph(pruned_graph: Dict[str,Any], node:int, k:int) -> List[Tuple[int,float]]:
#     D = pruned_graph['distance_matrix']
#     neighbors = set()
#     for a,b in pruned_graph['edges']:
#         if a==node:
#             neighbors.add(b)
#         elif b==node:
#             neighbors.add(a)
#     neighs = [(nb, D[node][nb]) for nb in neighbors]
#     neighs.sort(key=lambda x: x[1])
#     return neighs[:k]

# def extract_local_subtree(pruned_graph: Dict[str,Any], root:int, max_nodes:int=12) -> Dict[str,Any]:
#     n = pruned_graph['num_nodes']
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     G.add_edges_from([tuple(e) for e in pruned_graph['edges']])
#     visited = [root]
#     queue = [root]
#     while queue and len(visited) < max_nodes:
#         cur = queue.pop(0)
#         for nb in G.neighbors(cur):
#             if nb not in visited:
#                 visited.append(nb)
#                 queue.append(nb)
#             if len(visited) >= max_nodes:
#                 break
#     nodes = visited[:max_nodes]
#     mapping = {nodes[i]: i for i in range(len(nodes))}
#     local_edges = []
#     for a,b in pruned_graph['edges']:
#         if a in mapping and b in mapping:
#             local_edges.append((mapping[a], mapping[b]))
#     D_full = pruned_graph['distance_matrix']
#     D = [[D_full[orig_a][orig_b] for orig_b in nodes] for orig_a in nodes]
#     return {'nodes': nodes, 'edges': local_edges, 'distance_matrix': D, 'orig_to_local': mapping, 'local_to_orig': nodes}

# # ---------------------------
# # 2-opt local improvement
# # ---------------------------
# def two_opt_improve(path: List[int], D: List[List[float]], time_budget: float = None) -> Tuple[List[int], float]:
#     """
#     First-improvement 2-opt (fast). Time-budget guards.
#     """
#     start = time.time()
#     n = len(path)
#     best = path[:]
#     best_cost = close_tour_cost(best, D)
#     improved = True
#     while improved:
#         improved = False
#         for i in range(1, n-2):
#             for j in range(i+1, n):
#                 new_path = best[:i] + best[i:j][::-1] + best[j:]
#                 new_cost = close_tour_cost(new_path, D)
#                 if new_cost < best_cost:
#                     best = new_path
#                     best_cost = new_cost
#                     improved = True
#                     break
#             if improved:
#                 break
#             if time_budget is not None and (time.time() - start) > time_budget:
#                 return best, best_cost
#     return best, best_cost

# # ---------------------------
# # Beam search (main) - accepts sampling and state backends
# # ---------------------------
# def beam_search_quantum_refactored(pruned_graph: Dict[str,Any],
#                                    beam_size:int,
#                                    topk:int,
#                                    mode:str,
#                                    restarts:int,
#                                    use_montanaro:bool,
#                                    force_montanaro:bool,
#                                    n_max_quantum_user:int,
#                                    max_runtime:float,
#                                    sampling_backend,
#                                    state_backend,
#                                    k_cap:int = 64) -> Dict[str,Any]:
#     """
#     Main beam search algorithm. sampling_backend used for measurement-based Grover operations.
#     state_backend used for amplitude-level Montanaro feasibility checks.
#     """
#     if sampling_backend is None:
#         sampling_backend = AerSimulator()
#     if state_backend is None:
#         try:
#             state_backend = AerSimulator(method='statevector')
#         except Exception:
#             logger.warning("statevector backend unavailable; mounting statevector calls will fall back to sampling.")
#             state_backend = sampling_backend

#     n = pruned_graph['num_nodes']
#     D = pruned_graph['distance_matrix']

#     # Mode tuning
#     if mode == 'speed':
#         beam_size = beam_size if beam_size is not None else 3
#         topk = min(topk, 8)
#         enable_2opt = False
#         restarts = 1
#     elif mode == 'balanced':
#         beam_size = beam_size if beam_size is not None else 4
#         topk = min(topk, 12)
#         enable_2opt = True
#         restarts = restarts if restarts is not None else 1
#     else:  # accuracy
#         beam_size = beam_size if beam_size is not None else 8
#         topk = min(topk, 24)
#         enable_2opt = True
#         restarts = restarts if restarts is not None else 2

#     if n_max_quantum_user is not None:
#         n_max_quantum = n_max_quantum_user
#     else:
#         if n <= 10:
#             n_max_quantum = 10
#         elif n <= 20:
#             n_max_quantum = 6
#         elif n <= 50:
#             n_max_quantum = 4
#         else:
#             n_max_quantum = 0

#     branch = average_branching_factor(pruned_graph)
#     montanaro_allowed = use_montanaro and (n_max_quantum >= 3)
#     if not force_montanaro:
#         if n > 20 or branch > 4.0:
#             montanaro_allowed = False

#     start_time = time.time()
#     deadline = start_time + max_runtime
#     def time_left(): return max(0.0, deadline - time.time())

#     best_global = {'path': None, 'cost': float('inf')}
#     best_runs: List[Dict[str,Any]] = []
#     base_seed = int(time.time()) & 0x7fffffff

#     ENUMERATION_SAFE_LIMIT = 10000
#     BATCH_GROVER_MAX = 6

#     for restart_idx in range(restarts):
#         if time.time() > deadline:
#             break
#         rng = random.Random(base_seed + restart_idx)
#         np_rng = np.random.RandomState(base_seed + restart_idx)

#         # Initial beam - deterministic start at 0 (you can randomize starts via option)
#         beam = [{'path':[0], 'visited': (1<<0), 'cost': 0.0}]
#         beam_min = max(2, min(beam_size, 3))
#         run_quantum_time = 0.0
#         quantum_calls = 0
#         logs = {'steps': []}

#         for step in range(1, n):
#             if time.time() > deadline:
#                 break
#             if step % 5 == 0:
#                 logger.info(f"[RUN {restart_idx+1}/{restarts}] step {step}/{n-1}, beam={len(beam)}, time_left={time_left():.1f}s")
#             expansions = []
#             step_record = {'step': step, 'in_beam': deepcopy(beam), 'expansions': []}

#             # batch first part of beam
#             batches = []
#             for item in beam[:BATCH_GROVER_MAX]:
#                 last = item['path'][-1]
#                 visited = item['visited']
#                 neighs = topk_successors_from_pruned_graph(pruned_graph, last, topk)
#                 candidates = [t[0] for t in neighs if not (visited & (1<<t[0]))][:k_cap]
#                 dists = [pruned_graph['distance_matrix'][last][c] for c in candidates]
#                 batches.append({'item': item, 'candidates': candidates, 'dists': dists})

#             remaining_items = beam[BATCH_GROVER_MAX:] if len(beam) > BATCH_GROVER_MAX else []

#             # Process the batches
#             for batch in batches:
#                 item = batch['item']
#                 candidates = batch['candidates']
#                 dists = batch['dists']
#                 if len(candidates) == 0:
#                     rem = [i for i in range(n) if not (item['visited'] & (1<<i))]
#                     if not rem: continue
#                     chosen = min(rem, key=lambda x: pruned_graph['distance_matrix'][item['path'][-1]][x])
#                     new_path = item['path'] + [chosen]
#                     new_cost = item['cost'] + pruned_graph['distance_matrix'][item['path'][-1]][chosen]
#                     expansions.append({'path': new_path, 'visited': item['visited'] | (1<<chosen), 'cost': new_cost})
#                     step_record['expansions'].append({'from': item['path'], 'chosen': chosen, 'grover_info': {'fallback': True}})
#                     continue

#                 # pick marked set with slack then cap it
#                 min_d = min(dists)
#                 slack = 1.10
#                 marked = [i for i,d in enumerate(dists) if d <= min_d * slack]
#                 K_mark = max(1, min(4, len(dists)))
#                 if len(marked) > K_mark:
#                     ords = sorted(range(len(dists)), key=lambda i: dists[i])
#                     marked = ords[:K_mark]

#                 index_qubits = max(1, math.ceil(math.log2(max(1, len(candidates)))))
#                 N = 1 << index_qubits
#                 M = max(1, len(marked))
#                 repetitions = compute_grover_reps(index_qubits=index_qubits, marked_count=M, n_max_quantum=n_max_quantum)


#                 per_call_budget = max(0.5, time_left() * 0.05)
#                 grover_res = grover_search_marked_indices_circuit(marked, index_qubits, repetitions, backend=sampling_backend, shots=256, start_time=start_time, time_budget_remaining=per_call_budget)
                
#                 # robust grover result handling
#                 chosen_idx = None
#                 grover_info = {}
#                 chosen_indices = []
#                 if 'error' in grover_res:
#                     grover_info = {'fallback': True, 'reason': grover_res.get('error')}
#                 else:
#                     quantum_calls += 1
#                     # safe access to timing (some flows may not include timing)
#                     run_quantum_time += float(grover_res.get('timing', {}).get('total_time_s', 0.0))
#                     # prefer sampling counts if present
#                     counts = grover_res.get('counts', None)
#                     if counts:
#                         # pick top measured outcome(s)
#                         freq = sorted(counts.items(), key=lambda x: x[1], reverse=True)
#                         try:
#                             measured_int = int(freq[0][0], 2)
#                         except Exception:
#                             measured_int = None
#                         if measured_int is None or measured_int < 0 or measured_int >= (1 << index_qubits):
#                             grover_info = {'fallback': True, 'reason': 'invalid_measured_index'}
#                         else:
#                             # map measured index (which indexes the index space) to candidate index
#                             # If you produced 'marked' indices earlier (list of indexes into candidates),
#                             # ensure mapping exists; otherwise fallback to classical best
#                             if 'marked' in locals() and len(marked) > 0:
#                                 if measured_int < len(marked):
#                                     chosen_idx = marked[measured_int]
#                                     grover_info = {'counts': counts, 'timing': grover_res.get('timing')}
#                                 else:
#                                     grover_info = {'fallback': True, 'reason': 'measured_out_of_marked_range'}
#                             else:
#                                 grover_info = {'fallback': True, 'reason': 'no_marked_map'}
#                     else:
#                         # No counts => maybe statevector or unusual result; try statevector route
#                         if 'statevector' in grover_res:
#                             sv = grover_res['statevector']
#                             probs = np.abs(np.array(sv))**2
#                             measured_int = int(np.argmax(probs))
#                             if 'marked' in locals() and len(marked) > 0:
#                                 if measured_int < len(marked):
#                                     chosen_idx = marked[measured_int]
#                                     grover_info = {'state_probs_top': float(np.max(probs)), 'timing': grover_res.get('timing')}
#                                 else:
#                                     grover_info = {'fallback': True, 'reason': 'state_measured_out_of_range'}
#                             else:
#                                 grover_info = {'fallback': True, 'reason': 'no_marked_map_statevec'}
#                         else:
#                             grover_info = {'fallback': True, 'reason': 'no_counts_no_statevector'}

#                 # final fallback if no chosen index determined
#                 if chosen_idx is None:
#                     # deterministic classical fallback: choose nearest (min_idx) or best by dists
#                     chosen_idx = min(range(len(candidates)), key=lambda x: dists[x])
#                     chosen_indices = [marked.index(chosen_idx) if 'marked' in locals() and chosen_idx in marked else 0]
#                     grover_info = grover_info if grover_info else {'fallback': True, 'reason': 'no_choice'}
#                 else:
#                     # final fallback if somehow chosen_idx is None
#                     chosen_idx = min(range(len(candidates)), key=lambda x: dists[x])
#                     if 'marked' in locals() and chosen_idx in marked:
#                         chosen_indices = [marked.index(chosen_idx)]
#                     else:
#                         chosen_indices = [0]
#                     grover_info = grover_info if grover_info else {'fallback': True, 'reason': 'no_choice'}
#                 # expand multiple measured candidates for robustness
#                 for mi in chosen_indices:
#                     candidate_idx = marked[mi] if (0 <= mi < len(marked)) else marked[0]
#                     chosen = candidates[candidate_idx]
#                     new_path = item['path'] + [chosen]
#                     new_cost = item['cost'] + pruned_graph['distance_matrix'][item['path'][-1]][chosen]
#                     new_visited = item['visited'] | (1<<chosen)
#                     expansion = {'from': item['path'], 'chosen': chosen, 'new_path': new_path, 'new_cost': new_cost, 'grover_info': grover_info}

#                     # optional Montanaro feasibility (use state_backend)
#                     if montanaro_allowed:
#                         subtree = extract_local_subtree(pruned_graph, chosen, max_nodes=n_max_quantum)
#                         s = len(subtree['nodes'])
#                         if not force_montanaro and s > 10:
#                             expansion['montanaro'] = {'skipped': True, 'reason': 'subtree_too_large_for_auto'}
#                         else:
#                             remaining = n - len(new_path)
#                             avg_edge = 0.0
#                             if len(subtree['edges']) > 0:
#                                 vals = [subtree['distance_matrix'][a][b] for a,b in subtree['edges']]
#                                 avg_edge = sum(vals)/len(vals)
#                             estimated_extra = remaining * max(1.0, avg_edge)
#                             cost_bound = new_cost + estimated_extra
#                             if time_left() < 0.5:
#                                 expansion['montanaro'] = {'skipped': True, 'reason': 'time_low'}
#                             else:
#                                 ma_budget = min(max(0.5, time_left()*0.05), 8.0)
#                                 ma_res = montanaro_practical_feasibility_improved(subtree, cost_bound, backend=state_backend, start_time=start_time, time_budget_remaining=ma_budget, enumeration_safe_limit=ENUMERATION_SAFE_LIMIT)
#                                 expansion['montanaro'] = {'result': ma_res}
#                                 if 'timing' in ma_res:
#                                     run_quantum_time += ma_res['timing'].get('total_time_s', 0.0) if isinstance(ma_res.get('timing'), dict) else 0.0
#                                     quantum_calls += 1
#                     expansions.append({'path': new_path, 'visited': new_visited, 'cost': new_cost})
#                     step_record['expansions'].append(expansion)

#             # Remaining items processed similarly but single candidate expansion to control cost
#             for item in remaining_items:
#                 if time.time() > deadline: break
#                 last = item['path'][-1]
#                 visited = item['visited']
#                 neighs = topk_successors_from_pruned_graph(pruned_graph, last, topk)
#                 candidates = [t[0] for t in neighs if not (visited & (1<<t[0]))][:k_cap]
#                 dists = [pruned_graph['distance_matrix'][last][c] for c in candidates]
#                 if len(candidates) == 0:
#                     rem = [i for i in range(n) if not (visited & (1<<i))]
#                     if not rem: continue
#                     chosen = min(rem, key=lambda x: pruned_graph['distance_matrix'][last][x])
#                     new_path = item['path'] + [chosen]
#                     new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
#                     expansions.append({'path': new_path, 'visited': item['visited'] | (1<<chosen), 'cost': new_cost})
#                     step_record['expansions'].append({'from': item['path'], 'chosen': chosen, 'grover_info': {'fallback': True}})
#                     continue
#                 min_idx = int(np.argmin(dists))
#                 min_d = dists[min_idx]
#                 slack = 1.10
#                 marked = [i for i,d in enumerate(dists) if d <= min_d * slack]
#                 if len(marked) > 3:
#                     ords = sorted(range(len(dists)), key=lambda i: dists[i])
#                     marked = ords[:3]
#                 index_qubits = max(1, math.ceil(math.log2(max(1, len(candidates)))))
#                 N = 1 << index_qubits
#                 M = max(1, len(marked))
#                 repetitions = compute_grover_reps(index_qubits=index_qubits, marked_count=M, n_max_quantum=n_max_quantum)

#                 per_call_budget = max(0.5, time_left() * 0.05)
#                 grover_res = grover_search_marked_indices_circuit(marked, index_qubits, repetitions, backend=sampling_backend, shots=256, start_time=start_time, time_budget_remaining=per_call_budget)
                

#                 # robust grover result handling
#                 chosen_idx = None
#                 grover_info = {}
#                 if 'error' in grover_res:
#                     grover_info = {'fallback': True, 'reason': grover_res.get('error')}
#                 else:
#                     quantum_calls += 1
#                     # safe access to timing (some flows may not include timing)
#                     run_quantum_time += float(grover_res.get('timing', {}).get('total_time_s', 0.0))
#                     # prefer sampling counts if present
#                     counts = grover_res.get('counts', None)
#                     if counts:
#                         # pick top measured outcome(s)
#                         freq = sorted(counts.items(), key=lambda x: x[1], reverse=True)
#                         try:
#                             measured_int = int(freq[0][0], 2)
#                         except Exception:
#                             measured_int = None
#                         if measured_int is None or measured_int < 0 or measured_int >= (1 << index_qubits):
#                             grover_info = {'fallback': True, 'reason': 'invalid_measured_index'}
#                         else:
#                             # map measured index (which indexes the index space) to candidate index
#                             # If you produced 'marked' indices earlier (list of indexes into candidates),
#                             # ensure mapping exists; otherwise fallback to classical best
#                             if 'marked' in locals() and len(marked) > 0:
#                                 if measured_int < len(marked):
#                                     chosen_idx = marked[measured_int]
#                                     grover_info = {'counts': counts, 'timing': grover_res.get('timing')}
#                                 else:
#                                     grover_info = {'fallback': True, 'reason': 'measured_out_of_marked_range'}
#                             else:
#                                 grover_info = {'fallback': True, 'reason': 'no_marked_map'}
#                     else:
#                         # No counts => maybe statevector or unusual result; try statevector route
#                         if 'statevector' in grover_res:
#                             sv = grover_res['statevector']
#                             probs = np.abs(np.array(sv))**2
#                             measured_int = int(np.argmax(probs))
#                             if 'marked' in locals() and len(marked) > 0:
#                                 if measured_int < len(marked):
#                                     chosen_idx = marked[measured_int]
#                                     grover_info = {'state_probs_top': float(np.max(probs)), 'timing': grover_res.get('timing')}
#                                 else:
#                                     grover_info = {'fallback': True, 'reason': 'state_measured_out_of_range'}
#                             else:
#                                 grover_info = {'fallback': True, 'reason': 'no_marked_map_statevec'}
#                         else:
#                             grover_info = {'fallback': True, 'reason': 'no_counts_no_statevector'}

#                 # final fallback if no chosen index determined
#                 if chosen_idx is None:
#                     # deterministic classical fallback: choose nearest (min_idx) or best by dists
#                     chosen_idx = min(range(len(candidates)), key=lambda x: dists[x])
#                     grover_info = grover_info if grover_info else {'fallback': True, 'reason': 'no_choice'}




#                 chosen = candidates[chosen_idx]
#                 new_path = item['path'] + [chosen]
#                 new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
#                 new_visited = item['visited'] | (1<<chosen)
#                 expansions.append({'path': new_path, 'visited': new_visited, 'cost': new_cost})
#                 step_record['expansions'].append({'from': item['path'], 'chosen': chosen, 'grover_info': grover_info})
#                 if time.time() > deadline:
#                     break

#             # prune & diversify beam
#             expansions.sort(key=lambda x: x['cost'])
#             beam_new = expansions[:min(len(expansions), beam_size)]
#             if len(beam_new) < beam_size:
#                 # Build classical-ranked fallback list (unique by path)
#                 seen_paths = set()
#                 for b in beam_new:
#                     seen_paths.add(tuple(b['path']))
#                 # candidates from expansions past the top window, then greedy classical successors if needed
#                 backfill_candidates = []
#                 # first try next best expansions
#                 for b in expansions:
#                     t = tuple(b['path'])
#                     if t in seen_paths:
#                         continue
#                     backfill_candidates.append(b)
#                     seen_paths.add(t)
#                     if len(backfill_candidates) + len(beam_new) >= beam_size:
#                         break


#                 # if still short, generate classical greedy successors from current beam items (deterministic)
#                 if len(beam_new) + len(backfill_candidates) < beam_size:
#                     for item in beam:  # original beam items
#                         last = item['path'][-1]
#                         rem = [i for i in range(n) if not (item['visited'] & (1<<i))]
#                         rem_sorted = sorted(rem, key=lambda x: pruned_graph['distance_matrix'][last][x])
#                         for cand in rem_sorted:
#                             new_path = item['path'] + [cand]
#                             new_cost = item['cost'] + pruned_graph['distance_matrix'][last][cand]
#                             entry = {'path': new_path, 'visited': item['visited'] | (1<<cand), 'cost': new_cost}
#                             t = tuple(entry['path'])
#                             if t in seen_paths:
#                                 continue
#                             backfill_candidates.append(entry)
#                             seen_paths.add(t)
#                             if len(beam_new) + len(backfill_candidates) >= beam_size:
#                                 break
#                         if len(beam_new) + len(backfill_candidates) >= beam_size:
#                             break
#                 # append backfills to reach exactly beam_size
#                 need = beam_size - len(beam_new)
#                 beam_new.extend(backfill_candidates[:need])
            
#             if len(beam_new) < beam_min:
#                 # try to pad with best expansions (even duplicates), then random mutations
#                 idx = 0
#                 while len(beam_new) < beam_min and idx < len(expansions):
#                     if tuple(expansions[idx]['path']) not in {tuple(b['path']) for b in beam_new}:
#                         beam_new.append(expansions[idx])
#                     idx += 1
#                 # if still short, duplicate best
#                 while len(beam_new) < beam_min and beam_new:
#                     beam_new.append(deepcopy(beam_new[-1]))

#             beam = beam_new    

#             if rng.random() < 0.05 and len(beam) >= 2 and mode != 'speed':
#                 idx = rng.randrange(1, len(beam))
#                 item = beam[idx]
#                 if len(item['path']) >= 3:
#                     p = item['path'][:]
#                     p[-2], p[-1] = p[-1], p[-2]
#                     beam[idx] = {'path': p, 'visited': item['visited'], 'cost': path_cost(p, D)}
#             logs['steps'].append(step_record)
#             if time.time() > deadline:
#                 break
#             if any(len(b['path']) == n for b in beam):
#                 break

#         # Complete tours greedily and pick best
#         full_tours = []
#         for b in beam:
#             cur = b['path'][:]
#             visited = b['visited']
#             # greedily fill
#             while len(cur) < n:
#                 last = cur[-1]
#                 remaining = [i for i in range(n) if not (visited & (1<<i))]
#                 if not remaining: break
#                 nextc = min(remaining, key=lambda x: pruned_graph['distance_matrix'][last][x])
#                 cur.append(nextc)
#                 visited |= (1<<nextc)
#             # Only accept if cur contains all nodes
#             if len(cur) == n:
#                 # compute closed tour cost (includes return to start)
#                 cost = close_tour_cost(cur, D)
#                 full_tours.append({'path': cur, 'cost': cost})
#         # If no full tours, skip best selection for this restart
#         best = min(full_tours, key=lambda x: x['cost']) if full_tours else None

#         # apply 2-opt if best exists and is a full tour
#         if enable_2opt and best is not None and time_left() > 2.0:
#             opt_budget = min(10.0, time_left() * 0.25)
#             improved_path, improved_cost = two_opt_improve(best['path'], D, time_budget=opt_budget)
#             best = {'path': improved_path, 'cost': improved_cost}

#         if best is not None:
#             if best['cost'] < best_global.get('cost', float('inf')):
#                 best_global = best
#         if time.time() > deadline:
#             break

#     result = {
#         'best_tour': best_global,
#         'runs': best_runs,
#         'mode': mode,
#         'beam': beam_size,
#         'topk': topk,
#         'restarts': restarts,
#         'n_max_quantum': n_max_quantum,
#         'montanaro_allowed': montanaro_allowed,
#         'wall_time_s': time.time() - start_time
#     }
#     return result

# # ---------------------------
# # CLI + reporting
# # ---------------------------
# def default_output_paths(input_json_path: str):
#     base = os.path.splitext(os.path.basename(input_json_path))[0]
#     out_dir = "quantum_run_outputs"
#     os.makedirs(out_dir, exist_ok=True)
#     ts = int(time.time())
#     return {'json': os.path.join(out_dir, f"{base}_beamrun_{ts}.json"), 'md': os.path.join(out_dir, f"{base}_beamrun_{ts}.md")}

# def parse_opt_tour(opt_path: str) -> List[int]:
#     with open(opt_path, 'r') as f:
#         lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
#     if 'TOUR_SECTION' in lines:
#         idx = lines.index('TOUR_SECTION')
#         seq = []
#         for l in lines[idx+1:]:
#             if l == '-1' or l.upper() == 'EOF':
#                 break
#             try:
#                 seq.append(int(l)-1)
#             except:
#                 continue
#         return seq
#     seq = []
#     for l in lines:
#         try:
#             seq.append(int(l)-1)
#         except:
#             continue
#     return [x for x in seq if x >= 0]

# def pretty_md_report(result: Dict[str,Any], input_info: Dict[str,Any], opt_info: Dict[str,Any]=None) -> str:
#     md = []
#     md.append("# Quantum-Assisted Beam Search Report (Fixed)")
#     md.append("")
#     md.append("## Input")
#     for k,v in input_info.items():
#         md.append(f"- **{k}**: {v}")
#     md.append("")
#     md.append("## Summary")
#     best = result.get('best_tour')
#     if best and best.get('path'):
#         md.append(f"- Best tour cost: **{best['cost']}**")
#         md.append(f"- Path (length {len(best['path'])}): {best['path']}")
#     else:
#         md.append("- No tour found.")
#     md.append(f"- Wall time (s): {result.get('wall_time_s'):.2f}")
#     md.append(f"- Mode: {result.get('mode')}")
#     md.append(f"- Montanaro allowed: {result.get('montanaro_allowed')}")
#     if opt_info is not None:
#         md.append("")
#         md.append("## Optimality comparison")
#         md.append(f"- Provided opt tour length: {opt_info.get('opt_cost')}")
#         md.append(f"- Gap: {opt_info.get('gap_percent'):.2f}%")
#     return "\n".join(md)

# def main_cli():
#     parser = argparse.ArgumentParser(description="Quantum-Assisted Beam Search TSP (fixed backends).")
#     parser.add_argument('pruned_graph', help='Path to pruned graph JSON')
#     parser.add_argument('--mode', choices=['speed','balanced','accuracy'], default='balanced')
#     parser.add_argument('--beam', type=int, default=None)
#     parser.add_argument('--k', type=int, default=16)
#     parser.add_argument('--restarts', type=int, default=1)
#     parser.add_argument('--use_montanaro', action='store_true')
#     parser.add_argument('--force_montanaro', action='store_true')
#     parser.add_argument('--n_max_quantum', type=int, default=None)
#     parser.add_argument('--max_runtime', type=int, default=300)
#     parser.add_argument('--k_cap', type=int, default=64)
#     parser.add_argument('--opt_tour_path', type=str, default=None)
#     args = parser.parse_args()

#     pruned = load_pruned_graph(args.pruned_graph)
#     n = pruned['num_nodes']
#     edges = [list(e) for e in pruned['edges']]
#     D = pruned['distance_matrix']

#     if not check_connectivity(n, edges):
#         logger.info("[INFO] Graph not connected - repairing...")
#         edges = ensure_connected(n, edges, D)
#         pruned['edges'] = edges
#         logger.info("[INFO] Connectivity repaired.")

#     # create sampling backend and statebackend
#     sampling_backend = AerSimulator()
#     try:
#         state_backend = AerSimulator(method='statevector')
#     except Exception:
#         # fallback: some Aer versions may not support 'method' attribute
#         logger.warning("Warning: statevector backend not available; statevector calls will fall back to sampling backend.")
#         state_backend = sampling_backend

#     logger.info(f"[START] Running fixed search on {args.pruned_graph} (n={n}), mode={args.mode}, beam={args.beam}, k={args.k}, restarts={args.restarts}")
#     t0 = time.time()
#     result = beam_search_quantum_refactored(pruned, beam_size=args.beam, topk=args.k, mode=args.mode, restarts=args.restarts,
#                                            use_montanaro=args.use_montanaro, force_montanaro=args.force_montanaro,
#                                            n_max_quantum_user=args.n_max_quantum, max_runtime=args.max_runtime,
#                                            sampling_backend=sampling_backend, state_backend=state_backend, k_cap=args.k_cap)
#     t1 = time.time()
#     result['wall_time_total_s'] = t1 - t0

#     opt_info = None
#     if args.opt_tour_path:
#         try:
#             opt_path = parse_opt_tour(args.opt_tour_path)
#             opt_cost = close_tour_cost(opt_path, D)
#             best = result.get('best_tour')
#             if best and best.get('path') and len(best['path']) == n:
#                 gap = 100.0 * (best['cost'] - opt_cost) / opt_cost
#             else:
#                 gap = None
#             opt_info = {'opt_cost': opt_cost, 'gap_percent': gap, 'opt_path': opt_path}
#         except Exception as e:
#             logger.warning("[WARN] Failed to parse opt tour: %s", e)

#     out_paths = default_output_paths(args.pruned_graph)
#     out_json = {'input': {'path': args.pruned_graph, 'n': n, 'mode': args.mode, 'beam': args.beam, 'k': args.k, 'restarts': args.restarts}, 'result': result, 'opt_info': opt_info}
#     save_json(out_json, out_paths['json'])
#     md = pretty_md_report(result, out_json['input'], opt_info)
#     save_text(md, out_paths['md'])
#     logger.info(f"[DONE] Results saved: {out_paths['json']}, {out_paths['md']}")
#     if result.get('best_tour') and result['best_tour']['path'] is not None:
#         logger.info(" Best tour cost: %.3f, path len: %d", result['best_tour']['cost'], len(result['best_tour']['path']))
#     if opt_info is not None and opt_info.get('gap_percent') is not None:
#         logger.info(" Gap vs opt: %.2f%% (opt=%s)", opt_info['gap_percent'], opt_info['opt_cost'])
#     logger.info(" Wall time total: %.2fs", result['wall_time_total_s'])
#     return 0

# if __name__ == "__main__":
#     try:
#         sys.exit(main_cli())
#     except KeyboardInterrupt:
#         logger.info("User interrupted.")
#         sys.exit(1)










































































































































































#####
"""
quantum_accelerators_fixed.py

Production-level refactor of quantum-assisted beam search:
- Fixed index mapping bug in Grover results
- Robust backend handling with explicit types
- Deduplicated code with shared functions
- Seed management for reproducibility
- Research metrics collection

Author: Your Name
Date: 2024
Research: Quantum-Accelerated Beam Search for TSP
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import math
import os
import sys
import time
import logging
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------
# Logging configuration - FIXED VERSION
# ---------------------------
# Configure root logger FIRST
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    force=True
)

# Get the root logger
root_logger = logging.getLogger()

# Now suppress Qiskit modules WITHOUT affecting the root logger
qiskit_loggers = ['qiskit', 'qiskit.transpiler', 'qiskit.compiler', 'qiskit.execute']
for log_name in qiskit_loggers:
    logging.getLogger(log_name).setLevel(logging.WARNING)

# Create your application logger
logger = logging.getLogger("quantum_beam")
logger.propagate = True  # Let messages propagate to root logger (which is set to INFO)

# Now continue with your imports...
import networkx as nx
import numpy as np
import random

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCXGate



# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger("quantum_beam")

# # Suppress logs from Qiskit's transpiler and compiler modules
# logging.getLogger('qiskit.transpiler').setLevel(logging.WARNING)
# logging.getLogger('qiskit.compiler').setLevel(logging.WARNING)

# # Ensure your main logger is set to INFO (optional, for your own logs)
# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
# logger = logging.getLogger("quantum_beam")
# ---------------------------
# Configuration and Data Classes
# ---------------------------

@dataclass
class BackendConfig:
    """Configuration for quantum backends."""
    sampling_backend: Any
    state_backend: Any
    shots: int = 512
    optimization_level: int = 1

@dataclass
class SearchConfig:
    """Configuration for beam search."""
    beam_size: int
    topk: int
    mode: str
    restarts: int
    use_montanaro: bool
    force_montanaro: bool
    n_max_quantum: Optional[int]
    max_runtime: float
    k_cap: int = 64
    randomize_start: bool = False
    enable_2opt: bool = True
    seed: Optional[int] = None

@dataclass
class SearchMetrics:
    """Metrics for research analysis."""
    wall_time_s: float = 0.0
    quantum_time_s: float = 0.0
    quantum_calls: int = 0
    classical_expansions: int = 0
    grover_successes: int = 0
    grover_failures: int = 0
    montanaro_checks: int = 0
    circuit_stats: Dict[str, Any] = field(default_factory=dict)
    beam_history: List[Dict] = field(default_factory=list)
    
    @property
    def grover_success_rate(self) -> float:
        total = self.grover_successes + self.grover_failures
        return self.grover_successes / total if total > 0 else 0.0

class RandomManager:
    """Manages random seeds for reproducibility."""
    
    def __init__(self, seed: Optional[int] = None):
        self.base_seed = seed if seed is not None else int(time.time()) & 0x7fffffff
        self.counter = 0
        
    def get_rng(self) -> random.Random:
        """Get seeded random generator."""
        seed = self.base_seed + self.counter
        self.counter += 1
        return random.Random(seed)
    
    def get_numpy_rng(self) -> np.random.Generator:
        """Get seeded numpy random generator."""
        seed = self.base_seed + self.counter
        self.counter += 1
        return np.random.default_rng(seed)

# ---------------------------
# I/O & Graph Utilities
# ---------------------------

def load_pruned_graph(path: str) -> Dict[str, Any]:
    """Load pruned graph from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    for key in ('num_nodes', 'edges', 'distance_matrix'):
        if key not in data:
            raise ValueError(f"Pruned graph JSON missing required key: {key}")
    return data

def save_json(obj: Any, path: str):
    """Save object as JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_text(text: str, path: str):
    """Save text to file."""
    with open(path, 'w') as f:
        f.write(text)

def check_connectivity(n_nodes: int, edges: List[List[int]]) -> bool:
    """Check if graph is connected."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges])
    return nx.is_connected(G)

def ensure_connected(n_nodes: int, edges: List[List[int]], D: List[List[float]]) -> List[List[int]]:
    """Ensure graph connectivity by adding minimal edges."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges])
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return edges
    
    new_edges = edges.copy()
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
                            best_pair = (a, b)
        if best_pair:
            new_edges.append([best_pair[0], best_pair[1]])
            G.add_edge(*best_pair)
            components = list(nx.connected_components(G))
    return new_edges

# ---------------------------
# Quantum Circuit Utilities
# ---------------------------

def compute_grover_reps(index_qubits: int, marked_count: int, n_max_quantum: Optional[int] = None) -> int:
    """
    Compute optimal Grover repetitions.
    
    Args:
        index_qubits: Number of index qubits
        marked_count: Number of marked states
        n_max_quantum: Optional cap on repetitions
        
    Returns:
        Number of Grover iterations
    """
    N = 1 << max(1, index_qubits)
    M = max(1, int(marked_count))
    R = int(round((math.pi / 4.0) * math.sqrt(N / float(M))))
    lb = max(1, int(max(1, math.sqrt(N) / 3.0)))
    R = max(R, lb)
    
    if n_max_quantum is not None:
        R = min(R, max(1, int(n_max_quantum)))
    return R

def _apply_multi_controlled_z(qc: QuantumCircuit, qubits: List[int]):
    """
    Apply multi-controlled Z gate using MCX trick.
    
    Args:
        qc: Quantum circuit
        qubits: List of qubit indices (last one is target)
    """
    if len(qubits) == 0:
        return
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    if len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        mcx = MCXGate(len(controls))
        qc.append(mcx, controls + [target])
    qc.h(target)

def build_phase_oracle_marked_indices(marked_indices: List[int], index_qubits: int) -> QuantumCircuit:
    """
    Build phase oracle for marked indices.
    
    Args:
        marked_indices: List of indices to mark
        index_qubits: Number of qubits in index register
        
    Returns:
        Quantum circuit implementing the oracle
    """
    qr = QuantumRegister(index_qubits, 'idx')
    qc = QuantumCircuit(qr, name='oracle')
    
    for idx in marked_indices:
        bitstr = format(idx, '0{}b'.format(index_qubits))[::-1]
        zero_positions = [i for i, b in enumerate(bitstr) if b == '0']
        for p in zero_positions:
            qc.x(qr[p])
        _apply_multi_controlled_z(qc, [qr[i] for i in range(index_qubits)])
        for p in zero_positions:
            qc.x(qr[p])
    return qc

def build_grover_diffusion(index_qubits: int) -> QuantumCircuit:
    """Build Grover diffusion operator."""
    qr = QuantumRegister(index_qubits, 'idx')
    qc = QuantumCircuit(qr, name='diffusion')
    qc.h(qr)
    qc.x(qr)
    _apply_multi_controlled_z(qc, [qr[i] for i in range(index_qubits)])
    qc.x(qr)
    qc.h(qr)
    return qc

def grover_search_marked_indices_circuit(
    marked_indices: List[int],
    index_qubits: int,
    repetitions: int,
    backend_config: BackendConfig,
    shots: int = 512,
    start_time: Optional[float] = None,
    time_budget_remaining: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run Grover search with proper backend handling.
    
    Args:
        marked_indices: List of indices to mark
        index_qubits: Number of index qubits
        repetitions: Number of Grover iterations
        backend_config: Backend configuration
        shots: Number of measurement shots
        start_time: Start time for budget tracking
        time_budget_remaining: Remaining time budget
        
    Returns:
        Dictionary with results and metadata
    """
    if index_qubits < 1:
        index_qubits = 1
    
    # Time budget check
    if time_budget_remaining is not None and start_time is not None:
        if (time.time() - start_time) > time_budget_remaining:
            return {'error': 'time_budget_exceeded_before_build'}
    
    # Build circuit
    qr = QuantumRegister(index_qubits, 'idx')
    cr = ClassicalRegister(index_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    
    oracle = build_phase_oracle_marked_indices(marked_indices, index_qubits)
    diffusion = build_grover_diffusion(index_qubits)
    
    for _ in range(repetitions):
        qc.compose(oracle, qubits=qr, inplace=True)
        qc.compose(diffusion, qubits=qr, inplace=True)
        
        if time_budget_remaining is not None and start_time is not None:
            if (time.time() - start_time) > time_budget_remaining:
                return {'error': 'time_budget_exceeded_during_build'}
    
    # Run on appropriate backend
    t0 = time.time()
    try:
        # Check if we should use statevector backend
        use_statevector = hasattr(backend_config.state_backend, 'name') and 'statevector' in str(backend_config.state_backend.name).lower()
        
        if use_statevector:
            # Statevector simulation
            qc_nomeas = qc.copy()
            qc_nomeas.remove_final_measurements(inplace=False)
            compiled = transpile(
                qc_nomeas,
                backend=backend_config.state_backend,
                optimization_level=backend_config.optimization_level
            )
            job = backend_config.state_backend.run(compiled)
            result = job.result()
            statevector = result.get_statevector(compiled)
            t1 = time.time()
            
            return {
                'statevector': statevector,
                'timing': {'total_time_s': t1 - t0},
                'qc': qc_nomeas,
                'compiled_qc': compiled
            }
        else:
            # Sampling simulation
            qc.measure(qr, cr)
            compiled = transpile(
                qc,
                backend=backend_config.sampling_backend,
                optimization_level=backend_config.optimization_level
            )
            job = backend_config.sampling_backend.run(compiled, shots=shots)
            result = job.result()
            counts = result.get_counts()
            t1 = time.time()
            
            return {
                'counts': counts,
                'timing': {'total_time_s': t1 - t0},
                'qc': qc,
                'compiled_qc': compiled
            }
            
    except Exception as e:
        return {'error': f'simulation_failed: {str(e)}'}

# ---------------------------
# Held-Karp DP Enumeration
# ---------------------------

def held_karp_enumeration(subtree: Dict[str, Any], max_mask_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Enumerate paths using Held-Karp DP.
    
    Args:
        subtree: Subgraph information
        max_mask_size: Maximum size of mask to consider
        
    Returns:
        List of path assignments with costs
    """
    nodes = list(range(len(subtree['nodes'])))
    n = len(nodes)
    D = subtree['distance_matrix']
    max_mask_size = n if max_mask_size is None else min(n, max_mask_size)
    
    dp: Dict[Tuple[int, int], Tuple[float, int]] = {}
    for i in range(n):
        mask = 1 << i
        dp[(mask, i)] = (0.0, -1)
    
    for size in range(2, max_mask_size + 1):
        masks = [m for m in range(1, 1 << n) if bin(m).count("1") == size]
        for mask in masks:
            for j in range(n):
                if not (mask & (1 << j)):
                    continue
                prev_mask = mask ^ (1 << j)
                best_cost = float('inf')
                best_prev = -1
                
                for k in range(n):
                    if k == j:
                        continue
                    if not (prev_mask & (1 << k)):
                        continue
                    if (prev_mask, k) not in dp:
                        continue
                    
                    prev_cost = dp[(prev_mask, k)][0]
                    cost = prev_cost + D[k][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = k
                
                if best_prev >= 0:
                    dp[(mask, j)] = (best_cost, best_prev)
    
    assignments = []
    for (mask, last), (cost, prev) in dp.items():
        path = []
        cur_mask = mask
        cur_last = last
        
        while cur_last != -1:
            path.append(cur_last)
            entry = dp.get((cur_mask, cur_last), None)
            if entry is None:
                break
            _, p = entry
            cur_mask ^= (1 << cur_last)
            cur_last = p
        
        path = path[::-1]
        assignments.append({
            'bitmask': mask,
            'last': last,
            'path': path,
            'cost': cost
        })
    
    # Deduplicate
    best_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for a in assignments:
        key = (a['bitmask'], a['last'])
        if key not in best_map or a['cost'] < best_map[key]['cost']:
            best_map[key] = a
    
    return list(best_map.values())

# ---------------------------
# Montanaro Feasibility Check
# ---------------------------

def montanaro_practical_feasibility_improved(
    subtree: Dict[str, Any],
    cost_bound: float,
    backend_config: BackendConfig,
    start_time: Optional[float] = None,
    time_budget_remaining: Optional[float] = None,
    enumeration_safe_limit: int = 10000
) -> Dict[str, Any]:
    """
    Montanaro-style feasibility check using Held-Karp and Grover.
    
    Args:
        subtree: Local subtree to check
        cost_bound: Cost bound for feasibility
        backend_config: Backend configuration
        start_time: Start time for budget tracking
        time_budget_remaining: Remaining time budget
        enumeration_safe_limit: Safe limit for enumeration
        
    Returns:
        Feasibility result with metadata
    """
    t0 = time.time()
    enumerated = held_karp_enumeration(subtree)
    t1 = time.time()
    
    if len(enumerated) == 0:
        return {'feasible': False, 'reason': 'no_enumerated'}
    
    marked = [i for i, a in enumerate(enumerated) if a['cost'] <= cost_bound]
    if len(marked) == 0:
        return {
            'feasible': False,
            'reason': 'no_marked',
            'num_enumerated': len(enumerated)
        }
    
    if len(enumerated) > enumeration_safe_limit:
        return {
            'feasible': False,
            'reason': 'enumeration_too_large',
            'num_enumerated': len(enumerated)
        }
    
    index_qubits = max(1, math.ceil(math.log2(len(enumerated))))
    M = len(marked)
    R = max(1, int(math.floor((math.pi / 4.0) * math.sqrt((1 << index_qubits) / M))))
    
    grover_res = grover_search_marked_indices_circuit(
        marked,
        index_qubits,
        R,
        backend_config,
        shots=256,
        start_time=start_time,
        time_budget_remaining=time_budget_remaining
    )
    
    if 'error' in grover_res:
        return {
            'feasible': False,
            'reason': 'time_budget_exceeded_or_error',
            'details': grover_res
        }
    
    # Process result
    if 'statevector' in grover_res:
        sv = grover_res['statevector']
        probs = np.abs(np.array(sv)) ** 2
        measured_int = int(np.argmax(probs))
        assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
        
        return {
            'feasible': assgn is not None and assgn['cost'] <= cost_bound,
            'measured_assignment': assgn,
            'num_enumerated': len(enumerated),
            'marked_count': len(marked),
            'state_probs_top': float(np.max(probs)),
            'timing': grover_res['timing']
        }
    else:
        counts = grover_res.get('counts', {})
        if not counts:
            return {
                'feasible': False,
                'reason': 'no_counts',
                'details': grover_res
            }
        
        measured = max(counts.items(), key=lambda x: x[1])[0]
        measured_int = int(measured, 2)
        assgn = enumerated[measured_int] if measured_int < len(enumerated) else None
        
        return {
            'feasible': assgn is not None and assgn['cost'] <= cost_bound,
            'measured_assignment': assgn,
            'num_enumerated': len(enumerated),
            'marked_count': len(marked),
            'grover_counts': counts,
            'timing': grover_res['timing']
        }

# ---------------------------
# Utility Functions
# ---------------------------

def path_cost(path: List[int], Dmat: List[List[float]]) -> float:
    """Compute cost of a path."""
    c = 0.0
    for i in range(1, len(path)):
        c += Dmat[path[i - 1]][path[i]]
    return c

def close_tour_cost(path: List[int], Dmat: List[List[float]]) -> float:
    """Compute closed tour cost."""
    c = path_cost(path, Dmat)
    c += Dmat[path[-1]][path[0]]
    return c

def average_branching_factor(pruned_graph: Dict[str, Any]) -> float:
    """Compute average branching factor."""
    n = pruned_graph['num_nodes']
    degs = [0] * n
    for a, b in pruned_graph['edges']:
        degs[a] += 1
        degs[b] += 1
    nz = [d for d in degs if d > 0]
    return float(sum(nz)) / len(nz) if len(nz) > 0 else 0.0

def topk_successors_from_pruned_graph(
    pruned_graph: Dict[str, Any],
    node: int,
    k: int
) -> List[Tuple[int, float]]:
    """Get top-k successors from pruned graph."""
    D = pruned_graph['distance_matrix']
    neighbors = set()
    
    for a, b in pruned_graph['edges']:
        if a == node:
            neighbors.add(b)
        elif b == node:
            neighbors.add(a)
    
    neighs = [(nb, D[node][nb]) for nb in neighbors]
    neighs.sort(key=lambda x: x[1])
    return neighs[:k]

def extract_local_subtree(
    pruned_graph: Dict[str, Any],
    root: int,
    max_nodes: int = 12
) -> Dict[str, Any]:
    """Extract local subtree for quantum processing."""
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
    
    for a, b in pruned_graph['edges']:
        if a in mapping and b in mapping:
            local_edges.append((mapping[a], mapping[b]))
    
    D_full = pruned_graph['distance_matrix']
    D = [[D_full[orig_a][orig_b] for orig_b in nodes] for orig_a in nodes]
    
    return {
        'nodes': nodes,
        'edges': local_edges,
        'distance_matrix': D,
        'orig_to_local': mapping,
        'local_to_orig': nodes
    }

# ---------------------------
# 2-opt Local Improvement
# ---------------------------

def two_opt_improve(
    path: List[int],
    D: List[List[float]],
    time_budget: Optional[float] = None
) -> Tuple[List[int], float]:
    """2-opt local improvement with time budget."""
    start = time.time()
    n = len(path)
    best = path[:]
    best_cost = close_tour_cost(best, D)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = close_tour_cost(new_path, D)
                
                if new_cost < best_cost:
                    best = new_path
                    best_cost = new_cost
                    improved = True
                    break
            
            if improved:
                break
            
            if time_budget is not None and (time.time() - start) > time_budget:
                return best, best_cost
    
    return best, best_cost

# ---------------------------
# Core Grover Result Processing (FIXED)
# ---------------------------

def process_grover_result(
    grover_res: Dict[str, Any],
    marked: List[int],
    candidates: List[int],
    dists: List[float],
    metrics: SearchMetrics
) -> Tuple[int, Dict[str, Any]]:
    """
    CORRECTED: Process Grover result with proper index mapping.
    
    Args:
        grover_res: Result from Grover search
        marked: List of marked indices into candidates list
        candidates: List of candidate node indices
        dists: List of distances for candidates
        metrics: Metrics tracker
        
    Returns:
        Tuple of (chosen_candidate_index, grover_info)
    """
    # Default fallback
    fallback_idx = min(range(len(candidates)), key=lambda x: dists[x])
    
    if 'error' in grover_res:
        metrics.grover_failures += 1
        return fallback_idx, {'fallback': True, 'reason': grover_res['error']}
    
    # Extract measurement result
    measured_int = None
    grover_info = {}
    
    if 'counts' in grover_res:
        counts = grover_res['counts']
        freq = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        try:
            measured_int = int(freq[0][0], 2)
            grover_info = {'counts': counts, 'timing': grover_res.get('timing')}
        except Exception as e:
            metrics.grover_failures += 1
            return fallback_idx, {'fallback': True, 'reason': f'counts_parse_error: {str(e)}'}
    
    elif 'statevector' in grover_res:
        sv = grover_res['statevector']
        probs = np.abs(np.array(sv)) ** 2
        measured_int = int(np.argmax(probs))
        grover_info = {
            'state_probs_top': float(np.max(probs)),
            'timing': grover_res.get('timing')
        }
    
    else:
        metrics.grover_failures += 1
        return fallback_idx, {'fallback': True, 'reason': 'no_valid_result'}
    
    # CORRECTED: Map measured_int to candidate index
    # The marked list contains indices into the candidates list
    # measured_int is an index into the search space (0 to 2^index_qubits - 1)
    # We need to check if measured_int corresponds to a marked index
    # CORRECT mapping: measured_int is a candidate index
    if 0 <= measured_int < len(candidates):
        if measured_int in marked:
            metrics.grover_successes += 1
            return measured_int, grover_info

    
    
    # Fallback if mapping failed
    metrics.grover_failures += 1
    return fallback_idx, {'fallback': True, 'reason': 'mapping_failed'}

# ---------------------------
# Beam Item Expansion (Refactored)
# ---------------------------

def expand_beam_item(
    item: Dict[str, Any],
    pruned_graph: Dict[str, Any],
    config: SearchConfig,
    backend_config: BackendConfig,
    metrics: SearchMetrics,
    time_left: Callable[[], float],
    start_time: float,
    random_mgr: RandomManager
) -> List[Dict[str, Any]]:
    """
    Expand a single beam item with quantum assistance.
    
    Args:
        item: Current beam item
        pruned_graph: Pruned graph data
        config: Search configuration
        backend_config: Backend configuration
        metrics: Metrics tracker
        time_left: Function returning remaining time
        start_time: Start time
        random_mgr: Random manager
        
    Returns:
        List of expanded items
    """
    last = item['path'][-1]
    visited = item['visited']
    
    # Get candidate successors
    neighs = topk_successors_from_pruned_graph(pruned_graph, last, config.topk)
    candidates = [t[0] for t in neighs if not (visited & (1 << t[0]))][:config.k_cap]
    dists = [pruned_graph['distance_matrix'][last][c] for c in candidates]
    
    if not candidates:
        # Classical fallback
        rem = [i for i in range(pruned_graph['num_nodes']) if not (visited & (1 << i))]
        if not rem:
            return []
        chosen = min(rem, key=lambda x: pruned_graph['distance_matrix'][last][x])
        new_path = item['path'] + [chosen]
        new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
        
        metrics.classical_expansions += 1
        return [{
            'path': new_path,
            'visited': item['visited'] | (1 << chosen),
            'cost': new_cost
        }]
    
    # Quantum candidate selection
    min_d = min(dists)
    slack = 1.10
    marked = [i for i, d in enumerate(dists) if d <= min_d * slack]
    
    # Limit marked candidates
    K_mark = max(1, min(4, len(dists)))
    if len(marked) > K_mark:
        ords = sorted(range(len(dists)), key=lambda i: dists[i])
        marked = ords[:K_mark]
    
    # Run Grover search
    index_qubits = max(1, math.ceil(math.log2(max(1, len(candidates)))))
    repetitions = compute_grover_reps(
        index_qubits=index_qubits,
        marked_count=len(marked),
        n_max_quantum=config.n_max_quantum
    )
    
    per_call_budget = max(0.5, time_left() * 0.05)

    logger.info(
        f"ABOUT TO CALL GROVER | "
        f"candidates={len(candidates)}, "
        f"marked={len(marked)}, "
        f"index_qubits={index_qubits}, "
        f"reps={repetitions}"
    )

    
    grover_res = grover_search_marked_indices_circuit(
        marked,
        index_qubits,
        repetitions,
        backend_config,
        shots=256,
        start_time=start_time,
        time_budget_remaining=per_call_budget
    )
    
    # Process Grover result (with corrected mapping)
    chosen_idx, grover_info = process_grover_result(
        grover_res, marked, candidates, dists, metrics
    )
    
    # Track quantum time
    if 'timing' in grover_res:
        metrics.quantum_time_s += grover_res['timing'].get('total_time_s', 0.0)
    metrics.quantum_calls += 1
    
    # Create expansion
    chosen = candidates[chosen_idx]
    new_path = item['path'] + [chosen]
    new_cost = item['cost'] + pruned_graph['distance_matrix'][last][chosen]
    new_visited = item['visited'] | (1 << chosen)
    
    expansion = {
        'path': new_path,
        'visited': new_visited,
        'cost': new_cost,
        'grover_info': grover_info
    }
    
    # Optional Montanaro check
    if config.use_montanaro and config.n_max_quantum and config.n_max_quantum >= 3:
        subtree = extract_local_subtree(pruned_graph, chosen, max_nodes=config.n_max_quantum)
        if len(subtree['nodes']) <= 10 or config.force_montanaro:
            remaining = pruned_graph['num_nodes'] - len(new_path)
            avg_edge = np.mean([pruned_graph['distance_matrix'][a][b] for a, b in pruned_graph['edges']])
            estimated_extra = remaining * max(1.0, avg_edge)
            cost_bound = new_cost + estimated_extra
            
            if time_left() > 0.5:
                ma_budget = min(max(0.5, time_left() * 0.05), 8.0)
                ma_res = montanaro_practical_feasibility_improved(
                    subtree,
                    cost_bound,
                    backend_config,
                    start_time=start_time,
                    time_budget_remaining=ma_budget
                )
                expansion['montanaro'] = ma_res
                metrics.montanaro_checks += 1
    
    return [{
        'path': new_path,
        'visited': new_visited,
        'cost': new_cost
    }]

# ---------------------------
# Beam Pruning and Diversification
# ---------------------------

def prune_and_diversify(
    expansions: List[Dict[str, Any]],
    beam_size: int,
    n_nodes: int,
    D: List[List[float]],
    random_mgr: RandomManager,
    diversify: bool = True
) -> List[Dict[str, Any]]:
    """
    Prune expansions to beam size and add diversity.
    
    Args:
        expansions: List of expanded items
        beam_size: Target beam size
        n_nodes: Number of nodes
        D: Distance matrix
        random_mgr: Random manager
        diversify: Whether to add diversity
        
    Returns:
        Pruned and diversified beam
    """
    if not expansions:
        return []
    
    # Sort by cost
    expansions.sort(key=lambda x: x['cost'])
    
    # Take top beam_size
    beam_new = expansions[:min(len(expansions), beam_size)]
    
    # Add diversity if needed
    if len(beam_new) < beam_size and diversify:
        seen_paths = {tuple(item['path']) for item in beam_new}
        
        # Try to add more expansions
        for item in expansions[len(beam_new):]:
            if tuple(item['path']) not in seen_paths:
                beam_new.append(item)
                seen_paths.add(tuple(item['path']))
            if len(beam_new) >= beam_size:
                break
        
        # If still short, generate random completions
        if len(beam_new) < beam_size:
            rng = random_mgr.get_rng()
            base_item = beam_new[0] if beam_new else None
            if base_item:
                while len(beam_new) < beam_size:
                    # Create slightly mutated version
                    mutated = base_item.copy()
                    if len(mutated['path']) >= 2:
                        idx1 = rng.randint(1, len(mutated['path']) - 1)
                        idx2 = rng.randint(1, len(mutated['path']) - 1)
                        if idx1 != idx2:
                            path = mutated['path'][:]
                            path[idx1], path[idx2] = path[idx2], path[idx1]
                            mutated['path'] = path
                            mutated['cost'] = path_cost(path, D)
                            if tuple(mutated['path']) not in seen_paths:
                                beam_new.append(mutated)
                                seen_paths.add(tuple(mutated['path']))
    
    return beam_new

# ---------------------------
# Main Beam Search Algorithm
# ---------------------------

def beam_search_quantum_refactored(
    pruned_graph: Dict[str, Any],
    config: SearchConfig,
    backend_config: BackendConfig
) -> Dict[str, Any]:
    """
    Main beam search algorithm with quantum acceleration.
    
    Args:
        pruned_graph: Pruned graph data
        config: Search configuration
        backend_config: Backend configuration
        
    Returns:
        Search results with metrics
    """
    n = pruned_graph['num_nodes']
    D = pruned_graph['distance_matrix']
    
    # Initialize metrics
    metrics = SearchMetrics()
    random_mgr = RandomManager(config.seed)
    
    # Configure based on mode
    if config.mode == 'speed':
        config.beam_size = config.beam_size if config.beam_size is not None else 3
        config.topk = min(config.topk, 8)
        config.enable_2opt = False
        config.restarts = 1
    elif config.mode == 'balanced':
        config.beam_size = config.beam_size if config.beam_size is not None else 4
        config.topk = min(config.topk, 12)
        config.enable_2opt = True
        config.restarts = config.restarts if config.restarts is not None else 1
    else:  # accuracy
        config.beam_size = config.beam_size if config.beam_size is not None else 8
        config.topk = min(config.topk, 24)
        config.enable_2opt = True
        config.restarts = config.restarts if config.restarts is not None else 2
    
    # Set quantum parameters
    if config.n_max_quantum is None:
        if n <= 10:
            config.n_max_quantum = 10
        elif n <= 20:
            config.n_max_quantum = 6
        elif n <= 50:
            config.n_max_quantum = 4
        else:
            config.n_max_quantum = 0
    
    # Determine if Montanaro should be used
    branch = average_branching_factor(pruned_graph)
    montanaro_allowed = (
        config.use_montanaro and 
        config.n_max_quantum >= 3 and
        (config.force_montanaro or (n <= 20 and branch <= 4.0))
    )
    
    start_time = time.time()
    deadline = start_time + config.max_runtime
    
    def time_left() -> float:
        return max(0.0, deadline - time.time())
    
    best_global = {'path': None, 'cost': float('inf')}
    all_runs = []
    
    for restart_idx in range(config.restarts):
        if time.time() > deadline:
            break
        
        logger.info(f"[RUN {restart_idx + 1}/{config.restarts}] Starting with seed {config.seed + restart_idx if config.seed else 'random'}")
        
        # Initialize beam
        if config.randomize_start:
            rng = random_mgr.get_rng()
            start_node = rng.randint(0, n - 1)
        else:
            start_node = 0
        
        beam = [{
            'path': [start_node],
            'visited': (1 << start_node),
            'cost': 0.0
        }]
        
        run_log = {'restart': restart_idx, 'steps': []}
        
        # Main beam search loop
        for step in range(1, n):
            if time.time() > deadline:
                break
            
            if step % 5 == 0:
                logger.info(f"[RUN {restart_idx + 1}] step {step}/{n - 1}, beam={len(beam)}, time_left={time_left():.1f}s")
            
            # Expand all beam items
            expansions = []
            step_record = {'step': step, 'beam_size': len(beam), 'expansions': []}
            
            for item in beam:
                if time.time() > deadline:
                    break
                
                new_items = expand_beam_item(
                    item, pruned_graph, config, backend_config,
                    metrics, time_left, start_time, random_mgr
                )
                
                expansions.extend(new_items)
                
                for new_item in new_items:
                    step_record['expansions'].append({
                        'from': item['path'],
                        'to': new_item['path'][-1],
                        'cost': new_item['cost']
                    })
            
            # Prune and diversify
            beam = prune_and_diversify(
                expansions, config.beam_size, n, D, random_mgr,
                diversify=(config.mode != 'speed')
            )
            
            run_log['steps'].append(step_record)
            metrics.beam_history.append({
                'restart': restart_idx,
                'step': step,
                'beam_size': len(beam),
                'best_cost': min(b['cost'] for b in beam) if beam else float('inf')
            })
            
            # Check if we have complete tours
            if any(len(b['path']) == n for b in beam):
                break
        
        # Complete tours and select best for this restart
        full_tours = []
        for b in beam:
            if len(b['path']) == n:
                cost = close_tour_cost(b['path'], D)
                full_tours.append({'path': b['path'], 'cost': cost})
            elif len(b['path']) < n:
                # Complete greedily
                cur = b['path'][:]
                visited = b['visited']
                while len(cur) < n:
                    last = cur[-1]
                    remaining = [i for i in range(n) if not (visited & (1 << i))]
                    if not remaining:
                        break
                    nextc = min(remaining, key=lambda x: D[last][x])
                    cur.append(nextc)
                    visited |= (1 << nextc)
                
                if len(cur) == n:
                    cost = close_tour_cost(cur, D)
                    full_tours.append({'path': cur, 'cost': cost})
        
        # Apply 2-opt improvement if enabled
        if full_tours and config.enable_2opt and time_left() > 2.0:
            best_local = min(full_tours, key=lambda x: x['cost'])
            opt_budget = min(10.0, time_left() * 0.25)
            improved_path, improved_cost = two_opt_improve(
                best_local['path'], D, opt_budget
            )
            full_tours.append({
                'path': improved_path,
                'cost': improved_cost,
                '2opt': True
            })
        
        # Update global best
        if full_tours:
            best_local = min(full_tours, key=lambda x: x['cost'])
            run_log['best'] = best_local
            
            if best_local['cost'] < best_global['cost']:
                best_global = best_local
            
            all_runs.append(run_log)
        
        logger.info(f"[RUN {restart_idx + 1}] Best cost: {best_local['cost'] if 'best_local' in locals() else 'N/A'}")
    
    # Final metrics
    metrics.wall_time_s = time.time() - start_time
    
    return {
        'best_tour': best_global,
        'runs': all_runs,
        'metrics': {
            'wall_time_s': metrics.wall_time_s,
            'quantum_time_s': metrics.quantum_time_s,
            'quantum_calls': metrics.quantum_calls,
            'classical_expansions': metrics.classical_expansions,
            'grover_success_rate': metrics.grover_success_rate,
            'montanaro_checks': metrics.montanaro_checks,
            'avg_beam_size': np.mean([step['beam_size'] for step in metrics.beam_history]) if metrics.beam_history else 0
        },
        'config': {
            'mode': config.mode,
            'beam_size': config.beam_size,
            'topk': config.topk,
            'restarts': config.restarts,
            'n_max_quantum': config.n_max_quantum,
            'montanaro_allowed': montanaro_allowed,
            'seed': config.seed
        }
    }

# ---------------------------
# CLI Interface
# ---------------------------

def parse_opt_tour(opt_path: str) -> List[int]:
    """Parse optimal tour from file."""
    with open(opt_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    if 'TOUR_SECTION' in lines:
        idx = lines.index('TOUR_SECTION')
        seq = []
        for l in lines[idx + 1:]:
            if l == '-1' or l.upper() == 'EOF':
                break
            try:
                seq.append(int(l) - 1)
            except:
                continue
        return seq
    
    seq = []
    for l in lines:
        try:
            seq.append(int(l) - 1)
        except:
            continue
    
    return [x for x in seq if x >= 0]

def default_output_paths(input_json_path: str):
    """Generate default output paths."""
    base = os.path.splitext(os.path.basename(input_json_path))[0]
    out_dir = "quantum_run_outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    return {
        'json': os.path.join(out_dir, f"{base}_beamrun_{ts}.json"),
        'md': os.path.join(out_dir, f"{base}_beamrun_{ts}.md")
    }

def pretty_md_report(
    result: Dict[str, Any],
    input_info: Dict[str, Any],
    opt_info: Optional[Dict[str, Any]] = None
) -> str:
    """Generate markdown report."""
    md = []
    md.append("# Quantum-Assisted Beam Search Report")
    md.append("")
    md.append("## Input Parameters")
    for k, v in input_info.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Results")
    
    best = result.get('best_tour')
    if best and best.get('path'):
        md.append(f"- **Best tour cost**: {best['cost']:.3f}")
        md.append(f"- **Path length**: {len(best['path'])} nodes")
        md.append(f"- **Path**: {best['path'][:10]}{'...' if len(best['path']) > 10 else ''}")
    else:
        md.append("- No valid tour found.")
    
    metrics = result.get('metrics', {})
    md.append("")
    md.append("## Performance Metrics")
    md.append(f"- **Wall time**: {metrics.get('wall_time_s', 0):.2f}s")
    md.append(f"- **Quantum time**: {metrics.get('quantum_time_s', 0):.2f}s")
    md.append(f"- **Quantum calls**: {metrics.get('quantum_calls', 0)}")
    md.append(f"- **Grover success rate**: {metrics.get('grover_success_rate', 0):.1%}")
    md.append(f"- **Classical expansions**: {metrics.get('classical_expansions', 0)}")
    
    if opt_info is not None:
        md.append("")
        md.append("## Optimality Comparison")
        md.append(f"- **Optimal tour cost**: {opt_info.get('opt_cost', 'N/A')}")
        if opt_info.get('gap_percent') is not None:
            md.append(f"- **Optimality gap**: {opt_info['gap_percent']:.2f}%")
    
    return "\n".join(md)

def main_cli():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Quantum-Assisted Beam Search for TSP"
    )
    parser.add_argument('pruned_graph', help='Path to pruned graph JSON')
    parser.add_argument('--mode', choices=['speed', 'balanced', 'accuracy'],
                       default='balanced')
    parser.add_argument('--beam', type=int, default=None)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--use_montanaro', action='store_true')
    parser.add_argument('--force_montanaro', action='store_true')
    parser.add_argument('--n_max_quantum', type=int, default=None)
    parser.add_argument('--max_runtime', type=int, default=300)
    parser.add_argument('--k_cap', type=int, default=64)
    parser.add_argument('--opt_tour_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    # Load and prepare graph
    pruned = load_pruned_graph(args.pruned_graph)
    n = pruned['num_nodes']
    edges = [list(e) for e in pruned['edges']]
    D = pruned['distance_matrix']
    
    if not check_connectivity(n, edges):
        logger.info("Graph not connected - repairing...")
        edges = ensure_connected(n, edges, D)
        pruned['edges'] = edges
        logger.info("Connectivity repaired.")
    
    # Create backends
    sampling_backend = AerSimulator()
    try:
        state_backend = AerSimulator(method='statevector')
    except Exception:
        logger.warning("Statevector backend unavailable, using sampling backend for all operations")
        state_backend = sampling_backend
    
    backend_config = BackendConfig(
        sampling_backend=sampling_backend,
        state_backend=state_backend,
        shots=512,
        optimization_level=1
    )
    
    # Create search config
    search_config = SearchConfig(
        beam_size=args.beam,
        topk=args.k,
        mode=args.mode,
        restarts=args.restarts,
        use_montanaro=args.use_montanaro,
        force_montanaro=args.force_montanaro,
        n_max_quantum=args.n_max_quantum,
        max_runtime=args.max_runtime,
        k_cap=args.k_cap,
        seed=args.seed
    )
    
    # Run search
    logger.info(f"Starting quantum beam search on {args.pruned_graph} (n={n})")
    logger.info(f"Mode: {args.mode}, Beam: {args.beam}, K: {args.k}, Restarts: {args.restarts}")
    
    start_time = time.time()
    result = beam_search_quantum_refactored(pruned, search_config, backend_config)
    end_time = time.time()
    
    result['wall_time_total_s'] = end_time - start_time
    
    # Process optimal tour if provided
    opt_info = None
    if args.opt_tour_path and os.path.exists(args.opt_tour_path):
        try:
            opt_path = parse_opt_tour(args.opt_tour_path)
            if len(opt_path) == n:
                opt_cost = close_tour_cost(opt_path, D)
                best = result.get('best_tour')
                if best and best.get('path') and len(best['path']) == n:
                    gap = 100.0 * (best['cost'] - opt_cost) / opt_cost
                else:
                    gap = None
                opt_info = {
                    'opt_cost': opt_cost,
                    'gap_percent': gap,
                    'opt_path': opt_path
                }
        except Exception as e:
            logger.warning(f"Failed to parse optimal tour: {e}")
    
    # Save results
    out_paths = default_output_paths(args.pruned_graph)
    
    out_json = {
        'input': {
            'path': args.pruned_graph,
            'n': n,
            'mode': args.mode,
            'beam': args.beam,
            'k': args.k,
            'restarts': args.restarts,
            'seed': args.seed
        },
        'result': result,
        'opt_info': opt_info
    }
    
    save_json(out_json, out_paths['json'])
    
    md = pretty_md_report(
        result,
        out_json['input'],
        opt_info
    )
    save_text(md, out_paths['md'])
    
    logger.info(f"Results saved: {out_paths['json']}")
    logger.info(f"Report saved: {out_paths['md']}")
    
    if result.get('best_tour') and result['best_tour']['path']:
        logger.info(f"Best tour cost: {result['best_tour']['cost']:.3f}")
    
    if opt_info and opt_info.get('gap_percent') is not None:
        logger.info(f"Optimality gap: {opt_info['gap_percent']:.2f}%")
    
    logger.info(f"Total wall time: {result['wall_time_total_s']:.2f}s")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main_cli())
    except KeyboardInterrupt:
        logger.info("User interrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
