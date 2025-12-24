#!/usr/bin/env python3

import os
import sys
import json
import time
import math
import random
import argparse
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Callable

import numpy as np
import networkx as nx

# ---------------------------
# Logging
# ---------------------------

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("classical_beam")

# ---------------------------
# Configuration and Data Classes
# ---------------------------

@dataclass
class BackendConfig:
    """Kept for symmetry with quantum version (unused here)."""
    sampling_backend: Any = None
    state_backend: Any = None
    shots: int = 0
    optimization_level: int = 0

@dataclass
class SearchConfig:
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
    wall_time_s: float = 0.0
    quantum_time_s: float = 0.0
    quantum_calls: int = 0
    classical_expansions: int = 0
    grover_successes: int = 0
    grover_failures: int = 0
    montanaro_checks: int = 0
    circuit_stats: Dict[str, Any] = field(default_factory=dict)
    beam_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def grover_success_rate(self) -> float:
        total = self.grover_successes + self.grover_failures
        return self.grover_successes / total if total > 0 else 0.0

class RandomManager:
    def __init__(self, seed: Optional[int] = None):
        self.base_seed = seed if seed is not None else int(time.time()) & 0x7fffffff
        self.counter = 0

    def get_rng(self) -> random.Random:
        seed = self.base_seed + self.counter
        self.counter += 1
        return random.Random(seed)

# ---------------------------
# IO and Graph Utilities
# ---------------------------

def load_pruned_graph(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        data = json.load(f)
    for key in ('num_nodes', 'edges', 'distance_matrix'):
        if key not in data:
            raise ValueError(f"Missing key: {key}")
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

def ensure_connected(n_nodes: int, edges: List[List[int]], D: List[List[float]]) -> List[List[int]]:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges])

    components = list(nx.connected_components(G))
    if len(components) == 1:
        return edges

    new_edges = edges.copy()
    while len(components) > 1:
        best = None
        best_d = float('inf')
        for c1 in components:
            for c2 in components:
                if c1 == c2:
                    continue
                for a in c1:
                    for b in c2:
                        if D[a][b] < best_d:
                            best_d = D[a][b]
                            best = (a, b)
        new_edges.append([best[0], best[1]])
        G.add_edge(*best)
        components = list(nx.connected_components(G))

    return new_edges

# ---------------------------
# Core Helpers
# ---------------------------

def path_cost(path: List[int], D: List[List[float]]) -> float:
    return sum(D[path[i]][path[i+1]] for i in range(len(path)-1))

def close_tour_cost(path: List[int], D: List[List[float]]) -> float:
    return path_cost(path, D) + D[path[-1]][path[0]]

def topk_successors_from_pruned_graph(pruned_graph, node, k):
    neighs = []
    for a, b in pruned_graph['edges']:
        if a == node:
            neighs.append(b)
        elif b == node:
            neighs.append(a)
    neighs = list(set(neighs))
    D = pruned_graph['distance_matrix']
    neighs.sort(key=lambda x: D[node][x])
    return [(n, D[node][n]) for n in neighs[:k]]

def average_branching_factor(pruned_graph):
    degs = [0] * pruned_graph['num_nodes']
    for a, b in pruned_graph['edges']:
        degs[a] += 1
        degs[b] += 1
    return float(sum(degs)) / len(degs)

# ---------------------------
# Optimization
# ---------------------------

def two_opt_improve(path: List[int], D: List[List[float]], time_budget: float):
    start = time.time()
    best = path[:]
    best_cost = close_tour_cost(best, D)
    n = len(path)

    while time.time() - start < time_budget:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new = best[:]
                new[i:j] = reversed(new[i:j])
                cost = close_tour_cost(new, D)
                if cost < best_cost:
                    best = new
                    best_cost = cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost

# ---------------------------
# Classical Beam Expansion
# ---------------------------

def expand_beam_item_classical(
    item: Dict[str, Any],
    pruned_graph: Dict[str, Any],
    config: SearchConfig,
    metrics: SearchMetrics,
    random_mgr: RandomManager
) -> List[Dict[str, Any]]:

    last = item['path'][-1]
    visited = item['visited']
    D = pruned_graph['distance_matrix']

    neighs = topk_successors_from_pruned_graph(pruned_graph, last, config.topk)
    candidates = [n for n, _ in neighs if not (visited & (1 << n))][:config.k_cap]

    if not candidates:
        return []

    expansions = []
    for c in candidates[:1]:
        new_path = item['path'] + [c]
        new_cost = item['cost'] + D[last][c]
        expansions.append({
            'path': new_path,
            'visited': visited | (1 << c),
            'cost': new_cost
        })
        metrics.classical_expansions += 1

    return expansions

# ---------------------------
# Beam Pruning
# ---------------------------

def prune_and_diversify(expansions, beam_size):
    expansions.sort(key=lambda x: x['cost'])
    return expansions[:beam_size]

# ---------------------------
# Main Classical Beam Search
# ---------------------------

def beam_search_classical_refactored(
    pruned_graph: Dict[str, Any],
    config: SearchConfig
) -> Dict[str, Any]:

    start_time = time.time()
    deadline = start_time + config.max_runtime

    metrics = SearchMetrics()
    random_mgr = RandomManager(config.seed)

    n = pruned_graph['num_nodes']
    D = pruned_graph['distance_matrix']

    best_global = {'path': None, 'cost': float('inf')}
    all_runs = []

    for r in range(config.restarts):
        if time.time() > deadline:
            break

        start_node = random_mgr.get_rng().randint(0, n-1) if config.randomize_start else 0
        beam = [{
            'path': [start_node],
            'visited': (1 << start_node),
            'cost': 0.0
        }]

        for step in range(1, n):
            expansions = []
            for item in beam:
                expansions.extend(
                    expand_beam_item_classical(item, pruned_graph, config, metrics, random_mgr)
                )
            if not expansions:
                break
            beam = prune_and_diversify(expansions, config.beam_size)

            metrics.beam_history.append({
                'step': step,
                'beam_size': len(beam),
                'best_cost': min(b['cost'] for b in beam)
            })

        for b in beam:
            if len(b['path']) == n:
                cost = close_tour_cost(b['path'], D)
                if cost < best_global['cost']:
                    best_global = {'path': b['path'], 'cost': cost}

    metrics.wall_time_s = time.time() - start_time

    return {
        'best_tour': best_global,
        'metrics': {
            'wall_time_s': metrics.wall_time_s,
            'quantum_time_s': 0.0,
            'quantum_calls': 0,
            'classical_expansions': metrics.classical_expansions,
            'grover_success_rate': 0.0,
            'montanaro_checks': 0,
            'avg_beam_size': np.mean([b['beam_size'] for b in metrics.beam_history])
            if metrics.beam_history else 0
        }
    }

# ---------------------------
# CLI
# ---------------------------

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('pruned_graph')
    parser.add_argument('--mode', default='balanced')
    parser.add_argument('--beam', type=int, default=4)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--max_runtime', type=int, default=300)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    pruned = load_pruned_graph(args.pruned_graph)
    n = pruned['num_nodes']
    D = pruned['distance_matrix']

    if not check_connectivity(n, pruned['edges']):
        pruned['edges'] = ensure_connected(n, pruned['edges'], D)

    config = SearchConfig(
        beam_size=args.beam,
        topk=args.k,
        mode=args.mode,
        restarts=args.restarts,
        use_montanaro=False,
        force_montanaro=False,
        n_max_quantum=None,
        max_runtime=args.max_runtime,
        seed=args.seed
    )

    result = beam_search_classical_refactored(pruned, config)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    sys.exit(main_cli())
