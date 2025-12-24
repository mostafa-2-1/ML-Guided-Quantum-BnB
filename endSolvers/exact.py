import os
import json
import time
import numpy as np
import networkx as nx
from itertools import combinations
from newTrain import parse_tsp

RESULTS_DIR = "exactSolver_results"
GRAPH_DIR = "graphs_chosen"
os.makedirs(RESULTS_DIR, exist_ok=True)

def euclidean_matrix(coords):
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = np.linalg.norm(coords[i]-coords[j])
    return D

def held_karp(D):
    n = D.shape[0]
    C = {}

    # Initialize for subsets of size 1 (start at node 0)
    for k in range(1, n):
        C[(1 << k, k)] = (D[0, k], [0, k])

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            for k in subset:
                prev_bits = bits & ~(1 << k)
                res = []
                for m in subset:
                    if m == k:
                        continue
                    cost, path = C[(prev_bits, m)]
                    res.append((cost + D[m, k], path + [k]))
                C[(bits, k)] = min(res, key=lambda x: x[0])

    # Close the tour
    bits = (2**n - 1) - 1  # all except 0
    res = []
    for k in range(1, n):
        cost, path = C[(bits, k)]
        res.append((cost + D[k, 0], path + [0]))
    min_cost, min_path = min(res, key=lambda x: x[0])
    return min_cost, min_path

def solve_graph(filepath):
    tsp = parse_tsp(filepath)
    coords = tsp["coords"]
    n = coords.shape[0]
    if n > 12:
        print(f"[Skip] {filepath} has {n} nodes > 12 for exact solver")
        return None

    D = euclidean_matrix(coords)
    start = time.time()
    cost, path = held_karp(D)
    runtime = time.time() - start
    return {"instance": os.path.basename(filepath).replace(".tsp", ""),
            "n": n,
            "tour_cost": float(cost),
            "tour": path,
            "runtime_sec": runtime}

def main():
    for fname in os.listdir(GRAPH_DIR):
        if not fname.endswith(".tsp"):
            continue
        filepath = os.path.join(GRAPH_DIR, fname)
        print(f"[Exact Solver] Solving {fname} ...")
        result = solve_graph(filepath)
        if result:
            out_path = os.path.join(RESULTS_DIR, f"{result['instance']}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
