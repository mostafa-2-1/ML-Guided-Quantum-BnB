import os
import json
import time
import math
import numpy as np
import sys
import networkx as nx
from itertools import combinations

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from newTrain import parse_tsp


# ---------------------------
# Utilities
# ---------------------------

def build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = math.dist(coords[i], coords[j])
    return D


def load_edgelist_graph(edgelist_path, n):
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    with open(edgelist_path) as f:
        for line in f:
            u, v, w = line.strip().split()
            u, v, w = int(u), int(v), float(w)
            D[u, v] = w
            D[v, u] = w

    return D



def load_pruned_graph(json_path, coords=None):
    with open(json_path) as f:
        data = json.load(f)

    n = data["n"]
    edges = data["edges"]
    D = np.full((n, n), np.inf)
    np.fill_diagonal(D, 0)

    for edge in edges:
        if len(edge) == 3:
            u, v, w = edge
        elif len(edge) == 2:
            u, v = edge
            if coords is not None:
                # Compute Euclidean distance if coordinates are provided
                w = math.dist(coords[u], coords[v])
            else:
                raise ValueError(f"Edge has only 2 elements and no coords provided: {edge}")
        else:
            raise ValueError(f"Edge has unexpected format: {edge}")
        
        D[u][v] = w
        D[v][u] = w

    return D, n


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

def solve_and_save(D, n, out_path, graph_name):
    
    result = {
        "graph": graph_name,
        "n": n,
        "solved": False
    }

    # if np.isinf(D).any():
    #     result["reason"] = "graph not fully connected after pruning"
    #     with open(out_path, "w") as f:
    #         json.dump(result, f, indent=2)
    #     return


    if n > 12:
        result["reason"] = "n > 12"
    else:
        start = time.time()
        try:
            cost, path = held_karp(D)
            result["solved"] = True
            result["optimal_cost"] = cost
            result["optimal_tour"] = path
            result["runtime_sec"] = time.time() - start
        except Exception as e:
            result["reason"] = str(e)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        

def main():
    graphs_chosen = "graphs_chosen"
    edgegnn_only = "savedGraphs_EdgeGNN_only"
    cascade_root = "savedGraphs"
    out_root = "exactSolver_results"
    os.makedirs(out_root, exist_ok=True)

    factors = [d for d in os.listdir(cascade_root) if d.startswith("factor_")]

    for fname in os.listdir(graphs_chosen):
        if not fname.endswith(".tsp"):
            continue
        tsp_path = os.path.join(graphs_chosen, fname)
        tsp = parse_tsp(tsp_path)
        coords = tsp["coords"]
        n = len(coords)

        D_full = build_distance_matrix(coords)

        # ---- FULL GRAPH ----
        full_dir = os.path.join(out_root, "full")
        os.makedirs(full_dir, exist_ok=True)
        solve_and_save(
            D_full, n,
            os.path.join(full_dir, fname.replace(".tsp", ".json")),
            fname
        )

        # ---- EdgeGNN-only ----
        pruned_path = os.path.join(edgegnn_only, fname.replace(".tsp", ".json"))
        if os.path.exists(pruned_path):
            Dp, _ = load_pruned_graph(pruned_path, coords)
            out_dir = os.path.join(out_root, "edgeGNN_only")
            os.makedirs(out_dir, exist_ok=True)
            solve_and_save(
                Dp, n,
                os.path.join(out_dir, fname.replace(".tsp", ".json")),
                fname
            )

        # ---- Cascade (per factor) ----
        for factor in factors:
            edgelist_path = os.path.join(
                cascade_root,
                factor,
                fname.replace(".tsp", ".edgelist")
            )

            if not os.path.exists(edgelist_path):
                continue

            # Load the graph as networkx
            G = nx.read_edgelist(edgelist_path, data=(('weight', float),), nodetype=int)
            for v in range(n):
                if v not in G:
                    G.add_node(v)
            # Convert to distance matrix for held_karp
            Dp = np.full((n, n), np.inf)
            np.fill_diagonal(Dp, 0)
            for u, v, d in G.edges(data=True):
                Dp[u, v] = d['weight']
                Dp[v, u] = d['weight']

            out_dir = os.path.join(out_root, "edgeGNN_cascade", factor)
            os.makedirs(out_dir, exist_ok=True)

            solve_and_save(
                Dp,
                n,
                os.path.join(out_dir, fname.replace(".tsp", ".json")),
                fname
            )



if __name__ == "__main__":
    main()
