import os
import sys
import time
import json
import numpy as np
import networkx as nx
from itertools import combinations

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from newTrain import parse_tsp
from preprocess_data import build_candidate_edges

GRAPH_DIR = "graphs_chosen"
EDGEGNN_ONLY = "savedGraphs_EdgeGNN_only"
CASCADE_ROOT = "savedGraphs"
RESULTS_DIR = "heuristicSolver_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

BEAM_WIDTH = 10
START_NODE = 0


def build_full_graph(coords):
    edges, D = build_candidate_edges(coords)
    G = nx.Graph()
    n = coords.shape[0]
    G.add_nodes_from(range(n))
    for u, v in edges:
        G.add_edge(u, v, weight=D[u][v])
    return G, D


def beam_search_tsp(G, D, beam_width):
    n = G.number_of_nodes()
    start = START_NODE

    beam = [{
        "node": start,
        "visited": frozenset([start]),
        "path": [start],
        "cost": 0.0
    }]

    for _ in range(n - 1):
        candidates = []
        for state in beam:
            u = state["node"]
            for v in G.neighbors(u):
                if v in state["visited"]:
                    continue
                candidates.append({
                    "node": v,
                    "visited": state["visited"] | {v},
                    "path": state["path"] + [v],
                    "cost": state["cost"] + D[u][v]
                })
        if not candidates:
            return None
        candidates.sort(key=lambda x: x["cost"])
        beam = candidates[:beam_width]

    # Close the tour
    completed = []
    for state in beam:
        u = state["node"]
        if G.has_edge(u, start):
            completed.append({
                "path": state["path"] + [start],
                "cost": state["cost"] + D[u][start]
            })
    if not completed:
        return None
    best = min(completed, key=lambda x: x["cost"])
    return best["path"], best["cost"]


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
            w = float(np.linalg.norm(coords[u]-coords[v])) if coords is not None else 0
        else:
            raise ValueError(f"Unexpected edge format: {edge}")
        D[u, v] = w
        D[v, u] = w
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(n):
            if D[u,v] < np.inf:
                G.add_edge(u, v, weight=D[u,v])
    return G, D, n


def solve_and_save(G, D, n, out_path, graph_name):
    t0 = time.time()
    result = {
        "graph": graph_name,
        "n": n,
        "beam_width": BEAM_WIDTH,
        "solved": False
    }
    try:
        res = beam_search_tsp(G, D, BEAM_WIDTH)
        if res is None:
            result["feasible"] = False
        else:
            path, cost = res
            result["solved"] = True
            result["feasible"] = True
            result["tour_cost"] = float(cost)
            result["tour"] = path
        result["runtime_sec"] = time.time() - t0
    except Exception as e:
        result["error"] = str(e)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    factors = [d for d in os.listdir(CASCADE_ROOT) if d.startswith("factor_")]

    for fname in os.listdir(GRAPH_DIR):
        if not fname.endswith(".tsp"):
            continue
        tsp_path = os.path.join(GRAPH_DIR, fname)
        tsp = parse_tsp(tsp_path)
        coords = tsp["coords"]
        n = coords.shape[0]

        # --- FULL GRAPH ---
        G_full, D_full = build_full_graph(coords)
        full_dir = os.path.join(RESULTS_DIR, "full")
        os.makedirs(full_dir, exist_ok=True)
        solve_and_save(G_full, D_full, n, os.path.join(full_dir, fname.replace(".tsp", ".json")), fname)

        # --- EdgeGNN-only ---
        pruned_path = os.path.join(EDGEGNN_ONLY, fname.replace(".tsp", ".json"))
        if os.path.exists(pruned_path):
            G_p, D_p, n_p = load_pruned_graph(pruned_path, coords)
            out_dir = os.path.join(RESULTS_DIR, "edgeGNN_only")
            os.makedirs(out_dir, exist_ok=True)
            solve_and_save(G_p, D_p, n_p, os.path.join(out_dir, fname.replace(".tsp", ".json")), fname)

        # --- Cascade per factor ---
        for factor in factors:
            edgelist_path = os.path.join(CASCADE_ROOT, factor, fname.replace(".tsp", ".edgelist"))
            if not os.path.exists(edgelist_path):
                continue
            G = nx.read_edgelist(edgelist_path, data=(('weight', float),), nodetype=int)
            for v in range(n):
                if v not in G:
                    G.add_node(v)
            D = np.full((n, n), np.inf)
            np.fill_diagonal(D, 0)
            for u, v, d in G.edges(data=True):
                D[u,v] = d['weight']
                D[v,u] = d['weight']
            out_dir = os.path.join(RESULTS_DIR, "edgeGNN_cascade", factor)
            os.makedirs(out_dir, exist_ok=True)
            solve_and_save(G, D, n, os.path.join(out_dir, fname.replace(".tsp", ".json")), fname)


if __name__ == "__main__":
    main()
