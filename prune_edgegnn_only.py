import os
import json
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from torch_geometric.data import Data
import random

from newTrain import (
    EdgeGNN,
    parse_tsp,
    parse_opt_tour,
    build_candidate_edges,
    compute_node_edge_features
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# ----------- SAFETY GUARD HELPERS ----------------
# -------------------------------------------------

def enforce_topk_per_node(edges, probs, n, k):
    by_node = defaultdict(list)
    for (u, v), p in zip(edges, probs):
        by_node[u].append((p, (u, v)))
        by_node[v].append((p, (u, v)))

    keep = set()
    for v in range(n):
        top = sorted(by_node[v], reverse=True)[:k]
        for _, e in top:
            keep.add(tuple(sorted(e)))
    return keep


def enforce_mst(D):
    G = nx.Graph()
    n = D.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=D[i, j])
    T = nx.minimum_spanning_tree(G)
    return {tuple(sorted(e)) for e in T.edges()}


def enforce_mst_plus(D):
    return enforce_mst(D)


def nn_multistart_feasible(G, D, tries=10):
    n = G.number_of_nodes()
    starts = random.sample(list(G.nodes()), min(tries, n))

    for start in starts:
        visited = {start}
        cur = start

        while len(visited) < n:
            nbrs = [(D[cur][v], v) for v in G.neighbors(cur) if v not in visited]
            if not nbrs:
                break
            _, nxt = min(nbrs)
            visited.add(nxt)
            cur = nxt

        if len(visited) == n and G.has_edge(cur, start):
            return True

    return False


def repair_low_degree_nodes(G, D, min_deg):
    n = len(G.nodes())
    for v in G.nodes():
        while G.degree(v) < min_deg:
            candidates = sorted(
                [(D[v][u], u) for u in range(n) if u != v and not G.has_edge(v, u)]
            )
            if not candidates:
                break
            _, u = candidates[0]
            G.add_edge(v, u, weight=D[v][u])


def reconnect_graph_if_needed(G, D):
    if nx.is_connected(G):
        return G, False

    components = list(nx.connected_components(G))
    base = components[0]
    reconnected = False

    for comp in components[1:]:
        min_edge = None
        min_dist = float("inf")
        for u in base:
            for v in comp:
                if D[u][v] < min_dist:
                    min_dist = D[u][v]
                    min_edge = (u, v)
        if min_edge:
            G.add_edge(*min_edge, weight=D[min_edge[0]][min_edge[1]])
            base |= comp
            reconnected = True

    return G, reconnected

# -------------------------------------------------
# ---------------- PRUNING ------------------------
# -------------------------------------------------

def prune_tsp_instance_edgegnn(
    tsp_path,
    opt_tour_path,
    gnn,
    threshold
):
    tsp = parse_tsp(tsp_path)
    coords = tsp["coords"]
    n = coords.shape[0]

    opt_tour = parse_opt_tour(opt_tour_path)
    if opt_tour is None:
        return None

    edges, D = build_candidate_edges(coords)
    node_feat, edge_index_np, edge_attr_np = compute_node_edge_features(coords, edges, D)

    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float, device=device),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long, device=device),
        edge_attr=torch.tensor(edge_attr_np, dtype=torch.float, device=device),
    )

    with torch.no_grad():
        gnn_probs = torch.sigmoid(
            gnn(data.x, data.edge_index, data.edge_attr)
        ).cpu().numpy()

    thr = threshold

    if n <= 100:
        k_top = max(5, int(0.1 * n))
        min_deg = 2
        nearest_keep = 2
    elif n <= 500:
        k_top = max(10, int(0.1 * n))
        min_deg = 4
        nearest_keep = 6
        thr *= 0.8
    else:
        k_top = max(20, int(0.05 * n))
        min_deg = 4
        nearest_keep = 8
        thr *= 0.65

    protected_edges = set()
    for v in range(n):
        nbrs = sorted([(D[v][u], u) for u in range(n) if u != v])
        for i in range(min(nearest_keep, len(nbrs))):
            protected_edges.add(tuple(sorted((v, nbrs[i][1]))))

    pruned_edges = {
        tuple(sorted(edges[i]))
        for i in range(len(edges))
        if gnn_probs[i] >= thr
    }

    pruned_edges |= protected_edges
    pruned_edges |= enforce_topk_per_node(edges, gnn_probs, n, k_top)
    pruned_edges |= enforce_mst_plus(D)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in pruned_edges:
        G.add_edge(u, v, weight=D[u][v])

    G, reconnected = reconnect_graph_if_needed(G, D)
    repair_low_degree_nodes(G, D, min_deg)

    final_edges = list(G.edges())

    if n <= 55:
        feasible_solver = nn_multistart_feasible(G, D, tries=15)
    else:
        feasible_solver = nx.is_connected(G)

    opt_edges = {
        tuple(sorted((opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])))
        for i in range(len(opt_tour))
    }

    optimal_preserved = opt_edges.issubset(
        {tuple(sorted(e)) for e in final_edges}
    )

    return {
        "instance": os.path.basename(tsp_path),
        "n": n,
        "edges_kept": len(final_edges),
        "feasible_solver": feasible_solver,
        "optimal_preserved": optimal_preserved,
        "reconnected": reconnected,
        "edges": final_edges
    }

# -------------------------------------------------
# ---------------- MAIN ---------------------------
# -------------------------------------------------

def main():
    out_dir = "savedGraphs_EdgeGNN_only"
    os.makedirs(out_dir, exist_ok=True)

    with open("results/seed_42/best_threshold.json") as f:
        threshold = json.load(f)["threshold"]

    gnn = None

    for folder in ["tsplib_data", "synthetic_tsplib"]:
        for fname in os.listdir(folder):
            if not fname.endswith(".tsp"):
                continue

            tsp = os.path.join(folder, fname)
            opt = tsp.replace(".tsp", ".opt.tour")
            if not os.path.exists(opt):
                continue

            if gnn is None:
                dummy = parse_tsp(tsp)
                edges, D = build_candidate_edges(dummy["coords"])
                node_feat, _, edge_attr = compute_node_edge_features(dummy["coords"], edges, D)
                gnn = EdgeGNN(node_feat.shape[1], edge_attr.shape[1]).to(device)
                gnn.load_state_dict(
                    torch.load("results/seed_42/best_model.pt", map_location=device)
                )
                gnn.eval()

            print(f"[EdgeGNN-only pruning] {fname}")
            rep = prune_tsp_instance_edgegnn(tsp, opt, gnn, threshold)
            if rep is None:
                continue

            name = os.path.splitext(fname)[0]
            with open(f"{out_dir}/{name}.json", "w") as f:
                json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
