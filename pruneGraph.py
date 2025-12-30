
# import os
# import json
# import torch
# import joblib
# import numpy as np
# import math
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.metrics import precision_score, recall_score
# from torch_geometric.data import Data

# from newTrain import EdgeGNN
# from newTrain import parse_tsp, parse_opt_tour, build_candidate_edges, compute_node_edge_features

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -------------------------------------------------
# # ----------- SAFETY GUARD HELPERS ----------------
# # -------------------------------------------------

# def enforce_topk_per_node(edges, probs, n, k):
#     by_node = defaultdict(list)
#     for (u, v), p in zip(edges, probs):
#         by_node[u].append((p, (u, v)))
#         by_node[v].append((p, (u, v)))

#     keep = set()
#     for v in range(n):
#         top = sorted(by_node[v], reverse=True)[:k]
#         for _, e in top:
#             keep.add(tuple(sorted(e)))
#     return keep



# def enforce_mst_plus(D):
#     return enforce_mst(D)

# import random

# def nn_multistart_feasible(G, D, tries=10):
#     n = G.number_of_nodes()
#     starts = random.sample(list(G.nodes()), min(tries, n))

#     for start in starts:
#         visited = {start}
#         tour = [start]
#         cur = start

#         while len(visited) < n:
#             nbrs = [(D[cur][v], v) for v in G.neighbors(cur) if v not in visited]
#             if not nbrs:
#                 break
#             _, nxt = min(nbrs)
#             visited.add(nxt)
#             tour.append(nxt)
#             cur = nxt

#         if len(visited) == n and tour[0] in G.neighbors(tour[-1]):
#             return True

#     return False




# def repair_low_degree_nodes(G, D, min_deg=2):
#     n = len(G.nodes())
#     for v in G.nodes():
#         while G.degree(v) < min_deg:
#             candidates = sorted(
#                 [(D[v][u], u) for u in range(n) if u != v and not G.has_edge(v, u)]
#             )
#             if not candidates:
#                 break
#             _, u = candidates[0]
#             G.add_edge(v, u, weight=D[v][u])



# def enforce_mst(D):
#     G = nx.Graph()
#     n = D.shape[0]
#     for i in range(n):
#         for j in range(i + 1, n):
#             G.add_edge(i, j, weight=D[i, j])
#     T = nx.minimum_spanning_tree(G)
#     return {tuple(sorted(e)) for e in T.edges()}


# # -------------------------------------------------
# # -------- FEASIBILITY & SOLVER -------------------
# # -------------------------------------------------

# def reconnect_graph_if_needed(G, D):
#     if nx.is_connected(G):
#         return G, False

#     components = list(nx.connected_components(G))
#     base = components[0]
#     reconnected = False

#     for comp in components[1:]:
#         min_edge = None
#         min_dist = float("inf")
#         for u in base:
#             for v in comp:
#                 if D[u][v] < min_dist:
#                     min_dist = D[u][v]
#                     min_edge = (u, v)
#         if min_edge:
#             G.add_edge(*min_edge, weight=D[min_edge[0]][min_edge[1]])
#             base |= comp
#             reconnected = True

#     return G, reconnected


# def nearest_neighbor_tsp(G, D):
#     n = len(G.nodes())
#     visited = {0}
#     tour = [0]
#     cur = 0

#     while len(visited) < n:
#         nbrs = [(D[cur][v], v) for v in G.neighbors(cur) if v not in visited]
#         if not nbrs:
#             return None
#         _, nxt = min(nbrs)
#         visited.add(nxt)
#         tour.append(nxt)
#         cur = nxt

#     if tour[0] not in G.neighbors(tour[-1]):
#         return None
#     return tour


# # -------------------------------------------------
# # ---------------- PRUNING ------------------------
# # -------------------------------------------------

# # ---------------- ENHANCED PRUNING -------------------

# def prune_tsp_instance(
#     tsp_path,
#     opt_tour_path,
#     gnn,
#     cascade,
#     base_threshold,
#     factor_for_range,
#     save_json=True,
# ):
#     # ---------------- Parse instance ----------------
#     tsp_data = parse_tsp(tsp_path)
#     coords = tsp_data["coords"]
#     n = coords.shape[0]

#     opt_tour = parse_opt_tour(opt_tour_path)  # evaluation only
#     if opt_tour is None:
#         return None

#     edges, D = build_candidate_edges(coords)
#     node_feat, edge_index_np, edge_attr_np = compute_node_edge_features(coords, edges, D)

#     data = Data(
#         x=torch.tensor(node_feat, dtype=torch.float, device=device),
#         edge_index=torch.tensor(edge_index_np, dtype=torch.long, device=device),
#         edge_attr=torch.tensor(edge_attr_np, dtype=torch.float, device=device),
#     )

#     # ---------------- Inference ----------------
#     with torch.no_grad():
#         gnn_probs = torch.sigmoid(
#             gnn(data.x, data.edge_index, data.edge_attr)
#         ).cpu().numpy()

#     X_cascade = np.hstack([gnn_probs.reshape(-1, 1), edge_attr_np])
#     cascade_probs = cascade.predict_proba(X_cascade)[:, 1]

#     # ---------------- Size-aware pruning threshold ----------------
#     thr = base_threshold * factor_for_range(n)
#     if n <= 100:
#         k_top = max(5, int(0.1 * n))
#         min_deg = 2
#         nearest_keep = 2
#     elif n <= 500:
#         k_top = max(10, int(0.1 * n))
#         min_deg = 4
#         nearest_keep = 6
#         thr *=0.8
#     else:
#         k_top = max(20, int(0.05 * n))
#         min_deg = 4
#         nearest_keep = 8
#         thr *=0.65

#     protected_edges = set()

#     for v in range(n):
#         neighbors = sorted([(D[v][u], u) for u in range(n) if u != v])
#         for i in range(min(nearest_keep, len(neighbors))):
#             protected_edges.add(tuple(sorted((v, neighbors[i][1]))))

#     # ---------------- Core pruning ----------------
#     pruned_edges = {
#         tuple(sorted(edges[i]))
#         for i in range(len(edges))
#         if cascade_probs[i] >= thr
#     }
#     pruned_edges |= protected_edges
#     # ---------------- Safety guards ----------------
#     # 1. Top-k per node
#     pruned_edges |= enforce_topk_per_node(edges, cascade_probs, n, k_top)

#     # 2. MST edges
#     pruned_edges |= enforce_mst_plus(D)

   

#     # ---------------- Build graph ----------------
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     for u, v in pruned_edges:
#         G.add_edge(u, v, weight=D[u][v])

#     # ---------------- Structural repair ----------------
#     G, reconnected = reconnect_graph_if_needed(G, D)
#     repair_low_degree_nodes(G, D, min_deg=min_deg)

#     final_edges = list(G.edges())

#     # ---------------- Final feasibility checks ----------------
#     degree_ok = all(G.degree(v) >= min_deg for v in G.nodes())
#     connected = nx.is_connected(G)

#     if n<=55:
#         feasible_solver = nn_multistart_feasible(G, D, tries=15)
#     else:
#         feasible_solver = connected and all(G.degree(v) >= min_deg for v in G.nodes())

#     # ---------------- Optimal preservation ----------------
#     opt_edges = {
#         tuple(sorted((opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])))
#         for i in range(len(opt_tour))
#     }
#     optimal_preserved = opt_edges.issubset(
#         set(tuple(sorted(e)) for e in final_edges)
#     )

#     # ---------------- Report ----------------
#     report = {
#         "instance": os.path.basename(tsp_path),
#         "n": n,
#         "edges_kept": len(final_edges),
#         "degree_ok": degree_ok,
#         "connected": connected,
#         "reconnected": reconnected,
#         "feasible_solver": feasible_solver,
#         "optimal_preserved": optimal_preserved,
#     }

#     if save_json:
#         os.makedirs("savedGraphs", exist_ok=True)
#         name = os.path.splitext(os.path.basename(tsp_path))[0]
#         with open(f"savedGraphs/{name}_report.json", "w") as f:
#             json.dump(report, f, indent=2)

#     return report


# # ---------------- PROCESS FOLDERS WITH SIZE BUCKETS ----------------

# def process_folders(
#     folders,
#     model_path,
#     cascade_path,
#     cascade_thr_path,
#     factor_json_path,
# ):
#     with open(cascade_thr_path) as f:
#         base_threshold = json.load(f)["threshold"]

#     with open(factor_json_path) as f:
#         factor_raw = json.load(f)

#     def factor_for_range(n):
#         for k, v in factor_raw.items():
#             a, b = map(int, k.split("-"))
#             if a <= n <= b:
#                 return v[0]["factor"]
#         return 1.0

#     cascade = joblib.load(cascade_path)
#     gnn = None
#     all_reports = []

#     for folder in folders:
#         for fname in os.listdir(folder):
#             if not fname.endswith(".tsp"):
#                 continue

#             tsp = os.path.join(folder, fname)
#             opt = tsp.replace(".tsp", ".opt.tour")
#             if not os.path.exists(opt):
#                 continue

#             if gnn is None:
#                 dummy = parse_tsp(tsp)
#                 edges, D = build_candidate_edges(dummy["coords"])
#                 node_feat, _, edge_attr = compute_node_edge_features(dummy["coords"], edges, D)
#                 gnn = EdgeGNN(node_feat.shape[1], edge_attr.shape[1]).to(device)
#                 gnn.load_state_dict(torch.load(model_path, map_location=device))
#                 gnn.eval()

#             print(f"[Pruning] {fname}")
#             rep = prune_tsp_instance(
#                 tsp, opt, gnn, cascade, base_threshold, factor_for_range
#             )
#             if rep:
#                 all_reports.append(rep)

#     # ---- Aggregate summary by size ----
#     size_buckets = {
#         "small": {"range": "<=100", "reports": []},
#         "medium": {"range": "101-500", "reports": []},
#         "large": {"range": ">500", "reports": []},
#     }

#     for r in all_reports:
#         if r["n"] <= 100:
#             size_buckets["small"]["reports"].append(r)
#         elif r["n"] <= 500:
#             size_buckets["medium"]["reports"].append(r)
#         else:
#             size_buckets["large"]["reports"].append(r)

#     summary = {}
#     for key, info in size_buckets.items():
#         reps = info["reports"]
#         if reps:
#             summary[key] = {
#                 "instances": len(reps),
#                 "feasibility_rate": np.mean([r["feasible_solver"] for r in reps]),
#                 "optimal_preservation_rate": np.mean([r["optimal_preserved"] for r in reps]),
#                 "avg_edges_kept": np.mean([r["edges_kept"] for r in reps]),
#             }
#         else:
#             summary[key] = {
#                 "instances": 0,
#                 "feasibility_rate": None,
#                 "optimal_preservation_rate": None,
#                 "avg_edges_kept": None,
#             }

#     with open("savedGraphs/summary.json", "w") as f:
#         json.dump(summary, f, indent=2)

#     print("\n=== PRUNING SUMMARY BY SIZE ===")
#     print(json.dumps(summary, indent=2))



# def main():
#     # process_folders(
#     #     folders=["tsplib_data", "synthetic_tsplib"],
#     #     model_path="results/best_model_threshold.pt",
#     #     cascade_path="results/cascade_stage1.pkl",
#     #     cascade_thr_path="results/cascade_stage1_thr.json",
#     #     factor_json_path="results/full_factor_metrics.json",
#     # )
#     process_folders(
#         folders=["tsplib_data", "synthetic_tsplib"],
#         model_path="results/seed_42/best_model_threshold.pt",
#         cascade_path="results/seed_42/cascade_stage1.pkl",
#         cascade_thr_path="results/seed_42/cascade_stage1_thr.json",
#         factor_json_path="results/full_factor_metrics.json",
#     )
    


# if __name__ == "__main__":
#     main()








import os
import json
import torch
import joblib
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
from torch_geometric.data import Data

from newTrain import EdgeGNN
from newTrain import parse_tsp, parse_opt_tour, build_candidate_edges, compute_node_edge_features

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



def enforce_mst_plus(D):
    return enforce_mst(D)

import random

def nn_multistart_feasible(G, D, tries=10):
    n = G.number_of_nodes()
    starts = random.sample(list(G.nodes()), min(tries, n))

    for start in starts:
        visited = {start}
        tour = [start]
        cur = start

        while len(visited) < n:
            nbrs = [(D[cur][v], v) for v in G.neighbors(cur) if v not in visited]
            if not nbrs:
                break
            _, nxt = min(nbrs)
            visited.add(nxt)
            tour.append(nxt)
            cur = nxt

        if len(visited) == n and tour[0] in G.neighbors(tour[-1]):
            return True

    return False




def repair_low_degree_nodes(G, D, min_deg=2):
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



def enforce_mst(D):
    G = nx.Graph()
    n = D.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=D[i, j])
    T = nx.minimum_spanning_tree(G)
    return {tuple(sorted(e)) for e in T.edges()}


# -------------------------------------------------
# -------- FEASIBILITY & SOLVER -------------------
# -------------------------------------------------

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


def nearest_neighbor_tsp(G, D):
    n = len(G.nodes())
    visited = {0}
    tour = [0]
    cur = 0

    while len(visited) < n:
        nbrs = [(D[cur][v], v) for v in G.neighbors(cur) if v not in visited]
        if not nbrs:
            return None
        _, nxt = min(nbrs)
        visited.add(nxt)
        tour.append(nxt)
        cur = nxt

    if tour[0] not in G.neighbors(tour[-1]):
        return None
    return tour


# -------------------------------------------------
# ---------------- PRUNING ------------------------
# -------------------------------------------------

# ---------------- ENHANCED PRUNING -------------------


def prune_tsp_instance(
    tsp_path,
    opt_tour_path,
    gnn,
    cascade,
    base_threshold,
    factor,
    out_dir,
):
    tsp_data = parse_tsp(tsp_path)
    coords = tsp_data["coords"]
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

    X_cascade = np.hstack([gnn_probs.reshape(-1, 1), edge_attr_np])
    cascade_probs = cascade.predict_proba(X_cascade)[:, 1]

    thr = base_threshold * factor

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
        neighbors = sorted([(D[v][u], u) for u in range(n) if u != v])
        for i in range(min(nearest_keep, len(neighbors))):
            protected_edges.add(tuple(sorted((v, neighbors[i][1]))))

    pruned_edges = {
        tuple(sorted(edges[i]))
        for i in range(len(edges))
        if cascade_probs[i] >= thr
    }
    pruned_edges |= protected_edges
    pruned_edges |= enforce_topk_per_node(edges, cascade_probs, n, k_top)
    pruned_edges |= enforce_mst_plus(D)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in pruned_edges:
        G.add_edge(u, v, weight=D[u][v])

    # 1️⃣ Initial reconnect
    G, _ = reconnect_graph_if_needed(G, D)

    # 2️⃣ Fix degrees (this may create new isolated structures)
    repair_low_degree_nodes(G, D, min_deg=min_deg)

    # 3️⃣ FINAL reconnect — THIS IS THE CRITICAL ONE
    G, _ = reconnect_graph_if_needed(G, D)

    feasible = nx.is_connected(G)


    opt_edges = {
        tuple(sorted((opt_tour[i], opt_tour[(i + 1) % len(opt_tour)])))
        for i in range(len(opt_tour))
    }
    optimal_preserved = opt_edges.issubset(set(map(tuple, map(sorted, G.edges()))))

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(tsp_path))[0]
    assert nx.is_connected(G), f"Disconnected graph after pruning: {tsp_path}"


    nx.write_edgelist(G, f"{out_dir}/{name}.edgelist", data=["weight"])

    report = {
        "instance": name,
        "n": n,
        "factor": factor,
        "edges_kept": G.number_of_edges(),
        "feasible": feasible,
        "optimal_preserved": optimal_preserved,
    }

    with open(f"{out_dir}/{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


# -------------------------------------------------
# ---------------- PROCESS ------------------------
# -------------------------------------------------

def select_factors_for_range(entries):
    factors = [e["factor"] for e in entries]
    n = len(factors)

    if n <= 4:
        return factors

    idxs = sorted(set([
        0,
        n // 4,
        n // 2,
        (3 * n) // 4,
        n - 1
    ]))
    return [factors[i] for i in idxs]


def process_folders(
    folders,
    model_path,
    cascade_path,
    cascade_thr_path,
    factor_json_path,
):
    with open(cascade_thr_path) as f:
        base_threshold = json.load(f)["threshold"]

    with open(factor_json_path) as f:
        factor_raw = json.load(f)

    factor_map = {}
    for k, v in factor_raw.items():
        factor_map[k] = select_factors_for_range(v)

    cascade = joblib.load(cascade_path)
    gnn = None

    for folder in folders:
        for fname in os.listdir(folder):
            if not fname.endswith(".tsp"):
                continue

            tsp = os.path.join(folder, fname)
            opt = tsp.replace(".tsp", ".opt.tour")
            if not os.path.exists(opt):
                continue

            tsp_data = parse_tsp(tsp)
            n = tsp_data["coords"].shape[0]

            for k, factors in factor_map.items():
                a, b = map(int, k.split("-"))
                if not (a <= n <= b):
                    continue

                if gnn is None:
                    edges, D = build_candidate_edges(tsp_data["coords"])
                    node_feat, _, edge_attr = compute_node_edge_features(
                        tsp_data["coords"], edges, D
                    )
                    gnn = EdgeGNN(node_feat.shape[1], edge_attr.shape[1]).to(device)
                    gnn.load_state_dict(torch.load(model_path, map_location=device))
                    gnn.eval()

                for factor in factors:
                    out_dir = f"savedGraphs/factor_{factor:.5f}"
                    print(f"[Prune] {fname} | factor={factor:.5f}")
                    prune_tsp_instance(
                        tsp,
                        opt,
                        gnn,
                        cascade,
                        base_threshold,
                        factor,
                        out_dir,
                    )

def process_graphs_chosen(
    graphs_chosen_dir,
    model_path,
    cascade_path,
    cascade_thr_path,
    factor_json_path,
):
    with open(cascade_thr_path) as f:
        base_threshold = json.load(f)["threshold"]

    with open(factor_json_path) as f:
        factor_raw = json.load(f)

    factor_map = {
        k: select_factors_for_range(v)
        for k, v in factor_raw.items()
    }

    cascade = joblib.load(cascade_path)
    gnn = None

    for fname in os.listdir(graphs_chosen_dir):
        if not fname.endswith(".tsp"):
            continue

        tsp = os.path.join(graphs_chosen_dir, fname)
        opt_candidates = [
            os.path.join("synthetic_tsplib", os.path.basename(fname).replace(".tsp",".opt.tour")),
            os.path.join("tsplib_data", os.path.basename(fname).replace(".tsp",".opt.tour"))
        ]

        opt = next((f for f in opt_candidates if os.path.exists(f)), None)
        if opt is None:
            print(f"[Skip] No opt tour for {fname}")
            continue

        tsp_data = parse_tsp(tsp)
        coords = tsp_data["coords"]
        n = coords.shape[0]

        # pick correct factor range
        selected_factors = None
        for k, factors in factor_map.items():
            a, b = map(int, k.split("-"))
            if a <= n <= b:
                selected_factors = factors
                break

        if selected_factors is None:
            print(f"[Skip] No factor range for n={n} ({fname})")
            continue

        # lazy-load GNN ONCE
        if gnn is None:
            edges, D = build_candidate_edges(coords)
            node_feat, _, edge_attr = compute_node_edge_features(coords, edges, D)
            gnn = EdgeGNN(node_feat.shape[1], edge_attr.shape[1]).to(device)
            gnn.load_state_dict(torch.load(model_path, map_location=device))
            gnn.eval()

        for factor in selected_factors:
            out_dir = f"savedGraphs/factor_{factor:.5f}"
            print(f"[Prune] {fname} | n={n} | factor={factor:.5f}")

            prune_tsp_instance(
                tsp,
                opt,
                gnn,
                cascade,
                base_threshold,
                factor,
                out_dir,
            )



def main():
    # process_folders(
    #     folders=["tsplib_data", "synthetic_tsplib"],
    #     model_path="results/seed_42/best_model_threshold.pt",
    #     cascade_path="results/seed_42/cascade_stage1.pkl",
    #     cascade_thr_path="results/seed_42/cascade_stage1_thr.json",
    #     factor_json_path="results/full_factor_metrics.json",
    # )
    process_graphs_chosen(
        graphs_chosen_dir="graphs_chosen",
        model_path="results/seed_42/best_model_threshold.pt",
        cascade_path="results/seed_42/cascade_stage1.pkl",
        cascade_thr_path="results/seed_42/cascade_stage1_thr.json",
        factor_json_path="results/full_factor_metrics.json",
    )


if __name__ == "__main__":
    main()