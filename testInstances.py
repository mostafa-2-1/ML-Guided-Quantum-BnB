# # import torch
# # import joblib
# # import numpy as np
# # from torch_geometric.data import Batch
# # from newTrain import EdgeGNN, build_pyg_data_from_instance
# # import json

# # # === Load trained GNN ===
# # device = torch.device('cpu')
# # gnn = EdgeGNN(in_node_feats=6, in_edge_feats=9, hidden_dim=128, n_layers=4, dropout=0.3)
# # gnn.load_state_dict(torch.load("results/best_model.pt", map_location=device))
# # gnn.eval()

# # # === Load Stage 2 MLP and thresholds ===
# # mlp_stage2 = joblib.load("results/cascade_stage2.pkl")
# # with open("results/best_threshold.json", "r") as f:
# #     base_metrics = json.load(f)
# # with open("results/cascade_stage2_thr.json", "r") as f:
# #     stage2_metrics = json.load(f)

# # thr2 = stage2_metrics["threshold"]

# # # Print saved metrics
# # print("=== Saved model metrics ===")
# # print(f"[Base GNN] Threshold={base_metrics['threshold']:.4f}, "
# #       f"Precision={base_metrics['precision']:.4f}, Recall={base_metrics['recall']:.4f}, F1={base_metrics['f1']:.4f}")
# # print(f"[Cascade Stage 2] Threshold={thr2:.4f}, "
# #       f"Precision={stage2_metrics.get('precision',0):.4f}, Recall={stage2_metrics.get('recall',0):.4f}, "
# #       f"F1={stage2_metrics.get('f1',0):.4f}")

# # # === Load TSP instance ===
# # tsp_path = "tsplib_data/berlin52.tsp"
# # tour_path = "tsplib_data/berlin52.opt.tour"  # for validation
# # data = build_pyg_data_from_instance(tsp_path, tour_path)
# # edges = np.array(data.edges_list)

# # # Build optimal edges set
# # with open(tour_path, "r") as f:
# #     lines = f.readlines()
# # tour_nodes = [int(l.strip()) - 1 for l in lines if l.strip().isdigit()]
# # optimal_edges = set()
# # for i in range(len(tour_nodes)):
# #     a = tour_nodes[i]
# #     b = tour_nodes[(i+1) % len(tour_nodes)]
# #     optimal_edges.add(tuple(sorted([a, b])))

# # # === GNN predictions ===
# # with torch.no_grad():
# #     b = Batch.from_data_list([data]).to(device)
# #     logits = gnn(b.x, b.edge_index, b.edge_attr)
# #     gnn_probs = torch.sigmoid(logits).cpu().numpy()

# # # === Stage 2 cascade ===
# # # Here we simulate Stage 1 as identity since only Stage 2 is critical
# # # X_stage1 = np.hstack([gnn_probs.reshape(-1, 1), data.edge_attr.cpu().numpy()])
# # # p_stage2 = mlp_stage2.predict_proba(np.hstack([X_stage1, mlp_stage2.predict_proba(X_stage1)[:,1].reshape(-1,1)]))[:, 1]



# # # === Stage 1 probabilities ===
# # mlp_stage1 = joblib.load("results/cascade_stage1.pkl")  # make sure Stage 1 MLP is loaded

# # X_stage1 = np.hstack([gnn_probs.reshape(-1, 1), data.edge_attr.cpu().numpy()])
# # p_stage1 = mlp_stage1.predict_proba(X_stage1)[:, 1]

# # # === Stage 2 probabilities ===
# # X_stage2 = np.hstack([X_stage1, p_stage1.reshape(-1, 1)])  # 11 features as Stage 2 expects
# # p_stage2 = mlp_stage2.predict_proba(X_stage2)[:, 1]



# # # === Test multiple adjusted thresholds for per-instance recall ===
# # factors = [0.55, 0.6, 0.7, 0.8, 0.9, 1.0]  # multiply thr2 by these
# # print("\n=== Per-instance threshold analysis ===")
# # for f in factors:
# #     thr_adj = thr2 * f
# #     final_preds = (p_stage2 >= thr_adj).astype(int)
# #     pruned_edges_set = set(tuple(sorted(e)) for e in edges[final_preds==1])
# #     survived = len(optimal_edges & pruned_edges_set)
# #     print(f"Factor {f:.2f} -> Threshold {thr_adj:.4f} | "
# #           f"Optimal edges preserved: {survived}/{len(optimal_edges)} "
# #           f"({survived/len(optimal_edges)*100:.2f}%) | Pruned edges: {np.sum(final_preds)}")

# # # === Choose one factor for final pruning (example: 0.8) ===
# # chosen_factor = 0.8
# # thr_final = thr2 * chosen_factor
# # final_preds = (p_stage2 >= thr_final).astype(int)
# # pruned_edges = edges[final_preds == 1]
# # print(f"\nFinal pruned graph using factor {chosen_factor}: {len(pruned_edges)} edges")








import os
import json
import torch
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score
from torch_geometric.data import Data


from newTrain import EdgeGNN
from newTrain import parse_tsp, parse_opt_tour, build_candidate_edges, compute_node_edge_features
# assign cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import json

def visualize_pruning(coords, edges_full, edges_pruned, opt_tour=None, save_path=None):
    """
    Visualize TSP instance before/after pruning.
    - coords: np.ndarray (n,2)
    - edges_full: list of (i,j)
    - edges_pruned: list of (i,j)
    - opt_tour: optional list of node indices (to overlay true tour)
    """
    plt.figure(figsize=(10,5))

    # --- Left: full graph ---
    plt.subplot(1,2,1)
    G_full = nx.Graph()
    G_full.add_nodes_from(range(len(coords)))
    G_full.add_edges_from(edges_full)
    pos = {i: (coords[i,0], coords[i,1]) for i in range(len(coords))}
    nx.draw(G_full, pos, node_size=80, node_color='skyblue', edge_color='lightgray', with_labels=False)
    plt.title(f"Full graph ({len(edges_full)} edges)")

    # --- Right: pruned graph ---
    plt.subplot(1,2,2)
    G_pruned = nx.Graph()
    G_pruned.add_nodes_from(range(len(coords)))
    G_pruned.add_edges_from(edges_pruned)
    nx.draw(G_pruned, pos, node_size=80, node_color='lightgreen', edge_color='darkgreen', with_labels=False)

    #  draw true tour edges (if opt_tour given)
    if opt_tour is not None:
        tour_edges = [(opt_tour[i], opt_tour[(i+1)%len(opt_tour)]) for i in range(len(opt_tour))]
        nx.draw_networkx_edges(G_pruned, pos, edgelist=tour_edges, edge_color='red', width=2.0, style='dashed')

    plt.title(f"Pruned graph ({len(edges_pruned)} edges)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()



def prune_tsp_instance(
    tsp_path,
    opt_tour_path,
    model_path="results/best_model_threshold.pt",
    cascade_path="results/cascade_stage1.pkl",
    cascade_thr_path="results/cascade_stage1_thr.json",
    save_json=True,
):
    """Prune a TSP instance using EdgeGNN + cascade stage1.
    Returns: pruned_edges (list of (u,v)), precision, recall
    """
    # --- 1) parse tsp and get coords (parse_tsp returns dict with 'coords') ---
    tsp_data = parse_tsp(tsp_path)         # returns dict {'name', 'n', 'coords'}
    coords = tsp_data["coords"]            # numpy array (n,2)
    n = coords.shape[0]
    # --- 2) parse optimal tour ---
    opt_tour = parse_opt_tour(opt_tour_path)   # list of 0-based node indices

    # --- 3) build candidate edges and pairwise distances ---
    edges, D = build_candidate_edges(coords)   # edges: list[(i,j)], D: numpy (n,n)

    # --- 4) compute features: node_feat, edge_index_np (2,m), edge_attr_np (m,f) ---
    node_feat, edge_index_np, edge_attr_np = compute_node_edge_features(coords, edges, D)
    # ensure types
    edge_index_t = torch.tensor(edge_index_np, dtype=torch.long).to(device)   # (2,m)
    x_t = torch.tensor(node_feat, dtype=torch.float).to(device)
    edge_attr_t = torch.tensor(edge_attr_np, dtype=torch.float).to(device)

    # --- 5) Build PyG Data object
    data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t).to(device)

    # --- 6) Load EdgeGNN 
    in_node_feats = node_feat.shape[1]
    in_edge_feats = edge_attr_np.shape[1]
    gnn = EdgeGNN(in_node_feats, in_edge_feats).to(device)
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()

    # --- 7) GNN inference: get per-edge logits -> probs ---
    with torch.no_grad():
        logits = gnn(data.x, data.edge_index, data.edge_attr)   # shape (m,)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1) # shape (m,)

    # --- 8) Build cascade input features exactly like build_cascade_dataset did: [p, *edge_attr] ---
    #  build_cascade_dataset used feat = [p] + edge_attrs[idx].tolist()
    X_cascade = np.hstack([probs.reshape(-1, 1), edge_attr_np])  # shape (m, 1 + f_edge)

    # --- 9) Load cascade Stage 1 and threshold ---
    cascade = joblib.load(cascade_path)
    with open(cascade_thr_path, "r") as f:
        thr1 = json.load(f)["threshold"]*0.3#dynamically adjust for different graph sizes, so we can favor recall over precision

    cascade_probs = cascade.predict_proba(X_cascade)[:, 1]
    pruned_mask = (cascade_probs >= thr1)
    pruned_edges = [tuple(int(x) for x in edges[i]) for i in range(len(edges)) if pruned_mask[i]]

    # --- 10) Evaluate pruning vs .opt.tour ---
    # build set of undirected edges from opt tour (0-based)
    opt_edges_set = set()
    if opt_tour:
        # ensure opt_tour is list of 0-based ints; your parse returns 0-based already
        tour = opt_tour
        for a, b in zip(tour, tour[1:] + [tour[0]]):
            opt_edges_set.add(tuple(sorted((int(a), int(b)))))

    y_true = np.array([1 if tuple(sorted(e)) in opt_edges_set else 0 for e in edges], dtype=int)
    y_pred = pruned_mask.astype(int)

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    print(f"Pruned edges: kept {len(pruned_edges)} / {len(edges)} edges "
          f"({100.0 * len(pruned_edges) / max(1, len(edges)):.1f}%)")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}")

    # --- 11) Optionally save pruned graph ---
    out_dir = "savedGraphs"
    os.makedirs(out_dir, exist_ok=True)
    if save_json:
        base_name = os.path.basename(tsp_path)           
        name_no_ext = os.path.splitext(base_name)[0]     
        save_path = os.path.join(out_dir, f"{name_no_ext}_pruned.json")
        save_pruned_graph(save_path, coords, pruned_edges)
        print(f"Saved pruned graph to {save_path}")

    viz_path = os.path.join(out_dir, f"{name_no_ext}_viz.png")
    visualize_pruning(coords, edges, pruned_edges, opt_tour=opt_tour, save_path=viz_path)


    return pruned_edges, prec, rec


def save_pruned_graph(path, coords, pruned_edges, instance_file=None):
    """Save pruned graph as JSON, including distance matrix."""
    coords_arr = np.array(coords)
    n = coords_arr.shape[0]
    # Distance matrix
    D = np.linalg.norm(coords_arr[:, None, :] - coords_arr[None, :, :], axis=-1)
    data = {
        "name": os.path.basename(path),
        "num_nodes": n,
        "edges": pruned_edges,
        "distance_matrix": D.tolist(),
        "instance_file": instance_file  # keep track of original .tsp
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    tsp_file = "tsplib_data/berlin52.tsp"
    opt_tour_file = "tsplib_data/berlin52.opt.tour"

    # tsp_file = "synthetic_tsplib/synthetic_35.tsp"
    # opt_tour_file = "synthetic_tsplib/synthetic_35.opt.tour"

    edges_pruned, prec, rec = prune_tsp_instance(tsp_file, opt_tour_file)


if __name__ == "__main__":
    main()