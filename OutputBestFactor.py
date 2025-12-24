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





"""
learn_factors.py

This script implements Option B (fast pipeline) to learn a continuous threshold factor
for fixed graph-size ranges. It caches per-instance cascade probabilities so the
factor search only performs cheap vectorized thresholding and metric computation.

Outputs:
 - results/dynamic_factor_table.json  -> mapping range_id -> chosen factor
 - results/per_instance_factors.json -> raw per-graph best factor and metrics
 - cache_probs/<instance_name>.npz    -> cached arrays: cascade_probs, edges, y_true, base_threshold

Usage:
 python learn_factors.py --tsplib_dirs tsplib_data synthetic_tsplib --cascade results/cascade_stage1.pkl \
     --cascade_thr results/cascade_stage1_thr.json --model results/best_model_threshold.pt

Notes:
 - This assumes `newTrain` provides the following functions/classes:
    parse_tsp, parse_opt_tour, build_candidate_edges, compute_node_edge_features, EdgeGNN
 - Adjust paths in the CLI args if your project layout differs.
 - The script performs a dense search over factors = np.linspace(0.02, 1.0, 500).
 - For each graph it selects factors with recall >= 0.98; if none, falls back to >=0.95.
   From eligible factors it picks the one with the highest precision. Tie-break: smaller factor.
 - It aggregates per-range final factor by taking the median of per-instance chosen factors.

"""

import os
import argparse
import json
import glob
from collections import defaultdict

import numpy as np
import joblib
import torch
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from newTrain import parse_tsp, parse_opt_tour, build_candidate_edges, compute_node_edge_features, EdgeGNN

#output the best factor for each range
def get_range_id(n):
    # ranges as requested
    if n <= 20:
        return "1-20"
    if n <= 50:
        return "21-50"
    if n <= 100:
        return "51-100"
    if n <= 200:
        return "101-200"
    if n <= 500:
        return "201-500"
    if n <= 1000:
        return "501-1000"
    if n <= 2000:
        return "1001-2000"
    return "2001+"


def cache_exists(cache_dir, instance_name):
    return os.path.exists(os.path.join(cache_dir, instance_name + ".npz"))


def save_cache(cache_dir, instance_name, cascade_probs, edges, y_true, base_threshold):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, instance_name + ".npz")
    np.savez_compressed(path, cascade_probs=cascade_probs, edges=np.array(edges), y_true=y_true, base_threshold=base_threshold)


def load_cache(cache_dir, instance_name):
    path = os.path.join(cache_dir, instance_name + ".npz")
    data = np.load(path)
    return data['cascade_probs'], data['edges'], data['y_true'], float(data['base_threshold'])


def compute_and_cache_probs(tsp_path, opt_path, model_path, cascade_path, cascade_thr_path, device, cache_dir):
    """Run GNN + cascade once for this instance and cache: cascade_probs, edges, y_true, base_threshold"""
    # parse
    tsp = parse_tsp(tsp_path)
    coords = tsp['coords']
    n = coords.shape[0]
    instance_name = os.path.splitext(os.path.basename(tsp_path))[0]

    opt_tour = None
    if opt_path and os.path.exists(opt_path):
        opt_tour = parse_opt_tour(opt_path)

    # build candidate edges and features
    edges, D = build_candidate_edges(coords)
    node_feat, edge_index_np, edge_attr_np = compute_node_edge_features(coords, edges, D)

    # Prepare tensors
    x_t = torch.tensor(node_feat, dtype=torch.float).to(device)
    edge_index_t = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    edge_attr_t = torch.tensor(edge_attr_np, dtype=torch.float).to(device)

    # load gnn
    in_node_feats = node_feat.shape[1]
    in_edge_feats = edge_attr_np.shape[1]
    gnn = EdgeGNN(in_node_feats, in_edge_feats).to(device)
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()

    with torch.no_grad():
        logits = gnn(x_t, edge_index_t, edge_attr_t)
        gnn_probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # cascade X input
    X_cascade = np.hstack([gnn_probs.reshape(-1,1), edge_attr_np])
    cascade = joblib.load(cascade_path)
    cascade_probs = cascade.predict_proba(X_cascade)[:,1]

    # load base threshold
    with open(cascade_thr_path, 'r') as f:
        base_thr = json.load(f)['threshold']

    # Build y_true from opt tour
    opt_edges_set = set()
    if opt_tour:
        tour = opt_tour
        for a, b in zip(tour, tour[1:] + [tour[0]]):
            opt_edges_set.add(tuple(sorted((int(a), int(b)))))

    y_true = np.array([1 if tuple(sorted(e)) in opt_edges_set else 0 for e in edges], dtype=int)

    instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
    save_cache(cache_dir, instance_name, cascade_probs, edges, y_true, base_thr)
    return instance_name, cascade_probs, edges, y_true, base_thr


def evaluate_factors(cascade_probs, y_true, base_thr, factors):
    """Vectorized factor evaluation. Returns arrays of precision & recall for each candidate factor."""
    precs = np.zeros(len(factors))
    recs = np.zeros(len(factors))
    # vectorized loop
    for i, f in enumerate(factors):
        thr_adj = base_thr * f
        y_pred = (cascade_probs >= thr_adj).astype(int)
        precs[i] = precision_score(y_true, y_pred, zero_division=0)
        recs[i] = recall_score(y_true, y_pred, zero_division=0)
    return precs, recs


def main_cli(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dirs = args.tsplib_dirs
    cascade_path = args.cascade
    cascade_thr_path = args.cascade_thr
    model_path = args.model
    cache_dir = args.cache_dir

    # build list of tsp files
    tsp_files = []
    for d in dirs:
        tsp_files.extend(glob.glob(os.path.join(d, '*.tsp')))
    tsp_files = sorted(tsp_files)

    assert len(tsp_files) > 0, "No .tsp files found in provided directories"

    # factors grid
    factors = np.linspace(0.02, 1.0, 500)

    per_instance = {}
    per_range_factors = defaultdict(list)

    for tsp_path in tqdm(tsp_files, desc='Instances'):
        instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
        opt_path = tsp_path.replace('.tsp', '.opt.tour')

        # cache or compute
        if cache_exists(cache_dir, instance_name):
            cascade_probs, edges, y_true, base_thr = load_cache(cache_dir, instance_name)
        else:
            try:
                _, cascade_probs, edges, y_true, base_thr = compute_and_cache_probs(
                    tsp_path, opt_path, model_path, cascade_path, cascade_thr_path, device, cache_dir)
            except Exception as e:
                print(f"Skipping {instance_name} due to error during inference: {e}")
                continue
        

        # record
        tsp_meta = parse_tsp(tsp_path)
        n = tsp_meta['coords'].shape[0]
        range_id = get_range_id(n)

        # evaluate
        precs, recs = evaluate_factors(cascade_probs, y_true, base_thr, factors)

        # NEW: Store full results per range for printing later
        if "full_results" not in locals():
            full_results = defaultdict(list)

        for f, p, r2 in zip(factors, precs, recs):
            full_results[range_id].append({
                "factor": float(f),
                "precision": float(p),
                "recall": float(r2)
            })


        # choose candidates: recall >= 0.98 else fallback to >=0.95
        eligible_idx = np.where(recs >= 0.98)[0]
        used_threshold = 0.98
        if eligible_idx.size == 0:
            eligible_idx = np.where(recs >= 0.95)[0]
            used_threshold = 0.95

        chosen_factor = None
        chosen_prec = None
        chosen_rec = None

        if eligible_idx.size > 0:
            # among eligible pick max precision, tie-break smaller factor
            best_i = eligible_idx[np.argmax(precs[eligible_idx])]
            # if multiple with same precision pick smallest factor
            best_prec = precs[eligible_idx].max()
            candidates_same_prec = eligible_idx[precs[eligible_idx] == best_prec]
            best_i = candidates_same_prec[np.argmin(factors[candidates_same_prec])]

            chosen_factor = float(factors[best_i])
            chosen_prec = float(precs[best_i])
            chosen_rec = float(recs[best_i])
        else:
            # no eligible factor: pick factor that maximizes recall first then precision
            best_i = np.argmax(recs)
            chosen_factor = float(factors[best_i])
            chosen_prec = float(precs[best_i])
            chosen_rec = float(recs[best_i])
            used_threshold = float(recs[best_i])

        

        per_instance[instance_name] = {
            'n': int(n),
            'range': range_id,
            'chosen_factor': chosen_factor,
            'precision': chosen_prec,
            'recall': chosen_rec,
            'used_recall_threshold': used_threshold,
            'num_edges': int(len(edges))
        }

        per_range_factors[range_id].append(chosen_factor)
    
    

    print("\n==================== FACTOR PERFORMANCE PER RANGE ====================\n")

    # ---- NEW TOP-K SELECTION PARAMETERS ----
    K = 15                           # max number of factors per range
    recall_min = 0.95                # must satisfy ≥95%
    alpha = 2.0                      # recall weight
    beta = 1.0                       # precision weight

    def weighted_score(rec, prec):
        # Higher weight to recall:
        #   Example: giving up 2% recall must gain ~10% precision
        return alpha * rec + beta * prec

    topk_results = defaultdict(list)  # selected winners per range

    for r in ["1-20","21-50","51-100","101-200","201-500","501-1000","1001-2000","2001+"]:
        if r not in full_results:
            continue

        results = full_results[r]

        # ---- Filter only recall ≥95% ----
        valid = [x for x in results if x["recall"] >= recall_min]

        if not valid:
            print(f"\nRange {r}: NO factors reach >=95% recall")
            continue

        # ---- Compute weighted score ----
        for x in valid:
            x["score"] = weighted_score(x["recall"], x["precision"])

        # ---- Sort by score (desc) ----
        valid.sort(key=lambda x: x["score"], reverse=True)

        # ---- Keep only top K ----
        winners = valid[:K]
        topk_results[r] = winners

        # ---- PRINT ----
        print(f"\nRange {r}:  (top {K} factors by weighted score)")
        print("factor       score        recall      precision")
        print("------------------------------------------------")
        for x in winners:
            print(f"{x['factor']:.6f}   {x['score']:.4f}    {x['recall']:.4f}     {x['precision']:.4f}")

    # ---- SAVE TO JSON ----
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "full_factor_metrics.json"), "w") as f:
        json.dump(topk_results, f, indent=2)

    # Aggregate per range by median
    range_table = {}
    for r in ["1-20","21-50","51-100","101-200","201-500","501-1000","1001-2000","2001+"]:
        vals = per_range_factors.get(r, [])
        if len(vals) == 0:
            # fallback default
            range_table[r] = 0.3
        else:
            range_table[r] = float(np.median(vals))

    
    with open(os.path.join(out_dir, 'dynamic_factor_table.json'), 'w') as f:
        json.dump(range_table, f, indent=2)

    with open(os.path.join(out_dir, 'per_instance_factors.json'), 'w') as f:
        json.dump(per_instance, f, indent=2)

    print("Done. Saved results/dynamic_factor_table.json and results/per_instance_factors.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsplib_dirs', nargs='+', default=['tsplib_data', 'synthetic_tsplib'], help='folders with .tsp files')
    parser.add_argument('--cascade', default='results/cascade_stage1.pkl')
    parser.add_argument('--cascade_thr', default='results/cascade_stage1_thr.json')
    parser.add_argument('--model', default='results/best_model_threshold.pt')
    parser.add_argument('--cache_dir', default='cache_probs')
    args = parser.parse_args()
    main_cli(args)
