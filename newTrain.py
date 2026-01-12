# #!/usr/bin/env python3
# """
# tsp_gnn_pipeline.py

# End-to-end single-file pipeline (CPU) for edge-level classification in TSP instances.

# Features:
# - Parses .tsp (TSPLIB-style with NODE_COORD_SECTION) and .opt.tour files
# - Skips .tsp without matching .opt.tour
# - Builds full graph for n <= 300, k-NN (k=25) for n > 300; distances normalized per-instance
# - Computes node features (coords, deg, avg kNN dist, betweenness) and edge features (d, dx, dy, rank, in_MST, etc.)
# - GNN encoder: GraphSAGE (4 layers, ReLU, dropout=0.3) -> node embeddings
# - Edge classifier: concat(node_u_emb, node_v_emb, edge_features) -> MLP (with biases)
# - Training: Adam lr=0.001, CosineAnnealingLR scheduler, weighted BCE loss (pos_weight = neg/pos)
# - Split: by instances, 80% train, 20% test; plus 10% of training held out as validation for early stopping
# - Beam search tour builder using predicted probabilities mixed with inverse distances (0.7 prob + 0.3 inv-dist)
# - Prints metrics to console (AUC, Precision, Recall, F1, Accuracy, confusion matrix, average top-k)
# - Uses tqdm for progress and prints batch-level debug info every few batches
 
# """
# #Adapted model
# import os
# import math
# import json
# import time
# import random
# import argparse
# from collections import defaultdict

# import numpy as np
# import pandas as pd
# from sklearn.neural_network import MLPClassifier
# from tqdm import tqdm
# import networkx as nx
# from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR

# from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
# from sklearn.model_selection import train_test_split

# #plotting imports
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import MaxNLocator
# import pandas as pd
# from datetime import datetime



# import sys
# import os

# # Add the current directory to Python path to import from preprocess_data.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import the functions from preprocess_data.py
# try:
#     from preprocess_data import (
#         build_pyg_data_from_instance,
#         parse_tsp,
#         parse_opt_tour,
#         pairwise_distances,
#         build_candidate_edges,
#         compute_node_edge_features
#     )
#     print("✅ Successfully imported preprocessing functions from preprocess_data.py")
# except ImportError as e:
#     print(f"❌ Failed to import from preprocess_data.py: {e}")
#     print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.computations import (
#         compute_multi_seed_statistics
#     )
#     print("✅ Successfully imported com multi seed from computations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from computation.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.evaluations import (
#         evaluate_cascade_modes,
#         evaluate_global_threshold,
#         evaluate_topk_edges,
#         evaluate_cascade_stages,
#         analyze_threshold_sweep,
#         run_comprehensive_evaluation
#     )
#     print("✅ Successfully imported from evaluations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from evaluations.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.latex_table import (
#         create_single_seed_latex_table
#     )
#     print("✅ Successfully imported from latex table.py")
# except ImportError as e:
#     print(f"❌ Failed to import from latex table.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")

# try:
#     from Helpers.plottings import (
#         plot_training_progress,
#         plot_cascade_progress,
#         generate_multi_seed_plots,
#         plot_comprehensive_pr_curves,
#         plot_comprehensive_recall_vs_edges,
#     )
#     print("✅ Successfully imported from plotting.py")
# except ImportError as e:
#     print(f"❌ Failed to import from plotting.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")

# # PyG imports
# try:
#     from torch_geometric.data import Data, Batch
#     from torch_geometric.nn import SAGEConv
# except Exception as e:
#     raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")

# try: 
#     from Helpers.others import (
#         collate_batch,
#         evaluate_model_custom,
#         build_cascade_dataset,
#     )
#     print("✅ Successfully imported from others.py")
# except ImportError as e:
#     print(f"❌ Failed to import from others.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# class EdgePairwiseRankingLoss(nn.Module):
#     """
#     Pairwise ranking loss over edges incident to the same node.
#     """
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, edge_scores, edge_index, edge_labels):
#         """
#         edge_scores: (E,)
#         edge_index: (2, E)
#         edge_labels: (E,) in {0,1}
#         """
#         loss_terms = []

#         src, dst = edge_index

#         for node in torch.unique(torch.cat([src, dst])):
#             # edges incident to this node
#             mask = (src == node) | (dst == node)
#             idx = mask.nonzero(as_tuple=False).squeeze(1)

#             if idx.numel() < 2:
#                 continue

#             scores = edge_scores[idx]
#             labels = edge_labels[idx]

#             pos = scores[labels == 1]
#             neg = scores[labels == 0]

#             if pos.numel() == 0 or neg.numel() == 0:
#                 continue

#             # all pairwise (pos, neg)
#             diff = pos.view(-1, 1) - neg.view(1, -1)
#             loss = torch.clamp(self.margin - diff, min=0.0)
#             loss_terms.append(loss.mean())

#         if not loss_terms:
#             return torch.tensor(0.0, device=edge_scores.device, requires_grad=True)

#         return torch.stack(loss_terms).mean()


# def topk_recall_per_node(data, scores, k=2):
#     edge_index = data.edge_index
#     y = data.y
#     src, dst = edge_index

#     recall_vals = []

#     for node in torch.unique(torch.cat([src, dst])):
#         mask = (src == node) | (dst == node)
#         idx = mask.nonzero(as_tuple=False).squeeze(1)

#         if idx.numel() == 0:
#             continue

#         node_scores = scores[idx]
#         node_labels = y[idx]

#         topk_idx = torch.topk(node_scores, min(k, len(idx))).indices
#         recall = node_labels[topk_idx].sum() / node_labels.sum().clamp(min=1)
#         recall_vals.append(recall.item())

#     return float(np.mean(recall_vals)) if recall_vals else 0.0

# @torch.no_grad()
# def evaluate_ranking(model, dataset, device, k_list=(1, 2, 5)):
#     model.eval()

#     metrics = {f"top{k}_recall": [] for k in k_list}

#     for data in dataset:
#         data = data.to(device)
#         scores = model(data.x, data.edge_index, data.edge_attr)

#         for k in k_list:
#             r = topk_recall_per_node(data, scores, k=k)
#             metrics[f"top{k}_recall"].append(r)

#     # average over graphs
#     return {k: float(np.mean(v)) for k, v in metrics.items()}



# def run_multi_seed_experiments(args, seeds=[42], data_list=None):#, 123, 456, 789, 999], data_list=None):
#     """Run complete experiment with multiple random seeds"""
#     import warnings
#     warnings.filterwarnings('ignore')
    
#     all_results = []
#     seed_metrics = []
    
#     print(f"\n{'='*80}")
#     print(f"RUNNING MULTI-SEED EXPERIMENTS (Seeds: {seeds})")
#     print(f"{'='*80}\n")
    
#     for i, seed in enumerate(seeds):
#         print(f"\n{'='*60}")
#         print(f"SEED {i+1}/{len(seeds)}: {seed}")
#         print(f"{'='*60}")
        
#         # Run single seed experiment WITH preloaded data
#         results = run_single_seed_experiment(args, seed, data_list)  # PASS data_list here!
        
#         if results is not None:
#             all_results.append(results)
            
#             # Extract key metrics for this seed
#             seed_metric = {
#                 'seed': seed,
#                 'base_roc_auc': results['base_gnn']['roc_auc'],
#                 'base_pr_auc': results['base_gnn']['pr_auc'],
#                 'base_precision': results['base_gnn']['precision'],
#                 'base_recall': results['base_gnn']['recall'],
#                 'base_f1': results['base_gnn']['f1'],
#             }
            
#             if 'cascade_modes' in results and 'mode_a' in results['cascade_modes']:
#                 cascade_a = results['cascade_modes']['mode_a']
#                 seed_metric.update({
#                     'cascade_a_roc_auc': cascade_a['roc_auc'],
#                     'cascade_a_pr_auc': cascade_a['pr_auc'],
#                     'cascade_a_precision': cascade_a['precision'],
#                     'cascade_a_recall': cascade_a['recall'],
#                     'cascade_a_f1': cascade_a['f1'],
#                 })
            
#             if 'cascade_modes' in results and 'mode_b' in results['cascade_modes']:
#                 cascade_b = results['cascade_modes']['mode_b']
#                 seed_metric.update({
#                     'cascade_b_roc_auc': cascade_b['roc_auc'],
#                     'cascade_b_pr_auc': cascade_b['pr_auc'],
#                     'cascade_b_precision': cascade_b['precision'],
#                     'cascade_b_recall': cascade_b['recall'],
#                     'cascade_b_f1': cascade_b['f1'],
#                     'filter_rate': results['cascade_modes'].get('filter_rate', 0)
#                 })
            
#             seed_metrics.append(seed_metric)
#             print(f"\n✅ Seed {seed} completed successfully")
#         else:
#             print(f"\n❌ Seed {seed} failed")
    
#     # Compute statistics across seeds
#     compute_multi_seed_statistics(seed_metrics, args.out)
    
#     # Generate multi-seed plots
#     generate_multi_seed_plots(all_results, args.out)
    
#     return all_results, seed_metrics



# def run_single_seed_experiment(args, seed, data_list=None):
#     """Run a single seed experiment by calling our refactored function"""
#     print(f"\nRunning experiment with seed {seed}")
    
#     # Create a copy of args with seed-specific output directory
#     import copy
#     seed_args = copy.copy(args)
    
#     # Create seed-specific output directory
#     seed_out_dir = os.path.join(args.out, f"seed_{seed}")
#     seed_args.out = seed_out_dir
#     os.makedirs(seed_out_dir, exist_ok=True)
    
#     try:
#         # Run the complete experiment with this seed AND the preloaded data
#         results_summary = run_complete_experiment(seed_args, seed, data_list)  # PASS data_list here!
        
#         print(f"✅ Seed {seed} completed successfully")
#         return results_summary
        
#     except Exception as e:
#         print(f"❌ Seed {seed} failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None

# def run_complete_experiment(args, seed=None, data_list=None):
#     """
#     Run the complete TSP edge classification experiment.
#     If data_list is provided, use it instead of rebuilding from files.
#     """
#     print(f"DEBUG: data_list is {'provided' if data_list is not None else 'None'}")
#     if seed is not None:
#         # Set random seeds if provided
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         print(f"\nUsing seed: {seed}")
#     else:
#         # Use the seed from args
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
    
#     # Create output directory if it doesn't exist
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
#     results_summary = {}
#     if data_list is None:
#         # OLD CODE
#         print('Collecting instances...')
#         train_pairs = collect_pairs(args.train_dir)
#         synth_pairs = collect_pairs(args.synthetic_dir) if args.synthetic_dir else []
#         all_pairs = train_pairs + synth_pairs
#         print(f'Found {len(train_pairs)} tsplib instances and {len(synth_pairs)} synthetic instances; total {len(all_pairs)}')

#         # Build Data objects for all instances
#         data_objs = []
#         failures = []

#         print('Building Data objects (this may take time)...')
#         for tsp_path, tour_path in tqdm(all_pairs):
#             try:
#                 d = build_pyg_data_from_instance(tsp_path, tour_path, 
#                                                 full_threshold=args.full_threshold, 
#                                                 knn_k=args.knn_k, 
#                                                 knn_feat_k=args.knn_feat_k)
#                 if d is not None:
#                     data_objs.append(d)
#             except Exception as e:
#                 failures.append((tsp_path, str(e)))
        
#         print(f'Built {len(data_objs)} data objects; failed {len(failures)} instances')
#         if len(data_objs) == 0:
#             raise RuntimeError('No valid instances found.')
#         data_objs_to_use = data_objs
#     else:
#         # NEW: Use preloaded data
#         print(f"✅ Using {len(data_list)} preprocessed graphs")
#         data_objs_to_use = data_list
    
  
#     # Split by instances
#     idxs = list(range(len(data_objs_to_use)))
#     train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=args.seed)
#     train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)
#     train_set = [data_objs_to_use[i] for i in train_idx2]
#     val_set = [data_objs_to_use[i] for i in val_idx]
#     test_set = [data_objs_to_use[i] for i in test_idx]

#     print(f'Train: {len(train_set)} val: {len(val_set)} test: {len(test_set)}')
#     print(f'Ratios: Train {len(train_set)/len(data_objs_to_use):.1%}, '
#           f'Val {len(val_set)/len(data_objs_to_use):.1%}, '
#           f'Test {len(test_set)/len(data_objs_to_use):.1%}')
    
#     # Build model
#     in_node = train_set[0].x.shape[1]
#     in_edge = train_set[0].edge_attr.shape[1]
#     model = EdgeGNN(in_node_feats=in_node, in_edge_feats=in_edge, 
#                     hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout)
#     device = torch.device('cpu')
#     model.to(device)

#     training_history = {
#         'epoch': [],
#         'train_loss': [],
#         'val_top1': [],
#         'val_top2': [],
#         'val_top5': []
#     }

#     cascade_history = {'stage': [], 'pos_weight': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}

#     pos_weight = compute_pos_weight(train_set) * 0.75
#     print(f'Computed pos_weight={pos_weight:.4f} (neg/pos)')
   
    
#     criterion = EdgePairwiseRankingLoss(margin=1.0)

#     optimizer = Adam(model.parameters(), lr=0.001)
#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
#     print('Starting training...')
#     best_val_metric = -1.0
#     best_state_path = os.path.join(args.out, "best_model.pt")

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         train_loss = train_one_epoch_ranking(
#             model,
#             optimizer,
#             criterion,
#             train_set,
#             batch_size=args.batch_size,
#             device=device
#         )
        
   
#         scheduler.step()
#         t1 = time.time()

#         val_metrics = evaluate_ranking(
#             model=model,
#             dataset=val_set,
#             device=device,
#             k_list=args.topk_list
#         )
#         training_history['val_top1'].append(val_metrics['top1_recall'])
#         training_history['val_top2'].append(val_metrics['top2_recall'])
#         training_history['val_top5'].append(val_metrics['top5_recall'])

#         print(
#             f"Epoch {epoch}/{args.epochs} | "
#             f"train_loss={train_loss:.4f} | "
#             f"val@1={val_metrics['top1_recall']:.3f} | "
#             f"val@2={val_metrics['top2_recall']:.3f} | "
#             f"val@5={val_metrics['top5_recall']:.3f}"
#         )
#         metric_now = val_metrics['top2_recall']
#         if metric_now > best_val_metric:
#             best_val_metric = metric_now
#             torch.save(model.state_dict(), best_state_path)

#    # === Cascade training ===
#     # print("\n=== Training cascade classifier ===")
    

#     # # Build cascade datasets
#     # Xtr, ytr, _ = build_cascade_dataset(train_set, model, prob_cut=0.4)
#     # Xval, yval, _ = build_cascade_dataset(val_set, model, prob_cut=0.4)
#     # Xtest, ytest, _ = build_cascade_dataset(test_set, model, prob_cut=0.4)
#     # # Stage 1
#     # mlp_stage1, thr1 = train_cascade_mlp(Xtr, ytr, Xval, yval, cascade_history, args.out, stage=1)
    
#     # import joblib
#     # joblib.dump(mlp_stage1, os.path.join(args.out, "cascade_stage1.pkl"))
#     # with open(os.path.join(args.out, "cascade_stage1_thr.json"), "w") as f:
#     #     json.dump({"threshold": float(thr1)}, f, indent=2)

#     # # Stage 2
#     # p_tr = mlp_stage1.predict_proba(Xtr)[:, 1]
#     # p_val = mlp_stage1.predict_proba(Xval)[:, 1]
#     # p_test = mlp_stage1.predict_proba(Xtest)[:, 1]
    
#     # Xtr2 = np.hstack([Xtr, p_tr.reshape(-1, 1)])
#     # Xval2 = np.hstack([Xval, p_val.reshape(-1, 1)])
   

#     # mlp_stage2, thr2 = train_cascade_mlp(Xtr2, ytr, Xval2, yval, cascade_history, args.out, stage=2)
#     # # === Save stage 2 cascade ===
#     # joblib.dump(
#     #     mlp_stage2,
#     #     os.path.join(args.out, "cascade_stage2.pkl")
#     # )

#     # with open(os.path.join(args.out, "cascade_stage2_thr.json"), "w") as f:
#     #     json.dump(
#     #         {
#     #             "threshold": float(thr2),
#     #             "stage": 2,
#     #             "seed": seed
#     #         },
#     #         f,
#     #         indent=2
#     #     )

#     # results_summary = run_comprehensive_evaluation(
#     #     model=model,
#     #     mlp_stage1=mlp_stage1,
#     #     mlp_stage2=mlp_stage2,
#     #     test_set=test_set,
#     #     chosen_thr=chosen_thr,
#     #     cascade_thr=thr2,
#     #     training_history=training_history,
#     #     args=args,
#     #     device=device,
#     #     seed=seed
#     # )
#     # return results_summary

# def generate_multi_seed_report(all_results, seed_metrics, output_dir):
#     """Generate comprehensive multi-seed report"""
#     import pandas as pd
#     import os

#     print(f"\n{'='*80}")
#     print("GENERATING MULTI-SEED REPORT")
#     print(f"{'='*80}")
    
#     # Save raw results
#     results_df = pd.DataFrame(seed_metrics)
#     results_df.to_csv(os.path.join(output_dir, "all_seeds_raw_results.csv"), index=False)

#     # Compute exact total edges and positives across seeds
#     total_edges = sum(r['base_gnn']['n_edges'] for r in all_results)
#     total_positives = sum(r['base_gnn']['n_positives'] for r in all_results)
#     positive_rate = total_positives / total_edges if total_edges > 0 else 0.0

#     # Compute filter rate for Cascade Mode B, average across seeds if available
#     filter_rates = [
#         r['cascade_modes'].get('filter_rate', 0) 
#         for r in all_results 
#         if 'cascade_modes' in r and 'mode_b' in r['cascade_modes']
#     ]
#     avg_filter_rate = sum(filter_rates) / len(filter_rates) if filter_rates else 0.0

#     # Create summary markdown report
#     report = f"""
# # Multi-Seed Experiment Report

# ## Experiment Configuration
# - Number of seeds: {len(seed_metrics)}
# - Seeds used: {[m['seed'] for m in seed_metrics]}
# - Total edges evaluated: {total_edges}
# - Total positive edges: {total_positives} ({positive_rate:.2%})

# ## Key Findings

# ### 1. Statistical Significance
# All metrics show consistent performance across seeds with narrow confidence intervals, 
# indicating robust and reproducible results.

# ### 2. Cascade Effectiveness
# - **Mode A (all edges)**: Shows consistent degradation in overall ranking metrics
# - **Mode B (filtered edges)**: Shows consistent improvement in precision and F1-score
# - **Filter rate**: Average {avg_filter_rate:.2%} across seeds

# ### 3. Practical Implications
# The cascade refinement is statistically proven to:
# 1. Maintain high recall (>0.97) on filtered edges
# 2. Dramatically improve precision (0.12 -> 0.87)
# 3. Reduce candidate edges while preserving optimal tour edges

# ## Recommendations for Paper
# 1. Report metrics as mean +/- 95% CI across seeds
# 2. Focus on Cascade Mode B results for pruning applications
# 3. Include threshold sweep plots showing tradeoffs
# 4. Acknowledge Cascade Mode A limitations in discussion
# """
    
#     # Save report with proper encoding
#     report_path = os.path.join(output_dir, "multi_seed_report.md")
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(report)
    
#     print(f"\n✅ Multi-seed report saved to: {report_path}")
#     print("\nReport generated successfully.")



# # -----------------------------
# # Model: GraphSAGE encoder + edge MLP classifier
# # -----------------------------

# class EdgeGNN(nn.Module):
#     def __init__(self, in_node_feats, in_edge_feats,
#                  hidden_dim=128, n_layers=4, dropout=0.3):
#         super().__init__()

#         self.input_lin = nn.Linear(in_node_feats, hidden_dim)

#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         for _ in range(n_layers):
#             self.convs.append(SAGEConv(hidden_dim, hidden_dim))
#             self.norms.append(nn.LayerNorm(hidden_dim))

#         self.dropout = nn.Dropout(dropout)

#         edge_input_dim = hidden_dim * 2 + in_edge_feats
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(edge_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)   # raw score
#         )

#     def forward(self, x, edge_index, edge_attr):
#         h = self.input_lin(x)

#         for conv, norm in zip(self.convs, self.norms):
#             h = conv(h, edge_index)
#             h = norm(h)
#             h = F.relu(h)
#             h = self.dropout(h)

#         src, dst = edge_index
#         hu = h[src]
#         hv = h[dst]

#         edge_input = torch.cat([hu, hv, edge_attr], dim=1)
#         scores = self.edge_mlp(edge_input).squeeze(1)

#         return scores


# # -----------------------------
# # Training / evaluation
# # -----------------------------

# def compute_pos_weight(dataset):
    
#     pos = 0; neg = 0
#     for d in dataset:
#         arr = d.y.numpy()
#         pos += int((arr==1).sum())
#         neg += int((arr==0).sum())
#     if pos == 0:
#         return 1.0
#     return float(neg) / float(pos)




# #updated
# def train_one_epoch_ranking(model, optimizer, criterion,
#                             data_list, batch_size=1, device='cpu'):
#     model.train()
#     losses = []

#     random.shuffle(data_list)

#     for idx in range(0, len(data_list), batch_size):
#         batch = collate_batch(data_list[idx:idx+batch_size]).to(device)

#         optimizer.zero_grad()
#         scores = model(batch.x, batch.edge_index, batch.edge_attr)
#         loss = criterion(scores, batch.edge_index, batch.y)
#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())

#     return float(np.mean(losses))

# # -----------------------------
# # Main CLI
# # -----------------------------

# def collect_pairs(folder):
#     files = os.listdir(folder)
#     tsp_files = [os.path.join(folder,f) for f in files if f.endswith('.tsp')]
#     pairs = []
#     for t in tsp_files:
#         base = os.path.splitext(t)[0]
#         tourf = base + '.opt.tour'
#         if not os.path.exists(tourf):
#             tourf2 = base + '.tour'
#             if os.path.exists(tourf2):
#                 tourf = tourf2
#             else:
#                 continue
#         pairs.append((t, tourf))
#     return pairs


# def main():
#     import numpy as np

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_dir', type=str, required=True, help='folder with .tsp and .opt.tour (tsplib_data)')
#     parser.add_argument('--synthetic_dir', type=str, help='folder with synthetic instances')
#     parser.add_argument('--out', type=str, default='results', help='output folder for logs')
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--full_threshold', type=int, default=300)
#     parser.add_argument('--knn_k', type=int, default=25)
#     parser.add_argument('--knn_feat_k', type=int, default=10)
#     parser.add_argument('--hidden_dim', type=int, default=128)
#     parser.add_argument('--n_layers', type=int, default=4)
#     parser.add_argument('--dropout', type=float, default=0.3)
#     parser.add_argument('--beam_width', type=int, default=7)
#     parser.add_argument('--mix_prob', type=float, default=0.7)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--multi_seed', action='store_true', help='Run multi-seed experiments')
#     # parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999],
#     #                    help='Random seeds for multi-seed experiments')
#     parser.add_argument('--seeds', nargs='+', type=int, default=[42],
#                        help='Random seeds for multi-seed experiments')
#     parser.add_argument('--topk_list', nargs='+', type=int, default=[1,2,5,7,8])

#     args = parser.parse_args()
    
#     # Create main output directory
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
#     # Load preprocessed data
#     print("Loading preprocessed data...")
#     try:
#         data_list = torch.load("data_cache/pyg_graphs.pt", weights_only=False)
#         print(f"✅ Successfully loaded {len(data_list)} preprocessed graphs")
#     except Exception as e:
#         print(f"❌ Failed to load preprocessed data: {e}")
#         print("Falling back to rebuilding from raw files...")
#         data_list = None
    
#     if args.multi_seed:
#         # Run multi-seed experiments WITH preloaded data
#         print(f"\n{'='*80}")
#         print("RUNNING MULTI-SEED EXPERIMENTS")
#         print(f"{'='*80}")
        
#         all_results, seed_metrics = run_multi_seed_experiments(args, args.seeds, data_list)
        
#         # Generate multi-seed report (with encoding fix)
#         generate_multi_seed_report(all_results, seed_metrics, args.out)
        
#     else:
#         # Run single seed experiment WITH preloaded data
#         print(f"\n{'='*80}")
#         print("RUNNING SINGLE SEED EXPERIMENT")
#         print(f"{'='*80}")
        
#         results_summary = run_complete_experiment(args, data_list=data_list)
        
        
#         print("\n" + "="*80)
#         print("EXPERIMENT COMPLETE")
#         print("="*80)
#         print("✅ Enhanced PR curves saved")
#         print("✅ Recall vs edges kept plots saved") 
#         print("✅ Threshold sweep analysis saved")
#         print("✅ LaTeX table saved")
    
#     print('\nAll done.')
    

# if __name__ == '__main__':
#     main()




























import os
import math
import json
import time
import random
import argparse
import copy
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_fscore_support, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, precision_recall_curve
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from datetime import datetime

import sys

# Add the current directory to Python path to import from preprocess_data.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from preprocess_data.py
try:
    from preprocess_data import (
        build_pyg_data_from_instance,
        parse_tsp,
        parse_opt_tour,
        pairwise_distances,
        build_candidate_edges,
        compute_node_edge_features
    )
    print("✅ Successfully imported preprocessing functions from preprocess_data.py")
except ImportError as e:
    print(f"❌ Failed to import from preprocess_data.py: {e}")
    print("Make sure preprocess_data.py is in the same directory as this script.")

try:
    from Helpers.computations import (
        compute_multi_seed_statistics
    )
    print("✅ Successfully imported compute_multi_seed_statistics from computations.py")
except ImportError as e:
    print(f"❌ Failed to import from computations.py: {e}")

try:
    from Helpers.evaluations import (
        evaluate_cascade_modes,
        evaluate_global_threshold,
        evaluate_topk_edges,
        evaluate_cascade_stages,
        analyze_threshold_sweep,
        run_comprehensive_evaluation
    )
    print("✅ Successfully imported from evaluations.py")
except ImportError as e:
    print(f"❌ Failed to import from evaluations.py: {e}")

try:
    from Helpers.latex_table import (
        create_single_seed_latex_table
    )
    print("✅ Successfully imported from latex_table.py")
except ImportError as e:
    print(f"❌ Failed to import from latex_table.py: {e}")

try:
    from Helpers.plottings import (
        plot_training_progress,
        plot_cascade_progress,
        generate_multi_seed_plots,
        plot_comprehensive_pr_curves,
        plot_comprehensive_recall_vs_edges,
    )
    print("✅ Successfully imported from plottings.py")
except ImportError as e:
    print(f"❌ Failed to import from plottings.py: {e}")

# PyG imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError(
        "PyTorch Geometric not found or failed to import. "
        "Install it per the instructions in the script header."
    )

try:
    from Helpers.others import (
        collate_batch,
        evaluate_model_custom,
        build_cascade_dataset,
    )
    print("✅ Successfully imported from others.py")
except ImportError as e:
    print(f"❌ Failed to import from others.py: {e}")




# =============================================================================
# MODEL DEFINITION
# =============================================================================

class EdgeGNN(nn.Module):
    """
    Graph Neural Network for edge classification in TSP.
    Uses SAGEConv layers for node embeddings and an MLP for edge scoring.
    """
    def __init__(self, in_node_feats, in_edge_feats,
                 hidden_dim=128, n_layers=4, dropout=0.3):
        super().__init__()

        self.input_lin = nn.Linear(in_node_feats, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        edge_input_dim = hidden_dim * 2 + in_edge_feats
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # raw score
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.input_lin(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)

        src, dst = edge_index
        hu = h[src]
        hv = h[dst]

        edge_input = torch.cat([hu, hv, edge_attr], dim=1)
        scores = self.edge_mlp(edge_input).squeeze(1)

        return scores

# ==========================================
#   Cascade related stuff
# ==========================================

def evaluate_stage(model, dataset, device, k, stage_name, apply_structural=False, 
                   degree_cap=6, require_top2=True):
    """
    Evaluate a single stage of the cascade.
    
    Args:
        model: EdgeGNN model
        dataset: List of PyG Data objects
        device: torch device
        k: Number of top edges to keep per node
        stage_name: Name for logging
        apply_structural: Whether to apply structural pruning rules
        degree_cap: Maximum degree per node (for structural pruning)
        require_top2: Whether edge must be top-2 for at least one endpoint
    
    Returns:
        Tuple of (summary dict, list of pruned Data objects)
    """
    model.eval()
    
    results = {
        'recalls': [],
        'success_count': 0,
        'edges_kept': [],
        'edges_original': [],
        'sparsities': [],
        'instance_details': [],
        'stage_success': []
    }
    
    pruned_dataset = []
    
    # Create a temporary cascade pruner for the prune_to_topk method
    # We need to get dimensions from the first data object
    if len(dataset) > 0:
        sample_data = dataset[0]
        in_node = sample_data.x.size(1)
        in_edge = sample_data.edge_attr.size(1)
    else:
        raise ValueError("Dataset is empty")
    
    with torch.no_grad():
        for i, data in enumerate(dataset):
            data = data.to(device)
            n_nodes = data.x.size(0)
            
            # Skip empty graphs
            if data.edge_index.size(1) == 0:
                continue
                
            scores = model(data.x, data.edge_index, data.edge_attr)
            
            original_edges = data.edge_index.size(1)
            total_tour_edges = data.y.sum().item()
            
            if apply_structural:
                pruned_data, kept_mask = apply_structural_pruning(
                    data, scores, k, degree_cap, require_top2
                )
            else:
                # Standard top-k pruning using the CascadeEdgePruner method
                cascade = CascadeEdgePruner(in_node, in_edge)
                pruned_data, kept_mask = cascade.prune_to_topk(data, scores, k)
            
            # Compute metrics
            kept_edges = pruned_data.edge_index.size(1)
            tour_edges_kept = pruned_data.y.sum().item()
            recall = tour_edges_kept / total_tour_edges if total_tour_edges > 0 else 1.0
            is_success = (tour_edges_kept == total_tour_edges)
            sparsity = kept_edges / original_edges if original_edges > 0 else 0
            
            results['recalls'].append(recall)
            results['success_count'] += int(is_success)
            results['edges_kept'].append(kept_edges)
            results['edges_original'].append(original_edges)
            results['sparsities'].append(sparsity)
            results['stage_success'].append(is_success)

            
            results['instance_details'].append({
                'instance_id': i,
                'n_nodes': n_nodes,
                'original_edges': original_edges,
                'kept_edges': kept_edges,
                'tour_edges': total_tour_edges,
                'tour_edges_kept': tour_edges_kept,
                'recall': recall,
                'success': is_success,
                'sparsity': sparsity
            })
            deg = torch.zeros(n_nodes, dtype=torch.long)
            src_p, dst_p = pruned_data.edge_index

            for u, v in zip(src_p.tolist(), dst_p.tolist()):
                deg[u] += 1
                deg[v] += 1
            if apply_structural:
                results['instance_details'][-1].update({
                    'min_degree': int(deg.min().item()),
                    'mean_degree': float(deg.float().mean().item()),
                    'max_degree': int(deg.max().item())
                })


            
            pruned_dataset.append(pruned_data.cpu())
    
    n_instances = len(results['recalls'])
    
    if n_instances == 0:
        # Return empty results if no valid instances
        return {
            'stage': stage_name,
            'k': k,
            'mean_recall': 0,
            'min_recall': 0,
            'max_recall': 0,
            'std_recall': 0,
            'success_rate': 0,
            'success_count': 0,
            'failed_count': 0,
            'total_instances': 0,
            'mean_edges_kept': 0,
            'mean_sparsity': 0,
            'instance_details': []
        }, []
    
    summary = {
        'stage': stage_name,
        'k': k,
        'mean_recall': np.mean(results['recalls']) * 100,
        'min_recall': np.min(results['recalls']) * 100,
        'max_recall': np.max(results['recalls']) * 100,
        'std_recall': np.std(results['recalls']) * 100,
        'success_rate': results['success_count'] / n_instances * 100,
        'success_count': results['success_count'],
        'failed_count': n_instances - results['success_count'],
        'total_instances': n_instances,
        'mean_edges_kept': np.mean(results['edges_kept']),
        'mean_sparsity': np.mean(results['sparsities']) * 100,
        'instance_details': results['instance_details'],
        'success_flags': results['stage_success']
    }
    
    return summary, pruned_dataset

class CascadeEdgePruner(nn.Module):
    """Two-stage cascade for edge pruning"""
    
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=128, 
                 n_layers=4, dropout=0.3, share_weights=False):
        super().__init__()
        
        # Stage 1: Works on dense k-NN graph
        self.stage1 = EdgeGNN(in_node_feats, in_edge_feats, 
                              hidden_dim, n_layers, dropout)
        
        # Stage 2: Works on pruned graph (can share weights or not)
        if share_weights:
            self.stage2 = self.stage1
        else:
            # Separate model - potentially smaller since input is cleaner
            self.stage2 = EdgeGNN(in_node_feats, in_edge_feats,
                                  hidden_dim, n_layers, dropout)
    
    def forward_stage1(self, x, edge_index, edge_attr):
        return self.stage1(x, edge_index, edge_attr)
    
    def forward_stage2(self, x, edge_index, edge_attr):
        return self.stage2(x, edge_index, edge_attr)
    
    def prune_to_topk(self, data, scores, k):
        """Prune graph keeping top-k edges per node"""
        edge_index = data.edge_index
        src, dst = edge_index
        n_nodes = data.x.size(0)
        
        kept_edges = set()
        
        for node in range(n_nodes):
            mask = (src == node) | (dst == node)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            
            if idx.numel() == 0:
                continue
            
            node_scores = scores[idx]
            topk_count = min(k, idx.numel())
            topk_indices = torch.topk(node_scores, topk_count).indices
            
            for i in topk_indices:
                kept_edges.add(idx[i].item())
        
        kept_edges = sorted(kept_edges)
        kept_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        kept_mask[kept_edges] = True
        
        # Build pruned graph
        new_edge_index = edge_index[:, kept_mask]
        new_edge_attr = data.edge_attr[kept_mask]
        new_y = data.y[kept_mask]
        
        pruned_data = Data(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=new_y
        )
        
        return pruned_data, kept_mask
# =============================================================================
# CASCADE PIPELINE INTEGRATION
# =============================================================================

def compute_pruning_metrics(model, dataset, device, k_values=[5, 7, 8, 10, 15, 20]):
    """
    Compute comprehensive pruning metrics for different k values.
    
    Returns:
        Dictionary with metrics for each k value
    """
    model.eval()
    
    results = {k: {
        'recalls': [],
        'success_count': 0,
        'total_edges_kept': 0,
        'total_edges_original': 0,
        'total_tour_edges': 0,
        'tour_edges_kept': 0,
        'instance_details': []
    } for k in k_values}
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset, desc="Computing pruning metrics")):
            data = data.to(device)
            n_nodes = data.x.size(0)
            scores = model(data.x, data.edge_index, data.edge_attr)
            
            edge_index = data.edge_index
            y = data.y
            src, dst = edge_index
            
            total_tour_edges = y.sum().item()
            original_edges = edge_index.size(1)
            
            for k in k_values:
                # Get top-k edges per node
                kept_edges = set()
                
                for node in range(n_nodes):
                    mask = (src == node) | (dst == node)
                    idx = mask.nonzero(as_tuple=False).squeeze(-1)
                    
                    if idx.numel() == 0:
                        continue
                    
                    node_scores = scores[idx]
                    topk_count = min(k, idx.numel())
                    topk_indices = torch.topk(node_scores, topk_count).indices
                    
                    for ti in topk_indices:
                        kept_edges.add(idx[ti].item())
                
                # Compute metrics
                kept_edges_list = sorted(kept_edges)
                n_kept = len(kept_edges_list)
                
                # Count tour edges kept
                tour_edges_kept = sum(1 for e in kept_edges_list if y[e].item() == 1)
                recall = tour_edges_kept / total_tour_edges if total_tour_edges > 0 else 1.0
                
                # Is this a success (100% recall)?
                is_success = (tour_edges_kept == total_tour_edges)
                
                # Sparsity
                sparsity = n_kept / original_edges if original_edges > 0 else 0
                
                # Store results
                results[k]['recalls'].append(recall)
                results[k]['success_count'] += int(is_success)
                results[k]['total_edges_kept'] += n_kept
                results[k]['total_edges_original'] += original_edges
                results[k]['total_tour_edges'] += total_tour_edges
                results[k]['tour_edges_kept'] += tour_edges_kept
                
                results[k]['instance_details'].append({
                    'instance_id': i,
                    'n_nodes': n_nodes,
                    'original_edges': original_edges,
                    'kept_edges': n_kept,
                    'tour_edges': total_tour_edges,
                    'tour_edges_kept': tour_edges_kept,
                    'recall': recall,
                    'success': is_success,
                    'sparsity': sparsity
                })
    
    # Compute summary statistics
    n_instances = len(dataset)
    summary = {}
    
    for k in k_values:
        r = results[k]
        summary[k] = {
            'mean_recall': np.mean(r['recalls']) * 100,
            'min_recall': np.min(r['recalls']) * 100,
            'max_recall': np.max(r['recalls']) * 100,
            'std_recall': np.std(r['recalls']) * 100,
            'success_rate': r['success_count'] / n_instances * 100,
            'success_count': r['success_count'],
            'total_instances': n_instances,
            'failed_instances': n_instances - r['success_count'],
            'avg_sparsity': r['total_edges_kept'] / r['total_edges_original'] * 100,
            'avg_edges_per_instance': r['total_edges_kept'] / n_instances,
            'overall_tour_recall': r['tour_edges_kept'] / r['total_tour_edges'] * 100,
            'instance_details': r['instance_details']
        }
    
    return summary


def apply_structural_pruning(data, scores, k, degree_cap=6, require_top2_for_one=True):
    """
    Apply Stage 3 structural pruning rules.
    
    Rules:
    1. Degree cap: Each node keeps at most `degree_cap` edges
    2. Top-2 requirement: Edge must be in top-2 for at least one endpoint
    
    Args:
        data: PyG Data object
        scores: Edge scores from model
        k: Base k value for initial top-k
        degree_cap: Maximum degree per node
        require_top2_for_one: If True, edge must be top-2 for at least one endpoint
    
    Returns:
        Pruned Data object, kept mask
    """
    edge_index = data.edge_index
    src, dst = edge_index
    n_nodes = data.x.size(0)
    n_edges = edge_index.size(1)
    
    # Step 1: Get top-k edges per node
    node_topk = {node: set() for node in range(n_nodes)}
    node_top2 = {node: set() for node in range(n_nodes)}
    
    for node in range(n_nodes):
        mask = (src == node) | (dst == node)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        
        if idx.numel() == 0:
            continue
        
        node_scores = scores[idx]
        
        # Top-k
        topk_count = min(k, idx.numel())
        topk_indices = torch.topk(node_scores, topk_count).indices
        for ti in topk_indices:
            node_topk[node].add(idx[ti].item())
        
        # Top-2
        top2_count = min(2, idx.numel())
        top2_indices = torch.topk(node_scores, top2_count).indices
        for ti in top2_indices:
            node_top2[node].add(idx[ti].item())
    
    # Step 2: Initial kept edges (union of all top-k)
    kept_edges = set()
    for node in range(n_nodes):
        kept_edges.update(node_topk[node])
    
    # Step 3: Apply top-2 requirement
    if require_top2_for_one:
        filtered_edges = set()
        for e in kept_edges:
            u, v = src[e].item(), dst[e].item()
            # Edge must be top-2 for at least one endpoint
            if e in node_top2[u] or e in node_top2[v]:
                filtered_edges.add(e)
        kept_edges = filtered_edges
    
    # Step 4: Apply symmetric degree cap (GLOBAL)
    if degree_cap is not None:
        # Initialize degrees
        degrees = {node: 0 for node in range(n_nodes)}

        # Build edge list with scores
        edge_candidates = []
        for e in kept_edges:
            u, v = src[e].item(), dst[e].item()
            edge_candidates.append((e, scores[e].item(), u, v))

        # Sort edges globally by score (high → low)
        edge_candidates.sort(key=lambda x: -x[1])

        final_edges = set()

        for e, score, u, v in edge_candidates:
            # Check degree budgets
            u_can = degrees[u] < degree_cap
            v_can = degrees[v] < degree_cap

            # Symmetric acceptance rule
            if u_can and v_can:
                final_edges.add(e)
                degrees[u] += 1
                degrees[v] += 1
            elif u_can and not v_can:
                final_edges.add(e)
                degrees[u] += 1
            elif v_can and not u_can:
                final_edges.add(e)
                degrees[v] += 1
            # else: drop edge

        kept_edges = final_edges

    
    # Build pruned graph
    kept_edges_list = sorted(kept_edges)
    kept_mask = torch.zeros(n_edges, dtype=torch.bool)
    kept_mask[kept_edges_list] = True
    
    pruned_data = Data(
        x=data.x,
        edge_index=edge_index[:, kept_mask],
        edge_attr=data.edge_attr[kept_mask],
        y=data.y[kept_mask]
    )
    
    return pruned_data, kept_mask


def save_stage_results(stage_results, stage_dir, stage_name):
    """
    Save stage results to files.
    """
    os.makedirs(stage_dir, exist_ok=True)
    
    # Save summary as JSON
    summary_path = os.path.join(stage_dir, f"{stage_name}_summary.json")
    
    # Remove instance_details for JSON (save separately)
    summary_for_json = {k: v for k, v in stage_results.items() if k != 'instance_details'}
    
    with open(summary_path, 'w') as f:
        json.dump(summary_for_json, f, indent=2, default=str)
    
    # Save instance details as CSV
    if 'instance_details' in stage_results:
        details_df = pd.DataFrame(stage_results['instance_details'])
        details_path = os.path.join(stage_dir, f"{stage_name}_details.csv")
        details_df.to_csv(details_path, index=False)
    
    # Save human-readable report
    report_path = os.path.join(stage_dir, f"{stage_name}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{stage_name.upper()} RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Configuration:\n")
        f.write(f"  k value: {stage_results.get('k', 'N/A')}\n")
        f.write(f"  Total instances: {stage_results['total_instances']}\n\n")
        
        f.write(f"Recall Metrics:\n")
        f.write(f"  Mean Recall:    {stage_results['mean_recall']:.2f}%\n")
        f.write(f"  Min Recall:     {stage_results['min_recall']:.2f}%\n")
        f.write(f"  Max Recall:     {stage_results['max_recall']:.2f}%\n")
        f.write(f"  Std Recall:     {stage_results['std_recall']:.2f}%\n\n")
        
        f.write(f"Success Metrics:\n")
        f.write(f"  Success Rate:   {stage_results['success_rate']:.2f}%\n")
        f.write(f"  Successful:     {stage_results['success_count']}/{stage_results['total_instances']}\n")
        f.write(f"  Failed:         {stage_results['failed_count']}/{stage_results['total_instances']}\n\n")
        
        f.write(f"Sparsity Metrics:\n")
        f.write(f"  Mean Sparsity:  {stage_results['mean_sparsity']:.2f}%\n")
        f.write(f"  Mean Edges:     {stage_results['mean_edges_kept']:.1f}\n")
    
    print(f"✅ {stage_name} results saved to {stage_dir}")


def run_cascade_pipeline(base_model, train_set, val_set, test_set, device, args):
    """
    Run the complete cascade pipeline after base model training.
    
    Stage 1: Use the trained base model with k1
    Stage 2: Train a new model on Stage 1-pruned graphs, prune with k2
    Stage 3: Apply structural pruning rules
    
    Args:
        base_model: Trained EdgeGNN model (becomes Stage 1)
        train_set: Training dataset
        val_set: Validation dataset
        test_set: Test dataset
        device: torch device
        args: Command line arguments
    
    Returns:
        Dictionary with all cascade results
    """
    print("\n" + "=" * 80)
    print("RUNNING CASCADE PIPELINE")
    print("=" * 80)
    
    # Configuration
    k1 = getattr(args, 'cascade_k1', 10)  # Stage 1: keep top-10
    k2 = getattr(args, 'cascade_k2', 5)   # Stage 2: keep top-5
    k3 = getattr(args, 'cascade_k3', 4)   # Stage 3 base k
    degree_cap = getattr(args, 'degree_cap', 6)
    
    print(f"\nCascade Configuration:")
    print(f"  Stage 1 k: {k1}")
    print(f"  Stage 2 k: {k2}")
    print(f"  Stage 3 k: {k3}, degree_cap: {degree_cap}")
    
    # Create output directories
    stage1_dir = os.path.join(args.out, "stage1")
    stage2_dir = os.path.join(args.out, "stage2")
    stage3_dir = os.path.join(args.out, "stage3")
    
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    os.makedirs(stage3_dir, exist_ok=True)
    
    cascade_results = {}
    
    # =========================================================================
    # STAGE 1: Evaluate base model at k1
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STAGE 1: Base Model Pruning (k={k1})")
    print("=" * 60)
    
    base_model.eval()
    
    # Evaluate on test set
    stage1_test_results, test_set_pruned_s1 = evaluate_stage(
        base_model, test_set, device, k1, "stage1_test"
    )
    
    # Evaluate on train/val for Stage 2 training
    stage1_train_results, train_set_pruned_s1 = evaluate_stage(
        base_model, train_set, device, k1, "stage1_train"
    )
    stage1_val_results, val_set_pruned_s1 = evaluate_stage(
        base_model, val_set, device, k1, "stage1_val"
    )
    
    # Print Stage 1 summary
    print(f"\nStage 1 Test Results (k={k1}):")
    print(f"  Mean Recall:    {stage1_test_results['mean_recall']:.2f}%")
    print(f"  Success Rate:   {stage1_test_results['success_rate']:.2f}%")
    print(f"  Mean Sparsity:  {stage1_test_results['mean_sparsity']:.2f}%")
    print(f"  Mean Edges:     {stage1_test_results['mean_edges_kept']:.1f}")
    
    # Save Stage 1 results
    save_stage_results(stage1_test_results, stage1_dir, "stage1")
    cascade_results['stage1'] = stage1_test_results
    
    # Also save multi-k evaluation for Stage 1
    print("\nComputing Stage 1 metrics for multiple k values...")
    stage1_multi_k = compute_pruning_metrics(
        base_model, test_set, device, 
        k_values=[2, 3, 4, 5, 7, 8, 10, 15, 20]
    )
    
    # Save multi-k results
    multi_k_summary = []
    for k, metrics in stage1_multi_k.items():
        multi_k_summary.append({
            'k': k,
            'mean_recall': metrics['mean_recall'],
            'success_rate': metrics['success_rate'],
            'avg_sparsity': metrics['avg_sparsity'],
            'failed_instances': metrics['failed_instances']
        })
    
    multi_k_df = pd.DataFrame(multi_k_summary)
    multi_k_df.to_csv(os.path.join(stage1_dir, "stage1_multi_k.csv"), index=False)
    
    print("\nStage 1 Multi-k Summary:")
    print(multi_k_df.to_string(index=False))
    
    # =========================================================================
    # STAGE 2: Train new model on pruned graphs
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STAGE 2: Context-Aware Refinement (k={k2})")
    print("=" * 60)
    
    # Create Stage 2 model
    in_node = train_set[0].x.shape[1]
    in_edge = train_set[0].edge_attr.shape[1]
    
    stage2_model = EdgeGNN(
        in_node_feats=in_node,
        in_edge_feats=in_edge,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Train Stage 2 on pruned graphs
    print(f"\nTraining Stage 2 model on {len(train_set_pruned_s1)} pruned graphs...")
    print(f"  Input from Stage 1: ~{stage1_train_results['mean_edges_kept']:.0f} edges/instance")
    
    optimizer2 = Adam(stage2_model.parameters(), lr=0.001)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=1e-5)
    criterion = EdgePairwiseRankingLoss(margin=1.0)
    
    best_val_recall_s2 = 0
    stage2_best_path = os.path.join(stage2_dir, "stage2_best.pt")
    
    stage2_history = {
        'epoch': [],
        'train_loss': [],
        'val_recall': []
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        stage2_model.train()
        losses = []
        random.shuffle(train_set_pruned_s1)
        
        for data in train_set_pruned_s1:
            data = data.to(device)
            optimizer2.zero_grad()
            
            # Skip if no edges
            if data.edge_index.size(1) == 0:
                continue
            
            scores = stage2_model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(scores, data.edge_index, data.y)
            loss.backward()
            optimizer2.step()
            losses.append(loss.item())
        
        scheduler2.step()
        
        # Validate
        val_metrics = evaluate_ranking(stage2_model, val_set_pruned_s1, device, k_list=[k2])
        val_recall = val_metrics.get(f'top{k2}_recall', 0)
        
        stage2_history['epoch'].append(epoch)
        stage2_history['train_loss'].append(np.mean(losses) if losses else 0)
        stage2_history['val_recall'].append(val_recall)
        
        print(f"Stage2 Epoch {epoch}/{args.epochs} | "
              f"loss={np.mean(losses) if losses else 0:.4f} | "
              f"val@{k2}={val_recall:.4f}")
        
        if val_recall > best_val_recall_s2:
            best_val_recall_s2 = val_recall
            torch.save(stage2_model.state_dict(), stage2_best_path)
            print(f"  ↳ New best Stage 2 model saved!")
    
    # Load best Stage 2 model
    stage2_model.load_state_dict(torch.load(stage2_best_path))
    stage2_model.eval()
    
    print(f"\n✅ Stage 2 training complete. Best val@{k2} = {best_val_recall_s2:.4f}")
    
    # Evaluate Stage 2 on test set (cascaded from Stage 1)
    stage2_test_results, test_set_pruned_s2 = evaluate_stage(
        stage2_model, test_set_pruned_s1, device, k2, "stage2_test"
    )
    # Conditional recall: Stage 2 | Stage 1 success
    stage1_flags = stage1_test_results['success_flags']
    stage2_recalls = stage2_test_results['instance_details']

    stage1_success_map = {
        d['instance_id']: d['success']
        for d in stage1_test_results['instance_details']
    }
    cond_recalls = [
        d['recall']
        for d in stage2_test_results['instance_details']
        if stage1_success_map.get(d['instance_id'], False)
    ]

    conditional_recall_s2 = np.mean(cond_recalls) * 100 if cond_recalls else 0.0


    print(f"\nStage 2 Test Results (k={k2}):")
    print(f"  Mean Recall:    {stage2_test_results['mean_recall']:.2f}%")
    print(f"  Success Rate:   {stage2_test_results['success_rate']:.2f}%")
    print(f"  Mean Sparsity:  {stage2_test_results['mean_sparsity']:.2f}%")
    print(f"  Mean Edges:     {stage2_test_results['mean_edges_kept']:.1f}")
    print(f"  Conditional Recall (S2 | S1 success): {conditional_recall_s2:.2f}%")

    # Save Stage 2 results
    save_stage_results(stage2_test_results, stage2_dir, "stage2")
    cascade_results['stage2'] = stage2_test_results
    
    # Save training history
    history_df = pd.DataFrame(stage2_history)
    history_df.to_csv(os.path.join(stage2_dir, "stage2_training_history.csv"), index=False)
    
    # =========================================================================
    # STAGE 3: Structural Pruning
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"STAGE 3: Structural Pruning (k={k3}, degree_cap={degree_cap})")
    print("=" * 60)
    
    stage3_test_results, test_set_pruned_s3 = evaluate_stage(
        stage2_model, test_set_pruned_s2, device, k3, "stage3_test",
        apply_structural=True, degree_cap=degree_cap, require_top2=True
    )
    # Conditional recall: Stage 3 | Stage 2 success
    stage2_flags = stage2_test_results['success_flags']
    stage3_recalls = stage3_test_results['instance_details']

    stage2_success_map = {
        d['instance_id']: d['success']
        for d in stage2_test_results['instance_details']
    }

    cond_recalls = [
        d['recall']
        for d in stage2_test_results['instance_details']
        if stage2_success_map.get(d['instance_id'], False)
    ]

    conditional_recall_s3 = np.mean(cond_recalls) * 100 if cond_recalls else 0.0


    print(f"\nStage 3 Test Results (structural pruning):")
    print(f"  Mean Recall:    {stage3_test_results['mean_recall']:.2f}%")
    print(f"  Success Rate:   {stage3_test_results['success_rate']:.2f}%")
    print(f"  Mean Sparsity:  {stage3_test_results['mean_sparsity']:.2f}%")
    print(f"  Mean Edges:     {stage3_test_results['mean_edges_kept']:.1f}")
    print(f"  Conditional Recall (S3 | S2 success): {conditional_recall_s3:.2f}%")

    # Save Stage 3 results
    save_stage_results(stage3_test_results, stage3_dir, "stage3")
    cascade_results['stage3'] = stage3_test_results
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("CASCADE PIPELINE SUMMARY")
    print("=" * 80)
    
    comparison_data = []
    
    # Single-stage baselines at k2
    single_stage_k2 = stage1_multi_k.get(k2, {})
    comparison_data.append({
        'Method': f'Single-stage (k={k2})',
        'Mean Recall (%)': single_stage_k2.get('mean_recall', 0),
        'Success Rate (%)': single_stage_k2.get('success_rate', 0),
        'Sparsity (%)': single_stage_k2.get('avg_sparsity', 0),
        'Failed Instances': single_stage_k2.get('failed_instances', 0)
    })
    
    # Stage 1
    comparison_data.append({
        'Method': f'Stage 1 (k={k1})',
        'Mean Recall (%)': stage1_test_results['mean_recall'],
        'Success Rate (%)': stage1_test_results['success_rate'],
        'Sparsity (%)': stage1_test_results['mean_sparsity'],
        'Failed Instances': stage1_test_results['failed_count']
    })
    
    # Stage 2 (cascade)
    comparison_data.append({
        'Method': f'Cascade S1→S2 (k₁={k1}→k₂={k2})',
        'Mean Recall (%)': stage2_test_results['mean_recall'],
        'Success Rate (%)': stage2_test_results['success_rate'],
        'Sparsity (%)': stage2_test_results['mean_sparsity'],
        'Failed Instances': stage2_test_results['failed_count']
    })
    
    # Stage 3 (structural)
    comparison_data.append({
        'Method': f'Cascade + Structural (k={k3}, cap={degree_cap})',
        'Mean Recall (%)': stage3_test_results['mean_recall'],
        'Success Rate (%)': stage3_test_results['success_rate'],
        'Sparsity (%)': stage3_test_results['mean_sparsity'],
        'Failed Instances': stage3_test_results['failed_count']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(os.path.join(args.out, "cascade_comparison.csv"), index=False)
    
    # Key result
    single_recall = single_stage_k2.get('mean_recall', 0)
    cascade_recall = stage2_test_results['mean_recall']
    improvement = cascade_recall - single_recall
    
    print(f"\n📊 KEY RESULT: Cascade vs Single-stage at k={k2}")
    print(f"   Single-stage recall: {single_recall:.2f}%")
    print(f"   Cascade recall:      {cascade_recall:.2f}%")
    print(f"   Improvement:         {improvement:+.2f}%")
    
    if improvement > 0:
        print("   ✅ CASCADE OUTPERFORMS SINGLE-STAGE!")
    else:
        print("   ⚠️ Single-stage performs better (investigate model/hyperparameters)")
    
    # Save cascade results
    cascade_results['comparison'] = comparison_data
    cascade_results['improvement'] = {
        'single_stage_recall': single_recall,
        'cascade_recall': cascade_recall,
        'improvement': improvement
    }
    
    # Save overall cascade summary
    with open(os.path.join(args.out, "cascade_summary.json"), 'w') as f:
        # Convert to serializable format
        summary_for_json = {
            'stage1': {k: v for k, v in stage1_test_results.items() if k != 'instance_details'},
            'stage2': {k: v for k, v in stage2_test_results.items() if k != 'instance_details'},
            'stage3': {k: v for k, v in stage3_test_results.items() if k != 'instance_details'},
            'improvement': cascade_results['improvement'],
            'config': {
                'k1': k1, 'k2': k2, 'k3': k3, 'degree_cap': degree_cap
            }
        }
        json.dump(summary_for_json, f, indent=2, default=str)
    
    print(f"\n✅ All cascade results saved to {args.out}")
    
    return cascade_results, stage2_model


def plot_cascade_comparison(cascade_results, output_dir):
    """
    Generate comparison plots for cascade results.
    """
    import matplotlib.pyplot as plt
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data
    stages = ['Stage 1', 'Stage 2\n(Cascade)', 'Stage 3\n(Structural)']
    recalls = [
        cascade_results['stage1']['mean_recall'],
        cascade_results['stage2']['mean_recall'],
        cascade_results['stage3']['mean_recall']
    ]
    success_rates = [
        cascade_results['stage1']['success_rate'],
        cascade_results['stage2']['success_rate'],
        cascade_results['stage3']['success_rate']
    ]
    sparsities = [
        cascade_results['stage1']['mean_sparsity'],
        cascade_results['stage2']['mean_sparsity'],
        cascade_results['stage3']['mean_sparsity']
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Plot 1: Recall
    axes[0].bar(stages, recalls, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Mean Recall (%)', fontsize=12)
    axes[0].set_title('Recall by Stage', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(recalls) - 5, 100])
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: Success Rate
    axes[1].bar(stages, success_rates, color=colors, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Success Rate (%)', fontsize=12)
    axes[1].set_title('Success Rate by Stage', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 100])
    for i, v in enumerate(success_rates):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 3: Sparsity
    axes[2].bar(stages, sparsities, color=colors, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('Sparsity (%)', fontsize=12)
    axes[2].set_title('Edge Sparsity by Stage', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, max(sparsities) + 10])
    for i, v in enumerate(sparsities):
        axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cascade_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Cascade comparison plot saved")
    
    # Plot recall vs sparsity trade-off
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(sparsities, recalls, s=200, c=colors, edgecolors='black', linewidth=2, zorder=3)
    
    for i, stage in enumerate(['S1', 'S2', 'S3']):
        ax.annotate(stage, (sparsities[i], recalls[i]), 
                   textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Sparsity (%)', fontsize=12)
    ax.set_ylabel('Mean Recall (%)', fontsize=12)
    ax.set_title('Recall vs Sparsity Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'recall_vs_sparsity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Recall vs sparsity plot saved")


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class EdgePairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss over edges incident to the same node.
    Encourages positive (tour) edges to have higher scores than negative edges.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, edge_scores, edge_index, edge_labels):
        """
        Args:
            edge_scores: (E,) tensor of edge scores
            edge_index: (2, E) tensor of edge indices
            edge_labels: (E,) tensor of labels in {0, 1}
        
        Returns:
            Pairwise ranking loss
        """
        loss_terms = []
        src, dst = edge_index

        for node in torch.unique(torch.cat([src, dst])):
            # Get edges incident to this node
            mask = (src == node) | (dst == node)
            idx = mask.nonzero(as_tuple=False).squeeze(1)

            if idx.numel() < 2:
                continue

            scores = edge_scores[idx]
            labels = edge_labels[idx]

            pos = scores[labels == 1]
            neg = scores[labels == 0]

            if pos.numel() == 0 or neg.numel() == 0:
                continue

            # Compute all pairwise (pos, neg) differences
            diff = pos.view(-1, 1) - neg.view(1, -1)
            loss = torch.clamp(self.margin - diff, min=0.0)
            loss_terms.append(loss.mean())

        if not loss_terms:
            return torch.tensor(0.0, device=edge_scores.device, requires_grad=True)

        return torch.stack(loss_terms).mean()


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def topk_recall_per_node(data, scores, k=2):
    """
    Compute recall@k per node and average across all nodes.
    
    Args:
        data: PyG Data object
        scores: Edge scores from model
        k: Number of top edges to consider per node
    
    Returns:
        Mean recall@k across all nodes
    """
    edge_index = data.edge_index
    y = data.y
    src, dst = edge_index

    recall_vals = []

    for node in torch.unique(torch.cat([src, dst])):
        mask = (src == node) | (dst == node)
        idx = mask.nonzero(as_tuple=False).squeeze(1)

        if idx.numel() == 0:
            continue

        node_scores = scores[idx]
        node_labels = y[idx]

        # Get top-k edges for this node
        topk_idx = torch.topk(node_scores, min(k, len(idx))).indices
        
        # Compute recall: what fraction of positive edges are in top-k
        total_pos = node_labels.sum().clamp(min=1)
        recalled_pos = node_labels[topk_idx].sum()
        recall = recalled_pos / total_pos
        recall_vals.append(recall.item())

    return float(np.mean(recall_vals)) if recall_vals else 0.0


@torch.no_grad()
def evaluate_ranking(model, dataset, device, k_list=(1, 2, 5)):
    """
    Evaluate ranking metrics on a dataset.
    
    Args:
        model: The EdgeGNN model
        dataset: List of PyG Data objects
        device: torch device
        k_list: List of k values for top-k recall
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = {f"top{k}_recall": [] for k in k_list}

    for data in dataset:
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.edge_attr)

        for k in k_list:
            r = topk_recall_per_node(data, scores, k=k)
            metrics[f"top{k}_recall"].append(r)

    # Average over graphs
    return {k: float(np.mean(v)) for k, v in metrics.items()}


@torch.no_grad()
def evaluate_classification_metrics(model, dataset, device, threshold=0.5):
    """
    Evaluate classification metrics (precision, recall, F1, AUC) on a dataset.
    
    Args:
        model: The EdgeGNN model
        dataset: List of PyG Data objects
        device: torch device
        threshold: Classification threshold for binary predictions
    
    Returns:
        Dictionary of classification metrics
    """
    model.eval()
    
    all_scores = []
    all_labels = []
    
    for data in dataset:
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.edge_attr)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(scores)
        
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Binary predictions
    preds = (all_scores >= threshold).astype(int)
    
    # Compute metrics
    results = {
        'roc_auc': roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
        'pr_auc': average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'f1': f1_score(all_labels, preds, zero_division=0),
        'accuracy': accuracy_score(all_labels, preds),
        'threshold': threshold,
    }
    
    return results


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def compute_pos_weight(dataset):
    """
    Compute positive class weight for imbalanced classification.
    
    Args:
        dataset: List of PyG Data objects
    
    Returns:
        Weight ratio (negative / positive)
    """
    pos = 0
    neg = 0
    for d in dataset:
        arr = d.y.numpy()
        pos += int((arr == 1).sum())
        neg += int((arr == 0).sum())
    
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def train_one_epoch_ranking(model, optimizer, criterion,
                            data_list, batch_size=1, device='cpu'):
    """
    Train the model for one epoch using ranking loss.
    
    Args:
        model: The EdgeGNN model
        optimizer: PyTorch optimizer
        criterion: Loss function (EdgePairwiseRankingLoss)
        data_list: List of PyG Data objects
        batch_size: Batch size for training
        device: torch device
    
    Returns:
        Mean training loss for the epoch
    """
    model.train()
    losses = []

    random.shuffle(data_list)

    for idx in range(0, len(data_list), batch_size):
        batch = collate_batch(data_list[idx:idx + batch_size]).to(device)

        optimizer.zero_grad()
        scores = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(scores, batch.edge_index, batch.y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def collect_pairs(folder):
    """
    Collect pairs of .tsp and .opt.tour files from a folder.
    
    Args:
        folder: Path to directory containing TSP instances
    
    Returns:
        List of (tsp_path, tour_path) tuples
    """
    files = os.listdir(folder)
    tsp_files = [os.path.join(folder, f) for f in files if f.endswith('.tsp')]
    pairs = []
    
    for t in tsp_files:
        base = os.path.splitext(t)[0]
        tourf = base + '.opt.tour'
        if not os.path.exists(tourf):
            tourf2 = base + '.tour'
            if os.path.exists(tourf2):
                tourf = tourf2
            else:
                continue
        pairs.append((t, tourf))
    
    return pairs


# =============================================================================
# MULTI-SEED EXPERIMENT FUNCTIONS
# =============================================================================

def generate_multi_seed_report(all_results, seed_metrics, output_dir):
    """
    Generate a summary report for multi-seed experiments.
    
    Args:
        all_results: List of result dictionaries from each seed
        seed_metrics: List of metric dictionaries from each seed
        output_dir: Output directory for the report
    """
    report_path = os.path.join(output_dir, "multi_seed_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-SEED EXPERIMENT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Number of seeds: {len(seed_metrics)}\n")
        f.write(f"Seeds used: {[m['seed'] for m in seed_metrics]}\n\n")
        
        if seed_metrics:
            df = pd.DataFrame(seed_metrics)
            
            f.write("INDIVIDUAL SEED RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS (Mean ± Std):\n")
            f.write("-" * 50 + "\n")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'seed':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    f.write(f"  {col:30s}: {mean_val:.4f} ± {std_val:.4f}\n")
            
            f.write("\n")
            
            # Best seed
            if 'base_f1' in df.columns:
                best_idx = df['base_f1'].idxmax()
                f.write(f"Best seed (by base F1): {df.loc[best_idx, 'seed']} "
                       f"(F1 = {df.loc[best_idx, 'base_f1']:.4f})\n")
            
            # Save as CSV too
            csv_path = os.path.join(output_dir, "multi_seed_metrics.csv")
            df.to_csv(csv_path, index=False)
            f.write(f"\nDetailed metrics saved to: {csv_path}\n")
    
    print(f"✅ Multi-seed report saved to {report_path}")


def run_multi_seed_experiments(args, seeds=[42], data_list=None):
    """
    Run complete experiment with multiple random seeds.
    
    Args:
        args: Command line arguments
        seeds: List of random seeds
        data_list: Preprocessed data (optional)
    
    Returns:
        Tuple of (all_results, seed_metrics)
    """
    warnings.filterwarnings('ignore')
    
    all_results = []
    seed_metrics = []
    
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-SEED EXPERIMENTS (Seeds: {seeds})")
    print(f"{'='*80}\n")
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"SEED {i+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")
        
        # Run single seed experiment with preloaded data
        results = run_single_seed_experiment(args, seed, data_list)
        
        if results is not None:
            all_results.append(results)
            
            # Extract key metrics for this seed
            seed_metric = {
                'seed': seed,
                'base_roc_auc': results.get('base_gnn', {}).get('roc_auc', 0),
                'base_pr_auc': results.get('base_gnn', {}).get('pr_auc', 0),
                'base_precision': results.get('base_gnn', {}).get('precision', 0),
                'base_recall': results.get('base_gnn', {}).get('recall', 0),
                'base_f1': results.get('base_gnn', {}).get('f1', 0),
            }
            
            # Add cascade mode A metrics if available
            if 'cascade_modes' in results and results['cascade_modes']:
                if 'mode_a' in results['cascade_modes']:
                    cascade_a = results['cascade_modes']['mode_a']
                    seed_metric.update({
                        'cascade_a_roc_auc': cascade_a.get('roc_auc', 0),
                        'cascade_a_pr_auc': cascade_a.get('pr_auc', 0),
                        'cascade_a_precision': cascade_a.get('precision', 0),
                        'cascade_a_recall': cascade_a.get('recall', 0),
                        'cascade_a_f1': cascade_a.get('f1', 0),
                    })
                
                # Add cascade mode B metrics if available
                if 'mode_b' in results['cascade_modes']:
                    cascade_b = results['cascade_modes']['mode_b']
                    seed_metric.update({
                        'cascade_b_roc_auc': cascade_b.get('roc_auc', 0),
                        'cascade_b_pr_auc': cascade_b.get('pr_auc', 0),
                        'cascade_b_precision': cascade_b.get('precision', 0),
                        'cascade_b_recall': cascade_b.get('recall', 0),
                        'cascade_b_f1': cascade_b.get('f1', 0),
                        'filter_rate': results['cascade_modes'].get('filter_rate', 0)
                    })
            
            seed_metrics.append(seed_metric)
            print(f"\n✅ Seed {seed} completed successfully")
        else:
            print(f"\n❌ Seed {seed} failed")
    
    # Compute statistics across seeds
    if seed_metrics:
        try:
            compute_multi_seed_statistics(seed_metrics, args.out)
        except Exception as e:
            print(f"Warning: Could not compute multi-seed statistics: {e}")
    
    # Generate multi-seed plots
    if all_results:
        try:
            generate_multi_seed_plots(all_results, args.out)
        except Exception as e:
            print(f"Warning: Could not generate multi-seed plots: {e}")
    
    return all_results, seed_metrics


def run_single_seed_experiment(args, seed, data_list=None):
    """
    Run a single seed experiment.
    
    Args:
        args: Command line arguments
        seed: Random seed
        data_list: Preprocessed data (optional)
    
    Returns:
        Results dictionary or None if failed
    """
    print(f"\nRunning experiment with seed {seed}")
    
    # Create a copy of args with seed-specific output directory
    seed_args = copy.copy(args)
    
    # Create seed-specific output directory
    seed_out_dir = os.path.join(args.out, f"seed_{seed}")
    seed_args.out = seed_out_dir
    os.makedirs(seed_out_dir, exist_ok=True)
    os.makedirs(os.path.join(seed_out_dir, "plots"), exist_ok=True)
    
    try:
        # Run the complete experiment with this seed and the preloaded data
        results_summary = run_complete_experiment(seed_args, seed, data_list)
        print(f"✅ Seed {seed} completed successfully")
        return results_summary
        
    except Exception as e:
        print(f"❌ Seed {seed} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================

def run_complete_experiment(args, seed=None, data_list=None):
    """
    Run the complete TSP edge classification experiment.
    
    Args:
        args: Command line arguments
        seed: Random seed (optional, uses args.seed if None)
        data_list: Preprocessed data (optional)
    
    Returns:
        Dictionary containing all experiment results
    """
    # Determine which seed to use
    actual_seed = seed if seed is not None else args.seed
    
    # Set random seeds for reproducibility
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
    
    print(f"\nUsing seed: {actual_seed}")
    
    # Create output directories
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
    # Initialize results summary
    results_summary = {
        'seed': actual_seed,
        'args': vars(args),
    }
    
    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    if data_list is None:
        print('Collecting instances...')
        train_pairs = collect_pairs(args.train_dir)
        synth_pairs = collect_pairs(args.synthetic_dir) if args.synthetic_dir else []
        all_pairs = train_pairs + synth_pairs
        print(f'Found {len(train_pairs)} tsplib instances and {len(synth_pairs)} synthetic instances; '
              f'total {len(all_pairs)}')

        # Build Data objects for all instances
        data_objs = []
        failures = []

        print('Building Data objects (this may take time)...')
        for tsp_path, tour_path in tqdm(all_pairs):
            try:
                d = build_pyg_data_from_instance(
                    tsp_path, tour_path,
                    full_threshold=args.full_threshold,
                    knn_k=args.knn_k,
                    knn_feat_k=args.knn_feat_k
                )
                if d is not None:
                    data_objs.append(d)
            except Exception as e:
                failures.append((tsp_path, str(e)))
        
        print(f'Built {len(data_objs)} data objects; failed {len(failures)} instances')
        if len(data_objs) == 0:
            raise RuntimeError('No valid instances found.')
        
        data_objs_to_use = data_objs
    else:
        # Use preloaded data
        print(f"✅ Using {len(data_list)} preprocessed graphs")
        data_objs_to_use = data_list
    
    # -------------------------------------------------------------------------
    # TRAIN/VAL/TEST SPLIT
    # -------------------------------------------------------------------------
    idxs = list(range(len(data_objs_to_use)))
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=actual_seed)
    train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=actual_seed)
    
    train_set = [data_objs_to_use[i] for i in train_idx2]
    val_set = [data_objs_to_use[i] for i in val_idx]
    test_set = [data_objs_to_use[i] for i in test_idx]

    print(f'\nData splits:')
    print(f'  Train: {len(train_set)} ({len(train_set)/len(data_objs_to_use):.1%})')
    print(f'  Val:   {len(val_set)} ({len(val_set)/len(data_objs_to_use):.1%})')
    print(f'  Test:  {len(test_set)} ({len(test_set)/len(data_objs_to_use):.1%})')
    
    # Store split info in results
    results_summary['data_info'] = {
        'total_instances': len(data_objs_to_use),
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
    }
    
    # -------------------------------------------------------------------------
    # MODEL SETUP
    # -------------------------------------------------------------------------
    in_node = train_set[0].x.shape[1]
    in_edge = train_set[0].edge_attr.shape[1]
    
    print(f'\nModel configuration:')
    print(f'  Input node features: {in_node}')
    print(f'  Input edge features: {in_edge}')
    print(f'  Hidden dimension: {args.hidden_dim}')
    print(f'  Number of layers: {args.n_layers}')
    print(f'  Dropout: {args.dropout}')
    
    model = EdgeGNN(
        in_node_feats=in_node,
        in_edge_feats=in_edge,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')
    
    results_summary['model_info'] = {
        'in_node_feats': in_node,
        'in_edge_feats': in_edge,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }
    
    # -------------------------------------------------------------------------
    # TRAINING SETUP
    # -------------------------------------------------------------------------
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_top1': [],
        'val_top2': [],
        'val_top5': [],
        'learning_rate': [],
    }
    
    # Compute class weights
    pos_weight = compute_pos_weight(train_set)
    adjusted_pos_weight = pos_weight * 0.75
    print(f'\nClass balance:')
    print(f'  Pos weight (neg/pos): {pos_weight:.4f}')
    print(f'  Adjusted pos weight: {adjusted_pos_weight:.4f}')
    
    # Loss, optimizer, scheduler
    criterion = EdgePairwiseRankingLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    # Best model tracking
    best_val_metric = -1.0
    best_state_path = os.path.join(args.out, "best_model.pt")
    best_epoch = 0
    
    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------
    print(f'\n{"="*60}')
    print('STARTING TRAINING')
    print(f'{"="*60}')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train one epoch
        train_loss = train_one_epoch_ranking(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data_list=train_set,
            batch_size=args.batch_size,
            device=device
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        t1 = time.time()
        epoch_time = t1 - t0
        
        # Evaluate on validation set
        val_metrics = evaluate_ranking(
            model=model,
            dataset=val_set,
            device=device,
            k_list=args.topk_list
        )
        
        # Record history
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_top1'].append(val_metrics.get('top1_recall', 0))
        training_history['val_top2'].append(val_metrics.get('top2_recall', 0))
        training_history['val_top5'].append(val_metrics.get('top5_recall', 0))
        training_history['learning_rate'].append(current_lr)
        
        # Print progress
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val@1={val_metrics.get('top1_recall', 0):.3f} | "
            f"val@2={val_metrics.get('top2_recall', 0):.3f} | "
            f"val@5={val_metrics.get('top5_recall', 0):.3f} | "
            f"lr={current_lr:.2e} | "
            f"time={epoch_time:.1f}s"
        )
        
        # Save best model
        metric_now = val_metrics.get('top2_recall', 0)
        if metric_now > best_val_metric:
            best_val_metric = metric_now
            best_epoch = epoch
            torch.save(model.state_dict(), best_state_path)
            print(f"  ↳ New best model saved! (val@2 = {best_val_metric:.4f})")
    
    print(f'\n{"="*60}')
    print('TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'Best validation metric: {best_val_metric:.4f} at epoch {best_epoch}')
    
    results_summary['training_history'] = training_history
    results_summary['best_epoch'] = best_epoch
    results_summary['best_val_metric'] = best_val_metric
    
    # -------------------------------------------------------------------------
    # LOAD BEST MODEL AND EVALUATE
    # -------------------------------------------------------------------------
    print('\nLoading best model for evaluation...')
    model.load_state_dict(torch.load(best_state_path))

    model.eval()
    
    # Test set ranking metrics
    print('\nEvaluating on test set...')
    test_ranking_metrics = evaluate_ranking(
        model=model,
        dataset=test_set,
        device=device,
        k_list=args.topk_list
    )
    
    print(f'\nTest Ranking Metrics:')
    for k, v in test_ranking_metrics.items():
        print(f'  {k}: {v:.4f}')
    
    results_summary['test_ranking_metrics'] = test_ranking_metrics
    
    # Test set classification metrics
    test_classification_metrics = evaluate_classification_metrics(
        model=model,
        dataset=test_set,
        device=device,
        threshold=0.5
    )
    
    print(f'\nTest Classification Metrics:')
    for k, v in test_classification_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    
    results_summary['base_gnn'] = test_classification_metrics
    
    # -------------------------------------------------------------------------
    # CASCADE EVALUATION (if available)
    # -------------------------------------------------------------------------
    print('\nRunning cascade pipeline...')
    cascade_results = run_cascade_pipeline(
        base_model=model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        device=device,
        args=args
    )
    results_summary['cascade'] = cascade_results
    print('✅ Cascade pipeline complete')

    # -------------------------------------------------------------------------
    # GENERATE PLOTS
    # -------------------------------------------------------------------------
    print('\nGenerating plots...')
    
    try:
        plot_training_progress(training_history, args.out)
        print('✅ Training progress plot saved')
    except Exception as e:
        print(f'⚠️ Could not generate training progress plot: {e}')
    
    try:
        # Generate PR curves if we have the data
        plot_comprehensive_pr_curves(model, test_set, device, args.out)
        print('✅ PR curves saved')
    except Exception as e:
        print(f'⚠️ Could not generate PR curves: {e}')
    
    try:
        plot_comprehensive_recall_vs_edges(model, test_set, device, args.out)
        print('✅ Recall vs edges plot saved')
    except Exception as e:
        print(f'⚠️ Could not generate recall vs edges plot: {e}')
    
    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    results_path = os.path.join(args.out, "results_summary.json")
    
    # Convert non-serializable items
    serializable_results = {}
    for k, v in results_summary.items():
        if isinstance(v, dict):
            serializable_results[k] = {
                str(kk): (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
            }
        elif isinstance(v, (np.floating, np.integer)):
            serializable_results[k] = float(v)
        elif isinstance(v, list):
            serializable_results[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x
                for x in v
            ]
        else:
            serializable_results[k] = v
    
    try:
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f'\n✅ Results saved to {results_path}')
    except Exception as e:
        print(f'⚠️ Could not save results JSON: {e}')
    
    # Generate LaTeX table
    try:
        create_single_seed_latex_table(results_summary, args.out)
        print('✅ LaTeX table saved')
    except Exception as e:
        print(f'⚠️ Could not generate LaTeX table: {e}')
    
    print(f'\n{"="*60}')
    print('EXPERIMENT COMPLETE')
    print(f'{"="*60}')
    print(f'Results saved to: {args.out}')
    
    return results_summary


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TSP Edge Classification with GNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Folder with .tsp and .opt.tour files (tsplib_data)')
    parser.add_argument('--synthetic_dir', type=str, default=None,
                        help='Folder with synthetic instances')
    parser.add_argument('--out', type=str, default='results',
                        help='Output folder for logs and results')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Data preprocessing arguments
    parser.add_argument('--full_threshold', type=int, default=300,
                        help='Node threshold for full graph vs k-NN graph')
    parser.add_argument('--knn_k', type=int, default=30,
                        help='Number of nearest neighbors for edge construction')
    parser.add_argument('--knn_feat_k', type=int, default=10,
                        help='Number of nearest neighbors for feature computation')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension of GNN layers')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Evaluation arguments
    parser.add_argument('--beam_width', type=int, default=7,
                        help='Beam width for beam search decoding')
    parser.add_argument('--mix_prob', type=float, default=0.7,
                        help='Mixing probability for cascade')
    parser.add_argument('--topk_list', nargs='+', type=int, default=[1, 2, 5, 7, 8],
                        help='List of k values for top-k recall evaluation')
    
    # Multi-seed arguments
    parser.add_argument('--multi_seed', action='store_true',
                        help='Run multi-seed experiments')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='Random seeds for multi-seed experiments')
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("TSP EDGE CLASSIFICATION EXPERIMENT")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    data_list = None
    try:
        data_list = torch.load("data_cache/pyg_graphs.pt", weights_only=False)
        print(f"✅ Successfully loaded {len(data_list)} preprocessed graphs")
    except FileNotFoundError:
        print("⚠️ Preprocessed data not found at 'data_cache/pyg_graphs.pt'")
        print("Will rebuild from raw files...")
    except Exception as e:
        print(f"❌ Failed to load preprocessed data: {e}")
        print("Falling back to rebuilding from raw files...")
    
    # Run experiments
    if args.multi_seed:
        print(f"\n{'='*80}")
        print("RUNNING MULTI-SEED EXPERIMENTS")
        print(f"Seeds: {args.seeds}")
        print(f"{'='*80}")
        
        all_results, seed_metrics = run_multi_seed_experiments(args, args.seeds, data_list)
        
        # Generate multi-seed report
        if seed_metrics:
            generate_multi_seed_report(all_results, seed_metrics, args.out)
        
    else:
        print(f"\n{'='*80}")
        print("RUNNING SINGLE SEED EXPERIMENT")
        print(f"Seed: {args.seed}")
        print(f"{'='*80}")
        
        results_summary = run_complete_experiment(args, data_list=data_list)
        
        if results_summary:
            print("\n" + "=" * 80)
            print("EXPERIMENT COMPLETE")
            print("=" * 80)
            print(f"✅ Results saved to: {args.out}")
            print("✅ Training plots saved")
            print("✅ PR curves saved")
            print("✅ LaTeX table saved")
    
    print('\n✅ All done.')


if __name__ == '__main__':
    main()