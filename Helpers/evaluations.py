import os
import math
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

#plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import pandas as pd
from datetime import datetime



# PyG imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")


import sys
import os

try:
    from computations import (
        compute_metrics_continuous,
        knn_candidate_edges,
    )
    print("✅ Successfully imported from computations.py")
except ImportError as e:
    print(f"❌ Failed to import from computations.py: {e}")
    #print("Make sure preprocess_data.py is in the same directory as this script.")


try:
    from latex_table import (
        create_single_seed_latex_table
    )
    print("✅ Successfully imported from latex table.py")
except ImportError as e:
    print(f"❌ Failed to import from latex table.py: {e}")
    #print("Make sure preprocess_data.py is in the same directory as this script.")


# Import the functions from preprocess_data.py
try: 
    from plottings import (
        plot_comprehensive_pr_curves,
        plot_comprehensive_recall_vs_edges,
    )
    print("✅ Successfully imported from plottings.py")
except ImportError as e:
    print(f"❌ Failed to import from plottings.py: {e}")
    #print("Make sure preprocess_data.py is in the same directory as this script.")


try: 
    from others import (
        evaluate_model_custom,
        build_cascade_dataset,
    )
    print("✅ Successfully imported from others.py")
except ImportError as e:
    print(f"❌ Failed to import from others.py: {e}")
    #print("Make sure preprocess_data.py is in the same directory as this script.")




def evaluate_cascade_modes(base_model, cascade_mlp, test_set, stage1_thr=0.4, 
                          cascade_thr=0.5, device='cpu', mlp_stage1=None):
    """
    Evaluate cascade in two modes:
    Mode A: cascade on ALL edges
    Mode B: cascade only on stage-1 positives
    
    Returns metrics for both modes and for base GNN alone.
    """
    print("\n" + "="*80)
    print("CASCADE EVALUATION - TWO MODES")
    print("="*80)
    
    all_results = {}
    
    # -----------------------------------------------------------------
    # Mode A: Cascade on ALL edges (no filtering)
    # -----------------------------------------------------------------
    print("\n--- MODE A: Cascade on ALL edges ---")
    
    all_edges_data = []
    all_edges_labels = []
    all_edges_base_scores = []
    
    for d in tqdm(test_set, desc="Collecting all edges"):
        # Get base GNN scores for all edges
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            base_logits = base_model(b.x, b.edge_index, b.edge_attr)
            base_probs = torch.sigmoid(base_logits).cpu().numpy()
        
        # Build features for ALL edges (no filtering)
        edges = d.edges_list
        edge_attrs = d.edge_attr.cpu().numpy()
        
        for idx, (i, j) in enumerate(edges):
            base_score = base_probs[idx]
            
            # Build stage 1 feature vector [base_score, edge_features]
            stage1_feat = [base_score]
            stage1_feat.extend(edge_attrs[idx].tolist())
            
            # Get stage 1 probability (if mlp_stage1 is provided)
            if mlp_stage1 is not None:
                stage1_prob = mlp_stage1.predict_proba([stage1_feat])[:, 1][0]
            else:
                stage1_prob = 0.0  # Default if stage1 MLP not available
            
            # Build cascade stage 2 feature vector [stage1_feat, stage1_prob]
            stage2_feat = stage1_feat + [stage1_prob]  # Now 11 features
            
            all_edges_data.append(stage2_feat)
            all_edges_labels.append(int(d.y.cpu().numpy()[idx]))
            all_edges_base_scores.append(base_score)
    
    X_all = np.array(all_edges_data, dtype=float)
    y_all = np.array(all_edges_labels, dtype=int)
    base_scores_all = np.array(all_edges_base_scores)
    
    print(f"  Total edges: {len(y_all):,}")
    print(f"  Positive edges: {np.sum(y_all == 1):,} ({np.mean(y_all == 1)*100:.2f}%)")
    print(f"  Feature dimension: {X_all.shape[1]} (expected: 11)")
    
    # Get cascade predictions
    cascade_probs_all = cascade_mlp.predict_proba(X_all)[:, 1]
    
    # Compute metrics for MODE A
    metrics_a = compute_metrics_continuous(y_all, cascade_probs_all, cascade_thr)
    all_results['mode_a'] = metrics_a
    
    # -----------------------------------------------------------------
    # Mode B: Cascade only on stage-1 positives
    # -----------------------------------------------------------------
    print("\n--- MODE B: Cascade only on stage-1 positives ---")
    
    filtered_data = []
    filtered_labels = []
    filtered_base_scores = []
    
    for d in tqdm(test_set, desc="Collecting stage-1 positives"):
        # Get base GNN scores
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            base_logits = base_model(b.x, b.edge_index, b.edge_attr)
            base_probs = torch.sigmoid(base_logits).cpu().numpy()
        
        edges = d.edges_list
        edge_attrs = d.edge_attr.cpu().numpy()
        labels = d.y.cpu().numpy()
        
        for idx, (i, j) in enumerate(edges):
            base_score = base_probs[idx]
            
            # Only include if base_score >= stage1_thr
            if base_score >= stage1_thr:
                # Build stage 1 feature vector
                stage1_feat = [base_score]
                stage1_feat.extend(edge_attrs[idx].tolist())
                
                # Get stage 1 probability
                if mlp_stage1 is not None:
                    stage1_prob = mlp_stage1.predict_proba([stage1_feat])[:, 1][0]
                else:
                    stage1_prob = 0.0
                
                # Build cascade stage 2 feature vector
                stage2_feat = stage1_feat + [stage1_prob]  # 11 features
                
                filtered_data.append(stage2_feat)
                filtered_labels.append(int(labels[idx]))
                filtered_base_scores.append(base_score)
    
    if len(filtered_data) > 0:
        X_filtered = np.array(filtered_data, dtype=float)
        y_filtered = np.array(filtered_labels, dtype=int)
        base_scores_filtered = np.array(filtered_base_scores)
        
        print(f"  Filtered edges: {len(y_filtered):,} ({len(y_filtered)/len(y_all)*100:.1f}% of total)")
        print(f"  Positive edges in filtered: {np.sum(y_filtered == 1):,} ({np.mean(y_filtered == 1)*100:.2f}%)")
        
        # Get cascade predictions
        cascade_probs_filtered = cascade_mlp.predict_proba(X_filtered)[:, 1]
        
        # Compute metrics for MODE B
        metrics_b = compute_metrics_continuous(y_filtered, cascade_probs_filtered, cascade_thr)
        all_results['mode_b'] = metrics_b
        all_results['filter_rate'] = len(y_filtered) / len(y_all)
    else:
        print("  WARNING: No edges passed stage-1 threshold!")
        all_results['mode_b'] = None
    
    # -----------------------------------------------------------------
    # Base GNN alone (for comparison)
    # -----------------------------------------------------------------
    print("\n--- Base GNN alone (for comparison) ---")
    
    # Collect all base GNN scores (already computed)
    base_metrics = compute_metrics_continuous(y_all, base_scores_all, stage1_thr)
    all_results['base_gnn'] = base_metrics
    
    return all_results



def evaluate_global_threshold(model, val_set, device='cpu', recall_targets=[0.95, 0.98]):
    _, labels, preds = evaluate_model_custom(model, val_set)
    labels = np.array(labels)
    preds = np.array(preds)
    
    prec, rec, thr = precision_recall_curve(labels, preds)
    results = {}
    
    for target in recall_targets:
        valid_idxs = np.where(rec >= target)[0]
        if len(valid_idxs) > 0:
            idx = valid_idxs[np.argmax(prec[valid_idxs])]
            chosen_thr = thr[idx] if idx < len(thr) else 0.5
            yhat = (preds >= chosen_thr).astype(int)
            p, r, f, _ = precision_recall_fscore_support(labels, yhat, average='binary', zero_division=0)
            pr_auc = average_precision_score(labels, preds)
            results[target] = {'threshold': float(chosen_thr), 'precision': float(p),
                               'recall': float(r), 'f1': float(f), 'pr_auc': float(pr_auc)}
        else:
            results[target] = {'threshold': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                               'pr_auc': average_precision_score(labels, preds)}
    
    return results




def evaluate_topk_edges(model, dataset, k_values=[5, 10, 20], device='cpu'):
    """
    Evaluate top-k per node for multiple k values.
    """
    results = {}
    
    for k in k_values:
        all_labels = []
        all_preds = []
        all_probs = []
        
        for data in dataset:
            # Move data to device if needed
            data = data.to(device) if hasattr(data, 'to') else data
            edges = data.edges_list
            probs = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)).detach().cpu().numpy()
            prob_map = {tuple(e): p for e, p in zip(edges, probs)}

            
            # Per node: keep top-k
            topk_edges = set()
            for node in range(data.n_nodes):
                incident = [(prob_map.get((min(node, j), max(node, j)), 0.0), 
                           (min(node, j), max(node, j))) 
                           for j in range(data.n_nodes) if j != node]
                incident.sort(reverse=True)
                topk_edges.update([e for _, e in incident[:k]])
            
            preds = np.array([1 if tuple(e) in topk_edges else 0 for e in edges])
            all_preds.append(preds)
            all_labels.append(data.y.cpu().numpy())
            all_probs.extend(probs)
        
        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.array(all_probs)
            
            p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, 
                                                        average='binary', zero_division=0)
            pr_auc = average_precision_score(all_labels, all_probs)
            
            results[f'top{k}'] = {
                'precision': float(p), 
                'recall': float(r), 
                'f1': float(f), 
                'pr_auc': float(pr_auc)
            }
    
    return results




def evaluate_cascade_stages(base_model, mlp_stage1, mlp_stage2, test_set, device='cpu', recall_targets=[0.95, 0.98]):
    """
    Evaluate cascade stages using pre-trained MLPs on test set.
    """
    results = {}
    
    # Stage 0: Base GNN only
    _, labels, preds = evaluate_model_custom(base_model, test_set, device=device)
    
    # Evaluate at each recall target
    for target in recall_targets:
        # Find threshold for this recall target
        prec, rec, thr = precision_recall_curve(labels, preds)
        valid_idxs = np.where(rec >= target)[0]
        if len(valid_idxs) > 0:
            idx = valid_idxs[np.argmax(prec[valid_idxs])]
            chosen_thr = thr[idx] if idx < len(thr) else 0.5
            yhat = (preds >= chosen_thr).astype(int)
            p, r, f, _ = precision_recall_fscore_support(labels, yhat, average='binary', zero_division=0)
            pr_auc = average_precision_score(labels, preds)
            results[f'Base_GNN_recall_{target}'] = {
                'threshold': float(chosen_thr), 'precision': float(p),
                'recall': float(r), 'f1': float(f), 'pr_auc': float(pr_auc)
            }
    
    # Stage 1: Base + MLP1
    # Build features for test set
    X_test, y_test, _ = build_cascade_dataset(test_set, base_model, prob_cut=0.4)
    if len(X_test) > 0 and mlp_stage1 is not None:
        stage1_probs = mlp_stage1.predict_proba(X_test)[:, 1]
        
        for target in recall_targets:
            prec, rec, thr = precision_recall_curve(y_test, stage1_probs)
            valid_idxs = np.where(rec >= target)[0]
            if len(valid_idxs) > 0:
                idx = valid_idxs[np.argmax(prec[valid_idxs])]
                chosen_thr = thr[idx] if idx < len(thr) else 0.5
                yhat = (stage1_probs >= chosen_thr).astype(int)
                p, r, f, _ = precision_recall_fscore_support(y_test, yhat, average='binary', zero_division=0)
                pr_auc = average_precision_score(y_test, stage1_probs)
                results[f'Base_MLP1_recall_{target}'] = {
                    'threshold': float(chosen_thr), 'precision': float(p),
                    'recall': float(r), 'f1': float(f), 'pr_auc': float(pr_auc)
                }
    
    # Stage 2: Base + MLP1 + MLP2
    if len(X_test) > 0 and mlp_stage1 is not None and mlp_stage2 is not None:
        stage1_probs = mlp_stage1.predict_proba(X_test)[:, 1]
        X_test_stage2 = np.hstack([X_test, stage1_probs.reshape(-1, 1)])
        stage2_probs = mlp_stage2.predict_proba(X_test_stage2)[:, 1]
        
        for target in recall_targets:
            prec, rec, thr = precision_recall_curve(y_test, stage2_probs)
            valid_idxs = np.where(rec >= target)[0]
            if len(valid_idxs) > 0:
                idx = valid_idxs[np.argmax(prec[valid_idxs])]
                chosen_thr = thr[idx] if idx < len(thr) else 0.5
                yhat = (stage2_probs >= chosen_thr).astype(int)
                p, r, f, _ = precision_recall_fscore_support(y_test, yhat, average='binary', zero_division=0)
                pr_auc = average_precision_score(y_test, stage2_probs)
                results[f'Base_MLP1_MLP2_recall_{target}'] = {
                    'threshold': float(chosen_thr), 'precision': float(p),
                    'recall': float(r), 'f1': float(f), 'pr_auc': float(pr_auc)
                }
    
    return results




def analyze_threshold_sweep(results_summary, output_dir):
    """Analyze threshold sweep and generate summary statistics"""
    import pandas as pd
    
    analysis_data = []
    
    # Analyze Base GNN
    if 'base_gnn' in results_summary:
        base = results_summary['base_gnn']
        for thr, recall, edges_kept in zip(np.linspace(0, 1, 100), 
                                          base['recall_vs_edges'], 
                                          base['edges_kept']):
            analysis_data.append({
                'model': 'Base GNN',
                'threshold': thr,
                'recall': recall,
                'edges_kept': edges_kept,
                'edges_pruned': 1 - edges_kept
            })
    
    # Analyze Cascade Mode A
    if 'cascade_modes' in results_summary and 'mode_a' in results_summary['cascade_modes']:
        cascade_a = results_summary['cascade_modes']['mode_a']
        for thr, recall, edges_kept in zip(np.linspace(0, 1, 100), 
                                          cascade_a['recall_vs_edges'], 
                                          cascade_a['edges_kept']):
            analysis_data.append({
                'model': 'Cascade Mode A',
                'threshold': thr,
                'recall': recall,
                'edges_kept': edges_kept,
                'edges_pruned': 1 - edges_kept
            })
    
    # Analyze Cascade Mode B
    if 'cascade_modes' in results_summary and 'mode_b' in results_summary['cascade_modes']:
        cascade_b = results_summary['cascade_modes']['mode_b']
        for thr, recall, edges_kept in zip(np.linspace(0, 1, 100), 
                                          cascade_b['recall_vs_edges'], 
                                          cascade_b['edges_kept']):
            analysis_data.append({
                'model': 'Cascade Mode B',
                'threshold': thr,
                'recall': recall,
                'edges_kept': edges_kept,
                'edges_pruned': 1 - edges_kept
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(analysis_data)
    df.to_csv(os.path.join(output_dir, "threshold_sweep_analysis.csv"), index=False)
    
    # Generate summary statistics
    summary = []
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        # Find thresholds for key recall targets
        for target_recall in [0.90, 0.95, 0.98]:
            # Find threshold that achieves at least target recall
            valid = model_df[model_df['recall'] >= target_recall]
            if not valid.empty:
                best = valid.iloc[valid['edges_kept'].argmax()]  # Most edges while meeting recall target
                summary.append({
                    'model': model,
                    'target_recall': target_recall,
                    'threshold': best['threshold'],
                    'achieved_recall': best['recall'],
                    'edges_kept': best['edges_kept'],
                    'edges_pruned_pct': (1 - best['edges_kept']) * 100
                })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "threshold_sweep_summary.csv"), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("THRESHOLD SWEEP ANALYSIS SUMMARY")
    print("="*80)
    print("\nMinimum thresholds to achieve recall targets while maximizing edges kept:")
    print(summary_df.to_string(index=False))
    
    return df, summary_df


def run_comprehensive_evaluation(*, model, mlp_stage1, mlp_stage2, test_set, chosen_thr, cascade_thr, training_history, args, device, seed=None):
    # === Comprehensive evaluation ===
    print("\n" + "="*80)
    print("COMPREHENSIVE CASCADE EVALUATION WITH CONTINUOUS SCORES")
    print("="*80)
    import joblib
    joblib.dump(mlp_stage2, os.path.join(args.out, "cascade_stage2.pkl"))
    
    # Get base GNN scores on test set
    base_test_scores = []
    base_test_labels = []
    for d in tqdm(test_set, desc="Running base GNN on test set"):
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            logits = model(b.x, b.edge_index, b.edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
            base_test_scores.extend(probs.tolist())
            base_test_labels.extend(d.y.cpu().numpy().tolist())
    
    base_test_scores = np.array(base_test_scores)
    base_test_labels = np.array(base_test_labels)
    
    # Compute metrics for base GNN
    base_metrics = compute_metrics_continuous(
        base_test_labels,
        base_test_scores,
        chosen_thr
    )

    
    # Build cascade test datasets
    Xtest_stage1, ytest_stage1, _ = build_cascade_dataset(test_set, model, prob_cut=0.4)
    cascade_metrics_continuous = None
    
    if len(Xtest_stage1) > 0:
        p_stage1 = mlp_stage1.predict_proba(Xtest_stage1)[:, 1]
        Xtest_stage2 = np.hstack([Xtest_stage1, p_stage1.reshape(-1, 1)])
        cascade_scores = mlp_stage2.predict_proba(Xtest_stage2)[:, 1]
        
        cascade_metrics_continuous = compute_metrics_continuous(
            ytest_stage1, cascade_scores, cascade_thr
        )
    
    # Evaluate cascade in TWO MODES
    cascade_results = evaluate_cascade_modes(
        base_model=model,
        cascade_mlp=mlp_stage2,
        test_set=test_set,
        stage1_thr=0.4,
        cascade_thr=cascade_thr,
        device=device,
        mlp_stage1=mlp_stage1
    )
    cascade_ablation = evaluate_cascade_stages(
        base_model=model,
        mlp_stage1=mlp_stage1,
        mlp_stage2=mlp_stage2,
        test_set=test_set,
        device=device,
        recall_targets=[0.95, 0.98]
    )
    baseline_global = evaluate_global_threshold(model, val_set=test_set, device=device, recall_targets=[0.95, 0.98])
    baseline_topk = evaluate_topk_edges(model, test_set, device=device, k_values=[5, 10, 20])
    baseline_knn = knn_candidate_edges(dataset=test_set, k_values=[5, 10, 20])
    
    # Save all results
    results_summary =  {
        'base_gnn': base_metrics,
        'cascade_final': cascade_metrics_continuous,
        'cascade_modes': cascade_results,
        'cascade_ablation': cascade_ablation,
        'baselines': {
            'global_threshold': baseline_global,
            'topk': baseline_topk,
            'knn': baseline_knn
        },
        'training_history': training_history,
        'chosen_threshold': float(chosen_thr),
        'seed': seed if seed is not None else args.seed
    }
    

   
    
    with open(os.path.join(args.out, "comprehensive_metrics.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n✅ All metrics saved to: {os.path.join(args.out, 'comprehensive_metrics.json')}")
    
    # Inside run_comprehensive_evaluation (end)
    print("\n--- Generating enhanced PR curves (3 subplots) ---")
    plot_comprehensive_pr_curves(results_summary, args.out)

    print("\n--- Generating enhanced recall vs edges kept plots (3 subplots) ---")
    plot_comprehensive_recall_vs_edges(results_summary, args.out)

    print("\n--- Analyzing threshold sweep ---")
    threshold_sweep_df, threshold_summary = analyze_threshold_sweep(results_summary, args.out)

    print("\n--- Creating LaTeX table for single seed results ---")
    create_single_seed_latex_table(results_summary, args.out)

    
    return results_summary
