
import os
import math
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np



from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from datetime import datetime



import sys
import os

# Add the current directory to Python path to import from preprocess_data.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from preprocess_data.py
try:
    from latex_table import (
        create_multi_seed_latex_table
    )
    print("✅ Successfully imported from latex table.py")
except ImportError as e:
    print(f"❌ Failed to import from latex table.py: {e}")
    #print("Make sure preprocess_data.py is in the same directory as this script.")




def compute_metrics_continuous(y_true, y_scores, threshold=0.5):
    """
    Compute all metrics correctly using continuous scores for AUC.
    Returns a dictionary with all metrics.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, 
        precision_score, recall_score, f1_score,
        confusion_matrix, accuracy_score,
        precision_recall_curve
    )
    
    # Ensure we have numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. Continuous metrics (ROC-AUC, PR-AUC)
    roc_auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_scores)
    
    # 2. Threshold-based metrics at chosen threshold
    y_pred = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # 3. Generate precision-recall curve data
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_true, y_scores)
    
    # 4. Compute recall vs edges kept (for threshold sweep)
    edges_kept = []
    recall_values = []
    for thr in np.linspace(0, 1, 101):
        preds = (y_scores >= thr).astype(int)
        edges_kept.append(np.mean(preds))  # % of edges kept
        recall_values.append(recall_score(y_true, preds, zero_division=0))
    
    n_edges = len(y_true)
    n_positives = int(np.sum(y_true))
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'precisions_curve': precisions_curve.tolist(),
        'recalls_curve': recalls_curve.tolist(),
        'thresholds_curve': thresholds_curve.tolist(),
        'edges_kept': edges_kept,
        'recall_vs_edges': recall_values,
        'threshold_used': float(threshold),
        'n_edges': n_edges,
        'n_positives': n_positives
    }












def knn_candidate_edges(dataset, k_values=[5, 10, 20]):
    """
    Generate candidate edges using k-NN (distance matrix) and evaluate.
    Returns a dict keyed by k value.
    """
    results = {}
    
    for k in k_values:
        all_labels = []
        all_preds = []
        for data in dataset:
            D = data.D  # distance matrix
            edges = data.edges_list
            top_edges = set()
            for i in range(data.n_nodes):
                nn_idx = np.argsort(D[i])[1:k+1]  # skip self
                top_edges.update([tuple(sorted((i,j))) for j in nn_idx])
            
            preds = np.array([
                1 if tuple(sorted(e)) in top_edges else 0
                for e in edges
            ])

            all_preds.append(preds)
            all_labels.append(data.y.numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        p, r, f, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        pr_auc = average_precision_score(all_labels, all_preds)
        
        results[k] = {'precision': float(p), 'recall': float(r), 'f1': float(f), 'pr_auc': float(pr_auc)}
    
    return results






def precision_at_fixed_recalls(y_true, y_probs, recall_targets=[0.95, 0.98]):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    results = {}
    for rt in recall_targets:
        mask = recalls >= rt
        if not np.any(mask):
            results[rt] = {'precision': 0.0, 'threshold': 1.0}
        else:
            idx = np.argmax(mask)
            results[rt] = {
                'precision': float(precisions[idx]),
                'threshold': float(thresholds[idx]) if idx < len(thresholds) else 1.0
            }

    return results



def compute_multi_seed_statistics(seed_metrics, output_dir):
    """Compute mean ± std or 95% CI across seeds"""
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    df = pd.DataFrame(seed_metrics)
    
    print(f"\n{'='*80}")
    print("MULTI-SEED STATISTICS (mean ± 95% CI)")
    print(f"{'='*80}")
    
    stats_summary = []
    
    for col in df.columns:
        if col != 'seed':
            values = df[col].dropna()
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                n = len(values)
                if n > 1:
                    # 95% confidence interval using t-distribution
                    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
                    ci_width = ci[1] - ci[0]
                    stats_summary.append({
                        'metric': col,
                        'mean': mean,
                        'std': std,
                        '95%_CI_lower': ci[0],
                        '95%_CI_upper': ci[1],
                        'CI_width': ci_width,
                        'n_seeds': n
                    })
                    
                    print(f"\n{col}:")
                    print(f"  Mean: {mean:.4f}")
                    print(f"  Std:  {std:.4f}")
                    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
                    print(f"  CI Width: {ci_width:.4f}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(os.path.join(output_dir, "multi_seed_statistics.csv"), index=False)
    
    # Create LaTeX table for paper
    create_multi_seed_latex_table(stats_df, output_dir)
    
    return stats_df