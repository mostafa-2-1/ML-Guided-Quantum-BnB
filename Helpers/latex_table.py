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

def create_single_seed_latex_table(results_summary, output_dir):
    """Create LaTeX table for single seed results"""
    
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Edge Classification Performance (Single Seed)}
\\label{tab:single-seed-results}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Metric} & \\textbf{Base GNN} & \\textbf{Cascade Mode A} & \\textbf{Cascade Mode B} \\\\
\\midrule
"""
    
    # Extract metrics
    base = results_summary.get('base_gnn', {})
    cascade_a = results_summary.get('cascade_modes', {}).get('mode_a', {})
    cascade_b = results_summary.get('cascade_modes', {}).get('mode_b', {})
    
    metrics = [
        ('ROC-AUC', 'roc_auc'),
        ('PR-AUC', 'pr_auc'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1-Score', 'f1')
    ]
    
    for metric_name, metric_key in metrics:
        base_val = base.get(metric_key, '--')
        cascade_a_val = cascade_a.get(metric_key, '--')
        cascade_b_val = cascade_b.get(metric_key, '--')
        
        if base_val != '--':
            base_str = f"{base_val:.3f}"
        else:
            base_str = '--'
            
        if cascade_a_val != '--':
            cascade_a_str = f"{cascade_a_val:.3f}"
        else:
            cascade_a_str = '--'
            
        if cascade_b_val != '--':
            cascade_b_str = f"{cascade_b_val:.3f}"
        else:
            cascade_b_str = '--'
        
        latex_table += f"{metric_name} & {base_str} & {cascade_a_str} & {cascade_b_str} \\\\\n"
    
    # Add filter rate for Cascade Mode B
    filter_rate = results_summary.get('cascade_modes', {}).get('filter_rate', '--')
    if filter_rate != '--':
        latex_table += f"\\midrule\nFilter Rate & -- & -- & {filter_rate:.3f} \\\\\n"
    
    latex_table += """\\bottomrule
    \\end{tabular}
    \\end{table}
    """
    
    # Save LaTeX table
    tex_path = os.path.join(output_dir, "single_seed_latex_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)
    
    print(f"✅ LaTeX table saved to: {tex_path}")
    print("\nLaTeX table preview:")
    print(latex_table)





def create_multi_seed_latex_table(stats_df, output_dir):
    """
    Create LaTeX table showing mean ± 95% CI for multi-seed experiments.
    Columns: Metric | Base GNN | Cascade Mode A | Cascade Mode B | Filter Rate
    """
    import os

    # Define metrics and display names
    metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']
    metric_names = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1-Score']

    # Start LaTeX table
    latex = r"""\begin{table}[htbp]
\centering
\caption{Multi-Seed Performance (Mean ± 95\% CI)}
\label{tab:multi-seed-results}
\begin{tabular}{lcccc}
\toprule
Metric & Base GNN & Cascade Mode A & Cascade Mode B & Filter Rate \\
\midrule
"""

    for key, name in zip(metrics, metric_names):
        # Helper to get mean ± CI string for each mode
        def get_val(mode_prefix):
            col = f"{mode_prefix}_{key}"
            row = stats_df[stats_df['metric'] == col]
            if not row.empty:
                mean = row['mean'].values[0]
                lo = row['95%_CI_lower'].values[0]
                hi = row['95%_CI_upper'].values[0]
                return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
            else:
                return "--"

        base_str = get_val('base')
        a_str = get_val('cascade_a')
        b_str = get_val('cascade_b')
        filter_str = get_val('filter_rate') if key == 'f1' else "--"  # Only show filter rate at last row

        latex += f"{name} & {base_str} & {a_str} & {b_str} & {filter_str} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # Save LaTeX file
    tex_path = os.path.join(output_dir, "multi_seed_latex_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex)

    print(f"✅ Multi-seed LaTeX table saved to: {tex_path}")
