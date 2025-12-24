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



import sys
import os

# Add the current directory to Python path to import from preprocess_data.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from preprocess_data.py

# PyG imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")






def plot_training_progress(history, output_dir):
    """Plot comprehensive training progress"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Progress Metrics', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Loss curves
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0,0].set_title('Training & Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    axes[0,1].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    axes[0,1].plot(epochs, history['val_recall'], 'r-', label='Recall', linewidth=2)
    axes[0,1].plot(epochs, history['val_f1'], 'b-', label='F1-Score', linewidth=2)
    axes[0,1].set_title('Validation Metrics')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Score')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    # AUC metrics
    axes[0,2].plot(epochs, history['val_auc_roc'], 'purple', label='ROC-AUC', linewidth=2)
    axes[0,2].plot(epochs, history['val_auc_pr'], 'orange', label='PR-AUC', linewidth=2)
    axes[0,2].set_title('AUC Metrics')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('AUC Score')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_ylim(0, 1)
    
    # Learning rate
    axes[1,0].plot(epochs, history['learning_rate'], 'm-', linewidth=2)
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    
    # Precision-Recall scatter
    axes[1,1].scatter(history['val_recall'], history['val_precision'], 
                     c=epochs, cmap='viridis', alpha=0.7, s=50)
    axes[1,1].set_title('Precision-Recall Evolution')
    axes[1,1].set_xlabel('Recall')
    axes[1,1].set_ylabel('Precision')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlim(0, 1)
    axes[1,1].set_ylim(0, 1)
    
    # F1 score progression
    axes[1,2].plot(epochs, history['val_f1'], 'b-', linewidth=3)
    axes[1,2].set_title('F1-Score Progression')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_ylabel('F1-Score')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'plots', 'training_progress.pdf'), bbox_inches='tight')
    plt.close()
    
    print("Saved training progress plots")



def plot_cascade_progress(history, output_dir, stage):
    """Plot cascade training progress across different pos_weight values"""
    df = pd.DataFrame(history)
    stage_df = df[df['stage'] == stage]
    
    if stage_df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Cascade Stage {stage} Training Progress', fontsize=16, fontweight='bold')
    
    # Precision by pos_weight
    axes[0,0].plot(stage_df['pos_weight'], stage_df['precision'], 'go-', linewidth=2, markersize=8)
    axes[0,0].set_title('Precision vs Positive Weight')
    axes[0,0].set_xlabel('Positive Weight')
    axes[0,0].set_ylabel('Precision')
    axes[0,0].grid(True, alpha=0.3)
    
    # Recall by pos_weight
    axes[0,1].plot(stage_df['pos_weight'], stage_df['recall'], 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_title('Recall vs Positive Weight')
    axes[0,1].set_xlabel('Positive Weight')
    axes[0,1].set_ylabel('Recall')
    axes[0,1].grid(True, alpha=0.3)
    
    # F1 by pos_weight
    axes[1,0].plot(stage_df['pos_weight'], stage_df['f1'], 'bo-', linewidth=2, markersize=8)
    axes[1,0].set_title('F1-Score vs Positive Weight')
    axes[1,0].set_xlabel('Positive Weight')
    axes[1,0].set_ylabel('F1-Score')
    axes[1,0].grid(True, alpha=0.3)
    
    # 3D scatter of precision, recall, f1
    scatter = axes[1,1].scatter(stage_df['precision'], stage_df['recall'], 
                               c=stage_df['f1'], s=100, cmap='viridis', alpha=0.7)
    axes[1,1].set_title('Precision-Recall-F1 Tradeoff')
    axes[1,1].set_xlabel('Precision')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add annotations for pos_weight values
    for i, row in stage_df.iterrows():
        axes[1,1].annotate(f"w={row['pos_weight']}", 
                          (row['precision'], row['recall']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=axes[1,1], label='F1-Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'cascade_stage_{stage}_progress.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cascade stage {stage} progress plot")



def plot_comprehensive_pr_curves(results_summary, output_dir):
    """Create comprehensive PR curves with threshold annotations"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Base GNN PR curve
    if 'base_gnn' in results_summary:
        base = results_summary['base_gnn']
        axes[0].plot(base['recalls_curve'], base['precisions_curve'], 
                    label=f"Base GNN (AUC={base['pr_auc']:.3f})", linewidth=2)
        # Mark the chosen threshold
        axes[0].scatter(base['recall'], base['precision'], 
                       color='red', s=100, zorder=5, 
                       label=f"Threshold={base['threshold_used']:.3f}")
        axes[0].set_title('Base GNN Precision-Recall Curve', fontweight='bold')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
    
    # Plot 2: Cascade Mode A PR curve
    if 'cascade_modes' in results_summary and 'mode_a' in results_summary['cascade_modes']:
        cascade_a = results_summary['cascade_modes']['mode_a']
        axes[1].plot(cascade_a['recalls_curve'], cascade_a['precisions_curve'],
                    label=f"Cascade Mode A (AUC={cascade_a['pr_auc']:.3f})", linewidth=2)
        axes[1].scatter(cascade_a['recall'], cascade_a['precision'], 
                       color='red', s=100, zorder=5,
                       label=f"Threshold={cascade_a['threshold_used']:.3f}")
        axes[1].set_title('Cascade Mode A Precision-Recall Curve', fontweight='bold')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
    
    # Plot 3: Cascade Mode B PR curve
    if 'cascade_modes' in results_summary and 'mode_b' in results_summary['cascade_modes']:
        cascade_b = results_summary['cascade_modes']['mode_b']
        axes[2].plot(cascade_b['recalls_curve'], cascade_b['precisions_curve'],
                    label=f"Cascade Mode B (AUC={cascade_b['pr_auc']:.3f})", linewidth=2)
        axes[2].scatter(cascade_b['recall'], cascade_b['precision'], 
                       color='red', s=100, zorder=5,
                       label=f"Threshold={cascade_b['threshold_used']:.3f}")
        axes[2].set_title('Cascade Mode B Precision-Recall Curve', fontweight='bold')
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plots", "comprehensive_pr_curves.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "comprehensive_pr_curves.pdf"),
                bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive PR curves saved")



def plot_comprehensive_recall_vs_edges(results_summary, output_dir):
    """Create comprehensive recall vs edges kept plots"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Base GNN
    if 'base_gnn' in results_summary:
        base = results_summary['base_gnn']
        edges_kept = base['edges_kept']
        recall_values = base['recall_vs_edges']
        
        axes[0].plot(edges_kept, recall_values, linewidth=2, label='Base GNN')
        # Mark the chosen threshold point
        chosen_idx = np.abs(np.array(edges_kept) - (1 - base['threshold_used'])).argmin()
        axes[0].scatter(edges_kept[chosen_idx], recall_values[chosen_idx], 
                       color='red', s=100, zorder=5,
                       label=f"Recall={recall_values[chosen_idx]:.3f}, Edges={edges_kept[chosen_idx]:.3f}")
        
        axes[0].set_title('Base GNN: Recall vs %Edges Kept', fontweight='bold')
        axes[0].set_xlabel('Fraction of Edges Kept')
        axes[0].set_ylabel('Recall')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
    
    # Plot 2: Cascade Mode A
    if 'cascade_modes' in results_summary and 'mode_a' in results_summary['cascade_modes']:
        cascade_a = results_summary['cascade_modes']['mode_a']
        edges_kept = cascade_a['edges_kept']
        recall_values = cascade_a['recall_vs_edges']
        
        axes[1].plot(edges_kept, recall_values, linewidth=2, label='Cascade Mode A')
        chosen_idx = np.abs(np.array(edges_kept) - (1 - cascade_a['threshold_used'])).argmin()
        axes[1].scatter(edges_kept[chosen_idx], recall_values[chosen_idx], 
                       color='red', s=100, zorder=5,
                       label=f"Recall={recall_values[chosen_idx]:.3f}, Edges={edges_kept[chosen_idx]:.3f}")
        
        axes[1].set_title('Cascade Mode A: Recall vs %Edges Kept', fontweight='bold')
        axes[1].set_xlabel('Fraction of Edges Kept')
        axes[1].set_ylabel('Recall')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
    
    # Plot 3: Cascade Mode B
    if 'cascade_modes' in results_summary and 'mode_b' in results_summary['cascade_modes']:
        cascade_b = results_summary['cascade_modes']['mode_b']
        edges_kept = cascade_b['edges_kept']
        recall_values = cascade_b['recall_vs_edges']
        
        axes[2].plot(edges_kept, recall_values, linewidth=2, label='Cascade Mode B')
        chosen_idx = np.abs(np.array(edges_kept) - (1 - cascade_b['threshold_used'])).argmin()
        axes[2].scatter(edges_kept[chosen_idx], recall_values[chosen_idx], 
                       color='red', s=100, zorder=5,
                       label=f"Recall={recall_values[chosen_idx]:.3f}, Edges={edges_kept[chosen_idx]:.3f}")
        
        axes[2].set_title('Cascade Mode B: Recall vs %Edges Kept', fontweight='bold')
        axes[2].set_xlabel('Fraction of Edges Kept')
        axes[2].set_ylabel('Recall')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plots", "comprehensive_recall_vs_edges.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "comprehensive_recall_vs_edges.pdf"),
                bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive recall vs edges plots saved")




def generate_multi_seed_plots(all_results, output_dir):
    """Generate plots showing variation across seeds"""
    import matplotlib.pyplot as plt
    
    # Plot 1: Metric variations across seeds
    metrics_to_plot = [
        ('base_roc_auc', 'Base ROC-AUC'),
        ('base_pr_auc', 'Base PR-AUC'),
        ('base_f1', 'Base F1'),
        ('cascade_b_roc_auc', 'Cascade B ROC-AUC'),
        ('cascade_b_pr_auc', 'Cascade B PR-AUC'),
        ('cascade_b_f1', 'Cascade B F1'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric_key, metric_name) in enumerate(metrics_to_plot):
        if i < len(axes):
            # Extract metric values across seeds
            values = []
            for result in all_results:
                if metric_key.startswith('base'):
                    if 'base_gnn' in result:
                        values.append(result['base_gnn'][metric_key.split('_', 1)[1]])
                elif metric_key.startswith('cascade_b'):
                    if 'cascade_modes' in result and 'mode_b' in result['cascade_modes']:
                        metric_name_clean = metric_key.replace('cascade_b_', '')
                        values.append(result['cascade_modes']['mode_b'][metric_name_clean])

            
            if values:
                axes[i].bar(range(len(values)), values, alpha=0.7)
                axes[i].axhline(y=np.mean(values), color='r', linestyle='-', 
                               label=f'Mean: {np.mean(values):.3f}')
                axes[i].axhline(y=np.mean(values) + np.std(values), color='r', linestyle='--', 
                               alpha=0.5, label=f'±1 std')
                axes[i].axhline(y=np.mean(values) - np.std(values), color='r', linestyle='--', alpha=0.5)
                axes[i].set_title(f'{metric_name} across {len(values)} seeds')
                axes[i].set_xlabel('Seed Index')
                axes[i].set_ylabel(metric_name.split()[-1])
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plots", "multi_seed_metric_variations.png"), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "plots", "multi_seed_metric_variations.pdf"),
                bbox_inches='tight')
    plt.close()
    
    print(f"✅ Multi-seed variation plots saved")