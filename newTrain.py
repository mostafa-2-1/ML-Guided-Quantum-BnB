#!/usr/bin/env python3
"""
tsp_gnn_pipeline.py

End-to-end single-file pipeline (CPU) for edge-level classification in TSP instances.

Features:
- Parses .tsp (TSPLIB-style with NODE_COORD_SECTION) and .opt.tour files
- Skips .tsp without matching .opt.tour
- Builds full graph for n <= 300, k-NN (k=25) for n > 300; distances normalized per-instance
- Computes node features (coords, deg, avg kNN dist, betweenness) and edge features (d, dx, dy, rank, in_MST, etc.)
- GNN encoder: GraphSAGE (4 layers, ReLU, dropout=0.3) -> node embeddings
- Edge classifier: concat(node_u_emb, node_v_emb, edge_features) -> MLP (with biases)
- Training: Adam lr=0.001, CosineAnnealingLR scheduler, weighted BCE loss (pos_weight = neg/pos)
- Split: by instances, 80% train, 20% test; plus 10% of training held out as validation for early stopping
- Beam search tour builder using predicted probabilities mixed with inverse distances (0.7 prob + 0.3 inv-dist)
- Prints metrics to console (AUC, Precision, Recall, F1, Accuracy, confusion matrix, average top-k)
- Uses tqdm for progress and prints batch-level debug info every few batches
 
"""

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

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# PyG imports
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")

# Focal Loss (not used by defaul)
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, logits, targets):
        # logits: float tensor (N,), targets: float tensor {0,1}
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        w = self.alpha * targets + (1.0 - targets)  
        loss = - w * (1 - pt) ** self.gamma * torch.log(pt + 1e-12)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# Pos_weight scaling
def finetune_with_scaled_pos_weight(model_template, train_set, val_set, base_state, scale, epochs, lr, device, in_node_feats, in_edge_feats, hidden_dim, n_layers, dropout):
    # model: EdgeGNN instance (uninitialized weights will be loaded from base_state)
    model = model_template(in_node_feats, in_edge_feats, hidden_dim, n_layers, dropout).to(device)
    model.load_state_dict({k: v.clone().to(device) for k, v in base_state.items()})
    model.to(device)
    pos_weight = compute_pos_weight(train_set)
    scaled = float(pos_weight * scale)
    criterion_local = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(scaled, dtype=torch.float))
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val = None
    best_model = None
    for ep in range(epochs):
        _ = train_one_epoch_custom(model, optimizer, criterion_local, train_set, batch_size=1, device=device)
        val_res, val_labels, val_preds = evaluate_model_custom(model, val_set, batch_size=1, device=device, criterion=criterion_local)
        scheduler.step()
        
        thr = choose_threshold_for_precision_target(val_labels, val_preds, min_recall=0.85)
        prec = precision_at_threshold(val_labels, val_preds, thr)
        rec = recall_at_threshold(val_labels, val_preds, thr)
        if best_val is None or prec > best_val['precision']:
            best_val = {'precision': prec, 'recall': rec, 'thr': thr, 'epoch': ep}
            best_model = {k:v.cpu() for k,v in model.state_dict().items()}
    return best_model, best_val


from sklearn.metrics import precision_recall_curve

def choose_threshold_for_precision_target(labels, probs, min_recall=0.85):
    prec, rec, thr = precision_recall_curve(labels, probs)
    # find thresholds where rec >= min_recall, pick the one with highest precision
    valid = np.where(rec >= min_recall)[0]
    if len(valid) == 0:
        # fallback: return best-F1 threshold
        f1 = 2*(prec*rec)/(prec+rec+1e-12)
        idx = np.nanargmax(f1)
        return thr[idx] if idx < len(thr) else 0.5
    idx = valid[np.argmax(prec[valid])]
    return thr[idx] if idx < len(thr) else 0.5

def precision_at_threshold(labels, probs, thr):
    preds = (probs >= thr).astype(int)
    from sklearn.metrics import precision_score, recall_score
    return float(precision_score(labels, preds)), float(recall_score(labels, preds))
from sklearn.metrics import precision_score, recall_score
def recall_at_threshold(y_true, y_pred_probs, thr=0.5):
    """Compute recall given threshold."""
    y_pred = (y_pred_probs >= thr).astype(int)
    return recall_score(y_true, y_pred, zero_division=0)
# Create cascade datasets based on finetuned models
def build_cascade_dataset(data_list, model, prob_cut=0.5, device='cpu'):
    # returns X (num_examples x feat_dim), y (labels)
    Xs = []
    Ys = []
    for d in data_list:
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            logits = model(b.x, b.edge_index, b.edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
        edges = d.edges_list
        # compute extra edge features aligned with edges: d_norm, dx, dy, rank_mean, in_MST etc.
        # reuse the edge attributes (d.edge_attr) if available; but ensure alignment
        edge_attrs = d.edge_attr.cpu().numpy() if hasattr(d, 'edge_attr') else None
        for idx, p in enumerate(probs):
            if p >= prob_cut:
                feat = []
                feat.append(p)
                if edge_attrs is not None:
                    feat.extend(edge_attrs[idx].tolist())
                # optionally add node features: degs, avg_knn, bc by indexing
                Xs.append(feat)
                Ys.append(int(d.y.cpu().numpy()[idx]))
    X = np.array(Xs, dtype=float)
    y = np.array(Ys, dtype=int)
    return X, y
#Train a small MLP classifier on cascade dataset
from sklearn.neural_network import MLPClassifier
# def train_cascade_mlp(X_train, y_train, X_val, y_val):
#     mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', alpha=1e-4, max_iter=500)
#     mlp.fit(X_train, y_train)
#     # pick threshold on val to meet precision/recall constraints
#     val_probs = mlp.predict_proba(X_val)[:,1]
#     thr = choose_threshold_for_precision_target(y_val, val_probs, min_recall=0.85)
#     return mlp, thr
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
import numpy as np

def train_cascade_mlp(X_train, y_train, X_val, y_val, target_prec=0.80, min_recall=0.80):
    print("\n=== Training precision-recall balanced MLP cascade (enhanced) ===")

    best_model, best_thr = None, None
    best_p, best_r, best_f1 = 0, 0, 0
    best_posw = None

    # Sweep over several positive weights to rebalance class importance
    for pos_weight in [1.0, 1.5, 2.0, 3.0, 4.0]:
        print(f"\n--- Trying pos_weight={pos_weight} ---")
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=400,
            learning_rate_init=1e-3,
            random_state=42
        )

        # Assign sample weights (simulate class_weight)
        sample_weight = np.ones(len(y_train))
        sample_weight[y_train == 1] = pos_weight

        mlp.fit(X_train, y_train, sample_weight=sample_weight)
        probs = mlp.predict_proba(X_val)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = [2*p*r/(p+r+1e-8) for p, r in zip(precisions, recalls)]

        # Default values in case no threshold meets targets
        p, r, thr, f1 = 0, 0, 0.5, 0

        # Find all thresholds that meet both P and R targets
        candidates = [(p, r, t, f1) for p, r, t, f1 in zip(precisions, recalls, thresholds, f1_scores)
                      if p >= target_prec and r >= min_recall]

        if candidates:
            # Pick candidate with best combined F1
            best_cand = max(candidates, key=lambda x: x[3])
            p, r, thr, f1 = best_cand
            print(f"pos_weight={pos_weight} → thr={thr:.3f}, P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
        else:
            # fallback to global F1-optimal threshold
            best_idx = int(np.argmax(f1_scores))
            p, r, thr, f1 = precisions[best_idx], recalls[best_idx], thresholds[best_idx], f1_scores[best_idx]
            print(f"pos_weight={pos_weight} → best F1={f1:.3f} (P={p:.3f}, R={r:.3f}, thr={thr:.3f})")

        # Safe comparison
        if f1 > best_f1 or (p >= best_p and r >= best_r):
            best_model, best_thr, best_p, best_r, best_f1, best_posw = mlp, thr, p, r, f1, pos_weight

    # Summary
    if best_model is None:
        raise RuntimeError("No valid cascade configuration reached target metrics.")

    print("\n   Best cascade model:")
    print(f"   pos_weight={best_posw}, threshold={best_thr:.3f}")
    print(f"   Validation Precision={best_p:.3f}, Recall={best_r:.3f}, F1={best_f1:.3f}")

    # Optional fine-tuning around best threshold
    if best_p < target_prec or best_r < min_recall:
        print("  Fine-tuning threshold for better balance...")
        probs = best_model.predict_proba(X_val)[:, 1]
        scan_range = np.linspace(max(0, best_thr - 0.1), min(1, best_thr + 0.1), 20)
        for thr in scan_range:
            preds = (probs >= thr).astype(int)
            tp = np.sum((y_val == 1) & (preds == 1))
            fp = np.sum((y_val == 0) & (preds == 1))
            fn = np.sum((y_val == 1) & (preds == 0))
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if prec >= target_prec and rec >= min_recall and f1 > best_f1:
                best_thr, best_p, best_r, best_f1 = thr, prec, rec, f1

        print(f" After fine-tune: P={best_p:.3f}, R={best_r:.3f}, thr={best_thr:.3f}")

    return best_model, best_thr






#Search wrapper
def search_for_precision_recall(target_prec_low, target_prec_high, min_recall,
                              model_template, train_set, val_set, best_state_saved, in_node_feats, in_edge_feats,
                              hidden_dim, n_layers, dropout, device):
    model = model_template(in_node_feats, in_edge_feats, hidden_dim, n_layers, dropout).to(device)
    scales = [1.0, 0.75, 0.5, 0.33, 0.25]
    results = []
    for s in scales:
        print(f"Trying pos_weight scale={s}")
        model_state, best_val = finetune_with_scaled_pos_weight(model_template, train_set, val_set, best_state_saved, 
            scale=s, epochs=8, lr=1e-4, device=device, in_node_feats=in_node_feats, in_edge_feats=in_edge_feats,
            hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        # load into model and evaluate on val
        model.load_state_dict(model_state); model.to(device)
        val_res, val_labels, val_probs = evaluate_model_custom(model, val_set, batch_size=1, device=device, criterion=None)
        thr = choose_threshold_for_precision_target(val_labels, val_probs, min_recall=min_recall)
        p, r = precision_at_threshold(val_labels, val_probs, thr)
        print(f"scale={s} -> val P={p:.4f}, R={r:.4f} thr={thr:.4f}")
        if p >= target_prec_low and p <= target_prec_high and r > min_recall:
            print("Found model satisfying constraints")
            return model_state, thr
    # fallback to cascade:
    print("No scaled pos_weight hit targets; trying cascade classifier...")
    # build cascade dataset with prob_cut=0.4 
    Xtr, ytr = build_cascade_dataset(train_set, model, prob_cut=0.4)
    Xval, yval = build_cascade_dataset(val_set, model, prob_cut=0.4)
    mlp, thr_mlp = train_cascade_mlp(Xtr, ytr, Xval, yval)
    # validate on val_set aggregated and compute final P/R
    # apply cascade to val and compute metrics -> if meets, return mlp and thr_mlp
    return ('cascade', mlp, thr_mlp)


# -----------------------------
# Parsing utilities
# -----------------------------

def parse_tsp(filepath):
    """Parse a .tsp (TSPLIB) file. Returns dict with name, n, coords (numpy n x 2).
    Assumes NODE_COORD_SECTION present and Euclidean coordinates.
    """
    name = os.path.basename(filepath)
    n = None
    coords = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('NAME'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    name = parts[1].strip()
            if up.startswith('DIMENSION'):
                parts = line.split(':', 1)
                n = int(parts[1].strip())
            if up.startswith('NODE_COORD_SECTION'):
                # read n lines
                for _ in range(n):
                    ln = next(f).strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    idx = int(parts[0]) - 1
                    x = float(parts[1]); y = float(parts[2])
                    coords[idx] = (x, y)
                break
    if n is None:
        raise ValueError(f"DIMENSION not found in {filepath}")
    coords_arr = np.zeros((n,2), dtype=float)
    for i in range(n):
        coords_arr[i] = coords[i]
    return {'name': name, 'n': n, 'coords': coords_arr}


def parse_opt_tour(filepath):
    """Parse a .opt.tour / .tour file and return tour list (0-based indices)
    """
    tour = []
    with open(filepath, 'r') as f:
        started = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('TOUR_SECTION'):
                started = True
                continue
            if not started:
                continue
            if line == '-1' or up == 'EOF':
                break
            parts = line.split()
            for p in parts:
                if p == '-1':
                    break
                try:
                    tour.append(int(p)-1)
                except:
                    pass
    return tour

# -----------------------------
# Graph building & features
# -----------------------------

def pairwise_distances(coords):
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    return D


def build_candidate_edges(coords, full_threshold=300, knn_k=25):
    n = coords.shape[0]
    D = pairwise_distances(coords)
    if n <= full_threshold:
        edges = [(i,j) for i in range(n) for j in range(i+1,n)]
    else:
        # k-NN mutual/unidirectional mix: include top-k neighbors for each node
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        edges_set = set()
        for i in range(n):
            k = min(knn_k, n-1)
            dists, idxs = tree.query(coords[i], k=k+1)
            # idxs includes self at pos 0
            if hasattr(idxs, '__iter__'):
                neighbors = [int(x) for x in idxs if int(x) != i]
            else:
                neighbors = []
            for j in neighbors:
                a,b = (i,j) if i<j else (j,i)
                edges_set.add((a,b))
        edges = sorted(list(edges_set))
    return edges, D


def compute_node_edge_features(coords, edges, D, knn_k=10):
    """Compute node-level and edge-level features.
    Node features: x,y (normalized), degree (in candidate graph), avg_kNN_dist, betweenness
    Edge features: d (normalized), dx, dy, rank_i, rank_j, in_MST, is_knn1_i/j
    Returns: node_feat (n x f1), edge_index (2 x m), edge_attr (m x f2)
    """
    n = coords.shape[0]
    mean_d = D[np.triu_indices(n,1)].mean()
    coords_np = np.array(coords, dtype=float)
    bbox_w = coords_np[:,0].max() - coords_np[:,0].min() + 1e-6
    bbox_h = coords_np[:,1].max() - coords_np[:,1].min() + 1e-6

    # candidates degrees
    deg = np.zeros(n, dtype=int)
    for (i,j) in edges:
        deg[i] += 1; deg[j] += 1

    # kNN avg distt per node
    from scipy.spatial import KDTree
    tree = KDTree(coords_np)
    avg_knn = np.zeros(n, dtype=float)
    for i in range(n):
        kk = min(knn_k+1, n)
        dists, idxs = tree.query(coords_np[i], kk)
        if hasattr(dists, '__iter__'):
            arr = np.array([d for d,idx in zip(dists, idxs) if idx!=i])
            avg_knn[i] = arr.mean() if arr.size>0 else 0.0
        else:
            avg_knn[i] = 0.0

    # Betweennesss centrality (approxd)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i,j) in edges:
        G.add_edge(i,j,weight=float(D[i,j]))
    try:
        bc = np.fromiter(nx.betweenness_centrality(G).values(), dtype=float)
    except Exception:
        # fallback to zeros if too slow
        bc = np.zeros(n, dtype=float)

    # node features matrix
    node_feat = np.zeros((n, 6), dtype=float)
    node_feat[:,0] = coords_np[:,0] / bbox_w
    node_feat[:,1] = coords_np[:,1] / bbox_h
    node_feat[:,2] = deg / (n+1)
    node_feat[:,3] = avg_knn / (mean_d + 1e-12)
    node_feat[:,4] = bc
    node_feat[:,5] = (np.arange(n) / float(n))  # positional index

    # per-node rank matrix
    sorted_idx = np.argsort(D, axis=1)
    rank_matrix = np.empty_like(sorted_idx)
    for i in range(n):
        rank = np.empty(n, dtype=int)
        rank[sorted_idx[i]] = np.arange(n)
        rank_matrix[i] = rank

    # MST membership
    T = nx.minimum_spanning_tree(G, weight='weight')
    T_edges = set(tuple(sorted(e)) for e in T.edges())

    # edge_attr
    edge_index = [[], []]
    edge_attr = []
    for (i,j) in edges:
        d = float(D[i,j])
        feat = [
            d / (mean_d + 1e-12),  # normalized distance
            abs(coords_np[i,0] - coords_np[j,0]) / bbox_w,
            abs(coords_np[i,1] - coords_np[j,1]) / bbox_h,
            float(rank_matrix[i,j]) / float(n),
            float(rank_matrix[j,i]) / float(n),
            1.0 if (i,j) in T_edges else 0.0,
            1.0 if rank_matrix[i,j] <= 1 else 0.0,
            1.0 if rank_matrix[j,i] <= 1 else 0.0,
            (deg[i] + deg[j]) / (2.0 * n)
        ]
        edge_index[0].append(i); edge_index[1].append(j)
        edge_attr.append(feat)
    edge_index = np.array(edge_index, dtype=int)
    edge_attr = np.array(edge_attr, dtype=float)
    return node_feat, edge_index, edge_attr

# -----------------------------
# PyG Data builder
# -----------------------------

def build_pyg_data_from_instance(tsp_path, tour_path, full_threshold=300, knn_k=25, knn_feat_k=10):
    meta = parse_tsp(tsp_path)
    coords = meta['coords']
    n = meta['n']
    # parse tour
    tour = parse_opt_tour(tour_path)
    if not tour or len(tour) != n:
        # invalid tour, skip
        return None
    edges, D = build_candidate_edges(coords, full_threshold=full_threshold, knn_k=knn_k)
    # compute features
    node_feat, edge_index_np, edge_attr = compute_node_edge_features(coords, edges, D, knn_k=knn_feat_k)

    # labels: 1 if edge in tour
    true_edges = set()
    for a,b in zip(tour, tour[1:]):
        true_edges.add(tuple(sorted((a,b))))
    true_edges.add(tuple(sorted((tour[-1], tour[0]))))
    labels = []
    for (i,j) in edges:
        labels.append(1 if (i,j) in true_edges else 0)
    labels = np.array(labels, dtype=float)

    # Build PyG Data
    x = torch.tensor(node_feat, dtype=torch.float)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    # store raw for use
    data.coords = coords
    data.D = D
    data.name = meta['name']
    data.n_nodes = n
    data.edges_list = edges
    data.true_tour = tour
    return data

# -----------------------------
# Model: GraphSAGE encoder + edge MLP classifier
# -----------------------------

class EdgeGNN(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=128, n_layers=4, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        # input projection
        self.input_lin = nn.Linear(in_node_feats, hidden_dim)
        for i in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        # edge classifier: concat(u_emb, v_emb, edge_attr) -> MLP
        edge_input_dim = hidden_dim * 2 + in_edge_feats
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)  
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [N, in_node_feats]
        h = self.input_lin(x)
        for i in range(self.n_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = self.dropout(h)
        # compute edge logits
        # edge_index shape [2, m] with i<j ordering
        src = edge_index[0]
        dst = edge_index[1]
        hu = h[src]
        hv = h[dst]
        edge_input = torch.cat([hu, hv, edge_attr], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(dim=1)
        return logits

# -----------------------------
# Training / evaluation
# -----------------------------

def compute_pos_weight(dataset):
    
    pos = 0; neg = 0
    for d in dataset:
        arr = d.y.numpy()
        pos += int((arr==1).sum())
        neg += int((arr==0).sum())
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def collate_batch(data_list):
    # PyG Batch
    batch = Batch.from_data_list(data_list)
    return batch


def train_one_epoch(model, optimizer, data_list, batch_size=1, device='cpu', verbose_every=20):
    model.train()
    losses = []
    all_logits = []
    all_labels = []
    random.shuffle(data_list)
    n_batches = max(1, len(data_list) // batch_size)
    for idx in range(0, len(data_list), batch_size):
        batch_items = data_list[idx: idx+batch_size]
        batch = collate_batch(batch_items).to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labels = batch.y
        # weighted BCE with pos_weight computed per-epoch externally
        # We will compute pos_weight outside and set criterion each epoch.
        loss = criterion(logits, labels)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        losses.append(loss_value)
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            all_logits.append(probs)
            all_labels.append(labels.cpu().numpy())
        # debug print
        if (idx // batch_size) % verbose_every == 0:
            print(f"[train] batch {idx//batch_size+1}/{n_batches} loss={loss_value:.4f}")
    if len(all_labels)>0:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_logits)
    else:
        all_labels = np.array([]); all_preds = np.array([])
    return np.mean(losses), all_labels, all_preds


def train_one_epoch_custom(model, optimizer, criterion, data_list, batch_size=1, device='cpu', verbose_every=20):
    model.train()
    losses = []
    all_logits = []
    all_labels = []
    random.shuffle(data_list)
    n_batches = max(1, len(data_list) // batch_size)
    for idx in range(0, len(data_list), batch_size):
        batch_items = data_list[idx: idx+batch_size]
        batch = collate_batch(batch_items).to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        labels = batch.y
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            all_logits.append(probs)
            all_labels.append(labels.cpu().numpy())
        if (idx // batch_size) % verbose_every == 0:
            print(f"[fine-tune] batch {idx//batch_size+1}/{n_batches} loss={loss.item():.4f}")
    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_logits)
    else:
        all_labels = np.array([]); all_preds = np.array([])
    return np.mean(losses), all_labels, all_preds


def evaluate_model(model, data_list, batch_size=1, device='cpu'):
    model.eval()
    all_labels = []
    all_preds = []
    losses = []
    with torch.no_grad():
        for idx in range(0, len(data_list), batch_size):
            batch_items = data_list[idx: idx+batch_size]
            batch = collate_batch(batch_items).to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            labels = batch.y
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
    if len(all_labels)>0:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
    else:
        all_labels = np.array([]); all_preds = np.array([])
    # metrics
    res = {}
    if all_labels.size > 0:
        try:
            res['roc_auc'] = float(roc_auc_score(all_labels, all_preds))
        except Exception:
            res['roc_auc'] = float('nan')
        try:
            res['pr_auc'] = float(average_precision_score(all_labels, all_preds))
        except Exception:
            res['pr_auc'] = float('nan')
        yhat = (all_preds >= 0.5).astype(int)
        p,r,f,_ = precision_recall_fscore_support(all_labels, yhat, average='binary', zero_division=0)
        acc = float(accuracy_score(all_labels, yhat))
        cm = confusion_matrix(all_labels, yhat).tolist()
        res.update({'precision': float(p), 'recall': float(r), 'f1': float(f), 'accuracy': acc, 'confusion_matrix': cm})
    else:
        res.update({'roc_auc': float('nan'), 'pr_auc': float('nan'), 'precision':0,'recall':0,'f1':0,'accuracy':0,'confusion_matrix':[[0,0],[0,0]]})
    res['loss'] = float(np.mean(losses) if losses else float('nan'))
    return res, all_labels, all_preds

# Custom model evaluatiomon with external criterion
def evaluate_model_custom(model, data_list, batch_size=1, device='cpu', criterion=None, verbose=False):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for idx in range(0, len(data_list), batch_size):
            batch_items = data_list[idx: idx+batch_size]
            batch = collate_batch(batch_items).to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            labels = batch.y

            if criterion is not None:
                loss = criterion(logits, labels)
                losses.append(loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            all_logits.append(probs)
            all_labels.append(labels.cpu().numpy())

    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_logits)
    else:
        all_labels = np.array([]); all_preds = np.array([])

    avg_loss = np.mean(losses) if len(losses) > 0 else None
    return avg_loss, all_labels, all_preds


# -----------------------------
# Beam search for tour reconstruction
# -----------------------------

def beam_search_tours(data, edge_probs, beam_width=7, max_solutions=3, mix_prob=0.7):
    """
    data: Data object (has coords, D, edges_list, true_tour)
    edge_probs: numpy array aligned with edges_list giving probability for each undirected edge (i<j)
    mix_prob: weight for probability (0..1), remainder used for inverse normalized distance
    Returns list of (cost, tour)
    """
    n = data.n_nodes
    D = data.D
    edges = data.edges_list
    # build map (a,b)->prob
    prob_map = {}
    for (i,j), p in zip(edges, edge_probs):
        prob_map[(i,j)] = p
        prob_map[(j,i)] = p
    # compute neighbor lists sorted by combined score
    maxD = D.max() if D.max()>0 else 1.0
    neigh = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i==j: continue
            p = prob_map.get((i,j), 1e-9)
            invd = 1.0/(1.0 + D[i,j]/maxD)
            score = mix_prob * p + (1.0-mix_prob) * invd
            neigh[i].append((score, j))
        neigh[i].sort(reverse=True, key=lambda x: x[0])
    # beam search
    # seed nodes: choose nodes with highest sum of neighbor scores
    seed_scores = [(sum(s for s,_ in neigh[i][:10]), i) for i in range(n)]
    seed_scores.sort(reverse=True)
    seeds = [i for _,i in seed_scores[:min(len(seed_scores), beam_width//2)]]
    if not seeds:
        seeds = [0]
    beam = [([s], set([s]), 0.0) for s in seeds]
    completed = []
    max_iters = n
    while beam and len(completed) < max_solutions:
        new_beam = []
        for path, visited, score in beam:
            last = path[-1]
            # expand top neighbors
            candidates = neigh[last][:min(len(neigh[last]), 10)]
            for sc, nb in candidates:
                if nb in visited: continue
                new_path = path + [nb]
                new_visited = set(visited); new_visited.add(nb)
                # incremental score uses negative log of prob-like score
                new_score = score - math.log(max(sc, 1e-12)) + D[last, nb]/(maxD+1e-12)
                new_beam.append((new_path, new_visited, new_score))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: x[2])
        beam = new_beam[:beam_width]
        # extract completed
        remaining = []
        for pth, vis, sc in beam:
            if len(pth) == n:
                completed.append((pth, sc))
            else:
                remaining.append((pth, vis, sc))
        beam = remaining[:beam_width]
    # compute costs
    tours = []
    for pth, sc in completed:
        cost = 0.0
        for a,b in zip(pth, pth[1:]):
            cost += D[a,b]
        cost += D[pth[-1], pth[0]]
        tours.append((cost, pth))
    tours.sort(key=lambda x: x[0])
    return tours

# -----------------------------
# Utilities: top-k accuracy etc.
# -----------------------------

def average_topk_accuracy(data_list, preds_list, k=3):
    # For each instance, compute how many true-positive edges are in top-k per node on average
    vals = []
    for data, probs in zip(data_list, preds_list):
        edges = data.edges_list
        n = data.n_nodes
        # build mapping edge->prob
        prob_map = {e: p for e,p in zip(edges, probs)}
        # per node top-k
        found_counts = []
        true_edges = set()
        t = data.true_tour
        for a,b in zip(t, t[1:]): true_edges.add(tuple(sorted((a,b))))
        true_edges.add(tuple(sorted((t[-1], t[0]))))
        for node in range(n):
            # collect incident edges
            inc = [(prob_map.get((min(node,other), max(node,other)), 0.0), (min(node,other), max(node,other))) for other in range(n) if other!=node]
            inc.sort(reverse=True)
            topk = [e for _,e in inc[:k]]
            found = sum(1 for e in topk if e in true_edges)
            found_counts.append(found/float(k))
        vals.append(np.mean(found_counts))
    return float(np.mean(vals)) if vals else 0.0

# -----------------------------
# Main CLI
# -----------------------------

def collect_pairs(folder):
    files = os.listdir(folder)
    tsp_files = [os.path.join(folder,f) for f in files if f.endswith('.tsp')]
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


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='folder with .tsp and .opt.tour (tsplib_data)')
    parser.add_argument('--synthetic_dir', type=str, required=True, help='folder with synthetic instances')
    parser.add_argument('--out', type=str, default='results', help='output folder for logs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--full_threshold', type=int, default=300)
    parser.add_argument('--knn_k', type=int, default=25)
    parser.add_argument('--knn_feat_k', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--beam_width', type=int, default=7)
    parser.add_argument('--mix_prob', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    print('Collecting instances...')
    train_pairs = collect_pairs(args.train_dir)
    synth_pairs = collect_pairs(args.synthetic_dir)
    all_pairs = train_pairs + synth_pairs
    print(f'Found {len(train_pairs)} tsplib instances and {len(synth_pairs)} synthetic instances; total {len(all_pairs)}')

    # Build Data objects for all instances (skip ones missing tour or invalid)
    data_objs = []
    failures = []
    print('Building Data objects (this may take time)...')
    for tsp_path, tour_path in tqdm(all_pairs):
        try:
            d = build_pyg_data_from_instance(tsp_path, tour_path, full_threshold=args.full_threshold, knn_k=args.knn_k, knn_feat_k=args.knn_feat_k)
            if d is not None:
                data_objs.append(d)
        except Exception as e:
            failures.append((tsp_path, str(e)))
    print(f'Built {len(data_objs)} data objects; failed {len(failures)} instances')
    if len(data_objs) == 0:
        raise RuntimeError('No valid instances found. Check folders and formats.')

    # Split by instances: 80% train, 20% test
    idxs = list(range(len(data_objs)))
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=args.seed)
    # further split train -> train/val 90/10
    train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)
    train_set = [data_objs[i] for i in train_idx2]
    val_set = [data_objs[i] for i in val_idx]
    test_set = [data_objs[i] for i in test_idx]

    print(f'Train: {len(train_set)} val: {len(val_set)} test: {len(test_set)}')

    # build model
    in_node = train_set[0].x.shape[1]
    in_edge = train_set[0].edge_attr.shape[1]
    model = EdgeGNN(in_node_feats=in_node, in_edge_feats=in_edge, hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout)
    device = torch.device('cpu')
    model.to(device)

    pos_weight = compute_pos_weight(train_set)
    print(f'Computed pos_weight={pos_weight:.4f} (neg/pos)')

    #  Slightly reduce pos_weight to favor recall
    recall_boost_factor = 0.75   # try 0.5–0.8 range to favor recall
    pos_weight *= recall_boost_factor   
    print(f'Adjusted pos_weight for recall boost: {pos_weight:.4f}')

    global criterion
    # ---  loss function ---
    use_focal_loss = True 

    if use_focal_loss:
        print("Using Focal Loss (recall-focused)")
        criterion = BinaryFocalLoss(gamma=1.0, alpha=1.5)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float))


    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)


    print('Starting training...')

    best_val_metric = -1.0
    best_state_path = os.path.join(args.out, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = train_one_epoch(
            model, optimizer, train_set,
            batch_size=args.batch_size, device=device, verbose_every=10
        )
        val_res, val_labels, val_preds = evaluate_model(
            model, val_set, batch_size=args.batch_size, device=device
        )
        scheduler.step()
        t1 = time.time()

        print(f"Epoch {epoch}/{args.epochs} | time={(t1 - t0):.1f}s | "
              f"train_loss={train_loss:.4f} | val_loss={val_res['loss']:.4f} | "
              f"AUC={val_res['roc_auc']:.4f} | P={val_res['precision']:.4f} | "
              f"R={val_res['recall']:.4f} | F1={val_res['f1']:.4f}")
        print(f"→ Epoch {epoch}: Recall={val_res['recall']:.4f}  Precision={val_res['precision']:.4f}  F1={val_res['f1']:.4f}  AUC={val_res['roc_auc']:.4f}")

        # --- Save best so far (using F1 or ROC-AUC as criterion) ---
        metric_now = val_res["f1"] 
        if metric_now > best_val_metric:
            best_val_metric = metric_now
            torch.save(model.state_dict(), best_state_path)
            print(f" New best model saved (epoch {epoch}, F1={metric_now:.4f})")

    # Load the best model after training
    model.load_state_dict(torch.load(best_state_path, map_location=device))
    print(f"\nTraining finished. Best validation F1={best_val_metric:.4f}")


    # === Threshold tuning on validation set ===
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    print("\n=== Threshold tuning (recall-prioritized) ===")
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)

    prec, rec, thr = precision_recall_curve(val_labels, val_preds)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-12)

    target_recall = 0.95
    valid_idxs = np.where(rec >= target_recall)[0]

    if len(valid_idxs) > 0:
        idx = valid_idxs[np.argmax(prec[valid_idxs])]
        chosen_thr = thr[idx] if idx < len(thr) else 0.5
        print(f" Threshold for recall≥{target_recall:.2f}: thr={chosen_thr:.4f}, "
              f"P={prec[idx]:.4f}, R={rec[idx]:.4f}, F1={f1[idx]:.4f}")
    else:
        chosen_thr = 0.5
        print(f"No threshold achieves recall≥{target_recall:.2f}; using 0.5 default.")

    # === Evaluate base GNN on test set using chosen threshold ===
    print("\n=== Evaluating base GNN on TEST set using chosen threshold ===")
    # Ensure we have the best weights loaded
    model.load_state_dict(torch.load(best_state_path, map_location=device))
    model.to(device)

    # Evaluate model (returns metrics using default 0.5 threshold in evaluate_model)
    test_res, test_labels, test_preds = evaluate_model(model, test_set, batch_size=args.batch_size, device=device)

    # Apply chosen threshold to get binary predictions and classification metrics
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    test_preds_arr = np.array(test_preds)
    test_labels_arr = np.array(test_labels)

    preds_bin = (test_preds_arr >= chosen_thr).astype(int)  
    prec_test = precision_score(test_labels_arr, preds_bin, zero_division=0)
    rec_test = recall_score(test_labels_arr, preds_bin, zero_division=0)
    f1_test = f1_score(test_labels_arr, preds_bin, zero_division=0)
    cm_test = confusion_matrix(test_labels_arr, preds_bin)

    print(f"[Base GNN @ thr={chosen_thr:.4f}] Test Precision={prec_test:.4f} Recall={rec_test:.4f} F1={f1_test:.4f}")
    # Save base model tuned with chosen threshold (metadata)
    torch.save(model.state_dict(), os.path.join(args.out, "best_model_threshold.pt"))
    with open(os.path.join(args.out, "best_threshold.json"), "w") as f:
        json.dump({"threshold": float(chosen_thr),
                    "precision": float(prec_test),
                    "recall": float(rec_test),
                    "f1": float(f1_test)}, f, indent=2)
    print(f" Saved recall-tuned base GNN to best_model_threshold.pt")

    print("Confusion matrix (Test):")
    print(cm_test)

    # Keep test_res available for later prints
    test_res['precision'] = float(prec_test)
    test_res['recall'] = float(rec_test)
    test_res['f1'] = float(f1_test)
    test_res['confusion_matrix'] = cm_test.tolist()


    print("\n=== Skipping fine-tuning — training cascade classifier directly ===")

    # Load best saved GNN weights
    model.load_state_dict(torch.load(best_state_path, map_location=device))
    model.to(device)

    # Build cascade datasets from GNN predictions
    Xtr, ytr = build_cascade_dataset(train_set, model, prob_cut=0.4)
    Xval, yval = build_cascade_dataset(val_set, model, prob_cut=0.4)
    Xtest, ytest = build_cascade_dataset(test_set, model, prob_cut=0.4)

    # === Stage 1: Train the first cascade MLP ===
    mlp_stage1, thr1 = train_cascade_mlp(Xtr, ytr, Xval, yval)
    import joblib
    joblib.dump(mlp_stage1, os.path.join(args.out, "cascade_stage1.pkl"))
    with open(os.path.join(args.out, "cascade_stage1_thr.json"), "w") as f:
        json.dump({"threshold": float(thr1)}, f, indent=2)
    print(" Saved cascade Stage 1 MLP and threshold.")

    # === Stage 2: Refinement cascade ===
    # Get probabilities from stage 1
    p_tr   = mlp_stage1.predict_proba(Xtr)[:, 1]
    p_val  = mlp_stage1.predict_proba(Xval)[:, 1]
    p_test = mlp_stage1.predict_proba(Xtest)[:, 1]   

    # Add them as an extra input feature
    Xtr2   = np.hstack([Xtr,  p_tr.reshape(-1, 1)])
    Xval2  = np.hstack([Xval, p_val.reshape(-1, 1)])
    Xtest2 = np.hstack([Xtest, p_test.reshape(-1, 1)]) 

    print("\n=== Training second-stage cascade (refinement) ===")
    mlp_stage2, thr2 = train_cascade_mlp(Xtr2, ytr, Xval2, yval)

    # === Evaluate final cascade on test set ===
    print("\n=== Evaluating final (Stage 2) cascade classifier on test edges ===")
    p_test2 = mlp_stage2.predict_proba(Xtest2)[:, 1]   
    preds_test2 = (p_test2 >= thr2).astype(int)

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

    p  = precision_score(ytest, preds_test2)    
    r  = recall_score(ytest, preds_test2)
    f1 = f1_score(ytest, preds_test2)
    cm = confusion_matrix(ytest, preds_test2)

    print(f"\n[Final Cascade Stage 2] Precision={p:.4f} Recall={r:.4f} F1={f1:.4f}")
    joblib.dump(mlp_stage2, os.path.join(args.out, "cascade_stage2.pkl"))
    with open(os.path.join(args.out, "cascade_stage2_thr.json"), "w") as f:
        json.dump({"threshold": float(thr2),
                   "precision": float(p),
                   "recall": float(r),
                   "f1": float(f1)}, f, indent=2)
    print(" Saved final cascade Stage 2 MLP and threshold.")

    print("Confusion matrix:\n", cm)

    # show candidate threshold for 0.85 precision
    precisions, recalls, thresholds = precision_recall_curve(ytest, p_test2)    
    for p_, r_, t_ in zip(precisions, recalls, thresholds):
        if p_ >= 0.85:
            print(f"\nCandidate threshold for 0.85 precision:")
            print(f"Precision={p_:.3f}, Recall={r_:.3f}, Threshold={t_:.3f}")
            break


    # === Log base test metrics as well ===
    print('\n--- Base model test set results (for comparison) ---')
    print('Loss:', test_res['loss'])
    print('ROC AUC:', test_res['roc_auc'])
    print('PR AUC:', test_res['pr_auc'])
    print('Precision:', test_res['precision'])
    print('Recall:', test_res['recall'])
    print('F1:', test_res['f1'])
    print('Accuracy:', test_res['accuracy'])
    print('Confusion matrix:', test_res['confusion_matrix'])

    # compute average top-k (k=3) on test
    # need per-instance predictions -> run model on each instance to get per-edge probs
    per_inst_preds = []
    for d in test_set:
        model.eval()
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            logits = model(b.x, b.edge_index, b.edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
            per_inst_preds.append(probs)
    avg_top3 = average_topk_accuracy(test_set, per_inst_preds, k=3)
    print('Average top-3 per-node accuracy (test):', avg_top3)

    # For each test instance, run beam search to try to reconstruct tour
    print('\nRunning beam search reconstruction on test instances...')
    recon_results = []
    for d, probs in zip(test_set, per_inst_preds):
        tours = beam_search_tours(d, probs, beam_width=args.beam_width, max_solutions=3, mix_prob=args.mix_prob)
        true_cost = 0.0
        for a,b in zip(d.true_tour, d.true_tour[1:]): true_cost += d.D[a,b]
        true_cost += d.D[d.true_tour[-1], d.true_tour[0]]
        found = False
        found_equal = False
        for cost, tour in tours:
            if cost == true_cost:
                found_equal = True
            # canonical comparison by rotation/reverse
            def canon(t):
                mn = min(t); idx = t.index(mn); rt = t[idx:]+t[:idx]; rev = rt[::-1]; return tuple(rt) if tuple(rt)<tuple(rev) else tuple(rev)
            if tours and canon(tours[0][1]) == canon(d.true_tour):
                found = True
        recon_results.append({'name': d.name, 'n': d.n_nodes, 'true_cost': float(true_cost), 'found_equal_cost': found_equal})
    print('Recon results sample:', recon_results[:5])

    print('\nAll done.')

if __name__ == '__main__':
    main()
