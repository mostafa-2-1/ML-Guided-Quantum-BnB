
import os
import sys
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from preprocess_data import (
        parse_opt_tour
    )
    print("✅ Successfully imported parse opt tour from preprocess.py")
except ImportError as e:
    print(f"❌ Failed to import from preprocess.py: {e}")


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# FIXED PARSING FUNCTION
# =============================================================================
def parse_opt_tour(tour_path):
    tour = []
    with open(tour_path, "r") as f:
        in_section = False
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_section = True
                continue
            if not in_section:
                continue
            if line == "-1" or line == "EOF":
                break
            # Split line into multiple node indices
            for token in line.split():
                tour.append(int(token) - 1)  # convert to 0-based
    return tour

def compute_tour_metrics(kept_edges, edge_mapping, opt_tour, n_nodes):
    """
    Compute metrics comparing pruned edges to the optimal tour.
    opt_tour: list of node indices (0-based)
    """

    # Convert kept edges to undirected set (u,v) with u < v
    kept_set = set()
    for e in kept_edges:
        u, v = edge_mapping[e]
        kept_set.add((min(u, v), max(u, v)))

    # Convert tour to edge set
    opt_edge_set = set()
    for i in range(len(opt_tour)):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % len(opt_tour)]
        opt_edge_set.add((min(u, v), max(u, v)))

    kept_opt = kept_set & opt_edge_set

    return {
        "tour_edge_recall": len(kept_opt) / len(opt_edge_set),
        "tour_edges_kept": len(kept_opt),
        "tour_edges_total": len(opt_edge_set),
        "tour_broken": len(kept_opt) < len(opt_edge_set)
    }

def compute_degree_stats(kept_edges, edge_mapping, n_nodes):
    degree = np.zeros(n_nodes, dtype=int)
    for e in kept_edges:
        u, v = edge_mapping[e]
        degree[u] += 1
        degree[v] += 1

    return {
        "avg_degree": float(degree.mean()),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max())
    }

def compute_connectivity(kept_edges, edge_mapping, n_nodes):
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in kept_edges:
        u, v = edge_mapping[e]
        union(u, v)

    roots = set(find(i) for i in range(n_nodes))
    return {
        "is_connected": len(roots) == 1,
        "num_connected_components": len(roots)
    }

def compute_knn_overlap(coords, kept_edges, edge_mapping, knn_k=10):
    from scipy.spatial import KDTree

    n = len(coords)
    tree = KDTree(coords)

    knn_sets = []
    for i in range(n):
        _, idx = tree.query(coords[i], k=min(knn_k + 1, n))
        knn_sets.append(set(j for j in idx if j != i))

    overlap = []
    for i in range(n):
        neighbors = set()
        for e in kept_edges:
            u, v = edge_mapping[e]
            if u == i:
                neighbors.add(v)
            elif v == i:
                neighbors.add(u)

        if knn_sets[i]:
            overlap.append(len(neighbors & knn_sets[i]) / len(knn_sets[i]))

    return {"avg_knn_overlap": float(np.mean(overlap)) if overlap else 0.0}

def compute_edge_length_stats(kept_edges, edge_mapping, D):
    full_lengths = D[np.triu_indices(D.shape[0], 1)]
    pruned_lengths = []

    for e in kept_edges:
        u, v = edge_mapping[e]
        pruned_lengths.append(D[u, v])

    pruned_lengths = np.array(pruned_lengths)

    return {
        "mean_edge_length_full": float(full_lengths.mean()),
        "mean_edge_length_pruned": float(pruned_lengths.mean()) if len(pruned_lengths) else 0.0,
        "length_reduction_ratio": float(
            pruned_lengths.mean() / (full_lengths.mean() + 1e-12)
        ) if len(pruned_lengths) else 0.0
    }



def parse_tsp_fixed(filepath):
    """
    Parse a .tsp file (TSPLIB). Returns dict:
      - 'name': instance name
      - 'n': number of nodes
      - 'coords': Nx2 array if NODE_COORD_SECTION exists
      - 'D': NxN distance matrix if EDGE_WEIGHT_TYPE EXPLICIT or computed from coords
      - 'edge_weight_type': the type of edge weights
    """
    name = os.path.basename(filepath)
    n = None
    coords = {}
    D = None
    edge_type = None
    edge_format = None
    reading_coords = False
    reading_edges = False
    edge_lines = []

    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        if not line:
            continue
        up = line.upper()
        
        if up.startswith('NAME'):
            parts = line.split(':', 1)
            if len(parts) > 1:
                name = parts[1].strip()
        elif up.startswith('DIMENSION'):
            n = int(line.split(':')[1].strip())
        elif up.startswith('EDGE_WEIGHT_TYPE'):
            edge_type = line.split(':')[1].strip().upper()
        elif up.startswith('EDGE_WEIGHT_FORMAT'):
            edge_format = line.split(':')[1].strip().upper()
        elif up.startswith('NODE_COORD_SECTION'):
            reading_coords = True
            reading_edges = False
        elif up.startswith('EDGE_WEIGHT_SECTION'):
            reading_edges = True
            reading_coords = False
        elif up.startswith('DISPLAY_DATA_SECTION') or up.startswith('EOF'):
            reading_edges = False
            reading_coords = False
        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    idx = int(parts[0]) - 1
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
                except:
                    pass
        elif reading_edges:
            edge_lines.extend(line.split())

    # Build coords array if present
    coords_arr = None
    if coords and len(coords) == n:
        coords_arr = np.zeros((n, 2), dtype=float)
        for idx in range(n):
            if idx in coords:
                coords_arr[idx] = coords[idx]

    # Build distance matrix if EXPLICIT
    D_arr = None
    if edge_type == 'EXPLICIT' and edge_lines:
        try:
            edge_vals = list(map(float, edge_lines))
            D_arr = np.zeros((n, n), dtype=float)
            
            if edge_format == 'UPPER_ROW':
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if idx < len(edge_vals):
                            D_arr[i, j] = edge_vals[idx]
                            D_arr[j, i] = edge_vals[idx]
                            idx += 1
            elif edge_format == 'FULL_MATRIX':
                idx = 0
                for i in range(n):
                    for j in range(n):
                        if idx < len(edge_vals):
                            D_arr[i, j] = edge_vals[idx]
                            idx += 1
            elif edge_format == 'LOWER_DIAG_ROW':
                idx = 0
                for i in range(n):
                    for j in range(i+1):
                        if idx < len(edge_vals):
                            D_arr[i, j] = edge_vals[idx]
                            D_arr[j, i] = edge_vals[idx]
                            idx += 1
            elif edge_format == 'UPPER_DIAG_ROW':
                idx = 0
                for i in range(n):
                    for j in range(i, n):
                        if idx < len(edge_vals):
                            D_arr[i, j] = edge_vals[idx]
                            D_arr[j, i] = edge_vals[idx]
                            idx += 1
        except Exception as e:
            print(f"Warning: Failed to parse EXPLICIT matrix: {e}")
            D_arr = None

    # If we have coordinates, compute Euclidean distances
    if coords_arr is not None and D_arr is None:
        diff = coords_arr[:, None, :] - coords_arr[None, :, :]
        D_arr = np.sqrt((diff ** 2).sum(axis=-1))
    
    # For GEO type, we need special handling
    if edge_type == 'GEO' and coords_arr is not None:
        D_arr = compute_geo_distances(coords_arr)
    elif edge_type == 'ATT' and coords_arr is not None:
        D_arr = compute_att_distances(coords_arr)

    result = {
        'name': name,
        'n': n,
        'edge_weight_type': edge_type
    }
    
    if coords_arr is not None:
        result['coords'] = coords_arr
    if D_arr is not None:
        result['D'] = D_arr
        
    return result


def compute_geo_distances(coords):
    """Compute GEO (geographic) distances as per TSPLIB spec."""
    n = len(coords)
    PI = 3.141592653589793
    
    # Convert to radians
    lat = np.zeros(n)
    lon = np.zeros(n)
    
    for i in range(n):
        deg_x = int(coords[i, 0])
        min_x = coords[i, 0] - deg_x
        lat[i] = PI * (deg_x + 5.0 * min_x / 3.0) / 180.0
        
        deg_y = int(coords[i, 1])
        min_y = coords[i, 1] - deg_y
        lon[i] = PI * (deg_y + 5.0 * min_y / 3.0) / 180.0
    
    RRR = 6378.388  # Earth radius in km
    
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            q1 = np.cos(lon[i] - lon[j])
            q2 = np.cos(lat[i] - lat[j])
            q3 = np.cos(lat[i] + lat[j])
            dij = int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
            D[i, j] = dij
            D[j, i] = dij
    
    return D


def compute_att_distances(coords):
    """Compute ATT (pseudo-Euclidean) distances as per TSPLIB spec."""
    n = len(coords)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            xd = coords[i, 0] - coords[j, 0]
            yd = coords[i, 1] - coords[j, 1]
            rij = np.sqrt((xd * xd + yd * yd) / 10.0)
            tij = int(round(rij))
            if tij < rij:
                dij = tij + 1
            else:
                dij = tij
            D[i, j] = dij
            D[j, i] = dij
    
    return D


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance matrix."""
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def enforce_connectivity(
    kept_edges,
    edge_mapping,
    n_nodes,
    D,
    candidate_edges=None
):
    """
    Ensure graph connectivity by adding minimal bridge edges.
    candidate_edges: optional set of allowed (u,v) edges
    """
    # --- Union-Find ---
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Union kept edges
    for e in kept_edges:
        u, v = edge_mapping[e]
        union(u, v)

    # Build components
    components = {}
    for i in range(n_nodes):
        r = find(i)
        components.setdefault(r, []).append(i)

    if len(components) == 1:
        return kept_edges  # already connected

    comp_roots = list(components.keys())

    # --- Find minimal bridges ---
    added_edges = []

    for i in range(len(comp_roots) - 1):
        c1 = components[comp_roots[i]]
        c2 = components[comp_roots[i + 1]]

        best = None
        best_dist = float("inf")

        for u in c1:
            for v in c2:
                if candidate_edges is not None:
                    if (min(u, v), max(u, v)) not in candidate_edges:
                        continue
                d = D[u, v]
                if d < best_dist:
                    best_dist = d
                    best = (u, v)

        if best is not None:
            added_edges.append(best)
            union(best[0], best[1])

    # --- Convert added edges to indices ---
    for (u, v) in added_edges:
        for idx, (a, b) in edge_mapping.items():
            if (a == u and b == v) or (a == v and b == u):
                kept_edges.append(idx)
                break

    return sorted(set(kept_edges))

# =============================================================================
# IMPORTS FROM PREPROCESSING
# =============================================================================
try:
    from preprocess_data import (
        build_candidate_edges,
        compute_node_edge_features
    )
    print("✅ Successfully imported preprocessing functions")
except ImportError as e:
    print(f"⚠️ Could not import from preprocess_data.py: {e}")
    print("  Using built-in functions instead")
    
    def build_candidate_edges(coords, full_threshold=300, knn_k=30):
        """Build candidate edges using KNN or full graph."""
        n = len(coords)
        D = pairwise_distances(coords)
        
        if n <= full_threshold:
            # Full graph
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        else:
            # KNN graph
            from scipy.spatial import KDTree
            tree = KDTree(coords)
            edges_set = set()
            for i in range(n):
                _, indices = tree.query(coords[i], k=min(knn_k+1, n))
                for j in indices:
                    if i != j:
                        edges_set.add((min(i, j), max(i, j)))
            edges = list(edges_set)
        
        return edges, D

# PyG imports
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    print("✅ Successfully imported PyTorch Geometric")
except ImportError as e:
    raise ImportError("PyTorch Geometric not found. Please install it.")

def prune_stage2_aggressive(data, scores, k2, aggressiveness):
    """
    Stage-2 pruning with aggressiveness control.
    aggressiveness ∈ [0,1]
    """
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

        deg = idx.numel()
        k_eff = int(np.ceil((1 - aggressiveness) * deg + aggressiveness * k2))
        k_eff = max(1, min(k_eff, deg))

        topk_indices = torch.topk(node_scores, k_eff).indices
        for i in topk_indices:
            kept_edges.add(idx[i].item())

    kept_edges_list = sorted(kept_edges)
    kept_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    kept_mask[kept_edges_list] = True

    pruned_data = Data(
        x=data.x,
        edge_index=edge_index[:, kept_mask],
        edge_attr=data.edge_attr[kept_mask],
        y=data.y[kept_mask] if data.y is not None else None
    )

    return pruned_data, kept_edges_list, kept_mask

# =============================================================================
# MODEL DEFINITION (Same as training)
# =============================================================================

class EdgeGNN(nn.Module):
    """Graph Neural Network for edge classification in TSP."""
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
            nn.Linear(hidden_dim // 2, 1)
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


# =============================================================================
# EXTENDED FEATURE COMPUTATION
# =============================================================================

def compute_extended_features(coords, edges, D, knn_k=10):
    """Compute extended node and edge features."""
    n = len(coords)
    coords = np.asarray(coords, dtype=float)
    
    # Basic statistics
    upper_tri = D[np.triu_indices(n, 1)]
    mean_d = upper_tri.mean() + 1e-12
    max_d = D.max() + 1e-12
    bbox_w = np.ptp(coords[:, 0]) + 1e-6
    bbox_h = np.ptp(coords[:, 1]) + 1e-6
    
    # KNN computation
    from scipy.spatial import KDTree
    tree = KDTree(coords)
    
    knn_k_actual = min(knn_k, n - 1)
    knn_dists = np.zeros((n, max(1, knn_k_actual)))
    knn_indices = np.zeros((n, max(1, knn_k_actual)), dtype=int)
    
    for i in range(n):
        k = min(knn_k + 1, n)
        dists, idxs = tree.query(coords[i], k=k)
        mask = idxs != i
        dists = dists[mask][:knn_k_actual]
        idxs = idxs[mask][:knn_k_actual]
        
        if len(dists) < knn_k_actual:
            dists = np.pad(dists, (0, knn_k_actual - len(dists)), constant_values=mean_d)
            idxs = np.pad(idxs, (0, knn_k_actual - len(idxs)), constant_values=0)
        
        knn_dists[i] = dists
        knn_indices[i] = idxs
    
    avg_knn = knn_dists.mean(axis=1)
    std_knn = knn_dists.std(axis=1)
    
    # Compute degree from edge list
    degree = np.zeros(n)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    max_degree = degree.max() + 1e-6
    
    # Local density
    local_density = 1.0 / (avg_knn + 1e-6)
    local_density = local_density / (local_density.max() + 1e-6)
    
    # Node features (6)
    node_feat = np.zeros((n, 6), dtype=np.float32)
    node_feat[:, 0] = (coords[:, 0] - coords[:, 0].min()) / bbox_w
    node_feat[:, 1] = (coords[:, 1] - coords[:, 1].min()) / bbox_h
    node_feat[:, 2] = avg_knn / mean_d
    node_feat[:, 3] = std_knn / mean_d
    node_feat[:, 4] = local_density
    node_feat[:, 5] = degree / max_degree
    
    # Rank matrix
    sorted_idx = np.argsort(D, axis=1)
    rank = np.zeros_like(sorted_idx)
    for i in range(n):
        rank[i, sorted_idx[i]] = np.arange(n)
    
    # KNN sets
    knn_sets = [set(knn_indices[i]) for i in range(n)]
    
    # Edge features (9)
    edge_index = [[], []]
    edge_attr = []
    
    for i, j in edges:
        edge_index[0].append(i)
        edge_index[1].append(j)
        
        d = D[i, j]
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        
        angle = np.arctan2(dy, dx)
        angle_norm = (angle + np.pi) / (2 * np.pi)
        
        is_knn_i = 1.0 if j in knn_sets[i] else 0.0
        is_knn_j = 1.0 if i in knn_sets[j] else 0.0
        
        log_d = np.log1p(d) / (np.log1p(max_d) + 1e-6)
        
        edge_attr.append([
            d / mean_d,
            abs(dx) / bbox_w,
            abs(dy) / bbox_h,
            rank[i, j] / n,
            rank[j, i] / n,
            is_knn_i,
            is_knn_j,
            angle_norm,
            log_d
        ])
    
    return (
        node_feat,
        np.array(edge_index, dtype=np.int64),
        np.array(edge_attr, dtype=np.float32)
    )


# =============================================================================
# GRAPH BUILDING FOR PRUNING
# =============================================================================

def build_pyg_data_for_pruning(tsp_path, knn_k=30, knn_feat_k=10, full_threshold=300,
                                expected_node_feats=6, expected_edge_feats=9):
    """Build a PyG Data object from a TSP file for pruning."""
    
    # Use fixed parser
    result = parse_tsp_fixed(tsp_path)
    
    if result is None:
        return None, None, None, None
    
    n = result.get('n', 0)
    coords = result.get('coords', None)
    D = result.get('D', None)
    
    if n == 0:
        return None, None, None, None
    
    # If no coords but we have D, create dummy coords for features
    if coords is None:
        if D is not None:
            # Use MDS to create coords from D
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=100)
            coords = mds.fit_transform(D)
            print(f"  Created coords from distance matrix using MDS")
        else:
            print(f"  Error: No coordinates or distance matrix")
            return None, None, None, None
    
    coords = np.asarray(coords, dtype=np.float64)
    
    # Compute D from coords if not available
    if D is None:
        D = pairwise_distances(coords)
    
    # Build candidate edges
    if n <= full_threshold:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        edges_set = set()
        for i in range(n):
            _, indices = tree.query(coords[i], k=min(knn_k+1, n))
            for j in indices:
                if i != j:
                    edges_set.add((min(i, j), max(i, j)))
        edges = list(edges_set)
    
    if len(edges) == 0:
        return None, None, None, None
    
    # Compute features
    try:
        node_feat, edge_index_np, edge_attr = compute_extended_features(
            coords, edges, D, knn_k=knn_feat_k
        )
    except Exception as e:
        print(f"  Feature computation failed: {e}")
        return None, None, None, None
    
    # Pad/truncate features
    actual_node_feats = node_feat.shape[1]
    actual_edge_feats = edge_attr.shape[1]
    
    if actual_node_feats < expected_node_feats:
        padding = np.zeros((node_feat.shape[0], expected_node_feats - actual_node_feats), dtype=np.float32)
        node_feat = np.hstack([node_feat, padding])
    elif actual_node_feats > expected_node_feats:
        node_feat = node_feat[:, :expected_node_feats]
    
    if actual_edge_feats < expected_edge_feats:
        padding = np.zeros((edge_attr.shape[0], expected_edge_feats - actual_edge_feats), dtype=np.float32)
        edge_attr = np.hstack([edge_attr, padding])
    elif actual_edge_feats > expected_edge_feats:
        edge_attr = edge_attr[:, :expected_edge_feats]
    
    # Create Data object
    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.zeros(len(edges), dtype=torch.float)
    )
    
    edge_mapping = {i: (edges[i][0], edges[i][1]) for i in range(len(edges))}
    edge_type = result.get('edge_weight_type')
    return data, coords, edge_mapping, D, edge_type


# =============================================================================
# PRUNING FUNCTIONS
# =============================================================================

def prune_to_topk(data, scores, k, device='cpu'):
    """Prune graph keeping top-k edges per node."""
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
    
    kept_edges_list = sorted(kept_edges)
    kept_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    kept_mask[kept_edges_list] = True
    
    pruned_data = Data(
        x=data.x,
        edge_index=edge_index[:, kept_mask],
        edge_attr=data.edge_attr[kept_mask],
        y=data.y[kept_mask] if data.y is not None else None
    )
    
    return pruned_data, kept_edges_list, kept_mask


def apply_structural_pruning(data, scores, k, degree_cap=6, require_top2=True):
    """Apply structural pruning rules."""
    edge_index = data.edge_index
    src, dst = edge_index
    n_nodes = data.x.size(0)
    n_edges = edge_index.size(1)
    
    node_topk = {node: set() for node in range(n_nodes)}
    node_top2 = {node: set() for node in range(n_nodes)}
    
    for node in range(n_nodes):
        mask = (src == node) | (dst == node)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        
        if idx.numel() == 0:
            continue
        
        node_scores = scores[idx]
        
        topk_count = min(k, idx.numel())
        topk_indices = torch.topk(node_scores, topk_count).indices
        for ti in topk_indices:
            node_topk[node].add(idx[ti].item())
        
        top2_count = min(2, idx.numel())
        top2_indices = torch.topk(node_scores, top2_count).indices
        for ti in top2_indices:
            node_top2[node].add(idx[ti].item())
    
    kept_edges = set()
    for node in range(n_nodes):
        kept_edges.update(node_topk[node])
    
    if require_top2:
        filtered_edges = set()
        for e in kept_edges:
            u, v = src[e].item(), dst[e].item()
            if e in node_top2[u] or e in node_top2[v]:
                filtered_edges.add(e)
        kept_edges = filtered_edges
    
    if degree_cap is not None:
        degrees = {node: 0 for node in range(n_nodes)}
        edge_candidates = [(e, scores[e].item(), src[e].item(), dst[e].item()) 
                          for e in kept_edges]
        edge_candidates.sort(key=lambda x: -x[1])
        
        final_edges = set()
        for e, score, u, v in edge_candidates:
            u_can = degrees[u] < degree_cap
            v_can = degrees[v] < degree_cap
            
            if u_can and v_can:
                final_edges.add(e)
                degrees[u] += 1
                degrees[v] += 1
            elif u_can:
                final_edges.add(e)
                degrees[u] += 1
            elif v_can:
                final_edges.add(e)
                degrees[v] += 1
        
        kept_edges = final_edges
    
    kept_edges_list = sorted(kept_edges)
    kept_mask = torch.zeros(n_edges, dtype=torch.bool)
    kept_mask[kept_edges_list] = True
    
    pruned_data = Data(
        x=data.x,
        edge_index=edge_index[:, kept_mask],
        edge_attr=data.edge_attr[kept_mask],
        y=data.y[kept_mask] if data.y is not None else None
    )
    
    return pruned_data, kept_edges_list, kept_mask


# =============================================================================
# FIXED OUTPUT FORMATS
# =============================================================================

def save_edge_list(kept_edges, edge_mapping, output_path):
    """Save pruned edges as a simple edge list."""
    with open(output_path, 'w') as f:
        f.write(f"# Pruned edge list\n")
        f.write(f"# Total edges: {len(kept_edges)}\n")
        for edge_idx in kept_edges:
            u, v = edge_mapping[edge_idx]
            f.write(f"{u} {v}\n")


def save_sparse_tsp(coords, kept_edges, edge_mapping, output_path, name="pruned"):
    """
    Save a pruned TSP as an EXPLICIT FULL_MATRIX for LKH.
    Pruned edges get their true distance, missing edges get a large value.
    """
    n = len(coords)
    # Build set of kept edges (undirected)
    kept_set = set()
    for edge_idx in kept_edges:
        u, v = edge_mapping[edge_idx]
        kept_set.add((u, v))
        kept_set.add((v, u))

    # Compute distance matrix
    coords = np.asarray(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))

    with open(output_path, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: Pruned graph with {len(kept_edges)} edges\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write(f"EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append("0")
                elif (i, j) in kept_set:
                    row.append(str(int(round(D[i, j]))))
                else:
                    row.append("9999999")
            f.write(" ".join(row) + "\n")
        f.write("EOF\n")


def save_adjacency_csv(n_nodes, kept_edges, edge_mapping, D, output_path):
    """Save as sparse adjacency matrix."""
    rows, cols, weights = [], [], []
    
    written = set()
    for edge_idx in kept_edges:
        u, v = edge_mapping[edge_idx]
        edge_key = (min(u, v), max(u, v))
        if edge_key not in written:
            rows.extend([u, v])
            cols.extend([v, u])
            weights.extend([D[u, v], D[v, u]])
            written.add(edge_key)
    
    df = pd.DataFrame({'row': rows, 'col': cols, 'weight': weights})
    df.to_csv(output_path, index=False)


def save_lkh_candidates(n_nodes, kept_edges, edge_mapping, D, output_path):
    from collections import defaultdict
    
    adj = defaultdict(list)
    for edge_idx in kept_edges:
        u, v = edge_mapping[edge_idx]
        d = int(round(D[u, v]))
        adj[u].append((v, d))
        adj[v].append((u, d))

    with open(output_path, "w") as f:
        f.write(f"{n_nodes}\n")
        for node in range(n_nodes):
            neighbors = sorted(adj[node], key=lambda x: x[1])
            if len(neighbors) >= n_nodes:
                print(f"Node {node+1} has {len(neighbors)} neighbors (should be < {n_nodes})")
            # Write: nodeID neighbor1ID distance1 neighbor2ID distance2 ...
            f.write(f"{node+1}")
            for neighbor, dist in neighbors:
                if neighbor == node:
                    print(f"Node {node+1} has a self-loop")
                if neighbor < 0 or neighbor >= n_nodes:
                    print(f"Node {node+1} has out-of-bounds neighbor {neighbor+1}")
                f.write(f" {neighbor+1} {dist}")
            f.write("\n")
        f.write("-1\n")



def save_concorde_format(coords, kept_edges, edge_mapping, D, output_path):
    """
    Save in Concorde format with penalties for non-candidate edges.
    FIXED: Properly scales distances to integers.
    """
    n = len(coords)
    D_max = float(D.max())
    PENALTY_FACTOR = 3.0   # safe default
    penalty = PENALTY_FACTOR * D_max
    SCALE = 100            # fixed global scale

    
    # Build candidate edge set
    candidate_edges = set()
    for edge_idx in kept_edges:
        u, v = edge_mapping[edge_idx]
        candidate_edges.add((min(u, v), max(u, v)))
    
    
    with open(output_path, 'w') as f:
        f.write(f"NAME: pruned_concorde\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: Pruned graph, {len(candidate_edges)} edges, scale={SCALE}\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
        f.write(f"EDGE_WEIGHT_SECTION\n")
        row = []
        INF = 10**9
        for i in range(n):
            row = []
            for j in range(i + 1, n):
                base = float(D[i, j])

                if (min(i, j), max(i, j)) in candidate_edges:
                    # Scale and ensure minimum distance of 1
                    w = base
                else:
                    w = base + penalty
                row.append(int(round(w * SCALE)))
            if row:
                f.write(" ".join(map(str, row)) + "\n")
        
        f.write("EOF\n")
    
    return SCALE# Return scale for reference

def compute_stage2_aggressiveness(
    base=1.0,
    stage1_recall=None,
    n_nodes=0,
    coord_type="EUC_2D"
):
    aggr = base

    # ---------------------------
    # f(recall_stage1)
    # ---------------------------
    if stage1_recall is not None:
        if stage1_recall < 0.98:
            aggr *= 0.6   # be conservative
        elif stage1_recall > 0.995:
            aggr *= 1.1   # prune harder

    # ---------------------------
    # g(n_nodes)
    # ---------------------------
    if n_nodes > 1000:
        aggr *= 0.7
    elif n_nodes > 500:
        aggr *= 0.85

    # ---------------------------
    # h(coord_type)
    # ---------------------------
    if coord_type in ("GEO", "ATT"):
        aggr *= 0.8

    # Clamp for safety
    return float(np.clip(aggr, 0.5, 1.3))

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_models(args, device):
    """Load trained models."""
    print("\n" + "=" * 60)
    print("LOADING TRAINED MODELS")
    print("=" * 60)
    
    model_config = {
        'in_node_feats': args.node_feats,
        'in_edge_feats': args.edge_feats,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout
    }
    
    print(f"\nModel config: {model_config}")
    
    print(f"\nLoading Stage 1 model from: {args.stage1_model}")
    stage1_model = EdgeGNN(**model_config).to(device)
    
    try:
        state_dict = torch.load(args.stage1_model, map_location=device)
        stage1_model.load_state_dict(state_dict)
        stage1_model.eval()
        print("✅ Stage 1 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Stage 1 model: {e}")
        return None, None
    
    stage2_model = None
    if args.stage2_model and os.path.exists(args.stage2_model):
        print(f"\nLoading Stage 2 model from: {args.stage2_model}")
        stage2_model = EdgeGNN(**model_config).to(device)
        
        try:
            state_dict = torch.load(args.stage2_model, map_location=device)
            stage2_model.load_state_dict(state_dict)
            stage2_model.eval()
            print("✅ Stage 2 model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load Stage 2 model: {e}")
            stage2_model = None
    
    return stage1_model, stage2_model


def prune_single_graph(tsp_path, stage1_model, stage2_model, device, args, k):
    """Prune a single TSP graph."""
    instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
    
    # Build PyG data - now returns D as well
    data, coords, edge_mapping, D, coord_type = build_pyg_data_for_pruning(
        tsp_path,
        knn_k=args.knn_k,
        knn_feat_k=args.knn_feat_k,
        full_threshold=args.full_threshold,
        expected_node_feats=args.node_feats,
        expected_edge_feats=args.edge_feats
    )
    
    if data is None:
        return None
    
    data = data.to(device)
    n_nodes = data.x.size(0)
    if n_nodes < args.min_nodes:
        return None
    original_edges = data.edge_index.size(1)
    
    # results = {
    #     'instance_name': instance_name,
    #     'n_nodes': n_nodes,
    #     'original_edges': original_edges,
    #     'coords': coords,
    #     'D': D,  # Use the D from parsing (proper handling of GEO, ATT, etc.)
    #     'edge_mapping': edge_mapping,
    #     'stages': {}
    # }
    
    # Stage 1
    with torch.no_grad():
        scores1 = stage1_model(data.x, data.edge_index, data.edge_attr)
    
    pruned_s1, kept_edges_s1, mask_s1 = prune_to_topk(data, scores1, k, device)
    
    candidate_edges = {
        (min(u, v), max(u, v))
        for (_, (u, v)) in edge_mapping.items()
    }
    
    kept_edges_s1 = enforce_connectivity(
        kept_edges=kept_edges_s1,
        edge_mapping=edge_mapping,
        n_nodes=n_nodes,
        D=D,
        candidate_edges=candidate_edges
    )
    # results['stages']['stage1'] = {
    #     'k': args.k1,
    #     'kept_edges': kept_edges_s1,
    #     'n_edges': len(kept_edges_s1),
    #     'sparsity': len(kept_edges_s1) / original_edges * 100,
    # }
    results = {
        'instance_name': instance_name,
        'n_nodes': n_nodes,
        'original_edges': original_edges,
        'coords': coords,
        'D': D,
        'edge_mapping': edge_mapping,
        'kept_edges': kept_edges_s1,
        'k': k,
        'sparsity': len(kept_edges_s1) / original_edges * 100 if original_edges > 0 else 0
    }
    return results
    # stage1_recall = None
    # tour_path = os.path.join(args.opt_tour_dir, f"{instance_name}.opt.tour")
    # if os.path.exists(tour_path):
    #     opt_tour = parse_opt_tour(tour_path)
    #     metrics = compute_tour_metrics(
    #         kept_edges_s1, edge_mapping, opt_tour, n_nodes
    #     )
    #     stage1_recall = metrics["tour_edge_recall"]
    # # ------------------------------------------------------------
    # # Compute instance-adaptive Stage-2 aggressiveness
    # # ------------------------------------------------------------
   
    # stage2_aggressiveness = compute_stage2_aggressiveness(
    #     base=1.0,
    #     stage1_recall=stage1_recall,
    #     n_nodes=n_nodes,
    #     coord_type=coord_type
    # )

    # # Stage 2
    # if stage2_model is not None and args.use_stage2:
    #     pruned_s1 = pruned_s1.to(device)
        
    #     with torch.no_grad():
    #         scores2 = stage2_model(pruned_s1.x, pruned_s1.edge_index, pruned_s1.edge_attr)
        
    #     pruned_s2, kept_edges_s2_local, mask_s2 = prune_stage2_aggressive(
    #         pruned_s1,
    #         scores2,
    #         args.k2,
    #         stage2_aggressiveness
    #     )

        
    #     s1_kept = kept_edges_s1
    #     kept_edges_s2 = [s1_kept[i] for i in kept_edges_s2_local]
    #     # ------------------------------------------------------------
    #     # Enforce connectivity after Stage 2
    #     # ------------------------------------------------------------
    #     candidate_edges = {
    #         (min(u, v), max(u, v))
    #         for (_, (u, v)) in edge_mapping.items()
    #     }

    #     kept_edges_s2 = enforce_connectivity(
    #         kept_edges=kept_edges_s2,
    #         edge_mapping=edge_mapping,
    #         n_nodes=n_nodes,
    #         D=D,
    #         candidate_edges=candidate_edges
    #     )

    #     results['stages']['stage2'] = {
    #         'k': args.k2,
    #         'aggressiveness': stage2_aggressiveness,
    #         'kept_edges': kept_edges_s2,
    #         'n_edges': len(kept_edges_s2),
    #         'sparsity': len(kept_edges_s2) / original_edges * 100,
    #     }
        
    #     # Stage 3
    #     if args.use_stage3:
    #         pruned_s3, kept_edges_s3_local, _ = apply_structural_pruning(
    #             pruned_s2, scores2[mask_s2], args.k3,
    #             degree_cap=args.degree_cap, require_top2=True
    #         )
            
    #         kept_edges_s3 = [kept_edges_s2[i] for i in kept_edges_s3_local]
            
    #         results['stages']['stage3'] = {
    #             'k': args.k3,
    #             'degree_cap': args.degree_cap,
    #             'kept_edges': kept_edges_s3,
    #             'n_edges': len(kept_edges_s3),
    #             'sparsity': len(kept_edges_s3) / original_edges * 100,
    #         }
    
    # return results

def save_k_sweep_graph(results, k_output_dir, args):
    """Save pruned graph for k-sweep with metadata."""
    instance_name = results['instance_name']
    kept_edges = results['kept_edges']
    coords = results['coords']
    edge_mapping = results['edge_mapping']
    D = results['D']
    n_nodes = results['n_nodes']
    k = results['k']
    sparsity = results['sparsity']
    
    instance_dir = os.path.join(k_output_dir, instance_name)
    os.makedirs(instance_dir, exist_ok=True)
    
    # Save all formats
    save_edge_list(kept_edges, edge_mapping,
                   os.path.join(instance_dir, f"{instance_name}_edges.txt"))
    
    save_sparse_tsp(coords, kept_edges, edge_mapping,
                    os.path.join(instance_dir, f"{instance_name}_sparse.tsp"), instance_name)
    
    save_adjacency_csv(n_nodes, kept_edges, edge_mapping, D,
                       os.path.join(instance_dir, f"{instance_name}_adjacency.csv"))
    
    save_lkh_candidates(n_nodes, kept_edges, edge_mapping, D,
                        os.path.join(instance_dir, f"{instance_name}_candidates.txt"))
    
    if args.save_concorde:
        scale = save_concorde_format(coords, kept_edges, edge_mapping, D,
                             os.path.join(instance_dir, f"{instance_name}_concorde.tsp"))
        results['concorde_scale'] = scale
    
    # Build comprehensive metadata
    metadata = {
        'instance_name': instance_name,
        'n_nodes': n_nodes,
        'original_edges': results['original_edges'],
        'kept_edges': len(kept_edges),
        'k': k,
        'sparsity': sparsity,
        'timestamp': datetime.now().isoformat()
    }
    
    # Structural metrics
    metadata.update(compute_degree_stats(kept_edges, edge_mapping, n_nodes))
    metadata.update(compute_connectivity(kept_edges, edge_mapping, n_nodes))
    metadata.update(compute_knn_overlap(coords, kept_edges, edge_mapping))
    metadata.update(compute_edge_length_stats(kept_edges, edge_mapping, D))
    
    # Optional optimal tour metrics
    tour_path = os.path.join(args.opt_tour_dir, f"{instance_name}.opt.tour")
    if os.path.exists(tour_path):
        opt_tour = parse_opt_tour(tour_path)
        if opt_tour:
            metadata.update(
                compute_tour_metrics(kept_edges, edge_mapping, opt_tour, n_nodes)
            )
    metadata['feasible'] = metadata['is_connected']
    if 'tour_broken' in metadata:
        metadata['feasible'] = metadata['feasible'] and (not metadata['tour_broken'])

    if 'concorde_scale' in results:
        metadata['concorde_scale'] = results['concorde_scale']
    
    # Save metadata
    metadata_path = os.path.join(instance_dir, f"{instance_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def run_k_sweep_pipeline(args):
    """Run k-sweep pipeline for stage1 pruning."""
    print("\n" + "=" * 70)
    print("K-SWEEP STAGE1 PRUNING PIPELINE")
    print("=" * 70)
    print(f"\nInput:  {args.input_dir}")
    print(f"Output base: {args.output_dir}")
    print(f"Minimum nodes: {args.min_nodes}")
    print(f"K values to sweep: {args.k_values}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load stage1 model only
    print(f"\nLoading Stage 1 model from: {args.stage1_model}")
    
    model_config = {
        'in_node_feats': args.node_feats,
        'in_edge_feats': args.edge_feats,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout
    }
    
    stage1_model = EdgeGNN(**model_config).to(device)
    
    try:
        state_dict = torch.load(args.stage1_model, map_location=device)
        stage1_model.load_state_dict(state_dict)
        stage1_model.eval()
        print("✅ Stage 1 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Stage 1 model: {e}")
        return
    
    # Collect TSP files
    tsp_files = [os.path.join(args.input_dir, f) 
                 for f in os.listdir(args.input_dir) if f.endswith('.tsp')]
    print(f"\nFound {len(tsp_files)} TSP files")
    
    # Create master summary DataFrame
    all_summaries = []
    
    # Run k-sweep
    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"Processing k={k}")
        print('='*60)
        
        # Create output directory for this k
        k_output_dir = os.path.join(args.output_dir, f"k{k}")
        os.makedirs(k_output_dir, exist_ok=True)
        
        k_summaries = []
        failed = []
        processed_count = 0
        skipped_count = 0
        
        for tsp_path in tqdm(tsp_files, desc=f"k={k}"):
            instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
            
            try:
                # Prune with this k
                results = prune_single_graph(
                    tsp_path, stage1_model, None
                    , device, args, k
                )
                
                if results is None:
                    # Check if it's because of min_nodes or build failure
                    # Try to parse just to check node count
                    parsed = parse_tsp_fixed(tsp_path)
                    if parsed and parsed.get('n', 0) < args.min_nodes:
                        skipped_count += 1
                    else:
                        failed.append((instance_name, "Failed to build graph"))
                    continue
                
                # Save graph and get metadata
                metadata = save_k_sweep_graph(results, k_output_dir, args)
                
                # Add to summaries
                summary_row = {
                    'instance': instance_name,
                    'k': k,
                    'n_nodes': results['n_nodes'],
                    'original_edges': results['original_edges'],
                    'kept_edges': len(results['kept_edges']),
                    'sparsity': results['sparsity']
                }
                
                # Add metrics from metadata
                for key in ['avg_degree', 'min_degree', 'max_degree', 
                           'is_connected', 'num_connected_components',
                           'avg_knn_overlap', 'mean_edge_length_full',
                           'mean_edge_length_pruned', 'length_reduction_ratio',
                           'tour_edge_recall', 'tour_edges_kept', 'tour_edges_total',
                           'tour_broken']:
                    if key in metadata:
                        summary_row[key] = metadata[key]
                
                k_summaries.append(summary_row)
                processed_count += 1
                
            except Exception as e:
                import traceback
                failed.append((instance_name, str(e)))
                traceback.print_exc()
        
        # Save k-specific summary
        if k_summaries:
            k_df = pd.DataFrame(k_summaries)
            k_summary_path = os.path.join(k_output_dir, f"summary_k{k}.csv")
            k_df.to_csv(k_summary_path, index=False)
            all_summaries.extend(k_summaries)
            
            print(f"\n  Processed: {processed_count}")
            print(f"  Skipped (<{args.min_nodes} nodes): {skipped_count}")
            print(f"  Failed: {len(failed)}")
            
            if failed:
                print("\n  Failed instances:")
                for name, reason in failed[:5]:
                    print(f"    - {name}: {reason}")
                if len(failed) > 5:
                    print(f"    ... and {len(failed) - 5} more")
        
        else:
            print(f"\n  No graphs processed for k={k}")
    
    # Save master summary
    if all_summaries:
        master_df = pd.DataFrame(all_summaries)
        master_summary_path = os.path.join(args.output_dir, "k_sweep_master_summary.csv")
        master_df.to_csv(master_summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("K-SWEEP COMPLETE")
        print('='*70)
        
        # Print statistics
        print(f"\nTotal processed instances across all k values: {len(master_df['instance'].unique())}")
        print(f"\nSummary statistics:")
        
        # Group by k and compute averages
        grouped = master_df.groupby('k').agg({
            'n_nodes': 'mean',
            'sparsity': 'mean',
            'kept_edges': 'mean',
            'avg_degree': 'mean',
            'tour_edge_recall': 'mean',
            'is_connected': lambda x: (x == True).mean() * 100
        }).round(2)
        
        grouped.columns = ['Avg Nodes', 'Avg Sparsity %', 'Avg Kept Edges', 
                          'Avg Degree', 'Avg Tour Recall %', '% Connected']
        
        print("\n" + str(grouped))
        
        print(f"\n✅ Results saved to: {args.output_dir}")
        print(f"   Master summary: {master_summary_path}")
    else:
        print("\n⚠️ No graphs were processed. Check if all instances have <{args.min_nodes} nodes.")



def save_pruned_graph(results, output_dir, stage_name, args):
    """Save pruned graph in multiple formats."""
    instance_name = results['instance_name']
    stage_results = results['stages'][stage_name]
    
    kept_edges = stage_results['kept_edges']
    coords = results['coords']
    edge_mapping = results['edge_mapping']
    D = results['D']
    n_nodes = results['n_nodes']
    
    instance_dir = os.path.join(output_dir, stage_name, instance_name)
    os.makedirs(instance_dir, exist_ok=True)
    
    # Save all formats
    save_edge_list(kept_edges, edge_mapping,
                   os.path.join(instance_dir, f"{instance_name}_edges.txt"))
    
    save_sparse_tsp(coords, kept_edges, edge_mapping,
                    os.path.join(instance_dir, f"{instance_name}_sparse.tsp"), instance_name)
    
    save_adjacency_csv(n_nodes, kept_edges, edge_mapping, D,
                       os.path.join(instance_dir, f"{instance_name}_adjacency.csv"))
    
    save_lkh_candidates(n_nodes, kept_edges, edge_mapping, D,
                        os.path.join(instance_dir, f"{instance_name}_candidates.txt"))
    
    if args.save_concorde:
        scale = save_concorde_format(coords, kept_edges, edge_mapping, D,
                             os.path.join(instance_dir, f"{instance_name}_concorde.tsp"))
        # stage_results['concorde_scale'] = scale


    

    

    # Metadata
    metadata = {
        'instance_name': instance_name,
        'n_nodes': n_nodes,
        'original_edges': results['original_edges'],
        'kept_edges': len(kept_edges),
        'sparsity': stage_results['sparsity'],
        'stage': stage_name,
        'timestamp': datetime.now().isoformat()
    }
    # Structural metrics
    metadata.update(compute_degree_stats(kept_edges, edge_mapping, n_nodes))
    metadata.update(compute_connectivity(kept_edges, edge_mapping, n_nodes))
    metadata.update(compute_knn_overlap(coords, kept_edges, edge_mapping))
    metadata.update(compute_edge_length_stats(kept_edges, edge_mapping, D))
    # Optional optimal tour
    tour_path = os.path.join(args.opt_tour_dir, f"{instance_name}.opt.tour")
    if os.path.exists(tour_path):
        opt_edges = parse_opt_tour(tour_path)
        metadata.update(
            compute_tour_metrics(kept_edges, edge_mapping, opt_edges, n_nodes)
        )
    if 'concorde_scale' in stage_results:
        metadata['concorde_scale'] = stage_results['concorde_scale']
    
    with open(os.path.join(instance_dir, f"{instance_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def run_pruning_pipeline(args):
    """Main pruning pipeline."""
    print("\n" + "=" * 70)
    print("TSP GRAPH PRUNING PIPELINE (FIXED)")
    print("=" * 70)
    print(f"\nInput:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"\nConfiguration:")
    print(f"  Node features: {args.node_feats}, Edge features: {args.edge_feats}")
    print(f"  Stage 1 k={args.k1}, Stage 2 k={args.k2} (enabled={args.use_stage2})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    stage1_model, stage2_model = load_models(args, device)
    if stage1_model is None:
        return
    
    # Collect TSP files
    tsp_files = [os.path.join(args.input_dir, f) 
                 for f in os.listdir(args.input_dir) if f.endswith('.tsp')]
    print(f"\nFound {len(tsp_files)} TSP files")
    
    # Process
    all_results = []
    failed = []
    
    for tsp_path in tqdm(tsp_files, desc="Pruning"):
        instance_name = os.path.splitext(os.path.basename(tsp_path))[0]
        
        try:
            results = prune_single_graph(tsp_path, stage1_model, stage2_model, device, args)
            
            if results is None:
                failed.append((instance_name, "Failed to build graph"))
                continue
            
            for stage_name in results['stages']:
                save_pruned_graph(results, args.output_dir, stage_name, args)
            
            summary = {'instance': instance_name, 'n_nodes': results['n_nodes']}
            for stage, data in results['stages'].items():
                summary[f'{stage}_edges'] = data['n_edges']
                summary[f'{stage}_sparsity'] = data['sparsity']
            all_results.append(summary)
            
        except Exception as e:
            import traceback
            failed.append((instance_name, str(e)))
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processed: {len(all_results)}/{len(tsp_files)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed instances:")
        for name, reason in failed[:10]:
            print(f"  - {name}: {reason}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.output_dir, "pruning_summary.csv"), index=False)
        
        print("\nStatistics:")
        for stage in ['stage1', 'stage2', 'stage3']:
            if f'{stage}_sparsity' in df.columns:
                print(f"  {stage}: {df[f'{stage}_edges'].mean():.1f} edges, "
                      f"{df[f'{stage}_sparsity'].mean():.1f}% sparsity")
    
    print(f"\n✅ Results saved to: {args.output_dir}")


# def main():
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--input_dir', default='graphs_chosen')
#     parser.add_argument('--output_dir', default='pruned_graphs')
#     parser.add_argument('--stage1_model', default='results/best_model.pt')
#     parser.add_argument('--stage2_model', default='results/stage2/stage2_best.pt')
    
#     # Model architecture
#     parser.add_argument('--node_feats', type=int, default=6)
#     parser.add_argument('--edge_feats', type=int, default=9)
#     parser.add_argument('--hidden_dim', type=int, default=256)
#     parser.add_argument('--n_layers', type=int, default=4)
#     parser.add_argument('--dropout', type=float, default=0.3)
    
#     # Preprocessing
#     parser.add_argument('--full_threshold', type=int, default=300)
#     parser.add_argument('--knn_k', type=int, default=30)
#     parser.add_argument('--knn_feat_k', type=int, default=10)
    
#     # Pruning
#     parser.add_argument('--k1', type=int, default=10)
#     parser.add_argument('--k2', type=int, default=5)
#     parser.add_argument('--k3', type=int, default=4)
#     parser.add_argument('--degree_cap', type=int, default=6)
#     parser.add_argument('--use_stage2', action='store_true', default=True)
#     parser.add_argument('--use_stage3', action='store_true', default=False)
#     parser.add_argument('--save_concorde', action='store_true', default=True)
#     parser.add_argument(
#         '--opt_tour_dir',
#         default='tsplib_data',
#         help='Directory containing *.opt.tour files'
#     )

#     args = parser.parse_args()
#     run_pruning_pipeline(args)


# if __name__ == '__main__':
#     main()

def main():
    parser = argparse.ArgumentParser()
    
    # Input/Output
    parser.add_argument('--input_dir', default='graphs_chosen')
    parser.add_argument('--output_dir', default='pruned_graphs_k')
    
    # Model architecture
    parser.add_argument('--stage1_model', default='results/best_model.pt')
    parser.add_argument('--node_feats', type=int, default=6)
    parser.add_argument('--edge_feats', type=int, default=9)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Preprocessing
    parser.add_argument('--full_threshold', type=int, default=300)
    parser.add_argument('--knn_k', type=int, default=30)
    parser.add_argument('--knn_feat_k', type=int, default=10)
    
    # K-sweep specific
    parser.add_argument('--k_sweep', action='store_true', default=True,
                       help='Run k-sweep instead of normal pipeline')
    parser.add_argument('--k_values', type=int, nargs='+', 
                       default=[2, 3, 4, 5, 7, 8, 10, 15, 20, 25],
                       help='k values to sweep')
    parser.add_argument('--min_nodes', type=int, default=350,
                       help='Minimum nodes to process (skip smaller instances)')
    
    # Output options
    parser.add_argument('--save_concorde', action='store_true', default=True)
    parser.add_argument('--opt_tour_dir', default='tsplib_data',
                       help='Directory containing *.opt.tour files')
    
    # For backward compatibility (normal pipeline)
    parser.add_argument('--stage2_model', default='resultss/stage2/stage2_best.pt')
    parser.add_argument('--use_stage2', action='store_true', default=True)
    parser.add_argument('--use_stage3', action='store_true', default=False)
    parser.add_argument('--k1', type=int, default=10)
    parser.add_argument('--k2', type=int, default=5)
    parser.add_argument('--k3', type=int, default=4)
    parser.add_argument('--degree_cap', type=int, default=6)
    
    args = parser.parse_args()
    
    if args.k_sweep:
        run_k_sweep_pipeline(args)
    else:
        run_pruning_pipeline(args)


if __name__ == '__main__':
    main()