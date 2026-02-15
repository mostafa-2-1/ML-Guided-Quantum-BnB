import os
import json
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from scipy.spatial import KDTree

K_VALUES = [2,3,4,5,6,7,8,9,10,12,15,17,20,22,25]

INPUT_DIR = "graphss_chosen"
OUTPUT_ROOT = "pruned_graphs_k_baseline"

MIN_N = 350
MAX_N = 1000

# =============================================================================
# OPTIMAL TOUR PARSING (for computing recall)
# =============================================================================

def parse_opt_tour(tour_path):
    """
    Parse a .opt.tour file from TSPLIB.
    Returns list of node indices (0-based) in tour order.
    """
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

# =============================================================================
# DISTANCE COMPUTATION FUNCTIONS
# =============================================================================

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

# =============================================================================
# TSP PARSER (identical to GNN version)
# =============================================================================

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

# =============================================================================
# METRICS COMPUTATION FUNCTIONS (identical to GNN version)
# =============================================================================

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
    """Compute degree statistics from kept edges."""
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
    """Check if graph is connected and count components."""
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
    """Compute overlap between kept edges and k-nearest neighbors."""
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
    """Compare edge lengths between full and pruned graphs."""
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

# =============================================================================
# BASELINE PRUNING FUNCTIONS
# =============================================================================

def build_nearest_k(D, k):
    """Keep k nearest neighbors per node."""
    n = D.shape[0]
    keep_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        idx = np.argsort(D[i])
        neighbors = [j for j in idx if j != i][:k]
        for j in neighbors:
            keep_mask[i, j] = True
            keep_mask[j, i] = True
    return keep_mask

def build_random_k(n, k):
    """Keep k random neighbors per node."""
    keep_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        choices = random.sample([j for j in range(n) if j != i], k)
        for j in choices:
            keep_mask[i, j] = True
            keep_mask[j, i] = True
    return keep_mask

# =============================================================================
# CONNECTIVITY ENFORCEMENT (fixed version)
# =============================================================================

def enforce_connectivity(kept_edges, edge_mapping, n_nodes, D, candidate_edges=None, edge_to_idx=None):
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
        kept_edges.append(edge_to_idx[(min(u, v), max(u, v))])

    return sorted(set(kept_edges))

# =============================================================================
# OUTPUT FUNCTIONS (matching GNN version exactly)
# =============================================================================

def save_edge_list(kept_edges, edge_mapping, output_path):
    """Save pruned edges as a simple edge list."""
    with open(output_path, 'w') as f:
        f.write(f"# Pruned edge list\n")
        f.write(f"# Total edges: {len(kept_edges)}\n")
        for edge_idx in kept_edges:
            u, v = edge_mapping[edge_idx]
            f.write(f"{u} {v}\n")

def save_sparse_tsp(D, kept_edges, edge_mapping, n_nodes, output_path, name="pruned"):
    kept_set = set()
    for edge_idx in kept_edges:
        u, v = edge_mapping[edge_idx]
        kept_set.add((u, v))
        kept_set.add((v, u))

    with open(output_path, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"COMMENT: Pruned graph with {len(kept_edges)} edges\n")
        f.write(f"DIMENSION: {n_nodes}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write(f"EDGE_WEIGHT_SECTION\n")
        for i in range(n_nodes):
            row = []
            for j in range(n_nodes):
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
    """Save in LKH candidate format."""
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
            f.write(f"{node+1}")
            for neighbor, dist in neighbors:
                f.write(f" {neighbor+1} {dist}")
            f.write("\n")
        f.write("-1\n")

def save_concorde_format(coords, kept_edges, edge_mapping, D, output_path):
    """
    Save in Concorde format with penalties for non-candidate edges.
    """
    n = len(coords)
    D_max = float(D.max())
    PENALTY_FACTOR = 3.0
    penalty = PENALTY_FACTOR * D_max
    SCALE = 100
    
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
        for i in range(n):
            row = []
            for j in range(i + 1, n):
                base = float(D[i, j])
                if (min(i, j), max(i, j)) in candidate_edges:
                    w = base
                else:
                    w = base + penalty
                row.append(int(round(w * SCALE)))
            if row:
                f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")
    
    return SCALE

def write_metadata(instance_name, n_nodes, original_edges, kept_edges, k, method, 
                   coords, edge_mapping, D, opt_tour_dir): #, edges_added=0):
    """
    Generate comprehensive metadata matching GNN version exactly.
    """
    # Get kept edges list
    kept_edges_list = list(kept_edges) if isinstance(kept_edges, set) else kept_edges
    
    # Base metadata
    metadata = {
        'instance_name': instance_name,
        'n_nodes': n_nodes,
        'original_edges': original_edges,
        'kept_edges': len(kept_edges_list),
        'k': k,
        'method': method,
        'sparsity': (len(kept_edges_list) / original_edges * 100) if original_edges > 0 else 0,
        'timestamp': datetime.now().isoformat()
        #'connectivity_edges_added': edges_added
    }
    
    # Compute all metrics (identical to GNN version)
    try:
        metadata.update(compute_degree_stats(kept_edges_list, edge_mapping, n_nodes))
    except Exception as e:
        print(f"  Warning: degree stats failed: {e}")
    
    try:
        metadata.update(compute_connectivity(kept_edges_list, edge_mapping, n_nodes))
    except Exception as e:
        print(f"  Warning: connectivity failed: {e}")
    
    try:
        metadata.update(compute_knn_overlap(coords, kept_edges_list, edge_mapping))
    except Exception as e:
        print(f"  Warning: knn overlap failed: {e}")
        metadata['avg_knn_overlap'] = 0.0
    
    try:
        metadata.update(compute_edge_length_stats(kept_edges_list, edge_mapping, D))
    except Exception as e:
        print(f"  Warning: edge length stats failed: {e}")
    
    # Add optimal tour metrics if available
    tour_path = os.path.join(opt_tour_dir, f"{instance_name}.opt.tour")
    if os.path.exists(tour_path):
        try:
            opt_tour = parse_opt_tour(tour_path)
            if opt_tour:
                metadata.update(
                    compute_tour_metrics(kept_edges_list, edge_mapping, opt_tour, n_nodes)
                )
        except Exception as e:
            print(f"  Warning: tour metrics failed: {e}")
    
    # Add feasible flag (connected and tour not broken if tour available)
    metadata['feasible'] = metadata.get('is_connected', False)
    if 'tour_broken' in metadata:
        metadata['feasible'] = metadata['feasible'] and (not metadata['tour_broken'])
    
    return metadata

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    random.seed(12345)
    np.random.seed(12345)
    
    # Create opt_tour directory path (assume it's in the same parent directory)
    opt_tour_dir = os.path.join(os.path.dirname(INPUT_DIR), "tsplib_data")
    if not os.path.exists(opt_tour_dir):
        opt_tour_dir = "tsplib_data"  # fallback
        print(f"Using fallback opt_tour_dir: {opt_tour_dir}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    tsp_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".tsp")]
    print(f"Found {len(tsp_files)} TSP files")
    
    # Master summary for all runs
    all_summaries = []

    for fname in tsp_files:
        instance_name = os.path.splitext(fname)[0]
        tsp_path = os.path.join(INPUT_DIR, fname)

        parsed = parse_tsp_fixed(tsp_path)
        if parsed is None or 'D' not in parsed:
            print(f"  Skipping {instance_name}: No distance matrix")
            continue

        n_nodes = parsed["n"]

        # Enforce node range
        if n_nodes < MIN_N or n_nodes > MAX_N:
            print(f"  Skipping {instance_name}: n={n_nodes} outside [{MIN_N}, {MAX_N}]")
            continue

        D = parsed["D"]
        coords = parsed.get("coords", None)
        
        # If no coordinates, create dummy coordinates using MDS
        if coords is None:
            print(f"  Warning: {instance_name} has no coordinates, creating from distance matrix")
            try:
                from sklearn.manifold import MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=100)
                coords = mds.fit_transform(D)
            except ImportError:
                print(f"  Error: sklearn not available, cannot create coordinates")
                continue

        print(f"\nProcessing {instance_name} (n={n_nodes})")

        # Build edge mapping (complete graph)
        edge_mapping = {}
        idx = 0
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                edge_mapping[idx] = (i, j)
                idx += 1
        edge_to_idx = {
            (min(i, j), max(i, j)): idx
            for idx, (i, j) in edge_mapping.items()
        }
        
        original_edges = len(edge_mapping)

        # Process both baseline methods
        for method in ["random_k", "nearest_k"]:
            print(f"  Method: {method}")
            
            for k in K_VALUES:
                # Create output directory structure matching GNN version
                out_dir = os.path.join(OUTPUT_ROOT, method, f"k{k}")
                os.makedirs(out_dir, exist_ok=True)

                # Generate initial mask
                if method == "nearest_k":
                    keep_mask = build_nearest_k(D, k)
                else:  # random_k
                    keep_mask = build_random_k(n_nodes, k)

                # Convert mask -> edge indices
                kept_edges = []
                for i in range(n_nodes):
                    for j in range(i+1, n_nodes):
                        if keep_mask[i, j]:
                            kept_edges.append(edge_to_idx[(min(i, j), max(i, j))])

                # original_edge_count = len(kept_edges)

                # # Enforce connectivity
                # kept_edges = enforce_connectivity(
                #     kept_edges=kept_edges,
                #     edge_mapping=edge_mapping,
                #     n_nodes=n_nodes,
                #     D=D,
                #     edge_to_idx=edge_to_idx
                # )
                # edges_added = len(kept_edges) - original_edge_count

                # Generate comprehensive metadata (matches GNN version)
                metadata = write_metadata(
                    instance_name=instance_name,
                    n_nodes=n_nodes,
                    original_edges=original_edges,
                    kept_edges=kept_edges,
                    k=k,
                    method=method,
                    coords=coords,
                    edge_mapping=edge_mapping,
                    D=D,
                    opt_tour_dir=opt_tour_dir
                    #edges_added=edges_added
                )
                
                # Add to master summary
                summary_row = {
                    'instance': instance_name,
                    'method': method,
                    'k': k,
                    'n_nodes': n_nodes,
                    'kept_edges': metadata['kept_edges'],
                    'sparsity': metadata['sparsity'],
                    'avg_degree': metadata['avg_degree'],
                    'is_connected': metadata['is_connected'],
                    'num_connected_components': metadata['num_connected_components'],
                    'avg_knn_overlap': metadata.get('avg_knn_overlap', 0),
                    'length_reduction_ratio': metadata.get('length_reduction_ratio', 0),
                    'tour_edge_recall': metadata.get('tour_edge_recall', 0),
                    'feasible': metadata.get('feasible', False),
                    #'connectivity_edges_added': edges_added
                }
                all_summaries.append(summary_row)

                # Convert kept_edges back to mask for output
                keep_mask_final = np.zeros((n_nodes, n_nodes), dtype=bool)
                for e_idx in kept_edges:
                    u, v = edge_mapping[e_idx]
                    keep_mask_final[u, v] = True
                    keep_mask_final[v, u] = True

                # Write ALL output formats (matching GNN version)
                instance_dir = os.path.join(out_dir, instance_name)
                os.makedirs(instance_dir, exist_ok=True)

                # 1. Edge list
                save_edge_list(kept_edges, edge_mapping,
                             os.path.join(instance_dir, f"{instance_name}_edges.txt"))

                # 2. Sparse TSP
                output_path = os.path.join(instance_dir, f"{instance_name}_sparse.tsp")
                save_sparse_tsp(D, kept_edges, edge_mapping, n_nodes, output_path, instance_name)


                # 3. Adjacency CSV
                save_adjacency_csv(n_nodes, kept_edges, edge_mapping, D,
                                 os.path.join(instance_dir, f"{instance_name}_adjacency.csv"))

                # 4. LKH candidates
                save_lkh_candidates(n_nodes, kept_edges, edge_mapping, D,
                                  os.path.join(instance_dir, f"{instance_name}_candidates.txt"))

                # 5. Concorde format
                scale = save_concorde_format(coords, kept_edges, edge_mapping, D,
                                           os.path.join(instance_dir, f"{instance_name}_concorde.tsp"))
                metadata['concorde_scale'] = scale

                # 6. Metadata JSON (comprehensive)
                metadata_path = os.path.join(instance_dir, f"{instance_name}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"    k={k}: kept {metadata['kept_edges']} edges, "
                      f"sparsity={metadata['sparsity']:.1f}%, "
                      f"recall={metadata.get('tour_edge_recall', 0):.3f}, "
                      f"connected={metadata['is_connected']}")

    # Save master summary
    if all_summaries:
        master_df = pd.DataFrame(all_summaries)
        master_summary_path = os.path.join(OUTPUT_ROOT, "baseline_master_summary.csv")
        master_df.to_csv(master_summary_path, index=False)
        
        print(f"\n{'='*70}")
        print("BASELINE PRUNING COMPLETE")
        print('='*70)
        
        # Print summary statistics grouped by method and k
        for method in ["nearest_k", "random_k"]:
            method_df = master_df[master_df['method'] == method]
            if len(method_df) > 0:
                print(f"\n{method.upper()} Summary:")
                grouped = method_df.groupby('k').agg({
                    'n_nodes': 'mean',
                    'kept_edges': 'mean',
                    'sparsity': 'mean',
                    'avg_degree': 'mean',
                    'tour_edge_recall': 'mean',
                    'is_connected': lambda x: (x == True).mean() * 100,
                    'feasible': lambda x: (x == True).mean() * 100
                }).round(3)
                print(grouped.to_string())
        
        print(f"\n✅ Results saved to: {OUTPUT_ROOT}")
        print(f"   Master summary: {master_summary_path}")
    else:
        print("\n⚠️ No graphs were processed")

if __name__ == "__main__":
    main()