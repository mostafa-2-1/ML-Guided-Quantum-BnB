import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx

# -----------------------------
# Parsing utilities
# -----------------------------

import numpy as np

def parse_tsp(filepath):
    """
    Parse a .tsp file (TSPLIB). Returns dict:
      - 'name': instance name
      - 'n': number of nodes
      - 'coords': Nx2 array if NODE_COORD_SECTION exists (Euclidean)
      - 'D': NxN distance matrix if EDGE_WEIGHT_TYPE EXPLICIT
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
        for line in f:
            line = line.strip()
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
                for _ in range(n):
                    ln = next(f).strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    idx = int(parts[0]) - 1
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
                reading_coords = False
            elif up.startswith('EDGE_WEIGHT_SECTION'):
                reading_edges = True
            elif up.startswith('DISPLAY_DATA_SECTION') or up.startswith('EOF'):
                reading_edges = False
            elif reading_edges:
                edge_lines.extend(line.split())

    # Build coords array if present
    coords_arr = None
    if coords:
        coords_arr = np.zeros((n, 2), dtype=float)
        for i in range(n):
            coords_arr[i] = coords[i]

    # Build distance matrix if EXPLICIT
    D_arr = None
    if edge_type == 'EXPLICIT' and edge_lines:
        edge_vals = list(map(float, edge_lines))
        D_arr = np.zeros((n, n), dtype=float)
        if edge_format == 'UPPER_ROW':
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    D_arr[i, j] = edge_vals[idx]
                    D_arr[j, i] = edge_vals[idx]
                    idx += 1
        elif edge_format == 'FULL_MATRIX':
            idx = 0
            for i in range(n):
                for j in range(n):
                    D_arr[i, j] = edge_vals[idx]
                    idx += 1
        elif edge_format == 'LOWER_DIAG_ROW':
            idx = 0
            for i in range(n):
                for j in range(i+1):
                    D_arr[i, j] = edge_vals[idx]
                    D_arr[j, i] = edge_vals[idx]  # mirror
                    idx += 1

        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {edge_format}")

    out = {'name': name, 'n': n}
    if coords_arr is not None:
        out['coords'] = coords_arr
    else:
        coords_arr = np.zeros((n,2), dtype=float)
    if D_arr is not None:
        out['D'] = D_arr

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
    return np.sqrt((diff ** 2).sum(axis=2))


def build_candidate_edges(coords, full_threshold=300, knn_k=25):
    n = coords.shape[0]
    D = pairwise_distances(coords)

    if n <= full_threshold:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        edges_set = set()

        for i in range(n):
            k = min(knn_k + 1, n)
            _, idxs = tree.query(coords[i], k=k)
            for j in idxs:
                if j != i:
                    a, b = (i, j) if i < j else (j, i)
                    edges_set.add((a, b))

        edges = sorted(edges_set)

    return edges, D



def compute_node_edge_features(coords, edges, D, knn_k=10):
    n = coords.shape[0]
    coords = np.asarray(coords, dtype=float)

    mean_d = D[np.triu_indices(n, 1)].mean()
    bbox_w = coords[:, 0].ptp() + 1e-6
    bbox_h = coords[:, 1].ptp() + 1e-6

    # ---- Node features ----
    from scipy.spatial import KDTree
    tree = KDTree(coords)

    avg_knn = np.zeros(n)
    for i in range(n):
        k = min(knn_k + 1, n)
        dists, idxs = tree.query(coords[i], k=k)
        avg_knn[i] = np.mean([d for d, j in zip(dists, idxs) if j != i])

    node_feat = np.stack([
        coords[:, 0] / bbox_w,
        coords[:, 1] / bbox_h,
        avg_knn / (mean_d + 1e-12)
    ], axis=1)

    # ---- Rank matrix ----
    sorted_idx = np.argsort(D, axis=1)
    rank = np.zeros_like(sorted_idx)
    for i in range(n):
        rank[i, sorted_idx[i]] = np.arange(n)

    # ---- Edge features ----
    edge_index = [[], []]
    edge_attr = []

    for i, j in edges:
        edge_index[0].append(i)
        edge_index[1].append(j)

        edge_attr.append([
            D[i, j] / (mean_d + 1e-12),
            abs(coords[i, 0] - coords[j, 0]) / bbox_w,
            abs(coords[i, 1] - coords[j, 1]) / bbox_h,
            rank[i, j] / n,
            rank[j, i] / n
        ])

    return (
        node_feat,
        np.array(edge_index, dtype=int),
        np.array(edge_attr, dtype=float)
    )


# -----------------------------
# PyG Data builder
# -----------------------------

def build_pyg_data_from_instance(tsp_path, tour_path,
                                 full_threshold=300,
                                 knn_k=25,
                                 knn_feat_k=10):

    meta = parse_tsp(tsp_path)
    coords = meta['coords']
    n = meta['n']

    tour = parse_opt_tour(tour_path)
    if not tour or len(tour) != n:
        return None

    edges, D = build_candidate_edges(coords, full_threshold, knn_k)

    node_feat, edge_index_np, edge_attr = compute_node_edge_features(
        coords, edges, D, knn_feat_k
    )

    # ---- Labels: edge in optimal tour ----
    true_edges = set()
    for a, b in zip(tour, tour[1:]):
        true_edges.add(tuple(sorted((a, b))))
    true_edges.add(tuple(sorted((tour[-1], tour[0]))))

    y = np.array([
        1.0 if (i, j) in true_edges else 0.0
        for i, j in edges
    ])

    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float)
    )

    data.coords = coords
    data.D = D
    data.edges_list = edges
    data.true_tour = tour
    data.n_nodes = n
    data.name = meta['name']

    return data

def process_folder(folder_path):
    all_data = []
    for f in tqdm(os.listdir(folder_path), desc=f"Processing {folder_path}"):
        if not f.endswith(".tsp"):
            continue
        tsp = os.path.join(folder_path, f)
        tour = tsp.replace(".tsp", ".opt.tour")
        if os.path.exists(tour):
            d = build_pyg_data_from_instance(tsp, tour)
            if d is not None:
                all_data.append(d)
    return all_data




if __name__ == "__main__":
    os.makedirs("data_cache", exist_ok=True)
    graphs = (
        process_folder("tsplib_data") +
        process_folder("synthetic_tsplib")
    )
    torch.save(graphs, "data_cache/pyg_graphs.pt")
    print(f"Saved {len(graphs)} graphs")