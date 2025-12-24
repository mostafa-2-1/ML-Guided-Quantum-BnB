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

def process_folder(folder_path):
    all_data = []
    files = sorted(os.listdir(folder_path))
    tsp_files = [f for f in files if f.endswith('.tsp')]
    for tsp_file in tqdm(tsp_files, desc=f"Processing {folder_path}"):
        tsp_path = os.path.join(folder_path, tsp_file)
        # corresponding .opt.tour
        tour_file = tsp_file.replace('.tsp', '.opt.tour')
        tour_path = os.path.join(folder_path, tour_file)
        if not os.path.exists(tour_path):
            # skip if no tour
            continue
        data = build_pyg_data_from_instance(tsp_path, tour_path)
        if data is not None:
            all_data.append(data)
    return all_data




if __name__ == "__main__":
    os.makedirs("data_cache", exist_ok=True)
    all_data = process_folder("tsplib_data") + process_folder("synthetic_tsplib")
    torch.save(all_data, "data_cache/pyg_graphs.pt")
    print(f"\n✅ Saved {len(all_data)} graphs")
