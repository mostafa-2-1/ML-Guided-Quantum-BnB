# superb_tsp_dataset.py
import os
import gzip
import tsplib95
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, InMemoryDataset

# --------- Step 0: Folder setup ---------
DATA_FOLDER = "tsplib_data"  # Folder with all your .tsp.gz and .opt.tour.gz

# --------- Step 1: Helper functions ---------
def load_tsp(filename):
    """Load a .tsp or .tsp.gz instance using tsplib95"""
    if filename.endswith(".gz"):
        with gzip.open(filename, 'rt') as f:
            problem = tsplib95.parse(f)
    else:
        problem = tsplib95.load(filename)
    return problem

def get_edge_features(problem):
    """Compute features and labels for all edges in the TSP instance"""
    coords = problem.node_coords
    n = len(coords)

    # Build distance matrix
    distance_matrix = np.zeros((n+1, n+1))
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j:
                distance_matrix[i,j] = np.linalg.norm(
                    np.array(coords[i]) - np.array(coords[j])
                )

    # Load optimal tour if exists
    try:
        opt_tour_file = problem.filename.replace(".tsp", ".opt.tour")
        if not os.path.exists(opt_tour_file):
            opt_tour_file = opt_tour_file + ".gz"
        if opt_tour_file.endswith(".gz"):
            with gzip.open(opt_tour_file, "rt") as f:
                opt_tour = [int(x) for x in f.read().split() if x.isdigit()]
        else:
            with open(opt_tour_file, "r") as f:
                opt_tour = [int(x) for x in f.read().split() if x.isdigit()]
        edge_set = set((opt_tour[i], opt_tour[i+1]) for i in range(len(opt_tour)-1))
        edge_set.add((opt_tour[-1], opt_tour[0]))  # loop back
    except Exception:
        edge_set = set()

    data_list = []
    # Generate all possible edges (u,v)
    for u, v in combinations(range(1, n+1), 2):
        dist = distance_matrix[u,v]
        # Rank among nearest neighbors for both u and v
        nn_rank_u = np.sum(distance_matrix[u,1:] < dist)
        nn_rank_v = np.sum(distance_matrix[v,1:] < dist)
        label = 1 if (u,v) in edge_set or (v,u) in edge_set else 0
        data_list.append({
            "instance": problem.name,
            "node_u": u,
            "node_v": v,
            "distance": dist,
            "nn_rank_u": nn_rank_u,
            "nn_rank_v": nn_rank_v,
            "label": label
        })
    return pd.DataFrame(data_list)

# --------- Step 2: Load all instances and build dataset ---------
all_dfs = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".tsp") or file.endswith(".tsp.gz"):
        filepath = os.path.join(DATA_FOLDER, file)
        try:
            problem = load_tsp(filepath)
            df = get_edge_features(problem)
            all_dfs.append(df)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"Full dataset size: {full_df.shape}")

# --------- Step 3: Normalize features ---------
scaler = StandardScaler()
full_df[["distance", "nn_rank_u", "nn_rank_v"]] = scaler.fit_transform(
    full_df[["distance", "nn_rank_u", "nn_rank_v"]]
)

# --------- Step 4: Prepare PyTorch Geometric Dataset ---------
class TSPDataset(InMemoryDataset):
    def __init__(self, dataframe, transform=None):
        super(TSPDataset, self).__init__(".", transform)
        self.data, self.slices = self.process_df(dataframe)

    def process_df(self, df):
        data_list = []
        for instance_name, group in df.groupby("instance"):
            # Node features (optional: use coordinates)
            node_indices = np.unique(np.concatenate([group["node_u"].values, group["node_v"].values]))
            num_nodes = len(node_indices)
            x = torch.ones((num_nodes, 1), dtype=torch.float)  # minimal node features
            # Edge features
            edge_index = torch.tensor([group["node_u"].values-1, group["node_v"].values-1], dtype=torch.long)
            edge_attr = torch.tensor(group[["distance","nn_rank_u","nn_rank_v"]].values, dtype=torch.float)
            y = torch.tensor(group["label"].values, dtype=torch.float).unsqueeze(1)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)
        return self.collate(data_list)

# Create dataset
dataset = TSPDataset(full_df)
print(f"PyG dataset created with {len(dataset)} graphs.")

# --------- Step 5: Split into train/test (optional) ---------
train_dataset = dataset[:int(0.8*len(dataset))]
test_dataset = dataset[int(0.8*len(dataset)):]

print(f"Train graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

# Now you’re ready to feed this into a GNN for training
