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
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")




def collate_batch(data_list):
    # PyG Batch
    batch = Batch.from_data_list(data_list)
    return batch


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





# Create cascade datasets based on finetuned models
def build_cascade_dataset(data_list, model, prob_cut=0.5, device='cpu'):
    # returns X (num_examples x feat_dim), y (labels)
    Xs = []
    Ys = []
    edge_indices = []
    # model.eval()
    for d in data_list:
        with torch.no_grad():
            b = Batch.from_data_list([d]).to(device)
            logits = model(b.x, b.edge_index, b.edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
        edges = d.edges_list
        edge_attrs = d.edge_attr.cpu().numpy() if hasattr(d, 'edge_attr') else None
        for idx, p in enumerate(probs):
            if p >= prob_cut:
                feat = []
                feat.append(p)
                if edge_attrs is not None:
                    feat.extend(edge_attrs[idx].tolist())
               
                Xs.append(feat)
                Ys.append(int(d.y.cpu().numpy()[idx]))
                edge_indices.append(idx)
    X = np.array(Xs, dtype=float)
    y = np.array(Ys, dtype=int)
    edge_indices = np.array(edge_indices, dtype=int)
    return X, y, edge_indices
