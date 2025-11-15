import os
import json
import joblib
import torch
import numpy as np
from newTrain import EdgeGNN

# === Path ===
OUT_DIR = "results"   

# === Load best base GNN just to confirm it loads fine ===
model_path = os.path.join(OUT_DIR, "best_model.pt")
model = EdgeGNN(in_node_feats=6, in_edge_feats=9, hidden_dim=128, n_layers=4, dropout=0.3)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# === Load stored metric JSONs ===
def load_json_metrics(name):
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

base_metrics  = load_json_metrics("best_threshold.json")
stage1_metrics = load_json_metrics("cascade_stage1_thr.json")
stage2_metrics = load_json_metrics("cascade_stage2_thr.json")

# === Print everythingy ===
print("\n=== 📊 Final Stored Model Metrics ===")

if base_metrics:
    print(f"[Base GNN @ thr={base_metrics['threshold']:.4f}] "
          f"Precision={base_metrics['precision']:.4f}  "
          f"Recall={base_metrics['recall']:.4f}  "
          f"F1={base_metrics['f1']:.4f}")
else:
    print("Base GNN metrics not found.")

if stage1_metrics:
    print(f"[Cascade Stage 1] Threshold={stage1_metrics['threshold']:.4f}")
else:
    print("Stage 1 metrics not found.")

if stage2_metrics:
    print(f"[Cascade Stage 2 FINAL] "
          f"Threshold={stage2_metrics['threshold']:.4f}  "
          f"Precision={stage2_metrics['precision']:.4f}  "
          f"Recall={stage2_metrics['recall']:.4f}  "
          f"F1={stage2_metrics['f1']:.4f}")
else:
    print("Stage 2 metrics not found.")

print("\nModel metrics printed successfully.")
