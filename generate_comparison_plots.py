import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

GNN_PATH = "structural_analysis_gnn"
KNN_PATH = "structural_analysis_baseline/nearest_k"
RAND_PATH = "structural_analysis_baseline/random_k"

OUTPUT_DIR = "structural_analysis_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

methods_all = {
    "GNN": GNN_PATH,
    "Nearest-k": KNN_PATH,
    "Random-k": RAND_PATH
}

methods_gap_only = {
    "GNN": GNN_PATH,
    "Nearest-k": KNN_PATH
}

# ============================================================
# GAP VS K (ONLY GNN + NEAREST)
# ============================================================

plt.figure(figsize=(8,6))

for name, path in methods_gap_only.items():
    file = os.path.join(path, "macro_k_summary.csv")
    if not os.path.exists(file):
        continue

    df = pd.read_csv(file)
    x = df["k"]
    y = df["mean_gap"]
    se = df["std_gap"] / np.sqrt(df["count"])

    plt.plot(x, y, linewidth=3, label=name)
    plt.fill_between(x, y - se, y + se, alpha=0.25)

plt.xlabel("k", fontsize=14)
plt.ylabel("Mean Gap", fontsize=14)
plt.title("Gap vs k (GNN vs Nearest-k)", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_k_gnn_vs_knn.png"), dpi=300)
plt.close()

# ============================================================
# SUCCESS VS AVG DEGREE (ALL THREE)
# ============================================================

plt.figure(figsize=(8,6))

for name, path in methods_all.items():
    file = os.path.join(path, "structural_degree_summary.csv")
    if not os.path.exists(file):
        continue

    df = pd.read_csv(file)
    plt.plot(df["bin_center"], df["success_rate"], linewidth=3, label=name)

plt.xlabel("Average Degree", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.title("Success vs Average Degree", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_avg_degree_overlay.png"), dpi=300)
plt.close()

# ============================================================
# CRITICAL K DISTRIBUTION (ALL THREE)
# ============================================================

plt.figure(figsize=(8,6))

for name, path in methods_all.items():
    file = os.path.join(path, "per_instance_critical_k.csv")
    if not os.path.exists(file):
        continue

    df = pd.read_csv(file)
    if df.empty:
        continue

    plt.hist(df["critical_k"], bins=8, alpha=0.5, label=name)

plt.xlabel("Critical k", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Critical k Distribution", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "critical_k_distribution.png"), dpi=300)
plt.close()



print("Overlay plots generated.")
