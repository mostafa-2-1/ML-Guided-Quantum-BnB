import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
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

COLORS = {
    "GNN": "#1f77b4",        # Blue
    "Nearest-k": "#ff7f0e",  # Orange
    "Random-k": "#2ca02c"    # Green
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
    plt.plot(df["bin_center"], df["success_rate"], linewidth=3, 
             label=name, color=COLORS[name])

plt.xlabel("Average Degree", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.title("Success Probability vs Average Degree", fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_avg_degree_raw.png"), dpi=300)
plt.close()

# ============================================================
# FIGURE 5.5: SUCCESS VS AVG DEGREE - WITH LOGISTIC FIT OVERLAY
# ============================================================

def logistic_function(x, beta, x0):
    """Standard logistic function: 1 / (1 + exp(-beta * (x - x0)))"""
    return expit(beta * (x - x0))

def load_logistic_params(path):
    """Load logistic parameters from CSV"""
    file = os.path.join(path, "logistic_parameters.csv")
    if not os.path.exists(file):
        return None
    df = pd.read_csv(file)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "beta": row["beta_hat"],
        "x0": row["x0_hat"],
        "beta_ci_low": row.get("beta_ci_low", None),
        "beta_ci_high": row.get("beta_ci_high", None),
        "x0_ci_low": row.get("x0_ci_low", None),
        "x0_ci_high": row.get("x0_ci_high", None)
    }

plt.figure(figsize=(10, 7))

# Store x0 values for annotation
x0_values = {}

for name, path in methods_all.items():
    # Load raw data
    degree_file = os.path.join(path, "structural_degree_summary.csv")
    if not os.path.exists(degree_file):
        continue
    
    df = pd.read_csv(degree_file)
    x_data = df["bin_center"]
    y_data = df["success_rate"]
    
    # Plot raw data as scatter points
    plt.scatter(x_data, y_data, alpha=0.5, s=50, color=COLORS[name], 
                label=f"{name} (data)")
    
    # Load and plot logistic fit
    params = load_logistic_params(path)
    
    if params is not None and name != "Random-k":
        beta = params["beta"]
        x0 = params["x0"]
        x0_values[name] = x0
        
        # Generate smooth curve for logistic fit
        x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit = logistic_function(x_smooth, beta, x0)
        
        plt.plot(x_smooth, y_fit, linewidth=3, color=COLORS[name], 
                 linestyle='-', label=f"{name} (fit: β={beta:.2f}, x₀={x0:.2f})")
        
        # Add vertical line at x0 (transition point)
        plt.axvline(x=x0, color=COLORS[name], linestyle='--', alpha=0.5, linewidth=1.5)
    
    elif name == "Random-k":
        # Random-k: flat line at zero
        plt.axhline(y=0, color=COLORS[name], linestyle='-', linewidth=2, 
                    alpha=0.7, label=f"{name} (baseline)")

# Add annotations for horizontal shift if both GNN and Nearest-k have x0
if "GNN" in x0_values and "Nearest-k" in x0_values:
    shift = x0_values["Nearest-k"] - x0_values["GNN"]
    mid_y = 0.5
    
    # Draw horizontal arrow showing the shift
    plt.annotate('', 
                 xy=(x0_values["GNN"], mid_y), 
                 xytext=(x0_values["Nearest-k"], mid_y),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Add text label for the shift
    mid_x = (x0_values["GNN"] + x0_values["Nearest-k"]) / 2
    plt.text(mid_x, mid_y + 0.08, f'Δx₀ = {shift:.2f}', 
             fontsize=12, ha='center', color='red', fontweight='bold')

plt.xlabel("Average Degree", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.title("Success Probability vs Average Degree (with Logistic Fits)", fontsize=16)
plt.legend(fontsize=10, loc='lower right')
plt.grid(alpha=0.3)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_avg_degree_logistic_overlay.png"), dpi=300)
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
