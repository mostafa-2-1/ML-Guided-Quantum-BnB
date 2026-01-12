import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from matplotlib.patches import Rectangle
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================================
# CONFIG — EDIT THESE PATHS ONLY
# ================================

PRUNED_ROOT = "pruned_graphs_k"
SOLVER_ROOT = "solver_results"
OUTPUT_DIR = "phase_transition_plots"

K_VALUES = [2, 3, 4, 5, 7, 8, 10, 15, 20, 25]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# Load FULL solver results
# ================================

full_results = pd.read_csv(os.path.join(SOLVER_ROOT, "full", "results.csv"))
full_results = full_results.rename(columns={"cost": "full_cost"})
full_results = full_results.set_index("instance")

# ================================
# Load PRUNED solver results
# ================================

solver_tables = {}

for k in K_VALUES:
    path = os.path.join(SOLVER_ROOT, f"k{k}", "results.csv")
    df = pd.read_csv(path)
    df = df.set_index("instance")
    solver_tables[k] = df

# ================================
# Load metadata JSONs
# ================================

records = []

for k in K_VALUES:
    k_dir = os.path.join(PRUNED_ROOT, f"k{k}")
    if not os.path.isdir(k_dir):
        continue
    
    for inst in os.listdir(k_dir):
        inst_dir = os.path.join(k_dir, inst)
        if not os.path.isdir(inst_dir):
            continue
        
        meta_path = os.path.join(inst_dir, f"{inst}_metadata.json")
        if not os.path.exists(meta_path):
            continue
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        instance = meta["instance_name"]
        
        # attach solver results
        if instance not in solver_tables[k].index:
            continue
        
        sol = solver_tables[k].loc[instance]
        
        record = {
            "instance": instance,
            "k": k,
            "n": meta["n_nodes"],
            "sparsity": meta["sparsity"],
            "avg_degree": meta["avg_degree"],
            "is_connected": meta["is_connected"],
            "feasible": meta["feasible"],
            "gap_rel": sol["gap_rel_percent"],
            "success": sol["success"]
        }
        
        records.append(record)

df = pd.DataFrame(records)

print("Loaded records:", len(df))
print("Data overview:")
print(f"Success rate: {df['success'].mean():.2%}")
print(f"Feasibility rate: {df['feasible'].mean():.2%}")
print(f"Average degree range: [{df['avg_degree'].min():.2f}, {df['avg_degree'].max():.2f}]")

# ================================
# Plot 1 — Gap vs Avg Degree
# ================================

plt.figure()
for connected in [True, False]:
    subset = df[df["is_connected"] == connected]
    plt.scatter(subset["avg_degree"], subset["gap_rel"],
                marker="o" if connected else "x",
                label=f"connected={connected}")

plt.xlabel("Average Degree")
plt.ylabel("Optimality Gap (%)")
plt.legend()
plt.title("Gap vs Avg Degree")
plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_avg_degree.png"), dpi=300)
plt.close()

# ================================
# Plot 2 — Gap vs Sparsity
# ================================

plt.figure()
plt.scatter(df["sparsity"], df["gap_rel"])
plt.xlabel("Sparsity (%)")
plt.ylabel("Optimality Gap (%)")
plt.title("Gap vs Sparsity")
plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_sparsity.png"), dpi=300)
plt.close()

# ================================
# Plot 3 — Logistic P(success) vs Avg Degree
# ================================

def logistic(x, beta, x0):
    z = np.clip(beta * (x - x0), -500, 500)  # prevent overflow
    return 1 / (1 + np.exp(-z))

# Filter out data with extreme values if needed
mask = ~df[["avg_degree", "success"]].isna().any(axis=1)
xdata = df.loc[mask, "avg_degree"].values
ydata = df.loc[mask, "success"].astype(int).values

print(f"\nLogistic regression data: {len(xdata)} points")

# Fit logistic
try:
    params, pcov = curve_fit(
        logistic,
        xdata,
        ydata,
        p0=(1.0, np.median(xdata)),
        maxfev=5000
    )
    
    beta_hat, x0_hat = params
    param_errors = np.sqrt(np.diag(pcov))
    print(f"Fitted parameters: beta={beta_hat:.3f} ± {param_errors[0]:.3f}, "
          f"x0={x0_hat:.3f} ± {param_errors[1]:.3f}")
    
except RuntimeError as e:
    print(f"Logistic fit failed: {e}")
    # Use median as fallback
    x0_hat = np.median(xdata)
    beta_hat = 1.0
    param_errors = [np.nan, np.nan]

# Try bootstrap with percentile method (more robust than BCa)
def fit_x0(x, y):
    """Helper function for bootstrap"""
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Check if we have enough variation
    if len(x) < 10 or len(np.unique(y)) < 2:
        return np.nan
    
    try:
        p, _ = curve_fit(
            logistic,
            x,
            y,
            p0=(1.0, np.median(x)),
            maxfev=5000
        )
        return p[1]  # Return x0
    except (RuntimeError, ValueError):
        return np.nan

# Bootstrap with simpler percentile method
rng = np.random.default_rng(42)
n_resamples = 2000  # Reduced for speed, increase if needed
x0_bootstrap = []

for i in range(n_resamples):
    # Resample with replacement
    indices = rng.integers(0, len(xdata), len(xdata))
    x_resampled = xdata[indices]
    y_resampled = ydata[indices]
    
    x0_val = fit_x0(x_resampled, y_resampled)
    if not np.isnan(x0_val):
        x0_bootstrap.append(x0_val)

x0_bootstrap = np.array(x0_bootstrap)

if len(x0_bootstrap) > 0:
    # Calculate percentile-based CI
    x0_low = np.percentile(x0_bootstrap, 2.5)
    x0_high = np.percentile(x0_bootstrap, 97.5)
    print(f"Bootstrap: {len(x0_bootstrap)} successful resamples out of {n_resamples}")
else:
    x0_low, x0_high = np.nan, np.nan
    print("Warning: Bootstrap failed to produce any valid estimates")

# Smooth curve for plotting
x_plot = np.linspace(min(xdata), max(xdata), 200)
y_plot = logistic(x_plot, beta_hat, x0_hat)

plt.figure(figsize=(10, 6))
plt.scatter(xdata, ydata, alpha=0.5, label="Data points", s=30)

# Only plot fit if we have valid parameters
if not np.isnan(beta_hat):
    plt.plot(x_plot, y_plot, 'r-', linewidth=2, label="Logistic fit")
    plt.axvline(x0_hat, color='red', linestyle='--', 
                label=f"Critical degree = {x0_hat:.2f}")
    
    # Plot CI if available
    if not np.isnan(x0_low):
        plt.axvspan(x0_low, x0_high, alpha=0.2, color='red', 
                   label=f"95% CI: [{x0_low:.2f}, {x0_high:.2f}]")

plt.xlabel("Average Degree")
plt.ylabel("P(success)")
plt.title(f"Solver Success Phase Transition (n={len(xdata)})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_phase_transition.png"), dpi=300)
plt.close()

print(f"\nCritical degree: {x0_hat:.3f}")
if not np.isnan(x0_low):
    print(f"95% CI: [{x0_low:.3f}, {x0_high:.3f}]")
else:
    print("95% CI: Could not compute")

# ================================
# Plot 4 — Feasibility vs Sparsity
# ================================

# Bin sparsity with explicit observed parameter
bins = np.linspace(df["sparsity"].min(), df["sparsity"].max(), 8)
df["sparsity_bin"] = pd.cut(df["sparsity"], bins)

# Group with observed=False to suppress warning
feas = df.groupby("sparsity_bin", observed=False)["feasible"].mean()
bin_centers = [b.mid for b in feas.index.categories]

plt.figure()
plt.plot(bin_centers, feas.values, marker="o", linewidth=2)
plt.xlabel("Sparsity (%)")
plt.ylabel("Fraction Feasible")
plt.title("Feasibility vs Sparsity")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "feasibility_vs_sparsity.png"), dpi=300)
plt.close()

# ================================
# Additional diagnostic plot: Success rate by avg_degree bins
# ================================

plt.figure(figsize=(10, 6))
# Create equal-sized bins for avg_degree
df_sorted = df.sort_values("avg_degree")
n_bins = min(10, len(df_sorted) // 5)  # Ensure enough points per bin
if n_bins > 1:
    df_sorted["degree_bin"] = pd.qcut(df_sorted["avg_degree"], n_bins, duplicates='drop')
    success_by_bin = df_sorted.groupby("degree_bin", observed=False)["success"].agg(['mean', 'count'])
    
    # Create custom x positions
    bin_centers = []
    bin_labels = []
    for bin_val in success_by_bin.index.categories:
        bin_centers.append(bin_val.mid)
        bin_labels.append(f"{bin_val.left:.1f}-{bin_val.right:.1f}")
    
    plt.bar(range(len(bin_centers)), success_by_bin['mean'].values, alpha=0.7)
    plt.xticks(range(len(bin_centers)), bin_labels, rotation=45, ha='right')
    
    # Add count labels on bars
    for i, (mean_val, count_val) in enumerate(zip(success_by_bin['mean'], success_by_bin['count'])):
        plt.text(i, mean_val + 0.02, f"n={count_val}", ha='center', fontsize=8)
    
    plt.xlabel("Average Degree (binned)")
    plt.ylabel("Success Rate")
    plt.title(f"Success Rate by Average Degree Bins ({n_bins} bins)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "success_rate_by_degree_bins.png"), dpi=300)
    plt.close()

print("\nAll plots saved to:", OUTPUT_DIR)
print(f"Total instances: {len(df)}")
print(f"Data summary:")
print(df[['avg_degree', 'sparsity', 'success', 'feasible', 'gap_rel']].describe())