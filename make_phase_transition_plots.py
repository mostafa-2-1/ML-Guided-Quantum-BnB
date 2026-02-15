# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.optimize import curve_fit
# from scipy.stats import bootstrap
# from matplotlib.patches import Rectangle
# import warnings

# # Suppress specific warnings
# warnings.filterwarnings('ignore', category=FutureWarning)

# # ================================
# # CONFIG — EDIT THESE PATHS ONLY
# # ================================

# PRUNED_ROOT = "pruned_graphs_k"
# SOLVER_ROOT = "solver_results/seed1"
# OUTPUT_DIR = "phase_transition_plots"

# K_VALUES = [2, 3, 4, 5, 7, 8, 10, 15, 20, 25]

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ================================
# # Load FULL solver results
# # ================================

# full_results = pd.read_csv(os.path.join(SOLVER_ROOT, "full", "results.csv"))
# full_results = full_results.rename(columns={"cost": "full_cost"})
# full_results = full_results.set_index("instance")

# # ================================
# # Load PRUNED solver results
# # ================================

# solver_tables = {}

# for k in K_VALUES:
#     path = os.path.join(SOLVER_ROOT, f"k{k}", "results.csv")
#     df = pd.read_csv(path)
#     df = df.set_index("instance")
#     solver_tables[k] = df

# # ================================
# # Load metadata JSONs
# # ================================

# records = []

# for k in K_VALUES:
#     k_dir = os.path.join(PRUNED_ROOT, f"k{k}")
#     if not os.path.isdir(k_dir):
#         continue
    
#     for inst in os.listdir(k_dir):
#         inst_dir = os.path.join(k_dir, inst)
#         if not os.path.isdir(inst_dir):
#             continue
        
#         meta_path = os.path.join(inst_dir, f"{inst}_metadata.json")
#         if not os.path.exists(meta_path):
#             continue
        
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
        
#         instance = meta["instance_name"]
        
#         # attach solver results
#         if instance not in solver_tables[k].index:
#             continue
        
#         sol = solver_tables[k].loc[instance]
        
#         record = {
#             "instance": instance,
#             "k": k,
#             "n": meta["n_nodes"],
#             "sparsity": meta["sparsity"],
#             "avg_degree": meta["avg_degree"],
#             "is_connected": meta["is_connected"],
#             "feasible": meta["feasible"],
#             "gap_rel": sol["gap_rel_percent"],
#             "success": sol["success"]
#         }
        
#         records.append(record)

# df = pd.DataFrame(records)

# print("Loaded records:", len(df))
# print("Data overview:")
# print(f"Success rate: {df['success'].mean():.2%}")
# print(f"Feasibility rate: {df['feasible'].mean():.2%}")
# print(f"Average degree range: [{df['avg_degree'].min():.2f}, {df['avg_degree'].max():.2f}]")

# # ================================
# # Plot 1 — Gap vs Avg Degree
# # ================================

# plt.figure()
# for connected in [True, False]:
#     subset = df[df["is_connected"] == connected]
#     plt.scatter(subset["avg_degree"], subset["gap_rel"],
#                 marker="o" if connected else "x",
#                 label=f"connected={connected}")

# plt.xlabel("Average Degree")
# plt.ylabel("Optimality Gap (%)")
# plt.legend()
# plt.title("Gap vs Avg Degree")
# plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_avg_degree.png"), dpi=300)
# plt.close()

# # ================================
# # Plot 2 — Gap vs Sparsity
# # ================================

# plt.figure()
# plt.scatter(df["sparsity"], df["gap_rel"])
# plt.xlabel("Sparsity (%)")
# plt.ylabel("Optimality Gap (%)")
# plt.title("Gap vs Sparsity")
# plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_sparsity.png"), dpi=300)
# plt.close()

# # ================================
# # Plot 3 — Logistic P(success) vs Avg Degree
# # ================================

# def logistic(x, beta, x0):
#     z = np.clip(beta * (x - x0), -500, 500)  # prevent overflow
#     return 1 / (1 + np.exp(-z))

# # Filter out data with extreme values if needed
# mask = ~df[["avg_degree", "success"]].isna().any(axis=1)
# xdata = df.loc[mask, "avg_degree"].values
# ydata = df.loc[mask, "success"].astype(int).values

# print(f"\nLogistic regression data: {len(xdata)} points")

# # Fit logistic
# try:
#     params, pcov = curve_fit(
#         logistic,
#         xdata,
#         ydata,
#         p0=(1.0, np.median(xdata)),
#         maxfev=5000
#     )
    
#     beta_hat, x0_hat = params
#     param_errors = np.sqrt(np.diag(pcov))
#     print(f"Fitted parameters: beta={beta_hat:.3f} ± {param_errors[0]:.3f}, "
#           f"x0={x0_hat:.3f} ± {param_errors[1]:.3f}")
    
# except RuntimeError as e:
#     print(f"Logistic fit failed: {e}")
#     # Use median as fallback
#     x0_hat = np.median(xdata)
#     beta_hat = 1.0
#     param_errors = [np.nan, np.nan]

# # Try bootstrap with percentile method (more robust than BCa)
# def fit_x0(x, y):
#     """Helper function for bootstrap"""
#     x = np.asarray(x)
#     y = np.asarray(y)
    
#     # Check if we have enough variation
#     if len(x) < 10 or len(np.unique(y)) < 2:
#         return np.nan
    
#     try:
#         p, _ = curve_fit(
#             logistic,
#             x,
#             y,
#             p0=(1.0, np.median(x)),
#             maxfev=5000
#         )
#         return p[1]  # Return x0
#     except (RuntimeError, ValueError):
#         return np.nan

# # Bootstrap with simpler percentile method
# rng = np.random.default_rng(42)
# n_resamples = 2000  # Reduced for speed, increase if needed
# x0_bootstrap = []

# for i in range(n_resamples):
#     # Resample with replacement
#     indices = rng.integers(0, len(xdata), len(xdata))
#     x_resampled = xdata[indices]
#     y_resampled = ydata[indices]
    
#     x0_val = fit_x0(x_resampled, y_resampled)
#     if not np.isnan(x0_val):
#         x0_bootstrap.append(x0_val)

# x0_bootstrap = np.array(x0_bootstrap)

# if len(x0_bootstrap) > 0:
#     # Calculate percentile-based CI
#     x0_low = np.percentile(x0_bootstrap, 2.5)
#     x0_high = np.percentile(x0_bootstrap, 97.5)
#     print(f"Bootstrap: {len(x0_bootstrap)} successful resamples out of {n_resamples}")
# else:
#     x0_low, x0_high = np.nan, np.nan
#     print("Warning: Bootstrap failed to produce any valid estimates")

# # Smooth curve for plotting
# x_plot = np.linspace(min(xdata), max(xdata), 200)
# y_plot = logistic(x_plot, beta_hat, x0_hat)

# plt.figure(figsize=(10, 6))
# plt.scatter(xdata, ydata, alpha=0.5, label="Data points", s=30)

# # Only plot fit if we have valid parameters
# if not np.isnan(beta_hat):
#     plt.plot(x_plot, y_plot, 'r-', linewidth=2, label="Logistic fit")
#     plt.axvline(x0_hat, color='red', linestyle='--', 
#                 label=f"Critical degree = {x0_hat:.2f}")
    
#     # Plot CI if available
#     if not np.isnan(x0_low):
#         plt.axvspan(x0_low, x0_high, alpha=0.2, color='red', 
#                    label=f"95% CI: [{x0_low:.2f}, {x0_high:.2f}]")

# plt.xlabel("Average Degree")
# plt.ylabel("P(success)")
# plt.title(f"Solver Success Phase Transition (n={len(xdata)})")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "logistic_phase_transition.png"), dpi=300)
# plt.close()

# print(f"\nCritical degree: {x0_hat:.3f}")
# if not np.isnan(x0_low):
#     print(f"95% CI: [{x0_low:.3f}, {x0_high:.3f}]")
# else:
#     print("95% CI: Could not compute")

# # ================================
# # Plot 4 — Feasibility vs Sparsity
# # ================================

# # Bin sparsity with explicit observed parameter
# bins = np.linspace(df["sparsity"].min(), df["sparsity"].max(), 8)
# df["sparsity_bin"] = pd.cut(df["sparsity"], bins)

# # Group with observed=False to suppress warning
# feas = df.groupby("sparsity_bin", observed=False)["feasible"].mean()
# bin_centers = [b.mid for b in feas.index.categories]

# plt.figure()
# plt.plot(bin_centers, feas.values, marker="o", linewidth=2)
# plt.xlabel("Sparsity (%)")
# plt.ylabel("Fraction Feasible")
# plt.title("Feasibility vs Sparsity")
# plt.grid(True, alpha=0.3)
# plt.savefig(os.path.join(OUTPUT_DIR, "feasibility_vs_sparsity.png"), dpi=300)
# plt.close()

# # ================================
# # Additional diagnostic plot: Success rate by avg_degree bins
# # ================================

# plt.figure(figsize=(10, 6))
# # Create equal-sized bins for avg_degree
# df_sorted = df.sort_values("avg_degree")
# n_bins = min(10, len(df_sorted) // 5)  # Ensure enough points per bin
# if n_bins > 1:
#     df_sorted["degree_bin"] = pd.qcut(df_sorted["avg_degree"], n_bins, duplicates='drop')
#     success_by_bin = df_sorted.groupby("degree_bin", observed=False)["success"].agg(['mean', 'count'])
    
#     # Create custom x positions
#     bin_centers = []
#     bin_labels = []
#     for bin_val in success_by_bin.index.categories:
#         bin_centers.append(bin_val.mid)
#         bin_labels.append(f"{bin_val.left:.1f}-{bin_val.right:.1f}")
    
#     plt.bar(range(len(bin_centers)), success_by_bin['mean'].values, alpha=0.7)
#     plt.xticks(range(len(bin_centers)), bin_labels, rotation=45, ha='right')
    
#     # Add count labels on bars
#     for i, (mean_val, count_val) in enumerate(zip(success_by_bin['mean'], success_by_bin['count'])):
#         plt.text(i, mean_val + 0.02, f"n={count_val}", ha='center', fontsize=8)
    
#     plt.xlabel("Average Degree (binned)")
#     plt.ylabel("Success Rate")
#     plt.title(f"Success Rate by Average Degree Bins ({n_bins} bins)")
#     plt.ylim(0, 1.1)
#     plt.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "success_rate_by_degree_bins.png"), dpi=300)
#     plt.close()

# print("\nAll plots saved to:", OUTPUT_DIR)
# print(f"Total instances: {len(df)}")
# print(f"Data summary:")
# print(df[['avg_degree', 'sparsity', 'success', 'feasible', 'gap_rel']].describe())






import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# CONFIG
# ============================================================

SOLVER_ROOT = "solver_results"      # change for baseline
PRUNED_ROOT = "pruned_graphs_k"           # change for baseline
OUTPUT_DIR = "structural_analysis_gnn"

K_VALUES = [2,3,4,5,6,7,8,9,10,12,15,17,20,22,25]
N_BOOT = 1000
SEEDS = list(range(1, 11))
RNG = np.random.default_rng(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

# records = []

# for k in K_VALUES:
#     results_path = os.path.join(SOLVER_ROOT, f"k{k}", "results.csv")
#     if not os.path.exists(results_path):
#         continue

#     df_k = pd.read_csv(results_path)

#     for _, row in df_k.iterrows():
#         instance = row["instance"]

#         meta_path = os.path.join(
#             PRUNED_ROOT, f"k{k}", instance, f"{instance}_metadata.json"
#         )

#         if not os.path.exists(meta_path):
#             continue

#         meta = pd.read_json(meta_path, typ="series")

#         records.append({
#             "instance": instance,
#             "k": k,
#             "n": meta["n_nodes"],
#             "avg_degree": meta["avg_degree"],
#             "sparsity": meta["sparsity"],
#             "is_connected": meta["is_connected"],
#             "feasible": meta["feasible"],
#             "gap": row["gap_rel_percent"],
#             "success": row["success"]
#         })

# df = pd.DataFrame(records)

# if len(df) == 0:
#     raise ValueError("No data found.")

# df.to_csv(os.path.join(OUTPUT_DIR, "full_structural_dataframe.csv"), index=False)

# print("Total records:", len(df))

records = []

for k in K_VALUES:
    # Collect all seeds for this k
    seed_data = {}

    for seed in SEEDS:
        results_path = os.path.join(SOLVER_ROOT, f"seed{seed}", f"k{k}", "results.csv")
        if not os.path.exists(results_path):
            continue

        df_k = pd.read_csv(results_path)

        for _, row in df_k.iterrows():
            instance = row["instance"]
            if instance not in seed_data:
                seed_data[instance] = {"gaps": [], "successes": []}
            seed_data[instance]["gaps"].append(row["gap_rel_percent"])
            seed_data[instance]["successes"].append(row["success"])

    # Aggregate per instance
    for instance, data in seed_data.items():
        meta_path = os.path.join(
            PRUNED_ROOT, f"k{k}", instance, f"{instance}_metadata.json"
        )
        if not os.path.exists(meta_path):
            continue

        meta = pd.read_json(meta_path, typ="series")

        records.append({
            "instance": instance,
            "k": k,
            "n": meta["n_nodes"],
            "avg_degree": meta["avg_degree"],
            "sparsity": meta["sparsity"],
            "is_connected": meta["is_connected"],
            "feasible": meta["feasible"],
            "gap": np.mean(data["gaps"]),
            "success": np.mean(data["successes"]),
            "n_seeds": len(data["successes"]),
            "success_std": np.std(data["successes"])
        })

df = pd.DataFrame(records)

if len(df) == 0:
    raise ValueError("No data found.")

df.to_csv(os.path.join(OUTPUT_DIR, "full_structural_dataframe.csv"), index=False)

print(f"Total records: {len(df)}")
print(f"Seeds aggregated: {len(SEEDS)}")
print(f"Success values: {sorted(df['success'].unique())}")

# ============================================================
# 1️⃣ MACRO LEVEL — SUCCESS & GAP VS k (WITH CI)
# ============================================================

macro = df.groupby("k").agg(
    success_rate=("success", "mean"),
    mean_gap=("gap", "mean"),
    std_gap=("gap", "std"),
    count=("success", "count")
).reset_index()

# Bernoulli standard error
macro["success_se"] = np.sqrt(
    macro["success_rate"] * (1 - macro["success_rate"]) / macro["count"]
)

macro["success_ci_low"] = macro["success_rate"] - 1.96 * macro["success_se"]
macro["success_ci_high"] = macro["success_rate"] + 1.96 * macro["success_se"]

macro.to_csv(os.path.join(OUTPUT_DIR, "macro_k_summary.csv"), index=False)

# Success vs k
plt.figure()
plt.errorbar(
    macro["k"],
    macro["success_rate"],
    yerr=1.96 * macro["success_se"],
    marker="o"
)
plt.xlabel("k")
plt.ylabel("Success Rate")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_k.png"), dpi=300)
plt.close()

# Gap vs k
plt.figure()
plt.errorbar(
    macro["k"],
    macro["mean_gap"],
    yerr=macro["std_gap"],
    marker="o"
)
plt.xlabel("k")
plt.ylabel("Mean Optimality Gap (%)")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_k.png"), dpi=300)
plt.close()

# ============================================================
# 2️⃣ STRUCTURAL VIEW — AVG DEGREE BINNING (WITH CI)
# ============================================================

num_bins = 12
bins = np.linspace(df["avg_degree"].min(), df["avg_degree"].max(), num_bins)
df["degree_bin"] = pd.cut(df["avg_degree"], bins)

grouped = df.groupby("degree_bin", observed=False)

struct = grouped.agg(
    success_rate=("success", "mean"),
    mean_gap=("gap", "mean"),
    std_gap=("gap", "std"),
    count=("success", "count")
).reset_index()

struct["bin_center"] = struct["degree_bin"].apply(lambda x: x.mid).astype(float)

struct = struct[struct["count"] >= 5]

# Bernoulli CI
struct["success_se"] = np.sqrt(
    struct["success_rate"] * (1 - struct["success_rate"]) / struct["count"]
)

struct["success_ci_low"] = struct["success_rate"] - 1.96 * struct["success_se"]
struct["success_ci_high"] = struct["success_rate"] + 1.96 * struct["success_se"]

struct.to_csv(os.path.join(OUTPUT_DIR, "structural_degree_summary.csv"), index=False)

# Success vs avg_degree
plt.figure()
plt.errorbar(
    struct["bin_center"],
    struct["success_rate"],
    yerr=1.96 * struct["success_se"],
    marker="o"
)
plt.xlabel("Average Degree")
plt.ylabel("Success Rate")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "success_vs_avg_degree.png"), dpi=300)
plt.close()

# Gap vs avg_degree
plt.figure()
plt.errorbar(
    struct["bin_center"],
    struct["mean_gap"],
    yerr=struct["std_gap"],
    marker="o"
)
plt.xlabel("Average Degree")
plt.ylabel("Mean Optimality Gap (%)")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "gap_vs_avg_degree.png"), dpi=300)
plt.close()

# ============================================================
# 3️⃣ LOGISTIC PHASE TRANSITION (RAW BERNOULLI + BOOTSTRAP)
# ============================================================

def logistic(x, beta, x0):
    z = np.clip(beta * (x - x0), -500, 500)
    return 1 / (1 + np.exp(-z))

x_raw = df["avg_degree"].values
y_raw = df["success"].astype(float).values

# Initial fit
params, _ = curve_fit(
    logistic,
    x_raw,
    y_raw,
    p0=(1.0, np.median(x_raw)),
    maxfev=10000
)

beta_hat, x0_hat = params

# Bootstrap
beta_samples = []
x0_samples = []

for _ in range(N_BOOT):
    idx = RNG.choice(len(df), len(df), replace=True)
    x_boot = x_raw[idx]
    y_boot = y_raw[idx]

    try:
        params_boot, _ = curve_fit(
            logistic,
            x_boot,
            y_boot,
            p0=(beta_hat, x0_hat),
            maxfev=5000
        )
        beta_samples.append(params_boot[0])
        x0_samples.append(params_boot[1])
    except:
        continue

beta_samples = np.array(beta_samples)
x0_samples = np.array(x0_samples)

beta_ci = np.percentile(beta_samples, [2.5, 97.5])
x0_ci = np.percentile(x0_samples, [2.5, 97.5])

# Confidence band
x_plot = np.linspace(min(x_raw), max(x_raw), 300)

y_boot_curves = [
    logistic(x_plot, b, x0)
    for b, x0 in zip(beta_samples, x0_samples)
]

y_boot_curves = np.array(y_boot_curves)

y_lower = np.percentile(y_boot_curves, 2.5, axis=0)
y_upper = np.percentile(y_boot_curves, 97.5, axis=0)
y_plot = logistic(x_plot, beta_hat, x0_hat)

# Plot
plt.figure(figsize=(10, 6))

jitter = RNG.normal(0, 0.015, size=len(y_raw))
plt.scatter(
    x_raw,
    y_raw + jitter,
    alpha=0.3,
    s=20,
    c="steelblue",
    label="Data points",
    zorder=3
)

plt.plot(x_plot, y_plot, "r-", linewidth=2, label="Logistic fit")

plt.fill_between(
    x_plot, y_lower, y_upper,
    alpha=0.2, color="red",
    label=f"95% CI"
)

plt.axvline(
    x0_hat, color="red", linestyle="--", alpha=0.7,
    label=f"Critical degree = {x0_hat:.1f} [{x0_ci[0]:.1f}, {x0_ci[1]:.1f}]"
)

plt.xlabel("Average Degree", fontsize=12)
plt.ylabel("P(success)", fontsize=12)
plt.title(f"Solver Success Phase Transition (n={len(df)})", fontsize=13)
plt.legend(fontsize=10, loc="lower right")
plt.ylim(-0.1, 1.1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_phase_transition.png"), dpi=300)
plt.close()

# Save logistic parameters
logistic_df = pd.DataFrame({
    "beta_hat": [beta_hat],
    "beta_ci_low": [beta_ci[0]],
    "beta_ci_high": [beta_ci[1]],
    "x0_hat": [x0_hat],
    "x0_ci_low": [x0_ci[0]],
    "x0_ci_high": [x0_ci[1]]
})

logistic_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_parameters.csv"), index=False)

print("Critical avg degree:", x0_hat)
print("x0 95% CI:", x0_ci)

# ============================================================
# 4️⃣ PER-INSTANCE CRITICAL k + BOOTSTRAP CI
# ============================================================

critical_k = []

for inst, sub in df.groupby("instance"):
    sub_sorted = sub.sort_values("k")
    success_ks = sub_sorted[sub_sorted["success"] >= 0.5]["k"]
    if len(success_ks) > 0:
        critical_k.append(success_ks.iloc[0])

critical_df = pd.DataFrame({"critical_k": critical_k})
critical_df.to_csv(os.path.join(OUTPUT_DIR, "per_instance_critical_k.csv"), index=False)

# Histogram
plt.figure()
plt.hist(critical_df["critical_k"], bins=10)
plt.xlabel("Critical k")
plt.ylabel("Number of Instances")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "critical_k_distribution.png"), dpi=300)
plt.close()

# Bootstrap mean critical k
mean_samples = []
ck_values = critical_df["critical_k"].values

for _ in range(N_BOOT):
    sample = RNG.choice(ck_values, len(ck_values), replace=True)
    mean_samples.append(np.mean(sample))

mean_samples = np.array(mean_samples)
mean_ci = np.percentile(mean_samples, [2.5, 97.5])

print("Mean critical k:", np.mean(ck_values))
print("Mean critical k 95% CI:", mean_ci)

print("\n=== Structural Navigability Analysis Complete (Statistically Rigorous) ===")


# ============================================================
# FINAL SUMMARY
# ============================================================

# Count failing instances
n_total = df["instance"].nunique()
n_never = n_total - len(critical_k)

ck_values = np.array(critical_k)

print("\n" + "=" * 60)
print("STRUCTURAL ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total data points: {len(df)}")
print(f"Instances: {df['instance'].nunique()}")
print(f"K values: {sorted(df['k'].unique())}")
print(f"\nLogistic fit:")
print(f"  β = {beta_hat:.4f}  CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
print(f"  d₀ = {x0_hat:.2f}  CI: [{x0_ci[0]:.2f}, {x0_ci[1]:.2f}]")
print(f"  Bootstrap samples: {len(x0_samples)}/{N_BOOT}")
print(f"\nCritical k:")
print(f"  Mean: {np.mean(ck_values):.2f}  CI: [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
print(f"  Always-failing instances: {n_never}/{n_total}")
print(f"\nOverall success rate: {df['success'].mean()*100:.1f}%")
print(f"Success rate at k≥10: {df[df['k']>=10]['success'].mean()*100:.1f}%")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("=" * 60)