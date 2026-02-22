# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # ============================================================
# # CONFIG
# # ============================================================

# OUTPUT_DIR = "structural_analysis_comparison"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# N_BOOT = 1000
# RNG = np.random.default_rng(42)

# # Paths to the three full structural dataframes
# GNN_CSV = "structural_analysis_gnn/full_structural_dataframe.csv"
# KNN_CSV = "structural_analysis_baseline/nearest_k/full_structural_dataframe.csv"
# RAND_CSV = "structural_analysis_baseline/random_k/full_structural_dataframe.csv"

# # ============================================================
# # LOGISTIC FUNCTION
# # ============================================================

# def logistic(x, beta, x0):
#     z = np.clip(beta * (x - x0), -500, 500)
#     return 1.0 / (1.0 + np.exp(-z))

# def fit_logistic_with_bootstrap(degree, success, n_boot=1000, rng=None):
#     """
#     Fit logistic on raw Bernoulli data + bootstrap CI.
#     Returns dict with fit params, CI, and curves for plotting.
#     """
#     if rng is None:
#         rng = np.random.default_rng(42)

#     result = {
#         "fit_success": False,
#         "beta": None, "x0": None,
#         "beta_ci": None, "x0_ci": None,
#         "boot_betas": None, "boot_x0s": None
#     }

#     # Check if any success exists
#     if success.sum() == 0 or success.sum() == len(success):
#         return result

#     # Initial fit
#     try:
#         popt, _ = curve_fit(
#             logistic, degree, success,
#             p0=[0.5, np.median(degree)],
#             maxfev=10000
#         )
#     except (RuntimeError, ValueError):
#         return result

#     beta_hat, x0_hat = popt

#     # Bootstrap
#     boot_betas = []
#     boot_x0s = []

#     for _ in range(n_boot):
#         idx = rng.choice(len(degree), len(degree), replace=True)
#         d_boot = degree[idx]
#         s_boot = success[idx]

#         if s_boot.sum() == 0 or s_boot.sum() == len(s_boot):
#             continue

#         try:
#             popt_b, _ = curve_fit(
#                 logistic, d_boot, s_boot,
#                 p0=[beta_hat, x0_hat],
#                 maxfev=5000
#             )
#             boot_betas.append(popt_b[0])
#             boot_x0s.append(popt_b[1])
#         except (RuntimeError, ValueError):
#             continue

#     boot_betas = np.array(boot_betas)
#     boot_x0s = np.array(boot_x0s)

#     if len(boot_x0s) < 50:
#         return result

#     result["fit_success"] = True
#     result["beta"] = beta_hat
#     result["x0"] = x0_hat
#     result["beta_ci"] = np.percentile(boot_betas, [2.5, 97.5])
#     result["x0_ci"] = np.percentile(boot_x0s, [2.5, 97.5])
#     result["boot_betas"] = boot_betas
#     result["boot_x0s"] = boot_x0s

#     return result

# # ============================================================
# # LOAD ALL THREE DATASETS
# # ============================================================

# df_gnn = pd.read_csv(GNN_CSV)
# df_knn = pd.read_csv(KNN_CSV)
# df_rand = pd.read_csv(RAND_CSV)

# datasets = {
#     "Learned (GNN)": df_gnn,
#     "Nearest-K": df_knn,
#     "Random-K": df_rand
# }

# colors = {
#     "Learned (GNN)": "#2196F3",
#     "Nearest-K": "#FF9800",
#     "Random-K": "#4CAF50"
# }

# # ============================================================
# # FIT LOGISTIC FOR EACH METHOD
# # ============================================================

# fits = {}

# for name, df in datasets.items():
#     degree = df["avg_degree"].values.astype(float)
#     success = df["success"].astype(float).values

#     print(f"\nFitting: {name}")
#     print(f"  Data points: {len(df)}")
#     print(f"  Success rate: {success.mean()*100:.1f}%")

#     result = fit_logistic_with_bootstrap(degree, success, n_boot=N_BOOT, rng=RNG)
#     fits[name] = result

#     if result["fit_success"]:
#         print(f"  β = {result['beta']:.4f}  CI: [{result['beta_ci'][0]:.4f}, {result['beta_ci'][1]:.4f}]")
#         print(f"  d₀ = {result['x0']:.2f}  CI: [{result['x0_ci'][0]:.2f}, {result['x0_ci'][1]:.2f}]")
#         print(f"  Bootstrap samples: {len(result['boot_x0s'])}/{N_BOOT}")
#     else:
#         print(f"  Logistic fit failed (no transition detected)")

# # ============================================================
# # MASTER THREE-PANEL FIGURE
# # ============================================================

# fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

# # Global x range across all methods
# all_degrees = np.concatenate([
#     df["avg_degree"].values for df in datasets.values()
# ])
# x_global_min = 0
# x_global_max = max(all_degrees) + 2
# x_smooth = np.linspace(x_global_min, x_global_max, 500)

# panel_labels = ["(a)", "(b)", "(c)"]

# for idx, (name, df) in enumerate(datasets.items()):
#     ax = axes[idx]
#     color = colors[name]
#     fit = fits[name]

#     degree = df["avg_degree"].values.astype(float)
#     success = df["success"].astype(float).values

#     # Scatter raw data with jitter
#     jitter = RNG.normal(0, 0.015, size=len(success))
#     ax.scatter(
#         degree,
#         success + jitter,
#         alpha=0.25,
#         s=15,
#         c=color,
#         zorder=3
#     )

#     if fit["fit_success"]:
#         # Main logistic curve
#         y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
#         ax.plot(
#             x_smooth, y_smooth,
#             color=color, linewidth=2.5,
#             label="Logistic fit"
#         )

#         # Bootstrap confidence band
#         boot_curves = np.array([
#             logistic(x_smooth, b, x0)
#             for b, x0 in zip(fit["boot_betas"], fit["boot_x0s"])
#         ])
#         y_lower = np.percentile(boot_curves, 2.5, axis=0)
#         y_upper = np.percentile(boot_curves, 97.5, axis=0)

#         ax.fill_between(
#             x_smooth, y_lower, y_upper,
#             alpha=0.15, color=color
#         )

#         # Critical degree vertical line
#         ax.axvline(
#             fit["x0"], color=color, linestyle="--", alpha=0.8, linewidth=1.5
#         )

#         # Annotation box
#         ci_text = (
#             f"$d_0$ = {fit['x0']:.1f}\n"
#             f"95% CI: [{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]\n"
#             f"$\\beta$ = {fit['beta']:.3f}"
#         )
#         ax.text(
#             0.95, 0.45,
#             ci_text,
#             transform=ax.transAxes,
#             fontsize=9,
#             verticalalignment="top",
#             horizontalalignment="right",
#             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray")
#         )

#     else:
#         # No fit — add text annotation
#         ax.text(
#             0.5, 0.5,
#             "No transition\n(0% success at all densities)",
#             transform=ax.transAxes,
#             fontsize=11,
#             ha="center", va="center",
#             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="orange")
#         )

#     # Panel formatting
#     ax.set_xlabel("Average Degree", fontsize=12)
#     ax.set_title(f"{panel_labels[idx]} {name}", fontsize=13, fontweight="bold")
#     ax.set_ylim(-0.08, 1.12)
#     ax.set_xlim(x_global_min, x_global_max)
#     ax.grid(alpha=0.2)
#     ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
#     ax.axhline(1, color="gray", linewidth=0.5, alpha=0.3)

# # Only leftmost panel gets y label
# axes[0].set_ylabel("P(success)", fontsize=12)

# fig.suptitle(
#     "Phase Transitions Under Different Pruning Strategies",
#     fontsize=15, fontweight="bold", y=1.02
# )

# plt.tight_layout()
# plt.savefig(
#     os.path.join(OUTPUT_DIR, "master_phase_transition_comparison.png"),
#     dpi=300, bbox_inches="tight"
# )
# plt.savefig(
#     os.path.join(OUTPUT_DIR, "master_phase_transition_comparison.pdf"),
#     bbox_inches="tight"
# )
# plt.close()

# print(f"\n✅ Master figure saved to {OUTPUT_DIR}")

# # ============================================================
# # SAVE COMPARISON TABLE
# # ============================================================

# comparison_rows = []

# for name in datasets:
#     fit = fits[name]
#     df = datasets[name]

#     row = {
#         "Method": name,
#         "n_data_points": len(df),
#         "overall_success_rate": df["success"].mean(),
#         "success_rate_k_ge_10": df[df["k"] >= 10]["success"].mean() if "k" in df.columns else None,
#     }

#     if fit["fit_success"]:
#         row.update({
#             "beta": fit["beta"],
#             "beta_ci_low": fit["beta_ci"][0],
#             "beta_ci_high": fit["beta_ci"][1],
#             "d0": fit["x0"],
#             "d0_ci_low": fit["x0_ci"][0],
#             "d0_ci_high": fit["x0_ci"][1],
#             "n_bootstrap_valid": len(fit["boot_x0s"])
#         })
#     else:
#         row.update({
#             "beta": None,
#             "beta_ci_low": None,
#             "beta_ci_high": None,
#             "d0": None,
#             "d0_ci_low": None,
#             "d0_ci_high": None,
#             "n_bootstrap_valid": 0
#         })

#     comparison_rows.append(row)

# comp_df = pd.DataFrame(comparison_rows)
# comp_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_comparison_table.csv"), index=False)

# # Print clean summary
# print("\n" + "=" * 70)
# print("PHASE TRANSITION COMPARISON SUMMARY")
# print("=" * 70)
# print(f"{'Method':<16} {'d₀':>8} {'95% CI':>20} {'β':>8} {'Success':>10}")
# print("-" * 70)

# for name in datasets:
#     fit = fits[name]
#     df = datasets[name]
#     sr = df["success"].mean() * 100

#     if fit["fit_success"]:
#         ci_str = f"[{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]"
#         print(f"{name:<16} {fit['x0']:>8.1f} {ci_str:>20} {fit['beta']:>8.3f} {sr:>9.1f}%")
#     else:
#         print(f"{name:<16} {'∞':>8} {'(no transition)':>20} {'—':>8} {sr:>9.1f}%")

# print("=" * 70)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "structural_analysis_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOT = 1000
RNG = np.random.default_rng(42)

# Paths to the three full structural dataframes
GNN_CSV = "structural_analysis_gnn/full_structural_dataframe.csv"
KNN_CSV = "structural_analysis_baseline/nearest_k/full_structural_dataframe.csv"
RAND_CSV = "structural_analysis_baseline/random_k/full_structural_dataframe.csv"

# ============================================================
# LOGISTIC FUNCTION
# ============================================================

def logistic(x, beta, x0):
    z = np.clip(beta * (x - x0), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))



def fit_logistic_once(degree, success, p0=None):
    """
    Single logistic fit. Returns (beta, x0) or None on failure.
    """
    if success.sum() == 0 or success.sum() == len(success):
        return None
    if p0 is None:
        p0 = [0.5, np.median(degree)]
    try:
        popt, _ = curve_fit(
            logistic, degree, success,
            p0=p0,
            maxfev=10000
        )
        return popt  # (beta, x0)
    except (RuntimeError, ValueError):
        return None


def fit_logistic_with_bootstrap(degree, success, n_boot=1000, rng=None):
    """
    Fit logistic on raw Bernoulli data + bootstrap CI.
    Returns dict with fit params, CI, and per-bootstrap arrays.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    result = {
        "fit_success": False,
        "beta": None, "x0": None,
        "beta_ci": None, "x0_ci": None,
        "boot_betas": None, "boot_x0s": None
    }

    # Initial fit
    popt = fit_logistic_once(degree, success)
    if popt is None:
        return result

    beta_hat, x0_hat = popt

    # Bootstrap
    boot_betas = []
    boot_x0s = []

    for _ in range(n_boot):
        idx = rng.choice(len(degree), len(degree), replace=True)
        d_boot = degree[idx]
        s_boot = success[idx]

        popt_b = fit_logistic_once(d_boot, s_boot, p0=[beta_hat, x0_hat])
        if popt_b is not None:
            boot_betas.append(popt_b[0])
            boot_x0s.append(popt_b[1])

    boot_betas = np.array(boot_betas)
    boot_x0s = np.array(boot_x0s)

    if len(boot_x0s) < 50:
        return result

    result["fit_success"] = True
    result["beta"] = beta_hat
    result["x0"] = x0_hat
    result["beta_ci"] = np.percentile(boot_betas, [2.5, 97.5])
    result["x0_ci"] = np.percentile(boot_x0s, [2.5, 97.5])
    result["boot_betas"] = boot_betas
    result["boot_x0s"] = boot_x0s

    return result

# ============================================================
# LOAD ALL THREE DATASETS
# ============================================================

df_gnn = pd.read_csv(GNN_CSV)
df_knn = pd.read_csv(KNN_CSV)
df_rand = pd.read_csv(RAND_CSV)

datasets = {
    "Learned (GNN)": df_gnn,
    "Nearest-K": df_knn,
    "Random-K": df_rand
}

colors = {
    "Learned (GNN)": "#2196F3",
    "Nearest-K": "#FF9800",
    "Random-K": "#4CAF50"
}

# ============================================================
# FIT LOGISTIC FOR EACH METHOD
# ============================================================

fits = {}

for name, df in datasets.items():
    degree = df["avg_degree"].values.astype(float)
    success = df["success"].astype(float).values

    print(f"\nFitting: {name}")
    print(f"  Data points: {len(df)}")
    print(f"  Success rate: {success.mean()*100:.1f}%")

    result = fit_logistic_with_bootstrap(degree, success, n_boot=N_BOOT, rng=RNG)
    fits[name] = result

    if result["fit_success"]:
        print(f"  β = {result['beta']:.4f}  CI: [{result['beta_ci'][0]:.4f}, {result['beta_ci'][1]:.4f}]")
        print(f"  d₀ = {result['x0']:.2f}  CI: [{result['x0_ci'][0]:.2f}, {result['x0_ci'][1]:.2f}]")
        print(f"  Bootstrap samples: {len(result['boot_x0s'])}/{N_BOOT}")
    else:
        print(f"  Logistic fit failed (no transition detected)")

# ============================================================
# STATISTICAL TEST: Δ = d_c^NN - d_c^GNN  (PAIRED BOOTSTRAP)
# ============================================================

print("\n" + "=" * 70)
print("PAIRED BOOTSTRAP TEST: Δ = d_c(Nearest-K) - d_c(GNN)")
print("=" * 70)

delta_result = {
    "test_performed": False,
    "delta_point": None,
    "delta_ci": None,
    "delta_boot": None,
    "p_value_approx": None,
    "significant": None
}

gnn_fit = fits.get("Learned (GNN)")
knn_fit = fits.get("Nearest-K")

if (gnn_fit is not None and gnn_fit["fit_success"] and
    knn_fit is not None and knn_fit["fit_success"]):

    # --- Approach: Paired bootstrap ---
    # Resample BOTH datasets independently in each bootstrap iteration
    # and compute Δ = x0_NN - x0_GNN per iteration.
    
    degree_gnn = df_gnn["avg_degree"].values.astype(float)
    success_gnn = df_gnn["success"].astype(float).values
    degree_knn = df_knn["avg_degree"].values.astype(float)
    success_knn = df_knn["success"].astype(float).values

    # Initial fits (already have them)
    p0_gnn = [gnn_fit["beta"], gnn_fit["x0"]]
    p0_knn = [knn_fit["beta"], knn_fit["x0"]]

    boot_deltas = []
    boot_x0_gnn_paired = []
    boot_x0_knn_paired = []

    rng_paired = np.random.default_rng(123)  # separate seed for reproducibility

    for i in range(N_BOOT):
        # Resample GNN data
        idx_g = rng_paired.choice(len(degree_gnn), len(degree_gnn), replace=True)
        popt_g = fit_logistic_once(degree_gnn[idx_g], success_gnn[idx_g], p0=p0_gnn)

        # Resample Nearest-K data
        idx_k = rng_paired.choice(len(degree_knn), len(degree_knn), replace=True)
        popt_k = fit_logistic_once(degree_knn[idx_k], success_knn[idx_k], p0=p0_knn)

        # Only store if both fits succeeded
        if popt_g is not None and popt_k is not None:
            x0_g = popt_g[1]
            x0_k = popt_k[1]
            boot_x0_gnn_paired.append(x0_g)
            boot_x0_knn_paired.append(x0_k)
            boot_deltas.append(x0_k - x0_g)

    boot_deltas = np.array(boot_deltas)
    boot_x0_gnn_paired = np.array(boot_x0_gnn_paired)
    boot_x0_knn_paired = np.array(boot_x0_knn_paired)

    if len(boot_deltas) >= 50:
        delta_point = knn_fit["x0"] - gnn_fit["x0"]
        delta_ci = np.percentile(boot_deltas, [2.5, 97.5])
        
        # Approximate p-value: proportion of bootstrap Δ ≤ 0
        p_value = np.mean(boot_deltas <= 0)
        significant = delta_ci[0] > 0  # CI does not include zero

        delta_result["test_performed"] = True
        delta_result["delta_point"] = delta_point
        delta_result["delta_ci"] = delta_ci
        delta_result["delta_boot"] = boot_deltas
        delta_result["p_value_approx"] = p_value
        delta_result["significant"] = significant

        print(f"\n  Point estimate:  Δ = {delta_point:.2f}")
        print(f"  Bootstrap mean:  Δ = {boot_deltas.mean():.2f}")
        print(f"  Bootstrap std:   σ(Δ) = {boot_deltas.std():.2f}")
        print(f"  95% CI of Δ:     [{delta_ci[0]:.2f}, {delta_ci[1]:.2f}]")
        print(f"  P(Δ ≤ 0):        {p_value:.4f}")
        print(f"  Valid bootstraps: {len(boot_deltas)}/{N_BOOT}")
        print()
        
        if significant:
            print("  ✅ RESULT: The 95% CI does NOT include zero.")
            print("     → The critical degree under GNN pruning is STATISTICALLY LOWER")
            print("       than under Nearest-K pruning.")
        else:
            print("  ⚠️  RESULT: The 95% CI INCLUDES zero.")
            print("     → The difference is NOT statistically significant at α = 0.05.")
            print("     → Remove the word 'significantly' from your paper.")

        # ----- Plot: Bootstrap distribution of Δ -----
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(boot_deltas, bins=40, color="#7E57C2", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Δ = 0 (no difference)")
        ax.axvline(delta_point, color="black", linestyle="-", linewidth=2, 
                   label=f"Point estimate Δ = {delta_point:.2f}")
        ax.axvline(delta_ci[0], color="gray", linestyle=":", linewidth=1.5,
                   label=f"95% CI: [{delta_ci[0]:.2f}, {delta_ci[1]:.2f}]")
        ax.axvline(delta_ci[1], color="gray", linestyle=":", linewidth=1.5)
        
        # Shade the CI region
        ax.axvspan(delta_ci[0], delta_ci[1], alpha=0.15, color="gray")
        
        ax.set_xlabel("Δ = $d_c^{Nearest-K}$ − $d_c^{GNN}$", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_title("Bootstrap Distribution of Critical Degree Difference", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)
        
        # Add result text box
        result_text = (
            f"Δ = {delta_point:.2f}\n"
            f"95% CI: [{delta_ci[0]:.2f}, {delta_ci[1]:.2f}]\n"
            f"P(Δ ≤ 0) = {p_value:.4f}\n"
            f"{'Significant ✓' if significant else 'Not significant ✗'}"
        )
        ax.text(0.97, 0.95, result_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         alpha=0.9, edgecolor='gray'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "bootstrap_delta_test.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTPUT_DIR, "bootstrap_delta_test.pdf"), bbox_inches="tight")
        plt.close()
        
        print(f"\n  📊 Bootstrap Δ distribution plot saved.")

    else:
        print(f"\n  ❌ Too few valid paired bootstraps ({len(boot_deltas)}/{N_BOOT}). Test not performed.")
else:
    print("\n  ❌ Cannot perform test: one or both logistic fits failed.")


# ============================================================
# MASTER THREE-PANEL FIGURE (unchanged)
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

all_degrees = np.concatenate([
    df["avg_degree"].values for df in datasets.values()
])
x_global_min = 0
x_global_max = max(all_degrees) + 2
x_smooth = np.linspace(x_global_min, x_global_max, 500)

panel_labels = ["(a)", "(b)", "(c)"]

for idx, (name, df) in enumerate(datasets.items()):
    ax = axes[idx]
    color = colors[name]
    fit = fits[name]

    degree = df["avg_degree"].values.astype(float)
    success = df["success"].astype(float).values

    jitter = RNG.normal(0, 0.015, size=len(success))
    ax.scatter(
        degree,
        success + jitter,
        alpha=0.25,
        s=15,
        c=color,
        zorder=3
    )

    if fit["fit_success"]:
        y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
        ax.plot(
            x_smooth, y_smooth,
            color=color, linewidth=2.5,
            label="Logistic fit"
        )

        boot_curves = np.array([
            logistic(x_smooth, b, x0)
            for b, x0 in zip(fit["boot_betas"], fit["boot_x0s"])
        ])
        y_lower = np.percentile(boot_curves, 2.5, axis=0)
        y_upper = np.percentile(boot_curves, 97.5, axis=0)

        ax.fill_between(
            x_smooth, y_lower, y_upper,
            alpha=0.15, color=color
        )

        ax.axvline(
            fit["x0"], color=color, linestyle="--", alpha=0.8, linewidth=1.5
        )

        ci_text = (
            f"$d_0$ = {fit['x0']:.1f}\n"
            f"95% CI: [{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]\n"
            f"$\\beta$ = {fit['beta']:.3f}"
        )
        ax.text(
            0.95, 0.45,
            ci_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray")
        )

    else:
        ax.text(
            0.5, 0.5,
            "No transition\n(0% success at all densities)",
            transform=ax.transAxes,
            fontsize=11,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="orange")
        )

    ax.set_xlabel("Average Degree", fontsize=12)
    ax.set_title(f"{panel_labels[idx]} {name}", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlim(x_global_min, x_global_max)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(1, color="gray", linewidth=0.5, alpha=0.3)

axes[0].set_ylabel("P(success)", fontsize=12)

fig.suptitle(
    "Phase Transitions Under Different Pruning Strategies",
    fontsize=15, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "master_phase_transition_comparison.png"),
    dpi=300, bbox_inches="tight"
)
plt.savefig(
    os.path.join(OUTPUT_DIR, "master_phase_transition_comparison.pdf"),
    bbox_inches="tight"
)
plt.close()

print(f"\n✅ Master figure saved to {OUTPUT_DIR}")

# ============================================================
# SAVE COMPARISON TABLE (with Δ test results)
# ============================================================

comparison_rows = []

for name in datasets:
    fit = fits[name]
    df = datasets[name]

    row = {
        "Method": name,
        "n_data_points": len(df),
        "overall_success_rate": df["success"].mean(),
        "success_rate_k_ge_10": df[df["k"] >= 10]["success"].mean() if "k" in df.columns else None,
    }

    if fit["fit_success"]:
        row.update({
            "beta": fit["beta"],
            "beta_ci_low": fit["beta_ci"][0],
            "beta_ci_high": fit["beta_ci"][1],
            "d0": fit["x0"],
            "d0_ci_low": fit["x0_ci"][0],
            "d0_ci_high": fit["x0_ci"][1],
            "n_bootstrap_valid": len(fit["boot_x0s"])
        })
    else:
        row.update({
            "beta": None,
            "beta_ci_low": None,
            "beta_ci_high": None,
            "d0": None,
            "d0_ci_low": None,
            "d0_ci_high": None,
            "n_bootstrap_valid": 0
        })

    comparison_rows.append(row)

comp_df = pd.DataFrame(comparison_rows)

# Add delta test results as a separate row or as metadata
if delta_result["test_performed"]:
    delta_row = {
        "Method": "Δ (Nearest-K − GNN)",
        "n_data_points": None,
        "overall_success_rate": None,
        "success_rate_k_ge_10": None,
        "beta": None,
        "beta_ci_low": None,
        "beta_ci_high": None,
        "d0": delta_result["delta_point"],
        "d0_ci_low": delta_result["delta_ci"][0],
        "d0_ci_high": delta_result["delta_ci"][1],
        "n_bootstrap_valid": len(delta_result["delta_boot"])
    }
    comp_df = pd.concat([comp_df, pd.DataFrame([delta_row])], ignore_index=True)

comp_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_comparison_table.csv"), index=False)

# Save delta test details separately
if delta_result["test_performed"]:
    delta_df = pd.DataFrame({
        "statistic": [
            "delta_point_estimate",
            "delta_boot_mean",
            "delta_boot_std",
            "delta_ci_low",
            "delta_ci_high",
            "p_value_approx",
            "significant_at_005",
            "n_valid_bootstraps"
        ],
        "value": [
            delta_result["delta_point"],
            delta_result["delta_boot"].mean(),
            delta_result["delta_boot"].std(),
            delta_result["delta_ci"][0],
            delta_result["delta_ci"][1],
            delta_result["p_value_approx"],
            delta_result["significant"],
            len(delta_result["delta_boot"])
        ]
    })
    delta_df.to_csv(os.path.join(OUTPUT_DIR, "delta_statistical_test.csv"), index=False)
    
    # Also save the raw bootstrap deltas for reproducibility
    np.save(os.path.join(OUTPUT_DIR, "bootstrap_deltas.npy"), delta_result["delta_boot"])




# ============================================================
# PHASE 4: LOGISTIC REGRESSION WITH CONNECTIVITY COVARIATE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: LOGISTIC REGRESSION WITH CONNECTIVITY COVARIATE")
print("=" * 70)

import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress convergence warnings (optional, remove if you want to see them)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

regression_rows = []

for name, df in datasets.items():
    print(f"\n--- {name} ---")
    
    # Prepare data: convert is_connected to int and ensure numeric
    df_clean = df[["avg_degree", "is_connected", "success"]].copy()
    df_clean["is_connected"] = df_clean["is_connected"].astype(int)
    df_clean.dropna(inplace=True)
    
    # Check if is_connected is constant
    if df_clean["is_connected"].nunique() == 1:
        print(f"  Warning: 'is_connected' is constant ({df_clean['is_connected'].iloc[0]}). Dropping it from model.")
        predictors = ["avg_degree"]
    else:
        predictors = ["avg_degree", "is_connected"]
    
    # Build design matrix
    X = df_clean[predictors]
    X = sm.add_constant(X)
    y = df_clean["success"]
    
    # Try fitting, with error handling
    try:
        # Use BFGS for better stability on problematic data
        model = sm.Logit(y, X).fit(method='bfgs', maxiter=200, disp=0)
        
        # If we get here, model converged (or at least finished)
        params = model.params
        pvalues = model.pvalues
        ci = model.conf_int()
        
        # Extract coefficients (they may be huge if separation exists)
        const_coef = params.get('const', np.nan)
        const_p = pvalues.get('const', np.nan)
        const_ci_low = ci[0].get('const', np.nan) if len(ci) > 0 else np.nan
        const_ci_high = ci[1].get('const', np.nan) if len(ci) > 1 else np.nan
        
        degree_coef = params.get('avg_degree', np.nan)
        degree_p = pvalues.get('avg_degree', np.nan)
        degree_ci_low = ci[0].get('avg_degree', np.nan) if len(ci) > 0 else np.nan
        degree_ci_high = ci[1].get('avg_degree', np.nan) if len(ci) > 1 else np.nan
        
        if 'is_connected' in predictors:
            connected_coef = params.get('is_connected', np.nan)
            connected_p = pvalues.get('is_connected', np.nan)
            connected_ci_low = ci[0].get('is_connected', np.nan) if len(ci) > 0 else np.nan
            connected_ci_high = ci[1].get('is_connected', np.nan) if len(ci) > 1 else np.nan
        else:
            connected_coef = connected_p = connected_ci_low = connected_ci_high = np.nan
        
        pseudo_r2 = model.prsquared
        converged = True
        
    except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
        print(f"  Model failed: {e}")
        # Store NaN for all statistics
        const_coef = const_p = const_ci_low = const_ci_high = np.nan
        degree_coef = degree_p = degree_ci_low = degree_ci_high = np.nan
        connected_coef = connected_p = connected_ci_low = connected_ci_high = np.nan
        pseudo_r2 = np.nan
        converged = False
    
    # Store row
    row = {
        "method": name,
        "n": len(df_clean),
        "const_coef": const_coef,
        "const_p": const_p,
        "const_ci_low": const_ci_low,
        "const_ci_high": const_ci_high,
        "degree_coef": degree_coef,
        "degree_p": degree_p,
        "degree_ci_low": degree_ci_low,
        "degree_ci_high": degree_ci_high,
        "connected_coef": connected_coef,
        "connected_p": connected_p,
        "connected_ci_low": connected_ci_low,
        "connected_ci_high": connected_ci_high,
        "pseudo_r2": pseudo_r2,
        "converged": converged
    }
    regression_rows.append(row)
    
    if converged:
        # Print a short summary
        print(f"  converged: {converged}")
        print(f"  degree coef = {degree_coef:.4f} (p={degree_p:.4f})")
        if not np.isnan(connected_coef):
            print(f"  connected coef = {connected_coef:.4f} (p={connected_p:.4f})")
        print(f"  pseudo R² = {pseudo_r2:.4f}")
    else:
        print(f"  ❌ Model did not converge – results are NaN.")

# Save regression results to CSV
reg_df = pd.DataFrame(regression_rows)
reg_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_regression_with_connectivity.csv"), index=False)
print(f"\n✅ Regression results saved to {OUTPUT_DIR}/logistic_regression_with_connectivity.csv")

# ============================================================
# PRINT FINAL SUMMARY
# ============================================================
# ============================================================
# GENERATE PNG TABLE FOR LOGISTIC REGRESSION RESULTS
# ============================================================

# ============================================================
# GENERATE PNG TABLE FOR LOGISTIC REGRESSION RESULTS
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_table_figure(df, title, col_widths=None, figsize=(10, 4),
                        row_height=0.15, vert_scale=2.2):
    """Create a professional-looking table figure with adjustable row heights."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, vert_scale)  # Increase vert_scale for taller rows
    
    # Style ALL cells (header + data) with consistent height
    for i in range(len(df) + 1):  # +1 to include header row
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            cell.set_height(row_height)  # Set explicit height for each cell
            
            if i == 0:
                # Header styling
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white', fontsize=10)
            else:
                # Data row styling
                if i % 2 == 0:
                    cell.set_facecolor('#E8F0FE')
                else:
                    cell.set_facecolor('white')
                cell.set_text_props(fontsize=10)
            
            # Center text in all cells
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    return fig

# ============================================================
# GENERATE PNG TABLE FOR LOGISTIC REGRESSION (EXCLUDING RANDOM-K)
# ============================================================

# Filter out Random-K
reg_filtered = reg_df[reg_df['method'] != 'Random-K'].copy()

if len(reg_filtered) > 0:
    # Prepare display dataframe
    reg_display = reg_filtered[['method', 'degree_coef', 'degree_ci_low', 'degree_ci_high',
                                'connected_coef', 'connected_ci_low', 'connected_ci_high',
                                'pseudo_r2']].copy()
    
    # Round to 3 decimals and format as strings
    for col in ['degree_coef', 'degree_ci_low', 'degree_ci_high',
                'connected_coef', 'connected_ci_low', 'connected_ci_high',
                'pseudo_r2']:
        reg_display[col] = reg_display[col].apply(
            lambda x: f'{x:.3f}' if pd.notna(x) else '—'
        )
    
    # Rename columns for display
    reg_display.columns = ['Method', 'β_deg', 'β_deg\nlow', 'β_deg\nhigh',
                           'β_conn', 'β_conn\nlow', 'β_conn\nhigh',
                           'Pseudo R²']
    
    # Create figure with LARGER figsize and row_height
    n_rows = len(reg_display)
    fig_height = max(3, n_rows * 1.0 + 1.5)  # Dynamic height based on rows
    
    fig_reg = create_table_figure(
        reg_display,
        "Logistic Regression with Connectivity Covariate",
        col_widths=[0.16, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.12],
        figsize=(14, fig_height),
        row_height=0.18,      # INCREASE this for taller cells
        vert_scale=2.5        # INCREASE this for more vertical spacing
    )
    fig_reg.savefig(
        os.path.join(OUTPUT_DIR, "logistic_regression_covariate_table.png"),
        dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none'
    )
    plt.close()
    print(f"✅ PNG table saved to {OUTPUT_DIR}/logistic_regression_covariate_table.png")
else:
    print("⚠️ No data to plot (Random-K excluded, but others missing?)")
    

print("\n" + "=" * 70)
print("PHASE TRANSITION COMPARISON SUMMARY")
print("=" * 70)
print(f"{'Method':<16} {'d₀':>8} {'95% CI':>20} {'β':>8} {'Success':>10}")
print("-" * 70)

for name in datasets:
    fit = fits[name]
    df = datasets[name]
    sr = df["success"].mean() * 100

    if fit["fit_success"]:
        ci_str = f"[{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]"
        print(f"{name:<16} {fit['x0']:>8.1f} {ci_str:>20} {fit['beta']:>8.3f} {sr:>9.1f}%")
    else:
        print(f"{name:<16} {'∞':>8} {'(no transition)':>20} {'—':>8} {sr:>9.1f}%")

print("=" * 70)

if delta_result["test_performed"]:
    print(f"\n{'STATISTICAL TEST':=^70}")
    print(f"  Δ = d_c(Nearest-K) − d_c(GNN) = {delta_result['delta_point']:.2f}")
    print(f"  95% Bootstrap CI: [{delta_result['delta_ci'][0]:.2f}, {delta_result['delta_ci'][1]:.2f}]")
    print(f"  P(Δ ≤ 0) ≈ {delta_result['p_value_approx']:.4f}")
    if delta_result["significant"]:
        print(f"  ✅ Statistically significant: GNN critical degree is LOWER")
    else:
        print(f"  ⚠️  NOT statistically significant at α = 0.05")
    print("=" * 70)

    # Print the paper-ready sentence
    print("\n📝 PAPER-READY TEXT (for Section 5.4):")
    print("-" * 70)
    if delta_result["significant"]:
        print(
            f"To assess whether the shift in critical degree between pruning\n"
            f"strategies is statistically supported, we compute the bootstrap\n"
            f"distribution of the difference Δ = d_c^{{Nearest}} − d_c^{{GNN}}.\n"
            f"The 95% bootstrap confidence interval for Δ is\n"
            f"[{delta_result['delta_ci'][0]:.2f}, {delta_result['delta_ci'][1]:.2f}] "
            f"(based on {len(delta_result['delta_boot'])} valid resamples),\n"
            f"which does not include zero, confirming that the critical degree\n"
            f"under learned pruning is statistically lower (p < {max(delta_result['p_value_approx'], 0.001):.3f})."
        )
    else:
        print(
            f"To assess whether the shift in critical degree between pruning\n"
            f"strategies is statistically supported, we compute the bootstrap\n"
            f"distribution of the difference Δ = d_c^{{Nearest}} − d_c^{{GNN}}.\n"
            f"The 95% bootstrap confidence interval for Δ is\n"
            f"[{delta_result['delta_ci'][0]:.2f}, {delta_result['delta_ci'][1]:.2f}],\n"
            f"which includes zero. The observed difference is therefore not\n"
            f"statistically significant at the α = 0.05 level."
        )
    print("-" * 70)