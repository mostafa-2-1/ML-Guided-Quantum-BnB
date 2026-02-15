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

def fit_logistic_with_bootstrap(degree, success, n_boot=1000, rng=None):
    """
    Fit logistic on raw Bernoulli data + bootstrap CI.
    Returns dict with fit params, CI, and curves for plotting.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    result = {
        "fit_success": False,
        "beta": None, "x0": None,
        "beta_ci": None, "x0_ci": None,
        "boot_betas": None, "boot_x0s": None
    }

    # Check if any success exists
    if success.sum() == 0 or success.sum() == len(success):
        return result

    # Initial fit
    try:
        popt, _ = curve_fit(
            logistic, degree, success,
            p0=[0.5, np.median(degree)],
            maxfev=10000
        )
    except (RuntimeError, ValueError):
        return result

    beta_hat, x0_hat = popt

    # Bootstrap
    boot_betas = []
    boot_x0s = []

    for _ in range(n_boot):
        idx = rng.choice(len(degree), len(degree), replace=True)
        d_boot = degree[idx]
        s_boot = success[idx]

        if s_boot.sum() == 0 or s_boot.sum() == len(s_boot):
            continue

        try:
            popt_b, _ = curve_fit(
                logistic, d_boot, s_boot,
                p0=[beta_hat, x0_hat],
                maxfev=5000
            )
            boot_betas.append(popt_b[0])
            boot_x0s.append(popt_b[1])
        except (RuntimeError, ValueError):
            continue

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
# MASTER THREE-PANEL FIGURE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

# Global x range across all methods
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

    # Scatter raw data with jitter
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
        # Main logistic curve
        y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
        ax.plot(
            x_smooth, y_smooth,
            color=color, linewidth=2.5,
            label="Logistic fit"
        )

        # Bootstrap confidence band
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

        # Critical degree vertical line
        ax.axvline(
            fit["x0"], color=color, linestyle="--", alpha=0.8, linewidth=1.5
        )

        # Annotation box
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
        # No fit — add text annotation
        ax.text(
            0.5, 0.5,
            "No transition\n(0% success at all densities)",
            transform=ax.transAxes,
            fontsize=11,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor="orange")
        )

    # Panel formatting
    ax.set_xlabel("Average Degree", fontsize=12)
    ax.set_title(f"{panel_labels[idx]} {name}", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlim(x_global_min, x_global_max)
    ax.grid(alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(1, color="gray", linewidth=0.5, alpha=0.3)

# Only leftmost panel gets y label
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
# SAVE COMPARISON TABLE
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
comp_df.to_csv(os.path.join(OUTPUT_DIR, "logistic_comparison_table.csv"), index=False)

# Print clean summary
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