import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "structural_analysis_comparison_connectivity"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BOOT = 1000
RNG = np.random.default_rng(42)

# Paths to the three full structural dataframes
GNN_CSV = "structural_analysis_gnn/full_structural_dataframe.csv"
KNN_CSV = "structural_analysis_baseline/nearest_k/full_structural_dataframe.csv"
RAND_CSV = "structural_analysis_baseline/random_k/full_structural_dataframe.csv"



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_table_figure(df, title, col_widths=None, figsize=(10, 4)):
    """Create a professional-looking table figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.1)
    
    # Row styling
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F0FE')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_height(0.08)
            # Enable text wrapping
            table[(i, j)].get_text().set_wrap(True)
            table[(i, j)].get_text().set_horizontalalignment('center')
            table[(i, j)].get_text().set_verticalalignment('center')
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    return fig

# ============================================================
# LOGISTIC FUNCTION
# ============================================================

def logistic(x, beta, x0):
    z = np.clip(beta * (x - x0), -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_once(degree, success, p0=None):
    """Single logistic fit. Returns (beta, x0) or None on failure."""
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
        return popt
    except (RuntimeError, ValueError):
        return None


def fit_logistic_with_bootstrap(degree, success, n_boot=1000, rng=None):
    """Fit logistic on raw Bernoulli data + bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(42)

    result = {
        "fit_success": False,
        "beta": None, "x0": None,
        "beta_ci": None, "x0_ci": None,
        "boot_betas": None, "boot_x0s": None
    }

    popt = fit_logistic_once(degree, success)
    if popt is None:
        return result

    beta_hat, x0_hat = popt

    boot_betas, boot_x0s = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(degree), len(degree), replace=True)
        popt_b = fit_logistic_once(degree[idx], success[idx], p0=[beta_hat, x0_hat])
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

# Colors for the three curves
curve_colors = {
    "P(success)": "#2196F3",           # Blue
    "P(connected)": "#333333",          # Black
    "P(success|connected)": "#E53935"   # Red
}

# ============================================================
# FIT ALL THREE CURVES FOR EACH METHOD
# ============================================================

all_fits = {}

for name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    degree = df["avg_degree"].values.astype(float)
    success = df["success"].astype(float).values
    connected = df["is_connected"].astype(float).values
    
    print(f"  Total data points: {len(df)}")
    print(f"  Overall success rate: {success.mean()*100:.1f}%")
    print(f"  Overall connectivity rate: {connected.mean()*100:.1f}%")
    # Check for successes on disconnected graphs
    disconnected_success = ((df["is_connected"] == False) & (df["success"] > 0)).any()
    if disconnected_success:
        count = ((df["is_connected"] == False) & (df["success"] > 0)).sum()
        print(f"  ⚠️  {count} disconnected graphs have success > 0 (i.e., at least one seed succeeded).")
    else:
        print(f"  ✅ No successes on disconnected graphs.")
    # 1. P(success) - overall
    print(f"\n  Fitting P(success)...")
    fit_success = fit_logistic_with_bootstrap(degree, success, n_boot=N_BOOT, rng=RNG)
    if fit_success["fit_success"]:
        print(f"    d₀ = {fit_success['x0']:.2f}  CI: [{fit_success['x0_ci'][0]:.2f}, {fit_success['x0_ci'][1]:.2f}]")
    else:
        print(f"    Fit failed")
    
    # 2. P(connected)
    print(f"\n  Fitting P(connected)...")
    fit_connected = fit_logistic_with_bootstrap(degree, connected, n_boot=N_BOOT, rng=RNG)
    if fit_connected["fit_success"]:
        print(f"    d₀ = {fit_connected['x0']:.2f}  CI: [{fit_connected['x0_ci'][0]:.2f}, {fit_connected['x0_ci'][1]:.2f}]")
    else:
        conn_rate = connected.mean()
        print(f"    Fit failed — connectivity rate = {conn_rate*100:.1f}%")
    
    # 3. P(success | connected)
    df_conn = df[df["is_connected"] == True].copy()
    print(f"\n  Fitting P(success | connected)...")
    print(f"    Connected subset: {len(df_conn)} / {len(df)} ({100*len(df_conn)/len(df):.1f}%)")
    
    if len(df_conn) > 50:
        degree_conn = df_conn["avg_degree"].values.astype(float)
        success_conn = df_conn["success"].astype(float).values
        print(f"    Success rate in connected: {success_conn.mean()*100:.1f}%")
        
        fit_cond = fit_logistic_with_bootstrap(degree_conn, success_conn, n_boot=N_BOOT, rng=RNG)
        if fit_cond["fit_success"]:
            print(f"    d₀ = {fit_cond['x0']:.2f}  CI: [{fit_cond['x0_ci'][0]:.2f}, {fit_cond['x0_ci'][1]:.2f}]")
        else:
            print(f"    Fit failed")
    else:
        fit_cond = {"fit_success": False}
        print(f"    Not enough connected samples for fitting")
    
    all_fits[name] = {
        "P(success)": fit_success,
        "P(connected)": fit_connected,
        "P(success|connected)": fit_cond,
        "conn_rate": connected.mean()
    }

# ============================================================
# MASTER THREE-PANEL FIGURE: CONNECTIVITY DECOMPOSITION
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

# Global x range
all_degrees = np.concatenate([df["avg_degree"].values for df in datasets.values()])
x_global_min = 0
x_global_max = max(all_degrees) + 2
x_smooth = np.linspace(x_global_min, x_global_max, 500)

panel_labels = ["(a)", "(b)", "(c)"]

for idx, (name, df) in enumerate(datasets.items()):
    ax = axes[idx]
    fits = all_fits[name]
    
    # === P(success) : Blue solid ===
    fit = fits["P(success)"]
    if fit["fit_success"]:
        y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
        ax.plot(x_smooth, y_smooth, color=curve_colors["P(success)"], 
                linewidth=2.5, label=f"P(success), $d_0$={fit['x0']:.1f}")
        
        # Bootstrap CI band
        boot_curves = np.array([
            logistic(x_smooth, b, x0) for b, x0 in zip(fit["boot_betas"], fit["boot_x0s"])
        ])
        y_lo = np.percentile(boot_curves, 2.5, axis=0)
        y_hi = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(x_smooth, y_lo, y_hi, alpha=0.15, color=curve_colors["P(success)"])
        
        # Vertical line at d0
        ax.axvline(fit["x0"], color=curve_colors["P(success)"], linestyle=":", alpha=0.6, linewidth=1.5)
    
    # === P(connected) : Black dashed ===
    fit = fits["P(connected)"]
    if fit["fit_success"]:
        y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
        ax.plot(x_smooth, y_smooth, color=curve_colors["P(connected)"], 
                linewidth=2.5, linestyle="--", label=f"P(connected), $d_0$={fit['x0']:.1f}")
        
        boot_curves = np.array([
            logistic(x_smooth, b, x0) for b, x0 in zip(fit["boot_betas"], fit["boot_x0s"])
        ])
        y_lo = np.percentile(boot_curves, 2.5, axis=0)
        y_hi = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(x_smooth, y_lo, y_hi, alpha=0.1, color=curve_colors["P(connected)"])
        
        ax.axvline(fit["x0"], color=curve_colors["P(connected)"], linestyle=":", alpha=0.4, linewidth=1.5)
    else:
        # If always/nearly connected, draw horizontal line
        conn_rate = fits["conn_rate"]
        if conn_rate > 0.9:
            ax.axhline(conn_rate, color=curve_colors["P(connected)"], linestyle="--", 
                       linewidth=2, alpha=0.7, label=f"P(connected) ≈ {conn_rate:.2f}")
    
    # === P(success | connected) : Red solid ===
    fit = fits["P(success|connected)"]
    if fit["fit_success"]:
        y_smooth = logistic(x_smooth, fit["beta"], fit["x0"])
        ax.plot(x_smooth, y_smooth, color=curve_colors["P(success|connected)"], 
                linewidth=2.5, label=f"P(success|conn), $d_0$={fit['x0']:.1f}")
        
        boot_curves = np.array([
            logistic(x_smooth, b, x0) for b, x0 in zip(fit["boot_betas"], fit["boot_x0s"])
        ])
        y_lo = np.percentile(boot_curves, 2.5, axis=0)
        y_hi = np.percentile(boot_curves, 97.5, axis=0)
        ax.fill_between(x_smooth, y_lo, y_hi, alpha=0.15, color=curve_colors["P(success|connected)"])
        
        ax.axvline(fit["x0"], color=curve_colors["P(success|connected)"], linestyle=":", alpha=0.6, linewidth=1.5)
    
    # Panel formatting
    ax.set_xlabel("Average Degree", fontsize=12)
    ax.set_title(f"{panel_labels[idx]} {name}", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(x_global_min, x_global_max)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(1, color="gray", linewidth=0.5, alpha=0.3)

axes[0].set_ylabel("Probability", fontsize=12)

fig.suptitle(
    "Connectivity Decomposition: P(success) vs P(connected) vs P(success | connected)",
    fontsize=15, fontweight="bold", y=1.02
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "connectivity_decomposition.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "connectivity_decomposition.pdf"), bbox_inches="tight")
plt.close()

print(f"\n✅ Connectivity decomposition figure saved to {OUTPUT_DIR}")

# ============================================================
# SUMMARY TABLE 
# ============================================================

print("\n" + "=" * 105)
print("CONNECTIVITY DECOMPOSITION SUMMARY")
print("=" * 105)
print(f"{'Method':<18} {'Curve':<24} {'d₀':>8} {'d₀ 95% CI':>22} {'β':>10} {'β 95% CI':>22}")
print("-" * 105)

for name in datasets:
    fits = all_fits[name]
    for curve_name in ["P(success)", "P(connected)", "P(success|connected)"]:
        fit = fits[curve_name]
        if fit["fit_success"]:
            d0_ci = f"[{fit['x0_ci'][0]:.1f}, {fit['x0_ci'][1]:.1f}]"
            beta_ci = f"[{fit['beta_ci'][0]:.3f}, {fit['beta_ci'][1]:.3f}]"
            print(f"{name:<18} {curve_name:<24} {fit['x0']:>8.1f} {d0_ci:>22} {fit['beta']:>10.3f} {beta_ci:>22}")
        else:
            print(f"{name:<18} {curve_name:<24} {'—':>8} {'(no transition)':>22} {'—':>10} {'—':>22}")
    print("-" * 105)

print("=" * 105)

# Save summary to CSV
summary_rows = []
for name in datasets:
    fits = all_fits[name]
    df = datasets[name]
    
    for curve_name in ["P(success)", "P(connected)", "P(success|connected)"]:
        fit = fits[curve_name]
        row = {
            "method": name,
            "curve": curve_name,
            "d0": fit["x0"] if fit["fit_success"] else None,
            "d0_ci_low": fit["x0_ci"][0] if fit["fit_success"] else None,
            "d0_ci_high": fit["x0_ci"][1] if fit["fit_success"] else None,
            "beta": fit["beta"] if fit["fit_success"] else None,
            "beta_ci_low": fit["beta_ci"][0] if fit["fit_success"] else None,
            "beta_ci_high": fit["beta_ci"][1] if fit["fit_success"] else None,
            "n_bootstrap": len(fit["boot_x0s"]) if fit["fit_success"] else 0,
            "fit_success": fit["fit_success"]
        }
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "connectivity_decomposition_summary.csv"), index=False)

print(f"\n✅ Summary table saved to {OUTPUT_DIR}/connectivity_decomposition_summary.csv")

# ============================================================
# GENERATE PNG TABLE OF CONNECTIVITY DECOMPOSITION SUMMARY
# ============================================================

# Prepare display dataframe: round values, format CI columns
display_df = summary_df[summary_df["fit_success"]].copy()  # only successful fits
if len(display_df) > 0:
    # Round numeric columns
    for col in ["d0", "d0_ci_low", "d0_ci_high", "beta", "beta_ci_low", "beta_ci_high"]:
        display_df[col] = display_df[col].round(2)
    
    # Create combined CI strings for nicer display (optional)
    display_df["d0 CI"] = display_df.apply(
        lambda r: f"[{r['d0_ci_low']:.2f}, {r['d0_ci_high']:.2f}]", axis=1
    )
    display_df["β CI"] = display_df.apply(
        lambda r: f"[{r['beta_ci_low']:.2f}, {r['beta_ci_high']:.2f}]", axis=1
    )
    
    # Select and rename columns for final table
    final_table = display_df[["method", "curve", "d0", "d0 CI", "beta", "β CI"]].copy()
    final_table.columns = ["Method", "Curve", "d₀", "d₀ 95% CI", "β", "β 95% CI"]
    
    # Create figure
    fig = create_table_figure(
        final_table,
        "Connectivity Decomposition Summary",
        col_widths=[0.18, 0.22, 0.10, 0.16, 0.10, 0.16],
        figsize=(14, len(final_table) * 0.6 + 1)
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, "connectivity_decomposition_table.png"),
        dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none'
    )
    plt.close()
    print(f"✅ PNG table saved to {OUTPUT_DIR}/connectivity_decomposition_table.png")
else:
    print("⚠️ No successful fits to display in PNG table.")

# ============================================================
# KEY INSIGHT PRINTOUT
# ============================================================

print("\n" + "=" * 85)
print("KEY INSIGHTS")
print("=" * 85)

for name in datasets:
    fits = all_fits[name]
    print(f"\n{name}:")
    
    d0_success = fits["P(success)"]["x0"] if fits["P(success)"]["fit_success"] else None
    d0_conn = fits["P(connected)"]["x0"] if fits["P(connected)"]["fit_success"] else None
    d0_cond = fits["P(success|connected)"]["x0"] if fits["P(success|connected)"]["fit_success"] else None
    
    if d0_success and d0_cond:
        shift = d0_success - d0_cond
        print(f"  d₀(success) = {d0_success:.1f}")
        print(f"  d₀(success|connected) = {d0_cond:.1f}")
        print(f"  Shift due to connectivity: {shift:+.1f}")
        
        if d0_conn:
            print(f"  d₀(connected) = {d0_conn:.1f}")
            if d0_conn > d0_cond:
                print(f"  → Connectivity is the DOMINANT bottleneck (d0_conn > d0_cond)")
            else:
                print(f"  → Solver difficulty dominates (d0_cond > d0_conn)")
    elif d0_success:
        print(f"  d₀(success) = {d0_success:.1f}")
        print(f"  P(success|connected) fit failed — likely all connected graphs succeed/fail uniformly")
    else:
        print(f"  No successful logistic fits")

print("\n" + "=" * 85)