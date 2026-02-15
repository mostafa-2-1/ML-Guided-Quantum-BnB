import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

GNN_PATH = "structural_analysis_gnn"
KNN_PATH = "structural_analysis_baseline/nearest_k"
RAND_PATH = "structural_analysis_baseline/random_k"

OUTPUT_DIR = "table_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD LOGISTIC PARAMETERS
# ============================================================

def load_logistic(path, method_name):
    file = os.path.join(path, "logistic_parameters.csv")
    if not os.path.exists(file):
        return {
            "Method": method_name,
            "beta": 0,
            "beta_low": "-",
            "beta_high": "-",
            "x0": "-",
            "x0_low": "-",
            "x0_high": "-"
        }

    df = pd.read_csv(file)
    row = df.iloc[0]

    return {
        "Method": method_name,
        "beta": round(row["beta_hat"], 4),
        "beta_low": round(row["beta_ci_low"], 4),
        "beta_high": round(row["beta_ci_high"], 4),
        "x0": round(row["x0_hat"], 4),
        "x0_low": round(row["x0_ci_low"], 4),
        "x0_high": round(row["x0_ci_high"], 4),
    }

logistic_rows = [
    load_logistic(GNN_PATH, "GNN"),
    load_logistic(KNN_PATH, "Nearest-k"),
    load_logistic(RAND_PATH, "Random-k")
]

df_logistic = pd.DataFrame(logistic_rows)

# ============================================================
# WRITE LOGISTIC LATEX TABLE
# ============================================================

latex_logistic = r"""\begin{tabular}{lcccccc}
\hline
Method & $\beta$ & $\beta_{low}$ & $\beta_{high}$ & $x_0$ & $x_{0,low}$ & $x_{0,high}$ \\
\hline
"""

for _, row in df_logistic.iterrows():
    latex_logistic += (
        f"{row['Method']} & {row['beta']} & {row['beta_low']} & "
        f"{row['beta_high']} & {row['x0']} & {row['x0_low']} & {row['x0_high']} \\\\\n"
    )

latex_logistic += r"""\hline
\end{tabular}
"""

with open(os.path.join(OUTPUT_DIR, "logistic_parameters_table.tex"), "w") as f:
    f.write(latex_logistic)

# ============================================================
# LOAD CRITICAL K DATA
# ============================================================

def load_critical(path, method_name):
    file = os.path.join(path, "per_instance_critical_k.csv")
    if not os.path.exists(file):
        return pd.DataFrame(columns=["Method", "Critical_k"])

    df = pd.read_csv(file)
    df["Method"] = method_name
    df = df.rename(columns={"critical_k": "Critical_k"})
    return df[["Method", "Critical_k"]]

df_critical = pd.concat([
    load_critical(GNN_PATH, "GNN"),
    load_critical(KNN_PATH, "Nearest-k"),
    load_critical(RAND_PATH, "Random-k")
], ignore_index=True)

# ============================================================
# WRITE CRITICAL K LATEX TABLE
# ============================================================

latex_critical = r"""\begin{tabular}{lc}
\hline
Method & Critical $k$ \\
\hline
"""

for _, row in df_critical.iterrows():
    latex_critical += f"{row['Method']} & {row['Critical_k']} \\\\\n"

latex_critical += r"""\hline
\end{tabular}
"""

with open(os.path.join(OUTPUT_DIR, "critical_k_distribution_table.tex"), "w") as f:
    f.write(latex_critical)





# ============================================================
# GENERATE PNG TABLES
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_table_figure(df, title, col_widths=None, figsize=(10, 4)):
    """Create a professional-looking table figure"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.1)
    
    # Row styling with text wrapping
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F0FE')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_height(0.08)
            # Ensure text stays within cell boundaries
            cell = table[(i, j)]
            cell.get_text().set_wrap(True)
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    return fig

# ============================================================
# GENERATE LOGISTIC PARAMETERS PNG
# ============================================================

# Format the logistic dataframe for display
df_logistic_display = df_logistic.copy()
df_logistic_display.columns = ['Method', 'β', 'β (low)', 'β (high)', 'x₀', 'x₀ (low)', 'x₀ (high)']

# Create and save logistic parameters table
fig_logistic = create_table_figure(
    df_logistic_display, 
    'Logistic Regression Parameters Comparison',
    col_widths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
)
fig_logistic.savefig(
    os.path.join(OUTPUT_DIR, "logistic_parameters_table.png"), 
    dpi=300, 
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close()

# ============================================================
# GENERATE CRITICAL K STATISTICS PNG
# ============================================================

# Calculate statistics for critical k
critical_stats = df_critical.groupby('Method')['Critical_k'].agg([
    ('Mean', 'mean'),
    ('Std Dev', 'std'),
    ('Min', 'min'),
    ('Max', 'max'),
    ('Count', 'count')
]).round(2)
critical_stats.reset_index(inplace=True)

# Convert to string and format to ensure proper display
critical_stats_display = critical_stats.copy()
for col in critical_stats_display.columns[1:]:  # Skip 'Method' column
    critical_stats_display[col] = critical_stats_display[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

# Create and save critical k statistics table with adjusted figure size and column widths
fig_critical = create_table_figure(
    critical_stats_display,
    'Critical k Statistics by Method',
    col_widths=[0.18, 0.16, 0.16, 0.16, 0.16, 0.16],
    figsize=(12, 4)  # Wider figure to accommodate values
)
fig_critical.savefig(
    os.path.join(OUTPUT_DIR, "critical_k_stats_table.png"),
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close()

# ============================================================
# GENERATE CRITICAL K DISTRIBUTION VISUALIZATION
# ============================================================

# Create a box plot for critical k distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Prepare data for box plot
data_to_plot = [df_critical[df_critical['Method'] == method]['Critical_k'].values 
                for method in df_critical['Method'].unique()]
methods = df_critical['Method'].unique()

# Create box plot
bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True,
                boxprops=dict(facecolor='#4472C4', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))

# Style the plot
ax.set_ylabel('Critical k', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Critical k Values by Method', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean markers
for i, method in enumerate(methods):
    mean_val = df_critical[df_critical['Method'] == method]['Critical_k'].mean()
    ax.scatter(i+1, mean_val, color='green', s=100, zorder=5, marker='^', label='Mean' if i==0 else '')

if len(methods) > 0:
    ax.legend(loc='upper right')

fig.savefig(
    os.path.join(OUTPUT_DIR, "critical_k_distribution.png"),
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
plt.close()

print("PNG tables and visualizations generated.")

print("Combined tables generated.")
