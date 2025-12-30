import pandas as pd
import matplotlib.pyplot as plt

def show_table(df, title, col_width=0.15, font_size=10, scale=(1, 1.5)):
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(*scale)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#DDEEFF')
        else:
            cell.set_facecolor('#F7F7F7' if row % 2 == 0 else 'white')

    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


# =========================
# Load comparison table
# =========================

df = pd.read_csv("exact_comparison/comparison_table.csv")

# Optional: round numbers so it looks human
df = df.round({
    "full_cost": 2,
    "full_runtime": 3,
    "edgegnn_cost": 2,
    "edgegnn_runtime": 3,
    "cascade_best_cost": 2,
    "cascade_best_runtime": 3
})

show_table(df, "Exact Solver Comparison (Full vs Pruned)")
