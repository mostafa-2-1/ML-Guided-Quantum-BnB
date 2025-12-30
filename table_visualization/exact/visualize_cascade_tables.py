import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "exact_comparison"

def show_table(df, title):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E3F2FD')
        else:
            cell.set_facecolor('#FAFAFA' if row % 2 == 0 else 'white')

    ax.set_title(title, fontsize=15, pad=15)
    plt.tight_layout()
    plt.show()


for folder in os.listdir(BASE_DIR):
    if not folder.startswith("range_"):
        continue

    path = os.path.join(BASE_DIR, folder, "cascade_factors.csv")
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path)

    # Remove columns that contain any NaN values
    original_cols = len(df.columns)
    df = df.dropna(axis=1)  # axis=1 means columns
    removed_cols = original_cols - len(df.columns)
    
    if removed_cols > 0:
        print(f"Removed {removed_cols} columns with NaN values in {folder}")

    # Round numeric columns
    for col in df.columns:
        if "runtime" in col:
            df[col] = df[col].round(3)
        if "cost" in col:
            df[col] = df[col].round(2)

    show_table(df, f"Cascade Factors — {folder}")