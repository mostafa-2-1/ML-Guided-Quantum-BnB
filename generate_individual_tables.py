import os
import pandas as pd

GNN_PATH = "structural_analysis_gnn"
KNN_PATH = "structural_analysis_baseline/nearest_k"
RAND_PATH = "structural_analysis_baseline/random_k"

OUTPUT_DIR = "table_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

methods = {
    "gnn": GNN_PATH,
    "knn": KNN_PATH,
    "random": RAND_PATH
}

def write_latex(df, output_file):
    latex = df.to_latex(index=False, float_format="%.4f")
    with open(output_file, "w") as f:
        f.write(latex)

for key, path in methods.items():

    macro_file = os.path.join(path, "macro_k_summary.csv")
    degree_file = os.path.join(path, "structural_degree_summary.csv")

    if os.path.exists(macro_file):
        df_macro = pd.read_csv(macro_file)
        write_latex(df_macro,
            os.path.join(OUTPUT_DIR, f"macro_k_summary_{key}.tex"))

    if os.path.exists(degree_file):
        df_degree = pd.read_csv(degree_file)
        write_latex(df_degree,
            os.path.join(OUTPUT_DIR, f"structural_degree_summary_{key}.tex"))


import os
import pandas as pd
import matplotlib.pyplot as plt

GNN_PATH = "structural_analysis_gnn"
KNN_PATH = "structural_analysis_baseline/nearest_k"
RAND_PATH = "structural_analysis_baseline/random_k"

OUTPUT_DIR = "table_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

methods = {
    "gnn": GNN_PATH,
    "knn": KNN_PATH,
    "random": RAND_PATH
}

def write_latex(df, output_file):
    latex = df.to_latex(index=False, float_format="%.4f")
    with open(output_file, "w") as f:
        f.write(latex)

# ============================================================
# PNG GENERATION FUNCTION
# ============================================================

def create_table_figure(df, title, figsize=None):
    """Create a professional-looking table figure"""
    
    # Auto-adjust figure size based on table dimensions
    if figsize is None:
        num_cols = len(df.columns)
        num_rows = len(df)
        figsize = (max(10, num_cols * 1.5), max(4, num_rows * 0.5 + 2))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Format numeric columns to 4 decimal places
    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    
    # Calculate column widths based on number of columns
    num_cols = len(df_display.columns)
    col_widths = [1.0/num_cols] * num_cols
    
    # Create table
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.1)
    
    # Row styling with text wrapping
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
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
# PROCESS EACH METHOD
# ============================================================

method_names = {
    "gnn": "GNN",
    "knn": "Nearest-k",
    "random": "Random-k"
}

for key, path in methods.items():

    macro_file = os.path.join(path, "macro_k_summary.csv")
    degree_file = os.path.join(path, "structural_degree_summary.csv")

    # Process macro k summary
    if os.path.exists(macro_file):
        df_macro = pd.read_csv(macro_file)
        
        # Write LaTeX
        write_latex(df_macro,
            os.path.join(OUTPUT_DIR, f"macro_k_summary_{key}.tex"))
        
        # Generate PNG
        fig_macro = create_table_figure(
            df_macro,
            f'Macro k Summary - {method_names[key]}'
        )
        fig_macro.savefig(
            os.path.join(OUTPUT_DIR, f"macro_k_summary_{key}.png"),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

    # Process structural degree summary
    if os.path.exists(degree_file):
        df_degree = pd.read_csv(degree_file)
        
        # Write LaTeX
        write_latex(df_degree,
            os.path.join(OUTPUT_DIR, f"structural_degree_summary_{key}.tex"))
        
        # Generate PNG
        fig_degree = create_table_figure(
            df_degree,
            f'Structural Degree Summary - {method_names[key]}'
        )
        fig_degree.savefig(
            os.path.join(OUTPUT_DIR, f"structural_degree_summary_{key}.png"),
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()

print("Individual LaTeX tables and PNG visualizations generated.")


print("Individual LaTeX tables generated.")
