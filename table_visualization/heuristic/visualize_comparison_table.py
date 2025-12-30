import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def split_and_save_csv_table(csv_path, title, rows_per_table=30):
    """
    Split large CSV table into multiple smaller tables for better readability
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    
    print(f"📊 Processing: {title}")
    print(f"   Total rows: {total_rows}, Splitting into {np.ceil(total_rows/rows_per_table):.0f} tables")
    
    # Split dataframe into chunks
    chunks = []
    for i in range(0, total_rows, rows_per_table):
        chunk = df.iloc[i:i + rows_per_table]
        chunks.append(chunk)
    
    # Create and save each chunk
    saved_files = []
    for idx, chunk in enumerate(chunks):
        part_num = idx + 1
        chunk_title = f"{title} - Part {part_num}/{len(chunks)}"
        
        # Create larger figure for better readability
        num_rows = len(chunk)
        num_cols = len(chunk.columns)
        
        # Adjust figure size based on content
        fig_width = min(25, max(15, num_cols * 1.5))  # Width based on columns
        fig_height = min(40, max(12, num_rows * 0.6))  # Height based on rows
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=chunk.values,
            colLabels=chunk.columns,
            loc='center',
            cellLoc='center'
        )
        
        # Auto-adjust font size based on table size
        font_size = max(8, min(12, 14 - num_cols * 0.15))
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        
        # Adjust column widths
        table.auto_set_column_width([i for i in range(num_cols)])
        
        # Style header and rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2c3e50')  # Dark blue-gray
            else:
                # Alternating row colors
                cell.set_facecolor('#f8f9fa' if row % 2 == 0 else 'white')
                cell.set_text_props(color='black')
        
        # Add title and subtitle
        plt.suptitle(chunk_title, fontsize=14, fontweight='bold', y=0.98)
        plt.title(f"Rows {idx*rows_per_table+1} to {min((idx+1)*rows_per_table, total_rows)} of {total_rows}", 
                 fontsize=10, pad=10)
        
        plt.tight_layout()
        
        # Save with informative filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{safe_title}_part{part_num}_{base_name}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(filename)
        print(f"   ✅ Saved Part {part_num}: {filename} ({len(chunk)} rows)")
        plt.close()
    
    print(f"\n📁 All parts saved:")
    for f in saved_files:
        print(f"   • {f}")
    
    return saved_files


# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use the function
split_and_save_csv_table(
    "heuristic_comparison/comparison_table.csv",
    "",
    rows_per_table=30  # Adjust as needed
)