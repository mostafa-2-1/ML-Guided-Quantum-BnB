import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrollable_table_viewer import show_csv_table

BASE_DIR = "heuristic_comparison"  # or exact_comparison

for folder in sorted(os.listdir(BASE_DIR)):
    if not folder.startswith("range_"):
        continue

    path = os.path.join(BASE_DIR, folder, "cascade_factors.csv")
    if os.path.exists(path):
        show_csv_table(
            path,
            f"Cascade Factors — {folder}"
        )
