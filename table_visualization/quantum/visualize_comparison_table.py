
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from table_visualization.heuristic.visualize_comparison_table import split_and_save_csv_table
split_and_save_csv_table(
    "quantum_comparison/comparison_table.csv",
    "Quantum Solver – Cross-Method Comparison",
    rows_per_table=30
)
