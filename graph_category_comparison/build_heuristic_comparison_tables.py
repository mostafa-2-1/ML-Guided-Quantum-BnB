import os
import json
import pandas as pd
from collections import defaultdict

# =========================
# Paths
# =========================

RESULTS_ROOT = "heuristicSolver_results"
OUT_ROOT = "heuristic_comparison"

FULL_DIR = os.path.join(RESULTS_ROOT, "full")
EDGEGNN_ONLY_DIR = os.path.join(RESULTS_ROOT, "edgeGNN_only")
CASCADE_DIR = os.path.join(RESULTS_ROOT, "edgeGNN_cascade")

os.makedirs(OUT_ROOT, exist_ok=True)

# =========================
# Size ranges
# =========================

SIZE_RANGES = [
    (1, 20),
    (21, 50),
    (51, 100),
    (101, 200),
    (201, 500),
    (501, 1000),
    (1001, 2000),
    (2001, float("inf")),
]

def range_name(n):
    for lo, hi in SIZE_RANGES:
        if lo <= n <= hi:
            return f"range_{lo}_{'inf' if hi == float('inf') else hi}"
    raise ValueError(f"No range for n={n}")

# =========================
# Utilities
# =========================

def load_results(folder):
    results = {}
    if not os.path.exists(folder):
        return results

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(folder, fname)) as f:
            data = json.load(f)
        if data.get("solved", False):
            results[data["graph"]] = data
    return results

# =========================
# Load all results
# =========================

full_results = load_results(FULL_DIR)
edgegnn_results = load_results(EDGEGNN_ONLY_DIR)

# Cascade: graph -> factor -> data
cascade_results = defaultdict(dict)
factors = sorted(os.listdir(CASCADE_DIR))

for factor in factors:
    factor_dir = os.path.join(CASCADE_DIR, factor)
    if not os.path.isdir(factor_dir):
        continue

    for fname in os.listdir(factor_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(factor_dir, fname)) as f:
            data = json.load(f)
        if data.get("solved", False):
            cascade_results[data["graph"]][factor] = data

# =========================
# TABLE 1 — Cross-method comparison
# =========================

comparison_rows = []

for graph, full_data in full_results.items():
    if graph not in edgegnn_results:
        continue
    if graph not in cascade_results:
        continue
    if len(cascade_results[graph]) == 0:
        continue

    # Best cascade = lowest tour cost
    best_factor, best_data = min(
        cascade_results[graph].items(),
        key=lambda x: x[1]["tour_cost"]
    )

    row = {
        "graph": graph,
        "n": full_data["n"],

        "full_cost": full_data["tour_cost"],
        "full_runtime": full_data["runtime_sec"],

        "edgegnn_cost": edgegnn_results[graph]["tour_cost"],
        "edgegnn_runtime": edgegnn_results[graph]["runtime_sec"],

        "cascade_best_cost": best_data["tour_cost"],
        "cascade_best_runtime": best_data["runtime_sec"],
        "cascade_best_factor": best_factor,
    }

    comparison_rows.append(row)


comparison_df = pd.DataFrame(comparison_rows)

comparison_df.sort_values("n", inplace=True)

out_path = os.path.join(OUT_ROOT, "comparison_table.csv")
comparison_df.to_csv(out_path, index=False)

print(f"[✓] Saved heuristic comparison table → {out_path}")

# =========================
# TABLE 2 — Cascade-only (per range)
# =========================

graphs_by_range = defaultdict(list)

for graph, factor_map in cascade_results.items():
    any_data = next(iter(factor_map.values()))
    n = any_data["n"]
    graphs_by_range[range_name(n)].append(graph)

for rname, graphs in graphs_by_range.items():
    out_dir = os.path.join(OUT_ROOT, rname)
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for graph in graphs:
        row = {"graph": graph}

        for factor in factors:
            if factor in cascade_results[graph]:
                data = cascade_results[graph][factor]
                row[f"{factor}_cost"] = data["tour_cost"]
                row[f"{factor}_runtime"] = data["runtime_sec"]
            else:
                row[f"{factor}_cost"] = None
                row[f"{factor}_runtime"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("graph", inplace=True)

    out_path = os.path.join(out_dir, "cascade_factors.csv")
    df.to_csv(out_path, index=False)

    print(f"[✓] Saved heuristic cascade table → {out_path}")

print("\nAll heuristic comparison tables generated successfully.")
