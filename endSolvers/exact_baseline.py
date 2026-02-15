import os
import sys
import math
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25]
#SEEDS = [1, 2, 3]  # same as GNN code
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# ------------------ Distance functions (same as GNN code) ------------------

def compute_geo_distances(coords):
    n = len(coords)
    PI = 3.141592653589793
    lat = np.zeros(n)
    lon = np.zeros(n)
    for i in range(n):
        deg_x = int(coords[i, 0])
        min_x = coords[i, 0] - deg_x
        lat[i] = PI * (deg_x + 5.0 * min_x / 3.0) / 180.0
        deg_y = int(coords[i, 1])
        min_y = coords[i, 1] - deg_y
        lon[i] = PI * (deg_y + 5.0 * min_y / 3.0) / 180.0
    RRR = 6378.388
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            q1 = np.cos(lon[i] - lon[j])
            q2 = np.cos(lat[i] - lat[j])
            q3 = np.cos(lat[i] + lat[j])
            dij = int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
            D[i, j] = dij
            D[j, i] = dij
    return D

def compute_att_distances(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            xd = coords[i, 0] - coords[j, 0]
            yd = coords[i, 1] - coords[j, 1]
            rij = math.sqrt((xd * xd + yd * yd) / 10.0)
            tij = int(rij)
            dij = tij + 1 if tij < rij else tij
            D[i, j] = dij
            D[j, i] = dij
    return D

def compute_euc_2d(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            xd = coords[i, 0] - coords[j, 0]
            yd = coords[i, 1] - coords[j, 1]
            dij = int(math.sqrt(xd*xd + yd*yd) + 0.5)
            D[i, j] = dij
            D[j, i] = dij
    return D

# ------------------ Parse TSP ------------------

def parse_tsp(filepath):
    name = os.path.basename(filepath)
    n = None
    coords = {}
    D = None
    edge_type = None
    edge_format = None
    reading_coords = False
    reading_edges = False
    edge_lines = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            up = line.upper()
            if up.startswith('NAME'):
                parts = line.split(':', 1)
                if len(parts) > 1: name = parts[1].strip()
            elif up.startswith('DIMENSION'):
                n = int(line.split(':')[1].strip())
            elif up.startswith('EDGE_WEIGHT_TYPE'):
                edge_type = line.split(':')[1].strip().upper()
            elif up.startswith('EDGE_WEIGHT_FORMAT'):
                edge_format = line.split(':')[1].strip().upper()
            elif up.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                reading_edges = False
            elif up.startswith('EDGE_WEIGHT_SECTION'):
                reading_edges = True
                reading_coords = False
            elif line.startswith('DISPLAY_DATA_SECTION') or line.startswith('EOF'):
                reading_coords = False
                reading_edges = False
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0]) - 1
                        x = float(parts[1])
                        y = float(parts[2])
                        coords[idx] = (x, y)
                    except: pass
            elif reading_edges:
                edge_lines.extend(line.split())

    coords_arr = None
    if coords:
        coords_arr = np.zeros((n, 2), dtype=float)
        for i in range(n):
            coords_arr[i] = coords.get(i, (0, 0))

    D_arr = None
    if edge_type == 'EXPLICIT' and edge_lines:
        edge_vals = list(map(float, edge_lines))
        D_arr = np.zeros((n, n), dtype=float)
        if edge_format == 'UPPER_ROW':
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    if idx < len(edge_vals):
                        D_arr[i, j] = edge_vals[idx]
                        D_arr[j, i] = edge_vals[idx]
                        idx += 1
        elif edge_format == 'FULL_MATRIX':
            idx = 0
            for i in range(n):
                for j in range(n):
                    if idx < len(edge_vals):
                        D_arr[i, j] = edge_vals[idx]
                        idx += 1
        elif edge_format == 'LOWER_DIAG_ROW':
            idx = 0
            for i in range(n):
                for j in range(i+1):
                    if idx < len(edge_vals):
                        D_arr[i, j] = edge_vals[idx]
                        D_arr[j, i] = edge_vals[idx]
                        idx += 1
        elif edge_format == 'UPPER_DIAG_ROW':
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    if idx < len(edge_vals):
                        D_arr[i, j] = edge_vals[idx]
                        D_arr[j, i] = edge_vals[idx]
                        idx += 1
    elif edge_type == 'GEO' and coords_arr is not None:
        D_arr = compute_geo_distances(coords_arr)
    elif edge_type == 'ATT' and coords_arr is not None:
        D_arr = compute_att_distances(coords_arr)
    elif edge_type == 'EUC_2D' and coords_arr is not None:
        D_arr = compute_euc_2d(coords_arr)

    out = {'name': name, 'n': n, 'edge_weight_type': edge_type}
    if coords_arr is not None: out['coords'] = coords_arr
    if D_arr is not None: out['D'] = D_arr
    return out

# ------------------ Tour and LKH ------------------

def tour_length(tour: List[int], D: np.ndarray) -> float:
    n = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))

def solve_tsp_lkh(par_file: str, lkh_path: str = r"C:/LKH/LKH-3.exe",
                  time_limit: int = 3600, seed: Optional[int] = None) -> Tuple[Optional[List[int]], float, float]:

    par_file = os.path.abspath(par_file)
    tmp_dir = None
    if seed is not None:
        tmp_dir = tempfile.mkdtemp(prefix="lkh_seed_")
        tmp_par = os.path.join(tmp_dir, os.path.basename(par_file))
        with open(par_file, "r") as f:
            lines = f.readlines()
        has_seed = False
        with open(tmp_par, "w") as f:
            for line in lines:
                if line.strip().upper().startswith("SEED"):
                    f.write(f"SEED = {seed}\n")
                    has_seed = True
                else:
                    f.write(line)
            if not has_seed:
                f.write(f"\nSEED = {seed}\n")
        par_file = tmp_par

    sol_file = None
    try:
        with open(par_file, "r") as f:
            for line in f:
                if line.strip().startswith("OUTPUT_TOUR_FILE"):
                    sol_file = line.split("=")[1].strip()
        if sol_file is not None: sol_file = os.path.abspath(sol_file)
    except Exception as e:
        if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Failed to read .par file: {par_file} -> {e}")
        return None, float('inf'), 0.0

    t0 = time.perf_counter()
    try:
        subprocess.run([lkh_path, par_file], capture_output=True, text=True, timeout=time_limit)
        runtime = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"LKH timeout after {time_limit}s for {par_file}")
        return None, float('inf'), time.perf_counter() - t0

    if sol_file is None or not os.path.exists(sol_file):
        if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"LKH did not produce tour file for {par_file}")
        return None, float('inf'), runtime

    tour = []
    try:
        with open(sol_file, "r") as f:
            reading = False
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if line.upper() == "TOUR_SECTION":
                    reading = True
                    continue
                if line in ("-1", "EOF"): break
                if reading:
                    try: tour.append(int(line) - 1)
                    except: pass
    except Exception as e:
        if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Failed to parse LKH tour: {sol_file} -> {e}")
        return None, float('inf'), runtime

    if tmp_dir: shutil.rmtree(tmp_dir, ignore_errors=True)
    if not tour:
        print(f"No tour found in {sol_file}")
        return None, float('inf'), runtime

    return tour, float('inf'), runtime

def parse_opt_tour(filepath: str) -> Optional[List[int]]:
    if not os.path.exists(filepath): return None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        tour_section = False
        tour = []
        for line in lines:
            line = line.strip()
            if not tour_section:
                if line.upper().startswith("TOUR_SECTION"): tour_section = True
                continue
            if line in ("-1", "EOF", "END"): break
            if not line or line.startswith("COMMENT"): continue
            for num in line.split():
                if num in ("-1", "EOF"): break
                tour.append(int(num))
        if not tour: return None
        return tour
    except Exception as e:
        print(f"Error parsing optimal tour file: {e}")
        return None

# ------------------ Helper ------------------

def select_k_values(n):
    if n > 1500: return [8, 9, 10]
    return K_VALUES

# ------------------ Baseline Evaluation ------------------

def evaluate_baseline(instance_name: str, baseline_dir: str, full_par_dir: str, tsp_dir: str, time_limit: int = 3600) -> List[Dict]:
    """
    Evaluate one instance for one baseline (nearest_k or random_k), fully mirroring GNN evaluation.
    """
    rows = []
    full_tsp_path = os.path.join(tsp_dir, instance_name + ".tsp")
    parsed_full = parse_tsp(full_tsp_path)
    if parsed_full is None:
        raise RuntimeError(f"Failed to parse TSP: {full_tsp_path}")
    n = parsed_full["n"]
    D_full = parsed_full['D']
    np.fill_diagonal(D_full, 0)

    # Load optimal tour if exists
    optimal_cost = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    opt_tour_path = os.path.join(root_dir, "Desktop/SeniorProject_step1/tsplib_data", f"{instance_name}.opt.tour")
    optimal_tour_1based = parse_opt_tour(opt_tour_path)
    if optimal_tour_1based and len(optimal_tour_1based) == n:
        optimal_cost = tour_length([x-1 for x in optimal_tour_1based], D_full)

    k_vals = select_k_values(n)

    def run_seed(seed):
        seed_results = []
        print(f"\n==== Running seed {seed} for {instance_name} ====\n", flush=True)

        # Full graph using original full par
        full_par = os.path.join(full_par_dir, instance_name + ".par")
        tour_full, _, runtime_full = solve_tsp_lkh(full_par, time_limit=time_limit, seed=seed)
        if tour_full is None: raise RuntimeError(f"LKH failed on full graph {instance_name}, seed {seed}")
        cost_full = tour_length(tour_full, D_full)

        for k in k_vals:
            pruned_par = os.path.join(baseline_dir, f"k{k}", f"{instance_name}.par")
            if not os.path.exists(pruned_par): continue
            tour_pruned, _, runtime_pruned = solve_tsp_lkh(pruned_par, time_limit=time_limit, seed=seed)
            if tour_pruned is None:
                cost_pruned = float('inf')
                success = False
            else:
                cost_pruned = tour_length(tour_pruned, D_full)
                success = True
            gap_abs = cost_pruned - cost_full if math.isfinite(cost_pruned) else float('inf')
            gap_rel = (gap_abs / cost_full) * 100.0 if math.isfinite(cost_pruned) else float('inf')
            seed_results.append({
                "instance": instance_name,
                "seed": seed,
                "k": k,
                "runtime_full": runtime_full,
                "runtime_pruned": runtime_pruned,
                "cost_full": cost_full,
                "cost_pruned": cost_pruned,
                "gap_abs": gap_abs,
                "gap_rel_percent": gap_rel,
                "tour_found": success,
                "n_nodes": n
            })
        return seed_results

    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count())) as executor:
        futures = [executor.submit(run_seed, seed) for seed in SEEDS]
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results

# ------------------ Write baseline results ------------------

def write_baseline_results(rows: List[Dict], baseline_name: str):
    root_dir = os.getcwd()
    output_root = os.path.join(root_dir, "baseline_solver_results", baseline_name)
    os.makedirs(output_root, exist_ok=True)

    seeds = sorted(set(r["seed"] for r in rows))
    for seed in seeds:
        seed_dir = os.path.join(output_root, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_rows = [r for r in rows if r["seed"] == seed]

        # Full results
        full_dir = os.path.join(seed_dir, "full")
        os.makedirs(full_dir, exist_ok=True)
        df_full = pd.DataFrame([{
            "instance": r["instance"], "n": r["n_nodes"], "cost": r["cost_full"]
        } for r in seed_rows]).drop_duplicates(subset=["instance"])
        df_full.to_csv(os.path.join(full_dir, "results.csv"), index=False)

        # k results
        for k in sorted(set(r["k"] for r in seed_rows)):
            k_dir = os.path.join(seed_dir, f"k{k}")
            os.makedirs(k_dir, exist_ok=True)
            k_rows = []
            for r in seed_rows:
                if r["k"] != k: continue
                gap = r["gap_rel_percent"]
                success = math.isfinite(gap) and gap <= 0.7
                k_rows.append({
                    "instance": r["instance"],
                    "n": r["n_nodes"],
                    "cost": r["cost_pruned"],
                    "gap_abs": r["gap_abs"],
                    "gap_rel_percent": r["gap_rel_percent"],
                    "tour_found": r["tour_found"],
                    "success": success
                })
            if k_rows:
                df_k = pd.DataFrame(k_rows)
                df_k.to_csv(os.path.join(k_dir, "results.csv"), index=False)

# ------------------ Main ------------------

def main():
    tsp_dir = "graphss_chosen"
    baselines = {
        "nearest_k": "baseline_parameters/nearest_k",
        "random_k": "baseline_parameters/random_k"
    }
    full_par_dir = "parameters/full"

    for baseline_name, baseline_dir in baselines.items():
        print(f"\n=== Evaluating baseline: {baseline_name} ===\n")
        rows = []

        all_tsp_files = sorted(f for f in os.listdir(tsp_dir) if f.lower().endswith(".tsp"))
        for fname in all_tsp_files:
            instance_name = os.path.splitext(fname)[0]
            parsed = parse_tsp(os.path.join(tsp_dir, fname))
            if parsed is None:
                print(f"[skip] Could not parse {fname}")
                continue
            n = parsed["n"]
            if n < 350 or n > 1000:  # same filtering
                continue
            try:
                print(f"\n{'='*50}")
                print(f"Instance: {instance_name} (n={n})")
                print(f"{'='*50}")
                res_list = evaluate_baseline(instance_name, baseline_dir, full_par_dir, tsp_dir)
                rows.extend(res_list)
            except Exception as e:
                print(f"❌ Error for instance {instance_name}: {e}")
                import traceback
                traceback.print_exc()

        if rows:
            write_baseline_results(rows, baseline_name=baseline_name)
            print("\n✅ Results written to solver_results/")



if __name__ == "__main__":
    main()
