


import os
import sys
import math
import time
import shutil
import tempfile
import subprocess
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25]

def compute_geo_distances(coords):
    """Compute GEO (geographic) distances as per TSPLIB spec."""
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
            tij = int(rij)  # truncate, do NOT round
            if tij < rij:
                dij = tij + 1
            else:
                dij = tij
            D[i, j] = dij
            D[j, i] = dij
    return D



def compute_euc_2d(coords):
    """Compute TSPLIB EUC_2D distances (rounded Euclidean)."""
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


def parse_tsp(filepath):
    """
    Parse a .tsp file (TSPLIB). Returns dict with correct distance matrix
    based on EDGE_WEIGHT_TYPE (EUC_2D, GEO, ATT, EXPLICIT).
    """
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
            if not line:
                continue
            up = line.upper()
            if up.startswith('NAME'):
                parts = line.split(':', 1)
                if len(parts) > 1:
                    name = parts[1].strip()
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
            elif up.startswith('DISPLAY_DATA_SECTION') or up.startswith('EOF'):
                reading_edges = False
                reading_coords = False
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0]) - 1
                        x = float(parts[1])
                        y = float(parts[2])
                        coords[idx] = (x, y)
                    except:
                        pass
            elif reading_edges:
                edge_lines.extend(line.split())

    # Build coords array if present
    coords_arr = None
    if coords:
        coords_arr = np.zeros((n, 2), dtype=float)
        for i in range(n):
            coords_arr[i] = coords.get(i, (0, 0))

    # Build distance matrix based on EDGE_WEIGHT_TYPE
    D_arr = None
    
    # EXPLICIT: read from file
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
    
    # GEO: geographic coordinates
    elif edge_type == 'GEO' and coords_arr is not None:
        D_arr = compute_geo_distances(coords_arr)
    
    # ATT: pseudo-Euclidean
    elif edge_type == 'ATT' and coords_arr is not None:
        D_arr = compute_att_distances(coords_arr)
    
    # EUC_2D or default: Euclidean
    elif edge_type == 'EUC_2D' and coords_arr is not None:
        D_arr = compute_euc_2d(coords_arr)


    out = {
        'name': name, 
        'n': n,
        'edge_weight_type': edge_type
    }
    if coords_arr is not None:
        out['coords'] = coords_arr
    if D_arr is not None:
        out['D'] = D_arr

    return out


def tour_length(tour: List[int], D: np.ndarray) -> float:
    """Compute tour length with distance matrix D (0-based indices)."""
    n = len(tour)
    return float(sum(D[tour[i], tour[(i + 1) % n]] for i in range(n)))


def solve_tsp_lkh(par_file: str,
                  lkh_path: str = r"C:/LKH/LKH-3.exe",
                  time_limit: int = 3600,
                  seed: Optional[int] = None) -> Tuple[Optional[List[int]], float, float]:
    """
    Solve TSP using LKH solver with an existing .par file.
    If seed is provided, a temporary par file with SEED injected is used.
    Returns: (tour, reported_cost, runtime)
    """

    par_file = os.path.abspath(par_file)

    # --- If seed requested, create temporary par copy ---
    tmp_dir = None
    if seed is not None:
        import tempfile, shutil
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

        par_file = tmp_par  # solver will use temp par

    # ----------------- ORIGINAL SOLVER BELOW -----------------

    # Read OUTPUT_TOUR_FILE from .par file
    sol_file = None
    try:
        with open(par_file, "r") as f:
            for line in f:
                if line.strip().startswith("OUTPUT_TOUR_FILE"):
                    sol_file = line.split("=")[1].strip()
        if sol_file is not None:
            sol_file = os.path.abspath(sol_file)
    except Exception as e:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Failed to read .par file: {par_file} -> {e}")
        return None, float('inf'), 0.0

    t0 = time.perf_counter()
    try:
        subprocess.run(
            [lkh_path, par_file],
            capture_output=True,
            text=True,
            timeout=time_limit
        )
        runtime = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"LKH timeout after {time_limit}s for {par_file}")
        return None, float('inf'), time.perf_counter() - t0

    if sol_file is None or not os.path.exists(sol_file):
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"LKH did not produce tour file for {par_file}")
        return None, float('inf'), runtime

    # Parse tour (unchanged)
    try:
        with open(sol_file, "r") as f:
            tour = []
            reading = False
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.upper() == "TOUR_SECTION":
                    reading = True
                    continue
                if line in ("-1", "EOF"):
                    break
                if reading:
                    try:
                        tour.append(int(line) - 1)
                    except:
                        pass
    except Exception as e:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Failed to parse LKH tour: {sol_file} -> {e}")
        return None, float('inf'), runtime

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not tour:
        print(f"No tour found in {sol_file}")
        return None, float('inf'), runtime

    return tour, float('inf'), runtime

def parse_opt_tour(filepath: str) -> Optional[List[int]]:
    """Parse a TSPLIB .opt.tour file to extract the optimal tour (1-indexed)."""
    if not os.path.exists(filepath):
        print(f"    Optimal tour file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        tour_section = False
        tour = []
        
        for line in lines:
            line = line.strip()
            
            if not tour_section:
                if line.upper().startswith("TOUR_SECTION"):
                    tour_section = True
                continue
            
            # Stop when we hit -1, EOF, or end of section
            if line in ("-1", "EOF", "EOF\n", "END"):
                break
            
            # Skip empty lines or comments
            if not line or line.startswith("COMMENT"):
                continue
            
            # Parse tour nodes (1-indexed)
            try:
                # Handle multiple numbers on one line
                for num in line.split():
                    if num in ("-1", "EOF"):
                        break
                    node = int(num)
                    tour.append(node)
            except ValueError:
                # Skip non-integer lines
                continue
        
        if not tour:
            print(f"    No tour found in {filepath}")
            return None
        
        return tour
        
    except Exception as e:
        print(f"    Error parsing optimal tour file: {e}")
        return None

def evaluate_instance(instance_name: str,
                      full_tsp_path: str,
                      time_limit: int = 3600) -> List[Dict]:
    """
    Solve full graph once, then solve pruned graphs for multiple k.
    Returns list of result dicts (one per k).
    """

    results = []

    parsed_full = parse_tsp(full_tsp_path)
    if parsed_full is None:
        raise RuntimeError(f"Failed to parse TSP: {full_tsp_path}")

    n = parsed_full["n"]
    edge_type = parsed_full.get("edge_weight_type", "EUC_2D")

    if 'D' not in parsed_full or parsed_full['D'] is None:
        raise RuntimeError(f"No distance matrix for {full_tsp_path}")

    D_full = parsed_full['D']
    np.fill_diagonal(D_full, 0)

    # ---------- Load optimal tour if exists ----------
    optimal_cost = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    opt_tour_path = os.path.join(root_dir, "Desktop/SeniorProject_step1/tsplib_data", f"{instance_name}.opt.tour")

    optimal_tour_1based = parse_opt_tour(opt_tour_path)
    if optimal_tour_1based:
        optimal_tour_0based = [node - 1 for node in optimal_tour_1based]
        if len(optimal_tour_0based) == n:
            optimal_cost = tour_length(optimal_tour_0based, D_full)

    SEEDS = list(range(1, 11))
    for seed in SEEDS:

        print(f"\n==== Running seed {seed} ====\n")

        # Solve FULL
        print(f"  Solving FULL graph...", flush=True)
        full_par = os.path.join("parameters", "full", f"{instance_name}.par")

        tour_full, _, runtime_full = solve_tsp_lkh(
            full_par, lkh_path=r"C:/LKH/LKH-3.exe",
            time_limit=time_limit,
            seed=seed
        )

        if tour_full is None:
            raise RuntimeError(f"LKH failed on full graph {instance_name}, seed {seed}")

        cost_full = tour_length(tour_full, D_full)

        # Solve PRUNED exactly like before
        for k in K_VALUES:
            pruned_par = os.path.join("parameters", f"k{k}", f"{instance_name}.par")

            if not os.path.exists(pruned_par):
                continue

            tour_pruned, _, runtime_pruned = solve_tsp_lkh(
                pruned_par, lkh_path=r"C:/LKH/LKH-3.exe",
                time_limit=time_limit,
                seed=seed
            )

            if tour_pruned is None:
                cost_pruned = float('inf')
                success = False
            else:
                cost_pruned = tour_length(tour_pruned, D_full)
                success = True

            # Compute gap (unchanged)
            if math.isfinite(cost_pruned):
                gap_abs = cost_pruned - cost_full
                gap_rel = (gap_abs / cost_full) * 100.0
            else:
                gap_abs = float('inf')
                gap_rel = float('inf')

            results.append({
                "instance": instance_name,
                "seed": seed,   # NEW
                "k": k,
                "runtime_full": runtime_full,
                "runtime_pruned": runtime_pruned,
                "cost_full": cost_full,
                "cost_pruned": cost_pruned,
                "gap_abs": gap_abs,
                "gap_rel_percent": gap_rel,
                "tour_found": success
            })

    return results


from concurrent.futures import ThreadPoolExecutor, as_completed

# updated K selection for large graphs
def select_k_values(n):
    if n > 1500:
        return [8, 9, 10]
    return K_VALUES

# SEEDS = [1, 2, 3]  # only 3 seeds
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def evaluate_instance_parallel(instance_name: str,
                               full_tsp_path: str,
                               time_limit: int = 3600) -> List[Dict]:
    """
    Solve full graph once, then solve pruned graphs for multiple k.
    Returns list of result dicts (one per k).
    """

    results = []

    parsed_full = parse_tsp(full_tsp_path)
    if parsed_full is None:
        raise RuntimeError(f"Failed to parse TSP: {full_tsp_path}")

    n = parsed_full["n"]
    edge_type = parsed_full.get("edge_weight_type", "EUC_2D")

    if 'D' not in parsed_full or parsed_full['D'] is None:
        raise RuntimeError(f"No distance matrix for {full_tsp_path}")

    D_full = parsed_full['D']
    np.fill_diagonal(D_full, 0)

    # ---------- Load optimal tour if exists ----------
    optimal_cost = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    opt_tour_path = os.path.join(root_dir, "Desktop/SeniorProject_step1/tsplib_data", f"{instance_name}.opt.tour")

    optimal_tour_1based = parse_opt_tour(opt_tour_path)
    if optimal_tour_1based:
        optimal_tour_0based = [node - 1 for node in optimal_tour_1based]
        if len(optimal_tour_0based) == n:
            optimal_cost = tour_length(optimal_tour_0based, D_full)
    k_vals = select_k_values(n)
    def run_seed(seed):
        seed_results = []

        print(f"\n==== Running seed {seed} for {instance_name} ====\n", flush=True)

        full_par = os.path.join("parameters", "full", f"{instance_name}.par")
        tour_full, _, runtime_full = solve_tsp_lkh(
            full_par, lkh_path=r"C:/LKH/LKH-3.exe",
            time_limit=time_limit,
            seed=seed
        )
        if tour_full is None:
            raise RuntimeError(f"LKH failed on full graph {instance_name}, seed {seed}")
        cost_full = tour_length(tour_full, D_full)

        for k in k_vals:
            pruned_par = os.path.join("parameters", f"k{k}", f"{instance_name}.par")
            if not os.path.exists(pruned_par):
                continue

            tour_pruned, _, runtime_pruned = solve_tsp_lkh(
                pruned_par, lkh_path=r"C:/LKH/LKH-3.exe",
                time_limit=time_limit,
                seed=seed
            )

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



def write_solver_results(rows, output_root="solver_results"):
    """
    rows = list of dicts returned by evaluate_instance across all instances and seeds
    """

    root_dir = os.getcwd()
    output_root = os.path.join(root_dir, output_root)
    os.makedirs(output_root, exist_ok=True)

    # ---- group rows by seed ----
    seeds = sorted(set(r["seed"] for r in rows))

    for seed in seeds:
        seed_dir = os.path.join(output_root, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_rows = [r for r in rows if r["seed"] == seed]

        # --- FULL results ---
        full_dir = os.path.join(seed_dir, "full")
        os.makedirs(full_dir, exist_ok=True)

        full_rows = []
        for r in seed_rows:
            full_rows.append({
                "instance": r["instance"],
                "n": r["n_nodes"],
                "cost": r["cost_full"]
            })

        df_full = pd.DataFrame(full_rows).drop_duplicates(subset=["instance"])
        df_full.to_csv(os.path.join(full_dir, "results.csv"), index=False)

        # --- PRUNED results per k ---
        for k in sorted(set(r["k"] for r in seed_rows)):
            k_dir = os.path.join(seed_dir, f"k{k}")
            os.makedirs(k_dir, exist_ok=True)

            k_rows = []
            for r in seed_rows:
                if r["k"] != k:
                    continue

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
            path=os.path.join(k_dir, "results.csv")
            if k_rows:
                df_k = pd.DataFrame(k_rows)
                if os.path.exists(path):
                    old_df = pd.read_csv(path)
                    df_k = pd.concat([old_df, df_k], ignore_index=True)

                df_k.to_csv(path, index=False)





def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pruned TSP graphs using lkh solver"
    )
   
    parser.add_argument("--tsp_dir", default="graphss_chosen")
    parser.add_argument("--par_dir", default="parameters")
    parser.add_argument("--stage", default="stage2")
    
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--time_limit", type=int, default=3600)
    parser.add_argument("--output_csv", default="lkh.csv")
    parser.add_argument("--instance", type=str, default=None,
                    help="Run only a specific instance (without .tsp extension)")

    args = parser.parse_args()

   
    
    print("=" * 70)
    print("TSP EVALUATION WITH lkh")
    print("=" * 70)
   
    print(f"Full graphs:   {args.tsp_dir}")
   
    print("=" * 70)

    rows = []
    all_tsp_files = sorted(
        f for f in os.listdir(args.tsp_dir) if f.lower().endswith(".tsp")
    )

    for fname in all_tsp_files:
        instance_name = os.path.splitext(fname)[0]
        if args.instance is not None and instance_name != args.instance:
            continue
        full_tsp_path = os.path.join(args.tsp_dir, fname)

        parsed = parse_tsp(full_tsp_path)
        if parsed is None:
            print(f"[skip] Could not parse {fname}")
            continue

        n = parsed["n"]
        if (n < 350) or n > 1000:
            continue
      

        print(f"\n{'='*50}")
        print(f"Instance: {instance_name} (n={n})")
        print(f"{'='*50}")
        
        try:
           
            res_list = evaluate_instance_parallel(instance_name, full_tsp_path)
            rows.extend(res_list)
                

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    if not rows:
        print("\n❌ No instances evaluated.")
        return

    write_solver_results(rows)
    print("\n✅ Results written to solver_results/")

    print(f"\n✅ Saved results to {args.output_csv}")

   

if __name__ == "__main__":
    main()
