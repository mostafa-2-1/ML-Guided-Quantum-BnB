import os

# ------------------------
# CONFIG
# ------------------------
root_dir = os.getcwd()

# Baseline pruned graphs (ALREADY GENERATED)
baseline_pruned_dir = os.path.join(root_dir, "pruned_graphs_k_baseline")

# Output parameter + results dirs
params_dir = os.path.join(root_dir, "baseline_parameters")
results_dir = os.path.join(root_dir, "baseline_lkh_results")

# LKH executable
lkh_exe_path = r"C:\LKH\LKH.exe"  # adjust if needed

# K values
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25]

# Baseline types
baseline_types = ["nearest_k", "random_k"]

# ------------------------
# CREATE DIRS
# ------------------------
os.makedirs(params_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# ------------------------
# GENERATE .par FILES
# ------------------------
print("\n" + "=" * 60)
print("Generating LKH parameters for BASELINE pruned graphs")
print("=" * 60)

for baseline_type in baseline_types:
    print(f"\nBaseline: {baseline_type}")
    baseline_type_dir = os.path.join(baseline_pruned_dir, baseline_type)

    if not os.path.exists(baseline_type_dir):
        print(f"  ⚠️  Missing directory: {baseline_type_dir}")
        continue

    for k in k_values:
        k_dir = os.path.join(baseline_type_dir, f"k{k}")

        if not os.path.exists(k_dir):
            print(f"  ⚠️  Missing k={k} directory")
            continue

        # Output dirs
        param_k_dir = os.path.join(params_dir, baseline_type, f"k{k}")
        result_k_dir = os.path.join(results_dir, baseline_type, f"k{k}")
        os.makedirs(param_k_dir, exist_ok=True)
        os.makedirs(result_k_dir, exist_ok=True)

        tsp_files = [f for f in os.listdir(k_dir) if f.endswith(".tsp")]

        print(f"  k={k}: {len(tsp_files)} instances")

        for tsp_file in tsp_files:
            instance_name = tsp_file.replace("_sparse.tsp", "")
            tsp_path = os.path.join(k_dir, tsp_file)

            par_path = os.path.join(param_k_dir, f"{instance_name}.par")
            sol_path = os.path.join(result_k_dir, f"{instance_name}.sol")

            with open(par_path, "w") as f:
                f.write(f"PROBLEM_FILE = {tsp_path}\n")
                f.write(f"OUTPUT_TOUR_FILE = {sol_path}\n")
                f.write("RUNS = 1\n")
                f.write("MOVE_TYPE = 5\n")
                f.write("PRECISION = 1\n")
                f.write("TIME_LIMIT = 3600\n")

        print(f"  ✓ Parameters generated for {baseline_type} k={k}")

print("\n✓ Baseline LKH parameter generation complete.")

# ------------------------
# CREATE BATCH FILES
# ------------------------
print("\n" + "=" * 60)
print("Creating batch scripts")
print("=" * 60)

for baseline_type in baseline_types:
    for k in k_values:
        param_k_dir = os.path.join(params_dir, baseline_type, f"k{k}")

        if not os.path.exists(param_k_dir):
            continue

        batch_path = os.path.join(params_dir, f"run_{baseline_type}_k{k}.bat")

        with open(batch_path, "w") as f:
            f.write("@echo off\n")
            f.write(f"cd {root_dir}\n")
            f.write(f"echo Running {baseline_type} baseline, k={k}...\n\n")

            for par_file in os.listdir(param_k_dir):
                if par_file.endswith(".par"):
                    par_path = os.path.join(param_k_dir, par_file)
                    f.write(f'"{lkh_exe_path}" "{par_path}"\n')

            f.write("\necho Done.\n")
            f.write("pause\n")

        print(f"✓ Created {batch_path}")

print("\nAll baseline batch files created successfully.")
