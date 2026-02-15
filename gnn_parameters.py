
import os

# ------------------------
# Paths
# ------------------------
root_dir = os.getcwd()
full_dir = os.path.join(root_dir, "graphss_chosen")          # full TSP instances
pruned_base_dir = os.path.join(root_dir, "pruned_graphs_k")  # base directory for k-sweep
params_dir = os.path.join(root_dir, "parameters")
lkh_exe_path = r"C:\LKH\LKH.exe"  # adjust if needed
results_dir = os.path.join(root_dir, "lkh_results")

# K values to sweep
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 22, 25]

# Create directory structure
os.makedirs(params_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Create full parameters directory
full_params_dir = os.path.join(params_dir, "full")
os.makedirs(full_params_dir, exist_ok=True)

# Create results directories
full_results_dir = os.path.join(results_dir, "full")
os.makedirs(full_results_dir, exist_ok=True)

# Create directories for each k
for k in k_values:
    k_params_dir = os.path.join(params_dir, f"k{k}")
    os.makedirs(k_params_dir, exist_ok=True)
    
    k_results_dir = os.path.join(results_dir, f"k{k}")
    os.makedirs(k_results_dir, exist_ok=True)


def get_tsp_node_count(tsp_path):
    """Quickly parse TSP file to get number of nodes."""
    try:
        with open(tsp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.upper().startswith('DIMENSION'):
                    # Extract dimension value
                    parts = line.split(':')
                    if len(parts) >= 2:
                        return int(parts[1].strip())
                    # Alternative format: DIMENSION 666
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1].strip())
        return None
    except Exception as e:
        print(f"  Error reading {tsp_path}: {e}")
        return None

# ------------------------
# Get list of instances (from full directory)
# ------------------------
instance_names = []
for filename in os.listdir(full_dir):
    if filename.endswith('.tsp'):
        instance_name = os.path.splitext(filename)[0]
        full_tsp_path = os.path.join(full_dir, filename)
        n = get_tsp_node_count(full_tsp_path)
        if n is None:
            print(f"  ⚠️  Could not read dimension from {filename}")
            continue

        if n >= 350:
            instance_names.append(instance_name)
            print(f"  ✓ {instance_name}: {n} nodes (included)")
        else:
            print(f"  ✗ {instance_name}: {n} nodes (skipped)")

print(f"Found {len(instance_names)} instances")

# ------------------------
# Process full graphs
# ------------------------
print("\n" + "="*60)
print("Generating FULL graph parameters")
print("="*60)

for instance_name in instance_names:
    full_tsp_path = os.path.join(full_dir, f"{instance_name}.tsp")
    
    if not os.path.exists(full_tsp_path):
        print(f"⚠️  Full TSP not found for {instance_name}, skipping")
        continue
    
    # Create full .par file
    sol_file_full = os.path.join(full_results_dir, f"{instance_name}.sol")
    par_file_full = os.path.join(full_params_dir, f"{instance_name}.par")
    
    with open(par_file_full, "w") as f:
        f.write(f"PROBLEM_FILE = {full_tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {sol_file_full}\n")
        f.write("RUNS = 1\n")
        #f.write("MAX_CANDIDATES = 5\n")  # Could be lower for full graphs
        f.write("MOVE_TYPE = 5\n")
        f.write("PRECISION = 1\n")
        f.write("TIME_LIMIT = 3600\n")  # 1 hour time limit
    
    print(f"✓ Created full .par for {instance_name}")

# ------------------------
# Process pruned graphs for each k
# ------------------------
print("\n" + "="*60)
print("Generating PRUNED graph parameters")
print("="*60)

for k in k_values:
    print(f"\nProcessing k={k}")
    print("-" * 40)
    
    k_pruned_dir = os.path.join(pruned_base_dir, f"k{k}")
    
    # Check if k directory exists
    if not os.path.exists(k_pruned_dir):
        print(f"⚠️  Directory for k={k} not found: {k_pruned_dir}")
        continue
    
    # Get list of instance folders for this k
    instance_folders = []
    for item in os.listdir(k_pruned_dir):
        item_path = os.path.join(k_pruned_dir, item)
        if os.path.isdir(item_path):
            instance_folders.append(item)
    
    print(f"  Found {len(instance_folders)} instance folders for k={k}")
    
    for instance_name in instance_folders:
        instance_dir = os.path.join(k_pruned_dir, instance_name)
        
        # Look for the sparse TSP file
        sparse_tsp_path = os.path.join(instance_dir, f"{instance_name}_sparse.tsp")
        
        if not os.path.exists(sparse_tsp_path):
            # Try alternative naming
            alt_path = os.path.join(instance_dir, f"{instance_name}_concorde.tsp")
            if os.path.exists(alt_path):
                sparse_tsp_path = alt_path
            else:
                # Try to find any .tsp file in the directory
                for file in os.listdir(instance_dir):
                    if file.endswith('.tsp'):
                        sparse_tsp_path = os.path.join(instance_dir, file)
                        break
        
        if not os.path.exists(sparse_tsp_path):
            print(f"  ⚠️  No TSP file found for {instance_name} in k={k}")
            continue
        
        # Create pruned .par file
        sol_file_pruned = os.path.join(results_dir, f"k{k}", f"{instance_name}.sol")
        par_file_pruned = os.path.join(params_dir, f"k{k}", f"{instance_name}.par")
        
        with open(par_file_pruned, "w") as f:
            f.write(f"PROBLEM_FILE = {sparse_tsp_path}\n")
            f.write(f"OUTPUT_TOUR_FILE = {sol_file_pruned}\n")
            f.write("RUNS = 1\n")
            #f.write(f"MAX_CANDIDATES = {k}\n")  # Use k as max candidates
            f.write("MOVE_TYPE = 5\n")
            f.write("PRECISION = 1\n")
            f.write("TIME_LIMIT = 3600\n")  # 1 hour time limit
        
        print(f"  ✓ Created .par for {instance_name} (k={k})")

# ------------------------
# Create batch scripts for running LKH
# ------------------------
print("\n" + "="*60)
print("Creating batch scripts")
print("="*60)

# Create a batch file for full graphs
batch_full_path = os.path.join(params_dir, "run_full.bat")
with open(batch_full_path, "w") as f:
    f.write("@echo off\n")
    f.write(f"cd {root_dir}\n")
    f.write("echo Running LKH on FULL graphs...\n")
    
    for instance_name in instance_names:
        par_file = os.path.join(full_params_dir, f"{instance_name}.par")
        if os.path.exists(par_file):
            f.write(f'echo Processing {instance_name}...\n')
            f.write(f'"{lkh_exe_path}" {par_file}\n')
    
    f.write("echo Finished FULL graphs.\n")
    f.write("pause\n")

print(f"✓ Created batch file for full graphs: {batch_full_path}")

# Create batch files for each k
for k in k_values:
    batch_k_path = os.path.join(params_dir, f"run_k{k}.bat")
    k_params_subdir = os.path.join(params_dir, f"k{k}")
    
    if not os.path.exists(k_params_subdir):
        continue
    
    with open(batch_k_path, "w") as f:
        f.write("@echo off\n")
        f.write(f"cd {root_dir}\n")
        f.write(f"echo Running LKH on k={k} graphs...\n")
        
        # Get all .par files for this k
        for par_file in os.listdir(k_params_subdir):
            if par_file.endswith('.par'):
                instance_name = os.path.splitext(par_file)[0]
                full_par_path = os.path.join(k_params_subdir, par_file)
                f.write(f'echo Processing {instance_name} (k={k})...\n')
                f.write(f'"{lkh_exe_path}" {full_par_path}\n')
        
        f.write(f"echo Finished k={k} graphs.\n")
        f.write("pause\n")
    
    print(f"✓ Created batch file for k={k}: {batch_k_path}")

# Create a master batch file to run everything
batch_master_path = os.path.join(params_dir, "run_all.bat")
with open(batch_master_path, "w") as f:
    f.write("@echo off\n")
    f.write(f"cd {root_dir}\n")
    f.write("echo Running ALL LKH experiments...\n")
    f.write("\n")
    f.write("echo 1. Running FULL graphs...\n")
    f.write(f'call "{batch_full_path}"\n')
    f.write("\n")
    
    for k in k_values:
        batch_k_path = os.path.join(params_dir, f"run_k{k}.bat")
        if os.path.exists(batch_k_path):
            f.write(f"echo 2.{k}. Running k={k} graphs...\n")
            f.write(f'call "{batch_k_path}"\n')
            f.write("\n")
    
    f.write("echo ALL experiments completed!\n")
    f.write("pause\n")

print(f"✓ Created master batch file: {batch_master_path}")

# ------------------------
# Create summary structure for results
# ------------------------
print("\n" + "="*60)
print("Creating results summary structure")
print("="*60)

# Create CSV template for results
results_csv_path = os.path.join(results_dir, "results_summary.csv")
with open(results_csv_path, "w") as f:
    f.write("instance,n_nodes,k,graph_type,edges_kept,avg_degree,min_degree,sparsity,tour_cost,runtime_seconds,success_flag,gap_percent,timestamp\n")

print(f"✓ Created results summary template: {results_csv_path}")

# Create README with ASCII characters only
readme_path = os.path.join(results_dir, "README.md")
with open(readme_path, "w", encoding='utf-8') as f:
    f.write("# LKH Experiment Results\n\n")
    f.write("## Directory Structure\n")
    f.write("```\n")
    f.write("results/\n")
    f.write("|-- full/               # Full graph solutions\n")
    for k in k_values:
        if os.path.exists(os.path.join(results_dir, f"k{k}")):
            f.write(f"|-- k{k}/                 # k={k} solutions\n")
    f.write("|-- results_summary.csv # Summary of all results\n")
    f.write("`-- README.md           # This file\n")
    f.write("```\n\n")
    f.write("## Running Experiments\n")
    f.write("1. Run full graphs: `parameters/run_full.bat`\n")
    f.write("2. Run specific k: `parameters/run_kX.bat` where X is k value\n")
    f.write("3. Run all: `parameters/run_all.bat`\n\n")
    f.write("## Expected Output\n")
    f.write("- `.sol` files: TSP tours\n")
    f.write("- `.par` files: LKH parameters\n")
    f.write("- Summary CSV with metrics\n\n")
    f.write("## K Values Tested\n")
    f.write(", ".join(map(str, k_values)) + "\n")

print(f"Created README: {readme_path}")

# ------------------------
# Summary
# ------------------------
print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)

#total_pruned = sum(pruned_counts.values())
#print(f"\nGenerated:")
#print(f"  - Full graph parameters: {full_count}")
#print(f"  - Pruned graph parameters: {total_pruned}")

print(f"\nDirectory structure:")
print(f"  parameters/")
#print(f"  |-- full/              # Full graph parameters ({full_count} files)")
for k in k_values:
    if os.path.exists(os.path.join(params_dir, f'k{k}')):
        count = len([f for f in os.listdir(os.path.join(params_dir, f'k{k}')) if f.endswith('.par')])
        if count > 0:
            print(f"  |-- k{k}/              # k={k} parameters ({count} files)")
print(f"  |-- run_full.bat       # Run full graphs")
print(f"  |-- run_all.bat        # Run everything")
for k in k_values:
    if os.path.exists(os.path.join(params_dir, f'run_k{k}.bat')):
        print(f"  |-- run_k{k}.bat       # Run k={k} graphs")

print(f"\n  results/")
print(f"  |-- full/              # Full graph results")
for k in k_values:
    if os.path.exists(os.path.join(results_dir, f'k{k}')):
        print(f"  |-- k{k}/              # k={k} results")
print(f"  |-- results_summary.csv")
print(f"  `-- README.md")

print(f"\nTo run experiments:")
print(f"  1. Open parameters/run_all.bat (to run everything)")
print(f"  2. Or run individual batch files")
print(f"\nNote: Results will be saved to results/ directory")