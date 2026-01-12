import os
import glob

def parse_tsp_dimension(file_path):
    """Extract DIMENSION value from .tsp file"""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('DIMENSION'):
                    # Handle different formats: "DIMENSION : 10" or "DIMENSION: 10"
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            return int(parts[1].strip())
                    # Alternative: split by space and take last part
                    parts = line.split()
                    if len(parts) > 1:
                        return int(parts[-1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return None

def analyze_tsp_folders():
    folders = ['synthetic_tsplib', 'tsplib_data']
    ranges = [
        (1, 20, "1-20"),
        (21, 50, "21-50"), 
        (51, 100, "51-100"),
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, 2000, "1001-2000"),
        (2001, float('inf'), "2000+")
    ]
    
    # Initialize counters
    range_counts = {range_name: 0 for _, _, range_name in ranges}
    folder_totals = {folder: 0 for folder in folders}
    all_files_info = []
    
    print("Analyzing TSP files...\n")
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"⚠️  Folder '{folder}' not found, skipping...")
            continue
            
        print(f"📁 Scanning {folder}:")
        tsp_files = glob.glob(os.path.join(folder, "*.tsp"))
        print(f"   Found {len(tsp_files)} .tsp files")
        
        for tsp_file in tsp_files:
            dimension = parse_tsp_dimension(tsp_file)
            if dimension is not None:
                all_files_info.append((tsp_file, dimension))
                folder_totals[folder] += 1
                
                # Categorize into ranges
                for min_nodes, max_nodes, range_name in ranges:
                    if min_nodes <= dimension <= max_nodes:
                        range_counts[range_name] += 1
                        break
            else:
                print(f"   ⚠️  Could not read dimension from: {os.path.basename(tsp_file)}")
    
    # Print results
    print("\n" + "="*50)
    print("📊 NODE COUNT DISTRIBUTION")
    print("="*50)
    
    total_files = sum(range_counts.values())
    print(f"Total TSP files analyzed: {total_files}\n")
    
    for range_name in [r[2] for r in ranges]:
        count = range_counts[range_name]
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"Nodes {range_name:8}: {count:3} files ({percentage:5.1f}%)")
    
    print("\n" + "="*50)
    print("📁 FOLDER BREAKDOWN")
    print("="*50)
    for folder, count in folder_totals.items():
        if os.path.exists(folder):
            print(f"{folder:15}: {count:3} files")
    
    # Show some file examples
    print("\n" + "="*50)
    print("🔍 SAMPLE FILES (first 10)")
    print("="*50)
    for i, (file_path, dimension) in enumerate(all_files_info[:10]):
        print(f"{os.path.basename(file_path):30} → {dimension:4} nodes")
    
    return range_counts, all_files_info

if __name__ == "__main__":
    range_counts, all_files_info = analyze_tsp_folders()
    
    # Optional: Save detailed report
    with open("tsp_analysis_report.txt", "w") as f:
        f.write("TSP Files Analysis Report\n")
        f.write("=" * 30 + "\n\n")
        f.write("Node Range Distribution:\n")
        for range_name, count in range_counts.items():
            f.write(f"Nodes {range_name}: {count} files\n")
        f.write(f"\nTotal files: {sum(range_counts.values())}\n")
        f.write("\nDetailed file list:\n")
        for file_path, dimension in all_files_info:
            f.write(f"{os.path.basename(file_path)} - {dimension} nodes\n")
    
    print(f"\n📄 Detailed report saved to: tsp_analysis_report.txt")