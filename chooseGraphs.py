import os
import random
import shutil
from newTrain import parse_tsp

SOURCE_FOLDERS = ["tsplib_data", "synthetic_tsplib"]
OUT_DIR = "graphs_chosen"
MIN_N = 10
MAX_N = 25
NUM_GRAPHS = 60
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

candidates = []

print("[Scanning graphs...]")

for folder in SOURCE_FOLDERS:
    for fname in os.listdir(folder):
        if not fname.endswith(".tsp"):
            continue

        path = os.path.join(folder, fname)

        try:
            tsp = parse_tsp(path)
            n = tsp["coords"].shape[0]
        except Exception:
            continue

        if MIN_N <= n <= MAX_N:
            candidates.append((path, folder, fname, n))

print(f"Found {len(candidates)} candidate graphs")

if len(candidates) < NUM_GRAPHS:
    raise RuntimeError("Not enough graphs in the desired size range")

selected = random.sample(candidates, NUM_GRAPHS)

metadata = []

for src_path, folder, fname, n in selected:
    dst_path = os.path.join(OUT_DIR, fname)
    shutil.copy(src_path, dst_path)

    metadata.append({
        "name": fname.replace(".tsp", ""),
        "n": n,
        "source": folder
    })

with open(os.path.join(OUT_DIR, "selection_metadata.json"), "w") as f:
    import json
    json.dump(metadata, f, indent=2)

print("✅ Graph selection complete")
print(f"Saved {NUM_GRAPHS} graphs to `{OUT_DIR}/`")
