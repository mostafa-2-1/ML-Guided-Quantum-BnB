import os
import time
import numpy as np
from itertools import permutations

# Output folder
OUTPUT_FOLDER = "synthetic_tsplib"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- UTILS ----------
def save_tsp_file(name, coords):
    n = len(coords)
    filepath = os.path.join(OUTPUT_FOLDER, f"{name}.tsp")
    with open(filepath, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write(f"TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, 1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")
    return filepath

def save_tour_file(name, tour):
    n = len(tour)
    filepath = os.path.join(OUTPUT_FOLDER, f"{name}.opt.tour")
    with open(filepath, "w") as f:
        f.write(f"NAME : {name}.opt.tour\n")
        f.write(f"TYPE : TOUR\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("TOUR_SECTION\n")
        for node in tour:
            f.write(f"{node}\n")
        f.write("-1\nEOF\n")
    return filepath

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def total_tour_length(tour, coords):
    return sum(
        euclidean_distance(coords[tour[i] - 1], coords[tour[i + 1] - 1])
        for i in range(len(tour) - 1)
    ) + euclidean_distance(coords[tour[-1] - 1], coords[tour[0] - 1])

# ---------- SOLVERS ----------
def solve_exact(coords):
    n = len(coords)
    best_length = float('inf')
    best_tour = None
    for perm in permutations(range(1, n + 1)):
        length = total_tour_length(perm, coords)
        if length < best_length:
            best_length = length
            best_tour = perm
    return list(best_tour)

def solve_heuristic(coords):
    n = len(coords)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    while unvisited:
        last = tour[-1]
        next_node = min(unvisited, key=lambda x: euclidean_distance(coords[last], coords[x]))
        tour.append(next_node)
        unvisited.remove(next_node)
    return [i + 1 for i in tour]

# ---------- MAIN GENERATOR ----------
def generate_instance(name, n_nodes):
    print(f"\n Generating TSP instance '{name}' with {n_nodes} nodes...")
    start_time = time.time()

    max_coord = np.random.choice([5000, 10000, 20000])
    coords = np.random.randint(0, max_coord, size=(n_nodes, 2)).astype(float)

    # --- Augmentations ---
    if np.random.rand() < 0.5:  # random rotation
        theta = np.random.uniform(0, 2*np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        coords = coords @ R.T

    if np.random.rand() < 0.5:  # random scaling
        scale = np.random.uniform(0.8, 1.2)
        coords *= scale

    if np.random.rand() < 0.5:  # random Gaussia njitter
        coords += np.random.normal(0, 50, coords.shape)  # small coordinate noise

    tsp_file = save_tsp_file(name, coords)

    if n_nodes <= 10:
        print(f"  Solving EXACTLY for {n_nodes} nodes (this might take a sec)...")
        tour = solve_exact(coords)
    else:
        print(f" Using HEURISTIC for {n_nodes} nodes...")
        tour = solve_heuristic(coords)

    tour_file = save_tour_file(name, tour)
    elapsed = time.time() - start_time
    print(f" Finished {name}: {n_nodes} nodes | Time: {elapsed:.2f}s")
    return tsp_file, tour_file

# ---------- RUN ----------
if __name__ == "__main__":
    np.random.seed(42)  
    start_index = 1201 # DONT WORRY ABT THIS, I WAS JUST SPECIFING THE NUMBER TO HELP WITH NAMING    
    num_new = 300        

    # Generate variable sizes from 2 to 500+ nodes
    small = list(range(2, 21))
    medium = list(range(25, 101))
    large = list(range(102, 551))
    all_sizes = small + medium + large

    np.random.shuffle(all_sizes)

    sizes = []
    while len(sizes) < num_new:
        sizes.extend(all_sizes)
        np.random.shuffle(all_sizes)  # reshuffle each time
    sizes = sizes[:num_new]  # trim to exactly num_new
    total_start = time.time()
    for i, n in enumerate(sizes):
        idx = start_index + i
        generate_instance(f"synthetic_{idx}", n)

    total_time = time.time() - total_start
    print("\n Generation complete!")
    print(f" Total instances: {num_new}")
    print(f"  Total time: {total_time/60:.2f} minutes")
