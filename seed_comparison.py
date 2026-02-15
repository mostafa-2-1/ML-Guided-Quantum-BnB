import os
import pandas as pd

BASE_DIR = "solver_results"
SEEDS = ["seed1", "seed2", "seed3"]

def collect_success_rates():
    records = []

    for seed in SEEDS:
        seed_dir = os.path.join(BASE_DIR, seed)
        for k_folder in sorted(os.listdir(seed_dir)):
            # Skip full graphs
            if k_folder.lower() == "full":
                continue

            k_dir = os.path.join(seed_dir, k_folder)
            results_file = os.path.join(k_dir, "results.csv")

            if not os.path.isfile(results_file):
                continue

            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip().str.lower()

            if "success" not in df.columns:
                raise ValueError(
                    f"'success' column not found in {results_file}. Columns found: {df.columns.tolist()}"
                )

            success_rate = df["success"].astype(bool).mean() * 100.0

            records.append({
                "seed": seed,
                "k": k_folder,
                "success_rate_percent": round(success_rate, 2)
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = collect_success_rates()
    print(df)

    # Optional pivoted view
    pivot = df.pivot(index="k", columns="seed", values="success_rate_percent")
    print("\nSuccess Rate Comparison Across Seeds (%):\n")
    print(pivot)

    pivot.to_csv("seed_success_comparison.csv")
