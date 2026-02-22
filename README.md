Overview This project provides a systematic empirical analysis of how graph sparsification affects heuristic solver reliability in large-scale Euclidean Traveling Salesman Problems (**TSP**).

Rather than proposing a new solver, this work studies how different pruning strategies reshape the structural landscape on which a fixed state-of-the-art heuristic (**LKH**-3.0.13) operates.

We compare three pruning strategies:
- Learned pruning via a pretrained **GNN**
- Geometric nearest-k pruning
- Random-k pruning
Across fourteen benchmark instances and a controlled sparsity sweep, we model solver success probability as a function of realized average graph degree and identify phase transitions in navigability.

### Core Research Question

Is the observed success of learned pruning a consequence of structural information preservation, or merely a byproduct of sparsity?

### Experimental Scope

14 large-scale **TSPLIB** and **VLSI** benchmark instances 15 sparsity levels per instance 3 pruning strategies 10 **LKH** seeds per reduced graph

Total:
- **630** reduced graphs
- 6,**300** solver runs

### Methodological Pipeline

The experimental pipeline consists of four stages:

## Reduced Graph Construction

**GNN**-based top-k edge retention Nearest-k baseline Random-k baseline ## Structural Measurement Realized average degree Connectivity Tour recall Edge-length statistics ## Solver Evaluation **LKH**-3.0.13 10 independent seeds Success defined as relative gap ≤ 0.7% ## Statistical Modeling Logistic regression of P(success) vs average degree Bootstrap confidence intervals Connectivity decomposition Each reduced graph (instance × method × k) is treated as an independent structural observation.

### Setup Instructions

## Clone the repository Bash

git clone [https://github.com/mostafa-2-1/ML-Guided-Quantum-BnB.git](https://github.com/mostafa-2-1/ML-Guided-Quantum-BnB.git)

## Create a virtual environment

Bash

python -m venv venv venv\Scripts\activate  # Windows # source venv/bin/activate  # Linux/Mac ## Install dependencies Bash

pip install -r requirements.txt If missing:

Bash

pipreqs . --force --encoding=utf-8 ## External Dependency This project requires:

**LKH**-3.0.13 Download from the official **LKH** repository and compile locally.

⚠️ Solver paths must be configured in the experiment scripts.

### Reproducing Results

## Generate synthetic instances scripts/synthetic_tsp.py

## Process data

scripts/preprocess_tsp.py

## Train the GNN model

newTrain.py

## Prune graphs

4.1. **GNN** pruning gnn_pruning.py 4.2. Baseline pruning baseline_pruning.py

Results are generated in
- pruned_graphs_k (**GNN**)
- pruned_graphs_k_baseline (baselines)

## Generate parameters for LKH

5.1. **GNN** parameters gnn_parameters.py 5.2. Baseline parameters baseline_parameters.py

Results are generated in
- parameters (**GNN**)
- baseline_parameters (baselines)

## Solve reduced graphs using LKH

6.1. **GNN** solver endSolvers/exact_gnn.py 6.2. Baseline solver endSolvers/exact_baseline.py

Results are generated in
- solver_results (**GNN**)
- baseline_solver_results (baselines)

## Figures generation (success vs avg degree, gap vs k, success probability overlay)

generate_comparison_plots.py

Results are saved in
- structural_analysis_comparison
    - success_vs_avg_degree_overlay.png
    -  gap_vs_k_gnn_vs_knn.png
    - success_avg_degree_logistic_overlay.png

## Logistic parameters table and critical k distribution 

generate_combined_tables.py

Results are saved in
- table_comparison
    - critical_k_distribution.png
    - logistic_parameters_table.png

## Master logistic phase transition fit and logistic regression covariate table

generate_master_phase_transition_plot.py

Results are saved in
- structural_analysis_comparison
    - master_phase_transition_comparison.png
    - logistic_regression_covariate_table.png

## Connectivity decomposition figure

connectivity_decomposition_analysis.py

Results are saved in
- structural_analysis_comparison_connectivity
  - connectivity_decomposition.png

Statistical and quantitative outputs are generated with each file to provide more details.
