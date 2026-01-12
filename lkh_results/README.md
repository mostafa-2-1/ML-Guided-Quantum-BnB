# LKH Experiment Results

## Directory Structure
```
results/
|-- full/               # Full graph solutions
|-- k2/                 # k=2 solutions
|-- k3/                 # k=3 solutions
|-- k4/                 # k=4 solutions
|-- k5/                 # k=5 solutions
|-- k6/                 # k=6 solutions
|-- k7/                 # k=7 solutions
|-- k8/                 # k=8 solutions
|-- k10/                 # k=10 solutions
|-- k12/                 # k=12 solutions
|-- k15/                 # k=15 solutions
|-- k20/                 # k=20 solutions
|-- k25/                 # k=25 solutions
|-- results_summary.csv # Summary of all results
`-- README.md           # This file
```

## Running Experiments
1. Run full graphs: `parameters/run_full.bat`
2. Run specific k: `parameters/run_kX.bat` where X is k value
3. Run all: `parameters/run_all.bat`

## Expected Output
- `.sol` files: TSP tours
- `.par` files: LKH parameters
- Summary CSV with metrics

## K Values Tested
2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25
