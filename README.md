# SmoothE: Differentiable E-Graph Extraction
This repository contains scripts for setting up environments and reproducing results presented in the ASPLOS 2025 paper entitled _SmoothE: Differentiable E-Graph Extraction_.

## Setup

### Requirements (Est. Time: 30 mins)
Set up the necessar Python environment using conda: 
```
conda env create -f env.yaml
```
We used CUDA 11.7 for our experiments.


### CPLEX (optional) (Est. Time: 20 mins)
Install [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio).
You can follow the instructions at [CPLEX Education Intallation Tutorial](https://github.com/academic-initiative/documentation/blob/main/academic-initiative/how-to/How-to-download-IBM-ILOG-CPLEX/readme.md).


## Run Experiments

### Run SmoothE (Est. Time: 1 hr)
A machine equipped with an A100 GPU is highly recommended for this experiment.
```
bash run_smoothe.sh
```

### Run Heuristic Baselines (Est. Time: 5 mins)
```
python run_greedy_script.py
```

### Run ILP Baselines (Est. Time: 12 hrs) (optional)
We provide the resulting log files for this experiment in `./logs/`.
If AE skips this experiment, the provided log files will be automatically used to generate the tables and figures.
AE can also choose to skip the CPLEX expriment by skipping the third command. 

Note that the LIP performance is not deterministic, and the results may vary slightly from the provided log files due to the status of the machine.

Alternatively, if the AE wants to re-run the experiment, they can use the following command, and the resulting log files will overwrite the provided log files.
```
python launch.py --acyclic --dataset all --repeat 1 --method cplex 
python launch.py --acyclic --dataset all --repeat 1 --method cbc 
python launch.py --acyclic --dataset all --repeat 1 --method scip 
```

### Run Oracle Baselines (Est. Time: 3.5 days) (optional)
This experiments takes extremely long time to run because the time out is set to 10 hours for CPLEX to solve the ILP problem.
We provide the resulting log files for this experiment in `./logs/`.
If AE skips this experiment, the provided log files will be automatically used to generate the tables and figures.
Alternatively, if the AE wants to re-run the experiment, they can use the following command, and the resulting log files will overwrite the provided log files.
```
python launch.py --acyclic --dataset all --repeat 1 --method oracle 
```

### Run Genetic Baselines (Est. Time: 4 hours) (optional)
Run genetic algorithm on all the datasets for three times to report mean and standard deviation.
The whole experiment will take around 4 hours.
```
python run_genetic.py
 ```


## Generate Results (Instananeous)

### Tables
```
python table.py
```
This command will create **Table 2**, **Table 3**, and **Table 4** in `./table.md`.

### Figures
```
python figure.py
```
This command will create **Figure 4.pdf** and **Figure 6.pdf** in `./fig`.