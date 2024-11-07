#!/usr/bin/env python3

import os
import subprocess
import glob
from pathlib import Path


def run_genetic_script(input_file, params, cost_type, pkl_file):
    command = ["python", "genetic.py", "--input_file", input_file]

    if input_file.endswith(".dot"):
        command.append("--load_cost")
    command.append(f"--{cost_type}")
    command.append(pkl_file)

    # Split params into a list while preserving quoted strings
    params_list = params.split()
    command.extend(params_list)

    try:
        print(f"Executing: {' '.join(command)} for 1 times")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running genetic.py: {e}")


def run(exp_id):
    # Define the list of datasets
    datasets = ["rover", "flexc", "tensat", "diospyros", "impress"]

    # Define fixed parameters
    fix_params = "--num_of_generations 5000000 --choose_prob 0.4 --time_limit 60"

    # Define dataset-specific parameters
    dataset_params = {
        "rover": {
            "num_of_paths": 100,
            "num_of_tour_particips": 20
        },
        "flexc": {
            "num_of_paths": 50,
            "num_of_tour_particips": 15
        },
        "tensat": {
            "num_of_paths": 30,
            "num_of_tour_particips": 10
        },
        "diospyros": {
            "num_of_paths": 30,
            "num_of_tour_particips": 10
        },
        "impress": {
            "num_of_paths": 20,
            "num_of_tour_particips": 8
        },
    }

    for data_set in datasets:
        # Retrieve parameters for the current dataset
        params = fix_params
        ds_params = dataset_params.get(data_set, {})
        if not ds_params:
            print(f"Unknown dataset: {data_set}. Skipping...")
            continue

        params += f" --num_of_paths {ds_params['num_of_paths']} --num_of_tour_particips {ds_params['num_of_tour_particips']}"

        print(f"Running {data_set} with genetic")

        # Define the dataset directory
        dataset_dir = Path(f'dataset/{data_set}')

        nonlinear_cost_dir = Path(f'nonlinear_cost/dataset/{data_set}')

        # Process .json files
        json_files = list(dataset_dir.glob("*.json"))
        if json_files:
            print(f"Processing {len(json_files)} json files in {data_set}")
            for file_path in json_files:
                base_name = file_path.stem
                pkl_quad = nonlinear_cost_dir / f"{base_name}_quad_cost.pkl"
                pkl_mlp = nonlinear_cost_dir / f"{base_name}_mlp_cost.pkl"

                # Run quadratic_cost model
                run_genetic_script(input_file=str(file_path),
                                   params=params,
                                   cost_type="quad_cost",
                                   pkl_file=str(pkl_quad))

                # Run mlp_cost model
                run_genetic_script(input_file=str(file_path),
                                   params=params,
                                   cost_type="mlp_cost",
                                   pkl_file=str(pkl_mlp))
        # Process .dot files
        dot_files = list(dataset_dir.glob("*.dot"))
        if dot_files:
            print(f"Processing {len(dot_files)} dot files in {data_set}")
            for file_path in dot_files:
                base_name = file_path.stem
                pkl_quad = nonlinear_cost_dir / f"{base_name}_quad_cost.pkl"
                pkl_mlp = nonlinear_cost_dir / f"{base_name}_mlp_cost.pkl"

                # Run quadratic_cost model
                run_genetic_script(input_file=str(file_path),
                                   params=params,
                                   cost_type="quad_cost",
                                   pkl_file=str(pkl_quad))

                # Run mlp_cost model
                run_genetic_script(input_file=str(file_path),
                                   params=params,
                                   cost_type="mlp_cost",
                                   pkl_file=str(pkl_mlp))

        print("-" * 50)  # Separator for readability


if __name__ == "__main__":
    for i in range(3):
        run(i)
