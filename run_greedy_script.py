import os
import subprocess


def run_command(file_path, method):
    command = f"python dag_greedy.py --input_file {file_path} --method {method}"
    command += " --load_cost" if file_path.endswith("dot") else ""
    subprocess.run(command, shell=True)


def main():
    folder_path = "dataset/"
    for dataset_path in os.listdir(folder_path):
        print(f'Running heuristic for {dataset_path}')
        for file_name in os.listdir(os.path.join(folder_path, dataset_path)):
            if file_name.endswith('.json') or file_name.endswith('.dot'):
                file_path = os.path.join(folder_path, dataset_path, file_name)
                for method in ["faster", "baseline"]:
                    run_command(file_path, method)


if __name__ == "__main__":
    main()
