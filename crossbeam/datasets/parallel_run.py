#!/usr/bin/env python3

import os
import subprocess
import concurrent.futures

def generate_data(index):
    pid = os.getpid()  # Get the process ID
    split = "valid" if index % 2 == 0 else "train"
    
    out_dir = os.path.join(
        data_dir,
        f"t-{tout}-maxne-{maxne}-maxni-{maxni}-skip-{skip}-lambdaskip-{lambdaskip}-lambdafrac-{lambda_fraction}-shuffleops-{shuffle_ops}-pid-{pid}-{split}",
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Generating {split} in PID {pid}")
    subprocess.run(
        [
            "python3",
            "-m",
            "crossbeam.datasets.bottom_up_data_generation",
            "--domain=deepcoder",
            "--data_save_dir=" + out_dir,
            f"--split={split}",
            "--data_gen_seed=10000" if split == "valid" else "--data_gen_seed=0",
            "--data_gen_timeout=" + str(tout),
            "--num_tasks_per_weight=10" if split == "valid" else "--num_tasks_per_weight=100",
            "--num_searches=5",
            "--min_task_weight=3",
            "--max_task_weight=" + str(maxw),
            "--min_num_examples=5" if split == "valid" else f"--min_num_examples={maxne}",
            "--max_num_examples=" + str(maxne),
            "--min_num_inputs=1",
            "--max_num_inputs=" + str(maxni),
            "--skip_probability=" + str(skip),
            "--lambda_skip_probability=" + str(lambdaskip),
            "--lambda_fraction=" + str(lambda_fraction),
            "--shuffle_ops=" + str(shuffle_ops),
            "--num_datagen_proc=" + str(1),
            "--verbose=False",
        ]
    )

if __name__ == "__main__":
    data_dir = "./data/"
    tout = 3600
    maxw = 12
    maxne = 5
    maxni = 1
    skip = 0.0
    lambdaskip = 0.0
    lambda_fraction = 0.8
    shuffle_ops = False
    num_proc = 20  # Number of parallel executions

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_data, i) for i in range(num_proc)]
        concurrent.futures.wait(futures)
