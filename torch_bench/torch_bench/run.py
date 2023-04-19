#!/usr/bin/python3
import argparse

import time
import torch

from torch_bench.dataloader import PytorchLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run python dataloader")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--csv-path")
    args = parser.parse_args()

    loader = PytorchLoader()

    train_loader = loader.get_train_loader(batch_size=args.batch_size)

    num_samples = 0

    start = time.perf_counter()
    for sample, label in train_loader:
        label = label.to("cuda:0")
        sample = sample.to("cuda:0")
        num_samples += len(label)
    total_time = time.perf_counter() - start

    assert num_samples == 50_000

    print(f"Total time {total_time} nanoseconds")

    with open(args.csv_path, "a") as f:
        f.write(f"pytorch,{args.batch_size},{total_time}\n")
