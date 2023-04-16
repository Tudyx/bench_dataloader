#!/usr/bin/python3
import argparse

import time
import torch

from torch_bench.dataloader import PytorchLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run python dataloader")
    parser.add_argument("--batch-size", required=False, default=16)
    args = parser.parse_args()
    print(f"Hello from python called with batch-size { args.batch_size}")

    loader = PytorchLoader()

    train_loader = loader.get_train_loader(batch_size=args.batch_size)

    num_samples = 0

    start = time.perf_counter_ns()
    for sample, label in train_loader:
        label = label.to("cuda:0")
        sample = sample.to("cuda:0")
        num_samples += len(label)
    total_time = time.perf_counter_ns() - start

    assert num_samples == 50_000

    print(f"Total time {total_time} nanoseconds")

    # instantiate the loader
    # train
