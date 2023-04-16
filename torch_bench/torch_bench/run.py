#!/usr/bin/python3
import argparse

import time
import torch

from torch_bench.dataloader import PytorchLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run python dataloader")
    parser.add_argument("--batch-size", required=False, default=16)

    print("Hello from python")
    args = parser.parse_args()
    print(f"Hello from python called with batch-size { args.batch_size}")

    if not torch.cuda.is_available():
        raise Exception("Pytorch didn't encounter the GPU!")
    with torch.cuda.device("cuda:0") as d:
        a = torch.zeros(4, 3)
        print(a.get_device())

    loader = PytorchLoader()

    train_loader = loader.get_train_loader()

    num_samples = 0

    start = time.perf_counter_ns()
    for sample, label in train_loader:
        label = label.to("cuda:0")
        sample = sample.to("cuda:0")
        num_samples += len(sample)
        # break

    total_time = time.perf_counter_ns() - start

    print(f"Total time {total_time} nanoseconds")

    # instantiate the loader
    # train
