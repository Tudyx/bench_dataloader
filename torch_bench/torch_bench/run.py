#!/usr/bin/python3
import argparse

from torch_bench.dataloader import PytorchLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run python dataloader")
    parser.add_argument("--batch-size", required=False, default=16)

    print("Hello from python")
    args = parser.parse_args()
    print(f"Hello from python called with batch-size { args.batch_size}")

    loader = PytorchLoader()

    train_loader = loader.get_train_loader()
    for sample in train_loader:
        print(sample)
        break

    # instantiate the loader
    # train
