//! Benchmark
// Inspired by https://arxiv.org/pdf/2209.13705.pdf
// We compare only monothreaded version here, as for now ai-datalaoder is not parrallel.
use itertools::iproduct;
use std::process::Command;

const NB_EPOCH: usize = 1;
const BATCH_SIZE: &[usize] = &[16, 64, 128];
const LIBARIES: &[&[&str]] = &[RUST_CMD, PYTHON_CMD];
const RUST_CMD: &[&str] = &["cargo", "run", "--release", "--bin", "run", "--"];
// FIXME: config poetry to use a local venv and give the path here.
const PYTHON_CMD: &[&str] = &[
    "/home/teddy/.cache/pypoetry/virtualenvs/torch-bench-pHys0gd8-py3.10/bin/python",
    "torch_bench/torch_bench/run.py",
];

fn main() {
    let mut _nb_processed_image = 0;
    for _ in 0..NB_EPOCH {
        for (args, batch_size) in iproduct!(LIBARIES, BATCH_SIZE) {
            let mut command = Command::new(args[0]);
            command.env("TORCH_CUDA_VERSION", "cu117");
            for arg in args[1..].iter() {
                command.arg(arg);
            }
            command.arg("--batch-size").arg(batch_size.to_string());
            dbg!(&command);

            let output = command.output().expect("failed to execute process");
            dbg!(&output);
            if !output.status.success() {
                eprintln!("Error while executing {:?}", command);
            }
        }
    }
}
