//! Benchmark
// Inspired by https://arxiv.org/pdf/2209.13705.pdf
// We compare only monothreaded version here, as for now ai-datalaoder is not parrallel.
use chrono::Utc;
use colored::Colorize;
use itertools::iproduct;
use std::{
    error::Error,
    fs,
    io::{self, Write},
    process::Command,
};

const NB_EPOCH: usize = 100;
const BATCH_SIZE: &[usize] = &[16, 64, 128];
const LIBARIES: &[&[&str]] = &[RUST_CMD, PYTHON_CMD];
const RUST_CMD: &[&str] = &["cargo", "run", "--release", "--bin", "run", "--"];
const PYTHON_CMD: &[&str] = &[
    "torch_bench/.venv/bin/python",
    "torch_bench/torch_bench/run.py",
];

const DATASET: &str = "random-unique";

const NUM_WORKERS: &[usize] = &[2, 3, 4];

fn main() -> Result<(), Box<dyn Error>> {
    // Write header to the result csv.
    let csv_path = format!("res/run_{:?}.csv", Utc::now());
    fs::create_dir_all("res")?;
    fs::write(&csv_path, "library,batch_size,total_time,num_workers\n")?;

    for _ in 0..NB_EPOCH {
        for (args, batch_size, num_workers) in iproduct!(LIBARIES, BATCH_SIZE, NUM_WORKERS) {
            let mut command = Command::new(args[0]);
            // command.env("TORCH_CUDA_VERSION", "cu117");
            for arg in args[1..].iter() {
                command.arg(arg);
            }
            command.arg("--batch-size").arg(batch_size.to_string());
            command.arg("--csv-path").arg(&csv_path);
            command.arg("--dataset").arg(DATASET);
            command.arg("--num-workers").arg(num_workers.to_string());
            dbg!(&command);

            let output = command.output().expect("failed to execute process");
            io::stdout().write_all(&output.stdout).unwrap();
            if !output.status.success() {
                eprintln!("{} '{:?}':", "Error while executing".red(), command);
                io::stderr().write_all(&output.stderr).unwrap();
            }
        }
    }
    Ok(())
}
