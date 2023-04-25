mod dataset;
use ai_dataloader::Len;
use clap::{Parser, ValueEnum};
use dataset::RandomDataset;
use std::time::Duration;
use std::{error::Error, fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

use ai_dataloader::collate::{Collate, DefaultCollate, TorchCollate};
use ai_dataloader::indexable::DataLoader;

use crate::dataset::RandomUnique;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Dataset {
    Random,
    RandomUnique,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of sample within a batch.
    #[arg(short, long)]
    batch_size: usize,
    /// Path where to store the result.
    #[arg(short, long)]
    csv_path: PathBuf,
    #[arg(short, long, value_enum)]
    dataset: Dataset,
}

fn run_random(cli: Cli, dataset: RandomDataset) -> Result<(), Box<dyn Error>> {
    let loader = DataLoader::builder(dataset)
        .batch_size(cli.batch_size)
        .collate_fn(TorchCollate)
        .build();

    let device = tch::Device::Cuda(0);
    let mut num_sample = 0;
    let now = Instant::now();
    for (sample, label) in loader.iter() {
        let _sample = sample.to(device);
        num_sample += label.size1().unwrap();
    }
    let elapsed = now.elapsed();
    assert_eq!(num_sample, 50_000);

    report(elapsed, loader.len(), cli.batch_size, cli.csv_path)?;

    Ok(())
}

fn run_random_unique(cli: Cli, dataset: RandomUnique) -> Result<(), Box<dyn Error>> {
    let loader = DataLoader::builder(dataset)
        .batch_size(cli.batch_size)
        .collate_fn(TorchCollate)
        .build();

    let device = tch::Device::Cuda(0);
    let mut num_sample = 0;
    let now = Instant::now();
    for (sample, label) in loader.iter() {
        let _sample = sample.to(device);
        num_sample += label.size1().unwrap();
    }
    let elapsed = now.elapsed();
    assert_eq!(num_sample, 50_000);

    report(elapsed, loader.len(), cli.batch_size, cli.csv_path)?;

    Ok(())
}

fn report(
    total_time: Duration,
    num_batch: usize,
    batch_size: usize,
    csv_path: PathBuf,
) -> Result<(), Box<dyn Error>> {
    println!("Total loading time: {:.2?}", total_time);

    println!(
        "Average batch time: {:.2?}",
        total_time.div_f64(num_batch as f64)
    );
    println!(
        "Average sample time {:.2?}",
        (total_time
            .div_f64(num_batch as f64)
            .div_f64(batch_size as f64))
    );

    println!(
        "Sample per seconds {:.2?}",
        1. / (total_time
            .div_f64(num_batch as f64)
            .div_f64(batch_size as f64))
        .as_secs_f64()
    );

    // Append the result to the csv
    let mut csv_path = OpenOptions::new().write(true).append(true).open(csv_path)?;

    writeln!(
        csv_path,
        "ai-dataloader,{},{}",
        batch_size,
        total_time.as_secs_f64()
    )?;
    Ok(())
}

// To compute the samples_per_seconds
// We are taking the time per batch and calculating the speed per batch
// Then, we are taking the average. The first batch is considerably slower

// Generic version but not very useful.
#[allow(dead_code)]
fn run<D>(cli: Cli, dataset: D) -> Result<(), Box<dyn Error>>
where
    D: ai_dataloader::Dataset,
    DefaultCollate: Collate<D::Sample>, // Shouldn't be required
    TorchCollate: Collate<D::Sample>,
{
    let loader = DataLoader::builder(dataset)
        .batch_size(cli.batch_size)
        .collate_fn(TorchCollate)
        .build();

    let now = Instant::now();
    for sample in loader.iter() {
        // With generic version we can't really do anithing
        let _ = sample;
    }
    let elapsed = now.elapsed();

    report(elapsed, loader.len(), cli.batch_size, cli.csv_path)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    println!("cudart version {}", tch::utils::version_cudart());

    match cli.dataset {
        Dataset::Random => run_random(cli, RandomDataset::default())?,
        Dataset::RandomUnique => run_random_unique(cli, RandomUnique::default())?,
    };

    Ok(())
}
