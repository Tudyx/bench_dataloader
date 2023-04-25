mod dataset;
use ai_dataloader::Len;
use clap::Parser;
use dataset::RandomDataset;
use std::{error::Error, fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

use ai_dataloader::collate::TorchCollate;
use ai_dataloader::indexable::DataLoader;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Number of sample within a batch.
    #[arg(short, long)]
    batch_size: usize,
    /// Path where to store the result.
    #[arg(short, long)]
    csv_path: PathBuf,
}

// To compute the samples_per_seconds
// We are taking the time per batch and calculating the speed per batch
// Then, we are taking the average. The first batch is considerably slower

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    println!("cudart version {}", tch::utils::version_cudart());

    let dataset = RandomDataset::default();

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

    println!("Total loading time: {:.2?}", elapsed);

    println!(
        "Average batch time: {:.2?}",
        elapsed.div_f64(loader.len() as f64)
    );
    println!(
        "Average sample time {:.2?}",
        (elapsed
            .div_f64(loader.len() as f64)
            .div_f64(cli.batch_size as f64))
    );

    println!(
        "Sample per seconds {:.2?}",
        1. / (elapsed
            .div_f64(loader.len() as f64)
            .div_f64(cli.batch_size as f64))
        .as_secs_f64()
    );

    // Append the result to the csv
    let mut csv_path = OpenOptions::new()
        .write(true)
        .append(true)
        .open(cli.csv_path)?;

    writeln!(
        csv_path,
        "ai-dataloader,{},{}",
        cli.batch_size,
        elapsed.as_secs_f64()
    )?;

    Ok(())
}
