mod dataset;
use ai_dataloader::Len;
use clap::Parser;
use dataset::RandomDataset;
use std::time::Instant;

use ai_dataloader::collate::TorchCollate;
use ai_dataloader::indexable::DataLoader;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Batch size.
    #[arg(short, long)]
    batch_size: usize,
}

// To compute the samples_per_seconds
// We are taking the time per batch and calculating the speed per batch
// Then, we are taking the average. The first batch is considerably slower

fn main() {
    let cli = Cli::parse();
    println!("Hello, I've been run with {} batch size", cli.batch_size);

    let dataset = RandomDataset::default();

    let loader = DataLoader::builder(dataset)
        .batch_size(cli.batch_size)
        .collate_fn(TorchCollate)
        .build();

    let mut num_sample = 0;
    let now = Instant::now();
    for (_sample, label) in loader.iter() {
        num_sample += label.len();
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
}
