mod dataset;
use clap::Parser;
use dataset::RandomDataset;

use ai_dataloader::indexable::DataLoader;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Batch size.
    #[arg(short, long)]
    batch_size: usize,
}

fn main() {
    let cli = Cli::parse();
    println!("Hello, I've been run with {} batch size", cli.batch_size);

    let dataset = RandomDataset::default();

    let loader = DataLoader::builder(dataset)
        .batch_size(cli.batch_size)
        .build();

    for sample in loader.iter().take(1) {
        // dbg!(sample);
        break;
    }
}
