use clap::Parser;

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
}
