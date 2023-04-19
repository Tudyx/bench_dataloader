use std::{error::Error, path::PathBuf};

use polars::prelude::*;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    csv_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let df = CsvReader::from_path(cli.csv_path)?
        .has_header(true)
        .finish()?;

    println!("{}", df);

    // TODO: converte the time into seconds.

    let df = df
        .lazy()
        .groupby(["library", "batch_size"])
        .agg([
            mean("total_time").alias("mean"),
            col("total_time").std(1).alias("std"),
        ])
        .sort(
            "batch_size",
            SortOptions {
                multithreaded: true,
                descending: false,
                nulls_last: true,
            },
        )
        .sort(
            "library",
            SortOptions {
                multithreaded: true,
                descending: true,
                nulls_last: true,
            },
        )
        .with_columns([(lit(50_000) / col("mean")).alias("sample_per_seconds")])
        .collect()?;

    println!("{}", df);

    Ok(())
}
