use std::{error::Error, path::PathBuf};

use polars::prelude::*;

use plotly::{common::Title, layout::Axis, Bar, Layout, Plot};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long)]
    csv_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut cli = Cli::parse();

    let df = CsvReader::from_path(&cli.csv_path)?
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

    let x = df
        .column("batch_size")
        .unwrap()
        .iter()
        .map(|x| x.try_extract::<u32>().unwrap())
        .collect::<Vec<_>>();

    let pytorch_speed = df
        .filter(&df.column("library")?.equal("pytorch")?)?
        .column("sample_per_seconds")
        .unwrap()
        .iter()
        .map(|y| y.try_extract::<u32>().unwrap())
        .collect::<Vec<_>>();

    let ai_dataloader_speed = df
        .filter(&df.column("library")?.equal("ai-dataloader")?)?
        .column("sample_per_seconds")
        .unwrap()
        .iter()
        .map(|y| y.try_extract::<u32>().unwrap())
        .collect::<Vec<_>>();

    let pytorch_trace = Bar::new(x.clone(), pytorch_speed).name("PyTorch");
    let ai_dataloader_trace = Bar::new(x, ai_dataloader_speed).name("ai-dataloader");

    let layout = Layout::new()
        .title(Title::new(
            "Speed comparison between Pytorch and ai-dataloader",
        ))
        .x_axis(Axis::new().title(Title::new("batch size")))
        .y_axis(Axis::new().title(Title::new("speed (image/s)")));

    let mut plot = Plot::new();
    plot.set_layout(layout);
    plot.add_trace(pytorch_trace);
    plot.add_trace(ai_dataloader_trace);
    plot.show();
    println!("{}", plot.to_inline_html(Some("basic_bar_chart")));
    cli.csv_path.set_extension("html");

    plot.write_html(cli.csv_path);

    Ok(())
}
