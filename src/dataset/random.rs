use ai_dataloader::{Dataset, GetSample, Len};
use ndarray::Array3;
use nshare::ToNdarray3;
use std::path::PathBuf;
const NUM_CLASS: usize = 20;

/// Only the train part of random dataset for now.
pub struct RandomDataset {
    dataset_dir: PathBuf,
}

impl Default for RandomDataset {
    fn default() -> Self {
        Self {
            // FIXME: hardcoded
            dataset_dir: PathBuf::from(
                "/home/teddy/dev/rust/bench_dataloader/data/random/shared/train/",
            ),
        }
    }
}

impl Dataset for RandomDataset {}

impl Len for RandomDataset {
    fn len(&self) -> usize {
        50_000
    }
}

impl GetSample for RandomDataset {
    type Sample = (Array3<u8>, i32);

    fn get_sample(&self, index: usize) -> Self::Sample {
        let image_path = self.dataset_dir.join(format!("{index}.jpeg"));
        let image = image::open(image_path).unwrap();
        // The axes/dimensions created follow the pytorch convention : Color x Height x Width
        let image = image.into_rgb8().into_ndarray3();
        (image, (index % NUM_CLASS) as i32)
    }
}
