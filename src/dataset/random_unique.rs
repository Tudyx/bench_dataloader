use ai_dataloader::{Dataset, GetSample, Len};
use ndarray::Array3;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
const NUM_CLASS: usize = 20;

/// Only the train part of random dataset for now.
pub struct RandomUnique {
    image: Array3<u8>,
}

impl Default for RandomUnique {
    fn default() -> Self {
        Self {
            image: Array3::random((250, 250, 3), Uniform::new_inclusive(0, 255)),
        }
    }
}

impl Dataset for RandomUnique {}

impl Len for RandomUnique {
    fn len(&self) -> usize {
        50_000
    }
}

impl GetSample for RandomUnique {
    type Sample = (Array3<u8>, i32);

    fn get_sample(&self, index: usize) -> Self::Sample {
        (self.image.clone(), (index % NUM_CLASS) as i32)
    }
}
