use std::fs;
use std::io;
use std::io::{Read, Seek, SeekFrom};

use crate::dataset::{Dataset, Iter};
use crate::tensor::Tensor;

pub type MnistItem = ([u8; Mnist::IMAGE_SIZE], u8);
pub type MnistBatch = (Tensor, Tensor);

pub struct Mnist {
    images: Vec<[u8; Mnist::IMAGE_SIZE]>,
    labels: Vec<u8>,
}

impl Mnist {
    const IMAGE_SIZE: usize = 28 * 28;

    pub fn from_source(image_path: &str, label_path: &str) -> io::Result<Mnist> {
        let images = Self::read_images(image_path)?;
        let labels = Self::read_labels(label_path)?;

        Ok(Mnist { images, labels })
    }

    fn read_images(path: &str) -> io::Result<Vec<[u8; Self::IMAGE_SIZE]>> {
        let mut f = fs::File::open(path)?;

        let mut buf_32: [u8; 4] = [0; 4];

        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf_32)?;
        f.seek(SeekFrom::Current(8))?;

        let num_images = u32::from_be_bytes(buf_32);

        let mut images: Vec<[u8; Self::IMAGE_SIZE]> = Vec::with_capacity(num_images as usize);
        let mut buffer_image: [u8; Self::IMAGE_SIZE] = [0; Self::IMAGE_SIZE];

        for _ in 0..num_images {
            f.read_exact(&mut buffer_image)?;
            images.push(buffer_image);
        }
        Ok(images)
    }

    fn read_labels(path: &str) -> io::Result<Vec<u8>> {
        let mut f = fs::File::open(path)?;

        let mut buf_8: [u8; 1] = [0; 1];
        let mut buf_32: [u8; 4] = [0; 4];

        f.seek(SeekFrom::Start(4))?;
        f.read_exact(&mut buf_32)?;

        let num_labels = u32::from_be_bytes(buf_32);

        let mut labels: Vec<u8> = Vec::with_capacity(num_labels as usize);

        for _ in 0..num_labels {
            f.read_exact(&mut buf_8)?;
            labels.push(buf_8[0]);
        }
        Ok(labels)
    }

    pub fn collate(items: &[MnistItem]) -> Option<MnistBatch> {
        // create one-hot vec
        let mut image_batch = Vec::<f32>::with_capacity(items.len() * Mnist::IMAGE_SIZE);
        let mut label_batch = Vec::<u32>::with_capacity(items.len());

        for (image, label) in items {
            let image_f32 = image
                .iter()
                .map(|a| (*a as f32) / 255.0)
                .collect::<Vec<f32>>();

            image_batch.extend(image_f32);
            label_batch.push(*label as u32);
        }

        let image_tensor = Tensor::from_vec([items.len(), 28, 28, 1], image_batch);
        let label_tensor = Tensor::from_vec([items.len(), 1], label_batch);

        Some((image_tensor, label_tensor))
    }
}

impl Dataset for Mnist {
    type Item = MnistItem;

    fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        let image = self.images.get(index);
        let label = self.labels.get(index);

        image.map(|image| (*image, *label.unwrap()))
    }

    fn iter(&self) -> Iter<Self::Item> {
        Iter::new(self)
    }
}
