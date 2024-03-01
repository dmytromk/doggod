import os
import random
from glob import glob

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2

from src.common import config


def parse_image(filepath: str | bytes, image_size: int):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def train_preprocess(image):
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def configure_for_performance(dataset: DatasetV2, batch_size: int) -> DatasetV2:
    dataset = dataset.shuffle(buffer_size=1000, seed=config.SHUFFLE_SEED)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_dataset(filepath: str, batch_size: int, image_size: int) -> (DatasetV2, int):
    classes = os.listdir(filepath)
    filenames = glob(filepath + '/*/*')
    random.seed(config.SHUFFLE_SEED)
    random.shuffle(filenames)
    labels = [classes.index(name.split('/')[-2]) for name in filenames]

    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(lambda x: parse_image(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds, batch_size)

    return ds, len(filenames)
