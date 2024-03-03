import os
import random
from glob import glob

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV2


def parse_image(filepath: str | bytes, image_size: tuple[int, int]):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def train_preprocess(image, seed):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    image = tf.image.random_brightness(image, seed=seed, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, seed=seed, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def load_dataset(
        filepath: str,
        image_size: tuple[int, int],
        batch_size: int | None = None,
        shuffle: bool = True,
        seed: int = 132,
        preprocess: bool = False
) -> DatasetV2:
    classes = os.listdir(filepath)
    image_paths = glob(filepath + '/*/*')

    if shuffle:
        random.seed(seed)
        random.shuffle(image_paths)
    labels = [classes.index(name.split('/')[-2]) for name in image_paths]

    filenames_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    images_ds = filenames_ds.map(lambda x: parse_image(x, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess:
        images_ds = images_ds.map(lambda x: train_preprocess(x, seed), num_parallel_calls=tf.data.AUTOTUNE)

    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((images_ds, labels_ds))

    if batch_size is not None:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    dataset.class_names = classes

    dataset.file_paths = image_paths

    return dataset
