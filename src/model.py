import os
import tensorflow as tf

from keras import layers, models

from src.common.config import RESOURCES_DIR, IMG_SIZE, BATCH_SIZE
from src.dataset import load_dataset, train_preprocess


def build_model(filter_size, num_classes, img_size):
    model = models.Sequential([
        layers.Conv2D(filter_size, 3, activation='relu', padding='same', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(pool_size=2, strides=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(filter_size * 2, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=2, strides=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_train_model(filepath, filter_size, img_size, batch_size):
    train_dataset, train_size = load_dataset(f"{filepath}/train", batch_size, img_size)
    valid_dataset, valid_size = load_dataset(f"{filepath}/validation", batch_size, img_size)
    test_dataset, test_size = load_dataset(f"{filepath}/test", batch_size, img_size)

    train_dataset = train_dataset.map(lambda image, label: (train_preprocess(image), label),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    num_classes = len(os.listdir(f"{filepath}/train"))

    print(f"Number of breeds: {num_classes}")
    print(f"Number of training images: {train_size}")
    print(f"Number of validation images: {valid_size}")
    print(f"Number of test images: {test_size}")
    print(f"Number of images: {test_size + valid_size + train_size}")

    model = build_model(filter_size, num_classes, img_size)

    model.fit(train_dataset,
              steps_per_epoch=train_size // 32,
              batch_size=batch_size,
              epochs=10,
              validation_data=valid_dataset,
              validation_steps=valid_size // 32,
              verbose=1
              )

    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_size // 32)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")


build_train_model(f"{RESOURCES_DIR}/dog-api", 16, IMG_SIZE, BATCH_SIZE)
