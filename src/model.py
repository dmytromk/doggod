import os
import tensorflow as tf

from keras import layers, models

from src.common.config import RESOURCES_DIR, IMG_SIZE, BATCH_SIZE
from src.dataset import load_dataset, train_preprocess


def build_model(num_classes, img_size):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(*img_size, 3)),
        layers.MaxPooling2D(pool_size=2, strides=(2, 2)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=2, strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_train_model(filepath, img_size, batch_size):
    train_dataset = load_dataset(f"{filepath}/train", img_size, batch_size,
                                 seed=132, preprocess=True)
    valid_dataset = load_dataset(f"{filepath}/validation", img_size, batch_size, shuffle=False)
    test_dataset = load_dataset(f"{filepath}/test", img_size, batch_size, shuffle=False)

    print(f"Number of breeds: {len(train_dataset.class_names)}")

    model = build_model(len(train_dataset.class_names), img_size)

    model.fit(train_dataset,
              validation_data=valid_dataset,
              batch_size=batch_size,
              epochs=10,
              verbose=1
              )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")


build_train_model(f"{RESOURCES_DIR}/dog-api", (IMG_SIZE, IMG_SIZE), BATCH_SIZE)
