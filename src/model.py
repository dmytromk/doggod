import os
import tensorflow as tf

from keras import layers, models
from keras.src.callbacks import ModelCheckpoint

from src.common.config import RESOURCES_DIR, IMG_SIZE, BATCH_SIZE
from src.dataset import load_dataset


def build_model(num_classes, img_size):
    model = models.Sequential([
        layers.BatchNormalization(input_shape=(*img_size, 3)),

        layers.Conv2D(filters=16, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=128, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),

        layers.Conv2D(filters=256, kernel_size=3, activation='relu', kernel_initializer='he_normal'),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),

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

    checkpointer = ModelCheckpoint(filepath=f"{RESOURCES_DIR}/models/doggod.keras",
                                   verbose=1, save_best_only=True)

    #model.load_weights(filepath=f"{RESOURCES_DIR}/models/doggod.keras")
    model.fit(train_dataset,
              validation_data=valid_dataset,
              batch_size=batch_size,
              epochs=20,
              callbacks=[checkpointer],
              verbose=1
              )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")


build_train_model(f"{RESOURCES_DIR}/dog-api", (IMG_SIZE, IMG_SIZE), BATCH_SIZE)
