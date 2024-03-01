import os

from tensorflow.keras import layers, models

from src.common.config import RESOURCES_DIR, IMG_SIZE, BATCH_SIZE
from src.dataset import load_dataset


def build_model(filter_size, num_classes, img_size):
    model = models.Sequential([
        layers.Conv2D(filter_size, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filter_size * 2, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filter_size * 2, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def build_train_model(filepath, filter_size, img_size):
    train_dataset, train_size = load_dataset(f"{filepath}/train", BATCH_SIZE)
    valid_dataset, valid_size = load_dataset(f"{filepath}/validation", BATCH_SIZE)
    test_dataset, test_size = load_dataset(f"{filepath}/test", BATCH_SIZE)

    num_classes = len(os.listdir(f"{filepath}/train"))
    print(num_classes)
    model = build_model(filter_size, num_classes, img_size)

    model.fit(train_dataset,
              steps_per_epoch=train_size // 32,
              batch_size=32,
              epochs=10,
              validation_data=valid_dataset,
              validation_steps=valid_size // 32,
              verbose=1
              )

    test_loss, test_accuracy = model.evaluate_generator(test_dataset, steps=test_size // 32)
    print(test_loss)
    print(test_accuracy)
