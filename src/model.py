from keras import layers, losses, models, optimizers
from keras.applications import MobileNetV2
from keras.src.callbacks import ModelCheckpoint, EarlyStopping

from src.dataset import load_dataset
from src.common.config import RESOURCES_DIR, IMG_SIZE, BATCH_SIZE, SHUFFLE_SEED, EPOCHS


def build_model_scratch(num_classes, img_size):
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

        layers.Dense(num_classes, activation='softmax'),
    ])
    model.summary()

    return model


def build_model_pretrained(num_classes, img_size, pretrained_model):
    base_model = pretrained_model(input_shape=(*img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.summary()

    return model


def train_model(filepath, img_size, batch_size, epochs, pretrained_model=None):
    train_dataset = load_dataset(f"{filepath}/train", img_size, batch_size,
                                 seed=SHUFFLE_SEED, preprocess=True)
    validation_dataset = load_dataset(f"{filepath}/validation", img_size, batch_size, shuffle=False)
    test_dataset = load_dataset(f"{filepath}/test", img_size, batch_size, shuffle=False)

    num_classes = len(train_dataset.class_names)
    train_size = len(train_dataset.file_paths)
    validation_size = len(validation_dataset.file_paths)
    test_size = len(test_dataset.file_paths)

    print(f"Number of breeds: {num_classes}")
    print(f"Number of training images: {train_size}")
    print(f"Number of validation images: {validation_size}")
    print(f"Number of test images: {test_size}")

    if pretrained_model:
        model = build_model_pretrained(len(train_dataset.class_names), img_size, pretrained_model)
    else:
        model = build_model_scratch(len(train_dataset.class_names), img_size)

    model.compile(optimizer=optimizers.legacy.Adam(),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    filename = pretrained_model.__name__ if pretrained_model else 'Scratch'
    checkpointer = ModelCheckpoint(filepath=f"{RESOURCES_DIR}/models/Doggod{filename}",
                                   verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=3)
    model.fit(train_dataset,
              validation_data=validation_dataset,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[checkpointer, early_stopping],
              verbose=1
              )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")


train_model(f"{RESOURCES_DIR}/dog-api", (IMG_SIZE, IMG_SIZE), BATCH_SIZE, 10)
train_model(f"{RESOURCES_DIR}/dog-api", (IMG_SIZE, IMG_SIZE), BATCH_SIZE, EPOCHS, MobileNetV2)
