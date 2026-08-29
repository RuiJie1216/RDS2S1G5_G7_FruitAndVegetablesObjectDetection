"""
train_cnn.py
Algorithm 1: custom CNN classifier, trained from scratch.
Data source: dataset/classification/{train,val}/{class_name}/*.jpg  (built by data_prepare.py)
"""

import tensorflow as tf
from tensorflow.keras import layers, models

DATA_DIR = "dataset/classification"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20


def build_cnn(num_classes):
    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 3)),

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print(f"Number of classes: {len(class_names)} -> {class_names}")

    # Augmentation is done inside the graph, so no duplicate files on disk
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y))

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model = build_cnn(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("models/cnn_food_classifier.h5",
                                            save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                         epochs=EPOCHS, callbacks=callbacks)

    model.save("models/cnn_food_classifier.h5")
    print("Model saved to: models/cnn_food_classifier.h5")
    return history


if __name__ == "__main__":
    main()
