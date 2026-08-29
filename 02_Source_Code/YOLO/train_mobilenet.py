"""
train_mobilenet.py
Algorithm 2: MobileNetV2 transfer learning (ImageNet pretrained weights).
Data source: dataset/classification/{train,val}/{class_name}/*.jpg
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = "dataset/classification"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS_HEAD = 10       # stage 1: train the new classification head only
EPOCHS_FINETUNE = 10   # stage 2: unfreeze some layers and fine-tune


def build_mobilenet(num_classes):
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False  # freeze the backbone for stage 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base


def main():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print(f"Number of classes: {len(class_names)} -> {class_names}")

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    model, base = build_mobilenet(len(class_names))
    model.summary()

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)]

    print(">>> Stage 1: training the classification head (backbone frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

    print(">>> Stage 2: unfreezing the last layers for fine-tuning")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks)

    model.save("models/mobilenet_food_classifier.h5")
    print("Model saved to: models/mobilenet_food_classifier.h5")


if __name__ == "__main__":
    main()
