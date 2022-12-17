import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.lite import tflite
from keras import layers

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Dataset link: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
# Download the dataset and put the folders with the images of the dataset in data/dataset
dataset_path = "../data/dataset"
target_size = (150, 150)
batch_size = 64
epochs = 5
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.BinaryCrossentropy()

train_dataset = datagen.flow_from_directory(dataset_path,
                                         target_size=target_size,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         subset="training",
                                         class_mode="binary")

validation_dataset = datagen.flow_from_directory(dataset_path,
                                         target_size=target_size,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         subset="validation",
                                         class_mode="binary")

model = keras.Sequential([layers.Input(shape=(150, 150, 3)),
                         layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
                         layers.MaxPooling2D(pool_size=(2, 2)),
                         layers.Flatten(),
                         layers.Dense(units=32, activation="relu"),
                         layers.Dense(units=1, activation="sigmoid")], name="cnn")

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)

lr_str = str(learning_rate).replace(".", "_")
model.save(f"../model/cnn_surface_crack_detection_bs{batch_size}_epochs{epochs}_lr_{lr_str}.h5")

converter = tflite.TFLiteConverter.from_keras_model(model=model)
tflite_model = converter.convert()

with open("../model/cnn_surface_crack_detection_bs64_epochs5_lr_0_001.tflite", "wb") as file:
    file.write(tflite_model)
