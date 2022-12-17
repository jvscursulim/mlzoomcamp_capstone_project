import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import layers

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

dataset_path = ""
target_size = (150, 150)
batch_size = 20
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

