import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\rebek\Programme\Python\CNN\archive\train"
# Daten laden und vorverarbeiten
def load_and_preprocess_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(160, 160),
        batch_size=32)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        file_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(160, 160),
        batch_size=32)

    return train_ds, val_ds

train_ds, val_ds = load_and_preprocess_data()

# Modell erstellen
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(53)  # Angenommen, es gibt 53 Klassen (jede Karte im Deck)
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Modell trainieren
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Trainingsverlauf darstellen
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

# Modell evaluieren
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f"Test accuracy: {test_acc}")

model.save('my_model.keras')  # Speichert das gesamte Modell
