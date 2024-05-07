from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model


model = load_model(r'C:\Users\rebek\Programme\Python\CNN\TensorFlow\my_model.keras')

# Pfad zum Bild, das Sie testen möchten
image_path = r"C:\Users\rebek\Programme\Python\CNN\archive\train\ace of spades\012.jpg"

# Bild laden und Größe ändern
img = image.load_img(image_path, target_size=(160, 160))

# Das Bild in ein Array umwandeln
img_array = image.img_to_array(img)

# Pixelwerte skalieren
img_array = img_array / 255.0

# Ein Batch aus einem Bild erstellen (Modell erwartet eine Batch-Dimension)
img_array = np.expand_dims(img_array, axis=0)

# Modell verwenden, um das Bild vorherzusagen
predictions = model.predict(img_array)

# Die Klasse mit der höchsten Wahrscheinlichkeit auswählen
predicted_class = np.argmax(predictions, axis=1)

print(f"Predicted class: {predicted_class}")
