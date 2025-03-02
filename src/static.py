import cv2
import numpy as np
import tensorflow as tf
import os

# Cargar el modelo entrenado
model = tf.keras.models.load_model(r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\models\modelo_final.h5")

# Mapeo de etiquetas de emociones
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Cargar el clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Ruta de la imagen que deseas analizar
image_path = r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\data\IMG\angry3.jpg"  # Cambia esto por el nombre de la imagen que desees analizar

if not os.path.exists(image_path):
    print("Error: La imagen no existe.")
    exit()

# Cargar la imagen
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = cv2.resize(face, (48, 48))  # Redimensiona
    face = np.stack([face] * 3, axis=-1)  # Convierte a RGB (de 48x48 a 48x48x3)
    face = np.expand_dims(face, axis=0)  # Agrega dimensión batch


    # Predecir emoción
    prediction = model.predict(face)
    emotion = emotion_labels[np.argmax(prediction)]

    # Dibujar rectángulo y etiqueta
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Mostrar imagen con la emoción detectada
cv2.imshow("Reconocimiento de Emociones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
