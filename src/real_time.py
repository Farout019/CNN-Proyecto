import cv2
import numpy as np
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model(r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\models\modelo_final.h5")

# Mapeo de etiquetas de emociones
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Inicializar cámara
cap = cv2.VideoCapture("http://192.168.100.144:4747/video")

# Cargar el detector de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame")
        continue  

    # 🔄 Rotar imagen si es necesario
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 🔹 Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔍 Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # ✅ Optimización: Usar la predicción en un solo paso (evita latencia)
    processed_faces = []
    face_positions = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Recortar la cara detectada
        face = cv2.resize(face, (48, 48))  # Redimensionar a 48x48

        # 🔹 Convertir a formato adecuado para el modelo (RGB y normalizado)
        face = np.expand_dims(face, axis=-1)  # Agregar canal de profundidad
        face = np.repeat(face, 3, axis=-1)  # Convertir a 3 canales (RGB)
        face = np.expand_dims(face, axis=0)  # Agregar dimensión de batch
        face = face / 255.0  # Normalizar valores a [0,1]

        # Almacenar en una lista para procesarlas todas juntas
        processed_faces.append(face)
        face_positions.append((x, y, w, h))

    if processed_faces:
        # 🔥 **Optimización: Procesar todas las caras en un solo paso**
        predictions = model.predict(np.vstack(processed_faces))

        for i, (x, y, w, h) in enumerate(face_positions):
            emotion = emotion_labels[np.argmax(predictions[i])]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 📸 Mostrar el frame con detecciones
    cv2.imshow("Reconocimiento de Emociones", frame)

    # 🔴 Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔚 Liberar recursos
cap.release()
cv2.destroyAllWindows()
