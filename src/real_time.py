import cv2
import numpy as np
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model(r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\models\modelo_final.h5")

# Mapeo de etiquetas de emociones
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Conectar la webcam (en este caso uso la camara de mi celular al no contar con la webcam)
# cap = cv2.VideoCapture(0)  # Usa la webcam de la laptop en caso de contar con ella
cap = cv2.VideoCapture("http://192.168.100.144:4747/video") #comentar esta linea y descomentar la linea de arriba en caso de contar con webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame")
        continue  

    #  Rotar imagen si es necesario
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #  Optimización: 
    processed_faces = []
    face_positions = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Recortar la cara detectada
        face = cv2.resize(face, (48, 48))  # Redimensionar a 48x48

        
        face = np.expand_dims(face, axis=-1)  
        face = np.repeat(face, 3, axis=-1)  
        face = np.expand_dims(face, axis=0)  
        face = face / 255.0  # Normalizar valores a [0,1]

       
        processed_faces.append(face)
        face_positions.append((x, y, w, h))

    if processed_faces:
        predictions = model.predict(np.vstack(processed_faces))
        for i, (x, y, w, h) in enumerate(face_positions):
            emotion = emotion_labels[np.argmax(predictions[i])]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar el frame con detecciones
    cv2.imshow("Reconocimiento de Emociones", frame)

    # botn de salida salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
