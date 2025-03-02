import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta del modelo guardado
MODEL_PATH = r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\models\modelo_final.h5"

# Directorio de prueba
TEST_DIR = r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\data\test"

# Cargar el modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Preparar generador de datos para test
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Importante para que los índices coincidan con las predicciones
)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"🔹 Precisión en test: {test_acc*100:.2f}%")
print(f"🔹 Pérdida en test: {test_loss:.4f}")

# Obtener predicciones
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Reporte de clasificación
print("\n🔹 Clasificación por clases:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()
