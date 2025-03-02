import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directorios del dataset
train_dir = r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\data\train"
test_dir = r"C:\Users\enzoc\OneDrive\Escritorio\Proyecto_emociones\data\test"

# Generador de imágenes con normalización
datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes desde carpetas
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

print("Clases detectadas:", train_generator.class_indices)
