import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import argparse
from torchvision import transforms
import os

# Función para preprocesar la imagen (igual que se hizo durante el entrenamiento)
def preprocess_image(image_path, feature_extractor):
    # Cargar la imagen
    image = Image.open(image_path)

    # Aplicar las transformaciones necesarias usando el extractor de características
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

# Función para realizar la predicción
def predict(image_path, model_dir):
    # Cargar el modelo y el extractor de características desde el directorio guardado
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    
    # Preprocesar la imagen
    inputs = preprocess_image(image_path, feature_extractor)
    
    # Hacer la predicción con el modelo
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener las probabilidades de la predicción
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Obtener la clase predicha y la probabilidad más alta
    predicted_class = torch.argmax(probs, dim=-1).item()
    predicted_probability = probs[0, predicted_class].item()
    
    return predicted_class, predicted_probability

# Función principal para manejar la entrada y mostrar resultados
def main():
    # Parsear los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description="Predicción de imágenes con modelo entrenado.")
    parser.add_argument('--image_path', type=str, required=True, help="Ruta de la imagen para la predicción.")
    parser.add_argument('--model_dir', type=str, required=True, help="Ruta al directorio del modelo entrenado.")
    
    args = parser.parse_args()
    
    # Realizar la predicción
    predicted_class, predicted_probability = predict(args.image_path, args.model_dir)
    
    # Mapear la clase numérica a su nombre de clase
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]  # Las clases de CIFAR-10
    predicted_class_name = labels[predicted_class]
    
    # Mostrar los resultados
    print(f'Predicción: {predicted_class_name}, Probabilidad: {predicted_probability:.4f}')

if __name__ == '__main__':
    main()
