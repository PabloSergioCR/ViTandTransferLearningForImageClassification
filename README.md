**Clasificador de Imágenes con Vision Transformer (ViT)**

Este proyecto implementa un clasificador de imágenes utilizando un modelo preentrenado de Vision Transformer (ViT) disponible en Hugging Face. El modelo ha sido ajustado (fine-tuning) para clasificar imágenes de un conjunto de datos personalizado.

**Características principales**

- Modelo base: Vision Transformer (ViT) de Google (google/vit-base-patch16-224-in21k).

- Conjunto de datos: Personalizado, con clases específicas para una tarea definida.

- Entrenamiento realizado en Google Colab.

- Implementación basada en la biblioteca Hugging Face Transformers y datasets.

**Contenido del repositorio**

El repositorio contiene los siguientes archivos y carpetas principales:

- ImageSorterCreator.py: Código principal para el entrenamiento del modelo.
- inference.py: Código para realizar predicciones con el modelo entrenado.
- LICENSE: Archivo de licencia para el proyecto.
- README.md: Este archivo explicativo.
- Carpeta dataset/: Ejemplo de cómo estructurar los datos de entrenamiento y prueba.(no disponible)
- Carpeta outputs/: Modelos entrenados y métricas de evaluación.(no disponible)

**Requisitos del sistema**

Antes de ejecutar este proyecto, asegúrate de instalar las siguientes dependencias:

Dependencias principales

*bash*

*pip install torch torchvision transformers datasets evaluate*

Herramientas adicionales

Python 3.8 o superior.

Google Colab (opcional, para entrenamiento en la nube).

scikit-learn: Para calcular métricas adicionales.

**Entrenamiento del modelo**

Sigue los pasos de ImageSorterCreator.py para crear y entrenar el modelo


**Ejemplo de uso**

*bash*

*python inference.py --image_path ./sample_image.jpg --model_dir ./outputs/checkpoint-500*

Salida esperada:

*makefile*

Predicción: Class1 (95% de confianza)

**Modelo base**

El modelo utilizado, Vision Transformer (ViT), fue desarrollado por Google y está disponible en Hugging Face bajo la licencia Apache 2.0. Más información sobre el modelo base:

Hugging Face Model Card: ViT Base Patch16-224

**Licencia**

Este proyecto está licenciado bajo CC BY-NC 4.0. Esto significa que puedes usar y modificar este proyecto para fines no comerciales, siempre que se otorgue el crédito correspondiente al autor.

**Contribuciones**

¡Las contribuciones son bienvenidas! Si deseas mejorar este proyecto, abre un Pull Request o crea un Issue en este repositorio.

**Créditos**

Este proyecto fue desarrollado como un ejemplo de clasificación de imágenes utilizando modelos preentrenados. El modelo base pertenece a Google Research y fue ajustado utilizando herramientas de Hugging Face.
