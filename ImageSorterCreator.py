!pip install transformers datasets evaluate torch torchvision --no-deps
!pip install fsspec==2024.10.0 gcsfs==2024.10.0
!pip install datasets==2.14.5

#Importamos las librerías necesarias
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
from evaluate import load

# Cargamos dataset CIFAR-10
dataset = load_dataset("cifar10")

# Dividimos el dataset en entrenamiento y prueba
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Verificamos las clases
labels = train_dataset.features["label"].names
print(f"Clases: {labels}")

# Cargamos el extractor preentrenado
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Función de preprocesamiento
def preprocess(examples):
    examples["pixel_values"] = [feature_extractor(image, return_tensors="pt")["pixel_values"][0] for image in examples["img"]]
    return examples

# Preprocesamos los datasets
train_dataset = train_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

# Borramos datos originales para ahorrar memoria
train_dataset = train_dataset.remove_columns(["img"])
test_dataset = test_dataset.remove_columns(["img"])

#Cargamos el modleo preentrenado que va a clasificar las imágenes
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={i: label for i, label in enumerate(labels)},
    label2id={label: i for i, label in enumerate(labels)},
)

#Modificamos los hiperparámetros del modleo y las últimas capas para adaptarlo a nuestras necesidad. Como siempre este dependerá de la velocidad y precisión que queramos pudiendo hacelro a prueba y error hasta obtener los resultados que deseemos
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",  # Cambia a 'wandb' si usas Weights & Biases
    push_to_hub=False
)

# Métrica de evaluación
accuracy_metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

#Entrenamos el modelo configurando los parámetros del trainer previamente 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Entrenamos el modelo puede tardar 10 horas con los parámetros que aparecen en este código
trainer.train()

#Evaluamos el modelo
results = trainer.evaluate()
print(f"Resultados: {results}")

#Lo guardamos para no tener que entrenarlo cada vez que lo queramos usar
trainer.save_model("./cifar10-vit-model")
