import argparse
import os
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    AutoTokenizer
)
from src.config import Config

def compute_metrics(eval_pred):
    """Calcule la précision (accuracy) à chaque époque"""
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(config_path):
    cfg = Config.from_yaml(config_path)
    processed_path = cfg.data["processed_path"]
    model_name = cfg.model["name"]
    num_labels = cfg.model["num_labels"]
    output_dir = cfg.training["output_dir"]
    
    print(f"⚙️ Chargement des données tokenisées depuis {processed_path}...")
    dataset = load_from_disk(processed_path)
    
    # Mapping des 6 émotions du dataset (tristesse, joie, amour, colère, peur, surprise)
    emotions = ["Tristesse 😢", "Joie 😄", "Amour ❤️", "Colère 😡", "Peur 😨", "Surprise 😲"]
    id2label = {i: label for i, label in enumerate(emotions)}
    label2id = {label: i for i, label in enumerate(emotions)}

    print(f"🤖 Initialisation de {model_name} pour la classification d'émotions...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Paramètres d'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.training["learning_rate"],
        per_device_train_batch_size=cfg.training["batch_size"],
        per_device_eval_batch_size=cfg.training["batch_size"],
        num_train_epochs=cfg.training["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(20))

    print("🚀 Lancement de l'entraînement (Trainer Hugging Face)...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,  # <-- Le nouveau nom du paramètre !
    )

    trainer.train()

    print(f"💾 Sauvegarde du modèle final (prêt pour l'API) dans {output_dir}...")
    trainer.save_model(output_dir)
    print("✅ Entraînement terminé avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    main(args.config)