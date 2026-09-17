import argparse
import numpy as np
import evaluate as hf_evaluate
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, AutoTokenizer
from src.config import Config

def compute_metrics(eval_pred):
    metric = hf_evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main(config_path):
    cfg = Config.from_yaml(config_path)
    processed_path = cfg.data["processed_path"]
    output_dir = cfg.training["output_dir"]
    
    print("⚙️ Chargement du jeu de validation...")
    dataset = load_from_disk(processed_path)
    
    print(f"🤖 Chargement du modèle pour évaluation depuis {output_dir}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
    except OSError:
        print("❌ Modèle introuvable. Avez-vous lancé l'entraînement (src.train) ?")
        return

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    
    print("📊 Calcul des performances en cours...")
    results = trainer.evaluate()
    
    print("\n✅ Résultats de l'évaluation :")
    for key, value in results.items():
        print(f"  - {key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)