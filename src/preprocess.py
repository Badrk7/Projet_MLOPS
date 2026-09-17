import argparse
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.config import Config

def main(config_path):
    # 1. Chargement de la config
    cfg = Config.from_yaml(config_path)
    raw_path = cfg.data["raw_path"]
    processed_path = cfg.data["processed_path"]
    model_name = cfg.model["name"]
    max_length = cfg.model["max_length"]
    
    print(f"⚙️ Chargement des données brutes depuis {raw_path}...")
    dataset = load_from_disk(raw_path)
    
    print(f"🔄 Initialisation du Tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Fonction de tokenisation
    def tokenize_function(examples):
        # Le dataset dair-ai/emotion utilise la colonne 'text'
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    
    print("⏳ Tokenisation en cours (optimisée par lots)...")
    # L'argument batched=True permet d'accélérer drastiquement le traitement
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Sauvegarde des données transformées
    os.makedirs(processed_path, exist_ok=True)
    print(f"💾 Sauvegarde des données tokenisées dans {processed_path}...")
    tokenized_datasets.save_to_disk(processed_path)
    
    print("✅ Preprocessing (Tokenisation) terminé avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    main(args.config)