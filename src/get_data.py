import argparse
import os
from datasets import load_dataset
from src.config import Config

def main(config_path):
    # 1. Chargement de notre nouvelle configuration
    cfg = Config.from_yaml(config_path)
    dataset_name = cfg.data["dataset_name"]
    raw_path = cfg.data["raw_path"]
    
    print(f"📥 Téléchargement du dataset NLP : {dataset_name}...")
    
    # 2. Téléchargement depuis le Hub Hugging Face
    # trust_remote_code=True est recommandé pour les datasets récents
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    # 3. Création du dossier de destination s'il n'existe pas
    os.makedirs(raw_path, exist_ok=True)
    
    # 4. Sauvegarde ultra-rapide au format local Arrow
    print(f"💾 Sauvegarde locale dans le dossier : {raw_path}...")
    dataset.save_to_disk(raw_path)
    
    print("✅ Téléchargement terminé avec succès !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    main(args.config)