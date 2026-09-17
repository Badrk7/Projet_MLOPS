Détection d'Émotions (NLP) avec DistilBERT — Pipeline MLOps

Pipeline MLOps de bout en bout pour classifier des émotions dans du texte à l'aide du modèle DistilBERT et du dataset Hugging Face dair-ai/emotion (6 classes : Tristesse, Joie, Amour, Colère, Peur, Surprise).
Le projet intègre l'écosystème Hugging Face (transformers, datasets, evaluate) pour l'entraînement, le service du modèle avec FastAPI, et une interface interactive Streamlit.

Architecture du Projet

```
Projet_MLOPS/
├── configs/
│   └── config.yaml          # Nom du modèle, dataset, hyperparamètres d'entraînement
├── src/
│   ├── config.py            # Classe Config (chargement dynamique du YAML)
│   ├── get-data.py          # Téléchargement du dataset HF -> data/raw (format Arrow)
│   ├── preprocess.py        # Tokenisation des textes (AutoTokenizer) -> data/processed
│   ├── train.py             # Fine-tuning avec HF Trainer + sauvegarde locale du modèle
│   ├── api.py               # API FastAPI servant le modèle via pipeline("text-classification")
│   └── utils.py             # Helpers divers
├── app/
│   └── streamlit_app.py     # Interface web Streamlit appelant l'API FastAPI
├── tests/
│   └── test_pipeline.py     # Tests unitaires (pytest)
├── data/                    # Données brutes et tokenisées (gitignoré)
├── models/                  # Poids finaux du modèle (gitignoré)
├── Makefile                 # Raccourcis de commandes
├── Dockerfile               # Conteneurisation de l'API
├── requirements.txt         # Dépendances (PyTorch, Transformers, FastAPI...)
└── README.md                # Documentation
```

Étapes du pipeline : get-data → preprocess → train → API → Streamlit
Installation

Créer un environnement virtuel propre (.venv) et installer les dépendances nécessaires au Deep Learning et aux APIs :

python -m pip install -U pip -r requirements.txt

Exécution du Pipeline MLOps

Le traitement des données et l'entraînement se font de manière séquentielle en lisant la configuration centrale configs/config.yaml.

1. Ingestion des données :

python -m src.get-data --config configs/config.yaml

2. Prétraitement (Tokenisation) :

python -m src.preprocess --config configs/config.yaml

3. Entraînement du modèle (Fine-tuning) :

python -m src.train --config configs/config.yaml

(Le modèle final sera sauvegardé automatiquement dans le dossier models/distilbert-emotion avec ses tenseurs de poids et son tokenizer).
Déploiement : L'API FastAPI

Le backend charge le modèle sauvegardé localement en utilisant les pipelines d'Hugging Face et ouvre un endpoint /predict pour traiter le texte.

Lancement du serveur :

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

Test avec cURL :

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so incredibly happy today, everything is perfect!"}'

Interface Web : Streamlit

L'interface utilisateur permet de tester la détection d'émotions en temps réel de manière visuelle. L'API FastAPI doit impérativement tourner en parallèle dans un autre terminal.

Lancement de l'interface :

python -m streamlit run app/streamlit_app.py --server.port 8501

Docker

Construire et exécuter l'image Docker contenant l'API et le modèle entraîné :

make build_docker
make run_docker

Note : Le dossier models/ est requis pour la construction de l'image. L'étape d'entraînement (train.py) doit avoir été exécutée localement au moins une fois.
Les commandes peuvent être lancées avec le Makefile