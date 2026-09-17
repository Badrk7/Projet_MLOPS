from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# 1. Initialisation de l'API
app = FastAPI(
    title="API NLP - Émotions",
    description="API MLOps pour la classification d'émotions avec DistilBERT",
    version="1.0"
)

# 2. Chargement du modèle depuis le dossier local (généré par train.py)
print("⏳ Chargement du modèle en mémoire...")
try:
    # Le pipeline gère automatiquement la tokenisation et l'inférence
    classifier = pipeline("text-classification", model="models/distilbert-emotion")
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    classifier = None

# 3. Format des données attendues par l'API
class TextRequest(BaseModel):
    text: str

# 4. Le point d'entrée pour les prédictions
@app.post("/predict")
def predict_emotion(request: TextRequest):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas chargé.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")
    
    # Inférence
    result = classifier(request.text)[0]
    
    return {
        "label": result["label"],
        "score": result["score"]
    }

# Optionnel : un point d'entrée pour vérifier que l'API est en ligne
@app.get("/")
def health_check():
    return {"status": "ok", "model": "DistilBERT Emotion"}