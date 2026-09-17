from transformers import pipeline

def build_inference_pipeline(model_path="models/distilbert-emotion"):
    """Construit et retourne le pipeline NLP pour la prédiction d'émotions."""
    try:
        nlp_pipe = pipeline("text-classification", model=model_path)
        return nlp_pipe
    except Exception as e:
        print(f"❌ Erreur : Impossible de charger le modèle depuis {model_path}. Détails : {e}")
        return None