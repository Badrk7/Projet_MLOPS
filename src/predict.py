from src.pipeline import build_inference_pipeline

def main():
    print("🚀 Initialisation du pipeline de prédiction...")
    nlp_pipe = build_inference_pipeline()
    
    if not nlp_pipe:
        return

    # Données de test
    texts = [
        "I am so incredibly happy today, everything is perfect!",
        "This is the worst experience of my life, I am so angry.",
        "I feel a bit sad and lonely this evening."
    ]
    
    print("🔮 Prédictions en mode Batch :\n")
    predictions = nlp_pipe(texts)
    
    for text, pred in zip(texts, predictions):
        print(f"Texte : '{text}'")
        print(f"-> Émotion : {pred['label']} (Confiance : {pred['score']:.2%})\n")

if __name__ == "__main__":
    main()