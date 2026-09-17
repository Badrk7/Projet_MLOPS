import pytest
from src.config import Config
from src.api import TextRequest

def test_config_loads_expected_nlp_parameters():
    """Vérifie que la configuration charge bien les bons paramètres pour DistilBERT"""
    cfg = Config.from_yaml("configs/config.yaml")
    
    assert cfg.data["dataset_name"] == "dair-ai/emotion"
    assert cfg.model["name"] == "distilbert-base-uncased"
    assert cfg.model["num_labels"] == 6
    assert cfg.training["epochs"] > 0

def test_api_text_request_validation():
    """Vérifie que le modèle de données de l'API (Pydantic) valide bien les entrées"""
    # Une entrée valide
    req = TextRequest(text="I am extremely happy!")
    assert req.text == "I am extremely happy!"
    
    # Une entrée vide ne devrait techniquement pas bloquer Pydantic au niveau du type, 
    # mais vérifier qu'elle est bien parsée comme une chaîne vide.
    req_empty = TextRequest(text="")
    assert req_empty.text == ""

def test_config_missing_file_raises_error():
    """Vérifie que l'absence du fichier yaml lève bien une erreur"""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("configs/fichier_imaginaire.yaml")