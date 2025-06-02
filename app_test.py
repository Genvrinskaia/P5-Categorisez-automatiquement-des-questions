import numpy as np
import pytest

# On va simuler une structure équivalente à app.predict_top_5_dl
from app import tags_list

# Simulation d'un modèle
class MockModel:
    def predict_proba(self, X):
        # Retourne un tableau numpy de proba uniformes
        return np.array([[0.01 * i for i in range(1, len(tags_list)+1)]])

# Simulation d'un scaler
class MockScaler:
    def transform(self, X):
        return X  # Ne fait rien

# Simulation du USE (produit un vecteur compatible)
def mock_use(texts):
    return np.random.rand(1, 512)  # shape (1, 512)

# Fonction à tester (copiée/modifiée depuis app.predict_top_5_dl)
def predict_top_5_dl_mock(text):
    embed = mock_use([text])
    embed_scaled = MockScaler().transform(embed)
    probs = MockModel().predict_proba(embed_scaled)[0]

    tag_probs = list(zip(tags_list, probs))
    tag_probs_sorted = sorted(tag_probs, key=lambda x: x[1], reverse=True)
    return tag_probs_sorted[:5]

# Test
def test_prediction_mock():
    question = "Comment créer une API avec Python ?"
    result = predict_top_5_dl_mock(question)

    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(tag, tuple) and isinstance(tag[0], str) and isinstance(tag[1], float) for tag in result)

