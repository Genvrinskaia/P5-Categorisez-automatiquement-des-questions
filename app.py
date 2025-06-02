# ─── Imports ──────────────────────────────────────────────────────────────────
import sys
import streamlit as st
import tensorflow_hub as hub
import numpy as np
import joblib
import os
import mlflow
from mlflow.sklearn import load_model

# ─── Config TF Hub Cache ──────────────────────────────────────────────────────
if os.name == "nt":  # Windows (local)
    os.environ["TFHUB_CACHE_DIR"] = r"C:\envs\P5\tfhub_cache"
else:  # Linux (CI/CD)
    os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub_cache"

# ─── MLflow ───────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("file:///C:/envs/mlflow_P5/mlruns")

# ─── Chargements statiques ─────────────────────────────────────────────────────
st.write(f"Environnement Python actif : {os.environ.get('VIRTUAL_ENV', sys.prefix)}")

# Chargement du modèle de classification depuis MLflow
model_dir = os.path.join(os.path.dirname(__file__), "models", "USE_RL2_model")
model = joblib.load(os.path.join(model_dir, "model.joblib"))

# Chargement du scaler pour normaliser les embeddings
scaler_USE = joblib.load(os.path.join(model_dir, "scaler_USE.joblib"))

# MultiLabelBinarizer
mlb = joblib.load(os.path.join(model_dir, "P5_mlb.pkl"))
tags_list = mlb.classes_

# ─── Chargement paresseux du USE ───────────────────────────────────────────────
@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

use_model = None

def get_use_model():
    global use_model
    if use_model is None:
        use_model = load_use_model()
    return use_model

# ─── Interface ────────────────────────────────────────────────────────────────
st.title("Gaëlle_Genvrin_P5_API – Deep Learning + USE")
st.write("Entrez votre question StackOverflow ci-dessous 👇")

question = st.text_area("Question StackOverflow", height=150)

# ─── Prédiction ───────────────────────────────────────────────────────────────
def predict_top_5_dl(text):
    model_use = get_use_model()
    embed = model_use([text])  # Liste → Tensor
    embed_scaled = scaler_USE.transform(embed)  # Applique la normalisation

    probs = model.predict_proba(embed_scaled)[0]  # Prédiction des probabilités

    tag_probs = list(zip(tags_list, probs))
    tag_probs_sorted = sorted(tag_probs, key=lambda x: x[1], reverse=True)

    return tag_probs_sorted[:35]

# ─── Affichage ────────────────────────────────────────────────────────────────
if st.button("Générer les tags"):
    if not question.strip():
        st.warning("❗ Veuillez entrer une question.")
    else:
        top_preds = predict_top_5_dl(question)
        st.subheader("🏷️ Top 35 tags prédits")
        for tag, p in top_preds:
            st.write(f"**{tag}** — {p:.6f}")
