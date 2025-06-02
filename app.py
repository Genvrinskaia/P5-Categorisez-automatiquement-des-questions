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
else:  # Linux (CI/CD, ex: GitHub Actions)
    os.environ["TFHUB_CACHE_DIR"] = "/tmp/tfhub_cache"


# ─── MLflow ───────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("file:///C:/envs/mlflow_P5/mlruns")

# ─── Chargements ──────────────────────────────────────────────────────────────
st.write(f"Environnement Python actif : {os.environ.get('VIRTUAL_ENV', sys.prefix)}")

# 1. USE (Universal Sentence Encoder)
@st.cache_resource
def load_use_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

use_model = load_use_model()

# 2. Chargement du modèle de classification depuis MLflow
model_dir = r"C:/envs/P5_models/USE_RL2_model"
model = joblib.load(os.path.join(model_dir, "model.joblib"))

# 3. Chargement du scaler pour normaliser les embeddings
scaler_USE = joblib.load(r"C:\envs\P5\scaler_USE.joblib")

# 4. MultiLabelBinarizer
mlb = joblib.load(r"C:\envs\P5\P5_mlb.pkl")
tags_list = mlb.classes_

# ─── Interface ────────────────────────────────────────────────────────────────
st.title("Gaëlle_Genvrin_P5_API – Deep Learning + USE")
st.write("Entrez votre question StackOverflow ci-dessous 👇")

question = st.text_area("Question StackOverflow", height=150)

# ─── Prédiction ───────────────────────────────────────────────────────────────
def predict_top_5_dl(text):
    embed = use_model([text])  # Liste → Tensor
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
