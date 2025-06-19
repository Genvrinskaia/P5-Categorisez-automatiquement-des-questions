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
os.environ["TFHUB_CACHE_DIR"] = r"C:\envs\P5\tfhub_cache"

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

