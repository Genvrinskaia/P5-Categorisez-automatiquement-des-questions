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

