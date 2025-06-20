---
title: Tags Question StackOverflow
emoji: "🏷️"
colorFrom: "indigo"
colorTo: "purple"
sdk: streamlit
sdk_version: "1.25.0"
app_file: app.py
pinned: true
---

# Tags_Question_StackOverflow

Application Streamlit pour prédire automatiquement les tags des questions StackOverflow  
Basée sur un modèle Deep Learning avec Universal Sentence Encoder (USE).

---

## Objectif

L’objectif est de proposer une **API légère, rapide et utilisable en production**, capable de prédire des **tags multi-label** à partir de l’intitulé et du corps d’une question technique sur StackOverflow.

---

## Modèle utilisé

- **Universal Sentence Encoder (USE : encodage sémantique)** pour transformer la question en vecteur dense
- **Régression Logistique multi-label** avec `scikit-learn`
- Le modèle a été entraîné et sauvegardé avec **MLflow** pour le suivi des versions

---

## Technologies & outils

- `Streamlit` pour l'interface utilisateur
- `TensorFlow` et `TensorFlow Hub` pour charger USE
- `scikit-learn` pour la modélisation
- `MLflow` pour le suivi des expériences et la gestion du modèle
- `Git` & `GitHub` pour le versioning
- `GitHub Actions` pour le déploiement continu sur Hugging Face Spaces

---

## Usage

Entrez une question (titre + corps), puis cliquez sur **Générer les tags** pour obtenir les prédictions.

---

## Déploiement

Le projet est déployé automatiquement sur **Hugging Face Spaces** à l’adresse :

[https://huggingface.co/spaces/Genvrin/Tags_Question_StackOverflow](https://huggingface.co/spaces/Genvrin/Tags_Question_StackOverflow)

---
