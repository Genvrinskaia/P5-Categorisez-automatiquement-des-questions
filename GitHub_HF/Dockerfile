# Utilise une image officielle Python
FROM python:3.9-slim

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crée et définit un répertoire de travail
WORKDIR /app

# Copie les fichiers requirements en premier pour optimiser le cache Docker
COPY requirements.txt /app/

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le contenu de ton repo dans /app (y compris models)
COPY . /app

# Crée le dossier .streamlit avec les bons droits
RUN mkdir -p /app/.streamlit

# Ajoute un fichier de config minimal pour Streamlit
RUN echo "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
\n\
" > /app/.streamlit/config.toml

# Expose le port 7860 (facultatif mais recommandé)
EXPOSE 7860

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
