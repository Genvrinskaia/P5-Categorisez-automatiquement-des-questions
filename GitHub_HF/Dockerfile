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

# Copie les fichiers dans le conteneur
COPY . /app

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Crée le dossier .streamlit avec les bons droits
RUN mkdir -p /app/.streamlit

# Ajoute un fichier de config minimal si besoin
RUN echo "\
[server]\n\
headless = true\n\
port = 7860\n\
enableCORS = false\n\
\n\
" > /app/.streamlit/config.toml

# Lance l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
