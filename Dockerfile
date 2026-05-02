FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake spaCy language models into the image so the first request is instant.
# xx_ent_wiki_sm is the multilingual fallback used for Norwegian (no dedicated model).
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download de_core_news_sm && \
    python -m spacy download fr_core_news_sm && \
    python -m spacy download sv_core_news_sm && \
    python -m spacy download xx_ent_wiki_sm

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
