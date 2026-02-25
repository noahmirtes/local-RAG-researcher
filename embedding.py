# embedding.py

import ollama

def embed_text(text):
    request = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return request.embedding
