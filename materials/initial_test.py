import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

import ollama

import tiktoken

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

#from transformers import AutoTokenizer

def embed_text(text):
    request = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return request.embedding

# -------------------------------- #

book = "/Users/noah/REPOS/epub_to_text/input/The Penguin History of the Twentieth Century_The History of.epub"
def extract_epub_text(epub_path):
    book = epub.read_epub(epub_path)
    raw_chapters = [i.get_content() for i in book.get_items() if i.get_type() == ITEM_DOCUMENT]

    for chapter in raw_chapters:
        soup = BeautifulSoup(chapter, 'html.parser')

        print(soup.get_text())


def chunk_text(text, enc, token_size, overlap_tokens):

    tokens = enc.encode(text)
    num_tokens = len(tokens)

    chunks = []
    for start in range(0, num_tokens, token_size):
        start = max(0, start - overlap_tokens)
        end = min(num_tokens, (start + (token_size + overlap_tokens)))

        text_chunk = enc.decode(tokens[start:end])
        chunks.append(text_chunk.strip())

    return chunks


def embed_chunks(chunks : list[str], collection):

    embeddings = []
    for chunk in chunks:
        print('   embedding chunk')
        embedding = embed_text(chunk)

        if embedding:
            embeddings.append(embedding)

    print("adding embeddings to database")
    collection.add(
        ids=[f"id{i}" for i in range(0, len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )


# ------------------------------
# TEST HARNESS #


txt_path = "/Users/noah/REPOS/epub_to_text/output/The Penguin History of the Twentieth Century_trimmed.txt"
with open(txt_path, 'r') as f:
    text = f.read()

print("chunking text. . .")
enc = tiktoken.get_encoding("cl100k_base")
chunks = chunk_text(
    text=text,
    enc=enc,
    token_size=500,
    overlap_tokens=100
)

# trim chunks for test
#chunks = chunks[:50]

print("embedding text and adding to database. . .")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="collection")

embed_chunks(chunks, collection)


query_text = "democracy prevailing"
query_embedding = embed_text(query_text)


result = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print(result)
