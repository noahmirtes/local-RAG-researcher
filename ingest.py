import os

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import tiktoken
from hashlib import sha256

from embedding import embed_text
from utils import get_files_with_extensions, write_to_txt
from models import IngestDocument

from chroma import get_collection

# -------------------------------- #

def extract_epub_text(path, min_chapter_len : int = 200, chapter_separator=False):
    """
    Extract the text from an epub and save to .txt in a structured format

    min_chapter_len : the minimim number of characters a chapter must be to be extracted
    chapter_separated : include a dashed line to s
    """
    book = epub.read_epub(path)

    # skip if these words detected in. Must be lowercase
    skip_checks = [
        "table of contents",
        "foreword",
        "afterword",
        "index",
        "acknowledgements",
        "about the author",
        "also available",
        "published",
        "bookseller",
    ]

    text_blocks = []
    for idref, linear in book.spine:
        # skip non-linear items
        if linear == "no":
            continue

        # get the actual document by ID
        item = book.get_item_with_id(idref)
        if item.get_type() != ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")

        # get paragraph chunks
        chapter_pars = soup.find_all('p')

        # extract paragraph text
        paragraphs = [p.get_text() for p in chapter_pars]
        paragraphs = [p for p in paragraphs if p]

        chapter_text = None
        if len(paragraphs) > 0:
            chapter_text = "\n\n".join(paragraphs)
        if not chapter_text:
            chapter_text = soup.get_text()
        
        # skip short chapters
        chapter_len = len(chapter_text)
        if chapter_len < min_chapter_len:
            continue

        # trim chapter text for skip check
        ref_text = chapter_text[:300]
        if any(skip in ref_text for skip in skip_checks):
            continue

        # store chapter text
        text_blocks.append(chapter_text.strip())

    
    # join all text blocks and return
    if chapter_separator:
        separator = f"\n\n{"-" * 50}\n\n"
    else:
        separator = "\n\n"
    return book.title, separator.join(text_blocks)


def extract_txt_text():
    pass
def extract_pdf_text():
    pass



def chunk_text(text : str, enc, token_size : int = 800, overlap_tokens : int = 150):

    tokens = enc.encode(text)
    num_tokens = len(tokens)

    chunks = []
    for start in range(0, num_tokens, token_size):
        start = max(0, start - overlap_tokens)
        end = min(num_tokens, (start + (token_size + overlap_tokens)))

        text_chunk = enc.decode(tokens[start:end])
        chunks.append(text_chunk.strip())

    return chunks

# -------------------------------- #




def ingest_text(text : str, source_name : str, collection):
    """
    The main function that receives text, chunks, embeds, and adds to the vector database
    """
    try:
        # HASH INPUT
        book_hash_ojb = sha256(text.encode('utf-8'))
        book_hash = book_hash_ojb.hexdigest()


        # CHUNK
        enc = tiktoken.get_encoding("cl100k_base")
        chunks = chunk_text(
            text=text,
            enc=enc,
            token_size=500,
            overlap_tokens=100
        )


        # EMBED
        documents = []
        embeddings = []
        ids = []
        for i in range(0, len(chunks)):
            c = chunks[i]

            # hash
            chunk_hash_obj = sha256(c.encode('utf-8'))
            chunk_hash = chunk_hash_obj.hexdigest()

            # embed chunk text
            embed = embed_text(c)

            # construct chunk id
            chunk_id = f"{source_name}-{book_hash}-{i}-{chunk_hash}"
            
            documents.append(c)
            embeddings.append(embed)
            ids.append(chunk_id)

    except Exception as e:
        print(f"Failed to chunk text for {source_name} : {e}")
        return


    # ADD THE EMBEDDINGS TO THE DATABASE
    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )
    except Exception as e:
        print(f"Unable to add embeddings to database for {source_name}: {e}")






def main(INGEST_INPUT):

    ingest_files = get_files_with_extensions(INGEST_INPUT, ['.epub','.txt','.pdf'])

    print(f"Extracting text from {len(ingest_files)} files")
    INGEST_DOCUMENTS = []
    for f in ingest_files:
        ext = os.path.splitext(f)[-1].replace(".","")

        if ext == 'epub':
            name, text = extract_epub_text(f)
        elif ext == 'txt':
            text = extract_txt_text()
        elif ext == 'pdf':
            text = extract_pdf_text()

        ingest_document = IngestDocument(
            type=ext,
            path=f,
            name=name,
            text=text
        )
        INGEST_DOCUMENTS.append(ingest_document)


    # loop over all the data models and ingest the text
    print(f"Chunking, embedding, and adding to database the text from {len(ingest_files)} files")
    collection = get_collection()
    for ingest_document in INGEST_DOCUMENTS:
        print(f"    Ingesting {ingest_document.name}. . . ")
        ingest_text(
            text=ingest_document.text,
            source_name=ingest_document.name,
            collection=collection
        )





main("/Volumes/FILES/Noah's Library/Nonfiction/Culture & Politics")

