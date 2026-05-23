import fitz  # this is PyMuPDF — imported as fitz
import tiktoken
import json
import os

def extract_text_from_pdf(local_path):
    """
    Opens a PDF and extracts all text page by page.
    Returns the full text as one string.
    """
    doc = fitz.open(local_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"

    doc.close()
    return full_text


def split_into_chunks(text, chunk_size=300, overlap=50):
    """
    Splits text into chunks of ~300 tokens with 50-token overlap.
    
    Why overlap? So that if an answer spans the boundary between 
    two chunks, we don't miss it — each chunk shares 50 tokens 
    with the next one.
    """
    # cl100k_base is the tokenizer used by most modern LLMs
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Convert text into a list of token integers
    tokens = tokenizer.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        # Decode this slice of tokens back into readable text
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move forward by (chunk_size - overlap) so next chunk overlaps
        start += chunk_size - overlap

    return chunks


def process_documents(downloaded_files):
    """
    Takes the list of downloaded PDFs from loader.py,
    extracts text, chunks it, attaches metadata.
    Returns a list of chunk dicts ready for Elasticsearch.
    """
    all_chunks = []
    chunk_id = 0

    for doc in downloaded_files:
        print(f"\nProcessing: {doc['file_name']}")

        # Step 1: Extract raw text from the PDF
        text = extract_text_from_pdf(doc['local_path'])
        print(f"  Extracted {len(text)} characters of text")

        # Step 2: Split into chunks
        chunks = split_into_chunks(text)
        print(f"  Split into {len(chunks)} chunks")

        # Step 3: Attach metadata to each chunk
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                'chunk_id': f"chunk_{chunk_id}",
                'file_name': doc['file_name'],
                'drive_url': doc['drive_url'],
                'chunk_index': i,
                'text': chunk_text
            })
            chunk_id += 1

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


def save_chunks_to_json(chunks, output_path='data/chunks.json'):
    """
    Saves all chunks to a JSON file so we can inspect them
    and use them in the next step (Elasticsearch indexing).
    """
    os.makedirs('data', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")