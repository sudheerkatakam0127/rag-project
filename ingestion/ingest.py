import os
from ingestion.loader import load_pdfs_from_drive
from ingestion.chunker import process_documents, save_chunks_to_json
from ingestion.indexer import create_index, index_chunks

# Default folder ID — replace with yours
DEFAULT_FOLDER_ID = "1ZDSKsxf_jgCzj9-8Fpgm2f8JiIu7dg2X"


def run_ingestion():
    print("=== Starting ingestion ===")

    # Allow API to override folder ID via environment variable
    folder_id = os.environ.get("OVERRIDE_FOLDER_ID", DEFAULT_FOLDER_ID)
    print(f"Using folder ID: {folder_id}")

    # Step 1: Download PDFs from Google Drive
    print("\n[1] Loading PDFs from Google Drive...")
    downloaded_files = load_pdfs_from_drive(folder_id)

    # Step 2: Extract text and chunk
    print("\n[2] Extracting text and chunking...")
    chunks = process_documents(downloaded_files)

    # Step 3: Save to JSON
    print("\n[3] Saving chunks...")
    save_chunks_to_json(chunks)

    # Step 4: Index into Elasticsearch
    print("\n[4] Indexing into Elasticsearch...")
    create_index()
    index_chunks()

    print("\n=== Ingestion complete ===")
    return chunks


if __name__ == "__main__":
    run_ingestion()