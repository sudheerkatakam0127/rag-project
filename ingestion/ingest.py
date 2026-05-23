from ingestion.loader import load_pdfs_from_drive
from ingestion.chunker import process_documents, save_chunks_to_json

# Paste your Google Drive folder ID here (from Part 2 Step 4)
FOLDER_ID = "1ZDSKsxf_jgCzj9-8Fpgm2f8JiIu7dg2X"

def run_ingestion():
    print("=== Starting ingestion ===")

    # Step 1: Download PDFs from Google Drive
    print("\n[1] Loading PDFs from Google Drive...")
    downloaded_files = load_pdfs_from_drive(FOLDER_ID)

    # Step 2: Extract text and chunk
    print("\n[2] Extracting text and chunking...")
    chunks = process_documents(downloaded_files)

    # Step 3: Save to JSON
    print("\n[3] Saving chunks...")
    save_chunks_to_json(chunks)

    print("\n=== Ingestion complete ===")
    return chunks

if __name__ == "__main__":
    run_ingestion()