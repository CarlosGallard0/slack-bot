import os
import sys
import shutil
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.store import RAGSystem

load_dotenv()

def ingest():
    persist_directory = "db"
    
    print("--- RAG Ingestion Tool ---")
    
    if os.path.exists(persist_directory):
        choice = input(f"Database found in '{persist_directory}'. Do you want to overwrite it? (y/N): ").lower()
        if choice == 'y':
            print(f"Removing old database at {persist_directory}...")
            shutil.rmtree(persist_directory)
        else:
            print("Existing database will be kept. Adding new documents if any...")

    rag = RAGSystem()
    
    print("Starting ingestion process...")
    rag.initialize()
    
    print("Success! Data has been indexed and saved to disk.")
    print("The bot will now start faster as it will load the pre-built index.")

if __name__ == "__main__":
    ingest()

