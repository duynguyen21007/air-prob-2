import os
import sys
import csv
from pathlib import Path
from tqdm import tqdm

# Add the root directory to sys.path so we can import from src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import DATA_DIR

def build_index():
    data_csv_path = BASE_DIR / "data_icds.csv"
    chroma_persist_dir = DATA_DIR / "chroma_db"
    
    if chroma_persist_dir.exists():
        print(f"ChromaDB already exists at {chroma_persist_dir}. If you want to rebuild it, please delete the folder.")
        return

    print(f"Loading ICD-10 data from {data_csv_path}...")
    docs = []
    with open(data_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            icd_code = row.get("icd", "").strip()
            keyword = row.get("kw", "").strip()
            if icd_code and keyword:
                # e5 requires 'passage: ' prefix for database texts
                docs.append(Document(page_content=f"passage: {keyword}", metadata={"icd": icd_code}))

    print(f"Loaded {len(docs)} ICD-10 records.")
    
    print("Initializing HuggingFaceEmbeddings (intfloat/multilingual-e5-base)...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    
    print("Building Chroma vector store (this will take some time for embedding calculation)...")
    
    # We batch the documents to avoid memory issues and provide progress
    batch_size = 500
    vectorstore = None
    
    # Ensure directory exists
    chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding batches"):
        batch = docs[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(chroma_persist_dir),
                collection_name="icd10_collection"
            )
        else:
            vectorstore.add_documents(documents=batch)
            
    print(f"Successfully built and persisted Chroma database at {chroma_persist_dir}")

if __name__ == "__main__":
    build_index()
