import os
import sys
import csv
import re
from pathlib import Path
from tqdm import tqdm

# Attempt to resolve paths depending on where we are running
current_dir = Path.cwd()
if (current_dir / "data_rxnorm.csv").exists():
    # We are running on Colab or in the directory directly containing the CSV
    BASE_DIR = current_dir
    DATA_DIR = current_dir / "data"
else:
    # We are running in the standard local repo structure
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

sys.path.append(str(BASE_DIR))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def preprocess_rxnorm_text(text: str) -> str:
    text = text.lower()
    text = "".join([c if c.isalpha() or c.isdigit() else " " for c in text])
    text = re.sub(r'(\d)([a-z]+)', r'\1 \2', text)
    text = re.sub(r'([a-z]+)(\d)', r'\1 \2', text)
    text = " ".join(text.split())
    return text

def build_index():
    data_csv_path = BASE_DIR / "data_rxnorm.csv"
    chroma_persist_dir = DATA_DIR / "chroma_rxnorm_db"
    
    if chroma_persist_dir.exists():
        print(f"ChromaDB already exists at {chroma_persist_dir}. If you want to rebuild it, please delete the folder.")
        return

    print(f"Loading RxNorm data from {data_csv_path}...")
    docs = []
    with open(data_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxcui = row.get("RXCUI", "").strip()
            keyword = row.get("STR", "").strip()
            if rxcui and keyword:
                keyword = preprocess_rxnorm_text(keyword)
                # e5 requires 'passage: ' prefix for database texts
                docs.append(Document(page_content=f"passage: {keyword}", metadata={"rxcui": rxcui}))

    print(f"Loaded {len(docs)} RxNorm records.")
    
    print("Initializing HuggingFaceEmbeddings (intfloat/multilingual-e5-base)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'batch_size': 256}
    )
    
    print("Building Chroma vector store (this will take some time for embedding calculation)...")
    
    # We batch the documents to avoid memory issues and provide progress
    batch_size = 5000
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
                collection_name="rxnorm_collection"
            )
        else:
            vectorstore.add_documents(documents=batch)
            
    print(f"Successfully built and persisted Chroma database at {chroma_persist_dir}")

if __name__ == "__main__":
    build_index()
