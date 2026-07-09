import sys
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, DATA_DIR
from src.retrieval.hybrid_search import HybridSearcher

def main():
    stage4_dir = DATA_DIR / "stage4_rxnorm"
    data_csv_path = BASE_DIR / "data_icds.csv"
    chroma_persist_dir = DATA_DIR / "chroma_db"
    
    print("Initializing HybridSearcher...")
    searcher = HybridSearcher(str(data_csv_path), str(chroma_persist_dir))
    
    # 1. Collect all unique CHẨN_ĐOÁN strings
    unique_diagnoses = set()
    for doc_id in SAMPLE_IDS:
        in_file = stage4_dir / f"{doc_id}.json"
        if not in_file.exists():
            continue
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for ent in data:
                if ent["type"] == "CHẨN_ĐOÁN":
                    unique_diagnoses.add(ent["text"])
                    
    diagnoses_list = list(unique_diagnoses)
    print(f"Found {len(diagnoses_list)} unique diagnosis entities.")
    
    # 2. Write to CSV
    csv_file = BASE_DIR / "retrieval_result.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity Text", "Candidates"])
        
        for diag in tqdm(diagnoses_list, desc="Retrieving Candidates"):
            top_5 = searcher.get_top_k_icds(diag, k=5)
            
            candidates_lines = []
            for i, res in enumerate(top_5, 1):
                candidates_lines.append(f"Top {i}: {res}")
                
            candidates_str = "\n".join(candidates_lines)
            writer.writerow([diag, candidates_str])
            
    print(f"Results successfully saved to {csv_file}")

if __name__ == "__main__":
    main()
