import sys
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from src.config import SAMPLE_IDS, DATA_DIR
from src.retrieval.rxnorm_hybrid_search import RxNormHybridSearcher

def main():
    # We can read from stage3_assertions or stage4_rxnorm, both contain THUỐC entities.
    # stage4_rxnorm is the directory used in export_retrieval_csv.py.
    data_dir_to_read = DATA_DIR / "stage4_rxnorm"
    data_csv_path = BASE_DIR / "data_rxnorm.csv"
    chroma_persist_dir = DATA_DIR / "chroma_rxnorm_db"
    
    print("Initializing RxNormHybridSearcher...")
    searcher = RxNormHybridSearcher(str(data_csv_path), str(chroma_persist_dir))
    
    # 1. Collect all unique THUỐC strings
    unique_drugs = set()
    for doc_id in SAMPLE_IDS:
        in_file = data_dir_to_read / f"{doc_id}.json"
        if not in_file.exists():
            continue
        with open(in_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for ent in data:
                if ent["type"] == "THUỐC":
                    unique_drugs.add(ent["text"])
                    
    drugs_list = list(unique_drugs)
    print(f"Found {len(drugs_list)} unique drug entities.")
    
    # 2. Write to CSV
    csv_file = BASE_DIR / "rxnorm_retrieval_result.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity Text", "Candidates"])
        
        for drug in tqdm(drugs_list, desc="Retrieving Candidates"):
            top_5 = searcher.get_top_k_rxcuis(drug, k=5)
            
            candidates_lines = []
            for i, res in enumerate(top_5, 1):
                candidates_lines.append(f"Top {i}: {res}")
                
            candidates_str = "\n".join(candidates_lines)
            writer.writerow([drug, candidates_str])
            
    print(f"Results successfully saved to {csv_file}")

if __name__ == "__main__":
    main()
