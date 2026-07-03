import argparse
import json
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR / "input"
DATA_DIR = BASE_DIR / "data"
STAGE1_DIR = DATA_DIR / "stage1_ner"
STAGE2_DIR = DATA_DIR / "stage2_classify"

VALID_TYPES = {"TRIỆU_CHỨNG", "THUỐC", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

def review_stage1_file(doc_id: str):
    in_file = INPUT_DIR / f"{doc_id}.txt"
    out_file = STAGE1_DIR / f"{doc_id}.json"
    
    if not in_file.exists() or not out_file.exists():
        print(f"Files for doc {doc_id} not found.")
        return 0, 0
        
    with open(in_file, "r", encoding="utf-8") as f:
        source_text = f.read()
        
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entities = data.get("entities", [])
    
    print(f"\n=== Record {doc_id} ===")
    print(f"Source length: {len(source_text)} chars | Entities: {len(entities)}\n")
    
    errors = 0
    for ent in entities:
        text = ent["text"]
        start, end = ent["position"]
        
        extracted_substring = source_text[start:end]
        if extracted_substring == text:
            print(f"[{start}:{end}] \"{text}\"  ✓ match")
        else:
            print(f"[{start}:{end}] \"{text}\"  ✗ mismatch (found: \"{extracted_substring}\")")
            errors += 1
            
    print(f"\nPosition errors: {errors}/{len(entities)} ({(errors/len(entities)*100) if entities else 0:.1f}%)")
    return errors, len(entities)

def review_stage1_summary():
    if not STAGE1_DIR.exists():
         print("Stage 1 output directory not found.")
         return
         
    total_errors = 0
    total_entities = 0
    
    for file_path in STAGE1_DIR.glob("*.json"):
        doc_id = file_path.stem
        errors, entities = review_stage1_file(doc_id)
        total_errors += errors
        total_entities += entities
        
    if total_entities > 0:
        print(f"\n=== SUMMARY ===")
        print(f"Total Position errors: {total_errors}/{total_entities} ({(total_errors/total_entities)*100:.1f}%)")
    else:
        print("No entities found.")


def review_stage2_file(doc_id: str):
    in_file = INPUT_DIR / f"{doc_id}.txt"
    out_file = STAGE2_DIR / f"{doc_id}.json"
    
    if not in_file.exists() or not out_file.exists():
        print(f"Files for doc {doc_id} not found.")
        return
        
    with open(in_file, "r", encoding="utf-8") as f:
        source_text = f.read()
        
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Support both bare array and {"entities": [...]} formats
    entities = data if isinstance(data, list) else data.get("entities", [])
    type_counts = Counter()
    invalid_types = 0
    pos_errors = 0
    
    print(f"\n=== Record {doc_id} (Stage 2) ===")
    print(f"Source length: {len(source_text)} chars | Entities: {len(entities)}\n")
    
    for ent in entities:
        text = ent["text"]
        start, end = ent["position"]
        etype = ent.get("type", "MISSING")
        
        type_counts[etype] += 1
        
        # Check position
        extracted = source_text[start:end]
        pos_ok = "✓" if extracted == text else "✗"
        if extracted != text:
            pos_errors += 1
        
        # Check type validity
        type_ok = "✓" if etype in VALID_TYPES else "✗"
        if etype not in VALID_TYPES:
            invalid_types += 1
        
        print(f"[{start}:{end}] [{etype}] \"{text}\"  pos:{pos_ok} type:{type_ok}")
    
    print(f"\n--- Type Distribution ---")
    for t in sorted(type_counts.keys()):
        marker = "  " if t in VALID_TYPES else "⚠ "
        print(f"  {marker}{t}: {type_counts[t]}")
    
    print(f"\nPosition errors: {pos_errors}/{len(entities)}")
    print(f"Invalid types: {invalid_types}/{len(entities)}")


def review_stage2_summary():
    if not STAGE2_DIR.exists():
        print("Stage 2 output directory not found.")
        return
        
    total_entities = 0
    total_type_counts = Counter()
    total_pos_errors = 0
    total_invalid = 0
    file_count = 0
    
    for file_path in sorted(STAGE2_DIR.glob("*.json")):
        doc_id = file_path.stem
        in_file = INPUT_DIR / f"{doc_id}.txt"
        
        if not in_file.exists():
            continue
        
        with open(in_file, "r", encoding="utf-8") as f:
            source_text = f.read()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support both bare array and {"entities": [...]} formats
        entities = data if isinstance(data, list) else data.get("entities", [])
        file_count += 1
        
        for ent in entities:
            total_entities += 1
            etype = ent.get("type", "MISSING")
            total_type_counts[etype] += 1
            
            start, end = ent["position"]
            if source_text[start:end] != ent["text"]:
                total_pos_errors += 1
            if etype not in VALID_TYPES:
                total_invalid += 1
    
    print(f"\n=== Stage 2 Summary ({file_count} files) ===")
    print(f"Total entities: {total_entities}")
    print(f"\n--- Type Distribution ---")
    for t in sorted(total_type_counts.keys()):
        marker = "  " if t in VALID_TYPES else "⚠ "
        pct = (total_type_counts[t] / total_entities * 100) if total_entities else 0
        print(f"  {marker}{t}: {total_type_counts[t]} ({pct:.1f}%)")
    print(f"\nPosition errors: {total_pos_errors}/{total_entities}")
    print(f"Invalid types: {total_invalid}/{total_entities}")


def main():
    parser = argparse.ArgumentParser(description="Review Stage Output")
    parser.add_argument("--stage", type=int, required=True, help="Stage number to review (e.g. 1, 2)")
    parser.add_argument("--id", type=str, help="Document ID to review")
    parser.add_argument("--summary", action="store_true", help="Print summary of all processed files")
    
    args = parser.parse_args()
    
    if args.stage == 1:
        if args.id:
            review_stage1_file(args.id)
        elif args.summary:
            review_stage1_summary()
        else:
            print("Please provide either --id <doc_id> or --summary")
    elif args.stage == 2:
        if args.id:
            review_stage2_file(args.id)
        elif args.summary:
            review_stage2_summary()
        else:
            print("Please provide either --id <doc_id> or --summary")
    else:
        print(f"Review for Stage {args.stage} is not yet implemented.")

if __name__ == "__main__":
    main()

