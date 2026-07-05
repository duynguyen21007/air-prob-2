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
STAGE3_DIR = DATA_DIR / "stage3_assertions"
STAGE4_DIR = DATA_DIR / "stage4_rxnorm"

VALID_TYPES = {"TRIỆU_CHỨNG", "THUỐC", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
VALID_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
NO_ASSERTION_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}


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


def review_stage3_file(doc_id: str):
    in_file = INPUT_DIR / f"{doc_id}.txt"
    out_file = STAGE3_DIR / f"{doc_id}.json"
    
    if not in_file.exists() or not out_file.exists():
        print(f"Files for doc {doc_id} not found.")
        return
        
    with open(in_file, "r", encoding="utf-8") as f:
        source_text = f.read()
        
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entities = data if isinstance(data, list) else data.get("entities", [])
    
    print(f"\n=== Record {doc_id} (Stage 3) ===")
    print(f"Source length: {len(source_text)} chars | Entities: {len(entities)}\n")
    
    assertion_issues = 0
    for ent in entities:
        text = ent["text"]
        start, end = ent["position"]
        etype = ent.get("type", "MISSING")
        assertions = ent.get("assertions", [])
        
        # Check position
        extracted = source_text[start:end]
        pos_ok = "✓" if extracted == text else "✗"
        
        # Check assertion validity
        issues = []
        if etype in NO_ASSERTION_TYPES and assertions:
            issues.append(f"⚠ {etype} should have empty assertions")
        for a in assertions:
            if a not in VALID_ASSERTIONS:
                issues.append(f"⚠ invalid assertion: {a}")
        
        if issues:
            assertion_issues += 1
        
        assertions_str = str(assertions) if assertions else "[]"
        issue_str = " " + "; ".join(issues) if issues else ""
        print(f"[{start}:{end}] [{etype}] \"{text}\"  assertions={assertions_str}  pos:{pos_ok}{issue_str}")
    
    print(f"\nAssertion issues: {assertion_issues}/{len(entities)}")


def review_stage3_summary():
    if not STAGE3_DIR.exists():
        print("Stage 3 output directory not found.")
        return
        
    total_entities = 0
    assertion_counts = Counter()
    type_assertion_counts = Counter()  # (type, assertion) pairs
    lab_with_assertions = 0
    invalid_assertions = 0
    file_count = 0
    
    for file_path in sorted(STAGE3_DIR.glob("*.json")):
        doc_id = file_path.stem
        in_file = INPUT_DIR / f"{doc_id}.txt"
        
        if not in_file.exists():
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entities = data if isinstance(data, list) else data.get("entities", [])
        file_count += 1
        
        for ent in entities:
            total_entities += 1
            etype = ent.get("type", "MISSING")
            assertions = ent.get("assertions", [])
            
            for a in assertions:
                assertion_counts[a] += 1
                type_assertion_counts[(etype, a)] += 1
                if a not in VALID_ASSERTIONS:
                    invalid_assertions += 1
            
            if etype in NO_ASSERTION_TYPES and assertions:
                lab_with_assertions += 1
    
    print(f"\n=== Stage 3 Summary ({file_count} files) ===")
    print(f"Total entities: {total_entities}")
    
    print(f"\n--- Assertion Distribution ---")
    for a in sorted(assertion_counts.keys()):
        marker = "  " if a in VALID_ASSERTIONS else "⚠ "
        print(f"  {marker}{a}: {assertion_counts[a]}")
    
    no_assertion = sum(1 for fp in STAGE3_DIR.glob("*.json")
                       for ent in (json.load(open(fp, encoding="utf-8")) if isinstance(json.load(open(fp, encoding="utf-8")), list) else json.load(open(fp, encoding="utf-8")).get("entities", []))
                       if not ent.get("assertions", []))
    print(f"  (no assertions): {no_assertion}")
    
    print(f"\n--- Assertions by Type ---")
    for (t, a) in sorted(type_assertion_counts.keys()):
        print(f"  {t} + {a}: {type_assertion_counts[(t, a)]}")
    
    print(f"\nLab entities with assertions (should be 0): {lab_with_assertions}")
    print(f"Invalid assertion values: {invalid_assertions}")


def review_stage4_file(doc_id: str):
    in_file = INPUT_DIR / f"{doc_id}.txt"
    out_file = STAGE4_DIR / f"{doc_id}.json"
    
    if not in_file.exists() or not out_file.exists():
        print(f"Files for doc {doc_id} not found.")
        return
        
    with open(in_file, "r", encoding="utf-8") as f:
        source_text = f.read()
        
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    entities = data if isinstance(data, list) else data.get("entities", [])
    
    print(f"\n=== Record {doc_id} (Stage 4) ===")
    print(f"Source length: {len(source_text)} chars | Entities: {len(entities)}\n")
    
    for ent in entities:
        text = ent["text"]
        start, end = ent["position"]
        etype = ent.get("type", "MISSING")
        candidates = ent.get("candidates", [])
        
        if etype == "THUỐC":
            cand_str = str(candidates) if candidates else "[] (NO MATCH)"
            print(f"[{start}:{end}] [THUỐC] \"{text}\"  rxnorm: {cand_str}")
        else:
            if candidates:
                print(f"[{start}:{end}] [{etype}] \"{text}\"  ⚠ SHOULD NOT HAVE CANDIDATES: {candidates}")


def review_stage4_summary():
    if not STAGE4_DIR.exists():
        print("Stage 4 output directory not found.")
        return
        
    total_thuoc = 0
    thuoc_with_rxnorm = 0
    file_count = 0
    non_thuoc_with_candidates = 0
    
    for file_path in sorted(STAGE4_DIR.glob("*.json")):
        file_count += 1
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        entities = data if isinstance(data, list) else data.get("entities", [])
        
        for ent in entities:
            etype = ent.get("type", "MISSING")
            candidates = ent.get("candidates", [])
            if etype == "THUỐC":
                total_thuoc += 1
                if candidates:
                    thuoc_with_rxnorm += 1
            else:
                if candidates:
                    non_thuoc_with_candidates += 1
                    
    print(f"\n=== Stage 4 Summary ({file_count} files) ===")
    print(f"Total THUỐC entities: {total_thuoc}")
    if total_thuoc > 0:
        pct = (thuoc_with_rxnorm / total_thuoc) * 100
        print(f"THUỐC mapped to RxNorm: {thuoc_with_rxnorm}/{total_thuoc} ({pct:.1f}%)")
    print(f"Non-THUỐC with candidates (should be 0): {non_thuoc_with_candidates}")


def main():
    parser = argparse.ArgumentParser(description="Review Stage Output")
    parser.add_argument("--stage", type=int, required=True, help="Stage number to review (e.g. 1, 2, 3)")
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
    elif args.stage == 3:
        if args.id:
            review_stage3_file(args.id)
        elif args.summary:
            review_stage3_summary()
        else:
            print("Please provide either --id <doc_id> or --summary")
    elif args.stage == 4:
        if args.id:
            review_stage4_file(args.id)
        elif args.summary:
            review_stage4_summary()
        else:
            print("Please provide either --id <doc_id> or --summary")
    else:
        print(f"Review for Stage {args.stage} is not yet implemented.")

if __name__ == "__main__":
    main()

