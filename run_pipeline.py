import subprocess
import sys

stages = [
    "scripts/run_stage1_ner.py",
    "scripts/run_stage2_classify.py",
    "scripts/run_stage3_assertions.py",
    "scripts/run_stage4_rxnorm.py",
    "scripts/run_stage5_icd10.py",
    "scripts/run_stage6_merge.py"
]

def main():
    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Running {stage}...")
        print(f"{'='*50}\n")
        
        result = subprocess.run([sys.executable, stage])
        
        if result.returncode != 0:
            print(f"\n[ERROR] {stage} failed with exit code {result.returncode}.")
            print("Stopping pipeline.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Pipeline completed successfully!")

if __name__ == "__main__":
    main()
