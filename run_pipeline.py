import os
import sys
import argparse
import subprocess

stages = [
    "scripts/run_stage1_ner.py",
    # "scripts/run_stage2_classify.py",   # Skipped: classification done in Stage 1
    # "scripts/run_stage3_assertions.py", # Skipped: assertions set to [] by default
    "scripts/run_stage4_rxnorm.py",
    "scripts/run_stage5_icd10.py",
    "scripts/run_stage6_merge.py"
]

def main():
    parser = argparse.ArgumentParser(description="Run full clinical entity pipeline")
    parser.add_argument("--mock", action="store_true", help="Run entire pipeline in mock/offline mode (without LLM server)")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_LLM"] = "true"
        print("\n" + "="*60)
        print(" RUNNING PIPELINE IN MOCK / OFFLINE MODE (NO LLM SERVER)")
        print("="*60 + "\n")

    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Running {stage}...")
        print(f"{'='*50}\n")
        
        cmd = [sys.executable, stage]
        if args.mock:
            cmd.append("--mock")

        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n[ERROR] {stage} failed with exit code {result.returncode}.")
            print("Stopping pipeline.")
            sys.exit(result.returncode)
            
    print("\n[SUCCESS] Pipeline completed successfully!")

if __name__ == "__main__":
    main()
