# analysis/run_all_analysis.py
import subprocess
from pathlib import Path
import sys

def main():
    print("Running Hypothesis Test Analysis...")
    try:
        subprocess.run([sys.executable, "analysis/hypothesis_test.py"], check=True)
        print("All analysis completed.")
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()
