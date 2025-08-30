import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_glue_pbfa import main as eval_main

if __name__ == "__main__":
    eval_main()
