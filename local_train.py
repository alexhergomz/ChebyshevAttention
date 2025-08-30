import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_glue_imdb import main as train_main

if __name__ == "__main__":
    train_main()
