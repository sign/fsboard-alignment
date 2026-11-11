"""The full run: features, the five heads, their alignments, then the two output files.

    python -m fsboard_alignment

Each step is skipped if its output exists, so an interrupted run picks up where it stopped.
"""
import os

from . import align, features, suspects, train
from .data import RUNS

HEADS = {  # run -> training arguments
    "all": ["--minutes", "90"],  # aligns every clip
    "fold0": ["--holdout", "0", "--minutes", "75"],  # each fold head reads the half it never trained on
    "fold1": ["--holdout", "1", "--minutes", "75"],
    "local0": ["--holdout", "0", "--layers", "2", "--context", "8", "--minutes", "30"],  # same, ~1 s of context
    "local1": ["--holdout", "1", "--layers", "2", "--context", "8", "--minutes", "30"],
}

if __name__ == "__main__":
    features.main([])
    for run, args in HEADS.items():
        if not os.path.exists(f"{RUNS}/{run}/model.pt"):
            train.main([run, *args])
        if not os.path.exists(f"{RUNS}/{run}/align.jsonl"):
            align.main([run])
    suspects.main()
