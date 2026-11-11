"""A trained head's reading and forced alignment of every clip.

    python -m fsboard_alignment.align <run>

Writes runs/<run>/align.jsonl, one clip per line: its metadata, `pred` (the greedy reading, without the label),
and `chars`, the label's characters with start/end seconds (where the model fires for the character) and p, the
mean posterior there.  A character the alignment could not place has no times.
"""
import argparse
import json

import numpy as np
import torch

from .ctc import greedy, viterbi
from .data import RUNS
from .train import Head, load, log_probs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args(argv)
    ck = torch.load(f"{RUNS}/{a.run}/model.pt", map_location="cuda", weights_only=True)
    chars, data = ck["chars"], load()
    idx = {c: i for i, c in enumerate(chars)}
    model = Head(data[0][0].shape[1], len(chars), layers=ck["layers"], context=ck["context"]).cuda()
    model.load_state_dict(ck["model"])
    with open(f"{RUNS}/{a.run}/align.jsonl", "w") as out:
        for c, lp in log_probs(model, data):
            seconds = 2 / c[3]  # one output per frame pair
            target = [idx[ch] for ch in c[4]["text"]]
            aligned = [dict(c=chars[t], start=round(s[0] * seconds, 3), end=round((s[1] + 1) * seconds, 3),
                            p=round(float(np.exp(lp[s[0]:s[1] + 1, t]).mean()), 3)) if s else dict(c=chars[t])
                       for t, s in zip(target, viterbi(lp, target), strict=True)]
            out.write(json.dumps(c[4] | dict(duration=round(len(lp) * seconds, 3), chars=aligned,
                                             pred="".join(chars[i] for i in greedy(lp.argmax(-1).tolist())))) + "\n")


if __name__ == "__main__":
    main()
