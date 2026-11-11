"""Small character-level CTC head over the cached video features (features.py).

    python -m fsboard_alignment.train <run> [--holdout 0|1] [--layers 6] [--context 0] [--minutes 60]

Trains on every clip, or with --holdout on the other half only (data.fold), so that this head's reading of the
held-out half is a reading of clips it never saw.  --context 8 --layers 2 gives a head that sees about a second
around each output: it cannot learn to skip a repeat by looking ahead, so it reads both attempts.
Writes runs/<run>/{model.pt,log.txt}.  Training runs for --minutes with a cosine schedule.
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn

from .ctc import distance, greedy
from .data import FEATURES, RUNS, clips, fold

BATCH, LR = 32, 1e-3


class Head(nn.Module):
    def __init__(self, dim, vocab, hid=256, layers=6, context=0):
        super().__init__()
        self.context = context  # attention reach in outputs per layer; 0 = whole clip
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.Sequential(nn.Conv1d(hid, hid, 5, padding=2), nn.GELU())
        layer = nn.TransformerEncoderLayer(hid, hid // 64, hid * 4, 0.1, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hid)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, lengths):
        h = self.conv(self.inp(x).transpose(1, 2)).transpose(1, 2)
        T, H = h.shape[1:]  # noqa: N806
        pos = torch.arange(T, device=h.device)[:, None] / 10000 ** (torch.arange(0, H, 2, device=h.device) / H)
        h = h + torch.cat([pos.sin(), pos.cos()], -1)
        mask = torch.arange(T, device=h.device)[None] >= lengths[:, None]
        near = torch.arange(T, device=h.device)
        band = (near[:, None] - near[None]).abs() > self.context if self.context else None
        return self.out(self.norm(self.enc(h, mask=band, src_key_padding_mask=mask)))  # (B, T, V) logits


def load():
    """-> [(memory-mapped shard, start, end, fps, clip metadata with its md5)] for every clip with features."""
    meta, out = clips(), []
    for path in sorted(glob.glob(f"{FEATURES}/*.json")):
        info, shard = json.load(open(path)), np.load(path[:-5] + ".npy", mmap_mode="r")
        for i, md5 in enumerate(info["md5s"]):
            out.append((shard, info["offsets"][i], info["offsets"][i + 1], info["fps"][i], meta[md5] | {"md5": md5}))
    return out


def pad(batch):
    lengths = torch.tensor([c[2] - c[1] for c in batch])
    x = torch.zeros(len(batch), int(lengths.max()), batch[0][0].shape[1])
    for i, c in enumerate(batch):
        x[i, :lengths[i]] = torch.from_numpy(np.asarray(c[0][c[1]:c[2]], np.float32))
    return x.cuda(), lengths.cuda()


@torch.no_grad()
def log_probs(model, batch):
    """Yields (clip, (T, V) log-probs) for every clip, batched by length."""
    model.eval()
    order = sorted(batch, key=lambda c: c[2] - c[1])
    for k in range(0, len(order), 64):
        chunk = order[k:k + 64]
        x, lengths = pad(chunk)
        with torch.autocast("cuda", torch.bfloat16):
            lp = model(x, lengths).float().log_softmax(-1).cpu().numpy()
        for i, c in enumerate(chunk):
            yield c, lp[i, :int(lengths[i])]
    model.train()


def accuracy(model, batch, idx):
    edits = total = 0
    for c, lp in log_probs(model, batch):
        ref = [idx[ch] for ch in c[4]["text"]]
        edits, total = edits + distance(greedy(lp.argmax(-1).tolist()), ref), total + len(ref)
    return 1 - edits / max(1, total)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--holdout", type=int, choices=(0, 1))
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--context", type=int, default=0, help="attention reach per layer, in outputs (0 = whole clip)")
    ap.add_argument("--minutes", type=float, default=60)
    a = ap.parse_args(argv)
    random.seed(0), torch.manual_seed(0)
    out = f"{RUNS}/{a.run}"
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/log.txt", "a", buffering=1)

    def log(*m):
        s = " ".join(map(str, m))
        print(s, flush=True), logf.write(s + "\n")

    data = load()
    chars = ["<blank>"] + sorted({ch for c in data for ch in c[4]["text"]})
    idx = {c: i for i, c in enumerate(chars)}
    train = [c for c in data if fold(c[4]["md5"]) != a.holdout]
    held = [c for c in data if fold(c[4]["md5"]) == a.holdout][:1000]
    model = Head(data[0][0].shape[1], len(chars), layers=a.layers, context=a.context).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ctc = nn.CTCLoss(zero_infinity=True)
    log(vars(a), f"{len(train)} training clips, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    step, t0, done = 0, time.time(), False
    while not done:
        random.shuffle(train)  # batches of similar length: sort chunks of 50 batches, then shuffle the batches
        chunks = [sorted(train[k:k + BATCH * 50], key=lambda c: c[2] - c[1]) for k in range(0, len(train), BATCH * 50)]
        batches = [ch[k:k + BATCH] for ch in chunks for k in range(0, len(ch), BATCH)]
        random.shuffle(batches)
        for batch in batches:
            frac = (time.time() - t0) / 60 / a.minutes
            if frac >= 1:
                done = True
                break
            lr = LR * min(1, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr
            x, lengths = pad(batch)
            targets = [torch.tensor([idx[ch] for ch in c[4]["text"]]) for c in batch]
            with torch.autocast("cuda", torch.bfloat16):
                logits = model(x, lengths)
            loss = ctc(logits.float().log_softmax(-1).transpose(0, 1), torch.cat(targets).cuda(), lengths,
                       torch.tensor([len(t) for t in targets]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                log(f"step {step} loss {loss.item():.3f} lr {lr:.2e} {time.time() - t0:.0f}s")
    torch.save({"model": model.state_dict(), "chars": chars, "layers": a.layers, "context": a.context},
               f"{out}/model.pt")
    if held:
        log(f"FINAL step {step} held-out character accuracy {accuracy(model, held, idx):.1%} ({len(held)} clips)")


if __name__ == "__main__":
    main()
