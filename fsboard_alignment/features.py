"""Frozen MiCT-RANet18-FSBoard CNN features of the letterbox videos, cached for the CTC head (train.py).

    python -m fsboard_alignment.features [--part 0/1]

One forward pass per video (bf16, channels-last, fixed-shape batches; loader workers only decode).  Per frame the
512-channel feature map is pooled three ways: mean, max, and mean weighted by a motion prior (where the hands
move).  Frame pairs are averaged, so features are at half the video's frame rate (15 per second).
Writes shards of 500 clips: features/part<i>-<n>.npy (steps, 1536) float16 and .json (md5s, offsets, fps).
Resumable: clips already in a shard are skipped.  --part i/n takes every n-th clip, to split the work
over n GPUs.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from simple_video_utils.frames import read_frames_batched
from simple_video_utils.metadata import video_metadata
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import FEATURES, WEIGHTS, clips, letterbox
from .mictresnet import get_mictresnet

KERNELS = [((5, 3, 3), (5, 3, 3))] * 4  # the released FSBoard Large config (conf_fsboard_large_256p.ini)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) * 255
SIZE = 160  # frames are resized to this; it aligns as well as the model's 224 and is nearly twice as fast
SHARD = 500
FRAMES = 1280  # padded frames per batch
BUCKET = 64  # batches pad to a multiple of this many frames, so cuDNN autotuning sees few shapes


def cnn():
    """The CNN of the released MiCT-RANet18-FSBoard checkpoint: eval mode, channels-last weights, on the GPU."""
    model = get_mictresnet("resnet18", (5, 3, 3), KERNELS, (1, 1, 1, 1), pretrained=False)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("cnn.")})
    for mod in model.modules():
        if isinstance(mod, nn.Conv3d):
            mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last_3d)
        elif isinstance(mod, nn.Conv2d):
            mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
    return model.cuda().eval()


class Videos(Dataset):
    """md5 -> (uint8 frames (T, 256, 256, 3), fps, md5), or None if the video can't be decoded."""
    def __init__(self, md5s):
        self.md5s = md5s

    def __len__(self):
        return len(self.md5s)

    def __getitem__(self, i):
        path = letterbox(self.md5s[i])
        if not os.path.exists(path):  # a dead mount must stop the run, not look like unreadable videos
            raise FileNotFoundError(path)
        try:
            return torch.from_numpy(read_frames_batched(path)), float(video_metadata(path).fps), self.md5s[i]
        except (RuntimeError, ValueError):
            print("unreadable:", path, flush=True)
            return None


def padded(n):
    return -(-n // BUCKET) * BUCKET


def batches(videos):
    """Groups a video stream into batches of fixed shape: videos whose lengths pad to the same multiple of BUCKET,
    FRAMES // padded length of them per batch.  The partial batches follow at the end."""
    queues = {}
    for video in videos:
        if video is None:
            continue
        T = padded(len(video[0]))  # noqa: N806
        queue = queues.setdefault(T, [])
        queue.append(video)
        if len(queue) >= max(1, FRAMES // T):
            yield queue
            queues[T] = []
    yield from filter(None, queues.values())


def motion_prior(rgb):
    """Where things move, in place of the original model's optical flow: absolute frame difference of the grayscale
    video at half resolution, min-max normalized per frame, averaged over 3 frames, pooled to the CNN's feature
    map size.  rgb (T, 3, SIZE, SIZE) in 0-255 -> (T, SIZE / 16, SIZE / 16) in 0-255."""
    gray = F.interpolate(rgb.mean(1, keepdim=True), size=SIZE // 2, mode="bilinear", align_corners=False)
    diff = torch.zeros_like(gray)
    diff[1:] = (gray[1:] - gray[:-1]).abs()
    lo, hi = diff.amin((1, 2, 3), keepdim=True), diff.amax((1, 2, 3), keepdim=True)
    diff = (diff - lo) / (hi - lo).clamp(min=1e-6)
    smooth = F.avg_pool1d(diff.flatten(1).T[None], 3, 1, 1, count_include_pad=True)[0].T.view_as(diff)
    return 255 * F.adaptive_avg_pool2d(smooth, SIZE // 16)[:, 0]


def collate(batch):
    """Videos -> GPU tensors: normalized frames (B, T, 3, SIZE, SIZE), channels-last; motion priors; lengths."""
    lengths = [len(video) for video, _, _ in batch]
    T = padded(max(lengths))  # noqa: N806
    imgs = torch.zeros(len(batch), T, 3, SIZE, SIZE, device="cuda")
    maps = torch.zeros(len(batch), T, SIZE // 16, SIZE // 16, device="cuda")
    for i, (video, _, _) in enumerate(batch):
        v = video.cuda(non_blocking=True).permute(0, 3, 1, 2).float()
        v = F.interpolate(v, size=SIZE, mode="bilinear", align_corners=False)
        imgs[i, :len(v)] = (v - MEAN.cuda()) / STD.cuda()
        maps[i, :len(v)] = motion_prior(v)
    return imgs.transpose(1, 2).contiguous(memory_format=torch.channels_last_3d).transpose(1, 2), maps, lengths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args(argv)
    torch.backends.cudnn.benchmark = True
    os.makedirs(FEATURES, exist_ok=True)
    part, parts = map(int, a.part.split("/"))
    model = cnn()
    done = {m for path in glob.glob(f"{FEATURES}/*.json") for m in json.load(open(path))["md5s"]}
    todo = [m for m in list(clips())[part::parts] if m not in done]
    n = len(glob.glob(f"{FEATURES}/part{part}-*.json"))
    print(f"{len(todo)} clips to do ({len(done)} done)", flush=True)
    feats, md5s, fps, frames, t0 = [], [], [], 0, time.time()

    def write():
        nonlocal feats, md5s, fps, n
        name = f"{FEATURES}/part{part}-{n:04d}"
        np.save(name + ".npy", np.concatenate(feats))
        json.dump(dict(md5s=md5s, offsets=np.cumsum([0] + [len(f) for f in feats]).tolist(), fps=fps),
                  open(name + ".json.tmp", "w"))
        os.rename(name + ".json.tmp", name + ".json")  # the json marks the shard done
        print(f"shard {n}: {len(md5s)} clips, {frames / (time.time() - t0):.0f} frames/s", flush=True)
        feats, md5s, fps, n = [], [], [], n + 1

    loader = DataLoader(Videos(todo), batch_size=None, num_workers=6, prefetch_factor=4)
    for batch in batches(loader):
        imgs, maps, lengths = collate(batch)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            f = model(imgs).float()  # (B, 512, T, h, w)
        w = (maps + 1e-3)[:, None]
        f = torch.cat([f.mean((3, 4)), f.amax((3, 4)), (f * w).sum((3, 4)) / w.sum((3, 4))], 1).transpose(1, 2)
        for i, (_, rate, md5) in enumerate(batch):
            x = f[i, :lengths[i]]
            if len(x) % 2:
                x = torch.cat([x, x[-1:]])
            feats.append(x.view(len(x) // 2, 2, -1).mean(1).half().cpu().numpy())
            md5s.append(md5)
            fps.append(rate)
        frames += sum(lengths)
        if len(md5s) >= SHARD:
            write()
    if md5s:
        write()


if __name__ == "__main__":
    main()
