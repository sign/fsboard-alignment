"""FSboard's clips and where files are: inputs from the environment (see .env.example), products under the
working directory."""
import json
import os
from pathlib import Path

WEIGHTS = "weights/MiCT-RANet18_FSBoard_Large_69m_256p.pth"
FEATURES, RUNS, OUTPUT = "features", "runs", "output"


def clips():
    """md5 -> {file, text, subset, split, signer} for every clip in FSboard's metadata, in md5 order."""
    out = {}
    for path in sorted(Path(os.environ["FSBOARD_PATH"]).glob("*/metadata/*.json")):
        subset, split = path.stem.split("-")
        for c in json.loads(path.read_text()):
            out[c["clipFileMd5"]] = dict(file=f"{subset}/video_clips/{path.stem}/{c['clipFilename']}",
                                         text=c["phrase"].lower(), subset=subset, split=split, signer=c["signerId"])
    return dict(sorted(out.items()))


def original(file):
    """A clip's video as FSboard ships it; file is the clip's path in clips()."""
    return os.path.join(os.environ["FSBOARD_PATH"], file)


def letterbox(md5):
    return os.environ["FSBOARD_LETTERBOX_TEMPLATE"].format(md5=md5)


def fold(md5):
    """Which of the two cross-fitting halves a clip is in."""
    return int(md5, 16) % 2
