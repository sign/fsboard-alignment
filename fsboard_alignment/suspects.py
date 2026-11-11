"""The two deliverables, from the five heads' alignment files (align.py).

    python -m fsboard_alignment.suspects

Both are written to output/.

alignments.json  {md5: {file, text, subset, split, signer, duration, chars: [{c, start, end, p}]}}
    Forced alignment of the label by the head trained on everything (runs/all).  start/end (seconds) are where the
    model fires for the character (CTC spikes are short); a character's segment runs from its start to the next
    character's start.  p is the mean posterior there.

suspects.json  [{md5, file, text, subset, split, signer, reasons, reading, reading_local, reading_trained,
                 accuracy, accuracy_trained}]
    Clips whose label likely does not match the signing.  `reading` is the greedy reading of the whole-clip head
    that never trained on the clip (fold0 / fold1); `reading_local` the same from the head that only sees about
    a second of context (local0 / local1).  Whole-clip heads learn to stay silent through a long repeat, because
    the labels never include one; the local head cannot look ahead, so it reads both attempts.
    `reading_trained` is the all-data head's.  accuracy = 1 - edit distance / label length.  Reasons:
      false_start:<x>  either unseen reading goes back and spells on from there: it is closer to
      false_end:<x>    label[:j] + label[i:] (j - i >= 3) than to the label by at least 3 edits.
      repeat:<x>       false_start: back to the beginning (i <= 2), x = what was signed before restarting.
                       false_end: the ending signed again (j >= len - 1), x = that ending.
                       repeat: anything in between, x = the part signed twice.
      misread          the head trained on the clip still reads under 80% of the label (accuracy_trained), or
                       the whole-clip head that never saw it reads under 50% (accuracy) and found no repeat
"""
import json
import os

from .ctc import distance
from .data import OUTPUT, RUNS, fold


def repeat(pred, ref):
    """(i, j) if pred reads like ref[:j] + ref[i:], the signer going back from j to i (at least 3 characters) and
    spelling on from there; else None."""
    extra = len(pred) - len(ref)
    if extra < 3:
        return None
    base = distance(pred, ref)
    # tries every (i, j) whose repeat is about as long as the reading's surplus: fine for the few thousand
    # over-long readings; vectorize the edit distance if this ever has to run on everything
    spans = [(i, j) for i in range(len(ref)) for j in range(i + 3, len(ref) + 1) if j - i <= extra + 2]
    best = min(spans, key=lambda s: (distance(pred, ref[:s[1]] + ref[s[0]:]), s[1] - s[0]), default=None)
    return best if best and base - distance(pred, ref[:best[1]] + ref[best[0]:]) >= 3 else None


def reason(span, ref):
    i, j = span
    # the reading rarely pins the span to the character, so allow some slack at either end of the label
    if i <= 2:
        return f"false_start:{ref[:j]}"
    return f"false_end:{ref[i:]}" if j >= len(ref) - 1 else f"repeat:{ref[i:j]}"


def rows(path):
    with open(path) as f:
        yield from map(json.loads, f)


def main():
    unseen = {kind: {r["md5"]: r["pred"] for k in (0, 1) for r in rows(f"{RUNS}/{kind}{k}/align.jsonl")
                     if fold(r["md5"]) == k} for kind in ("fold", "local")}
    alignments, suspects = {}, []
    for row in rows(f"{RUNS}/all/align.jsonl"):
        md5, ref, trained = row.pop("md5"), row["text"], row.pop("pred")
        alignments[md5] = row
        reading, local = unseen["fold"][md5], unseen["local"][md5]
        acc, acc_trained = 1 - distance(reading, ref) / len(ref), 1 - distance(trained, ref) / len(ref)
        span = repeat(reading, ref) or repeat(local, ref)
        reasons = [reason(span, ref)] if span else []
        if acc_trained < 0.8 or (acc < 0.5 and not span):
            reasons.append("misread")
        if reasons:
            suspects.append(dict(md5=md5) | {k: row[k] for k in ("file", "text", "subset", "split", "signer")} | dict(
                reasons=reasons, reading=reading, reading_local=local, reading_trained=trained,
                accuracy=round(acc, 3), accuracy_trained=round(acc_trained, 3)))
    suspects.sort(key=lambda s: (s["reasons"][0].split(":")[0], s["accuracy"]))
    os.makedirs(OUTPUT, exist_ok=True)
    json.dump(alignments, open(f"{OUTPUT}/alignments.json", "w"))
    json.dump(suspects, open(f"{OUTPUT}/suspects.json", "w"), indent=1, ensure_ascii=False)
    count = {k: sum(r.split(":")[0] == k for s in suspects for r in s["reasons"])
             for k in ("false_start", "false_end", "repeat", "misread")}
    print(f"{len(alignments)} clips aligned; {len(suspects)} suspects: {count}")


if __name__ == "__main__":
    main()
