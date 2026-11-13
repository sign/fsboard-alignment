"""The two deliverables, from the five heads' alignment files (align.py).

    python -m fsboard_alignment.suspects

Both are written to output/.

alignments.json  {md5: {file, text, subset, split, signer, duration, chars: [{c, start, end, p}]}}
    Forced alignment of the label by the head trained on everything (runs/all).  start/end (seconds) are where the
    model fires for the character (CTC spikes are short); a character's segment runs from its start to the next
    character's start.  p is the mean posterior there.

suspects.json  [{md5, file, text, subset, split, signer, tier, level, reasons, reading, reading_local,
                 reading_trained, accuracy, accuracy_trained}]
    Clips whose label may not match the signing, worst first: tier "clear" (level 3), "likely" (2), "unlikely" (1).
    `reading` is the greedy reading of the whole-clip head
    that never trained on the clip (fold0 / fold1); `reading_local` the same from the head that only sees about
    a second of context (local0 / local1).  Whole-clip heads learn to stay silent through a long repeat, because
    the labels never include one; the local head cannot look ahead, so it reads both attempts.
    `reading_trained` is the all-data head's.  accuracy = 1 - edit distance / label length.  Reasons, clear tier:
      false_start:<x>  either unseen reading goes back and spells on from there: it is closer to
      false_end:<x>    label[:j] + label[i:] (j - i >= 3) than to the label by at least 3 edits.
      repeat:<x>       false_start: back to the beginning (i <= 2), x = what was signed before restarting.
                       false_end: the ending signed again (j >= len - 1), x = that ending.
                       repeat: anything in between, x = the part signed twice.
      misread          the head trained on the clip still reads under 80% of the label (accuracy_trained), or
                       the whole-clip head that never saw it reads under 50% (accuracy) and found no repeat
    likely tier (not clear):
      missing_word:<w>   any of the three readings skips this label word entirely: without it the label is closer
                         to the reading by the word's full length (spaces ignored)
    unlikely tier (neither of the above):
      same_error:<x>><y> all three readings, also the one trained on the label, read y where the label has x
                         (spaces and doubled letters ignored).  One reading's slip, like sumrises, is not flagged.
"""
import json
import os
import re
from difflib import SequenceMatcher

from .ctc import distance
from .data import OUTPUT, RUNS, fold

TIERS = ["unlikely", "likely", "clear"]  # level 1, 2, 3


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


def missing(pred, ref):
    """Label words the reading skips entirely: dropping the word saves all its characters (spaces ignored)."""
    words, pred = ref.split(" "), pred.replace(" ", "")
    base = distance(pred, "".join(words))
    return [w for i, w in enumerate(words) if len(words) > 1 and base >= len(w)
            and base - distance(pred, "".join(words[:i] + words[i + 1:])) == len(w)]


def errors(pred, ref):
    """{(label part, its start, what was read)} where the reading departs from the label; spaces and doubled
    letters are ignored."""
    pred, ref = (re.sub(r"(.)\1+", r"\1", s.replace(" ", "")) for s in (pred, ref))
    ops = SequenceMatcher(None, ref, pred, autojunk=False).get_opcodes()
    return {(ref[i:j], i, pred[k:m]) for tag, i, j, k, m in ops if tag != "equal"}


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
        tier = "clear"
        if not reasons:
            readings = (reading, local, trained)
            reasons = [f"missing_word:{w}" for w in dict.fromkeys(w for r in readings for w in missing(r, ref))]
            tier = "likely"
        if not reasons:
            shared = set.intersection(*(errors(r, ref) for r in readings))
            reasons = [f"same_error:{x}>{y}" for x, _, y in sorted(shared, key=lambda e: e[1])]
            tier = "unlikely"
        if reasons:
            suspects.append(dict(md5=md5) | {k: row[k] for k in ("file", "text", "subset", "split", "signer")} | dict(
                tier=tier, level=TIERS.index(tier) + 1, reasons=reasons, reading=reading, reading_local=local,
                reading_trained=trained,
                accuracy=round(acc, 3), accuracy_trained=round(acc_trained, 3)))
    suspects.sort(key=lambda s: (-s["level"], s["reasons"][0].split(":")[0], s["accuracy"]))
    os.makedirs(OUTPUT, exist_ok=True)
    json.dump(alignments, open(f"{OUTPUT}/alignments.json", "w"))
    json.dump(suspects, open(f"{OUTPUT}/suspects.json", "w"), indent=1, ensure_ascii=False)
    count = {t: sum(s["tier"] == t for s in suspects) for t in TIERS[::-1]}
    count |= {k: sum(r.split(":")[0] == k for s in suspects for r in s["reasons"])
              for k in ("false_start", "false_end", "repeat", "misread")}
    print(f"{len(alignments)} clips aligned; {len(suspects)} suspects: {count}")


if __name__ == "__main__":
    main()
