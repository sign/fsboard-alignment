"""CTC decoding without a model: greedy reading, forced alignment, edit distance."""
import numpy as np


def greedy(ids):
    """Per-frame argmax ids -> the reading: repeats merged, blanks (0) dropped."""
    return [i for k, i in enumerate(ids) if i and (k == 0 or i != ids[k - 1])]


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def viterbi(logp, target):
    """logp (T, V) log-probs, target char ids -> per target char (first frame, last frame), or None if not reached."""
    states = np.array([0] + [s for c in target for s in (c, 0)])
    S, T = len(states), len(logp)  # noqa: N806
    skip = np.zeros(S, bool)  # a character state may come straight from the previous character (no blank between)
    skip[3::2] = states[3::2] != states[1:-2:2]
    score = np.full(S, -np.inf)
    score[:2] = logp[0, states[:2]]
    back = np.zeros((T, S), np.int8)  # 0: stay, 1: from s-1, 2: from s-2
    for t in range(1, T):
        options = np.stack([score, np.r_[-np.inf, score[:-1]],
                            np.where(skip, np.r_[-np.inf, -np.inf, score[:-2]], -np.inf)])
        back[t] = options.argmax(0)
        score = options.max(0) + logp[t, states]
    s = S - 1 if S < 2 or score[S - 1] >= score[S - 2] else S - 2
    path = [s]
    for t in range(T - 1, 0, -1):
        s -= back[t, s]
        path.append(s)
    spans = {}
    for t, s in enumerate(path[::-1]):
        if s % 2:
            spans[s // 2] = (spans.get(s // 2, (t, t))[0], t)
    return [spans.get(k) for k in range(len(target))]
