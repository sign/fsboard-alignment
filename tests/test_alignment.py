import numpy as np

from fsboard_alignment.ctc import distance, greedy, viterbi
from fsboard_alignment.suspects import errors, missing, reason, repeat

REF = "call for more details"


def kind(pred, ref=REF):
    return reason(repeat(pred, ref), ref)


def test_repeats():
    assert kind("call for me call for more details").startswith("false_start:call for m")
    assert kind("7640 doctor bowmenbowen", "7640 doctor bowen") == "false_end:bowen"
    assert kind("call for mofor more details") == "repeat:for mo"
    assert repeat(REF, REF) is None
    assert repeat(REF + "sss", REF) is None


def test_missing_words_and_shared_errors():
    assert missing("always cover all bases", "always cover all the bases") == ["the"]
    assert missing("bad for the e nvironment", "bad for the environment") == []
    readings = ["the contton is high", "the conttos is high", "the contton is high"]
    assert set.intersection(*(errors(r, "the cotton is high") for r in readings)) == {("", 5, "n")}
    assert not errors("ddinner", "dinner")
    readings = ["the sumrises", "the sunris es", "the sun rises"]
    assert not set.intersection(*(errors(r, "the sun rises") for r in readings))


def test_viterbi_places_each_character_on_its_spike():
    logp = np.full((8, 3), -9.0)
    for t, c in enumerate([0, 1, 1, 0, 2, 0, 2, 0]):
        logp[t, c] = 0.0
    assert viterbi(logp, [1, 2, 2]) == [(1, 2), (4, 4), (6, 6)]
    assert greedy(logp.argmax(-1).tolist()) == [1, 2, 2]
    assert distance("kitten", "sitting") == 3
