"""Deterministic tests for the claim-aware boundary engine."""

from __future__ import annotations

import unittest

from boundary_engine import (
    END_PAD_MIN,
    END_PAD_MAX,
    align_end_to_words,
    align_start_to_words,
    apply_to_candidates,
    dedupe_identical_resolved_windows,
    find_natural_ending,
    is_open_thought,
    resolve_boundaries,
)


def _words_from_script(script, t0=0.0, word_dur=0.4, gap=0.05):
    """Build Whisper-like word timings from (text, optional_pause_after) tokens."""
    words = []
    t = t0
    for item in script:
        if isinstance(item, (int, float)):
            t += float(item)
            continue
        token = str(item)
        # Keep punctuation attached like Whisper often does.
        start = t
        end = t + word_dur
        words.append({"word": token, "start": round(start, 3), "end": round(end, 3)})
        t = end + gap
    return words


def _segments_from_words(words, group_size=6):
    segments = []
    for index in range(0, len(words), group_size):
        chunk = words[index:index + group_size]
        segments.append({
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(w["word"].strip() for w in chunk),
        })
    return segments


class BoundaryEngineTests(unittest.TestCase):
    def test_01_never_cut_middle_of_word(self):
        words = [
            {"word": "We", "start": 10.0, "end": 10.2},
            {"word": "build", "start": 10.25, "end": 10.7},
            {"word": "fast.", "start": 10.75, "end": 11.2},
        ]
        # Proposed end lands inside "build".
        aligned = align_end_to_words(10.4, words, pad=0.15)
        self.assertGreaterEqual(aligned, 10.7 + END_PAD_MIN - 1e-6)
        self.assertLessEqual(aligned, 10.7 + END_PAD_MAX + 1e-6)
        start = align_start_to_words(10.4, words)
        self.assertEqual(start, 10.25)

        candidate = {
            "start": 10.0,
            "end": 10.4,
            "text": "We build",
            "central_claim": {"start": 10.0, "end": 10.4, "text": "We build fast"},
        }
        resolved = resolve_boundaries(candidate, words=words, segments=_segments_from_words(words))
        self.assertTrue(resolved["boundary_debug"]["word_aligned"])
        # End must not sit inside any word body.
        for word in words:
            self.assertFalse(word["start"] < resolved["end"] < word["end"])

    def test_02_sentence_continues_eight_seconds_after_end(self):
        # Claim ends at ~5s, but the thought completes ~8s later.
        words = _words_from_script([
            "This", "changes", "everything", "because", 0.3,
            "we", "can", "build", "a", "home", "every",
            0.2, "sixty", "seconds", "instead", "of", "hours.",
        ], t0=0.0, word_dur=0.5, gap=0.1)
        # Force a premature end around "because"
        because = next(w for w in words if w["word"] == "because")
        complete = words[-1]
        self.assertGreaterEqual(complete["end"] - because["end"], 7.0)
        candidate = {
            "start": words[0]["start"],
            "end": because["end"],
            "text": "This changes everything because",
            "central_claim": {
                "start": words[0]["start"],
                "end": because["end"],
                "text": "This changes everything because",
            },
            "variant_type": "STANDARD",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 4)
        )
        self.assertGreaterEqual(resolved["end"], complete["end"] - 0.05)
        self.assertTrue(is_open_thought("This changes everything because"))
        self.assertFalse(is_open_thought("we can build a home every sixty seconds instead of hours."))

    def test_03_short_pause_before_payoff(self):
        words = _words_from_script([
            "The", "waitlist", "is", "huge.", 0.9,
            "That", "is", "why", "speed", "matters", "now.",
        ], t0=20.0, word_dur=0.45, gap=0.08)
        first_end = next(w for w in words if w["word"] == "huge.")["end"]
        payoff_end = words[-1]["end"]
        candidate = {
            "start": 20.0,
            "end": first_end,
            "text": "The waitlist is huge.",
            "central_claim": {"start": 20.0, "end": first_end, "text": "The waitlist is huge."},
            "variant_type": "STANDARD",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 5)
        )
        self.assertGreaterEqual(resolved["end"], payoff_end - 0.05)
        self.assertTrue(resolved["boundary_debug"]["payoff_detected"] or resolved["end"] >= payoff_end - 0.05)

    def test_04_punctuation_before_explanation_continues(self):
        words = _words_from_script([
            "It", "costs", "less.", 0.2,
            "Because", "shipping", "drops", "by", "thirty", "five", "percent.",
        ], t0=5.0, word_dur=0.4, gap=0.08)
        punct = next(w for w in words if w["word"] == "less.")["end"]
        explanation = words[-1]["end"]
        candidate = {
            "start": 5.0,
            "end": punct,
            "text": "It costs less.",
            "central_claim": {"start": 5.0, "end": punct, "text": "It costs less."},
            "variant_type": "STANDARD",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 4)
        )
        self.assertGreaterEqual(resolved["end"], explanation - 0.05)

    def test_05_question_followed_by_answer(self):
        words = _words_from_script([
            "Why", "build", "this", "way?", 0.25,
            "Because", "a", "home", "can", "be", "ready", "in", "under", "an", "hour.",
        ], t0=0.0, word_dur=0.4, gap=0.08)
        answer_start = next(w for w in words if w["word"] == "Because")["start"]
        answer_end = words[-1]["end"]
        candidate = {
            "start": answer_start,
            "end": answer_start + 2.0,
            "text": "Because a home can be ready",
            "central_claim": {
                "start": answer_start,
                "end": answer_end,
                "text": "Because a home can be ready in under an hour.",
            },
            "variant_type": "STANDARD",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 5)
        )
        self.assertLessEqual(resolved["start"], words[0]["start"] + 0.05)
        self.assertGreaterEqual(resolved["end"], answer_end - 0.05)
        self.assertIn(
            resolved["boundary_debug"]["start_reason"],
            {"question_before_answer", "minimal_setup_sentence", "claim_sentence_start"},
        )

    def test_06_claim_followed_by_numerical_explanation(self):
        words = _words_from_script([
            "Production", "scales", "fast.", 0.2,
            "We", "make", "a", "home", "every", "60", "seconds",
            "instead", "of", "four", "hours.",
        ], t0=100.0, word_dur=0.4, gap=0.08)
        claim_end = next(w for w in words if w["word"] == "fast.")["end"]
        full_end = words[-1]["end"]
        candidate = {
            "start": 100.0,
            "end": claim_end,
            "text": "Production scales fast.",
            "central_claim": {"start": 100.0, "end": claim_end, "text": "Production scales fast."},
            "variant_type": "EXTENDED",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 5)
        )
        self.assertGreaterEqual(resolved["end"], full_end - 0.05)

    def test_07_but_continuation(self):
        words = _words_from_script([
            "It", "sounds", "impossible", "but", 0.15,
            "the", "factory", "already", "ships", "weekly.",
        ], t0=0.0, word_dur=0.45, gap=0.08)
        but = next(w for w in words if w["word"] == "but")
        full_end = words[-1]["end"]
        candidate = {
            "start": 0.0,
            "end": but["end"],
            "text": "It sounds impossible but",
            "central_claim": {"start": 0.0, "end": but["end"], "text": "It sounds impossible but"},
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 4)
        )
        self.assertGreaterEqual(resolved["end"], full_end - 0.05)
        self.assertTrue(is_open_thought("It sounds impossible but"))

    def test_08_because_continuation(self):
        words = _words_from_script([
            "Costs", "fall", "because", 0.1,
            "modules", "share", "the", "same", "chassis", "design.",
        ], t0=50.0, word_dur=0.4, gap=0.08)
        because = next(w for w in words if w["word"] == "because")
        full_end = words[-1]["end"]
        candidate = {
            "start": 50.0,
            "end": because["end"],
            "text": "Costs fall because",
            "central_claim": {"start": 50.0, "end": because["end"], "text": "Costs fall because"},
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 4)
        )
        self.assertGreaterEqual(resolved["end"], full_end - 0.05)

    def test_09_correct_boundary_unchanged(self):
        words = _words_from_script([
            "Elon", "Musk", "was", "the", "first", "prototype", "recipient.",
        ], t0=33.0, word_dur=0.4, gap=0.08)
        start, end = words[0]["start"], words[-1]["end"]
        candidate = {
            "start": start,
            "end": end,
            "text": "Elon Musk was the first prototype recipient.",
            "central_claim": {
                "start": start,
                "end": end,
                "text": "Elon Musk was the first prototype recipient.",
            },
            "variant_type": "SHORT",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words)
        )
        self.assertAlmostEqual(resolved["start"], start, delta=0.08)
        # Allow end pad after final word.
        self.assertGreaterEqual(resolved["end"], end)
        self.assertLessEqual(resolved["end"], end + END_PAD_MAX + 0.05)
        self.assertLessEqual(resolved["boundary_debug"]["extension_seconds"], 0.3)
        self.assertLessEqual(resolved["boundary_debug"]["shrink_seconds"], 0.3)

    def test_10_overlong_dead_tail_shrinks(self):
        words = _words_from_script([
            "Assembly", "takes", "less", "than", "an", "hour.", 1.2,
            "Thanks", "for", "watching", "and", "subscribe", "today.",
        ], t0=70.0, word_dur=0.4, gap=0.08)
        payoff = next(w for w in words if w["word"] == "hour.")["end"]
        tail_end = words[-1]["end"]
        candidate = {
            "start": 70.0,
            "end": tail_end,
            "text": "Assembly takes less than an hour. Thanks for watching and subscribe today.",
            "central_claim": {
                "start": 70.0,
                "end": payoff,
                "text": "Assembly takes less than an hour.",
            },
            "variant_type": "SHORT",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 6)
        )
        self.assertLess(resolved["end"], tail_end - 0.5)
        self.assertLessEqual(resolved["end"], payoff + END_PAD_MAX + 0.35)
        self.assertGreater(resolved["boundary_debug"]["shrink_seconds"], 0.5)

    def test_11_strong_payoff_requires_extension(self):
        # Build ~12s of setup after a premature end before the payoff lands.
        script = ["Here", "is", "the", "key", "claim."]
        script.append(1.0)
        for token in ["The", "payoff", "arrives", "only", "after", "more", "context",
                      "about", "why", "factories", "beat", "site", "builds", "completely."]:
            script.append(token)
        words = _words_from_script(script, t0=0.0, word_dur=0.55, gap=0.12)
        claim_end = next(w for w in words if w["word"] == "claim.")["end"]
        payoff_end = words[-1]["end"]
        self.assertGreaterEqual(payoff_end - claim_end, 10.0)
        candidate = {
            "start": 0.0,
            "end": claim_end,
            "text": "Here is the key claim.",
            "central_claim": {"start": 0.0, "end": claim_end, "text": "Here is the key claim."},
            "variant_type": "EXTENDED",
        }
        resolved = resolve_boundaries(
            candidate, words=words, segments=_segments_from_words(words, 5)
        )
        self.assertGreaterEqual(resolved["boundary_debug"]["extension_seconds"], 9.0)
        self.assertGreaterEqual(resolved["end"], payoff_end - 0.2)

    def test_12_identical_resolved_windows_dedupe(self):
        words = _words_from_script([
            "We", "ship", "homes", "every", "week", "now.",
        ], t0=10.0, word_dur=0.4, gap=0.08)
        segments = _segments_from_words(words)
        base = {
            "start": 10.0,
            "end": words[-1]["end"],
            "text": "We ship homes every week now.",
            "central_claim": {
                "start": 10.0,
                "end": words[-1]["end"],
                "text": "We ship homes every week now.",
            },
        }
        short = dict(base, variant_type="SHORT", region_type="SHORT", candidate_id="s")
        standard = dict(base, variant_type="STANDARD", region_type="STANDARD", candidate_id="t")
        context = {"words": words, "segments": segments, "duration": 120.0, "scene_changes": []}
        applied = apply_to_candidates([short, standard], context, dedupe=True)
        self.assertEqual(len(applied), 1)
        # Explicit dedupe helper also collapses already-resolved identical windows.
        resolved = [
            resolve_boundaries(short, words=words, segments=segments),
            resolve_boundaries(standard, words=words, segments=segments),
        ]
        self.assertEqual(len(dedupe_identical_resolved_windows(resolved)), 1)

    def test_natural_ending_uses_same_engine_preserve_start(self):
        words = _words_from_script([
            "Setup", "line", "stays.", 0.2,
            "Because", "the", "payoff", "lands", "here.",
        ], t0=5.0, word_dur=0.4, gap=0.08)
        start = words[0]["start"]
        early_end = next(w for w in words if w["word"] == "stays.")["end"]
        context = {
            "words": words,
            "segments": _segments_from_words(words, 4),
            "duration": 60.0,
            "scene_changes": [],
        }
        end = find_natural_ending(
            start,
            early_end,
            context,
            central_claim={"start": start, "end": early_end, "text": "Setup line stays."},
        )
        self.assertGreater(end, early_end + 0.5)
        # Shrink path: overlong end should come back.
        long_end = words[-1]["end"] + 5.0
        # Append dead air words far after completion for shrink case
        dead = _words_from_script(
            ["Thanks", "for", "watching", "everyone", "today."],
            t0=words[-1]["end"] + 1.0,
            word_dur=0.4,
            gap=0.08,
        )
        all_words = words + dead
        context["words"] = all_words
        context["segments"] = _segments_from_words(all_words, 4)
        shrunk = find_natural_ending(
            start,
            dead[-1]["end"],
            context,
            central_claim={"start": start, "end": early_end, "text": "Setup line stays."},
            variant_mode="SHORT",
        )
        self.assertLess(shrunk, dead[-1]["end"] - 0.5)


if __name__ == "__main__":
    unittest.main()

