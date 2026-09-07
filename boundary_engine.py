"""Claim-aware boundary engine: resolve clip start/end without cutting words.

Viral moment (central claim / peak) and final clip window are different concepts.
This module finds the latest understandable start and the earliest semantically
complete end, then word-aligns both timestamps.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable, Optional

# Configurable end pad after the final spoken word (spec: 100-250 ms).
END_PAD_SECONDS = 0.15
END_PAD_MIN = 0.10
END_PAD_MAX = 0.25

MAX_COMPLETION_SEARCH = 20.0
MIN_CLIP_DURATION = 15.0
PREFERRED_MAX_DURATION = 55.0
HARD_MAX_DURATION = 65.0

TERMINAL_PUNCT = (".", "!", "?", "。", "！", "？")

OPEN_THOUGHT_ENDINGS = (
    r"\b(?:and|but|because|so|if|when|which|however|although|though|"
    r"or|nor|yet|unless|until|while|whereas|plus)\s*$",
    r"(?:that means|the reason is|for example|because of|such as|"
    r"in other words|which means|so that)\s*$",
)

OPEN_CONTINUATION_STARTS = re.compile(
    r"^(?:and|but|because|so|if|when|which|that|however|although|"
    r"for example|that means|the reason|because of|also|then)\b",
    re.I,
)

GENERIC_INTRO = re.compile(
    r"^(?:hey(?:\s+guys)?|hi(?:\s+everyone)?|hello|welcome(?:\s+back)?|"
    r"good\s+(?:morning|afternoon|evening)|thanks for (?:watching|tuning)|"
    r"don't forget to|subscribe|in this (?:video|episode)|today we|"
    r"let'?s (?:get started|dive in|talk about)|um+|uh+)\b",
    re.I,
)

SETUP_CUES = re.compile(
    r"\b(?:why|how|what if|the problem|the issue|imagine|suppose|"
    r"here's the thing|the question|people (?:think|say)|most (?:people|folks))\b",
    re.I,
)

PAYOFF_CUES = re.compile(
    r"\b(?:because|that's why|therefore|the answer|so we|which means|"
    r"in other words|the result|it turns out|actually|instead|"
    r"for example|meaning|so that|weil|deshalb|darum)\b",
    re.I,
)

QUESTION_START = re.compile(
    r"^(?:why|how|what|when|where|who|which|do|does|did|is|are|can|could|"
    r"would|should|will|have|has)\b",
    re.I,
)


def clamp_end_pad(pad: Optional[float]) -> float:
    if pad is None:
        return END_PAD_SECONDS
    try:
        value = float(pad)
    except (TypeError, ValueError):
        return END_PAD_SECONDS
    return max(END_PAD_MIN, min(END_PAD_MAX, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_words(words: Optional[Iterable[dict]]) -> list[dict]:
    normalized = []
    for item in words or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("word", item.get("text", ""))).strip()
        if not text:
            continue
        start = _safe_float(item.get("start"), -1.0)
        end = _safe_float(item.get("end"), -1.0)
        if end <= start or start < 0:
            continue
        normalized.append({"word": text, "start": start, "end": end})
    normalized.sort(key=lambda w: (w["start"], w["end"]))
    return normalized


def normalize_segments(segments: Optional[Iterable[dict]]) -> list[dict]:
    normalized = []
    for item in segments or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        start = _safe_float(item.get("start"), -1.0)
        end = _safe_float(item.get("end"), -1.0)
        if not text or end <= start or start < 0:
            continue
        normalized.append({"text": text, "start": start, "end": end})
    normalized.sort(key=lambda s: (s["start"], s["end"]))
    return normalized


def align_start_to_words(time_point: float, words: list[dict]) -> float:
    """Never begin inside a word; prefer shortly before the first selected word."""
    if not words:
        return max(0.0, round(time_point, 3))
    # If inside a word, snap to that word's start.
    for word in words:
        if word["start"] - 1e-3 <= time_point < word["end"] - 1e-3:
            return round(word["start"], 3)
    # Prefer the first word at/after the time, else the last word before it.
    later = [w for w in words if w["start"] >= time_point - 1e-3]
    if later:
        return round(later[0]["start"], 3)
    earlier = [w for w in words if w["end"] <= time_point + 1e-3]
    if earlier:
        return round(earlier[-1]["start"], 3)
    return max(0.0, round(time_point, 3))


def align_end_to_words(
    time_point: float,
    words: list[dict],
    *,
    pad: Optional[float] = None,
    source_duration: Optional[float] = None,
) -> float:
    """Never end before the final selected word completes; add safety pad."""
    pad = clamp_end_pad(pad)
    if not words:
        end = time_point + pad
        if source_duration and source_duration > 0:
            end = min(end, source_duration)
        return round(max(0.0, end), 3)

    # Word that contains the cut, or the last word fully before it.
    containing = None
    last_before = None
    for word in words:
        if word["start"] - 1e-3 <= time_point <= word["end"] + 1e-3:
            containing = word
            break
        if word["end"] <= time_point + 1e-3:
            last_before = word
    chosen = containing or last_before
    if chosen is None:
        # Cut is before all words — use first word end.
        chosen = words[0]
    end = chosen["end"] + pad
    # If the proposed cut is after the chosen word, keep any extra silence
    # only when it does not land inside a later word.
    if time_point > chosen["end"] and containing is None:
        later = next((w for w in words if w["start"] > chosen["end"] + 1e-3), None)
        if later is None or time_point <= later["start"]:
            end = max(end, time_point)
            # still never enter the next word
            if later is not None and end > later["start"]:
                end = later["start"]
        else:
            # time_point is inside/after a later word — align to that word
            for word in words:
                if word["start"] - 1e-3 <= time_point <= word["end"] + 1e-3:
                    end = word["end"] + pad
                    break
    if source_duration and source_duration > 0:
        end = min(end, float(source_duration))
    return round(max(0.0, end), 3)


def _ends_terminal(text: str) -> bool:
    return bool(text) and text.rstrip().endswith(TERMINAL_PUNCT)


def _is_question(text: str) -> bool:
    stripped = (text or "").strip()
    if stripped.endswith("?"):
        return True
    return bool(QUESTION_START.match(stripped)) and not _ends_terminal(stripped.replace("?", ""))


def is_open_thought(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.endswith(","):
        return True
    if stripped.endswith(("...", "…")):
        return True
    lower = stripped.lower()
    for pattern in OPEN_THOUGHT_ENDINGS:
        if re.search(pattern, lower, re.I):
            return True
    # Incomplete sentence: no terminal punctuation and hangs on a conjunction-like end.
    if not _ends_terminal(stripped):
        tokens = re.findall(r"[\w']+", lower)
        if tokens and tokens[-1] in {
            "and", "but", "because", "so", "if", "when", "which", "that",
            "or", "nor", "yet", "to", "the", "a", "an", "of", "for", "with",
        }:
            return True
        # Short hanging clause without punctuation often incomplete.
        if len(tokens) < 4:
            return True
    return False


def is_incomplete_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if _ends_terminal(stripped):
        return False
    return is_open_thought(stripped) or not stripped[-1].isalnum()


def _looks_attached(previous_text: str, next_text: str) -> bool:
    if not next_text:
        return False
    if OPEN_CONTINUATION_STARTS.match(next_text.strip()):
        return True
    if previous_text and not _ends_terminal(previous_text) and not _is_question(previous_text):
        return True
    if previous_text.rstrip().endswith(","):
        return True
    return False


def build_sentences(
    segments: list[dict],
    words: Optional[list[dict]] = None,
) -> list[dict]:
    """Group transcript items into sentence-like units with timings."""
    words = words or []
    items: list[dict]
    if words:
        items = [{"text": w["word"], "start": w["start"], "end": w["end"], "kind": "word"} for w in words]
    else:
        items = [{"text": s["text"], "start": s["start"], "end": s["end"], "kind": "seg"} for s in segments]

    sentences: list[dict] = []
    pending: list[dict] = []

    def flush():
        if not pending:
            return
        if pending[0]["kind"] == "word":
            # Reconstruct spacing from Whisper tokens when present.
            parts = []
            for item in pending:
                token = item["text"]
                if parts and not token[:1].isspace() and not parts[-1].endswith((" ", "\n")):
                    parts.append(" ")
                parts.append(token)
            text = "".join(parts).strip()
        else:
            text = " ".join(item["text"].strip() for item in pending).strip()
        sentences.append({
            "start": pending[0]["start"],
            "end": pending[-1]["end"],
            "text": text,
            "complete": _ends_terminal(text),
            "word_aligned": pending[0]["kind"] == "word",
        })
        pending.clear()

    for item in items:
        pending.append(item)
        text = item["text"].rstrip()
        if text.endswith(TERMINAL_PUNCT):
            flush()
        elif item["kind"] == "seg" and _ends_terminal(item["text"]):
            flush()
    flush()
    return sentences


def _sentence_index_covering(sentences: list[dict], time_point: float) -> int:
    if not sentences:
        return -1
    for index, sentence in enumerate(sentences):
        if sentence["start"] - 0.05 <= time_point <= sentence["end"] + 0.05:
            return index
    # nearest by start
    return min(range(len(sentences)), key=lambda i: abs(sentences[i]["start"] - time_point))


def _extract_claim(candidate: dict, sentences: list[dict]) -> dict:
    claim = candidate.get("central_claim")
    if isinstance(claim, dict) and claim.get("text"):
        start = _safe_float(claim.get("start"), _safe_float(candidate.get("start")))
        end = _safe_float(claim.get("end"), start)
        return {
            "start": start,
            "end": max(end, start),
            "text": str(claim.get("text") or claim.get("label") or "").strip(),
        }
    if isinstance(claim, dict) and (claim.get("start") is not None):
        start = _safe_float(claim.get("start"))
        end = _safe_float(claim.get("end"), start)
        index = _sentence_index_covering(sentences, start)
        text = sentences[index]["text"] if index >= 0 else str(candidate.get("text", ""))
        return {"start": start, "end": max(end, start), "text": text}
    # Fallback: use candidate window / middle sentence as claim proxy.
    start = _safe_float(candidate.get("start"))
    end = _safe_float(candidate.get("end"), start)
    index = _sentence_index_covering(sentences, (start + end) / 2.0 if end > start else start)
    if index < 0 and sentences:
        index = 0
    if index >= 0:
        sentence = sentences[index]
        return {"start": sentence["start"], "end": sentence["end"], "text": sentence["text"]}
    return {"start": start, "end": end, "text": str(candidate.get("text", "")).strip()}


def _variant_mode(candidate: dict, override: Optional[str] = None) -> str:
    if override:
        return str(override).upper()
    for key in ("variant_mode", "variant_type", "region_type"):
        value = candidate.get(key)
        if value:
            return str(value).upper()
    return "STANDARD"


def resolve_claim_aware_start(
    claim: dict,
    sentences: list[dict],
    *,
    variant: str = "STANDARD",
    original_start: Optional[float] = None,
) -> tuple[float, str]:
    """Latest natural start that still keeps the claim understandable."""
    if not sentences:
        start = claim["start"] if original_start is None else min(original_start, claim["start"])
        return round(max(0.0, start), 3), "claim_fallback_no_sentences"

    claim_index = _sentence_index_covering(sentences, claim["start"])
    if claim_index < 0:
        return round(max(0.0, claim["start"]), 3), "claim_time_fallback"

    claim_sentence = sentences[claim_index]
    start_index = claim_index
    reason = "claim_sentence_start"

    # Prefer question before answer.
    if claim_index > 0:
        previous = sentences[claim_index - 1]
        gap = claim_sentence["start"] - previous["end"]
        if gap <= 2.5 and _is_question(previous["text"]) and not _is_question(claim_sentence["text"]):
            start_index = claim_index - 1
            reason = "question_before_answer"

    # Minimal setup when claim is confusing alone (continuation / weak deictic start).
    needs_setup = bool(
        re.match(r"^(?:and|but|because|so|which|that|this|it|they|he|she|we)\b", claim_sentence["text"], re.I)
    ) or (variant in {"STANDARD", "EXTENDED"} and SETUP_CUES.search(claim_sentence["text"] or "") is None
          and len(claim_sentence["text"].split()) < 8)

    if start_index == claim_index and claim_index > 0 and needs_setup:
        previous = sentences[claim_index - 1]
        gap = claim_sentence["start"] - previous["end"]
        if gap <= 2.5 and not GENERIC_INTRO.match(previous["text"]) and not _is_question(claim_sentence["text"]):
            # SHORT: only pull setup if claim clearly continues previous thought.
            if variant == "SHORT" and not _looks_attached(previous["text"], claim_sentence["text"]):
                pass
            else:
                start_index = claim_index - 1
                reason = "minimal_setup_sentence"

    # EXTENDED may keep one additional story-turn sentence when still local.
    if variant == "EXTENDED" and start_index > 0:
        earlier = sentences[start_index - 1]
        gap = sentences[start_index]["start"] - earlier["end"]
        span = claim_sentence["start"] - earlier["start"]
        if gap <= 2.0 and span <= 14.0 and not GENERIC_INTRO.match(earlier["text"]):
            if SETUP_CUES.search(earlier["text"]) or _is_question(earlier["text"]):
                start_index = start_index - 1
                reason = "extended_story_turn"

    # Never include generic intros even if they sit immediately before.
    while start_index < claim_index and GENERIC_INTRO.match(sentences[start_index]["text"]):
        start_index += 1
        reason = "skipped_generic_intro"

    start = sentences[start_index]["start"]
    # Prefer latest understandable start — do not jump far earlier than the claim.
    if claim["start"] - start > 14.0 and start_index < claim_index:
        start_index = claim_index
        start = claim_sentence["start"]
        reason = "clamped_latest_natural_start"

    if original_start is not None and abs(original_start - start) < 0.05:
        reason = "unchanged_start"
    return round(max(0.0, start), 3), reason


def _text_between(sentences: list[dict], first: int, last: int) -> str:
    return " ".join(sentences[i]["text"] for i in range(first, last + 1)).strip()


def _has_payoff(text: str, claim_text: str) -> bool:
    if PAYOFF_CUES.search(text or ""):
        return True
    claim_tokens = set(re.findall(r"[a-z0-9']+", (claim_text or "").lower()))
    end_tokens = set(re.findall(r"[a-z0-9']+", (text or "").lower()))
    if claim_tokens and end_tokens:
        overlap = len(claim_tokens & end_tokens) / max(1, len(end_tokens))
        if overlap > 0.12 and _ends_terminal(text):
            return True
    if re.search(r"\d", text or "") and _ends_terminal(text or ""):
        return True
    return _ends_terminal(text or "")


def _dead_tail(sentences: list[dict], index: int) -> bool:
    text = sentences[index]["text"]
    if GENERIC_INTRO.match(text):
        return True
    lower = text.lower()
    if re.search(r"\b(?:subscribe|like and subscribe|thanks for watching|see you|bye)\b", lower):
        return True
    return False


def resolve_claim_aware_end(
    claim: dict,
    start: float,
    sentences: list[dict],
    *,
    words: Optional[list[dict]] = None,
    scene_changes: Optional[list[float]] = None,
    original_end: Optional[float] = None,
    variant: str = "STANDARD",
    source_duration: Optional[float] = None,
    search_horizon: float = MAX_COMPLETION_SEARCH,
) -> tuple[float, dict]:
    """Earliest semantically complete thought/payoff after the claim."""
    scene_changes = scene_changes or []
    words = words or []
    meta = {
        "sentence_completion": False,
        "payoff_detected": False,
        "question_answer_completion": False,
        "end_reason": "fallback_original_or_claim",
    }

    if not sentences:
        end = original_end if original_end is not None else claim["end"]
        return round(end, 3), meta

    claim_index = _sentence_index_covering(sentences, claim["end"] - 1e-3)
    if claim_index < 0:
        claim_index = _sentence_index_covering(sentences, claim["start"])
    if claim_index < 0:
        end = original_end if original_end is not None else claim["end"]
        return round(end, 3), meta

    # Search from the claim through +search_horizon (may shrink below original_end).
    latest_time = claim["end"] + float(search_horizon)
    if source_duration and source_duration > 0:
        latest_time = min(latest_time, float(source_duration))

    # Duration caps relative to resolved start.
    preferred_limit = start + PREFERRED_MAX_DURATION
    hard_limit = start + HARD_MAX_DURATION
    min_end = start + MIN_CLIP_DURATION

    # Question -> answer tracking.
    awaiting_answer = _is_question(sentences[claim_index]["text"])
    if claim_index > 0 and _is_question(sentences[claim_index - 1]["text"]):
        awaiting_answer = False  # claim itself is the answer body
        meta["question_answer_completion"] = False

    best_end = None
    best_meta = None
    last_index = claim_index

    for index in range(claim_index, len(sentences)):
        sentence = sentences[index]
        if sentence["start"] > latest_time + 1e-6:
            break
        if sentence["end"] - start > HARD_MAX_DURATION + 1e-6:
            break

        # Skip attaching across huge gaps unless still awaiting an answer within horizon.
        if index > claim_index:
            gap = sentence["start"] - sentences[index - 1]["end"]
            if gap > 2.5 and not awaiting_answer:
                break
            if gap > 4.0:
                break

        last_index = index
        text = sentences[index]["text"]
        window_text = _text_between(sentences, claim_index, index)

        if awaiting_answer and not _is_question(text) and _ends_terminal(text) and len(text.split()) >= 4:
            awaiting_answer = False
            meta["question_answer_completion"] = True

        open_thought = is_open_thought(text) or is_incomplete_sentence(text)
        next_sentence = sentences[index + 1] if index + 1 < len(sentences) else None
        gap_next = (
            (next_sentence["start"] - sentence["end"]) if next_sentence else 999.0
        )
        next_is_dead = bool(next_sentence) and (
            _dead_tail(sentences, index + 1)
            or bool(re.search(
                r"\b(?:thanks for watching|subscribe|like and subscribe|see you|bye)\b",
                next_sentence["text"],
                re.I,
            ))
        )
        elaborates = bool(next_sentence) and (
            _looks_attached(text, next_sentence["text"])
            or bool(OPEN_CONTINUATION_STARTS.match(next_sentence["text"]))
            or bool(PAYOFF_CUES.search(next_sentence["text"]))
            or bool(re.search(r"\d", next_sentence["text"]))
            or (
                index == claim_index
                and gap_next <= 2.0
                and len(next_sentence["text"].split()) >= 4
                and not GENERIC_INTRO.match(next_sentence["text"])
                and (
                    bool(SETUP_CUES.search(next_sentence["text"]))
                    or bool(PAYOFF_CUES.search(next_sentence["text"]))
                    or "payoff" in next_sentence["text"].lower()
                    or any(
                        token in next_sentence["text"].lower()
                        for token in re.findall(r"[a-z0-9']{4,}", (claim.get("text") or "").lower())
                    )
                )
            )
        )
        attached_next = bool(
            next_sentence
            and next_sentence["start"] <= latest_time
            and gap_next <= 2.5
            and elaborates
            and not next_is_dead
        )

        # Punctuation / pause / scene are candidates only.
        punct_candidate = _ends_terminal(text)
        pause_candidate = False
        if next_sentence:
            pause_candidate = gap_next >= 0.65 and not attached_next
        else:
            pause_candidate = True
        scene_candidate = any(
            sentence["end"] - 0.35 <= scene <= (next_sentence["start"] if next_sentence else sentence["end"] + 1.0)
            for scene in scene_changes
        )

        payoff = _has_payoff(text, claim["text"]) or _has_payoff(window_text, claim["text"])
        # Numerical explanation after a claim: keep going while digits explain the claim.
        if re.search(r"\d", text) and index > claim_index:
            payoff = True
        if index > claim_index and PAYOFF_CUES.search(text):
            payoff = True

        # Do not treat the bare claim sentence as done while discourse continues.
        complete = (
            punct_candidate
            and not open_thought
            and not attached_next
            and not awaiting_answer
        )

        # Prefer earliest complete thought; allow pause/scene only when thought is closed.
        if complete or (not open_thought and not awaiting_answer and not attached_next and (pause_candidate or scene_candidate) and punct_candidate):
            end_time = sentence["end"]
            # Strong payoff can exceed preferred duration up to hard max.
            if end_time > preferred_limit and not (payoff and end_time <= hard_limit):
                if best_end is not None:
                    break
                if not payoff:
                    continue
            reason = "semantic_completion"
            if meta.get("question_answer_completion"):
                reason = "question_answer_completion"
            elif payoff:
                reason = "payoff_completion"
            elif pause_candidate and punct_candidate:
                reason = "completed_before_pause"
            elif scene_candidate:
                reason = "completed_near_scene"
            candidate_meta = {
                "sentence_completion": True,
                "payoff_detected": bool(payoff),
                "question_answer_completion": bool(meta.get("question_answer_completion")),
                "end_reason": reason,
                "end_sentence_index": index,
            }
            best_end = end_time
            best_meta = candidate_meta
            # SHORT may stop at first closed thought once we are past pure claim setup,
            # but still prefer a payoff if this end has none and one arrives soon.
            if variant == "SHORT":
                if payoff or index > claim_index or not next_sentence or gap_next > 2.0:
                    break
                # else keep searching for nearby payoff
                best_end = None
                best_meta = None
                continue
            break

    if best_end is None:
        # Fallback: last sentence still within horizon / duration, avoiding open thought if possible.
        fallback_index = last_index
        while fallback_index > claim_index and is_open_thought(sentences[fallback_index]["text"]):
            # Prefer extending if next exists within horizon rather than ending open.
            if fallback_index + 1 < len(sentences) and sentences[fallback_index + 1]["end"] <= latest_time:
                fallback_index += 1
                if not is_open_thought(sentences[fallback_index]["text"]):
                    break
            else:
                break
        best_end = sentences[fallback_index]["end"]
        best_meta = {
            "sentence_completion": _ends_terminal(sentences[fallback_index]["text"]),
            "payoff_detected": _has_payoff(sentences[fallback_index]["text"], claim["text"]),
            "question_answer_completion": bool(meta.get("question_answer_completion")),
            "end_reason": "horizon_fallback",
            "end_sentence_index": fallback_index,
        }

    end = best_end
    meta.update(best_meta or {})

    # Shrink dead tails past the earliest complete thought (already earliest),
    # and shrink if original_end extends into trailing filler after completion.
    end_index = meta.get("end_sentence_index", claim_index)
    if isinstance(end_index, int) and end_index + 1 < len(sentences) and original_end is not None:
        # If original end reaches into dead-tail sentences after completion, stay shrunk.
        pass
    # Explicit shrink of dead-tail sentences included by a late original_end:
    if original_end is not None and end < original_end:
        meta["end_reason"] = meta.get("end_reason", "semantic_completion")
        if "shrink" not in meta["end_reason"]:
            meta["end_reason"] = meta["end_reason"] + "_shrink_dead_tail"

    # Ensure claim is covered.
    end = max(end, claim["end"])

    # Soft minimum duration: extend only to a complete sentence when short.
    if end < min_end:
        for index in range(claim_index, len(sentences)):
            if sentences[index]["end"] < min_end:
                if is_open_thought(sentences[index]["text"]):
                    continue
                continue
            if sentences[index]["end"] - start > HARD_MAX_DURATION:
                break
            if sentences[index]["end"] > latest_time and sentences[index]["end"] > (original_end or 0):
                # Allow min-duration fill only within hard constraints.
                if sentences[index]["end"] - start > PREFERRED_MAX_DURATION:
                    break
            if not is_open_thought(sentences[index]["text"]) and _ends_terminal(sentences[index]["text"]):
                end = sentences[index]["end"]
                meta["end_reason"] = "extended_to_min_duration_complete"
                break
            end = sentences[index]["end"]

    if source_duration and source_duration > 0:
        end = min(end, float(source_duration))
    return round(end, 3), meta


def window_text(words: list[dict], segments: list[dict], start: float, end: float) -> str:
    chosen_words = [w for w in words if w["end"] > start + 1e-3 and w["start"] < end - 1e-3]
    if chosen_words:
        parts = []
        for word in chosen_words:
            token = word["word"]
            if parts and not token[:1].isspace() and not str(parts[-1]).endswith((" ", "\n")):
                parts.append(" ")
            parts.append(token)
        return "".join(parts).strip()
    chosen = [s for s in segments if s["end"] > start + 1e-3 and s["start"] < end - 1e-3]
    return " ".join(s["text"] for s in chosen).strip()


def resolve_boundaries(
    candidate: dict,
    *,
    words: Optional[Iterable[dict]] = None,
    segments: Optional[Iterable[dict]] = None,
    scene_changes: Optional[Iterable[float]] = None,
    source_duration: Optional[float] = None,
    preserve_start: bool = False,
    end_only: bool = False,
    variant_mode: Optional[str] = None,
    end_pad: Optional[float] = None,
    search_horizon: float = MAX_COMPLETION_SEARCH,
) -> dict:
    """Return a shallow-copied candidate with claim-aware, word-aligned bounds."""
    adjusted = copy.deepcopy(candidate)
    word_list = normalize_words(words if words is not None else candidate.get("words"))
    segment_list = normalize_segments(
        segments if segments is not None else candidate.get("segments")
    )
    scenes = [float(s) for s in (scene_changes or []) if _safe_float(s, -1) >= 0]
    duration = source_duration
    if duration is None:
        duration = _safe_float(candidate.get("source_duration"), 0.0) or None

    original_start = _safe_float(adjusted.get("start"))
    original_end = _safe_float(adjusted.get("end"), original_start)
    sentences = build_sentences(segment_list, word_list)
    claim = _extract_claim(adjusted, sentences)
    variant = _variant_mode(adjusted, variant_mode)

    if preserve_start or end_only:
        resolved_start = original_start
        start_reason = "preserved_start"
    else:
        resolved_start, start_reason = resolve_claim_aware_start(
            claim, sentences, variant=variant, original_start=original_start
        )

    resolved_end, end_meta = resolve_claim_aware_end(
        claim,
        resolved_start,
        sentences,
        words=word_list,
        scene_changes=scenes,
        original_end=original_end,
        variant=variant,
        source_duration=duration,
        search_horizon=search_horizon,
    )

    # Word alignment — never cut a word.
    aligned_start = align_start_to_words(resolved_start, word_list)
    aligned_end = align_end_to_words(
        resolved_end,
        word_list,
        pad=end_pad,
        source_duration=duration,
    )
    if aligned_end <= aligned_start:
        aligned_end = align_end_to_words(
            max(resolved_end, aligned_start + 0.01),
            word_list,
            pad=end_pad,
            source_duration=duration,
        )

    extension = max(0.0, aligned_end - original_end)
    shrink = max(0.0, original_end - aligned_end)

    text = window_text(word_list, segment_list, aligned_start, aligned_end) or str(
        adjusted.get("text", "")
    ).strip()

    adjusted.update({
        "start": round(aligned_start, 3),
        "end": round(aligned_end, 3),
        "duration": round(aligned_end - aligned_start, 3),
        "text": text,
        "boundary_debug": {
            "central_claim": claim,
            "original_start": round(original_start, 3),
            "original_end": round(original_end, 3),
            "resolved_start": round(aligned_start, 3),
            "resolved_end": round(aligned_end, 3),
            "start_reason": start_reason,
            "end_reason": end_meta.get("end_reason"),
            "sentence_completion": bool(end_meta.get("sentence_completion")),
            "payoff_detected": bool(end_meta.get("payoff_detected")),
            "question_answer_completion": bool(end_meta.get("question_answer_completion")),
            "word_aligned": bool(word_list),
            "extension_seconds": round(extension, 3),
            "shrink_seconds": round(shrink, 3),
            "variant_mode": variant,
        },
    })
    return adjusted


def find_natural_ending(
    start: float,
    original_end: float,
    boundary_context: dict,
    *,
    central_claim: Optional[dict] = None,
    variant_mode: str = "STANDARD",
    end_pad: Optional[float] = None,
) -> float:
    """Recompute only the end using the claim-aware engine (expand or shrink)."""
    candidate = {
        "start": start,
        "end": original_end,
        "text": "",
        "central_claim": central_claim,
        "variant_type": variant_mode,
    }
    resolved = resolve_boundaries(
        candidate,
        words=boundary_context.get("words"),
        segments=boundary_context.get("segments"),
        scene_changes=boundary_context.get("scene_changes"),
        source_duration=boundary_context.get("duration"),
        preserve_start=True,
        end_only=True,
        variant_mode=variant_mode,
        end_pad=end_pad,
    )
    return float(resolved["end"])


def resolved_window_key(candidate: dict, tol: float = 0.05) -> tuple:
    start = round(_safe_float(candidate.get("start")) / tol) * tol
    end = round(_safe_float(candidate.get("end")) / tol) * tol
    return (round(start, 2), round(end, 2))


def dedupe_identical_resolved_windows(candidates: list[dict]) -> list[dict]:
    """Drop variants that resolved to the same semantic window."""
    kept: list[dict] = []
    seen: set[tuple] = set()
    for candidate in candidates:
        key = resolved_window_key(candidate)
        # Prefer keeping richer variant labels only when windows differ.
        if key in seen:
            # Mark debug if present
            debug = candidate.get("boundary_debug")
            if isinstance(debug, dict):
                debug = dict(debug)
                debug["deduped_identical_window"] = True
                candidate = dict(candidate)
                candidate["boundary_debug"] = debug
            continue
        seen.add(key)
        kept.append(candidate)
    return kept


def apply_to_candidates(
    candidates: list[dict],
    boundary_context: dict,
    *,
    end_pad: Optional[float] = None,
    dedupe: bool = True,
) -> list[dict]:
    """Resolve claim-aware boundaries for each candidate; optionally dedupe windows."""
    resolved = [
        resolve_boundaries(
            candidate,
            words=boundary_context.get("words"),
            segments=boundary_context.get("segments"),
            scene_changes=boundary_context.get("scene_changes"),
            source_duration=boundary_context.get("duration"),
            end_pad=end_pad,
        )
        for candidate in candidates
    ]
    if dedupe:
        return dedupe_identical_resolved_windows(resolved)
    return resolved
