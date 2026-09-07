"""Deterministic, transcript-only discovery. No model, media decode or network.

This is shallow discourse analysis, not an LLM substitute. Predicate/subject
structure, quantity context and relations between adjacent sentences seed
complete thoughts. Time windows are built around those seeds, not a time grid.
Ground-truth annotations are only consumed by the offline evaluation helper.
"""

import copy
import hashlib
import json
import re
import time
from pathlib import Path


SEMANTIC_VERSION = "transcript-semantic-v1.2"
PRE_GPT_SELECTION_VERSION = "quality-first-v1"
WINDOWS = {"SHORT": (15, 25, 21), "STANDARD": (25, 40, 32),
           "EXTENDED": (35, 55, 45)}
STOP = set("a an the this that these those it its is are was were be been being "
           "to of for from in on at as with and or but so then than if because "
           "i you he she we they my your our their me him her them have has had "
           "do does did will would could can should just really very about "
           "like yeah okay right actually know think well not no".split())
NUM_WORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
            r"thousand|million|billion|dozen|half|quarter)")
NUMBER = (r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|"
          + NUM_WORD + r"(?:[ -]+(?:and[ -]+)?" + NUM_WORD + r")*)")
QUANTITY = re.compile(
    r"(?<![\w.])(?:a\s+)?" + NUMBER
    + r"(?:\s*(?:to|–|-)\s*" + NUMBER + r")?"
    + r"(?:\s+(?:and\s+)?a\s+half)?"
    + r"[\s-]*(?P<unit>square\s+(?:feet|foot|meters?|metres?)|"
    r"seconds?|minutes?|hours?|days?|weeks?|months?|years?|stations?|"
    r"names?|people|customers?|users?|homes?|houses?|units?|"
    r"feet|foot|inches|meters?|metres?|miles?|kilometers?|kilometres?|"
    r"dollars?|euros?|bucks|percent|%|stories|story|bedrooms?|beds?|baths?)(?=\W|$)",
    re.I,
)
MONEY = re.compile(r"[$€£]\s*" + NUMBER, re.I)
PREDICATE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|can|could|will|would|should|"
    r"make|makes|made|build|builds|built|cost|costs|save|saves|saved|"
    r"need|needs|want|wants|love|loves|hate|hates|believe|feel|feels|"
    r"learned|realized|discovered|found|changed|became|started|took|takes|"
    r"means|shows|works|lets|allows|get|gets|got|try|plan|aim|think|told|"
    r"keep|keeps|store|stores|release|releases|ship|ships|shipped|"
    r"transport|transports|require|requires|produce|produces|consume|consumes)\b|"
    r"\b(?:it|that|there|here|he|she)['\u2019]s\b|"
    r"\b(?:we|they|you)['\u2019](?:re|ve|ll)\b|\bI['\u2019](?:m|ve|ll)\b", re.I)
RELATIONS = {
    "contrast": r"\b(?:but|however|instead|whereas|yet|unlike|rather than|compared to|until)\b",
    "reveal": r"\b(?:turns out|wait for it|first (?:recipient|time|ever)|"
              r"prototype|no one|nobody|surprisingly|unexpected|secret)\b",
    "explanation": r"\b(?:because|that's why|that is why|comes down to|depends on|"
                   r"which means|this means|so that|the reason|how we|how you|by using)\b",
    "problem": r"\b(?:problem|can't afford|cannot afford|too expensive|"
               r"biggest cost|biggest costs|struggle|failed|difficulty|challenge)\b",
    "solution": r"\b(?:solve|solution|made it easier|save|saves|reduce|"
                r"efficient|efficiently|instead|allows|able to|can)\b",
    "story": r"\b(?:then|when|until|suddenly|realized|discovered|told me|"
             r"last week|years ago|first time|grew up)\b",
    "emotion": r"\b(?:afraid|scared|proud|love|hate|regret|hurt|"
               r"can't afford|lost|bullied|my life|our mission)\b",
    "opinion": r"\b(?:I (?:think|believe|hate|love)|we (?:believe|hate)|"
               r"should|never|always|impossible|best|worst)\b",
}


def _tokens(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP]


def transcript_sentences(transcript):
    """Join Whisper segments across sentence breaks; use exact word boundaries.

    Segment-only transcripts use segment bounds (never invented word times).
    Long unpunctuated runs are bounded at input segment ends and marked incomplete.
    """
    segments = [s for s in transcript.get("segments", [])
                if str(s.get("text", "")).strip() and float(s.get("end", 0)) > float(s.get("start", 0))]
    words = [w for w in transcript.get("words", [])
             if str(w.get("word", "")).strip() and float(w.get("end", 0)) >= float(w.get("start", 0))]
    # Do not silently ignore a partial word stream.
    use_words = bool(words) and (not segments or (
        float(words[0]["start"]) <= float(segments[0]["start"]) + 1
        and float(words[-1]["end"]) >= float(segments[-1]["end"]) - 1))
    items = words if use_words else segments
    preserve_spacing = use_words and any(str(w["word"])[:1].isspace() for w in words)
    sentences, pending = [], []
    for item in items:
        original_text = str(item.get("word" if use_words else "text", ""))
        text = original_text if preserve_spacing else original_text.strip()
        if not text.strip():
            continue
        if pending and float(item["start"]) - float(pending[-1]["end"]) > 2.5:
            _finish_sentence(sentences, pending, use_words, preserve_spacing)
            pending = []
        pending.append({"text": text, "start": float(item["start"]), "end": float(item["end"])})
        terminal = bool(re.search(r'[.!?][\"\u201d\u2019]*$', text))
        segment_end = not use_words or any(abs(float(s["end"]) - float(item["end"])) < .02 for s in segments)
        if terminal or (segment_end and pending[-1]["end"] - pending[0]["start"] >= 22):
            _finish_sentence(sentences, pending, use_words, preserve_spacing)
            pending = []
    if pending:
        _finish_sentence(sentences, pending, use_words, preserve_spacing)
    return sentences


def _finish_sentence(sentences, items, word_aligned, preserve_spacing=False):
    text = ("" if preserve_spacing else " ").join(i["text"] for i in items).strip()
    sentences.append({"start": items[0]["start"], "end": items[-1]["end"],
                      "text": text, "complete": bool(re.search(r'[.!?][\"\u201d\u2019]*$', text)),
                      "word_aligned": word_aligned})


def numeric_claims(text):
    """Quantities need units and proposition/context; bare years aren't claims."""
    results = []
    for match in list(QUANTITY.finditer(text)) + list(MONEY.finditer(text)):
        context = text[max(0, match.start() - 90): min(len(text), match.end() + 110)]
        before = text[max(0, match.start() - 22):match.start()].lower()
        if re.search(r"(?:episode|chapter|part|season|version|year)\s*$", before):
            continue
        lower = context.lower()
        unit = match.groupdict().get("unit") or "currency"
        rate = bool(re.search(r"\b(?:every|per|each|an? hour|less than|only|"
                              r"from|to|under|over|about|around)\b", lower))
        outcome = bool(re.search(r"\b(?:make|build|built|home|homes|house|houses|"
                                 r"cost|ship|shipped|shipping|production|stations|"
                                 r"wait\s?list|names|space|save|earn|lost|grew|"
                                 r"customers|users|because|depends|size|wide|tall|"
                                 r"took|takes|setup|set up|instead|compared)\b", lower))
        meaningful = bool(PREDICATE.search(context)) and (rate or outcome)
        results.append({"quantity": match.group().strip(), "unit": unit,
                        "context": context.strip(), "meaningful": meaningful,
                        "rate_or_comparison": rate, "outcome_context": outcome})
    return results


def sentence_signals(text):
    words = re.findall(r"\b[\w']+\b", text)
    content = _tokens(text)
    relation = {k: bool(re.search(pattern, text, re.I)) for k, pattern in RELATIONS.items()}
    quantities = [q for q in numeric_claims(text) if q["meaningful"]]
    predicate = bool(PREDICATE.search(text))
    # A factual subject + predicate can seed a moment without trigger vocabulary.
    subject = bool(re.search(r"\b(?:I|we|you|he|she|they|it|this|that|"
                             r"[A-Z][a-z]{2,})\b", text)) or len(content) >= 4
    complete = bool(re.search(r'[.!?][\"\u201d\u2019]*$', text))
    proposition = predicate and subject and len(words) >= 7
    density = min(1.0, len(set(content)) / max(1, len(words)) / .65)
    score = (.18 * proposition + .10 * complete + .12 * density
             + .24 * min(1, len(quantities)) + .12 * relation["contrast"]
             + .14 * relation["reveal"] + .14 * relation["explanation"]
             + .10 * relation["opinion"] + .10 * relation["emotion"]
             + .14 * (relation["problem"] and relation["solution"])
             + .10 * (relation["story"] and proposition))
    return {"score": min(1.0, score), "proposition": proposition,
            "information_density": round(density, 4), "complete": complete,
            "numeric_claims": quantities, **relation}


def generate_semantic_candidates(transcript):
    """Return raw windows and locally deduplicated windows without any I/O."""
    sentences = transcript_sentences(transcript)
    raw = []
    for index, sentence in enumerate(sentences):
        claim_end = index
        is_question = sentence["text"].rstrip().endswith("?")
        if is_question:
            # Keep a question with a substantive answer, never an unanswered hook.
            while claim_end + 1 < len(sentences) and sentences[claim_end]["end"] - sentence["start"] < 20:
                if sentences[claim_end + 1]["start"] - sentences[claim_end]["end"] > 2.5:
                    break
                claim_end += 1
                if len(sentences[claim_end]["text"].split()) >= 6 and not sentences[claim_end]["text"].endswith("?"):
                    break
            if claim_end == index or sentences[claim_end]["text"].endswith("?"):
                continue
        claim_text = " ".join(s["text"] for s in sentences[index:claim_end + 1])
        sig = sentence_signals(claim_text)
        if is_question and sig["proposition"]:
            sig["score"] = min(1.0, sig["score"] + .18)
        if not sig["proposition"] or sig["score"] < .34:
            continue
        # Candidate starts are a short sentence context around this claim only.
        starts = [index]
        for earlier in range(index - 1, max(-1, index - 4), -1):
            if sentence["start"] - sentences[earlier]["start"] > 14:
                break
            if sentences[earlier + 1]["start"] - sentences[earlier]["end"] > 2.5:
                break
            starts.append(earlier)
        for variant, (minimum, maximum, target) in WINDOWS.items():
            options = []
            for first in starts:
                for last in range(claim_end, len(sentences)):
                    duration = sentences[last]["end"] - sentences[first]["start"]
                    if duration > maximum:
                        break
                    if last > claim_end and sentences[last]["start"] - sentences[last - 1]["end"] > 2.5:
                        break
                    if duration < minimum or not sentences[last]["complete"] or sentences[last]["text"].rstrip().endswith("?"):
                        continue
                    chosen = sentences[first:last + 1]
                    text = " ".join(s["text"] for s in chosen)
                    weak_start = bool(re.match(r"^(?:and|but|because|so|which|that|it|they|he|she)\b", chosen[0]["text"], re.I))
                    end_tokens = set(_tokens(chosen[-1]["text"]))
                    claim_tokens = set(_tokens(claim_text))
                    related = len(end_tokens & claim_tokens) / max(1, len(end_tokens))
                    payoff = last == claim_end or related > .12 or bool(re.search(RELATIONS["explanation"], chosen[-1]["text"], re.I))
                    context_cost = max(0, sentence["start"] - chosen[0]["start"]) / 30
                    boundary_score = (1.0 * payoff + .6 * (not weak_start) + .3 * related
                                      - abs(duration - target) / 35 - context_cost
                                      - .10 * (last - claim_end))
                    options.append((boundary_score, first, last, text, payoff, weak_start))
            if not options:
                continue
            _, first, last, text, payoff, weak_start = max(options, key=lambda x: (x[0], -x[1], -x[2]))
            start, end = sentences[first]["start"], sentences[last]["end"]
            all_sig = sentence_signals(text)
            numeric = all_sig["numeric_claims"]
            detected = [name for name in RELATIONS if all_sig[name]]
            detected += ["complete_proposition", "sentence_complete"]
            if numeric:
                detected.append("contextual_quantity")
            if is_question:
                detected.append("question_answer")
            if payoff:
                detected.append("payoff_completion")
            if not weak_start:
                detected.append("standalone_start")
            interest = round(min(1.0, sig["score"] * .78 + .10 * payoff
                                 + .07 * (not weak_start) + .05 * all_sig["information_density"]), 4)
            central = {"start": sentence["start"], "end": sentences[claim_end]["end"], "text": claim_text}
            identity = f"{start:.3f}:{end:.3f}:{central['start']:.3f}:{variant}"
            raw.append({
                "semantic_candidate_id": "semantic_" + hashlib.sha256(identity.encode()).hexdigest()[:12],
                "start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3),
                "transcript": text, "text": text, "semantic_interest_score": interest,
                "detected_signals": detected, "numeric_claims": numeric, "central_claim": central,
                "candidate_source": "semantic", "region_type": variant,
                "region_id": f"semantic_{index}", "region_rank": None,
                "local_interest_score": interest, "local_multimodal_score": round(interest * 100),
                "seed_modalities": ["text"], "anchor_types": ["text"],
                "anchor_strengths": {"semantic_interest": interest},
                "audio": {}, "visual": {},
                "boundary_evidence": {"word_aligned": all(s["word_aligned"] for s in sentences[first:last + 1]),
                                      "start_sentence": first, "end_sentence": last,
                                      "sentence_complete": True, "payoff_completion": payoff},
            })
    raw.sort(key=lambda c: (-c["semantic_interest_score"], c["start"], c["end"], c["semantic_candidate_id"]))
    kept = []
    for candidate in raw:
        duplicate = next((other for other in kept if _duplicate(candidate, other)), None)
        if duplicate:
            candidate["removed_reason"] = "semantic_temporal_text_duplicate"
            candidate["duplicate_of"] = duplicate["semantic_candidate_id"]
        else:
            kept.append(copy.deepcopy(candidate))
    return raw, kept


def _overlap(first, second):
    intersection = max(0.0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    return intersection / max(.001, min(first["end"] - first["start"], second["end"] - second["start"]))


def _duplicate(first, second):
    a, b = set(_tokens(first["text"])), set(_tokens(second["text"]))
    distinct = (first.get("region_type") != second.get("region_type")
                and abs(first["duration"] - second["duration"]) >= 9
                and abs(first["end"] - second["end"]) >= 6)
    # Never replace a rescued claim with a near duplicate that omits that claim.
    claims = [c.get("central_claim") for c in (first, second)]
    claims_fit = all(not c or (max(first["start"], second["start"]) <= c["start"]
                              and min(first["end"], second["end"]) >= c["end"]) for c in claims)
    return (_overlap(first, second) >= .84 and len(a & b) / max(1, len(a | b)) >= .72
            and not distinct and claims_fit)


def merge_candidate_sources(multimodal, semantic, maximum=20):
    """Preserve source fusion/dedupe; delegate only the final local selection."""
    merged = [dict(copy.deepcopy(c), candidate_source="multimodal") for c in multimodal]
    for candidate in semantic:
        duplicate = next((c for c in merged if c["candidate_source"] != "semantic" and _duplicate(candidate, c)), None)
        if duplicate:
            duplicate["candidate_source"] = "semantic+multimodal"
            for key in ("semantic_candidate_id", "semantic_interest_score", "detected_signals", "numeric_claims", "central_claim"):
                # Keep the highest-scoring semantic attribution for this window.
                if key not in duplicate:
                    duplicate[key] = copy.deepcopy(candidate[key])
        else:
            merged.append(copy.deepcopy(candidate))
    selected, _ = select_pre_gpt_candidates(merged, maximum)
    return merged, selected


def _selection_id(candidate):
    return candidate.get("semantic_candidate_id") or (
        f"multimodal_{candidate.get('region_id')}_{candidate.get('region_type')}_"
        f"{candidate['start']}_{candidate['end']}")


def _selection_quality(candidate):
    # Local priority only, not a calibrated viral score or a GPT score change.
    semantic = 100 * float(candidate.get("semantic_interest_score") or 0)
    media = float(candidate.get("local_multimodal_score") or 0)
    return max(semantic, media) if candidate["candidate_source"] != "semantic" else semantic


def _coverage(candidate):
    quantities = {q["quantity"].lower() for q in candidate.get("numeric_claims", []) if q.get("meaningful")}
    substantive = set(candidate.get("detected_signals", [])) & (
        set(RELATIONS) | {"question_answer", "contextual_quantity", "payoff_completion"})
    # Extra duration alone isn't evidence. Prefer shorter when evidence is equal.
    return len(quantities), len(substantive), -candidate["duration"]


def select_pre_gpt_candidates(merged, maximum=20):
    """Quality first, source floors rather than alternation, soft redundancy.

    Scores within one point are treated as a near tie when choosing a variant
    of the same region. Evidence coverage breaks that tie. A 20% representation
    floor protects each source when enough regions exist; hybrids count for both.
    No transcript processing or ground-truth information is used here.
    """
    if maximum <= 0:
        return [], {"version": PRE_GPT_SELECTION_VERSION, "decisions": {}, "steps": []}
    ranked = sorted(merged, key=lambda c: (-_selection_quality(c), c["start"], c["end"]))
    decisions = {_selection_id(c): {"rank_before_selection": rank,
                 "selection_quality": round(_selection_quality(c), 4), "selected": False}
                 for rank, c in enumerate(ranked, 1)}
    groups = {}
    for c in ranked:
        groups.setdefault(c.get("region_id") or _selection_id(c), []).append(c)
    representatives = []
    for variants in groups.values():
        best_score = max(_selection_quality(c) for c in variants)
        near_best = [c for c in variants if _selection_quality(c) >= best_score - 1.0]
        winner = max(near_best, key=lambda c: (_coverage(c), _selection_quality(c), -c["start"]))
        representatives.append(winner)
        for c in variants:
            if c is not winner:
                decisions[_selection_id(c)].update(
                    reason="region_representative", preferred_candidate=_selection_id(winner),
                    preferred_quality=round(_selection_quality(winner), 4),
                    preferred_source=winner["candidate_source"],
                    near_score_coverage_tiebreak=c in near_best)

    def supports(c, source):
        return source in c["candidate_source"].split("+")

    floor = max(1, maximum // 5)
    targets = {s: min(floor, sum(supports(c, s) for c in representatives)) for s in ("semantic", "multimodal")}
    selected, steps = [], []
    remaining = list(representatives)
    wordsets = {_selection_id(c): set(_tokens(c["text"])) for c in representatives}

    def priority(c):
        words = wordsets[_selection_id(c)]
        penalty, blocker = 0., None
        for kept in selected:
            other = wordsets[_selection_id(kept)]
            similarity = len(words & other) / max(1, len(words | other))
            # A different idea isn't rejected merely for sharing video time.
            redundancy = 18.0 * _overlap(c, kept) * similarity
            if redundancy > penalty:
                penalty, blocker = redundancy, _selection_id(kept)
        return _selection_quality(c) - penalty, penalty, blocker

    while remaining and len(selected) < maximum:
        deficits = {s: max(0, targets[s] - sum(supports(c, s) for c in selected)) for s in targets}
        floor_required = maximum - len(selected) <= sum(deficits.values())
        eligible = [c for c in remaining if not floor_required or any(
            deficits[s] and supports(c, s) for s in targets)]
        if not eligible:
            eligible = remaining
        winner = max(eligible, key=lambda c: (priority(c)[0], _selection_quality(c), -c["start"], -c["end"]))
        adjusted, penalty, blocker = priority(winner)
        selected.append(winner)
        remaining.remove(winner)
        step = {"position": len(selected), "candidate_id": _selection_id(winner),
                "candidate_source": winner["candidate_source"], "local_score": winner.get("local_multimodal_score"),
                "selection_quality": round(_selection_quality(winner), 4),
                "adjusted_priority": round(adjusted, 4), "diversity_penalty": round(penalty, 4),
                "redundant_with": blocker, "source_floor_required": floor_required}
        steps.append(step)
        decisions[_selection_id(winner)].update(selected=True, selected_position=len(selected), **{
            k: step[k] for k in ("adjusted_priority", "diversity_penalty", "source_floor_required")})
    for c in remaining:
        adjusted, penalty, blocker = priority(c)
        decisions[_selection_id(c)].update(reason="quality_diversity_budget", adjusted_priority=round(adjusted, 4),
            diversity_penalty=round(penalty, 4), redundant_with=blocker,
            cutoff_candidate=steps[-1] if steps else None)
    return selected, {"version": PRE_GPT_SELECTION_VERSION, "source_floors": targets,
                      "decisions": decisions, "steps": steps}


def refresh_pre_gpt_selection(features):
    """Selection-only cache migration; preserve detection/dedupe and GPT history."""
    started = time.perf_counter()
    selected, debug = select_pre_gpt_candidates(features["merged_candidates_after_dedupe"])
    features["selected_candidates"] = selected
    features["pre_gpt_selection_version"] = PRE_GPT_SELECTION_VERSION
    features["pre_gpt_selection_debug"] = debug
    features["pre_gpt_selection_seconds"] = round(time.perf_counter() - started, 6)
    features["candidate_selection_status"] = "local_only_not_ranked"
    # Existing recall describes the old selection until an explicit annotated audit.
    recall = features.get("semantic_ground_truth_recall", {})
    if "selected_for_gpt_not_sent" in recall:
        recall["selected_for_gpt_not_sent"] = {"status": "needs_selection_reevaluation"}
    return features


def update_semantic_features(features, transcript):
    """Augment cached/new media analysis; never rerun or relabel old GPT output."""
    fingerprint = hashlib.sha256(json.dumps(transcript, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if features.get("semantic_pass_version") == SEMANTIC_VERSION and features.get("semantic_transcript_fingerprint") == fingerprint:
        if features.get("pre_gpt_selection_version") != PRE_GPT_SELECTION_VERSION:
            return refresh_pre_gpt_selection(features), True
        return features, False
    started = time.perf_counter()
    raw, deduped = generate_semantic_candidates(transcript)
    semantic_seconds = time.perf_counter() - started
    multimodal = features.get("multimodal_candidates_after_dedupe", features.get(
        "deduplicated_candidates", features.get("raw_candidates", [])))
    merged, selected = merge_candidate_sources(multimodal, deduped)
    features["semantic_pass_version"] = SEMANTIC_VERSION
    features["semantic_transcript_fingerprint"] = fingerprint
    features["semantic_candidates_raw"] = raw
    features["semantic_candidates_after_dedupe"] = deduped
    features["semantic_ground_truth_recall"] = {"status": "not_evaluated", "reason": "no ground-truth annotations supplied"}
    features["multimodal_candidates_after_dedupe"] = [dict(copy.deepcopy(c), candidate_source="multimodal") for c in multimodal]
    features["merged_candidates_after_dedupe"] = merged
    features["selected_candidates"] = selected
    features["candidate_selection_status"] = "local_only_not_ranked"
    # This historical field remains the record of the previous request until
    # app.py actually invokes its existing ranker with the new selection.
    features.setdefault("candidates_sent_to_ai", [])
    for key in ("raw_candidates", "raw_candidate_variants", "deduplicated_candidates"):
        for candidate in features.get(key, []):
            candidate.setdefault("candidate_source", "multimodal")
    features["semantic_pass_timings"] = {
        "detection_and_dedupe_seconds": round(semantic_seconds, 6),
        "including_merge_seconds": round(time.perf_counter() - started, 6),
    }
    refresh_pre_gpt_selection(features)
    return features, True


def evaluate_recall(candidates, annotations):
    """Manual claim bounds are evaluation-only, never candidate generation input."""
    rows = []
    for truth in annotations:
        matches = []
        for candidate in candidates:
            overlap = max(0., min(candidate["end"], truth["end"]) - max(candidate["start"], truth["start"]))
            ratio = overlap / (truth["end"] - truth["start"])
            reason = []
            if ratio >= .30:
                reason.append("overlap_30_percent")
            if abs(candidate["start"] - truth["start"]) <= 8:
                reason.append("start_within_8_seconds")
            claim = truth.get("central_claim")
            if claim and candidate["start"] <= claim["start"] and candidate["end"] >= claim["end"]:
                reason.append("central_claim_fully_contained")
            if reason:
                matches.append({"candidate_id": candidate.get("semantic_candidate_id", candidate.get("candidate_id")),
                                "start": candidate["start"], "end": candidate["end"],
                                "semantic_interest_score": candidate.get("semantic_interest_score"),
                                "detected_signals": candidate.get("detected_signals", []),
                                "candidate_source": candidate.get("candidate_source", "multimodal"),
                                "overlap_ratio": round(ratio, 4), "match_reasons": reason})
        rows.append({**truth, "found": bool(matches), "matches": matches})
    return {"found": sum(r["found"] for r in rows), "total": len(rows), "clips": rows}


def augment_project(project_dir, annotations=None):
    """Explicit offline entrypoint: read existing JSON and update debug/selection."""
    project_dir = Path(project_dir)
    path = project_dir / "analysis_features.json"
    features = json.loads(path.read_text(encoding="utf-8"))
    transcript = json.loads((project_dir / "transcript.json").read_text(encoding="utf-8"))
    baseline = copy.deepcopy(features.get("raw_candidate_variants", features.get("raw_candidates", [])))
    features, changed = update_semantic_features(features, transcript)
    if annotations is not None:
        features["semantic_ground_truth_recall"] = {
            "status": "evaluated_offline", "baseline": evaluate_recall(baseline, annotations),
            "semantic_raw": evaluate_recall(features["semantic_candidates_raw"], annotations),
            "semantic_after_dedupe": evaluate_recall(features["semantic_candidates_after_dedupe"], annotations),
            "selected_for_gpt_not_sent": evaluate_recall(features["selected_candidates"], annotations),
        }
    if changed or annotations is not None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return features
