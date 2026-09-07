"""Global best-moments ranking calibration for Chopify AI scoring.

Pure helpers used by app.py after GPT evaluations. No Streamlit / media deps.
"""

from __future__ import annotations

import re
from typing import Any, Callable

RANKING_CALIBRATION_VERSION = "global-best-moments-v1"

FAMOUS_ENTITY_TERMS = (
    "elon musk", "elon", "musk", "tesla", "spacex", "openai", "chatgpt",
    "sam altman", "bill gates", "jeff bezos", "amazon", "google", "alphabet",
    "microsoft", "apple", "meta", "facebook", "instagram", "tiktok", "youtube",
    "twitter", "x.com", "nvidia", "amd", "intel", "netflix", "disney",
    "warner", "nba", "nfl", "fifa", "uefa", "olympics", "beyonce",
    "taylor swift", "kanye", "kim kardashian", "oprah", "trump", "biden",
    "obama", "putin", "zelensky", "nasa", "cnn", "bbc", "forbes", "wall street",
    "silicon valley", "warren buffett", "mark zuckerberg", "zuckerberg",
    "steve jobs", "tim cook", "sundar pichai", "satya nadella", "boxabl",
)

SUBSTANCE_TERMS = (
    "because", "therefore", "the answer", "that's why", "result", "revealed",
    "reveal", "secret", "mistake", "truth", "never", "impossible", "proves",
    "proof", "actually", "instead", "but", "however", "punchline", "payoff",
    "weil", "deshalb", "die antwort", "darum", "wahrheit", "fehler", "niemals",
    "unmöglich", "beweist", "stattdessen", "aber", "doch", "überrasch",
)

# Soft local ceilings — good transcript-only clips may still reach very-strong (82-88).
TRANSCRIPT_DOMINANT_CAP = 88
NINETY_PLUS_HARD_CAP = 89


def quality_tier_for_score(score: int | float) -> str:
    value = int(score)
    if value >= 90:
        return "exceptional"
    if value >= 82:
        return "strong"
    if value >= 75:
        return "good"
    if value >= 68:
        return "review"
    if value >= 55:
        return "weak"
    return "poor"


def score_band_label(score: int | float) -> str:
    value = int(score)
    if value >= 90:
        return "exceptional_best_in_video"
    if value >= 82:
        return "very_strong_best_few"
    if value >= 75:
        return "good_publishable"
    if value >= 68:
        return "decent_review_worthy"
    if value >= 55:
        return "weak_generic"
    return "poor"


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def has_famous_entity(text: str) -> bool:
    lowered = normalize_text(text)
    return any(term in lowered for term in FAMOUS_ENTITY_TERMS)


def has_memorable_substance(text: str, components: dict | None = None) -> bool:
    """Text-first substance check. Component scores are optional corroboration only."""
    lowered = normalize_text(text)
    components = components or {}
    numeric = bool(re.search(r"\b\d+(?:[.,]\d+)?%?\b", lowered))
    substance_hit = any(term in lowered for term in SUBSTANCE_TERMS)
    reaction_or_story = any(
        term in lowered for term in (
            "revealed", "exploded", "confess", "lawsuit", "bankrupt", "fired",
            "punchline", "betrayed", "scandal", "breakthrough",
        )
    )
    # Only strong payoff corroborates; hook/emotion alone can be celebrity-inflated.
    strong_payoff = int(components.get("payoff", 0) or 0) >= 11
    return bool(
        numeric or substance_hit or reaction_or_story or strong_payoff
        or ("?" in lowered and "!" in lowered)
    )


def celebrity_without_substance(text: str, components: dict | None = None) -> bool:
    return has_famous_entity(text) and not has_memorable_substance(text, components)


def component_total(components: dict) -> int:
    return int(sum(int(components.get(key, 0) or 0) for key in (
        "hook", "retention", "payoff", "emotion_novelty",
        "standalone", "shareability", "pacing",
    )))


def apply_celebrity_brand_guardrail(candidate: dict, components: dict) -> tuple[dict, list[str]]:
    """Prevent famous names from rescuing mediocre clips. Never boosts scores."""
    adjusted = dict(components)
    penalties: list[str] = []
    text = candidate.get("text") or ""
    if not celebrity_without_substance(text, adjusted):
        return adjusted, penalties

    if adjusted.get("shareability", 0) > 5:
        adjusted["shareability"] = min(int(adjusted["shareability"]), 5)
        penalties.append("celebrity_name_without_substance")
    if adjusted.get("hook", 0) > 16:
        adjusted["hook"] = min(int(adjusted["hook"]), 16)
        if "celebrity_name_without_substance" not in penalties:
            penalties.append("celebrity_name_without_substance")
    if adjusted.get("emotion_novelty", 0) > 8:
        adjusted["emotion_novelty"] = min(int(adjusted["emotion_novelty"]), 8)

    total = component_total(adjusted)
    if total >= 75:
        overflow = total - 74
        for key in ("shareability", "hook", "emotion_novelty", "retention"):
            if overflow <= 0:
                break
            floor = 3 if key != "retention" else 8
            reducible = max(0, int(adjusted[key]) - floor)
            cut = min(reducible, overflow)
            adjusted[key] = int(adjusted[key]) - cut
            overflow -= cut
        if "celebrity_name_without_substance" not in penalties:
            penalties.append("celebrity_name_without_substance")
    return adjusted, penalties


def apply_quality_guardrails(candidate: dict, components: dict) -> tuple[dict, list[str]]:
    """Local safety penalties. Avoid re-penalizing signals GPT already scored low."""
    adjusted = {key: int(components.get(key, 0) or 0) for key in (
        "hook", "retention", "payoff", "emotion_novelty",
        "standalone", "shareability", "pacing",
    )}
    penalties: list[str] = []
    text = normalize_text(candidate.get("text"))
    gpt_payoff = adjusted["payoff"]

    intro_pattern = re.compile(
        r"^(?:hey|hello|hi|welcome|what'?s up|good morning|hallo|hey leute|"
        r"herzlich willkommen|guten morgen|willkommen)\b"
    )
    context_pattern = re.compile(
        r"^(?:and then|like i said|as mentioned|this one|that |so yeah|"
        r"und dann|wie gesagt|wie erwähnt|dieses hier|das |also ja)\b"
    )
    sponsor_terms = (
        "sponsor", "sponsored by", "use my code", "link in the description",
        "werbung", "rabattcode", "link in der beschreibung",
    )
    if intro_pattern.search(text) or any(term in text[:180] for term in sponsor_terms):
        adjusted["hook"] = max(0, adjusted["hook"] - 12)
        adjusted["retention"] = max(0, adjusted["retention"] - 5)
        adjusted["standalone"] = max(0, adjusted["standalone"] - 2)
        penalties.append("intro_or_promo")
    if context_pattern.search(text):
        adjusted["hook"] = max(0, adjusted["hook"] - 7)
        adjusted["standalone"] = max(0, adjusted["standalone"] - 5)
        penalties.append("missing_context")

    dead_air = float(candidate.get("dead_air_ratio", 0) or 0)
    if dead_air >= 0.16:
        adjusted["pacing"] = max(0, adjusted["pacing"] - max(1, round(dead_air * 8)))
        adjusted["retention"] = max(0, adjusted["retention"] - round(dead_air * 10))
        penalties.append("dead_air")

    unfinished_end = re.search(
        r"\b(?:and|but|because|so|then|und|aber|weil|dass|also|dann)[,. ]*$",
        text,
    )
    question_without_resolution = (
        "?" in text
        and not any(term in text for term in (
            "because", "therefore", "the answer", "that's why",
            "weil", "deshalb", "die antwort", "darum",
        ))
    )
    # Double-penalty fix: GPT low payoff already encodes missing payoff.
    if unfinished_end or question_without_resolution:
        if gpt_payoff <= 6:
            if adjusted["retention"] > 0:
                adjusted["retention"] = max(0, adjusted["retention"] - 1)
            penalties.append("missing_payoff_acknowledged")
        else:
            adjusted["payoff"] = max(0, adjusted["payoff"] - 7)
            adjusted["retention"] = max(0, adjusted["retention"] - 3)
            penalties.append("missing_payoff")

    weak_evidence = not (
        re.search(r"\b\d+(?:[.,]\d+)?%?\b", text)
        or "?" in text or "!" in text
        or any(term in text for term in (
            "truth", "mistake", "problem", "never", "impossible", "surpris",
            "wahrheit", "fehler", "problem", "niemals", "unmöglich", "überrasch",
        ))
        or float(candidate.get("local_interest_score", 0) or 0) >= 0.5
        or (
            isinstance(candidate.get("audio"), dict)
            and candidate.get("audio", {}).get("energy_spikes", 0)
        )
    )
    if weak_evidence and component_total(adjusted) >= 85:
        adjusted["hook"] = min(adjusted["hook"], 17)
        adjusted["emotion_novelty"] = min(adjusted["emotion_novelty"], 9)
        adjusted["shareability"] = min(adjusted["shareability"], 6)
        penalties.append("limited_viral_evidence")

    celebrity_adjusted, celebrity_penalties = apply_celebrity_brand_guardrail(
        candidate, adjusted
    )
    adjusted = celebrity_adjusted
    penalties.extend(celebrity_penalties)
    return adjusted, penalties


def apply_score_confidence_caps(
    candidate: dict,
    components: dict,
    *,
    evidence_fn: Callable[[dict, dict], dict],
    confidence_fn: Callable[[dict, dict], tuple[float, dict]],
) -> dict:
    """Cap extreme scores without collapsing good clips out of the 75-85 band."""
    raw_score = component_total(components)
    evidence = evidence_fn(candidate, components)
    confidence_score, confidence_breakdown = confidence_fn(candidate, evidence)
    score_cap = 100
    cap_reasons: list[str] = []

    if evidence.get("transcript_dominant") and raw_score > TRANSCRIPT_DOMINANT_CAP:
        score_cap = TRANSCRIPT_DOMINANT_CAP
        cap_reasons.append(f"transcript_dominant_max_{TRANSCRIPT_DOMINANT_CAP}")

    if raw_score > 90:
        required_90_scores = (
            int(components.get("hook", 0)) >= 22
            and int(components.get("retention", 0)) >= 17
            and int(components.get("payoff", 0)) >= 13
            and int(components.get("standalone", 0)) >= 8
            and int(components.get("emotion_novelty", 0)) >= 10
        )
        name_only = celebrity_without_substance(candidate.get("text") or "", components)
        sufficient_evidence = (
            not name_only
            and int(evidence.get("strong_evidence_count", 0) or 0) >= 2
            and len(evidence.get("independent_source_groups", []) or []) >= 2
            and confidence_score >= 0.90
        )
        if not required_90_scores or not sufficient_evidence:
            score_cap = min(score_cap, NINETY_PLUS_HARD_CAP)
            cap_reasons.append("insufficient_independent_evidence_for_90_plus")
        if name_only:
            score_cap = min(score_cap, 74)
            cap_reasons.append("celebrity_without_substance_blocks_90")

    return {
        "raw_score": raw_score,
        "final_score": min(raw_score, score_cap),
        "score_cap": score_cap,
        "score_cap_reasons": cap_reasons,
        "confidence_score": confidence_score,
        "confidence_breakdown": confidence_breakdown,
        "evidence": evidence,
    }


def decide_publishability(
    *,
    model_decision: str,
    component_total_score: int,
    severe_gate_failures: list[str],
    narrow_gate_failures: list[str],
) -> str:
    """KEEP=publishable, BORDERLINE=review, REJECT=weak. No forced KEEP count."""
    if not severe_gate_failures and model_decision == "KEEP" and component_total_score >= 75:
        return "KEEP"
    if not severe_gate_failures and (
        component_total_score >= 68
        or (component_total_score >= 60 and len(narrow_gate_failures) <= 1)
        or (component_total_score >= 55 and len(narrow_gate_failures) == 1)
    ):
        return "BORDERLINE"
    return "REJECT"


def memorable_strength(components: dict) -> float:
    emotion = int(
        components.get("emotion_novelty", components.get("emotion_novelty_conflict", 0)) or 0
    )
    return (
        emotion * 1.2
        + int(components.get("payoff", 0) or 0) * 1.1
        + int(components.get("hook", 0) or 0) * 0.5
        + int(components.get("shareability", 0) or 0) * 0.4
    )


def same_story_key(text: str) -> frozenset[str]:
    tokens = [
        token for token in re.findall(r"\w+", normalize_text(text))
        if len(token) >= 4
    ]
    return frozenset(tokens[:24])


def same_story_similarity(first: str, second: str) -> float:
    a = same_story_key(first)
    b = same_story_key(second)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def variant_strength(components: dict, text: str = "") -> float:
    words = normalize_text(text).split()
    setup_penalty = 0.0
    if words and words[0] in {"so", "and", "well", "okay", "also", "und"}:
        setup_penalty += 2.0
    if len(words) > 140:
        setup_penalty += 1.5
    return (
        int(components.get("hook", 0) or 0)
        + int(components.get("payoff", 0) or 0) * 1.3
        + int(components.get("standalone", 0) or 0)
        - setup_penalty
    )


def demote_same_story_duplicates(
    ranked: list[dict], similarity_threshold: float = 0.72
) -> list[dict]:
    if len(ranked) < 2:
        return ranked
    ordered = sorted(
        ranked,
        key=lambda clip: (
            -variant_strength(
                clip.get("ai_scores") or clip.get("component_scores") or {},
                clip.get("text", ""),
            ),
            int(clip.get("relative_rank") or 10**6),
            -int(clip.get("score", 0) or 0),
        ),
    )
    winners: list[dict] = []
    demoted_ids: set[str] = set()
    for clip in ordered:
        text = clip.get("text", "")
        duplicate_of = None
        for winner in winners:
            if same_story_similarity(text, winner.get("text", "")) >= similarity_threshold:
                duplicate_of = winner
                break
        if duplicate_of is not None:
            demoted_ids.add(str(clip.get("candidate_id")))
            notes = list(clip.get("duplicate_notes") or [])
            notes.append(f"same_story_weaker_than:{duplicate_of.get('candidate_id')}")
            clip["duplicate_notes"] = notes
        else:
            winners.append(clip)

    survivors = [c for c in ranked if str(c.get("candidate_id")) not in demoted_ids]
    demoted = [c for c in ranked if str(c.get("candidate_id")) in demoted_ids]
    survivors.sort(key=lambda c: (
        int(c.get("relative_rank") or 10**6),
        -memorable_strength(c.get("ai_scores") or {}),
        -int(c.get("score", 0) or 0),
    ))
    demoted.sort(key=lambda c: (
        int(c.get("relative_rank") or 10**6),
        -int(c.get("score", 0) or 0),
    ))
    final = survivors + demoted
    for index, clip in enumerate(final, start=1):
        clip["relative_rank"] = index
        clip["is_same_story_demoted"] = str(clip.get("candidate_id")) in demoted_ids
    return final


def finalize_relative_ranks(ranked: list[dict]) -> list[dict]:
    """Unique relative_rank among ALL candidates; memorable substance can reorder close scores."""

    def sort_key(clip: dict):
        model_rank = clip.get("model_relative_rank")
        components = clip.get("ai_scores") or {}
        decision_priority = {"KEEP": 0, "BORDERLINE": 1, "REJECT": 2}.get(
            clip.get("decision"), 3
        )
        score = int(clip.get("score", 0) or 0)
        score_bucket = -((score + 2) // 4)
        return (
            decision_priority,
            int(model_rank) if model_rank is not None else 10**6,
            score_bucket,
            -memorable_strength(components),
            -score,
            str(clip.get("candidate_id") or ""),
        )

    ordered = demote_same_story_duplicates(sorted(ranked, key=sort_key))
    normalized = sorted(
        ordered,
        key=lambda c: (
            int(c.get("relative_rank") or 10**6),
            {"KEEP": 0, "BORDERLINE": 1, "REJECT": 2}.get(c.get("decision"), 3),
            -int(c.get("score", 0) or 0),
        ),
    )
    for index, clip in enumerate(normalized, start=1):
        clip["relative_rank"] = index
    return sorted(
        normalized,
        key=lambda clip: (
            {"KEEP": 0, "BORDERLINE": 1, "REJECT": 2}.get(clip.get("decision"), 3),
            int(clip.get("relative_rank") or 10**6),
            -memorable_strength(clip.get("ai_scores") or {}),
            -int(clip.get("score", 0) or 0),
        ),
    )


def ranking_sort_key(clip: dict, candidate_count: int = 100):
    return (
        {"KEEP": 0, "BORDERLINE": 1, "REJECT": 2}.get(clip.get("decision"), 3),
        int(clip.get("relative_rank") or candidate_count + 1),
        -memorable_strength(clip.get("ai_scores") or {}),
        -int(clip.get("score", 0) or 0),
    )


def global_best_moments_prompt_block() -> str:
    return " ".join((
        "GLOBAL BEST-MOMENTS RANKING: Do not evaluate candidates in isolation. "
        "Compare them against the strongest alternatives in the full candidate set.",
        "Before scoring individuals, conceptually identify the strongest hooks, reveals, "
        "payoffs, surprising claims, funny or memorable beats, stories, emotional or "
        "controversial moments, and standalone-worthy clips in this video, then rank each "
        "candidate relative to those peaks.",
        "Ask: is this one of the BEST moments in the entire video? Decent clips must NOT "
        "rank highly when stronger moments exist elsewhere in the candidate set.",
        "Ranking priorities in order: (1) Scroll Stop/Hook, (2) Curiosity/Retention, "
        "(3) Payoff, (4) Standalone Clarity, (5) Memorable Substance, (6) Shareability, "
        "(7) Pacing.",
        "CELEBRITY AND BRAND BIAS RULE: Do not rank highly mainly because a transcript "
        "mentions famous names, celebrities, brands, or trending companies such as Elon Musk, "
        "Tesla, SpaceX, or similar. A famous name is not viral value. Named entities help ONLY "
        "when the clip also has a strong claim, reveal, surprise, reaction, story, stakes, "
        "punchline, or payoff. Mediocre content plus a celebrity mention must not be rescued.",
        "SCORE CALIBRATION (absolute bands, calibrated against this video's best moments): "
        "90-100 exceptional best-in-video publishable; 82-89 very strong best-few; "
        "75-81 good publishable; 68-74 decent/review-worthy; 55-67 weak/generic; below 55 poor. "
        "Good non-celebrity clips with strong claims MUST be able to reach 75-85. Weak "
        "celebrity clips must stay clearly below the good range. 90+ requires ALL of: "
        "exceptional scroll-stop, strong retention, clear payoff, strong standalone, memorable "
        "substance, and relative superiority in this set. Famous names alone cannot create 90.",
        "Among candidates that tell the same story or claim, keep the strongest version "
        "(best hook + cleanest payoff + least setup) at the better relative_rank; near-duplicates "
        "must not occupy multiple top ranks.",
        "Assign relative_rank as the position among ALL candidates in this video (1 = best). "
        "Every candidate gets a unique relative_rank from 1 to N. Final ordering must reflect "
        "this relative comparison, not merely raw component totals — a stronger memorable "
        "moment may outrank a slightly higher raw component sum.",
    ))


def calibrated_scoring_bands_prompt() -> str:
    return (
        "Score scroll-stop 0-25, retention/forward momentum 0-20, payoff 0-15, "
        "emotion/novelty/conflict 0-15, standalone clarity 0-10, shareability/comment potential 0-10, "
        "and pacing 0-5. Do not return overall_score; Python sums these seven components locally. "
        "Use the global calibration bands: 90-100 exceptional, 82-89 very strong, 75-81 good, "
        "68-74 decent/review, 55-67 weak, <55 poor. A local total below 68 is normally not KEEP. "
        "A score of 75 may still be REJECT for no payoff, missing context, a poor start, or being "
        "clearly weaker than this set's best moments."
    )
