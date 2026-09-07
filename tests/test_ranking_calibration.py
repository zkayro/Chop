"""Deterministic tests for global best-moments ranking calibration."""

from __future__ import annotations

import unittest

from ranking_calibration import (
    TRANSCRIPT_DOMINANT_CAP,
    apply_celebrity_brand_guardrail,
    apply_quality_guardrails,
    apply_score_confidence_caps,
    celebrity_without_substance,
    component_total,
    decide_publishability,
    demote_same_story_duplicates,
    finalize_relative_ranks,
    global_best_moments_prompt_block,
    has_famous_entity,
    memorable_strength,
    quality_tier_for_score,
    score_band_label,
    variant_strength,
)


def _components(**overrides):
    base = {
        "hook": 18,
        "retention": 15,
        "payoff": 12,
        "emotion_novelty": 11,
        "standalone": 8,
        "shareability": 7,
        "pacing": 4,
    }
    base.update(overrides)
    return base


def _candidate(text, **extra):
    data = {
        "text": text,
        "dead_air_ratio": 0.0,
        "local_interest_score": 0.6,
        "audio": {},
        "visual": {},
        "anchor_types": ["text"],
    }
    data.update(extra)
    return data


def _passthrough_evidence(candidate, components):
    return {
        "strong_evidence_count": 2,
        "independent_source_groups": ["transcript", "audio"],
        "transcript_dominant": False,
        "visual_semantics_available": False,
    }


def _low_evidence(candidate, components):
    return {
        "strong_evidence_count": 1,
        "independent_source_groups": ["transcript"],
        "transcript_dominant": True,
        "visual_semantics_available": False,
    }


def _confidence(candidate, evidence):
    return 0.95, {"transcript": 0.3}


class RankingCalibrationTests(unittest.TestCase):
    def test_celebrity_mediocre_loses_to_strong_non_celebrity_claim(self):
        celeb_text = "Elon Musk walked into the room and waved at everyone briefly."
        claim_text = (
            "Most builders never check this: the panel fails at minus twenty "
            "because the seal shrinks, and that is why the whole unit leaks."
        )
        celeb_comp, celeb_pen = apply_celebrity_brand_guardrail(
            _candidate(celeb_text),
            _components(hook=20, retention=14, payoff=5, emotion_novelty=10, shareability=9),
        )
        claim_comp, _ = apply_celebrity_brand_guardrail(
            _candidate(claim_text),
            _components(hook=19, retention=16, payoff=13, emotion_novelty=12, shareability=8),
        )
        self.assertIn("celebrity_name_without_substance", celeb_pen)
        self.assertLess(component_total(celeb_comp), component_total(claim_comp))
        self.assertLess(component_total(celeb_comp), 75)

    def test_famous_brand_without_payoff_not_high(self):
        text = "Tesla is a big brand and SpaceX is also famous nowadays."
        adjusted, penalties = apply_celebrity_brand_guardrail(
            _candidate(text),
            _components(hook=22, retention=16, payoff=4, emotion_novelty=12, shareability=9),
        )
        self.assertTrue(celebrity_without_substance(text, _components(payoff=4)))
        self.assertIn("celebrity_name_without_substance", penalties)
        self.assertLess(component_total(adjusted), 75)
        self.assertEqual(score_band_label(component_total(adjusted)) in {
            "decent_review_worthy", "weak_generic", "poor", "good_publishable"
        }, True)
        self.assertNotEqual(score_band_label(component_total(adjusted)), "exceptional_best_in_video")

    def test_strong_numerical_claim_reaches_good_or_very_strong(self):
        text = (
            "We cut build time from 14 days to 48 hours because the foam cure "
            "now finishes overnight, and that payoff changes the whole factory."
        )
        components = _components(hook=20, retention=17, payoff=13, emotion_novelty=12, shareability=8)
        adjusted, penalties = apply_quality_guardrails(_candidate(text), components)
        total = component_total(adjusted)
        self.assertNotIn("celebrity_name_without_substance", penalties)
        self.assertGreaterEqual(total, 75)
        self.assertLessEqual(total, 100)
        self.assertIn(score_band_label(total), {
            "good_publishable", "very_strong_best_few", "exceptional_best_in_video"
        })

    def test_generic_clean_clip_does_not_outrank_memorable_moment(self):
        generic = {
            "candidate_id": "candidate_1",
            "decision": "KEEP",
            "score": 80,
            "model_relative_rank": 1,
            "text": "Here is a clean overview of the basic process from start to finish.",
            "ai_scores": _components(hook=18, retention=16, payoff=11, emotion_novelty=6, shareability=5),
        }
        memorable = {
            "candidate_id": "candidate_2",
            "decision": "KEEP",
            "score": 78,
            "model_relative_rank": 2,
            "text": "The shocking truth: the prototype exploded on takeoff, and nobody expected the silence after.",
            "ai_scores": _components(hook=17, retention=15, payoff=13, emotion_novelty=14, shareability=9),
        }
        ranked = finalize_relative_ranks([generic, memorable])
        by_id = {clip["candidate_id"]: clip for clip in ranked}
        self.assertEqual(by_id["candidate_2"]["relative_rank"], 1)
        self.assertGreater(
            memorable_strength(memorable["ai_scores"]),
            memorable_strength(generic["ai_scores"]),
        )

    def test_same_story_three_variants_strongest_wins(self):
        shared = (
            "the factory ships in sixty seconds after the foam cures overnight "
            "and that is why inventory collapsed"
        )
        weak = {
            "candidate_id": "candidate_1",
            "decision": "KEEP",
            "score": 76,
            "model_relative_rank": 1,
            "text": "So and well okay " + shared + " with a long slow setup before the point arrives finally.",
            "ai_scores": _components(hook=14, payoff=10, standalone=6),
        }
        mid = {
            "candidate_id": "candidate_2",
            "decision": "KEEP",
            "score": 78,
            "model_relative_rank": 2,
            "text": shared + " explained carefully for newcomers watching at home today.",
            "ai_scores": _components(hook=17, payoff=11, standalone=8),
        }
        strong = {
            "candidate_id": "candidate_3",
            "decision": "KEEP",
            "score": 79,
            "model_relative_rank": 3,
            "text": "Stop scrolling: " + shared + "!",
            "ai_scores": _components(hook=22, payoff=14, standalone=9),
        }
        ranked = demote_same_story_duplicates([weak, mid, strong])
        winner = min(ranked, key=lambda c: c["relative_rank"])
        self.assertEqual(winner["candidate_id"], "candidate_3")
        demoted = [c for c in ranked if c.get("is_same_story_demoted")]
        self.assertEqual(len(demoted), 2)
        self.assertGreater(
            variant_strength(strong["ai_scores"], strong["text"]),
            variant_strength(weak["ai_scores"], weak["text"]),
        )

    def test_good_clip_without_famous_entity_can_reach_75_85(self):
        text = (
            "Why do roofs fail in the first winter? Because the vapor barrier was "
            "installed backwards, and the numbers prove a 37 percent moisture spike."
        )
        self.assertFalse(has_famous_entity(text))
        components = _components(hook=20, retention=16, payoff=13, emotion_novelty=11, shareability=8, pacing=4)
        adjusted, _ = apply_quality_guardrails(_candidate(text), components)
        details = apply_score_confidence_caps(
            _candidate(text),
            adjusted,
            evidence_fn=_low_evidence,
            confidence_fn=lambda c, e: (0.55, {"transcript": 0.3}),
        )
        self.assertGreaterEqual(details["final_score"], 75)
        self.assertLessEqual(details["final_score"], 85)
        # Transcript-dominant ceiling remains available without collapsing the 75-85 band.
        self.assertEqual(TRANSCRIPT_DOMINANT_CAP, 88)
        high = dict(adjusted)
        high["hook"] = 25
        high["retention"] = 20
        high["payoff"] = 15
        high["emotion_novelty"] = 15
        high["standalone"] = 10
        high["shareability"] = 10
        high["pacing"] = 5
        capped = apply_score_confidence_caps(
            _candidate(text),
            high,
            evidence_fn=_low_evidence,
            confidence_fn=lambda c, e: (0.55, {"transcript": 0.3}),
        )
        self.assertEqual(capped["score_cap"], TRANSCRIPT_DOMINANT_CAP)
        self.assertEqual(capped["final_score"], TRANSCRIPT_DOMINANT_CAP)

    def test_weak_famous_stays_below_good_range(self):
        text = "Elon Musk smiled. Tesla looked shiny. That was the whole clip."
        adjusted, penalties = apply_quality_guardrails(
            _candidate(text),
            _components(hook=19, retention=14, payoff=5, emotion_novelty=9, shareability=8),
        )
        self.assertIn("celebrity_name_without_substance", penalties)
        self.assertLess(component_total(adjusted), 75)

    def test_multiple_good_clips_not_only_one_above_70(self):
        clips = []
        for index, text in enumerate((
            "The reveal: shipping dropped to forty eight hours because curing finished overnight.",
            "Investors gasped when the founder said margins flipped from minus twelve to plus nineteen.",
            "The funny failure: the demo door fell off, and that punchline sold the redesign.",
        ), start=1):
            comps = _components(hook=19, retention=16, payoff=12, emotion_novelty=11, shareability=8)
            adjusted, _ = apply_quality_guardrails(_candidate(text), comps)
            details = apply_score_confidence_caps(
                _candidate(text),
                adjusted,
                evidence_fn=_passthrough_evidence,
                confidence_fn=_confidence,
            )
            clips.append(details["final_score"])
        above = [score for score in clips if score >= 70]
        self.assertGreaterEqual(len(above), 2)

    def test_missing_payoff_penalized_once_not_triple(self):
        text = "What is the real reason factories stall mid-shift without any resolution here?"
        gpt_components = _components(payoff=4, retention=14, hook=16)
        adjusted, penalties = apply_quality_guardrails(_candidate(text), gpt_components)
        self.assertIn("missing_payoff_acknowledged", penalties)
        self.assertNotIn("missing_payoff", penalties)
        # Only a light retention nudge, not another -7 payoff cut.
        self.assertEqual(adjusted["payoff"], 4)
        self.assertGreaterEqual(adjusted["retention"], 13)
        # Decision path applies one gate, not stacked collapse below review by default.
        decision = decide_publishability(
            model_decision="KEEP",
            component_total_score=component_total(adjusted),
            severe_gate_failures=["no credible payoff"] if adjusted["payoff"] <= 4 else [],
            narrow_gate_failures=[],
        )
        self.assertEqual(decision, "REJECT")

    def test_strongest_in_set_gets_relative_rank_one(self):
        ranked = finalize_relative_ranks([
            {
                "candidate_id": "candidate_1",
                "decision": "BORDERLINE",
                "score": 70,
                "model_relative_rank": 3,
                "text": "A mild summary of workshop lighting choices.",
                "ai_scores": _components(hook=12, payoff=8, emotion_novelty=5),
            },
            {
                "candidate_id": "candidate_2",
                "decision": "KEEP",
                "score": 84,
                "model_relative_rank": 1,
                "text": "The impossible claim finally gets proven with the overnight cure result!",
                "ai_scores": _components(hook=22, payoff=14, emotion_novelty=13, shareability=9),
            },
            {
                "candidate_id": "candidate_3",
                "decision": "KEEP",
                "score": 77,
                "model_relative_rank": 2,
                "text": "A solid tip about sealing panels before winter storms arrive.",
                "ai_scores": _components(hook=18, payoff=11, emotion_novelty=8),
            },
        ])
        self.assertEqual(ranked[0]["candidate_id"], "candidate_2")
        self.assertEqual(ranked[0]["relative_rank"], 1)

    def test_prompt_contains_global_and_celebrity_rules(self):
        prompt = global_best_moments_prompt_block()
        self.assertIn("Do not evaluate candidates in isolation", prompt)
        self.assertIn("CELEBRITY AND BRAND BIAS RULE", prompt)
        self.assertIn("relative_rank", prompt)

    def test_quality_tiers_match_calibration_bands(self):
        self.assertEqual(quality_tier_for_score(91), "exceptional")
        self.assertEqual(quality_tier_for_score(83), "strong")
        self.assertEqual(quality_tier_for_score(76), "good")
        self.assertEqual(quality_tier_for_score(70), "review")
        self.assertEqual(quality_tier_for_score(60), "weak")
        self.assertEqual(quality_tier_for_score(40), "poor")

    def test_borderline_band_keeps_decent_clips_visible(self):
        decision = decide_publishability(
            model_decision="REJECT",
            component_total_score=70,
            severe_gate_failures=[],
            narrow_gate_failures=[],
        )
        self.assertEqual(decision, "BORDERLINE")


if __name__ == "__main__":
    unittest.main()
