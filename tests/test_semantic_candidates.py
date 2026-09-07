import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semantic_candidates import (generate_semantic_candidates, numeric_claims,
                                 transcript_sentences, merge_candidate_sources,
                                 update_semantic_features, evaluate_recall,
                                 select_pre_gpt_candidates, refresh_pre_gpt_selection)

ROOT = Path(__file__).resolve().parents[1]


class SemanticPassTests(unittest.TestCase):
    def test_contextual_numbers_and_spelled_quantities(self):
        examples = [
            "They have 160,000 names on the waitlist.",
            "We plan to make a home every 60 seconds.",
            "It takes four hours to build a home today.",
            "We will make a home in 20 to 40 minutes.",
            "We have a dozen stations in production.",
            "Full production will have 180 stations.",
            "The home is 1,000 to 1,100 square feet.",
            "It is 8.5 feet wide to reduce shipping cost.",
            "We ship at eight and a half feet wide.",
            "We can reduce shipping costs by 35%.",
            "It's about 1100 square feet and comes down to efficient space planning.",
        ]
        for text in examples:
            with self.subTest(text=text):
                self.assertTrue(any(c["meaningful"] for c in numeric_claims(text)))
        for text in ("Welcome to episode 2.", "This happened in 2026.", "Chapter 4 is here."):
            self.assertFalse(any(c["meaningful"] for c in numeric_claims(text)))

    def test_sentence_across_whisper_segments_and_decimal_word_times(self):
        data = {"segments": [{"start": 0, "end": 4, "text": "We build a home"},
                             {"start": 4, "end": 8, "text": "every 60 seconds."}]}
        sentences = transcript_sentences(data)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]["end"], 8)
        self.assertEqual(sentences[0]["text"], "We build a home every 60 seconds.")
        word_data = {"words": [{"word": w, "start": i, "end": i + 1}
                                for i, w in enumerate("It is 8.5 feet wide.".split())]}
        self.assertEqual(len(transcript_sentences(word_data)), 1)
        split_numbers = {"words": [{"word": w, "start": i, "end": i + 1}
                                   for i, w in enumerate([
                                       " They", " have", " over", " 160", ",000", " names",
                                       " on", " a", " wait", " list.",
                                       " It", " is", " 11", "00", " square", " feet."])]}
        joined = " ".join(s["text"] for s in transcript_sentences(split_numbers))
        self.assertIn("160,000 names", joined)
        self.assertIn("1100 square feet", joined)
        self.assertTrue(any(q["quantity"] == "160,000 names" for q in numeric_claims(joined)))

    def test_no_trigger_vocabulary_still_finds_explanation(self):
        data = {"segments": [
            {"start": 0, "end": 10, "text": "The outer shell keeps the internal temperature stable during the night."},
            {"start": 10, "end": 21, "text": "Water inside the walls stores heat and releases it into the room."},
            {"start": 21, "end": 31, "text": "This means the occupants can stay comfortable without using a heater."},
        ]}
        raw, _ = generate_semantic_candidates(data)
        self.assertTrue(raw)
        self.assertTrue(any("explanation" in c["detected_signals"] for c in raw))

    def test_no_unanswered_question_or_incomplete_end(self):
        self.assertEqual(generate_semantic_candidates({"segments": [
            {"start": 0, "end": 20, "text": "Why would you ever want to build a house this way?"}]}), ([], []))
        self.assertEqual(generate_semantic_candidates({"segments": [
            {"start": 0, "end": 20, "text": "This is the most important reason we need to"}]}), ([], []))
        raw, _ = generate_semantic_candidates({"segments": [
            {"start": 0, "end": 5, "text": "What is the answer?"},
            {"start": 25, "end": 45, "text": "We can build a home every 60 seconds instead of four hours."}]})
        self.assertFalse(any(c["central_claim"]["start"] == 0 for c in raw))

    def test_empty_and_determinism(self):
        self.assertEqual(generate_semantic_candidates({}), ([], []))
        data = {"segments": [{"start": 10, "end": 30, "text": "We can make a home every 60 seconds instead of waiting four hours."}]}
        self.assertEqual(generate_semantic_candidates(data), generate_semantic_candidates(data))

    def test_mix_survives_incomparable_scores_and_preserves_provenance(self):
        multi = [{"start": i * 60., "end": i * 60. + 20, "duration": 20,
                  "text": f"Existing visual event region {i}", "region_id": i,
                  "region_type": "SHORT", "local_multimodal_score": 100}
                 for i in range(25)]
        semantic = [{"start": 2000 + i * 60., "end": 2020 + i * 60., "duration": 20,
                     "text": f"A different complete idea {i}", "region_id": f"s{i}",
                     "region_type": "SHORT", "semantic_interest_score": .1,
                     "candidate_source": "semantic"} for i in range(25)]
        _, selected = merge_candidate_sources(multi, semantic)
        self.assertEqual(len(selected), 20)
        self.assertEqual(sum(c["candidate_source"] == "semantic" for c in selected), 4)
        # Quality dominates; the weaker source gets its floor, not half the slots.
        duplicate = dict(multi[0], candidate_source="semantic", semantic_candidate_id="s0",
                         semantic_interest_score=.8, detected_signals=[], numeric_claims=[],
                         central_claim={"start": 0, "end": 20, "text": multi[0]["text"]})
        merged, _ = merge_candidate_sources(multi, [duplicate])
        self.assertEqual(len(merged), 25)
        self.assertEqual(merged[0]["candidate_source"], "semantic+multimodal")

    def test_near_equal_variants_keep_extra_information_not_just_duration(self):
        base = {"start": 0., "end": 20., "duration": 20., "text": "A complete useful claim.",
                "region_id": "s1", "candidate_source": "semantic", "semantic_interest_score": .81,
                "semantic_candidate_id": "short", "numeric_claims": [{"quantity": "10 homes", "meaningful": True}]}
        longer = dict(base, end=40., duration=40., semantic_candidate_id="long",
                      semantic_interest_score=.807, numeric_claims=base["numeric_claims"] + [{"quantity": "60 seconds", "meaningful": True}])
        selected, debug = select_pre_gpt_candidates([base, longer])
        self.assertEqual(selected[0]["semantic_candidate_id"], "long")
        self.assertEqual(debug["decisions"]["short"]["preferred_candidate"], "long")
        filler = dict(longer, numeric_claims=base["numeric_claims"])
        self.assertEqual(select_pre_gpt_candidates([base, filler])[0][0]["semantic_candidate_id"], "short")

    def test_selection_only_preserves_detection_and_does_not_use_ground_truth(self):
        project = ROOT / "projects/20260907T100518Z_76efee8f"
        if not project.is_dir():
            self.skipTest("Optional local project missing")
        original = json.loads((project / "analysis_features.json").read_text(encoding="utf-8"))
        features = copy.deepcopy(original)
        with patch("semantic_candidates.generate_semantic_candidates", side_effect=AssertionError("must reuse candidates")), \
             patch("socket.socket", side_effect=AssertionError("no network")), \
             patch("subprocess.run", side_effect=AssertionError("no media")):
            result = refresh_pre_gpt_selection(features)
        for key in ("semantic_candidates_raw", "semantic_candidates_after_dedupe", "merged_candidates_after_dedupe",
                    "interest_anchors", "visual_events", "audio_events", "candidates_sent_to_ai", "final_ranking"):
            self.assertEqual(result.get(key), original.get(key), key)
        fixture = json.loads((ROOT / "tests/fixtures/boxabl_ground_truth.json").read_text())
        self.assertGreaterEqual(evaluate_recall(result["selected_candidates"], fixture["clips"])["found"], 4)
        self.assertLessEqual(len(result["selected_candidates"]), 20)
        # Translation in time must not change which candidate IDs win.
        shifted = copy.deepcopy(original["merged_candidates_after_dedupe"])
        for c in shifted:
            c["start"] += 10000
            c["end"] += 10000
        translated, _ = select_pre_gpt_candidates(shifted)
        self.assertEqual([c.get("semantic_candidate_id", c.get("region_id")) for c in result["selected_candidates"]],
                         [c.get("semantic_candidate_id", c.get("region_id")) for c in translated])

    def test_actual_project_recall_and_no_api_or_media(self):
        project = ROOT / "projects/20260907T100518Z_76efee8f"
        if not project.is_dir():
            self.skipTest("Optional local ground-truth project not present")
        fixture = json.loads((ROOT / "tests/fixtures/boxabl_ground_truth.json").read_text())
        transcript = json.loads((project / "transcript.json").read_text(encoding="utf-8"))
        features = json.loads((project / "analysis_features.json").read_text(encoding="utf-8"))
        original = copy.deepcopy(features)
        # Force the pass in memory even if the project already has cached results.
        features.pop("semantic_pass_version", None)
        with patch("subprocess.run", side_effect=AssertionError("media process forbidden")), \
             patch("socket.socket", side_effect=AssertionError("network forbidden")):
            result, changed = update_semantic_features(features, transcript)
        self.assertTrue(changed)
        for key in ("interest_anchors", "clustered_regions", "regions_selected_for_visual_analysis",
                    "visual_events", "audio_events", "performance", "config", "final_ranking",
                    "candidates_sent_to_ai", "ai_ranking"):
            self.assertEqual(result.get(key), original.get(key), key)
        raw, deduped = result["semantic_candidates_raw"], result["semantic_candidates_after_dedupe"]
        self.assertGreaterEqual(evaluate_recall(raw, fixture["clips"])["found"], 4)
        self.assertGreaterEqual(evaluate_recall(deduped, fixture["clips"])["found"], 4)
        self.assertLessEqual(len(result["selected_candidates"]), 20)
        for c in raw:
            self.assertGreaterEqual(c["duration"], 15)
            self.assertLessEqual(c["duration"], 55)
            self.assertLessEqual(c["start"], c["central_claim"]["start"])
            self.assertGreaterEqual(c["end"], c["central_claim"]["end"])
            self.assertTrue(c["text"].endswith((".", "!")))
        _, changed_again = update_semantic_features(result, transcript)
        self.assertFalse(changed_again)

    def test_cache_hit_upgrades_without_analysis(self):
        import multimodal_analysis as analysis
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"not a real video - must not decode")
            transcript = {"video": str(source), "duration": 21, "segments": [
                {"start": 0, "end": 21, "text": "We can build a home every 60 seconds instead of waiting four hours."}]}
            transcript_path = root / "transcript.json"
            transcript_path.write_text(json.dumps(transcript))
            features = {"success": True, "analysis_version": analysis.ANALYSIS_VERSION,
                        "fingerprint": analysis._analysis_fingerprint(source, transcript_path),
                        "duration": 21, "performance": {}}
            for key in ("global_audio_events", "interest_anchors", "clustered_regions",
                        "regions_selected_for_visual_analysis", "visual_events", "selected_candidates",
                        "deduplicated_candidates"):
                features[key] = []
            (root / "analysis_features.json").write_text(json.dumps(features))
            with patch.object(analysis, "analyze_audio", side_effect=AssertionError("audio forbidden")), \
                 patch.object(analysis, "analyze_video", side_effect=AssertionError("video forbidden")):
                result = analysis.analyze_project(root)
            self.assertTrue(result["semantic_candidates_raw"])

    def test_app_adapter_without_starting_dashboard(self):
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_multimodal_candidates")
        fake_features = {"duration": 100, "selected_candidates": [{
            "start": 10, "end": 30, "text": "Complete semantic idea.",
            "candidate_source": "semantic", "semantic_interest_score": .7,
            "local_multimodal_score": 70, "region_id": "semantic_2", "region_type": "SHORT"}]}
        env = {"json": json, "build_clip_candidates": lambda _: [],
               "surrounding_context_summary": lambda *args: {}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "adapter", "exec"), env)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis_features.json").write_text(json.dumps(fake_features))
            with patch("semantic_candidates.augment_project", return_value=fake_features):
                result = env["load_multimodal_candidates"](root, [])
        self.assertEqual(result[0]["candidate_source"], "semantic")
        self.assertEqual(result[0]["score"], 70)


if __name__ == "__main__":
    unittest.main()
