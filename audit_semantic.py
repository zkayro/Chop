"""Run only transcript discovery and recall evaluation on existing artifacts.

Default is read-only. --persist adds semantic debug fields and a local merged
selection to analysis_features.json, retaining the historical GPT run.
"""
import argparse
import json
from pathlib import Path

from semantic_candidates import augment_project, evaluate_recall, update_semantic_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    metadata = json.loads((args.project / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("project_id") != fixture["project_id"] or metadata.get("source_url") != fixture["source_url"]:
        parser.error("Ground truth does not belong to this project/source")
    before = json.loads((args.project / "analysis_features.json").read_text(encoding="utf-8"))
    transcript = json.loads((args.project / "transcript.json").read_text(encoding="utf-8"))
    baseline = evaluate_recall(before["raw_candidate_variants"], fixture["clips"])
    if args.persist:
        result = augment_project(args.project, fixture["clips"])
    else:
        result, _ = update_semantic_features(before, transcript)
    raw = result["semantic_candidates_raw"]
    deduped = result["semantic_candidates_after_dedupe"]
    after = evaluate_recall(deduped, fixture["clips"])
    selected = evaluate_recall(result["selected_candidates"], fixture["clips"])
    samples = []
    for row in after["clips"]:
        match = max(row["matches"], key=lambda c: (c["overlap_ratio"], c["semantic_interest_score"] or 0), default=None)
        samples.append({"klap": row["id"], "found": row["found"], "example": match})
    output = {
        "raw_count": len(raw), "deduped_count": len(deduped),
        "baseline_recall": f"{baseline['found']}/{baseline['total']}",
        "raw_recall": evaluate_recall(raw, fixture["clips"])["found"],
        "after_dedupe_recall": f"{after['found']}/{after['total']}",
        "selected_not_sent_recall": f"{selected['found']}/{selected['total']}",
        "selected_count": len(result["selected_candidates"]),
        "source_mix": {source: sum(c["candidate_source"] == source for c in result["selected_candidates"])
                       for source in ("semantic", "multimodal", "semantic+multimodal")},
        "timings": result["semantic_pass_timings"], "examples": samples,
        "persisted": args.persist, "api_calls": 0, "media_analysis_calls": 0,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
