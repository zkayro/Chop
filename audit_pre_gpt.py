"""Replay the original source alternation and audit selection, never detection.

The legacy policy exists only here for reproducible before/after diagnostics.
Ground truth is passed exclusively to evaluation, never to production selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from semantic_candidates import _duplicate, _overlap, evaluate_recall, refresh_pre_gpt_selection


def identity(c):
    return c.get("semantic_candidate_id") or f"multimodal_{c.get('region_id')}_{c.get('region_type')}_{c['start']}_{c['end']}"


def compact(c):
    return {"candidate_id": identity(c), "start": c["start"], "end": c["end"],
            "candidate_source": c["candidate_source"],
            "semantic_interest_score": c.get("semantic_interest_score"),
            "local_score": c.get("local_multimodal_score"), "region_id": c.get("region_id")}


def replay_alternation(merged, maximum=20):
    semantic = sorted((c for c in merged if c["candidate_source"] != "multimodal"),
                      key=lambda c: (-c["semantic_interest_score"], c["start"]))
    multimodal = sorted((c for c in merged if c["candidate_source"] != "semantic"),
                        key=lambda c: -c.get("local_multimodal_score", 0))
    records = {identity(c): {**compact(c), "semantic_rank": None, "multimodal_rank": None,
                             "selected": False, "events": []} for c in merged}
    for name, queue in (("semantic", semantic), ("multimodal", multimodal)):
        for rank, c in enumerate(queue, 1):
            records[identity(c)][f"{name}_rank"] = rank
    selected, used = [], set()
    for diverse in (True, False):
        for rank in range(max(len(semantic), len(multimodal), 0)):
            for name, queue in (("semantic", semantic), ("multimodal", multimodal)):
                if rank >= len(queue):
                    continue
                c = queue[rank]
                key = (c["start"], c["end"], c.get("region_id"), c.get("region_type"))
                if key in used:
                    continue
                blockers = [k for k in selected if (
                    (_overlap(c, k) >= .50 or c.get("region_id") == k.get("region_id"))
                    if diverse else _duplicate(c, k))]
                if blockers:
                    records[identity(c)]["events"].append({
                        "reason": "temporal_or_region_diversity" if diverse else "duplicate",
                        "source_turn": name, "queue_rank": rank + 1,
                        "preferred_candidates": [dict(compact(k), overlap=_overlap(c, k),
                                                      same_region=c.get("region_id") == k.get("region_id")) for k in blockers]})
                    continue
                selected.append(c)
                used.add(key)
                records[identity(c)].update(selected=True, selected_position=len(selected), source_turn=name)
                if len(selected) == maximum:
                    for record in records.values():
                        if not record["selected"]:
                            record["reason"] = (record["events"][-1]["reason"] if record["events"] else "budget_exhausted_before_queue_rank")
                            record["budget_cutoff"] = compact(c)
                            record["budget_cutoff_source_turn"] = name
                    return selected, records
    return selected, records


def audit(features, annotations):
    selected, records = replay_alternation(features["merged_candidates_after_dedupe"])
    gt = evaluate_recall(features["semantic_candidates_after_dedupe"], annotations)
    cohorts = []
    for truth in gt["clips"]:
        matches = []
        for match in truth["matches"]:
            record = records.get(match["candidate_id"])
            if record is None:
                original = next(c for c in features["semantic_candidates_after_dedupe"] if identity(c) == match["candidate_id"])
                representative = next((c for c in features["merged_candidates_after_dedupe"] if _duplicate(c, original)), None)
                record = {"selected": False, "reason": "merged_duplicate", "representative": compact(representative) if representative else None}
                if representative:
                    record["representative_selection"] = records[identity(representative)]
            matches.append({**match, "selection": record})
        best = max(matches, key=lambda m: (m["selection"].get("selected", False), m["semantic_interest_score"] or 0), default=None)
        cohorts.append({"klap": truth["id"], "representative": best, "all_matches": matches})
    return {"policy": "legacy_source_alternation", "input_digest": hashlib.sha256(
        json.dumps(features["semantic_candidates_after_dedupe"], sort_keys=True).encode()).hexdigest(),
        "raw_count": len(features["semantic_candidates_raw"]),
        "deduped_count": len(features["semantic_candidates_after_dedupe"]),
        "selected": [compact(c) for c in selected], "decisions": records,
        "recall": evaluate_recall(selected, annotations), "ground_truth_candidates": cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--save-before", action="store_true")
    parser.add_argument("--apply-selection", action="store_true")
    args = parser.parse_args()
    features = json.loads((args.project / "analysis_features.json").read_text(encoding="utf-8"))
    fixture = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    metadata = json.loads((args.project / "metadata.json").read_text(encoding="utf-8"))
    if metadata["project_id"] != fixture["project_id"] or metadata["source_url"] != fixture["source_url"]:
        parser.error("Wrong project for these annotations")
    result = audit(features, fixture["clips"])
    if args.save_before and args.apply_selection:
        parser.error("Save the baseline before changing selection")
    if args.save_before:
        assert result["selected"] == [compact(c) for c in features["selected_candidates"]], "Baseline replay must match stored selection exactly"
        path = args.project / "pre_gpt_selection_audit_before.json"
        # Preserve the original baseline on repeat runs.
        if path.exists():
            parser.error("Before audit already exists; refusing to replace the baseline")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.apply_selection:
        before_path = args.project / "pre_gpt_selection_audit_before.json"
        if not before_path.exists():
            parser.error("Save the before audit first")
        baseline = json.loads(before_path.read_text(encoding="utf-8"))
        if baseline["input_digest"] != result["input_digest"]:
            parser.error("Semantic candidate pool changed since baseline audit")
        started = time.perf_counter()
        refresh_pre_gpt_selection(features)
        elapsed = time.perf_counter() - started
        recall = evaluate_recall(features["selected_candidates"], fixture["clips"])
        features["semantic_ground_truth_recall"]["selected_for_gpt_not_sent"] = recall
        selected_ids = {identity(c) for c in features["selected_candidates"]}
        report = {"before_recall": baseline["recall"]["found"], "after_recall": recall["found"],
                  "input_digest": result["input_digest"], "selection_seconds": elapsed,
                  "source_mix": {s: sum(c["candidate_source"] == s for c in features["selected_candidates"])
                                 for s in ("semantic", "multimodal", "semantic+multimodal")},
                  "selected": [compact(c) for c in features["selected_candidates"]],
                  "recall": recall, "decisions": features["pre_gpt_selection_debug"]}
        for name, content in (("analysis_features.json", features), ("pre_gpt_selection_audit_after.json", report)):
            path = args.project / name
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        print(json.dumps({"before_recall": report["before_recall"], "after_recall": report["after_recall"],
                          "selection_seconds": elapsed, "source_mix": report["source_mix"],
                          "selected": report["selected"],
                          "ground_truth": [{"klap": r["id"], "found": r["found"], "matches": r["matches"]} for r in recall["clips"]]}, indent=2))
        return
    print(json.dumps({"recall": result["recall"]["found"], "total": result["recall"]["total"],
                      "candidates": [{"klap": r["klap"], "representative": r["representative"]} for r in result["ground_truth_candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
