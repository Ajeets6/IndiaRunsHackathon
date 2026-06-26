#!/usr/bin/env python3
"""Local submission audit beyond the official format validator."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import rank as ranker


def read_submission(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_needed_candidates(path: Path, candidate_ids: set[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for candidate in ranker.iter_candidates(path):
        cid = str(candidate.get("candidate_id", ""))
        if cid in candidate_ids:
            found[cid] = candidate
            if len(found) == len(candidate_ids):
                break
    return found


def reasoning_mentions_candidate_fact(reasoning: str, candidate: dict) -> bool:
    profile = candidate.get("profile", {})
    facts = [
        str(profile.get("current_title", "")),
        str(profile.get("current_company", "")),
        str(profile.get("location", "")),
        ranker.norm_years(float(profile.get("years_of_experience") or 0.0)),
    ]
    text = reasoning.lower()
    return any(fact and fact.lower() in text for fact in facts)


def audit_reasoning_grounding(row: dict[str, str], candidate: dict) -> list[str]:
    errors: list[str] = []
    reasoning = row.get("reasoning", "")
    if not reasoning.strip():
        return ["empty reasoning"]

    if not reasoning_mentions_candidate_fact(reasoning, candidate):
        errors.append("reasoning does not mention title, company, location, or years")

    blob = ranker.text_blob(candidate)
    _, evidence_hits, _, _ = ranker.score_evidence(candidate, blob)
    evidence_hit_set = set(evidence_hits)
    for label, _, _ in ranker.EVIDENCE_RULES:
        if label.lower() in reasoning.lower() and label not in evidence_hit_set:
            errors.append(f"ungrounded evidence label: {label}")

    if "core skills include" in reasoning.lower():
        skill_names = {str(skill.get("name", "")).lower() for skill in candidate.get("skills", [])}
        for _, (canonical, _) in ranker.CORE_SKILL_WEIGHTS.items():
            if canonical.lower() in reasoning.lower() and canonical.lower() not in skill_names:
                errors.append(f"ungrounded skill mention: {canonical}")

    return errors


def audit(rows: list[dict[str, str]], candidates: dict[str, dict], expect_rows: int) -> list[str]:
    errors: list[str] = []
    if len(rows) != expect_rows:
        errors.append(f"expected {expect_rows} rows, found {len(rows)}")

    ids = [row.get("candidate_id", "").strip() for row in rows]
    ranks = [row.get("rank", "").strip() for row in rows]
    reasonings = [row.get("reasoning", "") for row in rows]

    if len(set(ids)) != len(ids):
        errors.append("duplicate candidate_id values")
    if len(set(ranks)) != len(ranks):
        errors.append("duplicate rank values")
    if len(set(reasonings)) != len(reasonings):
        errors.append("duplicate reasoning strings")

    last_score = None
    for row in rows:
        cid = row.get("candidate_id", "").strip()
        if cid not in candidates:
            errors.append(f"{cid}: candidate_id not found in candidate file")
            continue

        try:
            score = float(row.get("score", ""))
        except ValueError:
            errors.append(f"{cid}: score is not a float")
            continue

        if last_score is not None and score > last_score:
            errors.append(f"{cid}: score increases from previous row")
        last_score = score

        for issue in audit_reasoning_grounding(row, candidates[cid]):
            errors.append(f"{cid}: {issue}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Redrob submission reasoning and ordering.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.jsonl"))
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--expect-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_submission(args.submission)
    candidates = load_needed_candidates(args.candidates, {row.get("candidate_id", "").strip() for row in rows})
    errors = audit(rows, candidates, args.expect_rows)
    if errors:
        print(f"Audit failed ({len(errors)} issue(s)):")
        for error in errors[:200]:
            print(f"- {error}")
        if len(errors) > 200:
            print(f"- ... {len(errors) - 200} more")
        sys.exit(1)
    print("Audit passed.")


if __name__ == "__main__":
    main()
