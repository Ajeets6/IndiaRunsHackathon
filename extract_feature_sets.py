#!/usr/bin/env python3
"""Extract data-driven feature-set candidates from the Redrob corpus.

This script explains how constants such as PRODUCT_COMPANIES can be derived
from the dataset instead of guessed manually. It requires pandas because this
is an exploratory profiling tool, not the no-dependency ranking step.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import rank as ranker


TARGET_ROLE_MIN_SCORE = 12.0
PRODUCT_INDUSTRY_SEEDS = {
    "Software",
    "SaaS",
    "AI/ML",
    "AI Services",
    "HealthTech AI",
    "Conversational AI",
    "Fintech",
    "Food Delivery",
    "E-commerce",
    "AdTech",
    "Gaming",
    "EdTech",
    "Internet",
}
SERVICE_INDUSTRIES = {
    "IT Services",
    "Consulting",
}


def blank_company_stats() -> dict[str, Any]:
    return {
        "career_mentions": 0,
        "current_mentions": 0,
        "product_industry_mentions": 0,
        "service_industry_mentions": 0,
        "target_title_mentions": 0,
        "evidence_mentions": 0,
        "core_skill_mentions": 0,
        "title_score_sum": 0.0,
        "evidence_score_sum": 0.0,
        "current_industries": Counter(),
        "career_industries": Counter(),
    }


def blank_title_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "title_score_sum": 0.0,
        "evidence_score_sum": 0.0,
        "core_skill_sum": 0,
        "india_count": 0,
        "product_company_count": 0,
    }


def blank_industry_stats() -> dict[str, Any]:
    return {
        "career_mentions": 0,
        "current_mentions": 0,
        "target_title_mentions": 0,
        "evidence_score_sum": 0.0,
        "core_skill_sum": 0,
    }


def round_float(value: float) -> float:
    return round(float(value), 4)


def top_counter(counter: Counter, limit: int = 5) -> str:
    return "; ".join(f"{name}:{count}" for name, count in counter.most_common(limit))


def analyze_candidate(
    candidate: dict[str, Any],
    company_stats: dict[str, dict[str, Any]],
    title_stats: dict[str, dict[str, Any]],
    industry_stats: dict[str, dict[str, Any]],
    skill_counts: Counter,
) -> None:
    profile = candidate.get("profile", {})
    current_company = str(profile.get("current_company", ""))
    current_industry = str(profile.get("current_industry", ""))
    current_title = str(profile.get("current_title", ""))
    country = str(profile.get("country", ""))
    blob = ranker.text_blob(candidate)
    evidence_score, evidence_hits, _, _ = ranker.score_evidence(candidate, blob)
    _, _, core_count, _ = ranker.score_skills(candidate)
    title_score = ranker.title_family_score(current_title)
    target_title = title_score >= TARGET_ROLE_MIN_SCORE

    title_row = title_stats[current_title]
    title_row["count"] += 1
    title_row["title_score_sum"] += title_score
    title_row["evidence_score_sum"] += evidence_score
    title_row["core_skill_sum"] += core_count
    title_row["india_count"] += int(country == "India")

    for skill in candidate.get("skills", []):
        name = str(skill.get("name", "")).strip()
        if name:
            skill_counts[name] += 1

    if current_company:
        company_row = company_stats[current_company]
        company_row["current_mentions"] += 1
        company_row["current_industries"][current_industry] += 1
        title_stats[current_title]["product_company_count"] += int(current_company in ranker.PRODUCT_COMPANIES)

    if current_industry:
        industry_row = industry_stats[current_industry]
        industry_row["current_mentions"] += 1
        industry_row["target_title_mentions"] += int(target_title)
        industry_row["evidence_score_sum"] += evidence_score
        industry_row["core_skill_sum"] += core_count

    for job in candidate.get("career_history", []):
        company = str(job.get("company", "")).strip()
        industry = str(job.get("industry", "")).strip()
        job_title = str(job.get("title", "")).strip()
        job_title_score = ranker.title_family_score(job_title)
        is_target_job = job_title_score >= TARGET_ROLE_MIN_SCORE

        if company:
            company_row = company_stats[company]
            company_row["career_mentions"] += 1
            company_row["product_industry_mentions"] += int(industry in PRODUCT_INDUSTRY_SEEDS)
            company_row["service_industry_mentions"] += int(industry in SERVICE_INDUSTRIES)
            company_row["target_title_mentions"] += int(is_target_job)
            company_row["evidence_mentions"] += int(bool(evidence_hits))
            company_row["core_skill_mentions"] += core_count
            company_row["title_score_sum"] += job_title_score
            company_row["evidence_score_sum"] += evidence_score
            company_row["career_industries"][industry] += 1

        if industry:
            industry_row = industry_stats[industry]
            industry_row["career_mentions"] += 1
            industry_row["target_title_mentions"] += int(is_target_job)
            industry_row["evidence_score_sum"] += evidence_score
            industry_row["core_skill_sum"] += core_count


def finalize_company_rows(company_stats: dict[str, dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for company, stats in company_stats.items():
        career_mentions = int(stats["career_mentions"])
        if career_mentions == 0:
            continue

        product_ratio = stats["product_industry_mentions"] / career_mentions
        service_ratio = stats["service_industry_mentions"] / career_mentions
        target_title_ratio = stats["target_title_mentions"] / career_mentions
        evidence_ratio = stats["evidence_mentions"] / career_mentions
        avg_title_score = stats["title_score_sum"] / career_mentions
        avg_evidence_score = stats["evidence_score_sum"] / career_mentions
        avg_core_skills = stats["core_skill_mentions"] / career_mentions
        product_likelihood_score = (
            45.0 * product_ratio
            + 25.0 * target_title_ratio
            + 20.0 * evidence_ratio
            + 1.2 * avg_title_score
            + 0.8 * avg_core_skills
            - 35.0 * service_ratio
        )
        service_likelihood_score = 70.0 * service_ratio + 20.0 * (1.0 - product_ratio)

        suggested_product = (
            career_mentions >= min_count
            and service_ratio < 0.55
            and (
                product_ratio >= 0.45
                or product_likelihood_score >= 34.0
                or (target_title_ratio >= 0.08 and evidence_ratio >= 0.12)
            )
        )
        suggested_service = career_mentions >= min_count and service_ratio >= 0.60 and product_ratio <= 0.20

        rows.append(
            {
                "company": company,
                "career_mentions": career_mentions,
                "current_mentions": int(stats["current_mentions"]),
                "product_industry_ratio": round_float(product_ratio),
                "service_industry_ratio": round_float(service_ratio),
                "target_title_ratio": round_float(target_title_ratio),
                "evidence_ratio": round_float(evidence_ratio),
                "avg_title_score": round_float(avg_title_score),
                "avg_evidence_score": round_float(avg_evidence_score),
                "avg_core_skills": round_float(avg_core_skills),
                "product_likelihood_score": round_float(product_likelihood_score),
                "service_likelihood_score": round_float(service_likelihood_score),
                "suggested_product_company": suggested_product,
                "suggested_service_company": suggested_service,
                "top_career_industries": top_counter(stats["career_industries"]),
                "top_current_industries": top_counter(stats["current_industries"]),
            }
        )
    return sorted(rows, key=lambda row: (-row["product_likelihood_score"], -row["career_mentions"], row["company"]))


def finalize_title_rows(title_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for title, stats in title_stats.items():
        count = int(stats["count"])
        rows.append(
            {
                "title": title,
                "count": count,
                "avg_title_score": round_float(stats["title_score_sum"] / count),
                "avg_evidence_score": round_float(stats["evidence_score_sum"] / count),
                "avg_core_skills": round_float(stats["core_skill_sum"] / count),
                "india_ratio": round_float(stats["india_count"] / count),
                "known_product_company_ratio": round_float(stats["product_company_count"] / count),
                "suggested_target_title": (stats["title_score_sum"] / count) >= TARGET_ROLE_MIN_SCORE,
            }
        )
    return sorted(rows, key=lambda row: (-row["avg_title_score"], -row["avg_evidence_score"], row["title"]))


def finalize_industry_rows(industry_stats: dict[str, dict[str, Any]], min_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for industry, stats in industry_stats.items():
        career_mentions = int(stats["career_mentions"])
        total_mentions = career_mentions + int(stats["current_mentions"])
        if total_mentions == 0:
            continue
        target_title_ratio = stats["target_title_mentions"] / max(career_mentions, 1)
        avg_evidence_score = stats["evidence_score_sum"] / max(total_mentions, 1)
        avg_core_skills = stats["core_skill_sum"] / max(total_mentions, 1)
        product_likelihood_score = 50.0 * target_title_ratio + 1.5 * avg_evidence_score + 0.8 * avg_core_skills
        suggested_product = (
            total_mentions >= min_count
            and (
                industry in PRODUCT_INDUSTRY_SEEDS
                or product_likelihood_score >= 16.0
            )
            and industry not in SERVICE_INDUSTRIES
        )
        rows.append(
            {
                "industry": industry,
                "career_mentions": career_mentions,
                "current_mentions": int(stats["current_mentions"]),
                "target_title_ratio": round_float(target_title_ratio),
                "avg_evidence_score": round_float(avg_evidence_score),
                "avg_core_skills": round_float(avg_core_skills),
                "product_likelihood_score": round_float(product_likelihood_score),
                "suggested_product_industry": suggested_product,
            }
        )
    return sorted(rows, key=lambda row: (-row["product_likelihood_score"], -row["career_mentions"], row["industry"]))


def finalize_skill_rows(skill_counts: Counter) -> list[dict[str, Any]]:
    rows = []
    core_keys = set(ranker.CORE_SKILL_WEIGHTS)
    for skill, count in skill_counts.items():
        canonical = ranker.CORE_SKILL_WEIGHTS.get(skill.lower(), ("", 0.0))
        rows.append(
            {
                "skill": skill,
                "count": count,
                "is_core_ranker_skill": skill.lower() in core_keys,
                "core_weight": canonical[1],
            }
        )
    return sorted(rows, key=lambda row: (-row["is_core_ranker_skill"], -row["core_weight"], -row["count"], row["skill"]))


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def py_set(name: str, values: list[str]) -> str:
    lines = [f"{name} = {{"]
    for value in values:
        lines.append(f"    {value!r},")
    lines.append("}")
    return "\n".join(lines)


def write_suggested_sets(path: Path, company_rows: list[dict[str, Any]], industry_rows: list[dict[str, Any]]) -> None:
    product_companies = sorted(row["company"] for row in company_rows if row["suggested_product_company"])
    service_companies = sorted(row["company"] for row in company_rows if row["suggested_service_company"])
    product_industries = sorted(row["industry"] for row in industry_rows if row["suggested_product_industry"])
    text = "\n\n".join(
        [
            "# Auto-generated by extract_feature_sets.py. Review before copying into rank.py.",
            py_set("SUGGESTED_PRODUCT_COMPANIES", product_companies),
            py_set("SUGGESTED_SERVICES_COMPANIES", service_companies),
            py_set("SUGGESTED_PRODUCT_INDUSTRIES", product_industries),
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract feature-set candidates from Redrob candidates.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("feature_profiles"))
    parser.add_argument("--limit", type=int, default=None, help="Optional candidate limit for quick profiling.")
    parser.add_argument("--min-company-count", type=int, default=10)
    parser.add_argument("--min-industry-count", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    company_stats: dict[str, dict[str, Any]] = defaultdict(blank_company_stats)
    title_stats: dict[str, dict[str, Any]] = defaultdict(blank_title_stats)
    industry_stats: dict[str, dict[str, Any]] = defaultdict(blank_industry_stats)
    skill_counts: Counter = Counter()

    seen = 0
    for candidate in ranker.iter_candidates(args.candidates, args.limit):
        analyze_candidate(candidate, company_stats, title_stats, industry_stats, skill_counts)
        seen += 1

    company_rows = finalize_company_rows(company_stats, args.min_company_count)
    title_rows = finalize_title_rows(title_stats)
    industry_rows = finalize_industry_rows(industry_stats, args.min_industry_count)
    skill_rows = finalize_skill_rows(skill_counts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(args.out_dir / "company_profile.csv", company_rows)
    write_table(args.out_dir / "title_profile.csv", title_rows)
    write_table(args.out_dir / "industry_profile.csv", industry_rows)
    write_table(args.out_dir / "skill_profile.csv", skill_rows)
    write_suggested_sets(args.out_dir / "suggested_feature_sets.py", company_rows, industry_rows)

    summary = {
        "candidates_seen": seen,
        "used_pandas": True,
        "company_rows": len(company_rows),
        "title_rows": len(title_rows),
        "industry_rows": len(industry_rows),
        "skill_rows": len(skill_rows),
        "suggested_product_companies": sum(row["suggested_product_company"] for row in company_rows),
        "suggested_service_companies": sum(row["suggested_service_company"] for row in company_rows),
        "suggested_product_industries": sum(row["suggested_product_industry"] for row in industry_rows),
    }
    (args.out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote feature profiles for {seen} candidates to {args.out_dir} using pandas.")


if __name__ == "__main__":
    main()
