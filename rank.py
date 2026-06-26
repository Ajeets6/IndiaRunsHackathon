#!/usr/bin/env python3
"""Offline Redrob candidate ranker.

The ranking step is intentionally stdlib-only: no network calls, no hosted LLMs,
and no local model imports. Optional dense similarities can be precomputed
offline and placed at artifacts/dense_similarity.csv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable


REFERENCE_DATE = date(2026, 6, 25)


TARGET_CITIES = {
    "pune",
    "noida",
}

ACCEPTABLE_CITIES = {
    "delhi ncr",
    "delhi",
    "gurgaon",
    "gurugram",
    "hyderabad",
    "mumbai",
    "bangalore",
    "bengaluru",
}

PRODUCT_COMPANIES = {
    "Swiggy",
    "Zomato",
    "Razorpay",
    "CRED",
    "Flipkart",
    "Meesho",
    "InMobi",
    "Nykaa",
    "Zoho",
    "Freshworks",
    "Ola",
    "Paytm",
    "PhonePe",
    "Dream11",
    "PharmEasy",
    "PolicyBazaar",
    "Unacademy",
    "Vedantu",
    "upGrad",
    "Sarvam AI",
    "Krutrim",
    "Haptik",
    "Observe.AI",
    "Rephrase.ai",
    "Mad Street Den",
    "Niramai",
    "Saarthi.ai",
    "Glance",
    "Google",
    "Amazon",
    "Meta",
    "Microsoft",
    "Netflix",
    "Yellow.ai",
    "Aganitha",
}

PRODUCT_INDUSTRIES = {
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
    "Marketplace",
    "Gaming",
    "EdTech",
    "Internet",
}

SERVICES_COMPANIES = {
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
    "HCL",
    "Tech Mahindra",
    "Mphasis",
    "Mindtree",
}

NON_TECH_TITLES = {
    "Marketing Manager",
    "HR Manager",
    "Accountant",
    "Sales Executive",
    "Content Writer",
    "Graphic Designer",
    "Civil Engineer",
    "Mechanical Engineer",
    "Customer Support",
    "Operations Manager",
    "Business Analyst",
    "Project Manager",
}

ROLE_WEIGHTS = {
    "Senior AI Engineer": 23.0,
    "Lead AI Engineer": 22.0,
    "Staff Machine Learning Engineer": 22.0,
    "Senior Machine Learning Engineer": 21.0,
    "Senior Applied Scientist": 20.0,
    "Senior NLP Engineer": 20.0,
    "Applied ML Engineer": 19.0,
    "Search Engineer": 19.0,
    "Recommendation Systems Engineer": 19.0,
    "Machine Learning Engineer": 18.0,
    "AI Engineer": 18.0,
    "NLP Engineer": 17.0,
    "Senior Software Engineer (ML)": 16.0,
    "ML Engineer": 15.0,
    "Senior Data Scientist": 14.0,
    "Data Scientist": 12.0,
    "AI Specialist": 10.0,
    "Senior Data Engineer": 9.0,
    "Senior Software Engineer": 8.0,
    "Data Engineer": 7.0,
    "Backend Engineer": 7.0,
    "Analytics Engineer": 6.5,
    "Software Engineer": 6.0,
    "Cloud Engineer": 4.0,
    "DevOps Engineer": 4.0,
    "Data Analyst": 3.0,
}

CORE_SKILL_WEIGHTS = {
    "information retrieval": ("Information Retrieval", 4.0),
    "semantic search": ("Semantic Search", 4.0),
    "vector search": ("Vector Search", 4.0),
    "embeddings": ("Embeddings", 3.8),
    "sentence transformers": ("Sentence Transformers", 3.6),
    "recommendation systems": ("Recommendation Systems", 3.8),
    "learning to rank": ("Learning to Rank", 4.0),
    "ranking": ("Ranking", 3.5),
    "bm25": ("BM25", 3.3),
    "faiss": ("FAISS", 3.4),
    "pinecone": ("Pinecone", 3.0),
    "qdrant": ("Qdrant", 3.0),
    "milvus": ("Milvus", 3.0),
    "weaviate": ("Weaviate", 3.0),
    "opensearch": ("OpenSearch", 2.8),
    "elasticsearch": ("Elasticsearch", 2.8),
    "llms": ("LLMs", 2.4),
    "rag": ("RAG", 2.7),
    "fine-tuning llms": ("Fine-tuning LLMs", 2.7),
    "lora": ("LoRA", 2.0),
    "qlora": ("QLoRA", 2.0),
    "peft": ("PEFT", 2.0),
    "hugging face transformers": ("Hugging Face Transformers", 2.2),
    "nlp": ("NLP", 2.4),
    "machine learning": ("Machine Learning", 2.3),
    "python": ("Python", 2.5),
    "feature engineering": ("Feature Engineering", 1.6),
    "data pipelines": ("Data Pipelines", 1.3),
    "spark": ("Spark", 1.1),
    "airflow": ("Airflow", 1.0),
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.0,
    "intermediate": 0.35,
    "advanced": 0.75,
    "expert": 1.0,
}

EVIDENCE_RULES = [
    ("production ranking system", ["ranking pipeline", "ranking system", "ranker", "learning-to-rank", "learning to rank"], 4.8),
    ("hybrid BM25+dense retrieval", ["hybrid retrieval", "bm25 + dense", "bm25 with dense", "sparse and dense", "bm25-only retrieval"], 4.6),
    ("semantic/vector search", ["semantic search", "vector search", "dense retrieval", "dense vector", "embedding-based search"], 4.2),
    ("recommendation systems", ["recommendation system", "recommender", "collaborative filtering", "matrix factorization"], 3.8),
    ("retrieval infrastructure", ["faiss", "hnsw", "pinecone", "qdrant", "milvus", "weaviate", "opensearch", "elasticsearch"], 3.8),
    ("LLM reranking or RAG", ["llm-based re-ranker", "llm rerank", "re-ranker", "reranking", "rag-based", "rag"], 3.5),
    ("embedding operations", ["embedding drift", "embedding versioning", "index refresh", "index versioning", "rollback"], 3.3),
    ("ranking evaluation", ["ndcg", "mrr", "map", "recall@k", "offline evaluation", "online/offline", "held-out eval"], 4.0),
    ("online A/B testing", ["a/b", "ab test", "online experiment"], 2.7),
    ("production deployment", ["production", "deployed", "serving", "p95", "latency", "real users"], 2.4),
    ("fine-tuning", ["fine-tuned", "fine tuning", "fine-tuning", "lora", "qlora", "peft"], 2.2),
    ("recruiter/search domain", ["candidate-jd", "candidate jd", "recruiter-facing", "recruiter search", "time-to-shortlist"], 3.2),
]

EVIDENCE_TOKEN_PATTERNS = {
    pattern
    for _, patterns, _ in EVIDENCE_RULES
    for pattern in patterns
    if re.fullmatch(r"[a-z0-9]+", pattern)
}

VISION_SPEECH_SKILLS = {
    "computer vision",
    "image classification",
    "object detection",
    "yolo",
    "gans",
    "diffusion models",
    "speech recognition",
    "tts",
    "robotics",
}


@dataclass
class CandidateScore:
    candidate_id: str
    stage1_score: float
    final_score: float
    display_score: float = 0.0
    rank: int = 0
    reasoning: str = ""
    components: dict[str, float] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def open_candidate_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def days_since(value: str | None) -> int | None:
    parsed = parse_date(value)
    if parsed is None:
        return None
    return (REFERENCE_DATE - parsed).days


def norm_years(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def serial_join(items: list[str], max_items: int = 4) -> str:
    shown = [item for item in items if item][:max_items]
    if not shown:
        return ""
    if len(shown) == 1:
        return shown[0]
    if len(shown) == 2:
        return f"{shown[0]} and {shown[1]}"
    return f"{', '.join(shown[:-1])}, and {shown[-1]}"


def text_blob(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.extend(
            [
                job.get("title", ""),
                job.get("company", ""),
                job.get("industry", ""),
                job.get("description", ""),
            ]
        )
    return " ".join(str(part) for part in parts if part).lower()


def assessment_lookup(signals: dict[str, Any]) -> dict[str, float]:
    raw = signals.get("skill_assessment_scores") or {}
    return {str(k).lower(): float(v) for k, v in raw.items() if isinstance(v, (int, float))}


def score_skills(candidate: dict[str, Any]) -> tuple[float, list[str], int, int]:
    signals = candidate.get("redrob_signals", {})
    assessments = assessment_lookup(signals)
    skill_score = 0.0
    core_skills: list[str] = []
    expert_zero_duration = 0

    for skill in candidate.get("skills", []):
        name = str(skill.get("name", "")).strip()
        key = name.lower()
        duration = int(skill.get("duration_months") or 0)
        proficiency = str(skill.get("proficiency", "")).lower()
        if proficiency == "expert" and duration == 0:
            expert_zero_duration += 1

        if key not in CORE_SKILL_WEIGHTS:
            continue

        canonical, base_weight = CORE_SKILL_WEIGHTS[key]
        if canonical not in core_skills:
            core_skills.append(canonical)

        prof_factor = PROFICIENCY_WEIGHT.get(proficiency, 0.0)
        duration_factor = clamp(duration / 60.0, 0.0, 1.0)
        endorsement_factor = clamp(float(skill.get("endorsements") or 0) / 60.0, 0.0, 1.0)
        assessment_factor = clamp(assessments.get(key, 0.0) / 100.0, 0.0, 1.0)
        trust = 0.62 + 0.16 * prof_factor + 0.12 * duration_factor + 0.06 * endorsement_factor + 0.04 * assessment_factor
        skill_score += base_weight * trust

    core_skills.sort()
    return min(skill_score, 34.0), core_skills, len(core_skills), expert_zero_duration


def title_family_score(title: str) -> float:
    if title in ROLE_WEIGHTS:
        return ROLE_WEIGHTS[title]
    lower = title.lower()
    score = 0.0
    if any(token in lower for token in ("ai", "machine learning", " ml", "nlp", "search", "recommendation")):
        score += 8.0
    if any(token in lower for token in ("engineer", "scientist")):
        score += 4.0
    if any(token in lower for token in ("senior", "lead", "staff", "principal")):
        score += 3.0
    if title in NON_TECH_TITLES:
        score -= 12.0
    return score


def score_role(candidate: dict[str, Any]) -> tuple[float, str, list[str]]:
    profile = candidate.get("profile", {})
    current_title = str(profile.get("current_title", ""))
    career_titles = [str(job.get("title", "")) for job in candidate.get("career_history", [])]
    current = title_family_score(current_title)
    past = max([title_family_score(title) for title in career_titles] + [0.0])
    score = current + min(past * 0.35, 8.0)

    lower_title = current_title.lower()
    if any(word in lower_title for word in ("senior", "lead", "staff")) and any(
        word in lower_title for word in ("ai", "machine learning", "ml", "nlp", "search", "recommendation")
    ):
        score += 2.5
    return clamp(score, -14.0, 31.0), current_title, career_titles


def score_experience(years: float) -> float:
    peak = 14.0 * math.exp(-((years - 7.0) / 2.0) ** 2)
    if 5.0 <= years <= 9.0:
        peak += 3.0
    if years < 4.0:
        peak -= 6.0
    elif years > 12.0:
        peak -= min(8.0, (years - 12.0) * 1.4)
    return peak


def score_location(candidate: dict[str, Any]) -> tuple[float, bool, bool]:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    country = str(profile.get("country", ""))
    location = str(profile.get("location", "")).lower()
    willing = bool(signals.get("willing_to_relocate"))
    target_city = any(city in location for city in TARGET_CITIES)
    acceptable_city = any(city in location for city in ACCEPTABLE_CITIES)

    score = 0.0
    if country == "India":
        score += 8.0
        if target_city:
            score += 5.0
        elif acceptable_city:
            score += 3.0
        elif willing:
            score += 1.5
    else:
        score -= 9.0
        if willing:
            score += 3.0
    return score, target_city, acceptable_city


def score_company(candidate: dict[str, Any]) -> tuple[float, bool, bool, int]:
    profile = candidate.get("profile", {})
    current_company = str(profile.get("current_company", ""))
    current_industry = str(profile.get("current_industry", ""))
    jobs = candidate.get("career_history", [])
    companies = [str(job.get("company", "")) for job in jobs]
    industries = [str(job.get("industry", "")) for job in jobs]

    product_hits = 0
    product_hits += sum(1 for company in companies if company in PRODUCT_COMPANIES)
    product_hits += sum(1 for industry in industries if industry in PRODUCT_INDUSTRIES)
    current_product = current_company in PRODUCT_COMPANIES or current_industry in PRODUCT_INDUSTRIES

    service_count = sum(1 for company in companies if company in SERVICES_COMPANIES)
    services_only = bool(jobs) and service_count == len(jobs) and product_hits == 0

    score = min(product_hits * 2.2, 13.0)
    if current_product:
        score += 4.0
    if current_company in SERVICES_COMPANIES:
        score -= 3.5
    if services_only:
        score -= 9.0
    return score, current_product, services_only, product_hits


def score_evidence(candidate: dict[str, Any], blob: str) -> tuple[float, list[str], bool, bool]:
    hits: list[str] = []
    score = 0.0
    tokenized_blob = " " + re.sub(r"[^a-z0-9]+", " ", blob) + " "

    for label, patterns, weight in EVIDENCE_RULES:
        if any(evidence_pattern_matches(blob, tokenized_blob, pattern) for pattern in patterns):
            hits.append(label)
            score += weight

    has_production = any(label in hits for label in ("production deployment", "production ranking system", "online A/B testing"))
    has_ir = any(
        label in hits
        for label in (
            "production ranking system",
            "hybrid BM25+dense retrieval",
            "semantic/vector search",
            "retrieval infrastructure",
            "ranking evaluation",
        )
    )
    return min(score, 40.0), hits, has_production, has_ir


def evidence_pattern_matches(blob: str, tokenized_blob: str, pattern: str) -> bool:
    if pattern in EVIDENCE_TOKEN_PATTERNS:
        return f" {pattern} " in tokenized_blob
    return pattern in blob


def score_behavior(candidate: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    signals = candidate.get("redrob_signals", {})
    behavior: dict[str, Any] = {}
    flags: list[str] = []
    score = 0.0

    open_to_work = bool(signals.get("open_to_work_flag"))
    if open_to_work:
        score += 7.0
    else:
        score -= 3.0

    last_active = str(signals.get("last_active_date", ""))
    inactive_days = days_since(last_active)
    behavior["inactive_days"] = inactive_days
    if inactive_days is None:
        flags.append("missing_last_active")
        score -= 1.0
    elif inactive_days < 0:
        flags.append("future_last_active")
        score -= 10.0
    elif inactive_days <= 14:
        score += 7.0
    elif inactive_days <= 30:
        score += 5.0
    elif inactive_days <= 60:
        score += 3.0
    elif inactive_days <= 120:
        score += 1.0
    elif inactive_days > 180:
        flags.append("long_inactive")
        score -= 4.0

    response_rate = float(signals.get("recruiter_response_rate") or 0.0)
    score += 5.0 * clamp(response_rate, 0.0, 1.0)
    if response_rate < 0.2:
        flags.append("low_recruiter_response")

    response_hours = float(signals.get("avg_response_time_hours") or 0.0)
    if response_hours <= 24:
        score += 3.0
    elif response_hours <= 72:
        score += 1.5
    elif response_hours > 168:
        score -= 2.5

    notice = int(signals.get("notice_period_days") or 0)
    if notice <= 30:
        score += 4.0
    elif notice <= 60:
        score += 1.0
    elif notice > 120:
        flags.append("very_long_notice")
        score -= 6.0
    elif notice > 90:
        flags.append("long_notice")
        score -= 4.0

    interview_completion = float(signals.get("interview_completion_rate") or 0.0)
    score += 3.0 * clamp(interview_completion, 0.0, 1.0)

    offer_acceptance = float(signals.get("offer_acceptance_rate", -1.0))
    if offer_acceptance >= 0:
        score += 2.0 * clamp(offer_acceptance, 0.0, 1.0)

    if bool(signals.get("willing_to_relocate")):
        score += 2.0

    github = float(signals.get("github_activity_score", -1.0))
    if github >= 70:
        score += 3.0
    elif github >= 35:
        score += 1.5
    elif github == -1:
        score -= 0.5

    score += 0.6 * sum(
        bool(signals.get(field))
        for field in ("verified_email", "verified_phone", "linkedin_connected")
    )
    score += min(float(signals.get("saved_by_recruiters_30d") or 0) / 8.0, 2.0)
    score += min(float(signals.get("profile_views_received_30d") or 0) / 150.0, 1.5)

    behavior.update(
        {
            "open_to_work": open_to_work,
            "last_active_date": last_active,
            "response_rate": response_rate,
            "response_hours": response_hours,
            "notice_days": notice,
            "willing_to_relocate": bool(signals.get("willing_to_relocate")),
            "github_activity_score": github,
        }
    )
    return clamp(score, -14.0, 33.0), behavior, flags


def score_education(candidate: dict[str, Any]) -> float:
    score = 0.0
    for edu in candidate.get("education", []):
        tier = str(edu.get("tier", ""))
        field_name = str(edu.get("field_of_study", "")).lower()
        if tier == "tier_1":
            score += 1.5
        elif tier == "tier_2":
            score += 0.6
        if any(token in field_name for token in ("computer", "data", "artificial intelligence", "machine learning")):
            score += 0.5
    return min(score, 3.0)


def duration_mismatch_years(candidate: dict[str, Any], years: float) -> float:
    career_months = sum(int(job.get("duration_months") or 0) for job in candidate.get("career_history", []))
    if career_months == 0:
        return 0.0
    return abs(career_months / 12.0 - years)


def score_penalties(
    candidate: dict[str, Any],
    facts: dict[str, Any],
    core_count: int,
    evidence_hits: list[str],
    has_production: bool,
    has_ir: bool,
    expert_zero_duration: int,
    services_only: bool,
) -> tuple[float, list[str]]:
    profile = candidate.get("profile", {})
    title = str(profile.get("current_title", ""))
    years = float(profile.get("years_of_experience") or 0.0)
    skill_names = {str(skill.get("name", "")).lower() for skill in candidate.get("skills", [])}
    flags: list[str] = []
    penalty = 0.0

    if expert_zero_duration:
        flags.append("expert_skill_zero_duration")
        penalty += min(18.0, expert_zero_duration * 7.0)

    mismatch = duration_mismatch_years(candidate, years)
    facts["experience_duration_mismatch_years"] = round(mismatch, 2)
    if mismatch > 5.0:
        flags.append("experience_duration_mismatch_gt5")
        penalty += 11.0
    elif mismatch > 3.0:
        flags.append("experience_duration_mismatch_gt3")
        penalty += 6.0

    if title in NON_TECH_TITLES and core_count >= 6:
        flags.append("non_tech_keyword_stuffer")
        penalty += 22.0

    if core_count >= 9 and len(evidence_hits) <= 3 and title not in ROLE_WEIGHTS:
        flags.append("keyword_stuffer_low_career_evidence")
        penalty += 16.0

    lower_title = title.lower()
    if "research" in lower_title and not has_production:
        flags.append("research_without_production")
        penalty += 10.0

    vision_speech_count = len(skill_names & VISION_SPEECH_SKILLS)
    if ("computer vision" in lower_title or vision_speech_count >= 4) and not has_ir:
        flags.append("vision_speech_without_ir")
        penalty += 9.0

    if services_only and not has_production:
        flags.append("services_only_without_production")
        penalty += 5.0

    jobs = candidate.get("career_history", [])
    if len(jobs) >= 5:
        avg_duration = sum(int(job.get("duration_months") or 0) for job in jobs) / len(jobs)
        if avg_duration < 18:
            flags.append("short_tenure_hopping")
            penalty += 4.0

    education_bad_dates = any(
        int(edu.get("end_year") or 0) < int(edu.get("start_year") or 0)
        for edu in candidate.get("education", [])
    )
    if education_bad_dates:
        flags.append("education_date_inconsistency")
        penalty += 6.0

    return penalty, flags


def load_dense_scores(artifacts_dir: Path | None) -> dict[str, float]:
    if artifacts_dir is None:
        return {}
    path = artifacts_dir / "dense_similarity.csv"
    if not path.exists():
        return {}

    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "candidate_id" not in reader.fieldnames or "similarity" not in reader.fieldnames:
            raise ValueError(f"{path} must have candidate_id,similarity columns")
        for row in reader:
            cid = str(row.get("candidate_id", "")).strip()
            if not cid:
                continue
            try:
                scores[cid] = float(row.get("similarity", 0.0))
            except ValueError:
                continue
    return scores


def score_candidate(candidate: dict[str, Any], dense_similarity: float = 0.0) -> CandidateScore:
    profile = candidate.get("profile", {})
    cid = str(candidate.get("candidate_id", ""))
    years = float(profile.get("years_of_experience") or 0.0)
    blob = text_blob(candidate)

    skill_score, core_skills, core_count, expert_zero_duration = score_skills(candidate)
    role_score, current_title, career_titles = score_role(candidate)
    experience_score = score_experience(years)
    location_score, target_city, acceptable_city = score_location(candidate)
    company_score, current_product, services_only, product_hits = score_company(candidate)
    evidence_score, evidence_hits, has_production, has_ir = score_evidence(candidate, blob)
    behavior_score, behavior_facts, behavior_flags = score_behavior(candidate)
    education_score = score_education(candidate)

    facts: dict[str, Any] = {
        "candidate_id": cid,
        "title": current_title,
        "career_titles": career_titles,
        "years": years,
        "company": str(profile.get("current_company", "")),
        "industry": str(profile.get("current_industry", "")),
        "location": str(profile.get("location", "")),
        "country": str(profile.get("country", "")),
        "core_skills": core_skills,
        "core_count": core_count,
        "evidence_hits": evidence_hits,
        "has_production": has_production,
        "has_ir": has_ir,
        "target_city": target_city,
        "acceptable_city": acceptable_city,
        "current_product": current_product,
        "services_only": services_only,
        "product_hits": product_hits,
        "expert_zero_duration": expert_zero_duration,
        "behavior": behavior_facts,
    }

    penalty_score, penalty_flags = score_penalties(
        candidate,
        facts,
        core_count,
        evidence_hits,
        has_production,
        has_ir,
        expert_zero_duration,
        services_only,
    )

    dense_score = clamp(float(dense_similarity), -1.0, 1.0) * 4.0
    fit_base = (
        role_score
        + skill_score
        + evidence_score
        + experience_score
        + location_score
        + company_score
        + education_score
        + dense_score
    )
    behavior_multiplier = 0.93 + clamp(behavior_score, -10.0, 28.0) / 160.0
    final_score = fit_base * behavior_multiplier + behavior_score - penalty_score

    stage1_score = (
        role_score
        + 0.78 * skill_score
        + 0.86 * evidence_score
        + 0.55 * experience_score
        + 0.55 * location_score
        + 0.60 * company_score
        + 0.45 * behavior_score
        + dense_score
        - 0.60 * penalty_score
    )

    components = {
        "role": round(role_score, 4),
        "skills": round(skill_score, 4),
        "evidence": round(evidence_score, 4),
        "experience": round(experience_score, 4),
        "location": round(location_score, 4),
        "company": round(company_score, 4),
        "behavior": round(behavior_score, 4),
        "education": round(education_score, 4),
        "dense": round(dense_score, 4),
        "penalty": round(penalty_score, 4),
        "behavior_multiplier": round(behavior_multiplier, 4),
    }

    flags = behavior_flags + penalty_flags
    return CandidateScore(
        candidate_id=cid,
        stage1_score=stage1_score,
        final_score=final_score,
        components=components,
        facts=facts,
        flags=flags,
    )


def iter_candidates(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array of candidates")
        for seen, candidate in enumerate(data):
            if limit is not None and seen >= limit:
                break
            yield candidate
        return

    seen = 0
    with open_candidate_file(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                break


def retrieve_stage1(
    candidates_path: Path,
    dense_scores: dict[str, float],
    stage1_size: int,
    limit: int | None,
) -> tuple[list[CandidateScore], int]:
    heap: list[tuple[float, str, CandidateScore]] = []
    total = 0

    for candidate in iter_candidates(candidates_path, limit):
        total += 1
        cid = str(candidate.get("candidate_id", ""))
        scored = score_candidate(candidate, dense_scores.get(cid, 0.0))
        item = (scored.stage1_score, cid, scored)
        if len(heap) < stage1_size:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] < heap[0][1]):
            heapq.heapreplace(heap, item)

    return [item[2] for item in heap], total


def rerank_stage2(stage1_candidates: list[CandidateScore], top_k: int) -> list[CandidateScore]:
    ordered = sorted(stage1_candidates, key=lambda item: (-item.final_score, item.candidate_id))
    selected = ordered[:top_k]

    if not selected:
        return []

    max_raw = max(item.final_score for item in selected)
    min_raw = min(item.final_score for item in selected)
    span = max(max_raw - min_raw, 1e-9)
    previous = 1.0

    for index, item in enumerate(selected, start=1):
        scaled = 0.420000 + 0.575000 * ((item.final_score - min_raw) / span)
        if index == 1:
            display = min(0.995000, scaled)
        else:
            display = min(previous - 0.000010, scaled)
        item.rank = index
        item.display_score = max(display, 0.001000)
        item.reasoning = make_reasoning(item, index)
        previous = item.display_score

    return selected


def behavior_phrase(facts: dict[str, Any]) -> str:
    behavior = facts.get("behavior", {})
    pieces: list[str] = []
    if behavior.get("open_to_work"):
        pieces.append("open to work")
    response_rate = behavior.get("response_rate")
    if isinstance(response_rate, (int, float)):
        pieces.append(f"{response_rate:.2f} recruiter response rate")
    notice = behavior.get("notice_days")
    if isinstance(notice, int):
        pieces.append(f"{notice}-day notice")
    if behavior.get("willing_to_relocate"):
        pieces.append("willing to relocate")
    last_active = behavior.get("last_active_date")
    if last_active:
        pieces.append(f"last active {last_active}")
    return serial_join(pieces, 4)


def concern_phrase(item: CandidateScore) -> str:
    facts = item.facts
    behavior = facts.get("behavior", {})
    years = float(facts.get("years") or 0.0)
    country = str(facts.get("country", ""))
    location = str(facts.get("location", ""))
    notice = behavior.get("notice_days")
    response_rate = behavior.get("response_rate")
    core_count = int(facts.get("core_count") or 0)

    if country != "India":
        return f"concern: location is {location}, outside India"
    if years < 5.0 or years > 9.0:
        return f"concern: {norm_years(years)} years is outside the JD's 5-9 year band"
    if isinstance(notice, int) and notice > 90:
        return f"concern: notice period is {notice} days"
    if isinstance(response_rate, (int, float)) and response_rate < 0.25:
        return f"concern: recruiter response rate is only {response_rate:.2f}"
    if not facts.get("target_city") and not facts.get("acceptable_city") and not behavior.get("willing_to_relocate"):
        return f"concern: location is {location}, not a target JD city"
    if facts.get("services_only"):
        return "concern: services-company history is heavier than the JD prefers"
    if not facts.get("current_product") and int(facts.get("product_hits") or 0) == 0:
        return "concern: limited product-company evidence"
    if not facts.get("has_ir") and core_count < 4:
        return "concern: retrieval/ranking evidence is thinner than top profiles"
    if "vision_speech_without_ir" in item.flags:
        return "concern: profile leans toward CV/speech without enough IR evidence"
    if facts.get("target_city"):
        return "location directly matches the Pune/Noida preference"
    if facts.get("acceptable_city"):
        return f"{location} is in the JD's acceptable India location set"
    if behavior.get("willing_to_relocate"):
        return "relocation flag helps Pune/Noida logistics"
    return "India location keeps outreach in scope"


def make_reasoning(item: CandidateScore, rank: int) -> str:
    facts = item.facts
    title = str(facts.get("title") or "Candidate")
    years = float(facts.get("years") or 0.0)
    company = str(facts.get("company") or "current employer")
    location = str(facts.get("location") or "unknown location")
    evidence = list(facts.get("evidence_hits") or [])
    core_skills = list(facts.get("core_skills") or [])

    if evidence:
        proof = f"career evidence includes {serial_join(evidence, 4)}"
    elif core_skills:
        proof = f"core skills include {serial_join(core_skills, 4)}"
    else:
        proof = "career history is adjacent to the role"

    if rank % 4 == 1:
        first = f"{title} with {norm_years(years)} years at {company}; {proof}."
    elif rank % 4 == 2:
        first = f"{title} at {company} brings {norm_years(years)} years; {proof}."
    elif rank % 4 == 3:
        first = f"{norm_years(years)}-year {title} profile from {company}; {proof}."
    else:
        first = f"{title} based in {location} with {norm_years(years)} years; {proof}."

    behavior = behavior_phrase(facts)
    concern = concern_phrase(item)
    if behavior:
        second = f"Redrob signals show {behavior}; {concern}."
    else:
        second = f"Redrob behavioral data is limited; {concern}."

    return f"{first} {second}"


def write_submission(path: Path, rows: list[CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for item in rows:
            writer.writerow(
                [
                    item.candidate_id,
                    item.rank,
                    f"{item.display_score:.6f}",
                    item.reasoning,
                ]
            )


def audit_rows(rows: list[CandidateScore], total_candidates: int, stage1_count: int, elapsed: float) -> dict[str, Any]:
    flag_counts: dict[str, int] = {}
    for item in rows:
        for flag in item.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    reasoning_values = [item.reasoning for item in rows]
    audit = {
        "total_candidates_seen": total_candidates,
        "stage1_candidates_reranked": stage1_count,
        "top_rows": len(rows),
        "runtime_seconds": round(elapsed, 3),
        "unique_candidate_ids": len({item.candidate_id for item in rows}),
        "unique_ranks": len({item.rank for item in rows}),
        "scores_non_increasing": all(rows[i].display_score >= rows[i + 1].display_score for i in range(len(rows) - 1)),
        "reasoning_non_empty": all(bool(item.reasoning.strip()) for item in rows),
        "unique_reasoning_strings": len(set(reasoning_values)),
        "top100_flag_counts": flag_counts,
        "top100_component_averages": component_averages(rows),
        "risk_candidates": [
            {
                "candidate_id": item.candidate_id,
                "rank": item.rank,
                "flags": item.flags,
                "components": item.components,
            }
            for item in rows
            if item.flags
        ],
    }
    return audit


def component_averages(rows: list[CandidateScore]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for item in rows for key in item.components})
    out: dict[str, float] = {}
    for key in keys:
        out[key] = round(sum(item.components.get(key, 0.0) for item in rows) / len(rows), 4)
    return out


def write_audit(artifacts_dir: Path, audit: dict[str, Any]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "ranking_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank Redrob candidates offline.")
    parser.add_argument("--candidates", type=Path, default=Path("candidates.jsonl"), help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", type=Path, default=Path("team_AjGreat.csv"), help="Output CSV path")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Artifact directory for optional dense scores and audit output")
    parser.add_argument("--stage1-size", type=int, default=1500, help="Number of candidates to keep for stage-2 reranking")
    parser.add_argument("--top-k", type=int, default=100, help="Number of ranked rows to output")
    parser.add_argument("--limit", type=int, default=None, help="Optional candidate limit for sandbox smoke tests")
    parser.add_argument("--no-audit", action="store_true", help="Do not write artifacts/ranking_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    if not args.candidates.exists():
        raise FileNotFoundError(f"Candidate file not found: {args.candidates}")
    if args.stage1_size < args.top_k:
        args.stage1_size = args.top_k

    dense_scores = load_dense_scores(args.artifacts)
    stage1_candidates, total_candidates = retrieve_stage1(
        args.candidates,
        dense_scores,
        args.stage1_size,
        args.limit,
    )
    rows = rerank_stage2(stage1_candidates, min(args.top_k, len(stage1_candidates)))
    write_submission(args.out, rows)

    elapsed = time.perf_counter() - start
    audit = audit_rows(rows, total_candidates, len(stage1_candidates), elapsed)
    if not args.no_audit:
        write_audit(args.artifacts, audit)

    print(
        f"Wrote {len(rows)} rows to {args.out} from {total_candidates} candidates "
        f"in {elapsed:.2f}s; stage1={len(stage1_candidates)}"
    )


if __name__ == "__main__":
    main()
