# Redrob Offline Candidate Ranker - India Runs Hackathon

This repository ranks the top 100 candidates for the Redrob Senior AI Engineer
job description. The production ranking step is deterministic, CPU-only, and
does not use network calls, hosted LLM APIs, local LLM inference, or GPUs.

## Reproduce Submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts ./artifacts
python validate_submission.py ./submission.csv
python audit_submission.py --candidates ./candidates.jsonl --submission ./submission.csv
```

Expected result:

- `submission.csv` with exactly 100 ranked rows.
- `validate_submission.py` prints `Submission is valid.`
- `audit_submission.py` prints `Audit passed.`
- `artifacts/ranking_audit.json` contains runtime, score-order, reasoning, and
  honeypot-risk diagnostics.

The measured full run on the local machine was about 16 seconds for 100,000
candidates.

## Methodology Summary

Deterministic offline two-stage ranker for the Senior AI Engineer JD. The JD is
first converted into `rubric.yaml`, a structured ideal-candidate profile that
defines what each resume is scored against. Stage 1 streams all 100K candidates
and retrieves the strongest 1500 using rubric-derived signals: role fit, core
AI/search/retrieval skills, career-history evidence, product-company
background, India/location fit, Redrob availability signals, and optional
precomputed dense similarity. Stage 2 reranks with capped structured
components rather than live embedding or LLM calls: senior ML/search roles,
production retrieval/ranking evidence, vector/hybrid search infrastructure,
ranking evaluation metrics, skill trust, product-company experience, and
behavioral readiness. Honeypot-like profiles are not hard-coded by ID; they are
penalized through generic trap checks such as non-tech keyword stuffing, expert
skills with zero duration, inconsistent experience duration, pure research/no
production evidence, CV/speech-only profiles, long inactivity, poor recruiter
response, and long notice periods. Reasoning is generated deterministically
from candidate JSON facts only.

## Pipeline

```text
candidates.jsonl
  -> score_candidate()
  -> score_penalties() applies honeypot-like penalties
  -> Stage 1 top 1500 retrieval
  -> Stage 2 final reranking
  -> submission.csv
  -> ranking_audit.json report
```

`ranking_audit.json` is only a report. It does not remove rows after ranking.
Honeypot handling happens during scoring through penalties.

## Files

- `rank.py`: main offline ranker and deterministic reasoning generator.
- `rubric.yaml`: JD-derived ideal-candidate profile and scoring rubric used as the comparison target.
- `submission.csv`: generated top-100 ranked output.
- `audit_submission.py`: local audit for ordering, uniqueness, candidate
  existence, and grounded reasoning.
- `validate_submission.py`: official format validator from the challenge bundle.
- `artifacts/ranking_audit.json`: full-run audit summary and risk flags.
- `extract_feature_sets.py`: pandas-based profiling script used to derive
  company, title, industry, and skill feature-set candidates.
- `feature_profiles/`: generated profiling outputs from the full corpus.
- `candidate_schema.json`, `job_description.docx`, `redrob_signals_doc.docx`,
  `submission_spec.docx`: challenge-provided references.

## Feature Set Extraction

Sets like `PRODUCT_COMPANIES`, `PRODUCT_INDUSTRIES`, and `SERVICES_COMPANIES`
were created from corpus profiling plus JD knowledge. The profiling script
derives candidate sets from repeated evidence in `candidates.jsonl`, including:

- company career/current counts
- dominant industries per company
- target AI/search/ML title ratios
- production retrieval/ranking evidence ratios
- core skill density
- service-industry ratios

Run:

```bash
python extract_feature_sets.py --candidates ./candidates.jsonl --out-dir ./feature_profiles
```

Outputs:

- `feature_profiles/company_profile.csv`
- `feature_profiles/title_profile.csv`
- `feature_profiles/industry_profile.csv`
- `feature_profiles/skill_profile.csv`
- `feature_profiles/suggested_feature_sets.py`
- `feature_profiles/profile_summary.json`

This profiling script requires pandas. The production `rank.py` command itself
does not require pandas.

## Papers Referenced

- ConFit v2: Improving Resume-Job Matching using Hypothetical Resume Embedding
  and Runner-Up Hard-Negative Mining  
  https://arxiv.org/abs/2502.12361  
  Used as inspiration for the JD-to-ideal-candidate / hypothetical-resume idea
  and asymmetric resume-JD matching.

- ConFit v3: Enhancing Resume-Job Matching Using LLM  
  https://arxiv.org/abs/2605.09760  
  Used as inspiration for the two-stage retrieve-then-rerank architecture. The
  LLM reranker was not used because the ranking step must be offline and CPU
  only.

- A Comparative Study of LSTM and BART for Resume Summarization  
  https://arxiv.org/abs/2306.13315  
  Reviewed for description generation, but not used directly because this task
  is ranking plus fact-grounded reasoning, not resume summarization.
