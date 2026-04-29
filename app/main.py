from __future__ import annotations

import json
import os
import re
from urllib import error as url_error
from urllib import request as url_request
import uuid
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from dataset_processor import (
    CATEGORY_PROFILE_PATH,
    DATASET_ROOT,
    SKILLS_OUTPUT_PATH,
    process_dataset,
)
from matcher import JobReranker, SemanticMatcher
from parser import analyze_resume_quality, read_text_from_file
from scorer import (
    calculate_overall_score,
    calculate_skill_match_ratio,
)
from skills_extractor import extract_skills, load_skills


ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT_DIR / "models" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_MODELS_DIR = ROOT_DIR / "models" / "trained"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

semantic_matcher = SemanticMatcher("all-MiniLM-L6-v2")
analysis_store: Dict[str, Dict[str, object]] = {}
resume_store: Dict[str, Dict[str, str]] = {}

ML_NUMERIC_FEATURES = [
    "num_skills",
    "num_projects",
    "num_experiences",
    "num_courses",
    "num_languages",
    "word_count",
    "avg_sentence_length",
    "section_count",
    "multi_domain_flag",
    "skill_density",
    "metric_count",
    "metric_ratio",
    "avg_bullet_length",
    "has_summary",
    "has_projects",
    "has_experience",
]
ML_TEXT_FEATURE = "resume_text"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))

PROGRAMMING_LANGUAGES = {
    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "rust",
    "kotlin",
    "swift",
    "php",
    "ruby",
    "r",
    "scala",
    "matlab",
    "sql",
    "bash",
}

DOMAIN_KEYWORDS = {
    "Software Engineering": {"software", "backend", "frontend", "api", "microservices", "devops", "deployment", "kubernetes"},
    "Data Science": {"machine learning", "data science", "model", "feature", "statistics", "forecast", "nlp"},
    "HR": {"recruitment", "talent", "onboarding", "employee", "hr", "policy", "workforce"},
    "Marketing": {"seo", "campaign", "brand", "conversion", "funnel", "ads", "marketing"},
    "Finance": {"finance", "valuation", "budget", "forecasting", "portfolio", "risk", "cash flow"},
    "Sales": {"sales", "quota", "pipeline", "account", "prospecting", "deal", "renewal"},
    "Product Management": {"product", "roadmap", "backlog", "discovery", "prioritization", "go-to-market"},
    "UI/UX Design": {"ux", "ui", "wireframe", "prototype", "usability", "figma", "design system"},
    "Cybersecurity": {"security", "siem", "incident", "vulnerability", "threat", "penetration", "soc"},
    "Business Analyst": {"requirements", "process", "kpi", "stakeholder", "gap analysis", "business analyst", "uat"},
}

JOB_DESCRIPTION_LIBRARY = [
    {
        "id": "se_backend_python",
        "title": "Backend Software Engineer (Python)",
        "domain": "Software Engineering",
        "description": (
            "We are hiring a Backend Software Engineer to design scalable APIs and microservices. "
            "Required: Python, SQL, REST API design, Docker, CI/CD, and debugging production systems. "
            "Experience with cloud platforms and performance optimization is preferred."
        ),
    },
    {
        "id": "ds_ml_engineer",
        "title": "Machine Learning Engineer",
        "domain": "Data Science",
        "description": (
            "Looking for an ML Engineer to build and deploy predictive models. "
            "Required: Python, scikit-learn, feature engineering, model evaluation, SQL, and MLOps basics. "
            "Nice to have: NLP, experiment tracking, and pipeline orchestration."
        ),
    },
    {
        "id": "hr_talent_acquisition",
        "title": "Talent Acquisition Specialist",
        "domain": "HR",
        "description": (
            "Seeking a Talent Acquisition Specialist to manage end-to-end hiring. "
            "Required: candidate sourcing, interview coordination, ATS tools, stakeholder communication, and onboarding support. "
            "Experience in employer branding and hiring analytics is a plus."
        ),
    },
    {
        "id": "marketing_growth",
        "title": "Growth Marketing Specialist",
        "domain": "Marketing",
        "description": (
            "We need a Growth Marketing Specialist to run multi-channel campaigns and optimize acquisition funnels. "
            "Required: SEO/SEM, campaign analytics, A/B testing, content strategy, and conversion optimization. "
            "Experience with GA4 and marketing automation tools preferred."
        ),
    },
    {
        "id": "finance_analyst",
        "title": "Financial Analyst",
        "domain": "Finance",
        "description": (
            "Hiring a Financial Analyst to support forecasting, budgeting, and business performance reporting. "
            "Required: financial modeling, variance analysis, Excel, SQL, and dashboarding. "
            "Experience with planning systems and risk analysis is beneficial."
        ),
    },
    {
        "id": "sales_account_exec",
        "title": "Account Executive",
        "domain": "Sales",
        "description": (
            "Seeking an Account Executive to drive pipeline and close mid-market/enterprise opportunities. "
            "Required: prospecting, qualification, CRM management, negotiation, and forecast discipline. "
            "Experience with Salesforce and consultative selling is preferred."
        ),
    },
    {
        "id": "pm_product_manager",
        "title": "Product Manager",
        "domain": "Product Management",
        "description": (
            "Looking for a Product Manager to lead discovery, prioritization, and cross-functional delivery. "
            "Required: roadmap planning, user stories, stakeholder alignment, metrics ownership, and experimentation. "
            "Technical collaboration with engineering and design teams is essential."
        ),
    },
    {
        "id": "ux_product_designer",
        "title": "Product Designer (UI/UX)",
        "domain": "UI/UX Design",
        "description": (
            "Hiring a Product Designer to create intuitive web/mobile experiences. "
            "Required: user research, wireframing, interaction design, prototyping, and usability testing. "
            "Figma and design-system experience strongly preferred."
        ),
    },
    {
        "id": "cyber_soc_analyst",
        "title": "SOC Analyst",
        "domain": "Cybersecurity",
        "description": (
            "Seeking a SOC Analyst to monitor threats and respond to security incidents. "
            "Required: SIEM monitoring, incident triage, vulnerability remediation, and log analysis. "
            "Hands-on knowledge of network security and endpoint controls is preferred."
        ),
    },
    {
        "id": "ba_business_analyst",
        "title": "Business Analyst",
        "domain": "Business Analyst",
        "description": (
            "Looking for a Business Analyst to gather requirements and improve operational workflows. "
            "Required: process mapping, stakeholder workshops, KPI definition, UAT support, and reporting. "
            "Experience with SQL and BI tools is a plus."
        ),
    },
    {
        "id": "accounting_tax_analyst",
        "title": "Tax and Accounting Analyst",
        "domain": "Accounting",
        "description": (
            "Seeking an Accounting Analyst for month-end close, reconciliation, and tax compliance support. "
            "Required: general ledger reconciliation, AP/AR management, financial reporting, and Excel proficiency. "
            "Experience with ERP systems like SAP FICO or QuickBooks is preferred."
        ),
    },
    {
        "id": "healthcare_operations_analyst",
        "title": "Healthcare Operations Analyst",
        "domain": "Healthcare",
        "description": (
            "Hiring a Healthcare Operations Analyst to improve patient workflows and reporting quality. "
            "Required: healthcare compliance awareness, clinical documentation handling, KPI reporting, and process coordination. "
            "Experience with EHR tools such as Epic or Cerner is a plus."
        ),
    },
    {
        "id": "education_curriculum_specialist",
        "title": "Curriculum and Instruction Specialist",
        "domain": "Education",
        "description": (
            "Looking for an Education Specialist to design curriculum and assessment frameworks. "
            "Required: curriculum planning, student performance tracking, instructional design, and classroom delivery strategy. "
            "Experience with LMS tools and blended learning is preferred."
        ),
    },
    {
        "id": "legal_contract_specialist",
        "title": "Legal Contract Specialist",
        "domain": "Legal",
        "description": (
            "Seeking a Legal Contract Specialist to review contracts and support compliance workflows. "
            "Required: legal research, contract review, policy drafting, and regulatory interpretation. "
            "Experience with legal research platforms and document control processes is beneficial."
        ),
    },
    {
        "id": "operations_process_manager",
        "title": "Operations Process Manager",
        "domain": "Operations",
        "description": (
            "We need an Operations Manager to optimize workflows and manage service delivery SLAs. "
            "Required: process optimization, cross-team coordination, operational reporting, and issue resolution governance. "
            "Experience with KPI dashboards and continuous improvement methods is expected."
        ),
    },
    {
        "id": "supply_chain_planner",
        "title": "Supply Chain Planning Analyst",
        "domain": "Supply Chain",
        "description": (
            "Hiring a Supply Chain Analyst to manage demand planning and inventory health. "
            "Required: demand forecasting, procurement coordination, logistics monitoring, and vendor management. "
            "Experience with ERP/SCM systems and inventory optimization is preferred."
        ),
    },
    {
        "id": "customer_support_specialist",
        "title": "Customer Support Specialist",
        "domain": "Customer Support",
        "description": (
            "Seeking a Customer Support Specialist to triage issues and improve resolution quality. "
            "Required: ticket handling, escalation management, customer communication, and SLA adherence. "
            "Experience with Zendesk, Freshdesk, or similar support tools is preferred."
        ),
    },
    {
        "id": "network_engineer_infra",
        "title": "Network Infrastructure Engineer",
        "domain": "Network Engineering",
        "description": (
            "Looking for a Network Engineer to manage routing, switching, and network reliability. "
            "Required: network troubleshooting, firewall configuration, monitoring, and incident response. "
            "Knowledge of Cisco/Juniper and observability tooling is preferred."
        ),
    },
    {
        "id": "cloud_platform_engineer",
        "title": "Cloud Platform Engineer",
        "domain": "Cloud Engineering",
        "description": (
            "Hiring a Cloud Engineer to build scalable cloud infrastructure and CI/CD automation. "
            "Required: AWS/Azure/GCP exposure, infrastructure as code, containerization, and platform reliability practices. "
            "Experience with Terraform and Kubernetes is highly valued."
        ),
    },
    {
        "id": "qa_automation_engineer",
        "title": "QA Automation Engineer",
        "domain": "QA Testing",
        "description": (
            "We need a QA Automation Engineer to improve product quality across releases. "
            "Required: test strategy, regression automation, API testing, defect lifecycle management, and release validation. "
            "Experience with Selenium/Cypress and CI pipelines is a plus."
        ),
    },
    {
        "id": "content_writer_seo",
        "title": "SEO Content Writer",
        "domain": "Content Writing",
        "description": (
            "Seeking a Content Writer to create high-quality SEO-driven and product-focused content. "
            "Required: long-form writing, editorial planning, content research, and clear technical communication. "
            "Experience with CMS workflows and keyword research platforms is preferred."
        ),
    },
    {
        "id": "graphic_visual_designer",
        "title": "Graphic and Visual Designer",
        "domain": "Graphic Design",
        "description": (
            "Looking for a Graphic Designer to produce campaign and brand assets across channels. "
            "Required: visual hierarchy, layout design, typography, and brand system consistency. "
            "Strong skills in Adobe Creative Suite and Figma are expected."
        ),
    },
    {
        "id": "hospitality_guest_relations",
        "title": "Guest Relations Manager",
        "domain": "Hospitality",
        "description": (
            "Hiring a Guest Relations Manager to improve service quality and customer satisfaction. "
            "Required: front-office operations, reservation handling, complaint resolution, and team coordination. "
            "Experience with hospitality PMS tools and service quality metrics is preferred."
        ),
    },
    {
        "id": "construction_site_engineer",
        "title": "Construction Site Engineer",
        "domain": "Construction",
        "description": (
            "Seeking a Site Engineer to coordinate construction execution, safety, and schedule adherence. "
            "Required: site planning, contractor coordination, progress reporting, and compliance checks. "
            "Experience with project planning tools and drawing management is important."
        ),
    },
]


def _clamp_float(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _normalize_text_for_lookup(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


JD_DESCRIPTION_TO_DOMAIN = {
    _normalize_text_for_lookup(item.get("description", "")): str(item.get("domain", ""))
    for item in JOB_DESCRIPTION_LIBRARY
    if item.get("description")
}


rf_regressor = None
quality_classifier = None
domain_classifier = None
preprocessor = None
quality_label_encoder = None
domain_label_encoder = None
LAST_OLLAMA_ERROR = ""
JOB_LIBRARY_INDEX: list[dict[str, object]] = []
quality_calibrator = None
domain_calibrator = None
confidence_thresholds: dict[str, dict[str, float]] = {}
job_reranker = None

DOMAIN_SUGGESTION_EXAMPLES: dict[str, dict[str, list[str]]] = {
    "Software Engineering": {
        "projects": ["Build a REST API with authentication", "Design a CI/CD pipeline for a sample app"],
        "courses": ["Distributed Systems", "System Design Fundamentals"],
    },
    "Data Science": {
        "projects": ["Customer churn prediction model", "Demand forecasting with time-series"],
        "courses": ["Applied Machine Learning", "Feature Engineering and Model Evaluation"],
    },
    "HR": {
        "projects": ["Recruitment funnel analytics dashboard", "Onboarding process optimization"],
        "courses": ["People Analytics", "Strategic Human Resource Management"],
    },
    "Marketing": {
        "projects": ["SEO content cluster strategy", "Campaign attribution dashboard"],
        "courses": ["Performance Marketing", "Marketing Analytics"],
    },
    "Finance": {
        "projects": ["Budget variance analysis dashboard", "Discounted cash-flow valuation model"],
        "courses": ["Corporate Finance", "Financial Modeling"],
    },
    "Product Management": {
        "projects": ["User onboarding optimization experiment", "Roadmap prioritization framework"],
        "courses": ["Product Strategy", "Product Analytics"],
    },
    "UI/UX Design": {
        "projects": ["Checkout flow redesign case study", "Design system for a mobile app"],
        "courses": ["User Experience Research", "Interaction Design"],
    },
    "Healthcare": {
        "projects": ["Patient workflow bottleneck analysis", "Clinical KPI reporting dashboard"],
        "courses": ["Healthcare Operations Management", "Clinical Data and EHR Fundamentals"],
    },
    "Accounting": {
        "projects": ["Month-end close automation tracker", "AP/AR reconciliation dashboard"],
        "courses": ["Financial Accounting", "Tax Compliance and Reporting"],
    },
    "Operations": {
        "projects": ["SLA breach root-cause reduction program", "Process cycle-time improvement initiative"],
        "courses": ["Operations Management", "Lean Six Sigma Foundations"],
    },
    "Supply Chain": {
        "projects": ["Inventory optimization model", "Supplier lead-time variability analysis"],
        "courses": ["Supply Chain Planning", "Procurement and Logistics Analytics"],
    },
    "Customer Support": {
        "projects": ["First-response time improvement project", "Ticket deflection knowledge-base initiative"],
        "courses": ["Customer Success Operations", "Service Quality Management"],
    },
}

DEFAULT_PROJECT_EXAMPLES = [
    "Role-relevant portfolio project with measurable outcomes",
    "Process improvement project with before/after metrics",
]

DEFAULT_COURSE_EXAMPLES = [
    "Domain fundamentals course",
    "Advanced practical course with capstone",
]

NOISE_SKILL_TERMS = {
    "and",
    "or",
    "as",
    "with",
    "for",
    "the",
    "a",
    "an",
    "in",
    "of",
    "to",
    "on",
    "by",
    "at",
    "from",
    "experience",
    "skills",
    "knowledge",
    "ability",
    "support",
    "team",
    "work",
    "management",
    "communication",
}

SHORT_SKILL_ALLOWLIST = {"r", "c", "go", "hr", "bi", "ui", "ux", "qa", "sql"}


def _top_k_predictions(probabilities: list[float], encoder, k: int = 2) -> list[dict]:
    indexed = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:k]
    output = []
    for idx, prob in indexed:
        label = str(encoder.inverse_transform([idx])[0])
        output.append({"label": label, "probability": round(float(prob), 4)})
    return output


def _normalize_feedback_list(value: object, default: list[str]) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned[:8] if cleaned else default
    if isinstance(value, str) and value.strip():
        # Accept newline/bullet formatted strings and split into list items.
        chunks = [
            part.strip(" -\t\r")
            for part in re.split(r"\n+|\s*•\s*|\s*-\s+", value)
            if part.strip(" -\t\r")
        ]
        return chunks[:8] if chunks else [value.strip()]
    return default


def _examples_for_domain(domain: str) -> tuple[list[str], list[str]]:
    profile = DOMAIN_SUGGESTION_EXAMPLES.get(domain, {})
    projects = profile.get("projects", DEFAULT_PROJECT_EXAMPLES)
    courses = profile.get("courses", DEFAULT_COURSE_EXAMPLES)
    return projects, courses


def _calibrated_confidence(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0

    ranked = sorted((_clamp_float(probability) for probability in probabilities), reverse=True)
    top_probability = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top_probability - runner_up)
    calibrated = (top_probability * 0.78) + (margin * 0.22)
    if top_probability >= 0.8:
        calibrated += 0.05
    return round(_clamp_float(calibrated), 4)


def _confidence_flags(probabilities: list[float], thresholds: dict[str, float]) -> dict[str, float | bool]:
    if not probabilities:
        return {"low_confidence": True, "top": 0.0, "margin": 0.0}

    ranked = sorted((_clamp_float(probability) for probability in probabilities), reverse=True)
    top_probability = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else 0.0
    margin = max(0.0, top_probability - runner_up)
    min_confidence = float(thresholds.get("min_confidence", 0.5))
    min_margin = float(thresholds.get("min_margin", 0.05))

    low_confidence = top_probability < min_confidence or margin < min_margin
    return {"low_confidence": low_confidence, "top": top_probability, "margin": margin}


def _build_job_library_index() -> list[dict[str, object]]:
    indexed_library: list[dict[str, object]] = []
    for item in JOB_DESCRIPTION_LIBRARY:
        description = str(item.get("description", ""))
        indexed_library.append(
            {
                **item,
                "skills": set(_sanitize_skill_terms(extract_skills(description, SKILLS_SET))),
            }
        )
    return indexed_library


def _job_match_summary(matched_skills: set[str], missing_skills: set[str]) -> str:
    matched_sample = ", ".join(sorted(matched_skills)[:3])
    missing_sample = ", ".join(sorted(missing_skills)[:3])

    if matched_sample and missing_sample:
        return f"Strong overlap in {matched_sample}; remaining gaps are {missing_sample}."
    if matched_sample:
        return f"Strong overlap in {matched_sample}."
    if missing_sample:
        return f"Primary gaps are {missing_sample}."
    return "Best fit is based on overall semantic alignment and role pattern matching."


def _rank_job_recommendations(resume_text: str, resume_skills: set[str], predicted_domain: str) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    resume_domain = _normalize_text_for_lookup(predicted_domain)

    for job in JOB_LIBRARY_INDEX:
        job_description = str(job.get("description", ""))
        job_skills = set(job.get("skills", set()))
        semantic_score = semantic_matcher.semantic_similarity(resume_text, job_description)
        skill_match_ratio = calculate_skill_match_ratio(resume_skills, job_skills)
        job_domain = _normalize_text_for_lookup(str(job.get("domain", "")))
        domain_alignment = 1.0 if resume_domain and resume_domain == job_domain else 0.35 if resume_domain and resume_domain in job_description.lower() else 0.0

        score = (
            semantic_score * 0.56
            + skill_match_ratio * 0.34
            + domain_alignment * 0.10
        )

        matched_skills = resume_skills.intersection(job_skills)
        missing_skills = job_skills.difference(resume_skills)
        confidence = _clamp_float((score * 0.82) + (max(0.0, semantic_score - skill_match_ratio) * 0.18))

        recommendations.append(
            {
                "id": job.get("id"),
                "title": job.get("title"),
                "domain": job.get("domain"),
                "description": job_description,
                "score": round(_clamp_float(score), 4),
                "confidence": round(confidence, 4),
                "semantic_score": round(_clamp_float(semantic_score), 4),
                "skill_match_ratio": round(_clamp_float(skill_match_ratio), 4),
                "matched_skills": sorted(matched_skills),
                "missing_skills": sorted(missing_skills),
                "reason": _job_match_summary(matched_skills, missing_skills),
            }
        )

    recommendations.sort(key=lambda item: (float(item["score"]), float(item["confidence"])), reverse=True)

    if job_reranker and recommendations:
        top_candidates = recommendations[:6]
        pairs = [(resume_text, str(item.get("description", ""))) for item in top_candidates]
        cross_scores = job_reranker.score_pairs(pairs)
        if cross_scores:
            min_score = min(cross_scores)
            max_score = max(cross_scores)
            scale = max_score - min_score
            for item, score in zip(top_candidates, cross_scores):
                cross_norm = (score - min_score) / scale if scale > 0 else 0.5
                semantic_score = float(item.get("semantic_score", 0.0))
                skill_match_ratio = float(item.get("skill_match_ratio", 0.0))
                blended = (
                    cross_norm * 0.55
                    + semantic_score * 0.25
                    + skill_match_ratio * 0.20
                )
                item["score"] = round(_clamp_float(blended), 4)
                item["confidence"] = round(
                    _clamp_float((blended * 0.85) + max(0.0, cross_norm - skill_match_ratio) * 0.15),
                    4,
                )
            recommendations = sorted(recommendations, key=lambda item: (float(item["score"]), float(item["confidence"])), reverse=True)

    return recommendations


def _infer_jd_domain(job_description: str) -> str:
    normalized = _normalize_text_for_lookup(job_description)
    if normalized in JD_DESCRIPTION_TO_DOMAIN:
        return JD_DESCRIPTION_TO_DOMAIN[normalized]

    best_domain = ""
    best_hits = 0
    text = normalized
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best_domain = domain
            best_hits = hits
    return best_domain


def _sanitize_skill_terms(skills: set[str] | list[str]) -> list[str]:
    cleaned: list[str] = []
    for raw in skills:
        token = re.sub(r"\s+", " ", str(raw).strip().lower())
        token = re.sub(r"[^a-z0-9+.#\- ]", "", token).strip()
        if not token:
            continue
        if token in NOISE_SKILL_TERMS:
            continue
        if len(token) <= 2 and token not in SHORT_SKILL_ALLOWLIST:
            continue
        if token.isdigit():
            continue
        cleaned.append(token)

    # preserve order while deduping
    return list(dict.fromkeys(cleaned))


def _parenthesize_examples(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    markers = ["such as", "for example", "e.g.,", "e.g."]

    for marker in markers:
        idx = lowered.find(marker)
        if idx == -1:
            continue

        marker_end = idx + len(marker)
        prefix = stripped[:marker_end]
        suffix = stripped[marker_end:].strip()
        if not suffix:
            return stripped

        if suffix.startswith(":"):
            suffix = suffix[1:].strip()

        trailing = ""
        if suffix.endswith("."):
            trailing = "."
            suffix = suffix[:-1].rstrip()

        if suffix.startswith("(") and suffix.endswith(")"):
            return stripped

        return f"{prefix} ({suffix}){trailing}"

    return stripped


def _ensure_jd_focus(items: list[str], section: str) -> list[str]:
    jd_markers = (
        "jd",
        "job description",
        "required",
        "role",
        "fit",
        "align",
        "missing skill",
        "matched skill",
    )

    focused: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue

        text = text.replace("(for this JD)", "").strip()
        text = re.sub(r"\s+", " ", text)
        if text.endswith(".."):
            text = text.rstrip(".") + "."

        lowered = text.lower()
        if any(marker in lowered for marker in jd_markers):
            focused.append(text)
            continue

        # Keep natural language; add a soft JD anchor only when needed.
        if section == "strengths":
            focused.append(f"{text} This supports the JD requirements.")
        elif section == "weaknesses":
            focused.append(f"{text} This may reduce JD fit.")
        else:
            focused.append(f"{text} Prioritize this for better JD alignment.")

    return focused


def _enrich_feedback(feedback: dict, data: dict) -> dict:
    suggestions = list(feedback.get("suggestions", []))
    strengths = list(feedback.get("strengths", []))
    weaknesses = list(feedback.get("weaknesses", []))

    matched_skills = _sanitize_skill_terms(data.get("matched_skills", []))
    missing_skills = _sanitize_skill_terms(data.get("missing_skills", []))

    if matched_skills:
        top_matched = "; ".join(matched_skills[:3])
        strengths.append(f"Strong alignment with JD-required skills ({top_matched}).")

    if missing_skills:
        top_missing = "; ".join(missing_skills[:3])
        weaknesses.append(f"Gap against JD-required skills ({top_missing}).")
        suggestions.append(f"Add proof of JD-required skills ({top_missing}) through experience bullets or projects.")

    project_examples = data.get("project_examples", DEFAULT_PROJECT_EXAMPLES)
    if int(data.get("num_projects", 0) or 0) < 2 and isinstance(project_examples, list) and project_examples:
        suggestions.append(
            f"Add project work examples such as ({project_examples[0]}; {project_examples[min(1, len(project_examples)-1)]})."
        )

    course_examples = data.get("course_examples", DEFAULT_COURSE_EXAMPLES)
    if int(data.get("num_courses", 0) or 0) < 2 and isinstance(course_examples, list) and course_examples:
        suggestions.append(
            f"Add relevant courses/certifications, for example ({course_examples[0]}; {course_examples[min(1, len(course_examples)-1)]})."
        )

    def dedupe(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = _parenthesize_examples(item)
            if normalized and normalized not in seen:
                out.append(normalized)
                seen.add(normalized)
        return out

    strengths = dedupe(strengths)
    weaknesses = dedupe(weaknesses)
    suggestions = dedupe(suggestions)

    strengths = _ensure_jd_focus(strengths, "strengths")
    weaknesses = _ensure_jd_focus(weaknesses, "weaknesses")
    suggestions = _ensure_jd_focus(suggestions, "suggestions")

    feedback["strengths"] = strengths[:8]
    feedback["weaknesses"] = weaknesses[:8]
    feedback["suggestions"] = suggestions[:8]
    return feedback


def _extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if not text:
        return {}

    # Try direct parse first.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fallback: pull first JSON object from free-form model output.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def generate_llm_feedback(data: dict) -> dict:
    global LAST_OLLAMA_ERROR
    LAST_OLLAMA_ERROR = ""

    base_prompt = (
        "You are a resume reviewer. Analyze the provided resume analysis data and return ONLY valid JSON "
        "with exactly these keys: strengths, weaknesses, suggestions. "
        "Every bullet in strengths, weaknesses, and suggestions must be explicitly related to the provided JD requirements. "
        "Each value must be a list of concise, specific bullet-style strings. Avoid generic statements. "
        "Use natural language and avoid repeating the same JD phrase in every bullet. "
        "Suggestions must include practical next steps. "
        "If projects are low, include concrete project examples. If courses are low, include concrete course examples. "
        "Whenever you include examples, put the example text inside parentheses. "
        "Do not use placeholder words or noisy tokens as skills. "
        "Always include at least one suggestion to improve structure flow when flow score is low. "
        "Do not include markdown, explanations, or extra keys.\n\n"
        "Resume analysis data:\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2)}"
    )

    strict_retry_prompt = (
        "Return ONLY JSON. No prose. "
        "Required schema:\n"
        "{\"strengths\":[\"...\",\"...\",\"...\"],\"weaknesses\":[\"...\",\"...\",\"...\"],\"suggestions\":[\"...\",\"...\",\"...\"]}\n"
        "Rules: each list must have at least 3 non-empty strings, and every string must be JD-focused.\n\n"
        "Resume analysis data:\n"
        f"{json.dumps(data, ensure_ascii=False, indent=2)}"
    )

    def complete_missing_prompt(partial_feedback: dict, missing_keys: list[str]) -> str:
        return (
            "Return ONLY JSON. No prose. "
            "Fill the missing lists for this resume review schema: strengths, weaknesses, suggestions. "
            f"Missing keys that must be non-empty: {', '.join(missing_keys)}. "
            "Keep existing provided lists as-is and generate only what is missing. "
            "All bullets must be concise, JD-focused, and practical.\n\n"
            "Existing partial feedback:\n"
            f"{json.dumps(partial_feedback, ensure_ascii=False, indent=2)}\n\n"
            "Resume analysis data:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    def request_ollama(prompt_text: str) -> dict:
        body = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
            },
            "keep_alive": "20m",
        }

        req = url_request.Request(
            OLLAMA_API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with url_request.urlopen(req, timeout=OLLAMA_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))

    def parse_feedback(payload: dict) -> dict:
        response_obj = payload.get("response", "")
        parsed = {}

        if isinstance(response_obj, dict):
            parsed = response_obj
        else:
            response_text = str(response_obj).strip()
            if response_text:
                parsed = _extract_json_object(response_text)

        # Some local models return actual output in `thinking` with empty `response`.
        if not parsed:
            thinking_obj = payload.get("thinking", "")
            thinking_text = str(thinking_obj).strip() if thinking_obj is not None else ""
            if thinking_text:
                parsed = _extract_json_object(thinking_text)

        # Additional compatibility for chat-style wrappers.
        if not parsed:
            message_obj = payload.get("message")
            if isinstance(message_obj, dict):
                content_text = str(message_obj.get("content", "")).strip()
                if content_text:
                    parsed = _extract_json_object(content_text)

        if not parsed:
            output_obj = payload.get("output", "")
            output_text = str(output_obj).strip()
            if output_text:
                parsed = _extract_json_object(output_text)

        if not parsed:
            return {}

        lowered = {str(k).strip().lower(): v for k, v in parsed.items()}

        # Support nested response shapes from local models.
        if not any(key in lowered for key in ("strengths", "weaknesses", "suggestions")):
            nested = lowered.get("data")
            if isinstance(nested, dict):
                lowered = {str(k).strip().lower(): v for k, v in nested.items()}

        # Support common key aliases from local models.
        strengths_val = lowered.get("strengths", lowered.get("strength", []))
        weaknesses_val = lowered.get("weaknesses", lowered.get("weakness", lowered.get("areas_for_improvement", [])))
        suggestions_val = lowered.get("suggestions", lowered.get("recommendations", lowered.get("improvements", [])))

        feedback = {
            "strengths": _normalize_feedback_list(strengths_val, []),
            "weaknesses": _normalize_feedback_list(weaknesses_val, []),
            "suggestions": _normalize_feedback_list(suggestions_val, []),
        }
        return feedback

    def missing_feedback_keys(feedback: dict) -> list[str]:
        keys = ["strengths", "weaknesses", "suggestions"]
        return [key for key in keys if not feedback.get(key)]

    def merge_feedback(primary: dict, secondary: dict) -> dict:
        merged = {
            "strengths": list(primary.get("strengths", [])),
            "weaknesses": list(primary.get("weaknesses", [])),
            "suggestions": list(primary.get("suggestions", [])),
        }
        for key in ("strengths", "weaknesses", "suggestions"):
            if not merged[key] and secondary.get(key):
                merged[key] = list(secondary.get(key, []))
        return merged

    try:
        payload = request_ollama(base_prompt)
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        LAST_OLLAMA_ERROR = f"Ollama request failed: {exc}"
        return {}

    feedback = parse_feedback(payload)
    missing_keys = missing_feedback_keys(feedback)
    if feedback and not missing_keys:
        return _enrich_feedback(feedback, data)

    # Repair pass: ask Ollama to fill only missing sections from partial output.
    if feedback and missing_keys:
        try:
            repair_payload = request_ollama(complete_missing_prompt(feedback, missing_keys))
            repaired = parse_feedback(repair_payload)
            feedback = merge_feedback(feedback, repaired)
            if not missing_feedback_keys(feedback):
                return _enrich_feedback(feedback, data)
        except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            pass

    # Retry once with stronger schema instructions.
    try:
        retry_payload = request_ollama(strict_retry_prompt)
    except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        LAST_OLLAMA_ERROR = f"Ollama retry failed: {exc}"
        return {}

    feedback = parse_feedback(retry_payload)
    missing_keys = missing_feedback_keys(feedback)

    if feedback and missing_keys:
        try:
            repair_payload = request_ollama(complete_missing_prompt(feedback, missing_keys))
            repaired = parse_feedback(repair_payload)
            feedback = merge_feedback(feedback, repaired)
            missing_keys = missing_feedback_keys(feedback)
        except (url_error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            pass

    if not feedback or missing_keys:
        missing_text = ", ".join(missing_keys) if missing_keys else "strengths, weaknesses, suggestions"
        LAST_OLLAMA_ERROR = f"Ollama JSON missing one or more required non-empty lists: {missing_text}."
        return {}

    return _enrich_feedback(feedback, data)


def ensure_artifacts() -> None:
    if not SKILLS_OUTPUT_PATH.exists() or not CATEGORY_PROFILE_PATH.exists():
        process_dataset(dataset_root=DATASET_ROOT)


def load_trained_models() -> None:
    global rf_regressor
    global quality_classifier
    global domain_classifier
    global preprocessor
    global quality_label_encoder
    global domain_label_encoder
    global quality_calibrator
    global domain_calibrator
    global confidence_thresholds
    global job_reranker

    rf_regressor = joblib.load(TRAINED_MODELS_DIR / "rf_regressor.pkl")
    quality_classifier = joblib.load(TRAINED_MODELS_DIR / "quality_classifier.pkl")
    domain_classifier = joblib.load(TRAINED_MODELS_DIR / "domain_classifier.pkl")
    preprocessor = joblib.load(TRAINED_MODELS_DIR / "preprocessor.pkl")
    quality_label_encoder = joblib.load(TRAINED_MODELS_DIR / "quality_label_encoder.pkl")
    domain_label_encoder = joblib.load(TRAINED_MODELS_DIR / "domain_label_encoder.pkl")

    calibrator_path = TRAINED_MODELS_DIR / "quality_calibrator.pkl"
    if calibrator_path.exists():
        quality_calibrator = joblib.load(calibrator_path)

    calibrator_path = TRAINED_MODELS_DIR / "domain_calibrator.pkl"
    if calibrator_path.exists():
        domain_calibrator = joblib.load(calibrator_path)

    thresholds_path = TRAINED_MODELS_DIR / "confidence_thresholds.json"
    if thresholds_path.exists():
        try:
            confidence_thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            confidence_thresholds = {}

    try:
        job_reranker = JobReranker()
    except RuntimeError:
        job_reranker = None


def _avg_sentence_length(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return 0.0

    total_words = 0
    for sentence in sentences:
        total_words += len(re.findall(r"\b\w+\b", sentence))

    return round(total_words / len(sentences), 2)


def _extract_section_entries(resume_text: str) -> Dict[str, list[str]]:
    section_names = ["summary", "skills", "projects", "experience", "education", "courses", "languages"]
    section_pattern = re.compile(
        r"^\s*(?:##\s*)?(?:\[\s*)?(summary|skills|projects|experience|education|courses|languages)(?:\s*\])?\s*[:\-]?\s*$",
        flags=re.IGNORECASE,
    )

    sections: Dict[str, list[str]] = {name: [] for name in section_names}
    current_section = None
    lines = [line.rstrip() for line in resume_text.splitlines() if line.strip()]

    for raw in lines:
        line = raw.strip()
        header_match = section_pattern.match(line)
        if header_match:
            current_section = header_match.group(1).lower()
            continue

        if current_section:
            cleaned = re.sub(r"^[-*>\u2022\s]+", "", line).strip()
            if cleaned:
                sections[current_section].append(cleaned)

    # Fallback for single-line/noisy resumes where sections are inline.
    if not any(sections.values()):
        inline = re.split(
            r"(?i)\b(summary|skills|projects|experience|education|courses|languages)\b\s*[:\-]?",
            resume_text,
        )
        if len(inline) > 2:
            idx = 1
            while idx < len(inline) - 1:
                section = inline[idx].lower().strip()
                content = inline[idx + 1].strip()
                next_cut = re.split(r"(?i)\b(summary|skills|projects|experience|education|courses|languages)\b", content)[0]
                entries = [
                    part.strip()
                    for part in re.split(r"\s*[;|]\s*|\s+-\s+", next_cut)
                    if part.strip()
                ]
                if section in sections and entries:
                    sections[section].extend(entries)
                idx += 2

    return sections


def _infer_multi_domain_flag(resume_text: str) -> int:
    text = resume_text.lower()
    matched_domains = 0
    for keywords in DOMAIN_KEYWORDS.values():
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= 2:
            matched_domains += 1

    return 1 if matched_domains >= 2 else 0


def extract_ml_features(resume_text: str, resume_skills: set[str]) -> Dict[str, float | int]:
    quality = analyze_resume_quality(resume_text)
    sections = _extract_section_entries(resume_text)

    num_skills = len(resume_skills)
    num_projects = len(sections.get("projects", []))
    num_experiences = len(sections.get("experience", []))
    num_courses = len(sections.get("courses", []))

    languages_from_section = {
        item.strip().lower()
        for item in sections.get("languages", [])
        if item.strip()
    }
    languages_from_text = {
        lang
        for lang in PROGRAMMING_LANGUAGES
        if re.search(rf"\b{re.escape(lang)}\b", resume_text, flags=re.IGNORECASE)
    }
    num_languages = len(languages_from_section.intersection(PROGRAMMING_LANGUAGES).union(languages_from_text))

    # Prefer parser-derived word count so quality and ML share a consistent text normalization base.
    word_count = int(quality.get("word_count", 0))
    avg_sentence_length = _avg_sentence_length(resume_text)

    section_presence = [
        bool(sections.get("skills")),
        bool(sections.get("projects")),
        bool(sections.get("experience")),
        bool(sections.get("education")),
        bool(sections.get("courses")),
        bool(sections.get("languages")),
        bool(sections.get("summary")),
    ]
    section_count = int(sum(section_presence))

    multi_domain_flag = _infer_multi_domain_flag(resume_text)

    metric_count = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", resume_text))
    metric_ratio = round(metric_count / max(word_count, 1), 4)

    bullets: list[str] = []
    for entries in sections.values():
        bullets.extend(entries)
    if not bullets:
        bullets = [
            re.sub(r"^[-*\u2022\s]+", "", line).strip()
            for line in resume_text.splitlines()
            if re.match(r"^\s*[-*\u2022]", line)
        ]
    bullet_lengths = [len(re.findall(r"\b\w+\b", item)) for item in bullets if item]
    avg_bullet_length = round(sum(bullet_lengths) / len(bullet_lengths), 2) if bullet_lengths else 0.0

    skill_density = round(num_skills / max(word_count, 1), 4)
    has_summary = 1 if sections.get("summary") else 0
    has_projects = 1 if sections.get("projects") else 0
    has_experience = 1 if sections.get("experience") else 0

    return {
        "num_skills": num_skills,
        "num_projects": num_projects,
        "num_experiences": num_experiences,
        "num_courses": num_courses,
        "num_languages": num_languages,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "section_count": section_count,
        "multi_domain_flag": multi_domain_flag,
        "skill_density": skill_density,
        "metric_count": metric_count,
        "metric_ratio": metric_ratio,
        "avg_bullet_length": avg_bullet_length,
        "has_summary": has_summary,
        "has_projects": has_projects,
        "has_experience": has_experience,
    }


ensure_artifacts()
SKILLS_SET = load_skills(SKILLS_OUTPUT_PATH)
JOB_LIBRARY_INDEX = _build_job_library_index()
load_trained_models()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/job_descriptions")
def job_descriptions():
    return jsonify({"items": JOB_DESCRIPTION_LIBRARY})


@app.post("/upload_resume")
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file provided."}), 400

    file = request.files["resume"]
    if not file.filename:
        return jsonify({"error": "Missing file name."}), 400

    extension = Path(file.filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".txt"}:
        return jsonify({"error": "Unsupported format. Use PDF, DOCX, or TXT."}), 400

    resume_id = str(uuid.uuid4())
    target_path = UPLOAD_DIR / f"{resume_id}{extension}"
    file.save(target_path)

    try:
        text = read_text_from_file(target_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to parse resume: {exc}"}), 400

    if not text.strip():
        return jsonify({"error": "Uploaded resume is empty or unreadable."}), 400

    resume_store[resume_id] = {
        "file_name": file.filename,
        "text": text,
    }

    return jsonify(
        {
            "resume_id": resume_id,
            "file_name": file.filename,
            "characters": len(text),
        }
    )


@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    resume_id = payload.get("resume_id")
    resume_text = payload.get("resume_text", "")
    job_description = payload.get("job_description", "")

    if resume_id and resume_id in resume_store:
        resume_text = resume_store[resume_id]["text"]

    if not resume_text.strip():
        return jsonify({"error": "Resume text is required (upload first or pass resume_text)."}), 400

    if not job_description.strip():
        return jsonify({"error": "Job description is required."}), 400

    similarity_score = semantic_matcher.semantic_similarity(resume_text, job_description)

    resume_skills = set(_sanitize_skill_terms(extract_skills(resume_text, SKILLS_SET)))
    jd_skills = set(_sanitize_skill_terms(extract_skills(job_description, SKILLS_SET)))

    matched_skills = sorted(resume_skills.intersection(jd_skills))
    missing_skills = sorted(jd_skills.difference(resume_skills))

    quality_analysis = analyze_resume_quality(resume_text)
    ml_features = extract_ml_features(resume_text, resume_skills)
    model_input = pd.DataFrame(
        [
            {
                ML_TEXT_FEATURE: resume_text,
                **ml_features,
            }
        ]
    )

    transformed_input = preprocessor.transform(model_input)

    final_score = int(round(float(rf_regressor.predict(transformed_input)[0])))

    quality_pred_idx = quality_classifier.predict(transformed_input)
    quality_label = str(quality_label_encoder.inverse_transform(quality_pred_idx)[0])

    domain_pred_idx = domain_classifier.predict(transformed_input)
    domain_label = str(domain_label_encoder.inverse_transform(domain_pred_idx)[0])

    if quality_calibrator:
        quality_probabilities = quality_calibrator.predict_proba(transformed_input)[0].tolist()
    else:
        quality_probabilities = quality_classifier.predict_proba(transformed_input)[0].tolist()

    if domain_calibrator:
        domain_probabilities = domain_calibrator.predict_proba(transformed_input)[0].tolist()
    else:
        domain_probabilities = domain_classifier.predict_proba(transformed_input)[0].tolist()
    quality_confidence = _calibrated_confidence(quality_probabilities)
    domain_confidence = _calibrated_confidence(domain_probabilities)
    quality_flags = _confidence_flags(quality_probabilities, confidence_thresholds.get("quality", {}))
    domain_flags = _confidence_flags(domain_probabilities, confidence_thresholds.get("domain", {}))
    quality_top2 = _top_k_predictions(quality_probabilities, quality_label_encoder, k=2)
    domain_top2 = _top_k_predictions(domain_probabilities, domain_label_encoder, k=2)
    skill_match_ratio = calculate_skill_match_ratio(resume_skills, jd_skills)
    overall_score = calculate_overall_score(
        semantic_similarity=similarity_score,
        skill_match_ratio=skill_match_ratio,
        quality_score=float(quality_analysis.get("quality_score", 0.0)),
    )

    jd_domain = _infer_jd_domain(job_description)
    job_recommendations = _rank_job_recommendations(resume_text, resume_skills, domain_label)
    best_job = job_recommendations[0] if job_recommendations else {}
    recommendation_confidence = float(best_job.get("confidence", 0.0)) if best_job else 0.0
    recommendation_score = float(best_job.get("score", 0.0)) if best_job else 0.0

    llm_feedback_input = {
        "final_score": final_score,
        "quality": quality_label,
        "predicted_domain": domain_label,
        "jd_domain": jd_domain,
        "job_description": job_description,
        "jd_required_skills": sorted(jd_skills),
        "num_skills": int(ml_features.get("num_skills", 0)),
        "num_projects": int(ml_features.get("num_projects", 0)),
        "num_experiences": int(ml_features.get("num_experiences", 0)),
        "num_courses": int(ml_features.get("num_courses", 0)),
        "num_languages": int(ml_features.get("num_languages", 0)),
        "missing_skills": missing_skills,
        "matched_skills": matched_skills,
        "word_count": int(ml_features.get("word_count", 0)),
        "format_flow_score": int(quality_analysis.get("format_flow_score", 0) or 0),
        "section_sequence": quality_analysis.get("section_sequence", []),
        "flow_feedback": quality_analysis.get("flow_feedback", ""),
        "best_job_title": best_job.get("title", ""),
        "best_job_domain": best_job.get("domain", ""),
        "best_job_score": recommendation_score,
        "best_job_confidence": recommendation_confidence,
    }

    example_domain = jd_domain or domain_label
    project_examples, course_examples = _examples_for_domain(example_domain)
    llm_feedback_input["project_examples"] = project_examples
    llm_feedback_input["course_examples"] = course_examples

    # Ollama-only mode: predefined rule-based feedback is disabled.
    llm_feedback = generate_llm_feedback(llm_feedback_input)
    if not llm_feedback:
        return (
            jsonify(
                {
                    "error": "Ollama feedback generation failed.",
                    "details": LAST_OLLAMA_ERROR or "No valid Ollama response received.",
                }
            ),
            503,
        )

    strengths = llm_feedback["strengths"]
    weaknesses = llm_feedback["weaknesses"]
    suggestions = llm_feedback["suggestions"]

    analysis_id = str(uuid.uuid4())
    result = {
        "analysis_id": analysis_id,
        "score": final_score,
        "final_score": final_score,
        "similarity_score": round(similarity_score, 4),
        "skill_match_ratio": round(skill_match_ratio, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "quality_label": quality_label,
        "quality": quality_label,
        "predicted_domain": domain_label,
        "feedback_source": "ollama",
        "confidence": {
            "quality": round(quality_confidence, 4),
            "domain": round(domain_confidence, 4),
            "recommendation": round(recommendation_confidence, 4),
        },
        "uncertainty": {
            "quality_low_confidence": bool(quality_flags["low_confidence"]),
            "domain_low_confidence": bool(domain_flags["low_confidence"]),
            "recommendation_low_confidence": recommendation_confidence < 0.5,
            "quality_margin": round(float(quality_flags["margin"]), 4),
            "domain_margin": round(float(domain_flags["margin"]), 4),
        },
        "top2_predictions": {
            "quality": quality_top2,
            "domain": domain_top2,
        },
        "predicted_category": domain_label,
        "category_confidence": round(domain_confidence, 4),
        "best_suitable_job": {
            "title": best_job.get("title", "Unknown"),
            "domain": best_job.get("domain", "Unknown"),
            "description": best_job.get("description", ""),
            "score": round(recommendation_score, 4),
            "confidence": round(recommendation_confidence, 4),
            "reason": best_job.get("reason", ""),
            "matched_skills": best_job.get("matched_skills", []),
            "missing_skills": best_job.get("missing_skills", []),
        },
        "job_recommendations": job_recommendations[:5],
        "quality_details": {
            "quality_score": quality_analysis.get("quality_score", 0),
            "format_flow_score": quality_analysis.get("format_flow_score", 0),
            "word_count": quality_analysis.get("word_count", 0),
            "sections_found": quality_analysis.get("sections_found", []),
            "section_sequence": quality_analysis.get("section_sequence", []),
            "flow_feedback": quality_analysis.get("flow_feedback", ""),
            "predicted_quality_label": quality_label,
            "ml_features": ml_features,
        },
        "legacy_overall_score": overall_score,
    }

    analysis_store[analysis_id] = result
    return jsonify(result)


@app.get("/results")
def results():
    analysis_id = request.args.get("analysis_id")

    if analysis_id:
        payload = analysis_store.get(analysis_id)
        if not payload:
            return jsonify({"error": "analysis_id not found."}), 404
        return jsonify(payload)

    if not analysis_store:
        return jsonify({"error": "No analysis has been performed yet."}), 404

    latest_id = next(reversed(analysis_store.keys()))
    return jsonify(analysis_store[latest_id])


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "skills_loaded": len(SKILLS_SET),
            "dataset_root": str(DATASET_ROOT),
            "ml_models_dir": str(TRAINED_MODELS_DIR),
            "ml_models_loaded": all(
                item is not None
                for item in [
                    rf_regressor,
                    quality_classifier,
                    domain_classifier,
                    preprocessor,
                    quality_label_encoder,
                    domain_label_encoder,
                ]
            ),
            "ollama_api_url": OLLAMA_API_URL,
            "ollama_model": OLLAMA_MODEL,
            "ollama_timeout_seconds": OLLAMA_TIMEOUT_SECONDS,
            "last_ollama_error": LAST_OLLAMA_ERROR,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
