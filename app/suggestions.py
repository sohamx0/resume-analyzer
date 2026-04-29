from __future__ import annotations

from typing import Dict, List, Set


def generate_suggestions(
    missing_skills: Set[str],
    quality_analysis: Dict[str, object],
    semantic_similarity: float,
    skill_match_ratio: float,
) -> List[str]:
    suggestions = []

    if missing_skills:
        top_missing = sorted(missing_skills)[:6]
        suggestions.append(
            f"Add evidence for missing skills in projects or experience: {', '.join(top_missing)}."
        )

    if skill_match_ratio < 0.5:
        suggestions.append("Tailor your resume summary and bullet points to mirror job requirements.")

    if semantic_similarity < 0.6:
        suggestions.append(
            "Rewrite project descriptions using role-specific language and measurable outcomes."
        )

    word_count = int(quality_analysis.get("word_count", 0))
    if word_count < 180:
        suggestions.append("Expand your resume with 2-3 detailed, result-focused project bullets.")
    elif word_count > 1500:
        suggestions.append("Reduce less relevant details and keep impact-focused bullets concise.")

    sections_found = set(quality_analysis.get("sections_found", []))
    for section in ["skills", "education", "projects"]:
        if section not in sections_found:
            suggestions.append(f"Add a clear '{section.title()}' section with structured bullets.")

    suggestions.append("Use metrics in achievements, such as %, revenue impact, or latency reduction.")
    suggestions.append("Mention tools and technologies explicitly in project and experience bullets.")

    deduped = []
    seen = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            deduped.append(suggestion)
            seen.add(suggestion)
    return deduped[:8]
