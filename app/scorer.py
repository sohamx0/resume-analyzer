from __future__ import annotations

from typing import Dict, List, Set, Tuple


def calculate_skill_match_ratio(resume_skills: Set[str], jd_skills: Set[str]) -> float:
    if not jd_skills:
        return 0.0
    matched = resume_skills.intersection(jd_skills)
    return len(matched) / len(jd_skills)


def calculate_overall_score(semantic_similarity: float, skill_match_ratio: float, quality_score: float) -> int:
    semantic_component = semantic_similarity * 100 * 0.30
    skill_component = skill_match_ratio * 100 * 0.40
    quality_component = quality_score * 0.30
    return int(round(semantic_component + skill_component + quality_component))


def build_strengths_weaknesses(
    quality_analysis: Dict[str, object],
    semantic_similarity: float,
    skill_match_ratio: float,
    missing_skills: Set[str],
) -> Tuple[List[str], List[str]]:
    strengths = list(quality_analysis.get("strengths", []))
    weaknesses = list(quality_analysis.get("weaknesses", []))

    if semantic_similarity >= 0.75:
        strengths.append("Strong semantic alignment with the job description.")
    elif semantic_similarity < 0.5:
        weaknesses.append("Resume content is not strongly aligned with the role narrative.")

    if skill_match_ratio >= 0.7:
        strengths.append("High overlap between resume and required job skills.")
    elif skill_match_ratio < 0.4:
        weaknesses.append("Skill overlap is low compared to job requirements.")

    if missing_skills:
        weaknesses.append(
            f"Missing important skills: {', '.join(sorted(missing_skills)[:8])}."
        )

    return strengths, weaknesses
