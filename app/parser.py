from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set

import spacy
from docx import Document
from pypdf import PdfReader


_SECTION_HEADERS = {"skills", "education", "projects", "experience", "summary"}
_FLOW_SECTIONS = ["summary", "skills", "experience", "projects", "education", "courses", "languages"]
_SECTION_HEADERS = set(_FLOW_SECTIONS)


def _load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp


_NLP = _load_nlp()


def read_text_from_file(file_path: str | Path) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_sections(text: str) -> Set[str]:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    found = set()
    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line)
        if normalized in _SECTION_HEADERS:
            found.add(normalized)
    return found


def extract_section_sequence(text: str) -> List[str]:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    sequence: List[str] = []
    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line).strip()
        if normalized in _SECTION_HEADERS:
            if not sequence or sequence[-1] != normalized:
                sequence.append(normalized)
    return sequence


def calculate_format_flow_score(section_sequence: List[str]) -> tuple[int, str]:
    if not section_sequence:
        return 25, "Section flow could not be detected clearly."

    expected_index = {name: idx for idx, name in enumerate(_FLOW_SECTIONS)}
    order_hits = 0
    for i in range(1, len(section_sequence)):
        if expected_index.get(section_sequence[i], -1) >= expected_index.get(section_sequence[i - 1], -1):
            order_hits += 1

    if len(section_sequence) <= 1:
        order_ratio = 0.6
    else:
        order_ratio = order_hits / (len(section_sequence) - 1)

    header_bonus = min(1.0, len(set(section_sequence)) / len(_FLOW_SECTIONS))
    score = int(round((order_ratio * 70) + (header_bonus * 30)))

    if score >= 75:
        feedback = "Section ordering and document flow are clear and professional."
    elif score >= 50:
        feedback = "Section flow is acceptable but could be ordered more consistently."
    else:
        feedback = "Section flow appears inconsistent; reorganize sections in a standard order."

    return score, feedback


def _sentence_lengths(text: str) -> List[int]:
    doc = _NLP(clean_text(text))
    lengths = []
    for sent in doc.sents:
        words = [token for token in sent if token.is_alpha]
        if words:
            lengths.append(len(words))
    return lengths


def analyze_resume_quality(text: str) -> Dict[str, object]:
    normalized = clean_text(text)
    words = re.findall(r"[a-zA-Z]{2,}", normalized.lower())
    word_count = len(words)

    length_score = 50
    length_feedback = "Resume length is acceptable."
    if word_count < 180:
        length_score = 25
        length_feedback = "Resume looks too short; add more detail on impact and projects."
    elif 180 <= word_count <= 1200:
        length_score = 90
        length_feedback = "Resume length is well balanced for screening."
    elif word_count > 1500:
        length_score = 40
        length_feedback = "Resume is quite long; tighten repetitive sections."

    repetition_score = 80
    repetition_feedback = "Keyword repetition looks healthy."
    if words:
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top_ratio = max(freq.values()) / max(1, word_count)
        if top_ratio > 0.1:
            repetition_score = 20
            repetition_feedback = "Heavy keyword repetition detected."
        elif top_ratio > 0.06:
            repetition_score = 45
            repetition_feedback = "Some keywords are repeated frequently; diversify wording."

    sections_found = extract_sections(text)
    section_sequence = extract_section_sequence(text)
    required_sections = {"skills", "education", "projects"}
    present_count = len(required_sections.intersection(sections_found))
    section_score = int((present_count / len(required_sections)) * 100)
    format_flow_score, flow_feedback = calculate_format_flow_score(section_sequence)

    sentence_lengths = _sentence_lengths(text)
    clarity_score = 70
    clarity_feedback = "Sentence clarity is generally good."
    if sentence_lengths:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        if avg_len < 6:
            clarity_score = 55
            clarity_feedback = "Many sentences are very short; add context for achievements."
        elif avg_len > 28:
            clarity_score = 50
            clarity_feedback = "Sentences are long; consider shorter, clearer bullet points."
        else:
            clarity_score = 88

    quality_score = round(
        (length_score * 0.25)
        + (repetition_score * 0.15)
        + (section_score * 0.25)
        + (clarity_score * 0.20)
        + (format_flow_score * 0.15),
        2,
    )

    strengths = []
    weaknesses = []

    if length_score >= 80:
        strengths.append("Resume has a practical level of detail.")
    else:
        weaknesses.append(length_feedback)

    if repetition_score >= 70:
        strengths.append("Keyword usage appears balanced.")
    else:
        weaknesses.append(repetition_feedback)

    if section_score >= 67:
        strengths.append("Core sections (skills, education, projects) are present.")
    else:
        weaknesses.append("Add clear Skills, Education, and Projects sections.")

    if clarity_score >= 80:
        strengths.append("Sentence structure is readable for recruiters.")
    else:
        weaknesses.append(clarity_feedback)

    if format_flow_score >= 75:
        strengths.append("Resume sections follow a clear and logical flow.")
    else:
        weaknesses.append(flow_feedback)

    return {
        "quality_score": quality_score,
        "word_count": word_count,
        "sections_found": sorted(sections_found),
        "length_feedback": length_feedback,
        "repetition_feedback": repetition_feedback,
        "clarity_feedback": clarity_feedback,
        "flow_feedback": flow_feedback,
        "format_flow_score": format_flow_score,
        "section_sequence": section_sequence,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }
