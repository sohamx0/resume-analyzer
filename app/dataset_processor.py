from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import spacy

from parser import clean_text, read_text_from_file


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT_DIR / "data"
DATASET_ROOT = DATA_ROOT / "data"
SKILLS_OUTPUT_PATH = DATA_ROOT / "skills.txt"
CATEGORY_PROFILE_PATH = DATA_ROOT / "category_profiles.json"


def _load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


_NLP = _load_nlp()


def iter_resume_files(dataset_root: Path) -> Iterable[Tuple[str, Path]]:
    for category_dir in dataset_root.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name.upper()
        if category == "RESUME":
            continue

        for file_path in category_dir.rglob("*"):
            if file_path.suffix.lower() in {".pdf", ".docx", ".txt"}:
                yield category, file_path


def iter_resume_csv_rows(root_dir: Path, dataset_root: Path) -> Iterable[Tuple[str, str]]:
    candidate_paths = [
        dataset_root / "Resume" / "Resume.csv",
        root_dir / "Resume" / "Resume.csv",
    ]

    existing_paths = [path for path in candidate_paths if path.exists()]
    for csv_path in existing_paths:
        with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue

                raw_category = (
                    row.get("Category")
                    or row.get("category")
                    or row.get("Job Category")
                    or "CSV_RESUME"
                )
                category = str(raw_category).strip().upper() or "CSV_RESUME"

                text_candidates = [
                    row.get("Resume"),
                    row.get("resume"),
                    row.get("Resume_str"),
                    row.get("resume_str"),
                    row.get("Text"),
                    row.get("text"),
                ]
                text = next((value for value in text_candidates if value), "")
                if not text:
                    text = " ".join(str(v) for v in row.values() if v)

                cleaned = clean_text(text)
                if cleaned:
                    yield category, cleaned


def _normalize_term(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9+.#/\- ]", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _expand_term_forms(token: str) -> List[str]:
    normalized = _normalize_term(token)
    if not normalized:
        return []

    forms = {normalized}
    if "/" in normalized:
        forms.add(normalized.replace("/", " "))
    if "-" in normalized:
        forms.add(normalized.replace("-", " "))
    return [form for form in forms if form]


def extract_terms(text: str) -> List[str]:
    normalized = clean_text(text)
    doc = _NLP(normalized)
    terms: List[str] = []

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            lemma = token.lemma_ if token.lemma_ else token.text
            for form in _expand_term_forms(lemma):
                if len(form) >= 2:
                    terms.append(form)

    try:
        for chunk in doc.noun_chunks:
            for form in _expand_term_forms(chunk.text):
                if len(form) >= 2:
                    terms.append(form)
    except (AttributeError, ValueError):
        pass

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "LANGUAGE", "EVENT", "WORK_OF_ART"}:
            for form in _expand_term_forms(ent.text):
                if len(form) >= 2:
                    terms.append(form)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#/\-]{1,}", normalized):
        terms.extend(_expand_term_forms(token))

    for acronym in re.findall(r"\b[A-Z]{2,6}\b", text):
        terms.extend(_expand_term_forms(acronym))

    return terms


def process_dataset(
    dataset_root: Path = DATASET_ROOT,
    skills_output_path: Path = SKILLS_OUTPUT_PATH,
    category_profile_path: Path = CATEGORY_PROFILE_PATH,
    min_frequency: int = 8,
    max_skills: int = 2000,
) -> Dict[str, object]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    global_counter = Counter()
    category_counters = defaultdict(Counter)
    processed_files = 0
    failed_files = 0

    for category, file_path in iter_resume_files(dataset_root):
        try:
            text = read_text_from_file(file_path)
            if not text.strip():
                continue
            terms = extract_terms(text)
            if not terms:
                continue

            global_counter.update(terms)
            category_counters[category].update(terms)
            processed_files += 1
        except Exception:
            failed_files += 1

    for category, text in iter_resume_csv_rows(ROOT_DIR, dataset_root):
        try:
            terms = extract_terms(text)
            if not terms:
                continue

            global_counter.update(terms)
            category_counters[category].update(terms)
            processed_files += 1
        except Exception:
            failed_files += 1

    frequent_skills = [
        term
        for term, freq in global_counter.most_common(max_skills)
        if freq >= min_frequency and len(term) >= 2
    ]

    skills_output_path.parent.mkdir(parents=True, exist_ok=True)
    skills_output_path.write_text("\n".join(sorted(set(frequent_skills))), encoding="utf-8")

    category_profiles = {}
    for category, counter in category_counters.items():
        top_terms = [term for term, _ in counter.most_common(120)]
        category_profiles[category] = top_terms

    category_profile_path.write_text(json.dumps(category_profiles, indent=2), encoding="utf-8")

    return {
        "processed_files": processed_files,
        "failed_files": failed_files,
        "skills_count": len(set(frequent_skills)),
        "skills_path": str(skills_output_path),
        "category_profile_path": str(category_profile_path),
    }


if __name__ == "__main__":
    summary = process_dataset()
    print(json.dumps(summary, indent=2))
