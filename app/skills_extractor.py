from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import spacy


def load_skills(skills_path: str | Path) -> Set[str]:
    path = Path(skills_path)
    if not path.exists():
        return set()

    skills = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    }
    return skills


def _load_nlp() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


_NLP = _load_nlp()


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9+.#/\- ]", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _expand_token_forms(token: str) -> Set[str]:
    normalized = _normalize_token(token)
    if not normalized:
        return set()

    forms = {normalized}
    if "/" in normalized:
        forms.add(normalized.replace("/", " "))
    if "-" in normalized:
        forms.add(normalized.replace("-", " "))
    return {form for form in forms if form}


def _candidate_terms(text: str) -> Set[str]:
    terms = set()

    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9+.#/\-]{1,}", text) if w]
    for w in words:
        terms.update(_expand_token_forms(w))

    acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
    for acronym in acronyms:
        terms.update(_expand_token_forms(acronym))

    doc = _NLP(text)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            lemma = token.lemma_ if token.lemma_ else token.text
            for form in _expand_token_forms(lemma):
                if len(form) >= 2:
                    terms.add(form)

    try:
        for chunk in doc.noun_chunks:
            for form in _expand_token_forms(chunk.text):
                if len(form) >= 2:
                    terms.add(form)
    except (AttributeError, ValueError):
        pass

    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "LANGUAGE", "EVENT", "WORK_OF_ART"}:
            for form in _expand_token_forms(ent.text):
                if len(form) >= 2:
                    terms.add(form)

    return terms


def extract_skills(text: str, skills_set: Set[str]) -> Set[str]:
    if not text or not skills_set:
        return set()

    candidates = _candidate_terms(text.lower())
    return {skill for skill in skills_set if skill in candidates}
