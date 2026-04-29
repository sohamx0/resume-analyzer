from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from sentence_transformers import SentenceTransformer, util

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None


class SemanticMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def semantic_similarity(self, resume_text: str, job_description: str) -> float:
        embeddings = self.model.encode([resume_text, job_description], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def load_category_profiles(profile_path: str | Path) -> Dict[str, list]:
        path = Path(profile_path)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def predict_category(self, resume_text: str, category_profiles: Dict[str, list]) -> Tuple[str, float]:
        if not category_profiles:
            return "UNKNOWN", 0.0

        categories = sorted(category_profiles.keys())
        profile_texts = [" ".join(category_profiles[cat]) for cat in categories]

        embeddings = self.model.encode([resume_text] + profile_texts, convert_to_tensor=True)
        resume_embedding = embeddings[0]
        profile_embeddings = embeddings[1:]

        scores = util.cos_sim(resume_embedding, profile_embeddings)[0].tolist()
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])

        best_category = categories[best_idx]
        confidence = max(0.0, min(1.0, float(scores[best_idx])))
        return best_category, confidence


class JobReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is unavailable. Ensure sentence-transformers is installed.")
        self.model = CrossEncoder(model_name)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]
