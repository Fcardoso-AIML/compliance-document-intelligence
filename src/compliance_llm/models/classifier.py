from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class ClassifierArtifacts:
    vectorizer: TfidfVectorizer
    mlb: MultiLabelBinarizer
    clf: OneVsRestClassifier


class ComplianceClassifier:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
        self.mlb = MultiLabelBinarizer()
        base = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.clf = OneVsRestClassifier(base)

    def fit(self, texts: List[str], labels: List[List[str]]) -> None:
        x = self.vectorizer.fit_transform(texts)
        y = self.mlb.fit_transform(labels)
        self.clf.fit(x, y)

    def predict(self, text: str, threshold: float = 0.35) -> Dict[str, float]:
        x = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(x)[0]
        return {label: float(p) for label, p in zip(self.mlb.classes_, probs) if p >= threshold}

    def save(self, path: str) -> None:
        joblib.dump(ClassifierArtifacts(self.vectorizer, self.mlb, self.clf), path)

    @classmethod
    def load(cls, path: str) -> "ComplianceClassifier":
        obj = cls()
        art: ClassifierArtifacts = joblib.load(path)
        obj.vectorizer = art.vectorizer
        obj.mlb = art.mlb
        obj.clf = art.clf
        return obj
