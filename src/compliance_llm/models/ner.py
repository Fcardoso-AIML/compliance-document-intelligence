import re
from typing import List, Dict

PATTERNS = [
    ("LAW", r"\b(GDPR|EU AI Act|AMLD|FATF|EBA|Basel)\b"),
    ("OBLIGATION", r"\b(must|shall|required|obligation|mandatory)\b"),
    ("RISK", r"\b(risk|breach|violation|penalty|sanction)\b"),
]


def extract_entities(text: str) -> List[Dict[str, str]]:
    entities = []
    for label, pattern in PATTERNS:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append({"text": m.group(0), "label": label})
    seen = set()
    uniq = []
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen:
            seen.add(key)
            uniq.append(ent)
    return uniq
