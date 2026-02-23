import json
from pathlib import Path
from typing import List, Dict


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(data_dir: Path) -> List[Dict]:
    files = sorted((data_dir / "raw").glob("*.jsonl"))
    corpus = []
    for fp in files:
        corpus.extend(read_jsonl(fp))
    return corpus
