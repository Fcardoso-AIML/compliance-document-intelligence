from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 60) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def to_chunk_records(doc: Dict, chunk_size: int = 350, overlap: int = 60) -> List[Dict]:
    out = []
    for idx, chunk in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)):
        out.append({
            "doc_id": doc["doc_id"],
            "chunk_id": f"{doc['doc_id']}_c{idx}",
            "text": chunk,
            "labels": doc.get("labels", []),
        })
    return out
