from compliance_llm.data.chunking import chunk_text


def test_chunking_basic():
    txt = "word " * 800
    chunks = chunk_text(txt, chunk_size=200, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c.split()) <= 200 for c in chunks)
