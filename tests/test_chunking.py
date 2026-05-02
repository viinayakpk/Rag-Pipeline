from app.services import chunking


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


def test_chunk_text_uses_token_based_limits(monkeypatch) -> None:
    monkeypatch.setattr(chunking, "_get_tokenizer", lambda: FakeTokenizer())

    text = " ".join(f"token{i}" for i in range(700))
    chunks = chunking.chunk_text(text)

    assert len(chunks) > 1
    assert all(chunk.token_count <= chunking.CHUNK_SIZE for chunk in chunks)
    assert chunks[0].chunk_index == 0
