def test_memgraph_imports() -> None:
    from memgraph import LocalEmbedder
    from memgraph.chunker import word_chunks
    from memgraph.fusion import rrf

    assert LocalEmbedder is not None
    assert word_chunks("hello world", chunk_size=2, overlap=0) == ["hello world"]
    assert rrf([]) == {}
