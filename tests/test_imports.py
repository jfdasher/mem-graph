def test_memgraph_imports() -> None:
    from memgraph import LocalEmbedder, OpenAIEmbedder, create_graph_schema, create_schema
    from memgraph.chunker import word_chunks
    from memgraph.embedders import Embedder
    from memgraph.fusion import rrf
    from memgraph.parsers import parse

    assert LocalEmbedder is not None
    assert word_chunks("hello world", chunk_size=2, overlap=0) == ["hello world"]
    assert rrf([]) == {}
