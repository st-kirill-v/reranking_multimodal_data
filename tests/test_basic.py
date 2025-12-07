def test_import():
    """Test basic imports"""
    try:
        from src.core.rag import SimpleTextRAG
        from src.api.server import app

        assert True
    except ImportError as e:
        assert False, f"Import error: {e}"


def test_rag_creation():
    """Test RAG instance creation"""
    from src.core.rag import SimpleTextRAG

    rag = SimpleTextRAG()
    assert rag is not None
    assert rag.documents == []
    assert rag.is_fitted == False
