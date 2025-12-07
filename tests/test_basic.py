def test_import():
    """Test basic imports"""
    try:
        # Импортируем правильный класс
        from src.core.rag import ModularRAG
        assert True
    except ImportError as e:
        assert False, f"Import error: {e}"

def test_rag_creation():
    """Test RAG instance creation"""
    from src.core.rag import ModularRAG
    rag = ModularRAG()
    assert rag is not None
    assert hasattr(rag, 'documents')
    assert hasattr(rag, 'is_fitted')

def test_basic():
    """Simple test"""
    assert 1 + 1 == 2