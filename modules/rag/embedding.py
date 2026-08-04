from collections.abc import Iterable

from config import DEFAULT_EMBEDDING_MODEL


def create_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Create the embedding model lazily to keep application startup lightweight."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model=None):
    """Convert text items into embedding vectors."""
    embedding_model = model or create_embedding_model()
    return embedding_model.encode(list(texts))