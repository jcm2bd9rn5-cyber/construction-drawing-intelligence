"""Gemini embeddings with separate document and query tasks."""
from collections.abc import Sequence
from google.genai import types
from app.core.config import Settings, get_settings
from app.ai.client import call_with_retry, create_client

class GeminiEmbedder:
    def __init__(self, client=None, settings: Settings | None = None):
        settings = settings or get_settings()
        self.settings = settings
        if not settings.gemini_api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required for embeddings")
        self.client = client or create_client(settings)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        if self.batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        self.embedding_api_requests = 0
        self.embedding_batches = 0

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []
        self.embedding_batches += 1

        def operation():
            self.embedding_api_requests += 1
            return self.client.models.embed_content(
                model=self.model,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dimensions
                ),
            )

        result = call_with_retry(
            operation, f"embedding ({task_type})", settings=self.settings
        )
        embeddings = list(result.embeddings or [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: requested {len(texts)}, "
                f"received {len(embeddings)}"
            )
        vectors = []
        for embedding in embeddings:
            values = list(embedding.values)
            if len(values) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} embedding dimensions; got {len(values)}"
                )
            vectors.append(values)
        return vectors

    def _embed(self, text: str, task_type: str) -> list[float]:
        return self._embed_batch([text], task_type)[0]

    def document(self, content: str, title: str = "") -> list[float]:
        return self.documents([content], title=title)[0]

    def documents(self, contents: Sequence[str], title: str = "") -> list[list[float]]:
        """Embed document chunks in bounded API batches while preserving order."""
        prepared = [f"{title}\n{content}" for content in contents]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            vectors.extend(
                self._embed_batch(
                    prepared[start : start + self.batch_size], "RETRIEVAL_DOCUMENT"
                )
            )
        return vectors

    def query(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")

def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_a = sum(a * a for a in left) ** 0.5
    norm_b = sum(b * b for b in right) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
