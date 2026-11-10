from types import SimpleNamespace
import pytest
from app.ai.embeddings import GeminiEmbedder, cosine_similarity
from app.ai.client import DailyQuotaExhausted, call_with_retry, quota_details
from app.core.config import Settings

class FakeModels:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.calls = []
    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * self.dimensions)])

def test_gemini_uses_asymmetric_retrieval_tasks(monkeypatch):
    from app.core.config import get_settings
    fake = FakeModels(get_settings().embedding_dimensions)
    embedder = GeminiEmbedder(client=SimpleNamespace(models=fake))
    assert len(embedder.document("本文", "題名")) == 768
    assert len(embedder.query("質問")) == 768
    assert fake.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert fake.calls[1]["config"].task_type == "RETRIEVAL_QUERY"

def test_cosine_similarity_handles_zero_and_orthogonal_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine_similarity([0, 0], [1, 0]) == 0


class BatchFakeModels:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.calls = []

    def embed_content(self, **kwargs):
        contents = kwargs["contents"]
        self.calls.append(kwargs)
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(index)] + [0.0] * (self.dimensions - 1))
                for index, _content in enumerate(contents)
            ]
        )


def test_document_embeddings_are_batched_and_ordered():
    settings = Settings(
        gemini_api_key="test",
        embedding_dimensions=4,
        embedding_batch_size=3,
        gemini_retry_attempts=1,
    )
    fake = BatchFakeModels(4)
    embedder = GeminiEmbedder(
        client=SimpleNamespace(models=fake), settings=settings
    )
    vectors = embedder.documents([f"chunk-{index}" for index in range(7)])

    assert [len(call["contents"]) for call in fake.calls] == [3, 3, 1]
    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0]
    assert embedder.embedding_api_requests == 3
    assert embedder.embedding_batches == 3


def test_query_embedding_is_single_query_request():
    settings = Settings(
        gemini_api_key="test",
        embedding_dimensions=4,
        embedding_batch_size=3,
        gemini_retry_attempts=1,
    )
    fake = BatchFakeModels(4)
    embedder = GeminiEmbedder(
        client=SimpleNamespace(models=fake), settings=settings
    )
    embedder.query("給水設備")
    assert len(fake.calls) == 1
    assert fake.calls[0]["contents"] == ["給水設備"]
    assert fake.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_daily_quota_is_classified_without_short_retries():
    class QuotaError(Exception):
        code = 429

    error = QuotaError(
        "429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
        "generativelanguage.googleapis.com/embed_content_free_tier_requests; "
        "retryDelay: '48330s'"
    )
    details = quota_details(error)
    assert details.daily is True
    assert details.metric.endswith("embed_content_free_tier_requests")
    assert details.retry_after_seconds == pytest.approx(48330)

    sleeps = []
    with pytest.raises(DailyQuotaExhausted) as raised:
        call_with_retry(
            lambda: (_ for _ in ()).throw(error),
            "embedding",
            settings=Settings(gemini_retry_attempts=5),
            sleep=sleeps.append,
            jitter=lambda _left, _right: 0,
        )
    assert raised.value.attempts == 1
    assert sleeps == []
