"""Local embeddings via sentence-transformers.

Local rather than hosted on purpose. The point of this project is a CI gate that
runs the evaluation on every pull request; if embedding needed a paid API key,
that gate would either cost money per PR or get stubbed out -- and a stubbed
gate is a fake gate.

BAAI/bge-small-en-v1.5: 384 dimensions, ~130MB, and it beats MiniLM on the MTEB
retrieval benchmarks at similar cost.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Sequence

import numpy as np

from sift.config import settings

_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    """Load the model once per process. First call downloads ~130MB."""
    from sentence_transformers import SentenceTransformer

    with _load_lock:
        return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
    """Embed corpus text. Returns L2-normalised float32 vectors.

    Normalising here means cosine similarity is a plain dot product, and it lets
    pgvector's `<=>` operator do the least work possible.
    """
    if not texts:
        return np.empty((0, settings.embedding_dim), dtype=np.float32)

    vectors = get_model().encode(
        list(texts),
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a search query.

    The prefix matters. BGE models are trained asymmetrically: queries get an
    instruction prefix, passages do not. Embedding a query as though it were a
    passage costs real retrieval quality for no visible error -- one of those
    bugs that only shows up as "the numbers are a bit worse than the paper".
    """
    vector = get_model().encode(
        settings.query_instruction + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.astype(np.float32)
