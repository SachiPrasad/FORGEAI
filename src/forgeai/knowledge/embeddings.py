"""Embedding generators (LiteLLM cloud/local API + zero-dependency offline semantic embedder)."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from forgeai.intake.llm_client import has_configured_llm_credentials
from forgeai.knowledge.schemas import EmbeddingProvider


class Embedder(Protocol):
    """Protocol implemented by all embedding backends."""

    provider_name: str
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalSemanticEmbedder:
    """Deterministic local dense embedder using word, bigram, and char-ngram signed hashing.

    Produces L2-normalized dense float32 vectors with zero external model downloads,
    ideal for offline execution, reproducible tests, and low-latency local retrieval.
    """

    provider_name = "local"

    def __init__(self, dimension: int = 384, model_name: str = "local-hash-ngram-v1") -> None:
        self.dimension = dimension
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec.tolist()

        # Unigrams (weight 2.0) + stemmed/prefix forms (weight 1.0)
        for tok in tokens:
            self._add_feature(vec, f"w:{tok}", weight=2.0)
            if len(tok) > 4:
                self._add_feature(vec, f"s:{tok[:4]}", weight=0.9)
            # Character 3-grams for morphological similarity
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    self._add_feature(vec, f"c:{tok[i : i + 3]}", weight=0.45)

        # Word bigrams for phrase affinity (e.g., "annual leave", "health insurance")
        for i in range(len(tokens) - 1):
            self._add_feature(vec, f"b:{tokens[i]}_{tokens[i + 1]}", weight=1.6)

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec.astype(float).tolist()

    def _add_feature(self, vec: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(digest, "little")
        idx = val % self.dimension
        sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
        vec[idx] += sign * weight


class LiteLLMEmbedder:
    """Provider-agnostic embedding backend powered by LiteLLM."""

    provider_name = "litellm"

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from litellm import embedding

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = embedding(model=self.model_name, input=batch)
            for item in response.data:
                vec = np.asarray(item["embedding"], dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0.0:
                    vec /= norm
                if self.dimension == 0:
                    self.dimension = int(vec.shape[0])
                all_vectors.append(vec.astype(float).tolist())
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


def create_embedder(
    provider: EmbeddingProvider = "auto",
    model: str | None = None,
    dimension: int = 384,
) -> Embedder:
    """Instantiate an embedding backend based on explicit choice or available API credentials."""
    resolved_model = model or "text-embedding-3-small"
    if provider == "local":
        return LocalSemanticEmbedder(dimension=dimension)
    if provider == "litellm":
        return LiteLLMEmbedder(model_name=resolved_model)

    # Auto mode: use LiteLLM if credentials/Ollama are configured, else local semantic embedder
    if has_configured_llm_credentials(resolved_model):
        return LiteLLMEmbedder(model_name=resolved_model)
    return LocalSemanticEmbedder(dimension=dimension)

