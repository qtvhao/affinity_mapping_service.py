"""
Embedding Client using Tenant LLM Service

A dedicated client for generating text embeddings via the tenant-llm service proxy.
"""

import os
import httpx
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Tenant LLM Service Configuration
TENANT_LLM_API_BASE_URL = os.getenv("TENANT_LLM_API_BASE_URL", "http://tenant-llm:8009")
CROSS_SERVICE_API_KEY = os.getenv("CROSS_SERVICE_API_KEY", "dev-cross-service-key-change-in-production")


class EmbeddingClient:
    """Client for generating text embeddings via Tenant LLM Service."""

    def __init__(
        self,
        tenant_id: str = "default",
        model_reference: str = "text-embedding-3-large@OpenAI",
        base_url: Optional[str] = None
    ):
        """
        Initialize the Embedding Client.

        Args:
            tenant_id: Tenant identifier (default: "default")
            model_reference: Model reference in format "model@factory" (default: text-embedding-3-large@OpenAI)
            base_url: Base URL for tenant-llm service (optional, uses TENANT_LLM_API_BASE_URL env var)
        """
        self.tenant_id = tenant_id
        self.model_reference = model_reference
        self.base_url = base_url or TENANT_LLM_API_BASE_URL
        self.embeddings_url = f"{self.base_url}/api/v1/openai/embeddings"

    def embed(self, text: str, timeout: int = 30) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            timeout: Request timeout in seconds

        Returns:
            list[float]: Embedding vector
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    self.embeddings_url,
                    headers={
                        "X-API-Key": CROSS_SERVICE_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "tenant_id": self.tenant_id,
                        "model_reference": self.model_reference,
                        "input": text
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["data"][0]["embedding"]
                else:
                    raise RuntimeError(f"Tenant LLM embedding error: {response.status_code} - {response.text}")

        except httpx.TimeoutException:
            raise RuntimeError(f"Tenant LLM embedding request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error calling Tenant LLM embeddings endpoint: {str(e)}")

    def embed_batch(self, texts: list[str], timeout: int = 60) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        Args:
            texts: List of texts to embed
            timeout: Request timeout in seconds

        Returns:
            list[list[float]]: List of embedding vectors
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    self.embeddings_url,
                    headers={
                        "X-API-Key": CROSS_SERVICE_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "tenant_id": self.tenant_id,
                        "model_reference": self.model_reference,
                        "input": texts
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return [item["embedding"] for item in result["data"]]
                else:
                    raise RuntimeError(f"Tenant LLM embedding error: {response.status_code} - {response.text}")

        except httpx.TimeoutException:
            raise RuntimeError(f"Tenant LLM embedding request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error calling Tenant LLM embeddings endpoint: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.

        Returns:
            int: Embedding dimension
        """
        # Generate a test embedding to determine dimension
        test_embedding = self.embed("test")
        return len(test_embedding)


def create_tenant_llm_embedding_client(
    tenant_id: str = "default",
    model_reference: str = "text-embedding-3-large@OpenAI",
    base_url: Optional[str] = None
) -> EmbeddingClient:
    """
    Factory function to create a Tenant LLM embedding client.

    Args:
        tenant_id: Tenant identifier (default: "default")
        model_reference: Model reference in format "model@factory" (default: text-embedding-3-large@OpenAI)
        base_url: Base URL for tenant-llm service (optional)

    Returns:
        EmbeddingClient: Configured embedding client instance
    """
    return EmbeddingClient(tenant_id=tenant_id, model_reference=model_reference, base_url=base_url)


class EmbeddingStore:
    """Simple in-memory store for text embeddings."""

    def __init__(self, embedding_client: EmbeddingClient):
        """
        Initialize the embedding store.

        Args:
            embedding_client: EmbeddingClient instance to use for generating embeddings
        """
        self.embedding_client = embedding_client
        self.texts: list[str] = []
        self.embeddings: list[list[float]] = []

    def add_text(self, text: str) -> int:
        """
        Add a text and generate its embedding.

        Args:
            text: Text to add

        Returns:
            int: Index of the added text
        """
        embedding = self.embedding_client.embed(text)
        self.texts.append(text)
        self.embeddings.append(embedding)
        return len(self.texts) - 1

    def add_texts(self, texts: list[str]) -> list[int]:
        """
        Add multiple texts and generate their embeddings in batch.

        Args:
            texts: List of texts to add

        Returns:
            list[int]: Indices of the added texts
        """
        embeddings = self.embedding_client.embed_batch(texts)
        start_idx = len(self.texts)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        return list(range(start_idx, len(self.texts)))

    def get_embedding(self, index: int) -> list[float]:
        """
        Get embedding by index.

        Args:
            index: Index of the text

        Returns:
            list[float]: Embedding vector
        """
        return self.embeddings[index]

    def get_text(self, index: int) -> str:
        """
        Get text by index.

        Args:
            index: Index of the text

        Returns:
            str: Text content
        """
        return self.texts[index]

    def cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        import math

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def find_similar(self, query: str, top_k: int = 5) -> list[tuple[int, str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            list[tuple[int, str, float]]: List of (index, text, similarity_score) tuples
        """
        if not self.texts:
            return []

        query_embedding = self.embedding_client.embed(query)

        similarities = [
            (i, text, self.cosine_similarity(query_embedding, emb))
            for i, (text, emb) in enumerate(zip(self.texts, self.embeddings))
        ]

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def __len__(self) -> int:
        """Return the number of stored texts."""
        return len(self.texts)
