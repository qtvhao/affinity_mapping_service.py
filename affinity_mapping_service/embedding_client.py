"""
Embedding Client for OpenAI API

A dedicated client for generating text embeddings using OpenAI's embedding models.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EmbeddingClient:
    """Client for generating text embeddings via OpenAI API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large"
    ):
        """
        Initialize the Embedding Client.

        Args:
            api_key: OpenAI API key (optional, loaded from OPENAI_API_KEY env var)
            model: Embedding model identifier (default: text-embedding-3-large)
        """
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            raise ValueError(
                "API key is required. Provide it via the api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            list[float]: Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a batch.

        Args:
            texts: List of texts to embed

        Returns:
            list[list[float]]: List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [item.embedding for item in response.data]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.

        Returns:
            int: Embedding dimension
        """
        # Generate a test embedding to determine dimension
        test_embedding = self.embed("test")
        return len(test_embedding)


def create_openai_embedding_client(
    api_key: str | None = None,
    model: str = "text-embedding-3-large"
) -> EmbeddingClient:
    """
    Factory function to create an OpenAI embedding client.

    Args:
        api_key: OpenAI API key (optional, loaded from env)
        model: Embedding model identifier (default: text-embedding-3-large)

    Returns:
        EmbeddingClient: Configured embedding client instance
    """
    return EmbeddingClient(api_key=api_key, model=model)


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
