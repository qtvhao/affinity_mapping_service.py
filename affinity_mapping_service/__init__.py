"""
Affinity Mapping Service Package
"""

from affinity_mapping_service.service import (
    AffinityMappingService,
    AffinityMappingSession,
    __version__,
    main,
)
from affinity_mapping_service.llm_client import (
    LLMClient,
    create_tenant_llm_client,
)
from affinity_mapping_service.embedding_client import (
    EmbeddingClient,
    EmbeddingStore,
    create_tenant_llm_embedding_client,
)

__all__ = [
    # Service classes
    "AffinityMappingService",
    "AffinityMappingSession",
    # LLM client
    "LLMClient",
    "create_tenant_llm_client",
    # Embedding client
    "EmbeddingClient",
    "EmbeddingStore",
    "create_tenant_llm_embedding_client",
    # Package metadata
    "__version__",
    "main",
]
