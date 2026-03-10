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
from affinity_mapping_service.anthropic_llm_client import AnthropicLLMClient
from affinity_mapping_service.agentic_loop import run_agentic_loop

__all__ = [
    # Service classes
    "AffinityMappingService",
    "AffinityMappingSession",
    # LLM clients
    "LLMClient",
    "create_tenant_llm_client",
    "AnthropicLLMClient",
    # Agentic loop
    "run_agentic_loop",
    # Embedding client
    "EmbeddingClient",
    "EmbeddingStore",
    "create_tenant_llm_embedding_client",
    # Package metadata
    "__version__",
    "main",
]
