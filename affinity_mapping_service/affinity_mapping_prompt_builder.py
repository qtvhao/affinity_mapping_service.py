"""
Affinity Mapping Prompt Builder

Constructs prompts, tool definitions, and tool result lookups for the
agentic affinity mapping loop.

Pattern from scaffold's FeatureDefinitionPromptBuilder — parameterized
read_document tool with enum of document IDs, plus a save_affinity_mapping
tool that captures the final markdown output.
"""

import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an expert in Domain-Driven Design and affinity mapping techniques.

## How You Work
1. Read ALL available documents by calling read_document once for EACH document listed in the enum.
2. Think between each read — identify recurring themes, shared concepts, and natural groupings.
3. After reading ALL documents, call save_affinity_mapping exactly once with the complete markdown content.

IMPORTANT — Tool Use Rules:
1. Call read_document once for EACH document ID in the enum — do NOT skip any.
2. Think between tool calls: build up your understanding of themes incrementally.
3. Call save_affinity_mapping exactly ONCE with the complete, final markdown.
4. Do NOT produce free-form text output outside of tool calls.

## Output Format (CRITICAL — must match exactly for downstream parsing)

Your output MUST use this exact structure:

### Cluster 1: [Cluster Name]
**Related Concepts:**
- [Concept 1]
- [Concept 2]
- [etc.]

**Key Relationships:**
[Describe relationships between concepts in this cluster]

### Cluster 2: [Cluster Name]
**Related Concepts:**
- [Concept 1]
- [Concept 2]
- [etc.]

**Key Relationships:**
[Describe relationships between concepts in this cluster]

[Continue with additional clusters...]

## Requirements
1. Create a MINIMUM of 12 clusters to ensure granular and comprehensive coverage.
2. Group related concepts into numbered clusters (Cluster 1, Cluster 2, etc.).
3. Each cluster should have a descriptive name focusing on a specific aspect.
4. Break down broad topics into more specific, focused clusters rather than combining them.
5. List related concepts under each cluster.
6. Describe the key relationships between concepts within each cluster.
7. Aim for clusters that are specific and well-defined rather than broad categories.
8. Provide ONLY the affinity mapping with grouped concept clusters. Start directly with the clusters.
9. Do NOT include any preamble, summary, or conclusion — ONLY the cluster sections."""


def build_system_prompt() -> list[dict]:
    """Return system prompt as content block array with cache_control."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


USER_PROMPT = (
    "Read ALL documents by calling read_document once for EACH document ID, "
    "then call save_affinity_mapping exactly once with the complete markdown content."
)


def build_tools(documents: list[str]) -> list[dict]:
    """
    Build tool definitions for the agentic affinity mapping loop.

    Args:
        documents: List of document contents (used to build enum of IDs)

    Returns:
        List of Anthropic tool definitions: read_document + save_affinity_mapping
    """
    doc_ids = [f"document_{i+1}" for i in range(len(documents))]

    tools = [
        {
            "name": "read_document",
            "description": (
                "Read a discovery document for analysis. "
                "Call this once for EACH document listed in the enum. "
                f"Available documents: {', '.join(doc_ids)}"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID of the document to read",
                        "enum": doc_ids,
                    }
                },
                "required": ["document_id"],
            },
        },
        {
            "name": "save_affinity_mapping",
            "description": (
                "Save the complete affinity mapping. Call this exactly once "
                "after reading ALL documents. The content must be the complete "
                "markdown with all clusters."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "markdown_content": {
                        "type": "string",
                        "description": "Complete markdown content with all affinity mapping clusters",
                    }
                },
                "required": ["markdown_content"],
            },
        },
    ]

    return tools


def build_tool_results(documents: list[str]) -> dict[str, dict[str, str]]:
    """
    Build tool results lookup mapping document IDs to content.

    Args:
        documents: List of document contents

    Returns:
        Dict with read_document -> {document_1: content, document_2: content, ...}
    """
    doc_lookup = {}
    for i, doc in enumerate(documents):
        doc_lookup[f"document_{i+1}"] = doc

    return {
        "read_document": doc_lookup,
    }
