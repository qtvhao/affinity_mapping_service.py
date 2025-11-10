"""
Affinity Mapping Service

A service for transforming documents into Bounded Context Specifications
using affinity mapping techniques.
"""

import re
from datetime import datetime
from pathlib import Path
from llm_client import create_deepinfra_client

__version__ = "0.1.0"


class AffinityMappingSession:
    """Represents an affinity mapping session with grouped concepts."""

    def __init__(self, content: str, log_dir: str = "logs"):
        """
        Initialize an Affinity Mapping Session.

        Args:
            content: The generated affinity mapping document
            log_dir: Directory to store log files (default: "logs")
        """
        self.content = content
        self.log_dir = Path(log_dir)
        self.log_file = None
        self._clusters = None

        # Write the affinity mapping document to a log file
        self._write_to_logfile()

    @property
    def clusters(self) -> list[dict]:
        """
        Parse and return clusters from the content.

        Returns:
            List of dictionaries with cluster information:
            [{
                'cluster_name': str,
                'cluster_number': int,
                'related_concepts': str,
                'key_relationships': str
            }]
        """
        if self._clusters is None:
            self._clusters = self._parse_clusters()
        return self._clusters

    def _parse_clusters(self) -> list[dict]:
        """Parse clusters from the generated content."""
        clusters = []

        # Pattern to match cluster sections
        cluster_pattern = r'###\s+Cluster\s+(\d+):\s+(.+?)\n\*\*Related Concepts:\*\*\n(.*?)\n\*\*Key Relationships:\*\*\n(.*?)(?=\n###|\Z)'

        matches = re.findall(cluster_pattern, self.content, re.DOTALL)

        for match in matches:
            cluster_number = int(match[0])
            cluster_name = match[1].strip()
            related_concepts = match[2].strip()
            key_relationships = match[3].strip()

            clusters.append({
                'cluster_name': cluster_name,
                'cluster_number': cluster_number,
                'related_concepts': related_concepts,
                'key_relationships': key_relationships
            })

        return clusters

    def _write_to_logfile(self) -> None:
        """Write the generated affinity mapping document to a log file."""
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"affinity_mapping_{timestamp}.md"

        # Write content to log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("# Affinity Mapping Session Log\n\n")
            f.write(f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(self.content)

        print(f"✓ Affinity mapping document written to: {self.log_file}")


class AffinityMappingService:
    """Main service class for affinity mapping operations."""

    def __init__(self, init_documents: list[str], api_key: str | None = None):
        """
        Initialize the Affinity Mapping Service.

        Args:
            init_documents: List of initial documents to process
            api_key: DeepInfra API key (optional, loaded from .env via DEEPINFRA_API_KEY)
        """
        self.documents = init_documents
        self.llm_client = create_deepinfra_client(api_key=api_key)

    def generate_affinity_mapping_session(self) -> AffinityMappingSession:
        """
        Generate an Affinity Mapping Session using DeepInfra GLM-4.6 model with init_documents as context.

        Returns:
            AffinityMappingSession: The generated affinity mapping document
        """
        # Combine all documents into context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(self.documents)
        ])

        # Create the prompt for DeepInfra
        prompt = f"""Based on the following documents, create an affinity mapping session that groups related concepts.

{context}

Please analyze these documents and create an affinity mapping with the following structure:

# Affinity Mapping Session

## Grouped Concept Clusters

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

[Continue with additional clusters as needed...]

Requirements:
1. Group related concepts into numbered clusters (Cluster 1, Cluster 2, etc.)
2. Each cluster should have a descriptive name
3. List related concepts under each cluster
4. Describe the key relationships between concepts within each cluster

Provide ONLY the affinity mapping with grouped concept clusters. Start directly with the clusters."""

        # Call DeepInfra API via LLM client
        system_prompt = "You are an expert in Domain-Driven Design and affinity mapping techniques."
        content = self.llm_client.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.7
        )

        return AffinityMappingSession(content=content)

    def process_document(self, document):
        """
        Process a document and transform it into a bounded context specification.

        Args:
            document: The input document to process

        Returns:
            A bounded context specification
        """
        # TODO: Implement document processing logic
        raise NotImplementedError("Document processing not yet implemented")


def main():
    """Main entry point for the service."""
    service = AffinityMappingService()
    print(f"Affinity Mapping Service v{__version__}")


if __name__ == "__main__":
    main()
