"""
Affinity Mapping Service

A service for transforming documents into Bounded Context Specifications
using affinity mapping techniques.
"""

import re
from datetime import datetime
from pathlib import Path
from affinity_mapping_service.llm_client import create_tenant_llm_client
from affinity_mapping_service.embedding_client import create_openai_embedding_client, EmbeddingClient

__version__ = "0.1.0"


class AffinityMappingSession:
    """Represents an affinity mapping session with grouped concepts."""

    def __init__(self, content: str, embedding_client: EmbeddingClient | None = None, log_dir: str = "logs"):
        """
        Initialize an Affinity Mapping Session.

        Args:
            content: The generated affinity mapping document
            embedding_client: Optional embedding client for generating cluster embeddings
            log_dir: Directory to store log files (default: "logs")
        """
        self.content = content
        self.embedding_client = embedding_client
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
                'key_relationships': str,
                'embedding': list[float] or None,
                'assigned': list[dict]  # [{chunk_content: str, similarity_score: float, embedding: list[float]}]
            }]
        """
        if self._clusters is None:
            self._clusters = self._parse_clusters()
        return self._clusters

    @clusters.setter
    def clusters(self, value: list[dict]) -> None:
        """
        Set the clusters list.

        Args:
            value: List of cluster dictionaries
        """
        self._clusters = value

    def _parse_clusters(self) -> list[dict]:
        """Parse clusters from the generated content and generate embeddings."""
        clusters = []

        # Pattern to match cluster sections
        cluster_pattern = r'###\s+Cluster\s+(\d+):\s+(.+?)\n\*\*Related Concepts:\*\*\n(.*?)\n\*\*Key Relationships:\*\*\n(.*?)(?=\n###|\Z)'

        matches = re.findall(cluster_pattern, self.content, re.DOTALL)

        for match in matches:
            cluster_number = int(match[0])
            cluster_name = match[1].strip()
            related_concepts = match[2].strip()
            key_relationships = match[3].strip()

            # Create cluster text for embedding (combine all cluster information)
            cluster_text = f"{cluster_name}\n{related_concepts}\n{key_relationships}"

            # Generate embedding if embedding_client is available
            embedding = None
            if self.embedding_client is not None:
                embedding = self.embedding_client.embed(cluster_text)

            clusters.append({
                'cluster_name': cluster_name,
                'cluster_number': cluster_number,
                'related_concepts': related_concepts,
                'key_relationships': key_relationships,
                'embedding': embedding,
                'assigned': []  # Initialize with empty list for assigned chunks
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

    def __init__(
        self,
        init_documents: list[str],
        tenant_id: str = "default",
        model_reference: str = "deepseek-ai/DeepSeek-R1-Turbo@OpenAI",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the Affinity Mapping Service.

        Args:
            init_documents: List of initial documents to process
            tenant_id: Tenant identifier for multi-tenant LLM config (default: "default")
            model_reference: Model reference in format "model_name@factory" (default: "deepseek-ai/DeepSeek-R1-Turbo@OpenAI")
            openai_api_key: OpenAI API key for embeddings (optional, loaded from .env via OPENAI_API_KEY)
            embedding_model: Embedding model to use (default: text-embedding-3-large)
        """
        self.documents = init_documents
        self.llm_client = create_tenant_llm_client(
            tenant_id=tenant_id,
            model_reference=model_reference
        )
        self.embedding_client = create_openai_embedding_client(
            api_key=openai_api_key,
            model=embedding_model
        )

    async def generate_affinity_mapping_session(self) -> AffinityMappingSession:
        """
        Generate an Affinity Mapping Session using Tenant LLM service with init_documents as context.

        Returns:
            AffinityMappingSession: The generated affinity mapping document
        """
        # Combine all documents into context
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(self.documents)
        ])

        # Create the prompt for Tenant LLM
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
1. Create a MINIMUM of 12 clusters to ensure granular and comprehensive coverage
2. Group related concepts into numbered clusters (Cluster 1, Cluster 2, etc.)
3. Each cluster should have a descriptive name focusing on a specific aspect
4. Break down broad topics into more specific, focused clusters rather than combining them
5. List related concepts under each cluster
6. Describe the key relationships between concepts within each cluster
7. Aim for clusters that are specific and well-defined rather than broad categories

Provide ONLY the affinity mapping with grouped concept clusters. Start directly with the clusters."""

        # Call Tenant LLM API via LLM client
        system_prompt = "You are an expert in Domain-Driven Design and affinity mapping techniques."
        content = await self.llm_client.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.7
        )

        return AffinityMappingSession(content=content, embedding_client=self.embedding_client)

    def chunk_document(self, document: str, chunk_size: int = 512, chunk_overlap: int = 128) -> list[str]:
        """
        Chunk a markdown document using LlamaIndex SentenceSplitter for more granular chunks.

        Args:
            document: The markdown document to chunk
            chunk_size: Maximum size of each chunk in characters (default: 512)
            chunk_overlap: Number of characters to overlap between chunks (default: 128)

        Returns:
            list[str]: List of document chunks (text content from each node)
        """
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document

        # Create a LlamaIndex Document object
        llama_doc = Document(text=document)

        # Initialize SentenceSplitter with chunk size configuration
        # This will create more granular chunks based on sentences and size limits
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Parse the document into nodes
        nodes = parser.get_nodes_from_documents([llama_doc])

        # Extract text content from each node
        chunks = [node.get_content() for node in nodes]

        return chunks

    def chunk_documents(self, documents: list[str], chunk_size: int = 512, chunk_overlap: int = 128) -> list[str]:
        """
        Chunk multiple documents using LlamaIndex SentenceSplitter.

        Args:
            documents: List of markdown documents to chunk
            chunk_size: Maximum size of each chunk in characters (default: 512)
            chunk_overlap: Number of characters to overlap between chunks (default: 128)

        Returns:
            list[str]: Flattened list of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            all_chunks.extend(chunks)
        return all_chunks

    def add_chunk_to_cluster(
        self,
        chunk_embedding: list[float],
        chunk_content: str,
        session: AffinityMappingSession,
        similarity_threshold: float = 0.7
    ) -> dict:
        """
        Add a chunk to the most similar cluster in the affinity mapping session.

        Uses cosine similarity to find the best matching cluster based on embeddings.
        If no cluster meets the similarity threshold, the chunk is assigned to a special
        'Uncategorized' cluster (cluster_number=999). When a chunk is assigned to a cluster,
        it is added to the cluster's 'assigned' list with chunk_content, similarity_score,
        and embedding.

        Args:
            chunk_embedding: The embedding vector for the chunk
            chunk_content: The text content of the chunk
            session: The AffinityMappingSession containing clusters
            similarity_threshold: Minimum cosine similarity to consider a match (default: 0.7)

        Returns:
            dict: Information about the cluster assignment with keys:
                - 'cluster_number': int (999 for Uncategorized)
                - 'cluster_name': str (cluster name or 'Uncategorized')
                - 'similarity_score': float (cosine similarity)
                - 'chunk_content': str (the input chunk content)
                - 'assigned': bool (always True, including for Uncategorized)
        """
        import numpy as np

        clusters = session.clusters

        if not clusters:
            return {
                'cluster_number': None,
                'cluster_name': None,
                'similarity_score': 0.0,
                'chunk_content': chunk_content,
                'assigned': False
            }

        # Calculate cosine similarity with each cluster
        best_cluster = None
        best_similarity = -1.0

        for cluster in clusters:
            cluster_embedding = cluster.get('embedding')

            if cluster_embedding is None:
                continue

            # Calculate cosine similarity
            similarity = np.dot(chunk_embedding, cluster_embedding) / (
                np.linalg.norm(chunk_embedding) * np.linalg.norm(cluster_embedding)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        # Check if best similarity meets threshold
        if best_cluster is not None and best_similarity >= similarity_threshold:
            # Add the chunk to the cluster's assigned list
            best_cluster['assigned'].append({
                'chunk_content': chunk_content,
                'similarity_score': float(best_similarity),
                'embedding': chunk_embedding
            })

            return {
                'cluster_number': best_cluster['cluster_number'],
                'cluster_name': best_cluster['cluster_name'],
                'similarity_score': float(best_similarity),
                'chunk_content': chunk_content,
                'assigned': True
            }
        else:
            # Chunk doesn't meet threshold - assign to 'Uncategorized' cluster (999)
            # First, check if the 'Uncategorized' cluster already exists
            uncategorized_cluster = None
            for cluster in clusters:
                if cluster['cluster_number'] == 999:
                    uncategorized_cluster = cluster
                    break

            # If it doesn't exist, create it
            if uncategorized_cluster is None:
                uncategorized_cluster = {
                    'cluster_number': 999,
                    'cluster_name': 'Uncategorized',
                    'embedding': None,  # No embedding for this special cluster
                    'assigned': []
                }
                clusters.append(uncategorized_cluster)

            # Add the chunk to the Uncategorized cluster
            uncategorized_cluster['assigned'].append({
                'chunk_content': chunk_content,
                'similarity_score': float(best_similarity) if best_similarity > -1 else 0.0,
                'embedding': chunk_embedding
            })

            return {
                'cluster_number': 999,
                'cluster_name': 'Uncategorized',
                'similarity_score': float(best_similarity) if best_similarity > -1 else 0.0,
                'chunk_content': chunk_content,
                'assigned': True
            }

    def add_chunks_to_clusters(
        self,
        chunk_contents: list[str],
        session: AffinityMappingSession,
        similarity_threshold: float = 0.7
    ) -> list[dict]:
        """
        Add multiple chunks to the most similar clusters in the affinity mapping session.

        Uses batch embedding generation for better performance, then calculates cosine
        similarity to find the best matching cluster for each chunk.

        Args:
            chunk_contents: List of text content for each chunk
            session: The AffinityMappingSession containing clusters
            similarity_threshold: Minimum cosine similarity to consider a match (default: 0.7)

        Returns:
            list[dict]: List of cluster assignment results, one for each chunk.
                Each dict contains:
                - 'cluster_number': int or None if no match
                - 'cluster_name': str or None if no match
                - 'similarity_score': float (cosine similarity)
                - 'chunk_content': str (the input chunk content)
                - 'assigned': bool (True if assigned to a cluster)
        """
        # Batch generate embeddings for all chunks
        chunk_embeddings = self.embedding_client.embed_batch(chunk_contents)

        # Assign each chunk to a cluster using the generated embeddings
        results = []
        for chunk_content, chunk_embedding in zip(chunk_contents, chunk_embeddings):
            result = self.add_chunk_to_cluster(
                chunk_embedding=chunk_embedding,
                chunk_content=chunk_content,
                session=session,
                similarity_threshold=similarity_threshold
            )
            results.append(result)

        return results

    async def process_documents_with_affinity_mapping(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7
    ) -> tuple[AffinityMappingSession, list[str], list[dict]]:
        """
        Complete workflow: generate affinity mapping, chunk documents, and assign chunks to clusters.

        This method orchestrates the full affinity mapping process:
        1. Generates an affinity mapping session from self.documents
        2. Chunks all documents into smaller pieces (with iterative re-chunking if needed)
        3. Assigns each chunk to the most similar cluster (or 'Uncategorized' if below threshold)

        If there are too few chunks compared to clusters, the method will iteratively reduce
        chunk_size to generate more chunks, ensuring better coverage of all clusters.

        Args:
            chunk_size: Maximum size of each chunk in characters (default: 1000)
            chunk_overlap: Number of overlapping characters between chunks (default: 200)
            similarity_threshold: Minimum cosine similarity to assign to a cluster (default: 0.7)

        Returns:
            tuple containing:
                - AffinityMappingSession: The generated session with clusters
                - list[str]: All document chunks
                - list[dict]: Cluster assignment results for each chunk
        """
        # Step 1: Generate affinity mapping session
        session = await self.generate_affinity_mapping_session()

        # Count regular clusters (exclude uncategorized cluster 999 if it exists)
        num_clusters = len([c for c in session.clusters if c['cluster_number'] != 999])

        # Step 2: Chunk all documents with iterative re-chunking if needed
        current_chunk_size = chunk_size
        current_overlap = chunk_overlap
        min_chunk_size = 100  # Minimum chunk size to avoid too small chunks

        chunks = self.chunk_documents(
            documents=self.documents,
            chunk_size=current_chunk_size,
            chunk_overlap=current_overlap
        )

        # Iteratively reduce chunk size if we have too few chunks
        # Target: at least 2x chunks compared to clusters for better coverage
        # This ensures enough chunks to distribute across all clusters
        target_chunk_count = num_clusters * 2

        iteration = 1
        max_iterations = 10

        while len(chunks) < target_chunk_count and current_chunk_size > min_chunk_size and iteration <= max_iterations:
            # Reduce chunk size by 30% and overlap proportionally
            current_chunk_size = int(current_chunk_size * 0.7)
            current_overlap = int(current_overlap * 0.7)

            # Ensure minimum chunk size
            if current_chunk_size < min_chunk_size:
                current_chunk_size = min_chunk_size
                current_overlap = int(min_chunk_size * 0.2)  # 20% overlap for minimum size

            print(f"  → Iteration {iteration}: Re-chunking with chunk_size={current_chunk_size}, overlap={current_overlap}")

            chunks = self.chunk_documents(
                documents=self.documents,
                chunk_size=current_chunk_size,
                chunk_overlap=current_overlap
            )

            print(f"  → Generated {len(chunks)} chunks (target: {target_chunk_count})")
            iteration += 1

        if iteration > max_iterations:
            print(f"  → Reached maximum iterations ({max_iterations}), stopping with {len(chunks)} chunks")

        # Step 3: Assign chunks to clusters
        results = self.add_chunks_to_clusters(
            chunk_contents=chunks,
            session=session,
            similarity_threshold=similarity_threshold
        )

        return session, chunks, results

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
