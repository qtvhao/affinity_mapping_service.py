"""
Affinity Mapping Service

A service for transforming documents into Bounded Context Specifications
using affinity mapping techniques.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__version__ = "0.1.0"


class AffinityMappingSession:
    """Represents an affinity mapping session with grouped concepts."""

    def __init__(self, content: str):
        """
        Initialize an Affinity Mapping Session.

        Args:
            content: The generated affinity mapping document
        """
        self.content = content


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
        # Use DeepInfra API endpoint and key
        if api_key is None:
            api_key = os.environ.get('DEEPINFRA_API_KEY')
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )

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
        prompt = f"""Based on the following documents, create an affinity mapping session that groups related concepts and identifies bounded contexts.

{context}

Please analyze these documents and create an affinity mapping that:
1. Identifies key concepts and themes
2. Groups related concepts together
3. Suggests potential bounded contexts
4. Highlights relationships between concepts

Provide the affinity mapping in a structured format."""

        # Call DeepInfra API with GLM-4.6 model
        response = self.client.chat.completions.create(
            model="zai-org/GLM-4.6",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Domain-Driven Design and affinity mapping techniques."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        # Extract the generated content
        content = response.choices[0].message.content

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
