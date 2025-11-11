"""
LLM Client for DeepInfra API

A dedicated client for interacting with LLM services via DeepInfra.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMClient:
    """Client for interacting with LLM services via DeepInfra API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepinfra.com/v1/openai",
        model: str = "zai-org/GLM-4.6"
    ):
        """
        Initialize the LLM Client.

        Args:
            api_key: DeepInfra API key (optional, loaded from DEEPINFRA_API_KEY env var)
            base_url: API base URL (default: DeepInfra OpenAI-compatible endpoint)
            model: Model identifier (default: zai-org/GLM-4.6)
        """
        if api_key is None:
            api_key = os.environ.get('DEEPINFRA_API_KEY')

        if not api_key:
            raise ValueError(
                "API key is required. Provide it via the api_key parameter "
                "or set DEEPINFRA_API_KEY environment variable."
            )

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        model: str | None = None
    ) -> str:
        """
        Generate a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (default: 0.7)
            model: Model to use (default: uses instance model)

        Returns:
            str: The generated completion content
        """
        if model is None:
            model = self.model

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response.choices[0].message.content

    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate completion with system and user prompts.

        Args:
            system_prompt: System role prompt
            user_prompt: User prompt
            temperature: Sampling temperature (default: 0.7)

        Returns:
            str: The generated completion content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.chat_completion(messages=messages, temperature=temperature)


def create_deepinfra_client(
    api_key: str | None = None,
    model: str = "zai-org/GLM-4.6"
) -> LLMClient:
    """
    Factory function to create a DeepInfra LLM client.

    Args:
        api_key: DeepInfra API key (optional, loaded from env)
        model: Model identifier (default: zai-org/GLM-4.6)

    Returns:
        LLMClient: Configured LLM client instance
    """
    return LLMClient(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
        model=model
    )
