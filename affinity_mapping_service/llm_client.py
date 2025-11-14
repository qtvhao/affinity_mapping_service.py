"""
LLM Client for Tenant LLM Service

A dedicated client for interacting with LLM services via the tenant-llm service.
This ensures consistent API key management and multi-tenant LLM configuration.
"""

import os
import httpx
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
TENANT_LLM_API_BASE_URL = os.getenv("TENANT_LLM_API_BASE_URL", "http://tenant-llm:8009")
CROSS_SERVICE_API_KEY = os.getenv("CROSS_SERVICE_API_KEY", "dev-cross-service-key-change-in-production")


class LLMClient:
    """Client for interacting with LLM services via Tenant LLM Service."""

    def __init__(
        self,
        tenant_id: str = "default",
        model_reference: str = "deepseek-ai/DeepSeek-R1-Turbo@OpenAI",
        base_url: str | None = None
    ):
        """
        Initialize the LLM Client.

        Args:
            tenant_id: Tenant identifier for multi-tenant LLM config
            model_reference: Model reference in format "model_name@factory" (e.g., "gpt-4o@OpenAI")
            base_url: Optional base URL override (default: TENANT_LLM_API_BASE_URL env var)
        """
        self.tenant_id = tenant_id
        self.model_reference = model_reference
        self.base_url = base_url or TENANT_LLM_API_BASE_URL
        self.proxy_url = f"{self.base_url}/api/v1/openai/chat/completions"

    async def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 180
    ) -> Optional[str]:
        """
        Generate a chat completion using tenant-llm service.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (optional)
            timeout: Request timeout in seconds (default: 180)

        Returns:
            str: The generated completion content, or None if failed
        """
        try:
            logger.info(f"Calling tenant-llm proxy: {self.proxy_url} with model {self.model_reference}")

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Build payload with max_tokens (use default of 4096 if None, as tenant-llm requires this field)
                payload = {
                    "tenant_id": self.tenant_id,
                    "model_reference": self.model_reference,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens if max_tokens is not None else 4096
                }

                response = await client.post(
                    self.proxy_url,
                    headers={
                        "X-API-Key": CROSS_SERVICE_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    logger.info(f"Generated content using Tenant LLM proxy: {self.model_reference}")
                    return content
                else:
                    logger.error(f"Tenant LLM proxy error: {response.status_code} - {response.text}")
                    return None

        except httpx.TimeoutException:
            logger.error(f"Tenant LLM proxy request timed out after {timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Error calling Tenant LLM proxy: {str(e)}")
            return None

    async def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Generate completion with system and user prompts.

        Args:
            system_prompt: System role prompt
            user_prompt: User prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            str: The generated completion content, or None if failed
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )


def create_tenant_llm_client(
    tenant_id: str = "default",
    model_reference: str = "deepseek-ai/DeepSeek-R1-Turbo@OpenAI"
) -> LLMClient:
    """
    Factory function to create a Tenant LLM client.

    Args:
        tenant_id: Tenant identifier (default: "default")
        model_reference: Model reference in format "model_name@factory"

    Returns:
        LLMClient: Configured LLM client instance
    """
    return LLMClient(
        tenant_id=tenant_id,
        model_reference=model_reference
    )
