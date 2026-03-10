"""
Anthropic LLM Client for Agentic Tool-Use Loops

Simplified port of scaffold's TenantLLMClient for the Anthropic Messages API.
Used by the agentic affinity mapping loop to drive multi-turn tool-use
conversations via the tenant-llm service.
"""

import os
import re
import logging
import httpx

logger = logging.getLogger(__name__)

# Reuse same env vars as llm_client.py
TENANT_LLM_API_BASE_URL = os.getenv("TENANT_LLM_API_BASE_URL", "http://tenant-llm:8009")
CROSS_SERVICE_API_KEY = os.getenv("CROSS_SERVICE_API_KEY", "dev-cross-service-key-change-in-production")


class AnthropicLLMClient:
    """Client for Anthropic Messages API via tenant-llm service."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or TENANT_LLM_API_BASE_URL

    async def generate_anthropic_messages(
        self,
        tenant_id: str,
        model_reference: str,
        system_prompt: list[dict],
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 16384,
        timeout: float = 600.0,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        agentic_session_id: str | None = None,
    ) -> dict:
        """
        Call Anthropic proxy with a full messages list (multi-turn support).

        Args:
            tenant_id: Tenant identifier
            model_reference: Model reference
            system_prompt: Content block array with cache_control
            messages: Full Anthropic messages list
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            tools: Tool definitions in Anthropic format
            tool_choice: Anthropic tool choice config
            agentic_session_id: Session ID for agentic loop tracking

        Returns:
            Raw Anthropic response dict

        Raises:
            RuntimeError: On non-200 HTTP response
        """
        payload: dict = {
            "tenant_id": tenant_id,
            "model_reference": model_reference,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if agentic_session_id is not None:
            payload["agentic_session_id"] = agentic_session_id

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/anthropic/messages",
                json=payload,
                headers={
                    "X-API-Key": CROSS_SERVICE_API_KEY,
                    "X-Service-Name": "preparation-strategy",
                },
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Anthropic proxy returned HTTP {response.status_code}: {response.text[:500]}"
            )

        result = response.json()
        content_blocks = result.get("content", [])

        block_types: dict[str, int] = {}
        for block in content_blocks:
            btype = block.get("type", "unknown")
            block_types[btype] = block_types.get(btype, 0) + 1

        logger.info(
            f"generate_anthropic_messages: stop_reason={result.get('stop_reason')}, "
            f"blocks={block_types}, "
            f"usage={{in={result.get('usage', {}).get('input_tokens', 0)}, "
            f"out={result.get('usage', {}).get('output_tokens', 0)}, "
            f"cached={result.get('usage', {}).get('cache_read_input_tokens', 0)}}}"
        )
        return result

    @staticmethod
    def get_anthropic_stop_reason(response: dict) -> str:
        """Get stop_reason from Anthropic response."""
        return response.get("stop_reason", "")

    @staticmethod
    def extract_anthropic_text(response: dict) -> str:
        """Extract concatenated text from all text content blocks."""
        content_blocks = response.get("content", [])
        text_parts = [
            block["text"]
            for block in content_blocks
            if block.get("type") == "text" and block.get("text")
        ]
        return "\n\n".join(text_parts) if text_parts else ""

    @staticmethod
    def strip_thinking_tags(content: str) -> str:
        """Remove <think>...</think> blocks from LLM response."""
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
