"""
Agentic Loop for Affinity Mapping

Standalone async function adapted from scaffold's AgenticLoopMixin.
Drives MiniMax-M2.5 (or any Anthropic-compatible model) through read-tool
calls (served from a pre-built lookup), interleaved thinking, and a save-tool
call that emits the final output.
"""

import logging
from uuid import uuid4
from affinity_mapping_service.anthropic_llm_client import AnthropicLLMClient

logger = logging.getLogger(__name__)

# Input size limit — tenant-llm accepts up to 205 KB; use half for safety
MAX_INPUT_BYTES = 205 * 1024


def _truncate_to_fit(content: str, max_bytes: int) -> str:
    """Truncate content to fit within a byte limit, preserving the start."""
    content_bytes = content.encode("utf-8")
    if len(content_bytes) <= max_bytes:
        return content

    notice = "\n\n[...CONTENT TRUNCATED DUE TO SIZE LIMIT...]"
    notice_bytes = len(notice.encode("utf-8"))
    available = max_bytes - notice_bytes
    truncated = content_bytes[:available].decode("utf-8", errors="ignore")
    logger.warning(
        f"[INPUT_SIZE] Truncated content from {len(content_bytes):,} to {available:,} bytes"
    )
    return truncated + notice


async def run_agentic_loop(
    anthropic_client: AnthropicLLMClient,
    system_prompt: list[dict],
    user_prompt: str,
    tools: list[dict],
    tool_results_lookup: dict[str, str | dict[str, str]],
    save_tool_name: str,
    tenant_id: str,
    model_reference: str,
    max_turns: int,
    temperature: float = 0.6,
    max_tokens: int = 16384,
    timeout: float = 600.0,
) -> str:
    """
    Run a multi-turn agentic tool-use loop.

    The model calls read tools (served from tool_results_lookup), thinks
    between calls, then calls the save tool to emit output. The loop ends
    when the save tool is called or max_turns is reached.

    Args:
        anthropic_client: AnthropicLLMClient instance
        system_prompt: Content block array with cache_control
        user_prompt: Initial user prompt string
        tools: Tool definitions (read + save tools)
        tool_results_lookup: Dict mapping read tool names to content
        save_tool_name: Name of the save tool that signals completion
        tenant_id: Tenant identifier
        model_reference: Model reference
        max_turns: Maximum number of LLM round-trips
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        Extracted markdown content from the save tool call, or "" if not produced.

    Raises:
        RuntimeError: If max_turns exhausted with 0 saves.
    """
    log_prefix = "AffinityMapping"
    messages: list[dict] = [
        {"role": "user", "content": user_prompt},
    ]

    text_fallback = ""
    saved_content = ""
    agentic_session_id = str(uuid4())
    logger.info(f"[{log_prefix}] Starting agentic loop with session_id={agentic_session_id}")

    for turn in range(max_turns):
        response = await anthropic_client.generate_anthropic_messages(
            tenant_id=tenant_id,
            model_reference=model_reference,
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            tools=tools,
            tool_choice={"type": "any"},
            agentic_session_id=agentic_session_id,
        )

        stop_reason = AnthropicLLMClient.get_anthropic_stop_reason(response)
        content_blocks = response.get("content", [])

        # Append full assistant response to messages (preserves reasoning chain)
        messages.append({"role": "assistant", "content": content_blocks})

        # Extract tool_use blocks from this turn
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if not tool_use_blocks:
            # Model stopped calling tools — extract any text as fallback
            text_fallback = AnthropicLLMClient.extract_anthropic_text(response)
            logger.info(
                f"[{log_prefix} turn {turn+1}] No tool calls, stop_reason={stop_reason} "
                f"— extracting text fallback ({len(text_fallback)} chars)"
            )
            break

        # Guard against MiniMax parallel tool call bug: if >1 save tool in a
        # single turn, process only the first, return error for duplicates.
        save_seen_this_turn = False

        # Build tool_result messages for each tool call
        tool_result_blocks = []
        for block in tool_use_blocks:
            tool_name = block.get("name", "")
            tool_use_id = block.get("id", "")

            if tool_name == save_tool_name:
                if save_seen_this_turn:
                    # Duplicate save in same turn — reject
                    logger.warning(
                        f"[{log_prefix} turn {turn+1}] Duplicate save tool "
                        f"in same turn — rejecting (MiniMax parallel bug guard)"
                    )
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "Error: call save tools ONE AT A TIME, one per turn.",
                        "is_error": True,
                    })
                    continue

                save_seen_this_turn = True
                content = block.get("input", {}).get("markdown_content", "")
                saved_content = content
                logger.info(
                    f"[{log_prefix} turn {turn+1}] Output saved via {tool_name} "
                    f"({len(content.encode('utf-8')):,} bytes)"
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": "Saved successfully.",
                })
            elif tool_name in tool_results_lookup:
                # Read tool — return content (may be str or dict for parameterized tools)
                lookup_value = tool_results_lookup[tool_name]
                if isinstance(lookup_value, dict):
                    # Parameterized tool: extract the key from tool input
                    tool_input = block.get("input", {})
                    param_key = None
                    for v in tool_input.values():
                        if isinstance(v, str) and v in lookup_value:
                            param_key = v
                            break
                    if param_key is not None:
                        content = lookup_value[param_key]
                    else:
                        content = (
                            f"Error: no matching entry for input {tool_input}. "
                            f"Available: {', '.join(lookup_value.keys())}"
                        )
                        logger.warning(
                            f"[{log_prefix} turn {turn+1}] {tool_name} called with "
                            f"unrecognized input {tool_input}"
                        )
                else:
                    content = lookup_value
                content = _truncate_to_fit(content, MAX_INPUT_BYTES // 2)
                logger.info(
                    f"[{log_prefix} turn {turn+1}] Serving {tool_name} "
                    f"({len(content.encode('utf-8')):,} bytes)"
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                })
            else:
                logger.warning(f"[{log_prefix} turn {turn+1}] Unknown tool call: {tool_name}")
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"Error: unknown tool '{tool_name}'",
                    "is_error": True,
                })

        # Mark the last tool_result block with cache_control
        if tool_result_blocks:
            tool_result_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        # Append tool results as a user message
        messages.append({"role": "user", "content": tool_result_blocks})

        # If save tool was called, we're done
        if saved_content:
            break
    else:
        if not saved_content:
            raise RuntimeError(
                f"[{log_prefix}] Agentic loop exhausted {max_turns} turns "
                f"with 0 saves — model spent all turns on read tools and "
                f"never called the save tool"
            )

    result = saved_content or text_fallback
    if result:
        result = AnthropicLLMClient.strip_thinking_tags(result)
    return result
