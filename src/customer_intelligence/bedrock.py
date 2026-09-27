"""LLM + embedding client.

Note on naming: the assessment prefers AWS Bedrock, but only OpenAI-compatible credentials were
available for this exercise (see README). This module keeps the `bedrock` name from the original
skeleton but wraps the OpenAI SDK against a custom `base_url`, using a Bedrock-style
`messages=[{"role": ..., "content": [{"text": ...}]}]` calling convention so the rest of the
codebase (ingest/qa/analysis) is agnostic to the underlying provider.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .config import settings
from .logging_setup import get_logger

logger = get_logger(__name__)


class ConverseResult:
    def __init__(self, text: str = "", tool_input: dict | None = None, stop_reason: str = "") -> None:
        self.text = text
        self.tool_input = tool_input or {}
        self.stop_reason = stop_reason


class _OpenAICompatClient:
    """
    Wraps the OpenAI python SDK against any OpenAI-compatible endpoint.
    Reads base_url, api_key, model from settings (loaded from .env).
    """

    def __init__(self) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package is required. pip install openai") from exc

        self._client = openai.OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def converse(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> ConverseResult:
        """
        Send a chat request. `messages` uses Bedrock-style format:
        [{"role": "user|assistant", "content": [{"text": "..."}]}, ...]
        Converted internally to OpenAI format and sent to the endpoint.
        """
        oai_messages = [{"role": "system", "content": system}]
        for m in messages:
            role = m.get("role", "user")
            content_blocks = m.get("content", [])
            if isinstance(content_blocks, list):
                text = "\n".join(
                    b.get("text", "") for b in content_blocks if isinstance(b, dict) and "text" in b
                )
            else:
                text = str(content_blocks)
            oai_messages.append({"role": role, "content": text})

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": oai_messages,
            "temperature": settings.gen_temperature,
            "max_tokens": settings.max_tokens,
        }

        if tools:
            oai_tools = []
            for t in tools:
                spec = t.get("toolSpec", t.get("toolspec", {}))
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec.get("inputSchema", spec.get("inputschema", {})).get("json", {}),
                    },
                })
            kwargs["tools"] = oai_tools
            kwargs["tool_choice"] = tool_choice or "auto"

        t0 = time.perf_counter()
        resp = self._call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
        latency = time.perf_counter() - t0
        logger.info("LLM call latency: %.3f sec", latency)
        self._log_usage(resp)

        choice = resp.choices[0]
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            try:
                payload = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool call arguments: %s", exc)
                payload = {}
            logger.debug("Tool call: %s(%s)", tc.function.name, json.dumps(payload))
            return ConverseResult(tool_input=payload, stop_reason=choice.finish_reason or "")

        text = choice.message.content or ""
        logger.debug("LLM response: %s", text)
        return ConverseResult(text=text, stop_reason=choice.finish_reason or "")

    def converse_json(self, system: str, user: str, schema: dict, tool_name: str) -> dict:
        """
        Force the LLM to respond with a JSON object that conforms to the given schema, by
        requiring a specific tool call.
        """
        logger.debug("Converse JSON: tool_name=%s", tool_name)
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Emit a structured {tool_name} result",
                    "parameters": schema,
                },
            }],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "temperature": settings.gen_temperature,
            "max_tokens": settings.max_tokens,
        }

        t0 = time.perf_counter()
        resp = self._call_with_retry(lambda: self._client.chat.completions.create(**kwargs))
        latency = time.perf_counter() - t0
        logger.info("LLM call latency: %.3f sec", latency)
        self._log_usage(resp)

        tc = resp.choices[0].message.tool_calls[0]
        result = json.loads(tc.function.arguments)
        logger.debug("Converse JSON result: %s", json.dumps(result))
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using the configured embedding model."""
        logger.debug("Embedding %d texts", len(texts))
        all_embeddings: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug("Embedding batch %d-%d", i, min(i + batch_size, len(texts)))
            t0 = time.perf_counter()

            def _do_embed(batch=batch):
                kwargs: dict[str, Any] = {"model": settings.embedding_model, "input": batch}
                if settings.embedding_dimensions:
                    kwargs["dimensions"] = settings.embedding_dimensions
                return self._client.embeddings.create(**kwargs)

            resp = self._call_with_retry(_do_embed)
            latency = time.perf_counter() - t0
            logger.info("Embedding call latency: %.3f sec", latency)
            all_embeddings.extend(item.embedding for item in resp.data)

        return all_embeddings

    # __ retry / logging helpers ____________________________________________________________
    @staticmethod
    def _log_usage(resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        try:
            payload = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
            logger.info("LLM call usage: %s", json.dumps(payload))
        except Exception:  # pragma: no cover - purely diagnostic
            logger.info("LLM call usage: %s", usage)

    @staticmethod
    def _call_with_retry(func, max_retries: int = 3, base_delay: float = 2.0):
        import openai as oai
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return func()
            except oai.RateLimitError as exc:
                wait = base_delay * (2 ** attempt)
                logger.warning("Rate limit error, retrying in %.1f seconds...", wait)
                time.sleep(wait)
                last_exc = exc
            except oai.APIConnectionError as exc:
                wait = base_delay * (2 ** attempt)
                logger.warning("API connection error, retrying in %.1f seconds...", wait)
                time.sleep(wait)
                last_exc = exc
            except oai.BadRequestError as exc:
                logger.error("Bad request error: %s", exc)
                raise
            except Exception as exc:
                logger.error("Unexpected error: %s", exc)
                raise
        assert last_exc is not None
        raise last_exc


bedrock = _OpenAICompatClient()