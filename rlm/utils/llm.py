"""OpenAI-compatible chat client and OpenRouter Jev decisions client."""

import json
import os
import time
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    """Chat completions client; OpenRouter is the default provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek/deepseek-v4-flash-0731", provider: str = "openrouter"):
        provider = provider.lower()
        if provider not in {"openrouter", "openai"}:
            raise ValueError("provider must be 'openrouter' or 'openai'")
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY")
        if not self.api_key:
            env_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            raise ValueError(f"API key is required. Set {env_name} or pass api_key.")
        
        self.model = model
        if provider == "openrouter":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"X-OpenRouter-Title": "jev-rlm"},
            )
        else:
            self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            token_arg = {"max_tokens": max_tokens} if max_tokens is not None and self.provider == "openrouter" else {}
            if max_tokens is not None and self.provider == "openai":
                token_arg = {"max_completion_tokens": max_tokens}
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **token_arg,
                **kwargs
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class JevClient:
    """Calls Jev's Choice, Score, and Noul primitives through OpenRouter."""

    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
    MAX_STATE_CHARS = 120_000

    def __init__(self, api_key: Optional[str] = None, model: str = "typesafe/jev-1.13", timeout: float = 30, trace_callback=None, progress_callback=None):
        import requests

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for Jev decisions")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()
        self.trace_callback = trace_callback
        self.progress_callback = progress_callback
        self._request_count = 0

    def decide(self, state: object, questions: dict) -> dict:
        request_payload = {"model": self.model, "state": state, "questions": questions}
        state_chars = len(json.dumps(state, ensure_ascii=False, default=str))
        if state_chars > self.MAX_STATE_CHARS:
            error = (f"Jev state is {state_chars:,} characters; the limit is "
                     f"{self.MAX_STATE_CHARS:,}. Split the state into smaller chunks and retry.")
            if self.progress_callback:
                self.progress_callback(f"Jev request rejected: {error}")
            if self.trace_callback:
                self.trace_callback({"type": "jev_error", "error": error,
                                     "state_chars": state_chars, "limit": self.MAX_STATE_CHARS})
            raise ValueError(error)
        if self.trace_callback:
            self.trace_callback({"type": "jev_request", "endpoint": self.ENDPOINT, "request": request_payload})
        self._request_count += 1
        request_number = self._request_count
        started = time.perf_counter()
        if self.progress_callback:
            self.progress_callback(f"Jev request {request_number} started ({state_chars:,} state characters)")
        try:
            response = self._session.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=request_payload,
                timeout=self.timeout,
            )
            if not response.ok:
                raise RuntimeError(f"OpenRouter Jev request failed ({response.status_code}): {response.text[:1000]}")
            payload = response.json()
            if not isinstance(payload.get("answers"), dict):
                raise RuntimeError("OpenRouter Jev response did not contain an answers object")
        except Exception as exc:
            if self.progress_callback:
                elapsed = time.perf_counter() - started
                self.progress_callback(f"Jev request {request_number} failed after {elapsed:.1f}s: {exc}")
            if self.trace_callback:
                self.trace_callback({"type": "jev_error", "request": request_payload, "error": str(exc)})
            raise
        if self.progress_callback:
            elapsed = time.perf_counter() - started
            self.progress_callback(f"Jev request {request_number} completed in {elapsed:.1f}s")
        if self.trace_callback:
            self.trace_callback({"type": "jev_response", "response": payload})
        return payload

    def choice(self, state: object, instructions: str, criteria: dict[str, str]) -> dict:
        """Choose among named categories using the supplied state and criteria."""
        if not criteria:
            raise ValueError("choice requires at least one option")
        key = "result"
        result = self.decide(state, {key: {"type": "choice", "instructions": instructions, "criteria": criteria}})
        return result["answers"][key]

    def score(self, state: object, instructions: str, levels: list[str]) -> dict:
        """Rate the supplied state using ordered textual levels."""
        if not 2 <= len(levels) <= 10:
            raise ValueError("score requires 2 to 10 ordered levels")
        key = "result"
        result = self.decide(state, {key: {"type": "score", "instructions": instructions, "criteria": levels}})
        return result["answers"][key]

    def noul(self, state: object, instructions: str, true_criteria: str, false_criteria: str) -> dict:
        """Make a structured yes/no decision about the supplied state."""
        key = "result"
        result = self.decide(state, {key: {"type": "noul", "instructions": instructions,
                                           "criteria": {"true": true_criteria, "false": false_criteria}}})
        return result["answers"][key]
