"""Minimal OpenAI-compatible chat client.

Works with DigitalOcean Serverless Inference (default), a vLLM / Ollama server
on a GPU Droplet, or anything else that speaks /v1/chat/completions.
"""
import json
import os
import re
import time

import httpx

DEFAULT_BASE_URL = "https://inference.do-ai.run/v1"
DEFAULT_MODEL = "openai-gpt-oss-120b"


class LLMError(RuntimeError):
    pass


def load_dotenv(path):
    """Read KEY=VALUE lines into os.environ without overriding real env vars."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLM:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=180.0, retries=2):
        self.retries = retries
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("MODEL_ACCESS_KEY", "")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        # Local servers (Ollama/vLLM) usually don't need a key.
        return bool(self.api_key) or "do-ai.run" not in self.base_url

    def chat(self, messages, temperature=0.2, max_tokens=4096) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(self.retries + 1):
            try:
                r = httpx.post(f"{self.base_url}/chat/completions", json=payload,
                               headers=headers, timeout=self.timeout)
            except httpx.HTTPError as e:
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"request failed: {e}") from e
            if (r.status_code == 429 or r.status_code >= 500) and attempt < self.retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        try:
            msg = r.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response: {r.text[:300]}") from e
        return msg.get("content") or ""

    def chat_json(self, messages, retries=2, **kw):
        """Call the model and parse a JSON value out of the reply, asking it to
        repair its output if parsing fails."""
        messages = list(messages)
        last = ""
        for _ in range(retries + 1):
            last = self.chat(messages, **kw)
            value = extract_json(last)
            if value is not None:
                return value
            messages += [
                {"role": "assistant", "content": last},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON value, no prose, no code fences."},
            ]
        raise LLMError(f"model did not return JSON: {last[:300]}")


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def extract_json(text: str):
    """Return the first JSON object/array found in text, or None.

    Open-weight models often wrap JSON in prose, code fences or <think> blocks,
    so we scan for the first position that decodes cleanly.
    """
    if not text:
        return None
    text = _THINK.sub("", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    return None
