"""
Shared extraction utilities for LLM-based answer extraction.

Supports two backends:
- Modal endpoint: Set EXTRACTION_ENDPOINT env var (e.g., https://your-modal-endpoint.modal.run/v1)
- OpenAI API: Set PROCESSOR=gpt-4o-mini and OPENAI_API_KEY

Modal takes precedence if both are set.
"""

import os
import time
from typing import Optional

import openai
from openai import OpenAI


# Regex pattern for "Answer: ..." extraction
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"


class ChatCompletionSampler:
    """
    Sample from chat completion API (OpenAI or Modal endpoint).
    """

    def __init__(
        self,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        endpoint = os.getenv("EXTRACTION_ENDPOINT", "https://modal-labs-civicmachines--vllm-inference-server-serve.modal.run")
        
        if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
            # OpenAI API
            self.client = OpenAI()
            self.model = "gpt-4o-mini"
        else:
            # Modal endpoint - use base_url with /v1 (OpenAI client appends /chat/completions)
            base_url = endpoint.rstrip("/") + "/v1"
            self.client = OpenAI(base_url=base_url, api_key="EMPTY")
            # Use provided model or default to the standard model
            self.model = model or os.getenv("EXTRACTION_MODEL", "ArliAI/gpt-oss-20b-Derestricted")
        
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                if response and response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content
                    return content.strip() if content else None
                return None
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return None
            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1


def get_extraction_sampler() -> Optional[ChatCompletionSampler]:
    """
    Factory function to get an extraction sampler if configured.
    
    Returns ChatCompletionSampler (defaults to Modal endpoint unless PROCESSOR=gpt-4o-mini is set).
    """
    return ChatCompletionSampler()


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} or \\fbox{...} content from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> Optional[str]:
    """Remove \\boxed{} wrapper from a string."""
    if s is None:
        return None
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
        return s  # Return original if format doesn't match

    left = "\\boxed{"

    if not s.startswith(left) or not s.endswith("}"):
        return s  # Return original if format doesn't match

    return s[len(left) : -1]
