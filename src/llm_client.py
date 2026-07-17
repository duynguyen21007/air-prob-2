"""
Shared vLLM client for all stages.
Connects to a locally-hosted vLLM server via the OpenAI-compatible API.
"""
import os
import time
from openai import OpenAI
from src.config import VLLM_BASE_URL, VLLM_API_KEY, VLLM_MODEL, VLLM_MAX_TOKENS, VLLM_TEMPERATURE


client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)


def get_response_for_single_chat(prompt: str) -> str:
    """
    Send a single-turn chat prompt to the vLLM server and return the response text.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            temperature=VLLM_TEMPERATURE,
            max_tokens=VLLM_MAX_TOKENS,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        response = response.choices[0].message.content
    except Exception as e:
        raise Exception(e)
    return response
