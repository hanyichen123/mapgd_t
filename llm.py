import os
from typing import Any, Optional

_client: Optional[Any] = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def call_openai(
    prompt: str,
    system_prompt: str = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 1000,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print(call_openai('你好'))
