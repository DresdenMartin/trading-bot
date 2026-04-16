"""Utilities for working with OpenAI models (GPT-5+ fallbacks).

If ANTHROPIC_API_KEY is set it takes precedence over OPENAI_API_KEY for all
call_chat_completion / call_web_search calls. Set ANTHROPIC_MODEL to override
the default (claude-haiku-4-5-20251001).
"""
import os
from typing import Optional

from openai import OpenAI


def _call_claude(system_prompt: str, user_prompt: str, *, max_output_tokens: int = 600) -> str:
    """Route a completion call through the Anthropic API."""
    import anthropic
    api_key = os.getenv('ANTHROPIC_API_KEY')
    model = os.getenv('ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=system_prompt or '',
        messages=[{'role': 'user', 'content': user_prompt}],
    )
    return msg.content[0].text if msg.content else ''


def _is_gpt5(model: Optional[str]) -> bool:
    return bool(model and model.strip().lower().startswith('gpt-5'))


def _is_gpt5_model(model: Optional[str]) -> bool:
    return _is_gpt5(model)


def _gpt5_min_tokens() -> int:
    try:
        configured = int(os.getenv('OPENAI_GPT5_MIN_OUTPUT_TOKENS', '600'))
    except Exception:
        configured = 600
    return max(256, configured)


def _extract_response_text(resp) -> str:
    """Pull assistant text from either Responses API or chat completion objects."""
    if resp is None:
        return ''
    # responses API objects expose `.output` and sometimes `.text`
    text_parts: list[str] = []
    output = getattr(resp, 'output', None)
    if output:
        for item in output:
            content = getattr(item, 'content', None)
            if not content:
                continue
            for part in content:
                text = getattr(part, 'text', None)
                if text:
                    text_parts.append(text)
    if text_parts:
        return ''.join(text_parts).strip()
    if hasattr(resp, 'text') and isinstance(resp.text, str):
        return resp.text.strip()
    # chat completion fallback
    choices = getattr(resp, 'choices', None)
    if choices:
        try:
            return choices[0].message.content.strip()
        except Exception:
            pass
    return ''


def _default_temperature(model: str, temperature: Optional[float]) -> float:
    if temperature is None:
        try:
            temperature = float(os.getenv('OPENAI_TEMPERATURE', '0'))
        except Exception:
            temperature = 0.0
    if _is_gpt5(model) and temperature == 0.0:
        return 1.0
    return temperature


def call_chat_completion(system_prompt: str, user_prompt: str, *, max_output_tokens: int = 600, temperature: Optional[float] = None) -> str:
    if os.getenv('ANTHROPIC_API_KEY'):
        return _call_claude(system_prompt, user_prompt, max_output_tokens=max_output_tokens)
    # ── original OpenAI path below ────────────────────────────────────────────
    """Invoke OpenAI chat with robust GPT-5 fallback.

    Parameters
    ----------
    system_prompt: str
        Instruction prompt (can be empty string).
    user_prompt: str
        User/content prompt.
    max_output_tokens: int
        Desired cap for generated tokens.
    temperature: Optional[float]
        Explicit temperature; if None uses env OPENAI_TEMPERATURE.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    client = OpenAI(api_key=api_key)
    temperature = _default_temperature(model, temperature)
    tokens = max_output_tokens

    if _is_gpt5(model):
        tokens = max(tokens, _gpt5_min_tokens())
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {'role': 'system', 'content': [{'type': 'input_text', 'text': system_prompt or ''}]},
                    {'role': 'user', 'content': [{'type': 'input_text', 'text': user_prompt}]},
                ],
                max_output_tokens=tokens,
                temperature=temperature,
            )
            txt = _extract_response_text(resp)
            if txt:
                return txt
        except Exception:
            # fall back below
            pass
        # Fallback to legacy chat completions for GPT-5
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt or ''},
                {'role': 'user', 'content': user_prompt},
            ],
            max_completion_tokens=tokens,
            temperature=temperature,
        )
        return _extract_response_text(resp)

    # Non GPT-5 path
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt or ''},
            {'role': 'user', 'content': user_prompt},
        ],
        max_completion_tokens=tokens,
        temperature=temperature,
    )
    return _extract_response_text(resp)


def _web_search_tool_type() -> str:
    return os.getenv('OPENAI_WEB_SEARCH_TOOL', 'web_search')


def call_web_search(system_prompt: str, user_prompt: str, *, max_output_tokens: int = 800, temperature: Optional[float] = None) -> str:
    if os.getenv('ANTHROPIC_API_KEY'):
        return _call_claude(system_prompt, user_prompt, max_output_tokens=max_output_tokens)
    # ── original OpenAI path below ────────────────────────────────────────────
    """Invoke OpenAI responses with web search tool enabled.

    Falls back to call_chat_completion if the tool/model is unavailable.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')
    model = os.getenv('OPENAI_WEB_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))
    client = OpenAI(api_key=api_key)
    temperature = _default_temperature(model, temperature)
    tokens = max_output_tokens
    if _is_gpt5(model):
        tokens = max(tokens, _gpt5_min_tokens())

    tool_type = _web_search_tool_type()
    try:
        resp = client.responses.create(
            model=model,
            tools=[{'type': tool_type}],
            input=[
                {'role': 'system', 'content': [{'type': 'input_text', 'text': system_prompt or ''}]},
                {'role': 'user', 'content': [{'type': 'input_text', 'text': user_prompt}]},
            ],
            max_output_tokens=tokens,
            temperature=temperature,
        )
        txt = _extract_response_text(resp)
        if txt:
            return txt
    except Exception:
        pass

    return call_chat_completion(system_prompt, user_prompt, max_output_tokens=tokens, temperature=temperature)
