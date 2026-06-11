"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


logger = logging.getLogger('mirofish.llm_client')


def _parse_llm_json(response: str) -> Dict[str, Any]:
    """
    Robust JSON parser for LLM outputs.

    Many LLMs (qwen, gemma, ollama-served models) append trailing text
    after the JSON block even with response_format=json_object. Also,
    JSON blocks are often wrapped in ```json ... ``` Markdown fences.

    Strategy:
    1. Strip Markdown fences
    2. json.loads (strict, fastest path)
    3. raw_decode (parses JSON prefix, ignores trailing text)
    4. Balanced-brace extraction (finds first complete {...} structure)
    5. Strip control chars + retry

    Raises ValueError with helpful snippet on all failures.
    """
    if not response or not response.strip():
        raise ValueError("LLM returned empty response")

    # 1. Strip Markdown fences
    cleaned = response.strip()
    cleaned = re.sub(r'^```(?:json|JSON)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    # 2. Fast path: complete JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e_strict:
        first_error = e_strict

    # 3. raw_decode - parses JSON prefix, ignores trailing text
    try:
        decoder = json.JSONDecoder()
        obj, end_idx = decoder.raw_decode(cleaned)
        trailing = cleaned[end_idx:].strip()
        if trailing:
            logger.warning(
                "LLM appended trailing text after JSON (%d chars), ignored. Preview: %s",
                len(trailing), trailing[:120]
            )
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"items": obj}
    except json.JSONDecodeError:
        pass

    # 4. Balanced-brace extraction: find first complete {...}
    start = cleaned.find('{')
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        result = json.loads(candidate)
                        logger.warning(
                            "Extracted JSON from messy LLM output (%d chars before, %d after)",
                            start, len(cleaned) - (i + 1)
                        )
                        return result
                    except json.JSONDecodeError:
                        break

    # 5. Last resort: strip control chars + retry
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
    if sanitized != cleaned:
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

    # All strategies failed
    snippet = cleaned[:200] + ('...' if len(cleaned) > 200 else '')
    raise ValueError(
        f"LLM returned invalid JSON (all parse strategies failed): "
        f"first_error={first_error.msg} at pos {first_error.pos}. "
        f"Response preview: {snippet}"
    )


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        import time
        max_attempts = 5
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                break
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "429" in error_str or "rate_limit" in error_str.lower() or "Rate limit" in error_str:
                    wait = min(2 ** attempt * 2, 30)  # 2s, 4s, 8s, 16s, 30s
                    logger.warning(f"Rate limit hit, retrying in {wait}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise last_error
        # 部分模型（如MiniMax M2.5）会在content中包含<think>思考内容，需要移除
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return _parse_llm_json(response)

