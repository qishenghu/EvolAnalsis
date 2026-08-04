"""
OpenAI-Compatible API Teacher LLM Backend.

This module provides the OpenAITeacherLLM class for calling OpenAI-compatible APIs
including GPT-4, Claude via OpenAI API, DashScope, DeepSeek API, vLLM Server, etc.
"""

import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger

from .base_teacher_llm import BaseTeacherLLM


class OpenAITeacherLLM(BaseTeacherLLM):
    """
    OpenAI-Compatible API Teacher LLM 后端
    
    ⭐ 支持：OpenAI GPT-4/4o、Claude via OpenAI API、DashScope、DeepSeek API、vLLM Server 等
    
    关键特性：
    - 通过 api_base 参数兼容各种 OpenAI-compatible 服务
    - 支持采集 log_prob（如果 API 支持）
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        collect_log_prob: bool = False,
        max_retries: int = 3,
        timeout: float = 120.0,
    ):
        """
        初始化 OpenAI-Compatible Teacher LLM。
        
        Args:
            model_name: 模型名称 (e.g., "gpt-4", "gpt-4-turbo", "qwen-max")
            api_base: API base URL (默认使用 OpenAI 官方 API)
            api_key: API key (默认从环境变量读取)
            temperature: 采样温度
            max_tokens: 最大生成 token 数（输出长度上限，不包括输入 prompt）
                       注意：这是输出限制，不是总 context 长度限制
            collect_log_prob: 是否采集 log probabilities
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
        """
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.collect_log_prob = collect_log_prob
        self._supports_log_prob = collect_log_prob
        self.max_retries = max(1, int(max_retries))
        self.timeout = timeout
        
        # 延迟导入 openai
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAITeacherLLM. "
                            "Install with: pip install openai")
        
        # 初始化 OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=api_base or "https://api.openai.com/v1",
            timeout=timeout,
            # Keep exactly one retry layer.  The explicit loop below records the
            # retry count and honours Retry-After; enabling SDK retries as well
            # silently multiplies the number of billable attempts.
            max_retries=0,
        )
        
        logger.info(f"[OpenAITeacherLLM] Initialized with model={model_name}, "
                   f"api_base={api_base or 'default'}, collect_log_prob={collect_log_prob}")

    @staticmethod
    def _jsonable_model(value: Any) -> Any:
        """Convert OpenAI SDK response models to JSON-safe plain values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): OpenAITeacherLLM._jsonable_model(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [OpenAITeacherLLM._jsonable_model(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return OpenAITeacherLLM._jsonable_model(
                model_dump(mode="json", exclude_none=True)
            )
        return str(value)

    @staticmethod
    def _safe_error_text(error: Exception) -> str:
        """Redact credential-shaped substrings before writing API errors."""
        text = str(error)
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-[REDACTED]", text)
        return text[:2000]

    @staticmethod
    def _retry_delay_seconds(error: Exception, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            try:
                if retry_after is not None:
                    return max(0.0, min(float(retry_after), 120.0))
            except (TypeError, ValueError):
                pass
        return min(2.0 ** attempt, 60.0) + random.uniform(0.0, 0.5)
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """
        调用 OpenAI-compatible API 生成响应。
        
        Args:
            messages: 对话历史
            **kwargs: 覆盖默认参数 (temperature, max_tokens, etc.)
            
        Returns:
            Tuple[str, Optional[Dict]]: (response_text, metadata)
        """
        for attempt in range(self.max_retries):
            started_at = time.monotonic()
            try:
                # 构建请求参数
                request_params = {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                if kwargs.get("top_p") is not None:
                    request_params["top_p"] = float(kwargs["top_p"])
                
                # 如果需要采集 log_prob（OpenAI API 支持）
                if self.collect_log_prob:
                    request_params["logprobs"] = True
                    request_params["top_logprobs"] = 1
                
                # 调用 API
                response = self.client.chat.completions.create(**request_params)
                latency_ms = (time.monotonic() - started_at) * 1000.0
                
                # 提取响应
                choice = response.choices[0]
                response_text = choice.message.content or ""

                # Reasoning 模型经 OpenRouter 返回时,思考内容在独立字段
                # (OpenRouter 归一化为 `reasoning`,DeepSeek 原生 API 为 `reasoning_content`),
                # 合并回 <think> 块使落盘轨迹保留完整教师推理;下游可用
                # filter_teacher_trajectories.py --strip_think 生成无思考版本。
                reasoning = (getattr(choice.message, "reasoning", None)
                             or getattr(choice.message, "reasoning_content", None))
                if reasoning and "<think>" not in response_text:
                    response_text = f"<think>\n{reasoning.strip()}\n</think>\n{response_text.lstrip()}"
                
                # 提取 log_prob（如果有）
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason is not None:
                    finish_reason = str(
                        getattr(finish_reason, "value", finish_reason)
                    )
                choice_extra = getattr(choice, "model_extra", None) or {}
                response_extra = getattr(response, "model_extra", None) or {}
                native_finish_reason = (
                    getattr(choice, "native_finish_reason", None)
                    or choice_extra.get("native_finish_reason")
                    or response_extra.get("native_finish_reason")
                )

                metadata = {
                    "finish_reason": finish_reason,
                    "native_finish_reason": native_finish_reason,
                    "response_id": getattr(response, "id", None),
                    "returned_model": getattr(response, "model", None),
                    "usage": self._jsonable_model(getattr(response, "usage", None)),
                    "latency_ms": latency_ms,
                    "retry_count": attempt,
                    "reasoning_present": bool(reasoning),
                }
                if self.collect_log_prob and choice.logprobs:
                    logprobs_content = choice.logprobs.content
                    if logprobs_content:
                        token_logprobs = []
                        tokens = []
                        for lp in logprobs_content:
                            token_logprobs.append(lp.logprob)
                            tokens.append(lp.token)
                        metadata["log_probs"] = token_logprobs
                        metadata["tokens"] = tokens
                
                return response_text, metadata
                
            except Exception as e:
                safe_error = self._safe_error_text(e)
                logger.warning(
                    f"[OpenAITeacherLLM] Attempt {attempt + 1}/"
                    f"{self.max_retries} failed: {safe_error}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self._retry_delay_seconds(e, attempt))
                else:
                    logger.error(f"[OpenAITeacherLLM] All {self.max_retries} attempts failed")
                    raise
    
    def reset(self):
        """OpenAI API 无状态，无需重置"""
        pass
    
    @property
    def supports_log_prob(self) -> bool:
        return self._supports_log_prob
    
    @property
    def model_name(self) -> str:
        return self._model_name
