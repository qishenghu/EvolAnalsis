"""
OpenAI-Compatible API Teacher LLM Backend.

This module provides the OpenAITeacherLLM class for calling OpenAI-compatible APIs
including GPT-4, Claude via OpenAI API, DashScope, DeepSeek API, vLLM Server, etc.
"""

import os
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
        self.max_retries = max_retries
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
            max_retries=max_retries,
        )
        
        logger.info(f"[OpenAITeacherLLM] Initialized with model={model_name}, "
                   f"api_base={api_base or 'default'}, collect_log_prob={collect_log_prob}")
    
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
        import time
        
        for attempt in range(self.max_retries):
            try:
                # 构建请求参数
                request_params = {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                
                # 如果需要采集 log_prob（OpenAI API 支持）
                if self.collect_log_prob:
                    request_params["logprobs"] = True
                    request_params["top_logprobs"] = 1
                
                # 调用 API
                response = self.client.chat.completions.create(**request_params)
                
                # 提取响应
                choice = response.choices[0]
                response_text = choice.message.content or ""
                
                # 提取 log_prob（如果有）
                metadata = {}
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
                
                return response_text, metadata if metadata else None
                
            except Exception as e:
                logger.warning(f"[OpenAITeacherLLM] Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
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

