"""
Base Teacher LLM Abstract Class.

This module defines the abstract interface for Teacher LLM backends.
All Teacher LLM implementations must inherit from BaseTeacherLLM.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseTeacherLLM(ABC):
    """
    Teacher LLM 抽象基类
    
    ⭐ 核心：统一 OpenAI-compatible API 和 vLLM 开源模型的接口
    
    接口要求（与 EnvWorker 兼容）：
    - __call__(messages, **kwargs) -> Tuple[str, Optional[Dict]]
    - 返回：(response_text, metadata)
    """
    
    @abstractmethod
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """
        调用 Teacher LLM 生成响应。
        
        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            **kwargs: 其他参数 (temperature, max_tokens, etc.)
            
        Returns:
            Tuple[str, Optional[Dict]]:
                - response_text: LLM 生成的文本
                - metadata: 可选的元数据，包括：
                    - log_probs: List[float] - token 级别的 log probabilities
                    - tokens: List[str] - 对应的 tokens
                    - token_ids: List[int] - 对应的 token IDs
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置状态（如果有的话）"""
        pass
    
    @property
    @abstractmethod
    def supports_log_prob(self) -> bool:
        """是否支持返回 log_prob"""
        pass
    
    @property
    def model_name(self) -> str:
        """返回模型名称"""
        return getattr(self, '_model_name', 'unknown')

