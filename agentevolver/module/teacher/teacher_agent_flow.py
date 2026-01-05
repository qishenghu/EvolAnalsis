"""
Teacher Agent Flow - Adapter for Teacher LLM to work with EnvWorker.

This module provides the TeacherAgentFlow class that wraps Teacher LLM
to be compatible with the BaseAgentFlow interface used by EnvWorker.
"""

from typing import List, Dict, Any, Optional, Callable

from omegaconf import DictConfig
from loguru import logger

from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from .base_teacher_llm import BaseTeacherLLM


class TeacherAgentFlow(BaseAgentFlow):
    """
    Teacher Agent Flow - 包装 Teacher LLM 以兼容 BaseAgentFlow 接口
    
    ⭐ 核心：作为 BaseTeacherLLM 和 EnvWorker 之间的适配层
    
    使用方式：
    1. 创建 TeacherAgentFlow 替代标准 AgentFlow
    2. TeacherAgentFlow 内部使用 Teacher LLM（OpenAI 或 vLLM）
    3. EnvWorker 无需修改，调用接口完全一致
    """
    
    def __init__(
        self,
        teacher_llm: BaseTeacherLLM,
        tokenizer: Any,
        config: DictConfig = None,
        **kwargs
    ):
        """
        初始化 Teacher Agent Flow。
        
        Args:
            teacher_llm: Teacher LLM 实例（OpenAI 或 vLLM）
            tokenizer: Tokenizer（用于与 BaseAgentFlow 兼容）
            config: 配置
        """
        self.teacher_llm = teacher_llm
        self._tokenizer = tokenizer
        self._config = config
        
        # 从配置中获取参数，如果没有配置则使用默认值
        if config is not None:
            self.max_steps = config.actor_rollout_ref.rollout.multi_turn.get("max_steps", 30)
            self.max_model_len = config.actor_rollout_ref.rollout.get("max_model_len", 8192)
            self.max_env_len = config.actor_rollout_ref.rollout.get("max_env_len", 4096)
        else:
            self.max_steps = 30
            self.max_model_len = 8192
            self.max_env_len = 4096
        
        # 存储最后一次调用的 metadata（包含 log_prob）
        self._last_metadata: Optional[Dict] = None
        
        logger.info(f"[TeacherAgentFlow] Initialized with teacher_llm={teacher_llm.model_name}, "
                   f"supports_log_prob={teacher_llm.supports_log_prob}")
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def config(self):
        return self._config
    
    @property
    def llm_chat_fn(self) -> Callable:
        """
        返回与标准 llm_chat_fn 兼容的函数。
        """
        return self._teacher_llm_chat_fn
    
    def _teacher_llm_chat_fn(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 Teacher LLM 并返回与标准 llm_chat_fn 兼容的格式。
        
        Args:
            messages: 对话历史
            **kwargs: 其他参数
            
        Returns:
            Dict with 'role' and 'content', optionally '_metadata'
        """
        response_text, metadata = self.teacher_llm(messages, **kwargs)
        
        # 存储 metadata 供后续使用
        self._last_metadata = metadata
        
        result = {
            "role": "assistant",
            "content": response_text,
        }
        
        # 如果有 metadata，添加到结果中
        if metadata:
            result["_metadata"] = metadata
        
        return result
    
    @property
    def supports_log_prob(self) -> bool:
        """是否支持 log_prob"""
        return self.teacher_llm.supports_log_prob
    
    @property
    def last_metadata(self) -> Optional[Dict]:
        """获取最后一次调用的 metadata（包含 log_prob）"""
        return self._last_metadata
    
    def get_llm_chat_fn(self, sampling_params: dict = None) -> Callable:
        """
        返回 LLM chat 函数（与 ParallelEnvManager.get_llm_chat_fn 接口一致）
        
        Args:
            sampling_params: 采样参数（可选）
            
        Returns:
            Callable: LLM chat 函数
        """
        def chat_fn(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
            # 合并采样参数
            merged_kwargs = {**(sampling_params or {}), **kwargs}
            return self._teacher_llm_chat_fn(messages, **merged_kwargs)
        
        return chat_fn
    
    def execute(self, *args, **kwargs):
        """
        执行方法 - 对于 Teacher trajectory 采集，实际执行由 EnvWorker 完成。
        这里提供一个占位符以保持接口兼容。
        """
        raise NotImplementedError(
            "TeacherAgentFlow.execute() should not be called directly. "
            "Use EnvWorker.execute() with TeacherAgentFlow's llm_chat_fn instead."
        )
    
    def reset(self):
        """重置状态"""
        self._last_metadata = None
        self.teacher_llm.reset()

