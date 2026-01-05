"""
Teacher LLM Module for LUFFY Teacher Experience Replay.

This module provides abstractions for different Teacher LLM backends:
- OpenAITeacherLLM: For OpenAI-compatible APIs (GPT-4, Claude, DashScope, etc.)
- VLLMTeacherLLM: For local vLLM models (Qwen, Llama, DeepSeek, etc.)
- TeacherAgentFlow: Adapter layer to integrate Teacher LLM with EnvWorker
"""

from .base_teacher_llm import BaseTeacherLLM
from .openai_teacher_llm import OpenAITeacherLLM
from .vllm_teacher_llm import VLLMTeacherLLM, create_teacher_llm
from .teacher_agent_flow import TeacherAgentFlow

__all__ = [
    "BaseTeacherLLM",
    "OpenAITeacherLLM", 
    "VLLMTeacherLLM",
    "create_teacher_llm",
    "TeacherAgentFlow",
]

