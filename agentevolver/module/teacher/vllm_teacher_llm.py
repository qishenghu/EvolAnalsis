"""
vLLM Local Model Teacher LLM Backend.

This module provides the VLLMTeacherLLM class for using local open-source models
via vLLM inference engine (Qwen, Llama, DeepSeek, Mistral, etc.)

Also provides the factory function create_teacher_llm() for creating Teacher LLM instances.
"""

import threading
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger

from .base_teacher_llm import BaseTeacherLLM


class VLLMTeacherLLM(BaseTeacherLLM):
    """
    vLLM 开源模型 Teacher LLM 后端
    
    ⭐ 支持：Qwen、Llama、DeepSeek、Mistral 等本地开源模型
    
    关键特性：
    - 使用 vLLM 进行高效推理
    - 天然支持 log_prob 采集（开源模型可以获取完整概率分布）
    - 支持 tensor_parallel 多卡推理
    
    参考：VERL 框架的 vLLMRollout 实现
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer: Any = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        collect_log_prob: bool = True,  # 开源模型默认采集 log_prob
        trust_remote_code: bool = True,
        max_num_seqs: Optional[int] = None,  # vLLM 并发序列数
    ):
        """
        初始化 vLLM Teacher LLM。
        
        Args:
            model_path: 本地模型路径
            tokenizer: tokenizer 实例（可选，默认自动加载）
            temperature: 采样温度
            max_tokens: 最大生成 token 数（输出长度上限，不包括输入 prompt）
                       注意：这是输出限制，不是总 context 长度限制
            tensor_parallel_size: 张量并行大小（多 GPU）
            gpu_memory_utilization: GPU 内存使用率
            collect_log_prob: 是否采集 log probabilities
            trust_remote_code: 是否信任远程代码
            max_num_seqs: vLLM 最大并发序列数（None 使用 vLLM 默认值，通常 256-1024）
        """
        self._model_path = model_path
        self._model_name = model_path.split('/')[-1]  # 使用路径最后一部分作为名称
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.collect_log_prob = collect_log_prob
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.max_num_seqs = max_num_seqs
        
        # 延迟导入 vLLM
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("vLLM and transformers are required for VLLMTeacherLLM. "
                            "Install with: pip install vllm transformers")
        
        # 初始化 tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = tokenizer
        
        # 初始化 vLLM 引擎
        logger.info(f"[VLLMTeacherLLM] Loading model from {model_path}...")
        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self.llm = LLM(**llm_kwargs)
        
        # 保存 SamplingParams 类引用
        self._SamplingParams = SamplingParams
        
        # ⭐ 添加线程锁保护 vLLM 调用（vLLM 在高并发下可能不是完全线程安全的）
        self._llm_lock = threading.Lock()
        
        logger.info(f"[VLLMTeacherLLM] Initialized with model={model_path}, "
                   f"tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}, "
                   f"max_num_seqs={max_num_seqs}, collect_log_prob={collect_log_prob}")
    
    def __call__(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Tuple[str, Optional[Dict]]:
        """
        调用 vLLM 开源模型生成响应。
        
        Args:
            messages: 对话历史
            **kwargs: 覆盖默认参数
            
        Returns:
            Tuple[str, Optional[Dict]]: (response_text, metadata)
        """
        try:
            # 使用 chat template 格式化消息
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 创建采样参数
            sampling_params = self._SamplingParams(
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                logprobs=1 if self.collect_log_prob else None,
            )
            
            # ⭐ 使用锁保护 vLLM 调用（确保线程安全）
            # 注意：虽然这会降低并发性能，但可以避免 vLLM 内部状态冲突
            # vLLM 的 generate() 在高并发多线程环境下可能出现内部状态冲突
            with self._llm_lock:
                outputs = self.llm.generate([prompt], sampling_params)
            
            if not outputs or len(outputs) == 0:
                raise ValueError("vLLM returned empty outputs")
            
            output = outputs[0]
            
            # 检查输出是否有效
            if not output.outputs or len(output.outputs) == 0:
                raise ValueError("vLLM output.outputs is empty")
            
            # 提取响应文本
            response_text = output.outputs[0].text
            
            # 提取 log_prob（如果采集）
            metadata = {}
            if self.collect_log_prob and output.outputs[0].logprobs:
                logprobs_list = output.outputs[0].logprobs
                # vLLM 返回的 logprobs 是 List[Dict[token_id, Logprob]]
                token_logprobs = []
                tokens = []
                for lp_dict in logprobs_list:
                    if lp_dict:
                        # 获取生成的 token 的 logprob
                        for token_id, logprob_obj in lp_dict.items():
                            token_logprobs.append(logprob_obj.logprob)
                            tokens.append(logprob_obj.decoded_token)
                            break  # 只取第一个（实际生成的 token）
                
                metadata["log_probs"] = token_logprobs
                metadata["tokens"] = tokens
                # 同时保存 token_ids，方便后续对齐
                metadata["token_ids"] = list(output.outputs[0].token_ids) if output.outputs[0].token_ids else []
            
            return response_text, metadata if metadata else None
            
        except Exception as e:
            logger.error(f"[VLLMTeacherLLM] Generation failed: {e}")
            raise
    
    def reset(self):
        """vLLM 无状态，无需重置"""
        pass
    
    @property
    def supports_log_prob(self) -> bool:
        """vLLM 开源模型始终支持 log_prob"""
        return True
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def model_path(self) -> str:
        return self._model_path


# ============================================================
# 工厂函数：根据配置创建 Teacher LLM
# ============================================================

def create_teacher_llm(config: Dict[str, Any]) -> BaseTeacherLLM:
    """
    工厂函数：根据配置创建 Teacher LLM 实例。
    
    Args:
        config: Teacher LLM 配置字典
            - type: "openai" 或 "vllm"
            - 其他参数根据 type 决定
            
    Returns:
        BaseTeacherLLM 实例
        
    Example config for OpenAI:
        {
            "type": "openai",
            "model_name": "gpt-4",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-xxx",
            "temperature": 0.0,
            "max_tokens": 4096,
            "collect_log_prob": false
        }
        
    Example config for vLLM:
        {
            "type": "vllm",
            "model_path": "/path/to/qwen-72b",
            "temperature": 0.0,
            "max_tokens": 4096,
            "tensor_parallel_size": 4,
            "gpu_memory_utilization": 0.85,
            "collect_log_prob": true
        }
    """
    from .openai_teacher_llm import OpenAITeacherLLM
    
    llm_type = config.get("type", "openai").lower()
    
    if llm_type == "openai":
        return OpenAITeacherLLM(
            model_name=config.get("model_name", "gpt-4"),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096),
            collect_log_prob=config.get("collect_log_prob", False),
            max_retries=config.get("max_retries", 3),
            timeout=config.get("timeout", 120.0),
        )
    elif llm_type == "vllm":
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for vllm backend")
        return VLLMTeacherLLM(
            model_path=model_path,
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 4096),
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.85),
            collect_log_prob=config.get("collect_log_prob", True),
            trust_remote_code=config.get("trust_remote_code", True),
            max_num_seqs=config.get("max_num_seqs", None),
        )
    else:
        raise ValueError(f"Unsupported teacher LLM type: {llm_type}. "
                        f"Supported types: 'openai', 'vllm'")

