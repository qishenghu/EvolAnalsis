#!/usr/bin/env python3
"""
Teacher Trajectory Collection Script

Collect trajectories from Teacher LLM (GPT-4, Claude, vLLM models) for LUFFY training.

Usage:
    # Using vLLM local model (recommended, with log_prob)
    python scripts/collect_teacher_trajectories.py \
        --env alfworld \
        --env_url http://localhost:8081 \
        --backend vllm \
        --model_path /data/models/Qwen2.5-72B-Instruct \
        --task_file data/alfworld/task_ids.txt \
        --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
        --n_per_task 2

    # Limit to maximum 100 tasks
    python scripts/collect_teacher_trajectories.py \
        --env alfworld \
        --env_url http://localhost:8081 \
        --backend vllm \
        --model_path /data/models/Qwen2.5-72B-Instruct \
        --task_file data/alfworld/task_ids.txt \
        --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
        --max_tasks 100 \
        --n_per_task 2

    # Using OpenAI API (no log_prob)
    python scripts/collect_teacher_trajectories.py \
        --env alfworld \
        --env_url http://localhost:8000 \
        --backend openai \
        --model_name gpt-4 \
        --api_base https://api.openai.com/v1 \
        --task_file data/alfworld/task_ids.txt \
        --output data/teacher_trajectories/alfworld_gpt4.jsonl \
        --collect_log_prob false
    
    # 大规模采集（2000 tasks × 10 rollouts，使用并行加速）
    python scripts/collect_teacher_trajectories.py \
        --env alfworld \
        --env_url http://localhost:8081 \
        --backend vllm \
        --model_path /data/models/Qwen2.5-72B-Instruct \
        --task_file data/alfworld/task_ids.txt \
        --output data/teacher_trajectories/alfworld_qwen72b.jsonl \
        --n_per_task 10 \
        --max_workers 16 \
        --save_every 100
"""

import argparse
import json
import os
import sys
import copy
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf, DictConfig

from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Reward
from agentevolver.client.env_client import EnvClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect teacher trajectories for LUFFY training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ===== 环境配置 =====
    parser.add_argument(
        "--env", type=str, required=True,
        choices=["alfworld", "webshop", "sciworld"],
        help="Environment name"
    )
    parser.add_argument(
        "--env_url", type=str, default="http://localhost:8000",
        help="Environment service URL"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to base config file (optional, for environment setup)"
    )
    
    # ===== Task 索引配置 =====
    parser.add_argument(
        "--task_file", type=str, required=True,
        help="Path to task IDs file (one task_id per line)"
    )
    parser.add_argument(
        "--task_start", type=int, default=0,
        help="Start index in task file (for distributed collection)"
    )
    parser.add_argument(
        "--task_end", type=int, default=None,
        help="End index in task file (exclusive, None for all)"
    )
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Maximum number of tasks to process (None for all). Applied after task_start/task_end."
    )
    
    # ===== Teacher LLM 后端配置 =====
    parser.add_argument(
        "--backend", type=str, required=True,
        choices=["openai", "vllm"],
        help="Teacher LLM backend: 'openai' for API, 'vllm' for local model"
    )
    
    # OpenAI 后端参数
    parser.add_argument(
        "--model_name", type=str, default="gpt-4",
        help="[OpenAI] Model name (e.g., gpt-4, gpt-4-turbo, claude-3-opus)"
    )
    parser.add_argument(
        "--api_base", type=str, default="https://api.openai.com/v1",
        help="[OpenAI] API base URL"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="[OpenAI] API key (default: use OPENAI_API_KEY env var)"
    )
    
    # vLLM 后端参数
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="[vLLM] Path to local model"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="[vLLM] Tensor parallel size for multi-GPU inference"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.85,
        help="[vLLM] GPU memory utilization ratio"
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=None,
        help="[vLLM] Maximum number of sequences processed in parallel (default: vLLM default, typically 256-1024)"
    )
    
    # ===== 模型特性配置 =====
    parser.add_argument(
        "--use_qwen3", action="store_true", default=False,
        help="Enable Qwen3 native thinking mode (appends /no_think to non-final turns)"
    )
    parser.add_argument(
        "--success_threshold", type=float, default=None,
        help="Success threshold for evaluate(). None=use env default. "
             "For SciWorld set to 70 (score scale 0-100, matching training config)."
    )

    # ===== 通用生成参数 =====
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--collect_log_prob", type=str, default="auto",
        choices=["true", "false", "auto"],
        help="Collect log probabilities. 'auto': true for vllm, false for openai"
    )
    
    # ===== 采集配置 =====
    parser.add_argument(
        "--n_per_task", type=int, default=1,
        help="Number of trajectories to collect per task (used as max attempts in stop_on_success mode)"
    )
    parser.add_argument(
        "--stop_on_success", action="store_true", default=False,
        help="Stop rolling for a task after getting the first successful trajectory. "
             "n_per_task becomes the max attempts limit. Most efficient for targeted collection."
    )
    parser.add_argument(
        "--filter_success", action="store_true", default=True,
        help="Only keep successful trajectories"
    )
    parser.add_argument(
        "--no_filter_success", action="store_false", dest="filter_success",
        help="Keep all trajectories including failed ones"
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Maximum retries per task on failure"
    )
    parser.add_argument(
        "--max_steps", type=int, default=30,
        help="Maximum steps per trajectory"
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="Maximum parallel workers for trajectory collection (recommended: 8-16 for 2000 tasks)"
    )
    
    # ===== 输出配置 =====
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file (skip already collected tasks)"
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save checkpoint every N tasks"
    )
    
    return parser.parse_args()


def load_task_ids(task_file: str, start: int = 0, end: Optional[int] = None) -> List[str]:
    """从文件加载 task IDs"""
    with open(task_file, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if end is None:
        end = len(task_ids)
    task_ids = task_ids[start:end]
    
    logger.info(f"Loaded {len(task_ids)} task IDs from {task_file} (range: [{start}, {end}))")
    return task_ids


def load_completed_task_ids(output_file: str) -> set:
    """从已有输出文件中加载已完成的 task IDs（用于断点续采）"""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        traj = json.loads(line)
                        completed.add(traj.get("task_id", ""))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Found {len(completed)} already completed tasks in {output_file}")
    return completed


def create_minimal_config(env_name: str, env_url: str, max_steps: int = 30) -> DictConfig:
    """创建最小配置用于 rollout
    
    参考 config/alfworld_grpo_3b.yaml 中的配置
    """
    config = OmegaConf.create({
        "env_service": {
            "env_url": env_url,
            "env_type": env_name,
        },
        "trainer": {
            "experiment_name": "teacher_trajectory_collection",
        },
        "actor_rollout_ref": {
            "rollout": {
                # ⭐ 关键参数：必须与训练配置一致
                "prompt_length": 20480,
                "response_length": 2048,
                "max_model_len": 25580,
                "max_env_len": 4096,
                # Context 管理
                "context_template": "linear",
                "context_template_train_sp_action": False,
                # Rollout 控制
                "sparse": True,
                "debug_llm_io": False,
                "use_qwen3": args.use_qwen3,
                "enable_request_id": False,
                # Multi-turn 配置
                "multi_turn": {
                    "enable": True,
                    "max_steps": max_steps,
                    "format": "llama3_json",
                },
            }
        },
        "data": {
            "max_prompt_length": 4000,
            "max_response_length": 21580,
        },
        "exp_manager": {
            "experience_replay": {
                "enable": False,
            }
        },
    })
    return config


class TeacherRolloutExecutor:
    """
    Teacher 轨迹采集执行器
    
    复用 AgentFlow 的 rollout 逻辑，只替换 LLM 调用部分。
    """
    
    def __init__(
        self,
        teacher_llm,
        tokenizer,
        config: DictConfig,
        env_url: str,
        collect_log_prob: bool = True,
        success_threshold: Optional[float] = None,
    ):
        self.teacher_llm = teacher_llm
        self.tokenizer = tokenizer
        self.config = config
        self.env_url = env_url
        self.collect_log_prob = collect_log_prob
        self.success_threshold = success_threshold
        
        # ⭐ 存储每轮 LLM 调用的 log_probs（分轮存储，便于对齐）
        # 格式：[{"turn_idx": int, "log_probs": [...], "token_ids": [...], "tokens": [...]}]
        self._turn_log_probs: List[Dict[str, Any]] = []
        
        self.max_steps = config.actor_rollout_ref.rollout.multi_turn.max_steps
        self.max_model_len = config.actor_rollout_ref.rollout.max_model_len
        self.max_env_len = config.actor_rollout_ref.rollout.max_env_len
        
        logger.info(f"[TeacherRolloutExecutor] Initialized with max_steps={self.max_steps}")
    
    def get_llm_chat_fn(self) -> Callable:
        """返回可用于 rollout 的 LLM chat 函数"""
        def llm_chat_fn(messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
            # 调用 Teacher LLM
            response_text, metadata = self.teacher_llm(messages, **kwargs)
            
            # ⭐ 分轮收集 log_prob（保留 token_ids 便于后续对齐）
            if metadata and "log_probs" in metadata:
                turn_info = {
                    "turn_idx": len(self._turn_log_probs),
                    "log_probs": metadata["log_probs"],
                    "token_ids": metadata.get("token_ids", []),
                    "tokens": metadata.get("tokens", []),
                }
                self._turn_log_probs.append(turn_info)
            
            return {
                "role": "assistant",
                "content": response_text,
            }
        
        return llm_chat_fn
    
    def reset_log_probs(self):
        """重置累积的 log_probs"""
        self._turn_log_probs = []
    
    def get_turn_log_probs(self) -> List[Dict[str, Any]]:
        """获取分轮存储的 log_probs"""
        return self._turn_log_probs.copy()
    
    def get_accumulated_log_probs(self) -> List[float]:
        """获取累积的 log_probs（仅 LLM 生成的 token）"""
        accumulated = []
        for turn in self._turn_log_probs:
            accumulated.extend(turn["log_probs"])
        return accumulated
    
    def execute_rollout(
        self,
        task: Task,
        rollout_id: str,
        thread_index: int = 0,
    ) -> Optional[Trajectory]:
        """
        执行单个任务的 rollout。
        
        Args:
            task: 任务对象
            rollout_id: rollout ID
            thread_index: 线程索引
            
        Returns:
            Trajectory 对象，如果失败则返回 None
        """
        from agentevolver.module.context_manager.cmt_linear import Linear_CMT
        from agentevolver.utils.utils import convert_tool_to_user_message
        
        self.reset_log_probs()
        
        env = EnvClient(base_url=self.env_url)
        instance_id = uuid.uuid4().hex
        
        try:
            # 1. 创建环境实例
            init_response = env.create_instance(
                env_type=task.env_type,
                task_id=task.task_id,
                instance_id=instance_id,
                params={'is_open_query': task.open_query}
            )
            
            init_messages: list = init_response["state"]
            assert isinstance(init_messages, list) and len(init_messages) in [2, 3], \
                f"init_messages must be list and its length must be 2 or 3, got {len(init_messages)}"
            
            # 获取 query
            if task.query is not None:
                init_messages[-1]["content"] = task.query
            else:
                task.query = init_messages[-1]["content"]
            
            # 2. 创建 Context Manager
            cmt = Linear_CMT(self.config, self.tokenizer)
            cmt.data_id = task.task_id
            cmt.rollout_id = rollout_id
            cmt.task_id = task.task_id
            cmt.instance_id = instance_id
            cmt.query = task.query
            
            # 初始化消息
            add_nothink = self.config.actor_rollout_ref.rollout.get("use_qwen3", True)
            cmt.save_init_input(init_messages, add_nothink)
            
            # 3. 执行 rollout 循环
            llm_chat_fn = self.get_llm_chat_fn()
            err_in_env = False
            
            for act_step in range(self.max_steps):
                # 准备 LLM 输入
                try:
                    step_input_message_arr = cmt.prepare_next_llm_context()
                except Exception as e:
                    logger.error(f"Error preparing context at step {act_step}: {e}")
                    break
                
                # 检查 token 溢出
                is_safe = cmt.check_context_token_num_safe(step_input_message_arr)
                if not is_safe:
                    logger.warning(f"Token overflow detected at step {act_step}.")
                    cmt.is_terminated = False
                    break
                
                # 调用 LLM
                try:
                    llm_output = llm_chat_fn(step_input_message_arr)
                except Exception as e:
                    logger.error(f"Error calling LLM at step {act_step}: {e}")
                    break
                
                # 保存 LLM 输出
                cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)
                
                # 与环境交互
                try:
                    env_output = env.step(
                        instance_id, 
                        {"content": cmt.prepare_world_interaction(), "role": "assistant"}
                    )
                    assert len(env_output['state']) == 1
                    env_output["state"] = env_output["state"][0]
                    if env_output["state"]["role"] == "tool":
                        env_output["state"] = convert_tool_to_user_message(
                            env_output["state"], self.tokenizer, format="qwen"
                        )
                except Exception as e:
                    logger.error(f"Error in env.step at step {act_step}: {e}")
                    err_in_env = True
                    cmt.is_terminated = False
                    state = {"content": str(e), "role": "user"}
                    env_output = {
                        "reward": 0,
                        "is_terminated": True,
                        "state": state,
                    }
                
                # 保存环境输出
                state = env_output["state"]
                state.pop('tool_calls', None)
                cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)
                
                # 检查是否终止
                cmt.is_terminated = env_output["is_terminated"]
                if cmt.is_terminated or err_in_env:
                    break
            
            # 4. 计算 reward
            try:
                eval_params = {"sparse": True}
                if self.success_threshold is not None:
                    eval_params["sciworld_success_threshold"] = self.success_threshold
                score = env.evaluate(instance_id, params=eval_params)
            except Exception as e:
                logger.warning(f"Error evaluating task {task.task_id}: {e}")
                score = 0.0
            
            if score >= 1:
                success_rate = 1.0
            else:
                success_rate = 0.0
            
            cmt.reward = Reward(
                outcome=score, 
                success_rate=success_rate, 
                madness=cmt.compute_madness(), 
                description="Teacher trajectory"
            )
            cmt.remove_last_context()
            
            # 释放环境实例
            env.release_instance(instance_id)
            
            return cmt
            
        except Exception as e:
            logger.error(f"Error in rollout for task {task.task_id}: {e}")
            try:
                env.release_instance(instance_id)
            except:
                pass
            return None
    
    def trajectory_to_dict(self, traj, teacher_model_name: str) -> Dict[str, Any]:
        """将轨迹转换为可序列化的字典"""
        # 获取 messages
        # ⭐ 使用 Linear_CMT 的 to_role_content 方法或直接访问 ExtendedMessage 属性
        messages = []
        if hasattr(traj, 'to_role_content'):
            # 使用 CMT 的内置方法转换
            messages = traj.to_role_content(traj.full_context)
        else:
            # 手动转换 ExtendedMessage 对象
            for step in traj.full_context:
                if isinstance(step, dict):
                    messages.append(step)
                elif hasattr(step, 'role') and hasattr(step, 'content'):
                    # ExtendedMessage 对象有 role 和 content 属性
                    # 优先使用 content_for_future（如果存在），否则使用 content
                    content = getattr(step, 'content_for_future', None) or getattr(step, 'content', '')
                    messages.append({"role": step.role, "content": content})
                else:
                    messages.append({"role": "unknown", "content": str(step)})
        
        # ⭐ 获取 log_probs（分轮存储和累积两种格式）
        turn_log_probs = self.get_turn_log_probs() if self.collect_log_prob else []
        accumulated_log_probs = self.get_accumulated_log_probs() if self.collect_log_prob else []
        has_log_prob = len(accumulated_log_probs) > 0
        
        traj_dict = {
            "task_id": traj.task_id,
            "data_id": traj.data_id,
            "rollout_id": traj.rollout_id,
            "messages": messages,
            "reward": traj.reward.outcome if traj.reward else 0.0,
            "success": traj.reward.outcome == 1.0 if traj.reward else False,
            "is_terminated": traj.is_terminated,
            "teacher_model": teacher_model_name,
            "metadata": {
                "is_teacher": True,
                "has_log_prob": has_log_prob,
                "collected_at": datetime.now().isoformat(),
            },
        }
        
        # ⭐ 保存两种格式的 log_probs：
        # 1. log_probs: 累积的 log_probs（仅 LLM 生成的 token，便于简单使用）
        # 2. log_probs_per_turn: 分轮的详细信息（包含 token_ids，便于精确对齐）
        #
        # 对于 multi-turn 场景：
        # - 使用相同 tokenizer 的模型（如 Qwen-3B 学习 Qwen-72B）：
        #   可以利用 token_ids 进行精确对齐
        # - 使用不同 tokenizer 的模型：
        #   应设置 use_log_prob=False，使用 LUFFY 方式（假设 π_old=1）
        if has_log_prob:
            traj_dict["log_probs"] = accumulated_log_probs  # 兼容旧格式
            traj_dict["log_probs_per_turn"] = turn_log_probs  # 新格式：分轮详细信息
            # 保存总 token 数（便于验证对齐）
            traj_dict["metadata"]["total_generated_tokens"] = len(accumulated_log_probs)
            traj_dict["metadata"]["num_turns"] = len(turn_log_probs)
        
        return traj_dict


def main():
    args = parse_args()
    
    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    log_file = args.output.replace('.jsonl', '.log')
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="100 MB")
    
    logger.info("=" * 60)
    logger.info("Teacher Trajectory Collection")
    logger.info("=" * 60)
    logger.info(f"Environment: {args.env}")
    logger.info(f"Environment URL: {args.env_url}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Task file: {args.task_file}")
    logger.info(f"Output: {args.output}")
    
    # 1. 加载 task IDs
    task_ids = load_task_ids(args.task_file, args.task_start, args.task_end)
    
    # 2. 应用 max_tasks 限制
    if args.max_tasks is not None and args.max_tasks > 0:
        original_count = len(task_ids)
        task_ids = task_ids[:args.max_tasks]
        logger.info(f"Limited to {len(task_ids)} tasks (from {original_count}) by --max_tasks={args.max_tasks}")
    
    # 3. 断点续采：加载已完成的 tasks
    if args.resume:
        completed = load_completed_task_ids(args.output)
        task_ids = [tid for tid in task_ids if tid not in completed]
        logger.info(f"Resuming: {len(task_ids)} tasks remaining")
    
    if not task_ids:
        logger.info("No tasks to collect. Exiting.")
        return
    
    # 4. 确定是否收集 log_prob
    if args.collect_log_prob == "auto":
        collect_log_prob = (args.backend == "vllm")
    else:
        collect_log_prob = (args.collect_log_prob == "true")
    
    logger.info(f"Collect log_prob: {collect_log_prob}")
    
    # 5. 创建配置
    config = create_minimal_config(args.env, args.env_url, args.max_steps)
    
    # 6. 创建 Teacher LLM
    from agentevolver.module.teacher import create_teacher_llm
    
    if args.backend == "openai":
        llm_config = {
            "type": "openai",
            "model_name": args.model_path,
            "api_base": args.api_base,
            "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "collect_log_prob": collect_log_prob,
        }
    else:  # vllm
        if not args.model_path:
            raise ValueError("--model_path is required for vllm backend")
        llm_config = {
            "type": "vllm",
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "collect_log_prob": collect_log_prob,
        }
        if args.max_num_seqs is not None:
            llm_config["max_num_seqs"] = args.max_num_seqs
    
    teacher_llm = create_teacher_llm(llm_config)
    logger.info(f"Teacher LLM: {teacher_llm.model_name}")
    logger.info(f"Teacher LLM supports log_prob: {teacher_llm.supports_log_prob}")
    
    # 7. 加载 tokenizer
    from transformers import AutoTokenizer
    
    if args.backend == "vllm":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        # OpenAI 后端使用一个通用 tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 8. 创建 Rollout 执行器（每个线程需要独立的执行器实例）
    # 注意：对于 vLLM 后端，多个线程共享同一个 LLM 实例是安全的（vLLM 内部处理并发）
    # 对于 OpenAI 后端，多个线程并发调用也是安全的
    
    output_file = args.output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # 线程安全的计数器和锁
    successful_count = [0]  # 使用列表以便在闭包中修改
    failed_count = [0]
    write_lock = threading.Lock()
    collected_trajectories = []  # 线程安全的列表（append 操作是原子的）
    checkpoint_counter = [0]  # 用于定期保存
    
    def process_single_rollout(task_id: str, rollout_idx: int, thread_index: int) -> Optional[Dict[str, Any]]:
        """处理单个 (task, rollout) 组合"""
        # 为每个线程创建独立的执行器实例（避免状态冲突）
        executor = TeacherRolloutExecutor(
            teacher_llm=teacher_llm,  # LLM 实例是线程安全的
            tokenizer=tokenizer,
            config=config,
            env_url=args.env_url,
            collect_log_prob=collect_log_prob,
            success_threshold=args.success_threshold,
        )
        
        task = Task(
            task_id=task_id,
            env_type=args.env,
            open_query=False,
            evaluator="env",
        )
        
        rollout_id = f"{task_id}_teacher_{rollout_idx}"
        
        for retry in range(args.max_retries):
            try:
                traj = executor.execute_rollout(
                    task=task,
                    rollout_id=rollout_id,
                    thread_index=thread_index,
                )
                
                if traj is None:
                    logger.warning(f"Task {task_id} rollout {rollout_idx} returned None, "
                                 f"retry {retry+1}/{args.max_retries}")
                    continue
                
                # 检查是否成功
                success = traj.reward and traj.reward.outcome >= 1.0
                
                # 如果需要过滤成功，且当前失败，则重试
                if args.filter_success and not success:
                    logger.debug(f"Task {task_id} rollout {rollout_idx} failed (reward={traj.reward.outcome if traj.reward else 0}), "
                               f"retry {retry+1}/{args.max_retries}")
                    continue
                
                # 转换为字典
                traj_dict = executor.trajectory_to_dict(traj, teacher_llm.model_name)
                
                logger.debug(f"Task {task_id} rollout {rollout_idx}: success={success}, "
                           f"steps={len(traj_dict['messages'])}")
                return traj_dict  # 成功，返回结果
                
            except Exception as e:
                logger.warning(f"Task {task_id} rollout {rollout_idx} error: {e}, "
                             f"retry {retry+1}/{args.max_retries}")
                if retry == args.max_retries - 1:
                    logger.error(f"Task {task_id} rollout {rollout_idx} failed after "
                               f"{args.max_retries} retries")
        
        return None  # 所有重试都失败
    
    # 9. 准备工作项
    def process_task_until_success(task_id: str, thread_index: int) -> Optional[Dict[str, Any]]:
        """stop_on_success 模式：对单个 task 持续 rollout 直到成功或达到上限"""
        for attempt in range(args.n_per_task):
            result = process_single_rollout(task_id, attempt, thread_index)
            if result is not None:
                logger.info(f"Task {task_id}: success on attempt {attempt+1}/{args.n_per_task}")
                return result
        logger.warning(f"Task {task_id}: no success after {args.n_per_task} attempts")
        return None

    if args.stop_on_success:
        # stop_on_success 模式：每个 task 一个工作项，内部循环 rollout
        all_work_items = [(task_id,) for task_id in task_ids]
        total_work = len(all_work_items)
        logger.info(f"Total work items: {total_work} tasks (stop_on_success, max {args.n_per_task} attempts each)")
    else:
        # 原始模式：每个 (task, rollout) 一个工作项
        all_work_items = []
        for task_id in task_ids:
            for rollout_idx in range(args.n_per_task):
                all_work_items.append((task_id, rollout_idx))
        total_work = len(all_work_items)
        logger.info(f"Total work items: {total_work} (tasks={len(task_ids)}, rollouts_per_task={args.n_per_task})")
    logger.info(f"Using {args.max_workers} parallel workers")
    
    # 10. 并行采集
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_work, desc="Collecting trajectories")
    except ImportError:
        pbar = None
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor_pool:
        # 提交所有任务
        future_to_item = {}
        thread_index = 0
        for item in all_work_items:
            if args.stop_on_success:
                task_id = item[0]
                future = executor_pool.submit(process_task_until_success, task_id, thread_index)
                future_to_item[future] = (task_id, 0)
            else:
                task_id, rollout_idx = item
                future = executor_pool.submit(process_single_rollout, task_id, rollout_idx, thread_index)
                future_to_item[future] = (task_id, rollout_idx)
            thread_index += 1

        # 收集结果
        for future in as_completed(future_to_item):
            task_id, rollout_idx = future_to_item[future]
            try:
                traj_dict = future.result()
                if traj_dict is not None:
                    # 线程安全地添加轨迹和更新计数器
                    with write_lock:
                        collected_trajectories.append(traj_dict)
                        checkpoint_counter[0] += 1
                        current_count = checkpoint_counter[0]
                        
                        # 更新成功/失败计数
                        success = traj_dict.get("success", False)
                        if success:
                            successful_count[0] += 1
                        else:
                            failed_count[0] += 1
                        
                        # 定期保存（在锁内检查，避免竞态条件）
                        if current_count % args.save_every == 0:
                            mode = 'a' if args.resume or current_count > args.save_every else 'w'
                            with open(output_file, mode) as f:
                                # 保存所有未保存的轨迹
                                for traj in collected_trajectories:
                                    f.write(json.dumps(traj, ensure_ascii=False) + '\n')
                            logger.info(f"Checkpoint saved: {current_count}/{total_work} items, "
                                       f"saved {len(collected_trajectories)} trajectories")
                            collected_trajectories.clear()  # 清空已保存的
                
                if pbar:
                    pbar.update(1)
                    
            except Exception as e:
                logger.error(f"Error processing task {task_id} rollout {rollout_idx}: {e}")
                if pbar:
                    pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # 保存剩余轨迹（线程安全）
    with write_lock:
        if collected_trajectories:
            mode = 'a' if args.resume else 'w'
            with open(output_file, mode) as f:
                for traj in collected_trajectories:
                    f.write(json.dumps(traj, ensure_ascii=False) + '\n')
            logger.info(f"Saved remaining {len(collected_trajectories)} trajectories")
    
    # 11. 统计结果
    if os.path.exists(output_file):
        total_collected = sum(1 for _ in open(output_file))
    else:
        total_collected = 0
    
    logger.info("=" * 60)
    logger.info("Collection completed!")
    logger.info(f"Total trajectories collected: {total_collected}")
    logger.info(f"Successful: {successful_count[0]}, Failed: {failed_count[0]}")
    logger.info(f"Output file: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
