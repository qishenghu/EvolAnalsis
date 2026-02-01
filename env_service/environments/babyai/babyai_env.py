# -*- coding: utf-8 -*-
"""
BabyAI Environment integration for AgentEvolver.

通过 HTTP 方式复用 AgentGym 提供的 BabyAI 服务器。
BabyAI 是一个基于 MiniGrid 的指令跟随环境，
agent 需要在网格世界中完成各种导航和操作任务。
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

import requests

from env_service.base import BaseEnv
from env_service.registry import Registry


@Registry.register("babyai")
class BabyaiEnv(BaseEnv):
    """
    BabyAI environment wrapper for AgentEvolver.
    
    This class adapts AgentGym's BabyAI environment to the BaseEnv interface,
    allowing it to be used with Experience Pool and GRPO training.
    
    BabyAI 包含 40 个不同难度的关卡，任务包括：
    - GoTo: 导航到特定目标
    - Open: 打开门
    - Pickup: 拾取物品
    - PutNext: 放置物品
    - Unlock: 解锁门
    - 以及各种组合任务
    """
    
    # BabyAI 有 40 个关卡，每个关卡可以用不同的种子生成不同实例
    # 总任务数 = 40 levels * seeds
    NUM_LEVELS = 40

    # Align with AgentGym paper / eval protocol (user-provided):
    # Train uses a curated subset of 18 BabyAI levels.
    BABYAI_ALLOWED_CATEGORIES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 21, 30, 31, 33, 36]
    # In AgentGym, BabyAI train indices are drawn from [0, 1804) with the level determined by (i % 40 + 1).
    BABYAI_BASE_TRAIN_SIZE = 1804
    
    def __init__(self, task_id: str = None, instance_id: str = None, params: Dict[str, Any] = None):
        """
        Initialize the BabyAI environment.
        
        Args:
            task_id (str): The task ID (data_idx in BabyAI format).
                          data_idx = level_idx * seeds + seed
            instance_id (str): The instance ID for this environment.
            params (Dict[str, Any]): Additional parameters including:
                - server_url: URL of the BabyAI server
        """
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}

        # 外部 BabyAI HTTP 服务器地址
        self.server_url: str = (
            self.params.get("server_url")
            or os.environ.get("BABYAI_SERVER_URL", "http://127.0.0.1:36002")
        ).rstrip("/")

        # 远端 BabyAI server 上的 env id
        self.remote_env_id: Optional[int] = None
        self.current_data_idx = None
        
        # Store state info
        self.current_observation = None
        self.current_available_actions = []
        self.is_done = False
        self.current_reward = 0.0
        self.current_score = 0.0

    # ---------------------------
    # Internal HTTP helpers
    # ---------------------------

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        resp = requests.post(url, json=json_body, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: Dict[str, Any] = None) -> Any:
        url = f"{self.server_url}{path}"
        resp = requests.get(url, params=params, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    def _ensure_remote_env(self):
        """确保在远端 BabyAI server 上创建了一个 env id。"""
        if self.remote_env_id is not None:
            return
        # AgentGym 的 /create 接口返回 {"id": int}
        data = self._post("/create", {})
        if "id" not in data:
            raise RuntimeError(f"BabyAI HTTP /create returned invalid data: {data}")
        self.remote_env_id = int(data["id"])
        print(f"Created BabyAI environment {self.remote_env_id}")
        
    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        Args:
            params (Dict[str, Any]): Additional parameters:
                - data_idx: Index to specify level and seed
        
        Returns:
            Dict[str, Any]: Initial state with "state" (list of messages) and other info.
        """
        params = params or {}
        
        # Determine data_idx
        if self.task_id is not None:
            try:
                data_idx = int(self.task_id)
            except ValueError:
                data_idx = 0
        else:
            data_idx = params.get("data_idx", 0)
        
        # 确保远端 env 已创建
        self._ensure_remote_env()

        # 调用 AgentGym 的 /reset 接口
        reset_payload = {
            "id": self.remote_env_id,
            "data_idx": int(data_idx),
        }
        reset_result = self._post("/reset", reset_payload)

        if "error" in reset_result:
            raise RuntimeError(
                f"Failed to reset BabyAI environment: {reset_result['error']}"
            )
        
        self.current_data_idx = data_idx
        self.current_observation = reset_result.get("observation", "")
        self.is_done = reset_result.get("done", False)
        self.current_reward = reset_result.get("reward", 0.0)
        self.current_score = reset_result.get("score", 0.0)
        
        # 解析 available actions
        self.current_available_actions = self._parse_available_actions(self.current_observation)
        
        # Format initial state as messages
        init_messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "assistant",
                "content": "OK. I'll follow your instructions and try my best to solve the task."
            },
            {
                "role": "user",
                "content": self.current_observation
            }
        ]
        
        return {
            "instance_id": self.instance_id,
            "state": init_messages,
            "available_actions": self.current_available_actions,
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for BabyAI environment."""
        return '''You are an intelligent agent in a grid-based environment. Your task is to navigate and interact with objects to achieve the given goal.

At each step, you will receive:
1. Your current goal (e.g., "go to the red ball", "pick up the blue key")
2. A description of what you can see in front of you
3. A list of available actions

You should choose from two actions: "THOUGHT" or "ACTION".
- If you choose "THOUGHT", first think about the current condition and plan, then output your action.
  Format: "Thought:\nyour thoughts.\n\nAction:\nyour next action"
- If you choose "ACTION", directly output the action.
  Format: "Action:\nyour next action"

Important:
1. The action must be chosen from the given available actions.
2. Actions include: "turn left", "turn right", "move forward", "pickup [object]", "drop", "toggle", "go to [object]", "go through [door]", etc.
3. Think strategically about the shortest path to your goal.
4. Pay attention to obstacles and locked doors.'''

    def _parse_available_actions(self, observation: str) -> List[str]:
        """Parse available actions from observation text."""
        actions = []
        # BabyAI observation format includes "Available actions: [...]"
        match = re.search(r'Available actions:\s*\[(.*?)\]', observation, re.DOTALL)
        if match:
            action_str = match.group(1)
            # Parse individual actions
            actions = [a.strip().strip('"\'') for a in action_str.split(',') if a.strip()]
        return actions
    
    def step(self, action: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a step in the environment.
        
        Args:
            action (Dict[str, Any]): Action to execute.
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            Dict[str, Any]: Step result with state, reward, is_terminated, info.
        """
        params = params or {}
        
        self._ensure_remote_env()
        
        # Parse action string
        if isinstance(action, dict):
            action_str = action.get("content", action.get("action", ""))
        elif isinstance(action, str):
            action_str = action
        else:
            raise ValueError(f"Invalid action format: {action}")
        
        if not action_str:
            raise ValueError("Action string cannot be empty")
        
        # Parse "Action:" from LLM output
        parsed_action = self._parse_action_from_llm_output(action_str)
        
        # Validate action
        if parsed_action is None:
            available_actions = self.current_available_actions or []
            invalid_obs = "Invalid action format. Please use 'Action: your_action' format."
            if available_actions:
                invalid_obs += "\nAvailable actions: " + str(available_actions)
            
            return {
                "state": [{"role": "user", "content": invalid_obs}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {
                    "available_actions": available_actions,
                    "invalid_action": True,
                },
                "instance_id": self.instance_id,
            }
        
        # Check if action is valid
        if self.current_available_actions and parsed_action.lower() not in [a.lower() for a in self.current_available_actions]:
            available_actions = self.current_available_actions
            invalid_obs = f"Action '{parsed_action}' is not available."
            invalid_obs += "\nAvailable actions: " + str(available_actions)
            
            return {
                "state": [{"role": "user", "content": invalid_obs}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {
                    "available_actions": available_actions,
                    "invalid_action": True,
                },
                "instance_id": self.instance_id,
            }

        # 调用 AgentGym 的 /step 接口
        step_payload = {
            "id": self.remote_env_id,
            "action": parsed_action,
        }
        step_result = self._post("/step", step_payload)

        if "error" in step_result:
            raise RuntimeError(f"Step failed: {step_result['error']}")
        
        # Update state
        self.current_observation = step_result.get("observation", "")
        self.is_done = step_result.get("done", False)
        self.current_reward = step_result.get("reward", 0.0)
        self.current_score = step_result.get("score", 0.0)
        self.current_available_actions = self._parse_available_actions(self.current_observation)
        
        # Format response
        state_messages = [
            {
                "role": "user",
                "content": self.current_observation
            }
        ]
        
        return {
            "state": state_messages,
            "reward": self.current_reward,
            "is_terminated": self.is_done,
            "info": {
                "available_actions": self.current_available_actions,
                "score": self.current_score,
            },
            "instance_id": self.instance_id,
        }
    
    def evaluate(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> float:
        """
        Evaluate the performance of the environment.
        
        Returns:
            float: Evaluation score (0.0 to 1.0).
        """
        # BabyAI's score is already normalized
        return float(self.current_score) if self.current_score else float(self.current_reward)
    
    def get_info(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about the environment.
        """
        return {
            "env_id": "babyai",
            "task_id": self.task_id,
            "data_idx": self.current_data_idx,
            "observation": self.current_observation,
            "available_actions": self.current_available_actions,
            "done": self.is_done,
            "reward": self.current_reward,
            "score": self.current_score,
        }
    
    def close(self):
        """
        Close the environment and release resources.
        """
        if self.remote_env_id is not None:
            try:
                print(f"Closing BabyAI environment {self.remote_env_id}")
                close_payload = {"id": self.remote_env_id}
                self._post("/close", close_payload)
                print(f"BabyAI environment {self.remote_env_id} closed successfully.")
            except Exception as e:
                print(f"Error closing BabyAI environment {self.remote_env_id}: {e}")
            finally:
                self.remote_env_id = None
    
    @staticmethod
    def get_query_list(split: str = "train", params: Dict[str, Any] = None) -> List[str]:
        """
        Get a list of task IDs for the specified split.
        
        ✅ Align with AgentGym eval protocol:
        - train: curated BabyAI_TRAIN_INDEX derived from categories (see AgentGym paper)
        - val/dev/test: use AgentGym eval indices from `env_service/environments/babyai/babyai_test.json` (90 tasks)
        
        Args:
            split (str): "train", "val", "test", "dev"
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            List[str]: List of task IDs.
        """
        _ = params or {}

        def _load_eval_indices() -> List[int]:
            here = os.path.dirname(__file__)
            test_path = os.path.join(here, "babyai_test.json")
            with open(test_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # item_id format: "babyai_1804"
            idxs: List[int] = []
            for it in data:
                item_id = str(it.get("item_id", ""))
                m = re.match(r"^babyai_(\d+)$", item_id)
                if not m:
                    raise ValueError(f"Unexpected BabyAI eval item_id: {item_id}")
                idxs.append(int(m.group(1)))
            return idxs

        if split == "train":
            # BABYAI_TRAIN_INDEX = [i for i in range(1804) if (i % 40 + 1) in BABYAI_ALLOWED_CATEGORIES]
            allowed = set(BabyaiEnv.BABYAI_ALLOWED_CATEGORIES)
            return [
                str(i)
                for i in range(BabyaiEnv.BABYAI_BASE_TRAIN_SIZE)
                if ((i % BabyaiEnv.NUM_LEVELS) + 1) in allowed
            ]

        # val/dev/test → AgentGym eval set
        return [str(i) for i in _load_eval_indices()]
    
    def _parse_action_from_llm_output(self, llm_output: str) -> Optional[str]:
        """
        Parse action from LLM output.
        
        Args:
            llm_output (str): Raw LLM output string.
        
        Returns:
            Optional[str]: Extracted action string.
        """
        if not llm_output or not isinstance(llm_output, str):
            return None
        
        llm_output_clean = llm_output.strip()
        
        # Strategy 1: Extract "Action:" segment
        action_parts = llm_output_clean.rsplit("Action:", 1)
        
        if len(action_parts) == 2:
            action_str = action_parts[1].strip()
            action_str = re.sub(r"^Action:\s*", "", action_str, flags=re.IGNORECASE).strip()
            action_str = action_str.split('\n')[0].strip()  # Take first line only
            
            if action_str:
                return action_str
        
        # Strategy 2: If no "Action:" found, check if it's a direct action
        if not re.match(r"^(Thought|Action):", llm_output_clean, re.IGNORECASE):
            return llm_output_clean.split('\n')[0].strip()
        
        return None

