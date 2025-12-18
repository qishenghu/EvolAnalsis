# -*- coding: utf-8 -*-
"""
AlfWorld Environment integration for AgentEvolver.

当前版本通过 **HTTP 方式复用 AgentGym 提供的 AlfWorld 服务器**，
而不在每个 Ray actor 内部创建 ALFWorld_Wrapper，从而避免每次
create_instance 都重新从磁盘加载全部 games。
"""

import os
import json
import re
from typing import Any, Dict, List, Optional

import requests

# 仅用于读取 mappings_train / mappings_test.json，复用 AgentGym 的任务划分
AGENTGYM_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "AgentGym",
        "agentenv-alfworld",
    )
)

from env_service.base import BaseEnv
from env_service.registry import Registry


@Registry.register("alfworld")
class AlfworldEnv(BaseEnv):
    """
    AlfWorld environment wrapper for AgentEvolver.
    
    This class adapts AgentGym's AlfWorld environment to the BaseEnv interface,
    allowing it to be used with Experience Pool and GRPO training.
    """
    
    def __init__(self, task_id: str = None, instance_id: str = None, params: Dict[str, Any] = None):
        """
        Initialize the AlfWorld environment.
        
        Args:
            task_id (str): The task ID (game index in the games list).
            instance_id (str): The instance ID for this environment.
            params (Dict[str, Any]): Additional parameters including:
                - data_path: Path to ALFWORLD_DATA (default: ~/.cache/alfworld)
                - config_path: Path to base_config.yaml
        """
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}

        # 外部 AlfWorld HTTP 服务器地址：
        # 优先使用 params["server_url"]，否则使用环境变量 ALFWORLD_SERVER_URL，
        # 默认 http://127.0.0.1:36001 （见 AgentGym README）。
        self.server_url: str = (
            self.params.get("server_url")
            or os.environ.get("ALFWORLD_SERVER_URL", "http://127.0.0.1:36001")
        ).rstrip("/")

        # 远端 AlfWorld server 上的 env id（即 agentenv-alfworld/server.py 中的 id）
        self.remote_env_id: Optional[int] = None
        self.current_game_index = None
        self.world_type = self.params.get("world_type", "Text")
        
        # Store initial state info
        self.current_observation = None
        self.current_available_actions = []
        self.is_done = False
        self.current_reward = 0.0

    # ---------------------------
    # Internal HTTP helpers
    # ---------------------------

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        resp = requests.post(url, json=json_body, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    def _ensure_remote_env(self):
        """确保在远端 AlfWorld server 上创建了一个 env id。"""
        if self.remote_env_id is not None:
            return
        # AgentGym 的 /create 接口不需要 body，返回 {"id": int}
        data = self._post("/create", {})
        # data 形如 {"id": 0}
        if "id" not in data:
            raise RuntimeError(f"AlfWorld HTTP /create returned invalid data: {data}")
        self.remote_env_id = int(data["id"])
        print(f"Created AlfWorld environment {self.remote_env_id}")
        
    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        Args:
            params (Dict[str, Any]): Additional parameters:
                - game_index: Index of the game to load (if task_id not provided)
                - world_type: "Text", "Embody", or "Hybrid" (default: "Text")
        
        Returns:
            Dict[str, Any]: Initial state with "state" (list of messages) and other info.
        """
        params = params or {}
        
        # Determine game index
        if self.task_id is not None:
            try:
                game_index = int(self.task_id)
            except ValueError:
                # If task_id is not a number, try to find it in games list
                game_index = self._find_game_index_by_task_id(self.task_id)
        else:
            game_index = params.get("game_index", 0)
        
        world_type = params.get("world_type", self.world_type)
        
        # 确保远端 env 已创建
        self._ensure_remote_env()

        # 调用 AgentGym 的 /reset 接口
        reset_payload = {
            "id": self.remote_env_id,
            "game": int(game_index),
            "world_type": world_type,
        }
        reset_result = self._post("/reset", reset_payload)

        if "error" in reset_result:
            raise RuntimeError(
                f"Failed to reset AlfWorld environment: {reset_result['error']}"
            )
        
        self.current_game_index = game_index
        self.current_observation = reset_result["observation"]
        self.current_available_actions = reset_result.get("available_actions", [])
        self.is_done = False
        self.current_reward = 0.0
        
        # Format initial state as messages (similar to appworld format)
        # AlfWorld uses a simple observation format
        init_messages = [
            {
                "role": "system",
                "content": 'Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You should choose from two actions: "THOUGHT" or "ACTION". If you choose "THOUGHT", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought:\nyour thoughts.\n\nAction:\nyour next action"; If you choose "ACTION", you should directly output the action in this turn. Your output must strictly follow this format:"Action:\nyour next action". After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.'
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
            "task_type": reset_result.get("task_type", ""),
            "available_actions": self.current_available_actions,
        }
    
    def step(self, action: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a step in the environment.
        
        Args:
            action (Dict[str, Any]): Action to execute. Expected format:
                - content: The action string (e.g., "go to kitchen" or "Thought: ... Action: go to kitchen")
                - or: raw action string
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            Dict[str, Any]: Step result with state, reward, is_terminated, info.
        """
        params = params or {}
        
        # 确保远端 env 已创建
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
        
        # Parse "Action:" from LLM output if present
        # LLM may output: "Thought: ... Action: go to kitchen" or "Action: go to kitchen"
        # We need to extract just the action part
        parsed_action = self._parse_action_from_llm_output(action_str)
        
        # If parsing failed (returned None), return error state
        if parsed_action is None:
            # Return error state with "Invalid action." message
            # Only return user message (consistent with other environments)
            state_messages = [
                {
                    "role": "user",
                    "content": "Invalid action."
                }
            ]
            
            return {
                "state": state_messages,
                "reward": self.current_reward,  # Use current reward (no change)
                "is_terminated": self.is_done,  # Use current termination status
                "info": {
                    "available_actions": self.current_available_actions,
                },
                "instance_id": self.instance_id,
            }
        
        action_str = parsed_action
        
        

        # 调用 AgentGym 的 /step 接口
        step_payload = {
            "id": self.remote_env_id,
            "action": action_str,
        }
        step_result = self._post("/step", step_payload)

        if "error" in step_result:
            raise RuntimeError(f"Step failed: {step_result['error']}")
        
        # Update state
        self.current_observation = step_result["observation"]
        self.current_available_actions = step_result.get("available_actions", [])
        self.is_done = step_result["done"]
        self.current_reward = step_result.get("reward", 0.0)
        
        # Format response as state message
        # Only return user message (consistent with other environments like appworld, openworld, bfcl)
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
            },
            "instance_id": self.instance_id,
        }
    
    def evaluate(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> float:
        """
        Evaluate the performance of the environment.
        
        Args:
            messages (Dict[str, Any]): Conversation messages (not used for AlfWorld).
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            float: Evaluation score (0.0 to 1.0). For AlfWorld, this is the reward.
        """
        # AlfWorld's reward is already computed during step
        # Return the current reward as the evaluation score
        return float(self.current_reward)
    
    def get_info(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about the environment.
        
        Args:
            messages (Dict[str, Any]): Additional messages.
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            Dict[str, Any]: Environment information.
        """
        return {
            "env_id": self.env_id,
            "task_id": self.task_id,
            "game_index": self.current_game_index,
            "observation": self.current_observation,
            "available_actions": self.current_available_actions,
            "done": self.is_done,
            "reward": self.current_reward,
        }
    
    def close(self):
        """
        Close the environment and release resources.
        
        Args:
            instance_id (str, optional): The instance ID (for compatibility with env_service).
                This is ignored as we use self.instance_id and self.remote_env_id.
        """
        if self.remote_env_id is not None:
            try:
                # 调用 AgentGym 的 /delete 接口，释放远端环境资源
                print(f"Deleting AlfWorld environment {self.remote_env_id}")
                delete_payload = {"id": self.remote_env_id}
                delete_result = self._post("/delete", delete_payload)
                
                if "error" in delete_result:
                    print(f"Warning: Failed to delete AlfWorld environment {self.remote_env_id}: {delete_result['error']}")
                else:
                    print(f"AlfWorld environment {self.remote_env_id} deleted successfully.")
            except Exception as e:
                print(f"Error deleting AlfWorld environment {self.remote_env_id}: {e}")
            finally:
                # 重置本地状态
                self.remote_env_id = None
    
    @staticmethod
    def get_query_list(split: str = "train", params: Dict[str, Any] = None) -> List[str]:
        """
        Get a list of task IDs (game indices) for the specified split.
        
        Args:
            split (str): The split to get queries for ("train", "val", "test", "dev").
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            List[str]: List of task IDs (as strings representing game indices).
        
        Note:
            - "train" → mappings_train.json (2420 tasks)
            - "val", "dev", "test" → mappings_test.json (200 tasks)
        """
        params = params or {}
        
        # Determine mapping file based on split
        # For validation, use test set (200 tasks)
        if split == "train":
            mapping_file = os.path.join(AGENTGYM_ROOT, "configs", "mappings_train.json")
        else:  # val, dev, test → all use test set
            mapping_file = os.path.join(AGENTGYM_ROOT, "configs", "mappings_test.json")
        
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        # Load mappings
        with open(mapping_file, "r") as f:
            mappings = json.load(f)
        
        # Return task IDs (using item_id as the index)
        # For simplicity, we return indices as strings
        # The actual game file path will be resolved in get_init_state
        task_ids = [str(mapping["item_id"]) for mapping in mappings]
        
        return task_ids
    
    def _parse_action_from_llm_output(self, llm_output: str) -> Optional[str]:
        """
        Parse action from LLM output that may contain "Thought:" and "Action:" format.
        
        According to the system prompt, LLM may output:
        - "Thought:\nyour thoughts.\n\nAction:\nyour next action"
        - "Action:\nyour next action"
        - Or just the action string directly
        
        This method follows a similar approach to AgentGym's BaseAdapter.parse_react().
        
        Args:
            llm_output (str): Raw LLM output string.
        
        Returns:
            Optional[str]: Extracted action string (cleaned and ready for environment).
                          Returns None if no valid "Action:" marker is found and the output
                          contains "Thought:" or other markers indicating it's not a direct action.
        """
        if not llm_output or not isinstance(llm_output, str):
            return None
        
        llm_output_clean = llm_output.strip()
        
        # Strategy 1: Try to extract "Action:" segment (most common case)
        # Split by "Action:" from the right (rsplit) to get the last action if multiple exist
        action_parts = llm_output_clean.rsplit("Action:", 1)
        
        if len(action_parts) == 2:
            # Found "Action:" marker
            action_str = action_parts[1].strip()
            
            # Remove any leading "Action:" markers (in case of nested formats)
            action_str = re.sub(r"^Action:\s*", "", action_str, flags=re.IGNORECASE).strip()
            
            # Clean up: remove trailing newlines and whitespace
            action_str = action_str.rstrip()
            
            if action_str:
                return action_str
        
        # Strategy 2: If no "Action:" found, check if it's a direct action
        # (i.e., doesn't start with "Thought:" or "Action:")
        if not re.match(r"^(Thought|Action):", llm_output_clean, re.IGNORECASE):
            # Looks like a direct action, return as-is
            return llm_output_clean
        
        # Strategy 3: If we have "Thought:" but no "Action:", 
        # the LLM output is invalid (missing "Action:" marker)
        # Return None to indicate parsing failure
        if "Thought:" in llm_output_clean and "Action:" not in llm_output_clean:
            print(f"Warning: LLM output contains 'Thought:' but no 'Action:' marker. "
                  f"Raw output: {llm_output_clean[:200]}...")
            return None
        
        # Fallback: if we can't determine the format, return None
        return None
    
    def _find_game_index_by_task_id(self, task_id: str) -> int:
        """
        Find game index by task_id string.
        
        Args:
            task_id (str): Task ID string.
        
        Returns:
            int: Game index.
        """
        # Try to find in train mappings
        for mapping_file in [
            os.path.join(AGENTGYM_ROOT, "configs", "mappings_train.json"),
            os.path.join(AGENTGYM_ROOT, "configs", "mappings_test.json"),
        ]:
            if os.path.exists(mapping_file):
                with open(mapping_file, "r") as f:
                    mappings = json.load(f)
                for mapping in mappings:
                    if mapping.get("task_id") == task_id:
                        return mapping["item_id"]
        
        # If not found, try to parse as integer
        try:
            return int(task_id)
        except ValueError:
            raise ValueError(f"Could not find game index for task_id: {task_id}")

