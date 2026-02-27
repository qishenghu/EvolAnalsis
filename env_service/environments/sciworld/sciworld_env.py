# -*- coding: utf-8 -*-
"""
ScienceWorld Environment integration for AgentEvolver.

通过 HTTP 方式复用 AgentGym 提供的 ScienceWorld 服务器。
ScienceWorld 是一个基于文本的科学实验模拟环境，
agent 需要通过一系列操作完成科学实验任务。
"""

import os
import re
import json
from typing import Any, Dict, List, Optional

import requests

from env_service.base import BaseEnv
from env_service.registry import Registry


@Registry.register("sciworld")
class SciworldEnv(BaseEnv):
    """
    ScienceWorld environment wrapper for AgentEvolver.
    
    This class adapts AgentGym's ScienceWorld environment to the BaseEnv interface,
    allowing it to be used with Experience Pool and GRPO training.
    
    ScienceWorld 包含 30 个科学实验任务类型，如：
    - 测量物体质量
    - 加热/冷却物体
    - 观察化学反应
    - 电路实验
    - 生物实验
    等
    """
    
    def __init__(self, task_id: str = None, instance_id: str = None, params: Dict[str, Any] = None):
        """
        Initialize the ScienceWorld environment.
        
        Args:
            task_id (str): The task ID (data_idx combining taskName and variationIdx).
            instance_id (str): The instance ID for this environment.
            params (Dict[str, Any]): Additional parameters including:
                - server_url: URL of the ScienceWorld server
        """
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}

        # 外部 ScienceWorld HTTP 服务器地址
        self.server_url: str = (
            self.params.get("server_url")
            or os.environ.get("SCIWORLD_SERVER_URL", "http://127.0.0.1:36004")
        ).rstrip("/")

        # ScienceWorld simplification preset (e.g. "easy", "teleportAction,openDoors", or "")
        self.simplification_str: str = (
            self.params.get("simplification_str")
            or os.environ.get("SCIWORLD_SIMPLIFICATION", "easy")
        )

        # 远端 ScienceWorld server 上的 env id
        self.remote_env_id: Optional[int] = None
        self.current_data_idx = None
        self.current_task_name = None
        self.current_var_num = None
        
        # Store state info
        self.current_observation = None
        self.current_task_description = None
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
        """确保在远端 ScienceWorld server 上创建了一个 env id。"""
        if self.remote_env_id is not None:
            return
        # AgentGym 的 /create 接口返回 {"id": int}
        data = self._post("/create", {})
        if "id" not in data:
            raise RuntimeError(f"ScienceWorld HTTP /create returned invalid data: {data}")
        self.remote_env_id = int(data["id"])
        print(f"Created ScienceWorld environment {self.remote_env_id}")
        
    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        Args:
            params (Dict[str, Any]): Additional parameters:
                - data_idx: Index to specify task and variation
        
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
            "simplification_str": self.simplification_str,
        }
        reset_result = self._post("/reset", reset_payload)

        if "error" in reset_result:
            raise RuntimeError(
                f"Failed to reset ScienceWorld environment: {reset_result['error']}"
            )
        
        self.current_data_idx = data_idx
        self.current_task_name = reset_result.get("task_name", "")
        self.current_var_num = reset_result.get("var_num", 0)
        self.current_task_description = reset_result.get("task_description", "")
        self.current_observation = reset_result.get("observation", "")
        self.is_done = reset_result.get("done", False)
        self.current_reward = reset_result.get("reward", 0.0)
        self.current_score = reset_result.get("score", 0.0)
        
        # Get action hints
        action_hints = self._get_action_hints()
        
        # Format initial state as messages
        init_messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "assistant",
                "content": "OK. I'll help you complete this scientific experiment step by step."
            },
            {
                "role": "user",
                "content": f"Task: {self.current_task_description}\n\nCurrent observation:\n{self.current_observation}\n\n{action_hints}"
            }
        ]
        
        return {
            "instance_id": self.instance_id,
            "state": init_messages,
            "task_name": self.current_task_name,
            "task_description": self.current_task_description,
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for ScienceWorld environment."""
        return '''You are a scientific experiment assistant in a text-based simulation environment. Your task is to perform scientific experiments by interacting with objects in the environment.

At each step, you will receive:
1. The task description (what experiment you need to perform)
2. Your current observation (what you can see/do)
3. OBJ candidates (the objects that can be interacted with in the current state).

Available actions:
[
{{"action": "open OBJ", "description": "open a container"}},
{{"action": "close OBJ", "description": "close a container"}},
{{"action": "activate OBJ", "description": "activate a device"}},
{{"action": "deactivate OBJ", "description": "deactivate a device"}},
{{"action": "connect OBJ to OBJ", "description": "connect electrical components"}},
{{"action": "disconnect OBJ", "description": "disconnect electrical components"}},
{{"action": "use OBJ [on OBJ]", "description": "use a device/item"}},
{{"action": "look around", "description": "describe the current room"}},
{{"action": "look at OBJ", "description": "describe an object in detail"}},
{{"action": "look in OBJ", "description": "describe a container's contents"}},
{{"action": "read OBJ", "description": "read a note or book"}},
{{"action": "move OBJ to OBJ", "description": "move an object to a container"}},
{{"action": "pick up OBJ", "description": "move an object to the inventory"}},
{{"action": "put down OBJ", "description": "drop an inventory item"}},
{{"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}},
{{"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}},
{{"action": "mix OBJ", "description": "chemically mix a container"}},
{{"action": "go to LOC", "description": "move to a new location"}},
{{"action": "eat OBJ", "description": "eat a food"}},
{{"action": "flush OBJ", "description": "flush a toilet"}},
{{"action": "focus on OBJ", "description": "signal intent on a task object"}},
{{"action": "wait", "description": "take no action for 10 iterations"}},
{{"action": "wait1", "description": "take no action for 1 iteration"}},
{{"action": "task", "description": "describe current task"}},
{{"action": "inventory", "description": "list your inventory"}}
]

In each turn, you should choose from two answer formats: "THOUGHT" or "ACTION".
- If you choose "THOUGHT", first analyze the task and current state, then output your action.
  Format: "Thought:\nyour thoughts.\n\nAction:\nyour next action"
- If you choose "ACTION", directly output the action.
  Format: "Action:\nyour next action"

Important:
1. Read the task description carefully.
2. Plan your experiment steps logically.
3. Pay attention to the objects and locations available.
4. OBJ in the selected action should be replaced with one of the OBJ candidates.
'''

    def _get_action_hints(self) -> str:
        """Get action hints from the server."""
        try:
            hints = self._get("/action_hint", {"id": self.remote_env_id})
            if isinstance(hints, dict):
                valid_objs = hints.get("possible_objects", [])
                
                # Keep only object candidates to reduce token usage.
                # Valid action templates already exist in the system prompt.
                hint_str = f"OBJ candidates: {valid_objs}"
                # if possible_actions:
                #    hint_str += f"Suggested actions: {possible_actions}\n"
                # if possible_objects:
                #    hint_str += f"Nearby objects: {possible_objects}"
                return hint_str
        except:
            pass
        return ""
    
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
        
        if parsed_action is None:
            action_hints = self._get_action_hints()
            invalid_obs = "Invalid action format. Please use 'Action: your_action' format."
            invalid_obs += f"\n\n{action_hints}"
            
            return {
                "state": [{"role": "user", "content": invalid_obs}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {
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
        
        # Get updated action hints
        action_hints = self._get_action_hints()
        
        # Format response
        state_messages = [
            {
                "role": "user",
                "content": f"{self.current_observation}\n\n{action_hints}"
            }
        ]
        
        return {
            "state": state_messages,
            "reward": self.current_reward,
            "is_terminated": self.is_done,
            "info": {
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
        # ScienceWorld exposes an integer-ish `score` (often interpreted as percentage).
        # Paper-style evaluation is non-negative: failed trajectories should not produce negative reward.
        # So we clamp score into [0, 100] then normalize to [0, 1].
        #
        # ⭐ AgentGym-style binary outcome:
        # Some AgentGym settings binarize the final outcome: only a perfect completion (score==1.0)
        # is treated as success; any partial progress (<1.0) is treated as failure (0.0).
        if self.current_score is None:
            return 0.0
        score = float(self.current_score)
        score = max(0.0, min(100.0, score))
        normalized = score / 100.0
        #return 1.0 if normalized >= 1.0 else 0.0
        return normalized

    def get_info(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about the environment.
        """
        return {
            "env_id": "sciworld",
            "task_id": self.task_id,
            "data_idx": self.current_data_idx,
            "task_name": self.current_task_name,
            "var_num": self.current_var_num,
            "task_description": self.current_task_description,
            "observation": self.current_observation,
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
                print(f"Closing ScienceWorld environment {self.remote_env_id}")
                close_payload = {"id": self.remote_env_id}
                self._post("/close", close_payload)
                print(f"ScienceWorld environment {self.remote_env_id} closed successfully.")
            except Exception as e:
                print(f"Error closing ScienceWorld environment {self.remote_env_id}: {e}")
            finally:
                self.remote_env_id = None
    
    @staticmethod
    def get_query_list(split: str = "train", params: Dict[str, Any] = None) -> List[str]:
        """
        Get a list of task IDs for the specified split.
        
        ✅ Align with AgentGym eval protocol:
        - val/dev/test: use AgentGym eval indices from `env_service/environments/sciworld/sciworld_test.json` (200 tasks)
        - train: return indices that DO NOT overlap with the eval indices.
          We approximate the candidate pool as [0, max(eval_idx)+1) (covers all eval indices)
          and remove eval indices. This guarantees no overlap, and works with AgentGym's
          `games[data_idx]` indexing as long as eval indices are valid.
        
        Args:
            split (str): "train", "val", "test", "dev"
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            List[str]: List of task IDs.
        """
        _ = params or {}

        def _load_eval_indices() -> List[int]:
            here = os.path.dirname(__file__)
            test_path = os.path.join(here, "sciworld_test.json")
            with open(test_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # item_id format: "sciworld_606"
            idxs: List[int] = []
            for it in data:
                item_id = str(it.get("item_id", ""))
                m = re.match(r"^sciworld_(\d+)$", item_id)
                if not m:
                    raise ValueError(f"Unexpected SciWorld eval item_id: {item_id}")
                idxs.append(int(m.group(1)))
            return idxs

        eval_idxs = _load_eval_indices()
        eval_set = set(eval_idxs)

        if split in ("val", "dev", "test"):
            return [str(i) for i in eval_idxs]

        # train: exclude eval indices
        max_eval = max(eval_idxs) if eval_idxs else 0
        candidate_max = max_eval + 1
        train_ids = [str(i) for i in range(candidate_max) if i not in eval_set]
        return train_ids
    
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
        
        # Remove </s> token if present
        if llm_output_clean.endswith("</s>"):
            llm_output_clean = llm_output_clean[:-4].strip()
        
        # Strategy 1: Extract "Action:" segment
        action_parts = llm_output_clean.rsplit("Action:", 1)
        
        if len(action_parts) == 2:
            action_str = action_parts[1].strip()
            action_str = re.sub(r"^Action:\s*", "", action_str, flags=re.IGNORECASE).strip()
            # Take first line/sentence
            action_str = action_str.split('\n')[0].strip()
            
            if action_str:
                return action_str
        
        # Strategy 2: If no "Action:" found, check if it's a direct action
        if not re.match(r"^(Thought|Action):", llm_output_clean, re.IGNORECASE):
            # Looks like a direct action
            return llm_output_clean.split('\n')[0].strip()
        
        # Strategy 3: If only "Thought:" without "Action:", return None
        if "Thought:" in llm_output_clean and "Action:" not in llm_output_clean:
            return None
        
        return None

