# -*- coding: utf-8 -*-
"""
WebShop Environment integration for AgentEvolver.

通过 HTTP 方式复用 AgentGym 提供的 WebShop 服务器。
WebShop 是一个模拟在线购物的环境，agent 需要根据用户指令
搜索、浏览和购买符合要求的商品。
"""

import os
import re
from typing import Any, Dict, List, Optional

import requests

from env_service.base import BaseEnv
from env_service.registry import Registry


@Registry.register("webshop")
class WebshopEnv(BaseEnv):
    """
    WebShop environment wrapper for AgentEvolver.
    
    This class adapts AgentGym's WebShop environment to the BaseEnv interface,
    allowing it to be used with Experience Pool and GRPO training.
    
    WebShop 环境模拟电商购物场景：
    - Agent 收到购物指令（如 "Find me a red dress under $50"）
    - Agent 可以搜索商品、浏览列表、查看详情、选择选项
    - 最终购买符合要求的商品获得奖励
    """
    
    # WebShop 使用 session_id 来标识不同的购物任务
    # 每个 session 对应一个随机生成的购物指令
    NUM_SESSIONS = 500  # 默认使用 500 个不同的 session
    
    def __init__(self, task_id: str = None, instance_id: str = None, params: Dict[str, Any] = None):
        """
        Initialize the WebShop environment.
        
        Args:
            task_id (str): The task ID (session_id in WebShop).
            instance_id (str): The instance ID for this environment.
            params (Dict[str, Any]): Additional parameters including:
                - server_url: URL of the WebShop server
        """
        self.task_id = task_id
        self.instance_id = instance_id
        self.params = params or {}

        # 外部 WebShop HTTP 服务器地址
        self.server_url: str = (
            self.params.get("server_url")
            or os.environ.get("WEBSHOP_SERVER_URL", "http://127.0.0.1:36003")
        ).rstrip("/")

        # 远端 WebShop server 上的 env id
        self.remote_env_id: Optional[int] = None
        self.current_session_id = None
        
        # Store state info
        self.current_observation = None
        self.current_instruction = None
        self.current_available_actions = {"has_search_bar": False, "clickables": []}
        self.is_done = False
        self.current_reward = 0.0

    # ---------------------------
    # Internal HTTP helpers
    # ---------------------------

    def _post(self, path: str, json_body: Dict[str, Any]) -> Any:
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
        """确保在远端 WebShop server 上创建了一个 env id。"""
        if self.remote_env_id is not None:
            return
        # WebShop 的 /create 接口直接返回 int（不是 dict）
        data = self._post("/create", {})
        if isinstance(data, int):
            self.remote_env_id = data
        elif isinstance(data, dict) and "id" in data:
            self.remote_env_id = int(data["id"])
        else:
            # 可能直接返回整数
            self.remote_env_id = int(data)
        print(f"Created WebShop environment {self.remote_env_id}")
        
    def get_init_state(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the initial state of the environment.
        
        Args:
            params (Dict[str, Any]): Additional parameters:
                - session_id: Session ID for the shopping task
        
        Returns:
            Dict[str, Any]: Initial state with "state" (list of messages) and other info.
        """
        params = params or {}
        
        # Determine session_id
        if self.task_id is not None:
            try:
                session_id = int(self.task_id)
            except ValueError:
                session_id = None  # Use random session
        else:
            session_id = params.get("session_id", None)
        
        # 确保远端 env 已创建
        self._ensure_remote_env()

        # 调用 WebShop 的 /reset 接口
        reset_payload = {
            "env_idx": self.remote_env_id,
            "session_id": session_id,
        }
        reset_result = self._post("/reset", reset_payload)

        if isinstance(reset_result, dict) and "error" in reset_result:
            raise RuntimeError(
                f"Failed to reset WebShop environment: {reset_result['error']}"
            )
        
        self.current_session_id = session_id
        
        # Get instruction and observation
        self.current_instruction = self._get("/instruction_text", {"env_idx": self.remote_env_id})
        self.current_observation = self._get("/observation", {"env_idx": self.remote_env_id})
        
        # Get available actions
        self.current_available_actions = self._get("/available_actions", {"env_idx": self.remote_env_id})
        
        self.is_done = False
        self.current_reward = 0.0
        
        # Format available actions for display
        action_desc = self._format_available_actions()
        
        # Format initial state as messages
        init_messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "assistant",
                "content": "OK. I'll help you find and purchase the item according to your requirements."
            },
            {
                "role": "user",
                "content": f"{self.current_observation}\n\n{action_desc}"
            }
        ]
        
        return {
            "instance_id": self.instance_id,
            "state": init_messages,
            "instruction": self.current_instruction,
            "available_actions": self.current_available_actions,
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for WebShop environment."""
        return '''You are a shopping assistant agent navigating an online store to find and purchase items for the user.

Your task is to:
1. Search for items using the search bar
2. Browse through search results
3. Click on items to view details
4. Select required options (size, color, etc.)
5. Add the item to cart when you find the right one
6. Complete the purchase

Available action types:
- search[query]: Search for items (only when search bar is available)
- click[element]: Click on a clickable element (product, option, button, etc.)

You should choose from two actions: "THOUGHT" or "ACTION".
- If you choose "THOUGHT", first analyze the current page and plan, then output your action.
  Format: "Thought:\nyour thoughts.\n\nAction:\nyour next action"
- If you choose "ACTION", directly output the action.
  Format: "Action:\nyour next action"

Important:
1. Read the shopping instruction carefully.
2. Pay attention to all requirements (color, size, price, features).
3. Use search[] to find items and click[] to interact.
4. Click "Buy Now" when you find the perfect match.'''

    def _format_available_actions(self) -> str:
        """Format available actions for display."""
        actions = self.current_available_actions
        parts = []
        
        if actions.get("has_search_bar"):
            parts.append("You can use: search[your query]")
        
        clickables = actions.get("clickables", [])
        if clickables:
            # Show first 20 clickables to avoid too long output
            display_clickables = clickables[:20]
            parts.append(f"Clickable elements: {display_clickables}")
            if len(clickables) > 20:
                parts.append(f"... and {len(clickables) - 20} more")
        
        return "\n".join(parts) if parts else "No actions available."
    
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
            action_desc = self._format_available_actions()
            invalid_obs = "Invalid action format. Please use 'Action: search[query]' or 'Action: click[element]' format."
            invalid_obs += f"\n\n{action_desc}"
            
            return {
                "state": [{"role": "user", "content": invalid_obs}],
                "reward": 0.0,
                "is_terminated": False,
                "info": {
                    "available_actions": self.current_available_actions,
                    "invalid_action": True,
                },
                "instance_id": self.instance_id,
            }

        # 调用 WebShop 的 /step 接口
        step_payload = {
            "env_idx": self.remote_env_id,
            "action": parsed_action,
        }
        step_result = self._post("/step", step_payload)

        if isinstance(step_result, dict) and "error" in step_result:
            raise RuntimeError(f"Step failed: {step_result['error']}")
        
        # Update state (step_result is StepResponse: state, reward, done, info)
        if isinstance(step_result, dict):
            self.current_observation = step_result.get("state", "")
            self.is_done = step_result.get("done", False)
            self.current_reward = step_result.get("reward", 0.0)
        else:
            # Handle tuple response
            self.current_observation = step_result[0] if len(step_result) > 0 else ""
            self.current_reward = step_result[1] if len(step_result) > 1 else 0.0
            self.is_done = step_result[2] if len(step_result) > 2 else False
        
        # Get updated available actions
        try:
            self.current_available_actions = self._get("/available_actions", {"env_idx": self.remote_env_id})
        except:
            self.current_available_actions = {"has_search_bar": False, "clickables": []}
        
        action_desc = self._format_available_actions()
        
        # Format response
        state_messages = [
            {
                "role": "user",
                "content": f"{self.current_observation}\n\n{action_desc}"
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
        
        Returns:
            float: Evaluation score (0.0 to 1.0).
        """
        return float(self.current_reward)
    
    def get_info(self, messages: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about the environment.
        """
        return {
            "env_id": "webshop",
            "task_id": self.task_id,
            "session_id": self.current_session_id,
            "instruction": self.current_instruction,
            "observation": self.current_observation,
            "available_actions": self.current_available_actions,
            "done": self.is_done,
            "reward": self.current_reward,
        }
    
    def close(self):
        """
        Close the environment and release resources.
        """
        # WebShop doesn't have a /close endpoint in the original implementation
        # Just reset local state
        self.remote_env_id = None
        print(f"WebShop environment released.")
    
    @staticmethod
    def get_query_list(split: str = "train", params: Dict[str, Any] = None) -> List[str]:
        """
        Get a list of task IDs (session IDs) for the specified split.
        
        Args:
            split (str): "train", "val", "test", "dev"
            params (Dict[str, Any]): Additional parameters.
        
        Returns:
            List[str]: List of task IDs (session IDs).
        """
        params = params or {}
        
        # WebShop has around 12k user instructions
        # We use session_id as task_id
        total_sessions = params.get("total_sessions", WebshopEnv.NUM_SESSIONS)
        
        if split == "train":
            # Use 80% for training
            task_ids = [str(i) for i in range(int(total_sessions * 0.8))]
        else:  # val, dev, test
            # Use 20% for testing
            task_ids = [str(i) for i in range(int(total_sessions * 0.8), total_sessions)]
        
        return task_ids
    
    def _parse_action_from_llm_output(self, llm_output: str) -> Optional[str]:
        """
        Parse action from LLM output.
        
        WebShop actions are in format: search[query] or click[element]
        
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
            action_str = action_str.split('\n')[0].strip()
            
            if action_str:
                return action_str
        
        # Strategy 2: Look for search[] or click[] patterns directly
        search_match = re.search(r'search\[([^\]]+)\]', llm_output_clean, re.IGNORECASE)
        if search_match:
            return f"search[{search_match.group(1)}]"
        
        click_match = re.search(r'click\[([^\]]+)\]', llm_output_clean, re.IGNORECASE)
        if click_match:
            return f"click[{click_match.group(1)}]"
        
        # Strategy 3: If no "Action:" found and not recognized pattern
        if not re.match(r"^(Thought|Action):", llm_output_clean, re.IGNORECASE):
            # Check if it looks like a valid action
            if llm_output_clean.startswith(("search[", "click[")):
                return llm_output_clean.split('\n')[0].strip()
        
        return None

