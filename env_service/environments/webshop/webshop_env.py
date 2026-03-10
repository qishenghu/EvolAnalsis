# -*- coding: utf-8 -*-
"""
WebShop Environment integration for AgentEvolver.

通过 HTTP 方式复用 AgentGym 提供的 WebShop 服务器。
WebShop 是一个模拟在线购物的环境，agent 需要根据用户指令
搜索、浏览和购买符合要求的商品。
"""

import json
import os
import re
from math import prod
from typing import Any, Dict, List, Optional

import requests

from env_service.base import BaseEnv
from env_service.registry import Registry

AGENTGYM_WEBSHOP_DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "AgentGym",
        "agentenv-webshop",
        "webshop",
        "data",
    )
)


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
    NUM_SESSIONS = 500  # Fallback only; prefer explicit split files or params.
    
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
        self.action_format = str(self.params.get("action_format", "react"))
        self.enable_action_sanitizer = bool(self.params.get("enable_action_sanitizer", True))
        self.invalid_action_penalty = float(self.params.get("invalid_action_penalty", -0.1))
        self.max_consecutive_invalid_actions = int(
            self.params.get("max_consecutive_invalid_actions", 2)
        )
        self.consecutive_invalid_actions = 0
        self.total_invalid_actions = 0
        self.invalid_action_penalty_total = 0.0

    def _use_react_tags(self) -> bool:
        return self.action_format == "react_tags"

    def _get_action_format_example(self) -> str:
        if self._use_react_tags():
            return (
                "<think>\n"
                "your thoughts.\n"
                "</think>\n"
                "<action>\n"
                "click[something] (or search[keywords])\n"
                "</action>"
            )
        return (
            "Thought:\n"
            "your thoughts.\n"
            "Action: \n"
            "click[something] (or search[keywords])"
        )

    def _get_invalid_action_format_message(self) -> str:
        if self._use_react_tags():
            return (
                "Invalid action format. Please use the format: '<think>\\n your thoughts.\\n</think>\\n"
                "<action>\\n search[query]\\n</action>' or '<think>\\n your thoughts.\\n</think>\\n"
                "<action>\\n click[element]\\n</action>'."
            )
        return (
            "Invalid action format. Please use the format: 'Thought:\\n your thoughts.\\n\\n"
            "Action:\\n search[query]' or 'Thought:\\n your thoughts.\\n\\nAction:\\n click[element]'."
        )

    @staticmethod
    def _parse_session_id(value: Any) -> Optional[int]:
        """Accept raw ints as well as ids like ``webshop_5238``."""
        if value is None:
            return None
        if isinstance(value, int):
            return value

        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)

        match = re.match(r"^webshop_(\d+)$", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _get_eval_split_path() -> str:
        return os.path.join(os.path.dirname(__file__), "webshop_test.json")

    @staticmethod
    def _get_train_split_path() -> str:
        return os.path.join(os.path.dirname(__file__), "webshop_train.json")

    @staticmethod
    def _get_default_product_file() -> str:
        return os.path.join(AGENTGYM_WEBSHOP_DATA_DIR, "items_shuffle_1000.json")

    @staticmethod
    def _get_default_attribute_file() -> str:
        return os.path.join(AGENTGYM_WEBSHOP_DATA_DIR, "items_ins_v2_1000.json")

    @classmethod
    def _load_eval_indices(cls) -> List[int]:
        test_path = cls._get_eval_split_path()
        return cls._load_split_indices_from_file(test_path, split_name="eval")

    @classmethod
    def _load_train_indices(cls) -> Optional[List[int]]:
        train_path = cls._get_train_split_path()
        if not os.path.exists(train_path):
            return None
        return cls._load_split_indices_from_file(train_path, split_name="train")

    @classmethod
    def _load_split_indices_from_file(cls, path: str, split_name: str) -> List[int]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"WebShop {split_name} split not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idxs: List[int] = []
        for item in data:
            item_id = str(item.get("item_id", ""))
            session_id = cls._parse_session_id(item_id)
            if session_id is None:
                raise ValueError(f"Unexpected WebShop {split_name} item_id: {item_id}")
            idxs.append(session_id)
        return idxs

    @classmethod
    def _infer_total_sessions(cls, params: Optional[Dict[str, Any]] = None) -> int:
        params = params or {}
        explicit_total = params.get("total_sessions")
        if explicit_total is not None:
            return int(explicit_total)

        inferred_from_data = cls._infer_total_sessions_from_default_data()
        if inferred_from_data is not None:
            return inferred_from_data

        eval_idxs = cls._load_eval_indices()
        # Without a canonical train split file, use the smallest range that
        # fully covers all known eval ids while guaranteeing no overlap.
        return max(eval_idxs) + 1 if eval_idxs else cls.NUM_SESSIONS

    @classmethod
    def _infer_total_sessions_from_default_data(cls) -> Optional[int]:
        product_file = cls._get_default_product_file()
        attribute_file = cls._get_default_attribute_file()
        if not (os.path.exists(product_file) and os.path.exists(attribute_file)):
            return None

        with open(product_file, "r", encoding="utf-8") as f:
            products = json.load(f)
        with open(attribute_file, "r", encoding="utf-8") as f:
            attributes = json.load(f)

        total_goals = 0
        seen_asins = set()
        for product in products[:1000]:
            asin = product.get("asin")
            if asin == "nan" or not asin or len(str(asin)) > 10 or asin in seen_asins:
                continue
            seen_asins.add(asin)

            attr_entry = attributes.get(asin)
            if not attr_entry:
                continue

            instruction = attr_entry.get("instruction")
            instruction_attributes = attr_entry.get("instruction_attributes")
            if instruction is None or not instruction_attributes:
                continue

            option_sizes: List[int] = []
            customization_options = product.get("customization_options") or {}
            for option_contents in customization_options.values():
                if option_contents is None:
                    continue
                valid_values = [
                    option_content.get("value", "").strip().replace("/", " | ").lower()
                    for option_content in option_contents
                    if option_content.get("value")
                ]
                if valid_values:
                    option_sizes.append(len(valid_values))

            total_goals += prod(option_sizes) if option_sizes else 1

        return total_goals

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
            session_id = self._parse_session_id(self.task_id)
        else:
            session_id = self._parse_session_id(params.get("session_id", None))
        
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
        self.consecutive_invalid_actions = 0
        self.total_invalid_actions = 0
        self.invalid_action_penalty_total = 0.0
        
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
        return f'''You are web shopping.
I will give you instructions about what to do.
You have to follow the instructions.
Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.
You can use search action if search is available.
You can click one of the buttons in clickables.
An click or search action should be of the following structure:
search[keywords]
click[value]

If the action is not valid, perform nothing.
Keywords in search are up to you, but the value in click must be a value in the list of available actions.
Remember that your keywords in search should be carefully designed.
In each turn, you must output your thought/reasoning and then output your action in the following format:
```
{self._get_action_format_example()}
```
'''

    def _format_available_actions(self) -> str:
        """Format available actions for display."""
        actions = self.current_available_actions
        parts = []
        
        if actions.get("has_search_bar"):
            parts.append("You can use: search[your query]")
        
        clickables = actions.get("clickables", [])
        if clickables:
            # Show first 20 clickables to avoid too long output
            display_clickables = clickables
            parts.append(f"Clickable elements: {display_clickables}")
            # if len(clickables) > 20:
            #     parts.append(f"... and {len(clickables) - 20} more")
        
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
        is_valid_action, invalid_reason, parsed_action = self._validate_parsed_action(parsed_action)
        if not is_valid_action:
            return self._build_invalid_action_response(invalid_reason)

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
        self.consecutive_invalid_actions = 0
        
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
        return float(self.current_reward + self.invalid_action_penalty_total)
    
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
            "success_rate": 1.0 if float(self.current_reward) >= 1.0 else 0.0,
            "consecutive_invalid_actions": self.consecutive_invalid_actions,
            "total_invalid_actions": self.total_invalid_actions,
            "invalid_action_penalty_total": self.invalid_action_penalty_total,
            "action_format": self.action_format,
        }
    
    def close(self):
        """
        Close the environment and release resources.
        """
        if self.remote_env_id is not None:
            try:
                print(f"Deleting WebShop environment {self.remote_env_id}")
                delete_payload = {"env_idx": self.remote_env_id}
                delete_result = self._post("/delete", delete_payload)
                if isinstance(delete_result, dict) and not delete_result.get("success", False):
                    print(
                        f"Warning: Failed to delete WebShop environment {self.remote_env_id}: {delete_result}",
                    )
            except Exception as e:
                print(f"Error deleting WebShop environment {self.remote_env_id}: {e}")
            finally:
                self.remote_env_id = None
                self.current_session_id = None
                self.current_observation = None
                self.current_instruction = None
                self.current_available_actions = {"has_search_bar": False, "clickables": []}
                self.is_done = False
                self.current_reward = 0.0
                self.consecutive_invalid_actions = 0
                self.total_invalid_actions = 0
                self.invalid_action_penalty_total = 0.0
        print("WebShop environment released.")
    
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
        
        eval_idxs = WebshopEnv._load_eval_indices()
        eval_set = set(eval_idxs)

        if split in ("val", "dev", "test"):
            return [str(i) for i in eval_idxs]

        cached_train_idxs = WebshopEnv._load_train_indices()
        if cached_train_idxs is not None:
            return [str(i) for i in cached_train_idxs]

        total_sessions = WebshopEnv._infer_total_sessions(params)
        if total_sessions <= 0:
            return []

        train_ids = [str(i) for i in range(total_sessions) if i not in eval_set]
        return train_ids
    
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
        if self.enable_action_sanitizer:
            action_matches = list(
                re.finditer(
                    r"(search|click)\s*\[([^\]\n]+)\]",
                    llm_output_clean,
                    flags=re.IGNORECASE,
                )
            )
            if action_matches:
                last_match = action_matches[-1]
                action_type = last_match.group(1).lower()
                action_arg = last_match.group(2).strip()
                if action_arg:
                    return f"{action_type}[{action_arg}]"

        if self._use_react_tags():
            action_tag_match = re.search(
                r"<action>(.*?)</action>",
                llm_output_clean,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if action_tag_match:
                action_str = action_tag_match.group(1).strip().split("\n")[0].strip()
                if action_str:
                    return action_str
        
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
        if not re.match(r"^(Thought|Action):", llm_output_clean, re.IGNORECASE) and not re.match(r"^<(think|action)>", llm_output_clean, re.IGNORECASE):
            # Check if it looks like a valid action
            if llm_output_clean.startswith(("search[", "click[")):
                return llm_output_clean.split('\n')[0].strip()
        
        return None

    def _validate_parsed_action(self, parsed_action: Optional[str]) -> tuple[bool, str, Optional[str]]:
        if parsed_action is None:
            return False, self._get_invalid_action_format_message(), None

        search_match = re.fullmatch(r"search\[([^\]]+)\]", parsed_action, flags=re.IGNORECASE)
        if search_match:
            if not self.current_available_actions.get("has_search_bar", False):
                return False, "Invalid action. Search is not available on this page.", None
            if not search_match.group(1).strip():
                return False, "Invalid action. Search query cannot be empty.", None
            return True, "", f"search[{search_match.group(1).strip()}]"

        click_match = re.fullmatch(r"click\[([^\]]+)\]", parsed_action, flags=re.IGNORECASE)
        if click_match:
            target = click_match.group(1).strip()
            clickables = [str(item) for item in self.current_available_actions.get("clickables", [])]
            target_lower = target.lower()
            canonical_target = None
            for item in clickables:
                item_lower = item.lower()
                if item_lower == target_lower:
                    canonical_target = item
                    break
                wrapped_match = re.fullmatch(r"click\[([^\]]+)\]", item, flags=re.IGNORECASE)
                if wrapped_match and wrapped_match.group(1).strip().lower() == target_lower:
                    canonical_target = item
                    break
            if canonical_target is None:
                return False, f"Invalid action. '{target}' is not a clickable element on this page.", None
            if re.fullmatch(r"click\[([^\]]+)\]", canonical_target, flags=re.IGNORECASE):
                return True, "", canonical_target
            return True, "", f"click[{canonical_target}]"

        return False, "Invalid action. Only search[...] and click[...] are allowed.", None

    def _build_invalid_action_response(self, invalid_reason: str) -> Dict[str, Any]:
        self.total_invalid_actions += 1
        self.consecutive_invalid_actions += 1
        self.invalid_action_penalty_total += self.invalid_action_penalty
        should_terminate = self.consecutive_invalid_actions >= self.max_consecutive_invalid_actions
        self.is_done = should_terminate

        action_desc = self._format_available_actions()
        invalid_obs = invalid_reason
        invalid_obs += f"\n\n{action_desc}"
        if should_terminate:
            invalid_obs += (
                f"\n\nEpisode terminated after {self.consecutive_invalid_actions} consecutive invalid actions."
            )

        return {
            "state": [{"role": "user", "content": invalid_obs}],
            "reward": self.invalid_action_penalty,
            "is_terminated": should_terminate,
            "info": {
                "available_actions": self.current_available_actions,
                "invalid_action": True,
                "consecutive_invalid_actions": self.consecutive_invalid_actions,
                "total_invalid_actions": self.total_invalid_actions,
                "invalid_action_penalty_total": self.invalid_action_penalty_total,
            },
            "instance_id": self.instance_id,
        }

