"""
ReMe 版 RawMemory：通过 HTTP 调用 ReMe 服务存/取原始轨迹。
"""

from __future__ import annotations

import json
from typing import List

import requests

from .memory import BaseMemory, MemoryItem, NullMemory
from .types import ExperienceOutput, APIExperienceOutput


class RawMemory(BaseMemory):
    """
    RawMemory（ReMe 实现）：
    - 存储：调用 ReMe 的 store_raw_trajectory / store_raw_trajectory_only_success
      将 conversation 直接作为 messages 发送，query 作为索引键。
    - 检索：调用 retrieve_raw_trajectory，通过 query 取回原始轨迹文本。
    """

    def __init__(
        self,
        reme_base_url: str = "http://localhost:8123/",
        reme_workspace_id: str = "task_workspace",
        success_only: bool = True,
        k_retrieval: int = 1,
    ):
        super().__init__()
        self.type = "raw"
        self.base_url = reme_base_url.rstrip("/") + "/"
        self.workspace_id = reme_workspace_id
        self.success_only = success_only
        self.k_retrieval = k_retrieval

    def _to_messages(self, conversation: List) -> List[dict]:
        """将 conversation 转为 role/content 列表，保持原顺序。"""
        msgs = []
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            if 'Below are some experiences that may be relevant' in content:
                continue  # skip injected experience
            msgs.append({"role": role, "content": content})
        return msgs

    def _store_endpoint(self) -> str:
        return (
            "store_raw_trajectory_only_success"
            if self.success_only
            else "store_raw_trajectory"
        )

    def _post(self, path: str, payload: dict):
        url = self.base_url + path
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ReMe call failed {url}, code={resp.status_code}, text={resp.text}")
        return resp

    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        if not self._storage_enabled:
            return

        # score 用 reward；ReMe 的 success_only/threshold 在 flow 层处理
        score = getattr(experience, "reward", 0) if hasattr(experience, "reward") else experience.reward
        messages = self._to_messages(experience.conversation)

        payload = {
            "workspace_id": self.workspace_id,
            "trajectories": [
                {
                    "messages": messages,
                    "score": score,
                    "metadata": {"query": task_query, "task_index": task_idx},
                }
            ],
        }

        self._post(self._store_endpoint(), payload)

    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        top_k = min(self.k_retrieval, k)
        payload = {
            "workspace_id": self.workspace_id,
            "query": task_query,
            "top_k": top_k,
        }
        resp = self._post("retrieve_raw_trajectory", payload)
        data = resp.json()
        mems = data.get("metadata", {}).get("memory_list", [])

        results = []
        for mem in mems:
            traj_text = mem.get("content", "")
            metadata = mem.get("metadata", {})
            task_query = metadata['query']
            score = metadata['score']
            result = "Success" if score == 1.0 else "Failure"
            exp_str = f"[TASK]: {task_query}\n[TRAJECTORY]: \n{traj_text}\n[RESULT]: {result}"
            results.append(MemoryItem(content=exp_str, metadata=metadata))
        return results






class StrategyMemory(BaseMemory):
    """
    StrategyMemory（ReMe 实现）：
    - 存储：调用 ReMe 的 store_task_memory，将 strategy 直接作为 messages 发送，query 作为索引键。
    - 检索：调用 retrieve_task_memory_vanilla，通过 query 取回 when_to_use 和 insight 文本。
    """

    def __init__(
        self,
        reme_base_url: str = "http://localhost:8123/",
        reme_workspace_id: str = "task_workspace",
        success_only: bool = True,
        k_retrieval: int = 1,
    ):
        super().__init__()
        self.type = "strategy"
        self.base_url = reme_base_url.rstrip("/") + "/"
        self.workspace_id = reme_workspace_id
        self.success_only = success_only
        self.k_retrieval = k_retrieval

    def _to_messages(self, conversation: List) -> List[dict]:
        """将 conversation 转为 role/content 列表，保持原顺序。"""
        msgs = []
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            if 'Below are some experiences that may be relevant' in content:
                continue  # skip injected experience
            msgs.append({"role": role, "content": content})
        return msgs

    def _store_endpoint(self) -> str:
        return (
            "summary_task_memory_only_success"
            if self.success_only
            else "summary_task_memory"
        )

    def _post(self, path: str, payload: dict):
        url = self.base_url + path
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"ReMe call failed {url}, code={resp.status_code}, text={resp.text}")
        return resp

    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        if not self._storage_enabled:
            return

        # score 用 reward；ReMe 的 success_only/threshold 在 flow 层处理
        score = getattr(experience, "reward", 0) if hasattr(experience, "reward") else experience.reward
        messages = self._to_messages(experience.conversation)

        payload = {
            "workspace_id": self.workspace_id,
            "trajectories": [
                {
                    "messages": messages,
                    "score": score,
                    "metadata": {"query": task_query, "task_index": task_idx},
                }
            ],
        }

        self._post(self._store_endpoint(), payload)

    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        top_k = min(self.k_retrieval, k)
        payload = {
            "workspace_id": self.workspace_id,
            "query": task_query,
            "top_k": top_k,
        }
        resp = self._post("retrieve_task_memory_vanilla", payload)
        data = resp.json()
        mems = data.get("metadata", {}).get("memory_list", [])

        results = []
        for mem in mems:
            when_to_use = mem.get("when_to_use", "")
            exp_text = mem.get("content", "")
            metadata = mem.get("metadata", {})

            exp_str = f"[WHEN TO USE]: {when_to_use}\n[INSIGHT]: {exp_text}"
            results.append(MemoryItem(content=exp_str, metadata=metadata))
        return results



class StrategyRewriteMemory(BaseMemory):
    """
    StrategyRewriteMemory（ReMe 实现）：
    - 存储：调用 ReMe 的 store_task_memory，将 strategy 直接作为 messages 发送，query 作为索引键。
    - 检索：调用 retrieve_task_memory_vanilla_rewrite, 通过 query 取回针对当前query的rewritten insight 文本。
    """

    def __init__(
        self,
        reme_base_url: str = "http://localhost:8123/",
        reme_workspace_id: str = "task_workspace",
        success_only: bool = True,
        k_retrieval: int = 1,
    ):
        super().__init__()
        self.type = "strategy_rewrite"
        self.base_url = reme_base_url.rstrip("/") + "/"
        self.workspace_id = reme_workspace_id
        self.success_only = success_only
        self.k_retrieval = k_retrieval

    def _to_messages(self, conversation: List) -> List[dict]:
        """将 conversation 转为 role/content 列表，保持原顺序。"""
        msgs = []
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            if 'Below are some experiences that may be relevant' in content:
                continue  # skip injected experience
            msgs.append({"role": role, "content": content})
        return msgs

    def _store_endpoint(self) -> str:
        return (
            "summary_task_memory_only_success"
            if self.success_only
            else "summary_task_memory"
        )

    def _post(self, path: str, payload: dict):
        url = self.base_url + path
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"ReMe call failed {url}, code={resp.status_code}, text={resp.text}")
        return resp

    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        if not self._storage_enabled:
            return

        # score 用 reward；ReMe 的 success_only/threshold 在 flow 层处理
        score = getattr(experience, "reward", 0) if hasattr(experience, "reward") else experience.reward
        messages = self._to_messages(experience.conversation)

        payload = {
            "workspace_id": self.workspace_id,
            "trajectories": [
                {
                    "messages": messages,
                    "score": score,
                    "metadata": {"query": task_query, "task_index": task_idx},
                }
            ],
        }

        self._post(self._store_endpoint(), payload)

    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        top_k = min(self.k_retrieval, k)
        payload = {
            "workspace_id": self.workspace_id,
            "query": task_query,
            "top_k": top_k,
        }
        resp = self._post("retrieve_task_memory_vanilla_rewrite", payload)
        data = resp.json()
        rewritten_insight = data.get("answer", "")

        results = []
        exp_text = f"[EXPERIENCE]: {rewritten_insight}"
        results.append(MemoryItem(content=exp_text, metadata={}))
        return results



class HybridTrajInsightMemory(BaseMemory):
    """
    HybridTrajInsightMemory（ReMe 实现）：
    - 存储：调用 ReMe 的 store_hybrid_traj_insight / store_hybrid_traj_insight_only_success
      将 conversation 直接作为 messages 发送，query 作为索引键。
    - 检索：调用 retrieve_hybrid_traj_insight，通过 query 取回原始轨迹文本。
    """

    def __init__(
        self,
        reme_base_url: str = "http://localhost:8123/",
        reme_workspace_id: str = "task_workspace",
        success_only: bool = True,
        k_retrieval: int = 1,
    ):
        super().__init__()
        self.type = "raw"
        self.base_url = reme_base_url.rstrip("/") + "/"
        self.workspace_id = reme_workspace_id
        self.success_only = success_only
        self.k_retrieval = k_retrieval

    def _to_messages(self, conversation: List) -> List[dict]:
        """将 conversation 转为 role/content 列表，保持原顺序。"""
        msgs = []
        for msg in conversation:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            if 'Below are some experiences that may be relevant' in content:
                continue  # skip injected experience
            msgs.append({"role": role, "content": content})
        return msgs

    def _store_endpoint(self) -> str:
        return (
            "store_hybrid_traj_insight_only_success"
            if self.success_only
            else "store_hybrid_traj_insight"
        )

    def _post(self, path: str, payload: dict):
        url = self.base_url + path
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"ReMe call failed {url}, code={resp.status_code}, text={resp.text}")
        return resp

    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        if not self._storage_enabled:
            return

        # score 用 reward；ReMe 的 success_only/threshold 在 flow 层处理
        score = getattr(experience, "reward", 0) if hasattr(experience, "reward") else experience.reward
        messages = self._to_messages(experience.conversation)

        payload = {
            "workspace_id": self.workspace_id,
            "trajectories": [
                {
                    "messages": messages,
                    "score": score,
                    "metadata": {"query": task_query, "task_index": task_idx},
                }
            ],
        }

        self._post(self._store_endpoint(), payload)

    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        top_k = min(self.k_retrieval, k)
        payload = {
            "workspace_id": self.workspace_id,
            "query": task_query,
            "top_k": top_k,
        }
        resp = self._post("retrieve_hybrid_traj_insight", payload)
        data = resp.json()
        mems = data.get("metadata", {}).get("memory_list", [])

        results = []
        for mem in mems:
            exp_text = mem.get("content", "")
            metadata = mem.get("metadata", {})
            task_query = metadata['query']
            score = metadata['score']
            result = "Success" if score == 1.0 else "Failure"
            exp_str = f"[TASK]: {task_query}\n{exp_text}\n[RESULT]: {result}"
            results.append(MemoryItem(content=exp_str, metadata=metadata))
        return results



MEMORY_CLS_MAP = {
    "raw": RawMemory,
    "strategy": StrategyMemory,
    "strategy_rewrite": StrategyRewriteMemory,
    "hybrid_traj_insight": HybridTrajInsightMemory,
    "null": NullMemory,
}