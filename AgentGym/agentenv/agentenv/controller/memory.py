"""
Memory模块：用于存储和检索经验（Experience）
适配AgentGym框架的ExperienceOutput格式
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import os
import uuid
import json
import re
import chromadb
import dspy
import random

lm_gpt41_mini = dspy.LM('gpt-4.1-mini')
lm_qwen_plus = dspy.LM('openai/qwen-plus')
lm_qwen_flash = dspy.LM('openai/qwen-flash')
lm_dict = {
    'gpt-4.1-mini': lm_gpt41_mini,
    'qwen-plus': lm_qwen_plus,
    'qwen-flash': lm_qwen_flash,
}
dspy.configure(experimental=True)

from .types import ExperienceOutput, APIExperienceOutput, ConversationMessage, APIConversationMessage


@dataclass
class MemoryItem:
    """可检索的知识单元"""
    content: str
    metadata: dict


class BaseMemory(ABC):
    """Memory接口：所有Memory模块的基类"""
    
    def __init__(self):
        self._storage_enabled = True
        self.type = None
    
    @abstractmethod
    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        """
        存储一个完成的experience
        
        Args:
            experience: 要存储的经验
            task_query: 任务查询字符串
            task_idx: 任务索引
        """
        pass
    
    @abstractmethod
    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        """
        为新任务检索k个最相关的记忆
        
        Args:
            task_query: 任务查询字符串
            k: 检索数量
            
        Returns:
            检索到的MemoryItem列表
        """
        pass
    
    def disable_storage(self):
        """禁用学习（用于评估）"""
        self._storage_enabled = False
    
    def enable_storage(self):
        """启用学习（用于训练）"""
        self._storage_enabled = True


class NullMemory(BaseMemory):
    """空Memory：不存储也不检索任何内容（用于baseline）"""
    
    def __init__(self):
        super().__init__()
        self.type = "null"
    
    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        pass
    
    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        return []


class RawMemory(BaseMemory):
    """
    RawMemory: 存储完整的原始轨迹到向量数据库(Chroma)
    使用向量相似度搜索进行检索
    """
    
    def __init__(
        self,
        local_exp_pool_path: str = "",
        collection_name: str = "exp_pool",
        success_only: bool = True,
        k_retrieval: int = 1,
        include_thought: bool = True,
    ):
        super().__init__()
        self.type = "raw"
        self.local_exp_pool_path = local_exp_pool_path
        self.collection_name = collection_name
        self.success_only = success_only
        self.k_retrieval = k_retrieval
        self.vector_store = None
        self._embedding = None
        self.inner_collection_name = f"{self.collection_name}_in_memory" + f"_{uuid.uuid4().hex}"
        self.include_thought = include_thought
        self._initialize_store()
    
    def _initialize_store(self):
        """初始化向量存储"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            # from langchain_community.embeddings import LocalAIEmbeddings
            from langchain_openai import OpenAIEmbeddings

            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            
            self._embedding = OpenAIEmbeddings(
                model="text-embedding-3-small",
                # openai_api_key="random-string",
                # base_url="http://127.0.0.1:8000/v1",
            )
            # self._embedding = HuggingFaceEmbeddings(
            #     model_name="BAAI/bge-large-en-v1.5",
            #     model_kwargs={"trust_remote_code": True},
            # )
            
            # 创建内存中的collection
            if not self.local_exp_pool_path:
                in_memory_collection = self.inner_collection_name
                self.vector_store = Chroma(
                    collection_name=in_memory_collection,
                    embedding_function=self._embedding,
                )
                return
            
            # 如果提供了路径，尝试加载已有数据
            if self.local_exp_pool_path.endswith(".jsonl"):
                # 从jsonl文件加载
                in_memory_collection = self.inner_collection_name
                self.vector_store = Chroma(
                    collection_name=in_memory_collection,
                    embedding_function=self._embedding,
                )
                with open(self.local_exp_pool_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        doc = Document(
                            page_content=item.get('page_content', ''),
                            metadata=item.get('metadata', {})
                        )
                        self.vector_store.add_documents([doc])
                print(f"[RawMemory] Loaded experiences from '{self.local_exp_pool_path}'")
                return

            if self.local_exp_pool_path.endswith(".pkl"):
                import pickle
                
                with open(self.local_exp_pool_path, 'rb') as f:
                    snapshot = pickle.load(f)

                    if len(snapshot["ids"]) == 0:
                        in_memory_collection = self.inner_collection_name
                        self.vector_store = Chroma(
                            collection_name=in_memory_collection,
                            embedding_function=self._embedding,
                        )
                        return
                with open(self.local_exp_pool_path, 'rb') as f:
                    snapshot = pickle.load(f)
                    mem_client = chromadb.Client()
                    mem_collection = mem_client.create_collection(self.inner_collection_name)
                    mem_collection.add(
                        ids=snapshot["ids"],
                        embeddings=snapshot["embeddings"],
                        documents=snapshot["documents"],
                        metadatas=snapshot["metadatas"],
                    )
                    self.vector_store = Chroma(
                        client=mem_client,
                        collection_name=self.inner_collection_name,
                        embedding_function=self._embedding,  # only used for queries / added docs
                    )
                    init_doc_cnt = len(self.vector_store.get(include=["documents"]).get("documents", []))
                    print(f"[RawMemory] Loaded {init_doc_cnt} experiences from '{self.local_exp_pool_path}'")
                    return
            
            # if os.path.isdir(self.local_exp_pool_path):
            #     # 从Chroma持久化目录加载
            #     try:
            #         source_store = Chroma(
            #             collection_name=self.collection_name,
            #             embedding_function=self._embedding,
            #             persist_directory=self.local_exp_pool_path,
            #         )
            #         data = source_store.get(include=["embeddings", "documents", "metadatas"])

            #         mem_client = chromadb.Client()
            #         mem_collection = mem_client.create_collection(self.collection_name + '_in_memory')
                    
            #         mem_collection.add(
            #             ids=data["ids"],
            #             embeddings=data["embeddings"],
            #             documents=data["documents"],
            #             metadatas=data["metadatas"],
            #         )

            #         self.vector_store = Chroma(
            #             client=mem_client,
            #             collection_name=self.collection_name + '_in_memory',
            #             embedding_function=self._embedding,  # only used for queries / added docs
            #         )
            #         init_doc_cnt = len(self.vector_store.get(include=["documents"]).get("documents", []))
            #         print(f"[RawMemory] Loaded {init_doc_cnt} experiences from '{self.local_exp_pool_path}'")

            #     except Exception as error:
            #         print(f"[RawMemory] Failed to load exp pool: {error}")
        except ImportError as e:
            print("[RawMemory] Warning: langchain not installed. Memory will not work.")
            print(f"[RawMemory] Error: {e}")
            self.vector_store = None
    
    def _format_trajectory(self, experience: ExperienceOutput | APIExperienceOutput) -> str:
        """将experience转换为轨迹字符串"""
        trajectory_str = ""
        conversation = experience.conversation
        include_thought = self.include_thought
        
        # 跳过conversation_start，从实际交互开始
        # 通常conversation_start有2条消息（system和assistant确认）
        start_idx = 2 if len(conversation) > 2 else 0
        if conversation[start_idx]['role'] == 'user':
            if 'Below are past task experiences' in conversation[start_idx]['content']:
                # This is the memory prompt, skip it
                start_idx += 1
        
        step_num = 1
        for i in range(start_idx, len(conversation)):
            msg = conversation[i]
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            
            if role in ["user", "human"]:
                trajectory_str += f"Observation #{step_num}: {content}\n"
            elif role in ["assistant", "gpt"]:
                if 'Thought:' in content and 'Action:' in content:
                    # it means the ReAct format contains thought + action
                    if include_thought:
                        thought = content.split('Thought:')[1].split('Action:')[0].strip()
                        trajectory_str += f"Thought #{step_num}: {thought}\n"
                    content = content.split('Action:')[1].strip()
                elif 'Action:' in content:
                    content = content.split('Action:')[1].strip()
                trajectory_str += f"Action #{step_num}: {content}\n"
                step_num += 1
        
        return trajectory_str.strip()
    
    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        """存储experience到向量数据库"""
        if not self._storage_enabled or self.vector_store is None:
            return
        
        # 只存储成功的经验（如果设置了success_only）
        reward = experience.reward
        if self.success_only and reward <= 0:
            return
        
        try:
            from langchain_core.documents import Document
            
            trajectory_str = self._format_trajectory(experience)
            result = "success" if reward > 0 else "failure"
            
            doc = Document(
                page_content=task_query,
                metadata={
                    'task_index': task_idx,
                    'task': task_query,
                    'task_query': task_query,
                    'reward': reward,
                    'result': result,
                    'trajectory': trajectory_str,
                }
            )
            self.vector_store.add_documents([doc])
            
            total = len(self.vector_store.get()['ids'])
            print(f"[RawMemory] Stored 1 trajectory. Total memories: {total}")
        except Exception as e:
            print(f"[RawMemory] Failed to store experience: {e}")
    
    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        """从向量数据库检索相似经验"""
        if self.vector_store is None:
            return []
        
        size_of_memory = len(self.vector_store.get()["ids"])
        if size_of_memory < self.k_retrieval:
            return []
        
        try:
            filter_dict = {'reward': 1} if self.success_only else {}
            documents = self.vector_store.similarity_search(
                task_query,
                k=min(self.k_retrieval, k),
                filter=filter_dict if filter_dict else None
            )
            
            memories = []
            for doc in documents:
                task_query_retrieved = doc.metadata.get('task_query', doc.metadata.get('task', ''))
                trajectory = doc.metadata.get('trajectory', '')
                result = doc.metadata.get('result', '')
                if not result:
                    reward = doc.metadata.get('reward', 0)
                    result = "success" if reward > 0 else "failure"
                
                exp_str = f"Task: {task_query_retrieved}\nTrajectory: {trajectory}\nResult: {result}"
                memories.append(MemoryItem(content=exp_str, metadata=doc.metadata))
            
            return memories
        except Exception as e:
            print(f"[RawMemory] Retrieval failed: {e}")
            return []
    
    def persist(self, path: str):
        """持久化Memory到文件"""
        if self.vector_store is None:
            return
        assert path.endswith(".pkl"), "Only .pkl files are supported for persistence"
        import pickle
        try:
            # snapshot = self.vector_store.get(include=["metadatas", "documents"])
            # documents = snapshot.get("documents", [])
            # metadatas = snapshot.get("metadatas", [])
            snapshot = self.vector_store.get(include=["embeddings", "documents", "metadatas"])
            assert 'embeddings' in snapshot, "embeddings not found in snapshot"
            assert 'documents' in snapshot, "documents not found in snapshot"
            assert 'metadatas' in snapshot, "metadatas not found in snapshot"
            
            with open(path, 'wb') as f:
                pickle.dump(snapshot, f)
            
            print(f"[RawMemory] Persisted {len(snapshot['documents'])} trajectories to '{path}'")
        except Exception as e:
            print(f"[RawMemory] Failed to persist: {e}")


class StrategyMemory(BaseMemory):
    """
    StrategyMemory：从原始轨迹中提炼简化的“策略/步骤”并做向量检索
    目标：提供低保真但高可用的操作纲要，减少上下文长度。
    """
    
    def __init__(
        self,
        local_exp_pool_path: str = "",
        collection_name: str = "strategy_pool",
        success_only: bool = True,
        k_retrieval: int = 1,
        max_strategy_chars: int = 1500,
        default_lm: str = 'gpt-4.1-mini',
        include_thought: bool = True,
    ):
        super().__init__()
        self.type = "strategy"
        self.local_exp_pool_path = local_exp_pool_path
        self.collection_name = collection_name
        self.success_only = success_only
        self.k_retrieval = k_retrieval
        self.max_strategy_chars = max_strategy_chars
        self.vector_store = None
        self._embedding = None
        self.inner_collection_name = f"{self.collection_name}_in_memory" + f"_{uuid.uuid4().hex}"
        self._initialize_store()
        self._strategy_predictor = None  # lazy init if dspy is available
        # 存储原始的字符串键，用于多进程序列化
        self.default_lm_key = default_lm
        self.include_thought = include_thought
        self.default_lm = lm_dict[default_lm]
    
    def _initialize_store(self):
        """初始化向量存储（与 RawMemory 类似，使用 OpenAIEmbeddings + Chroma）"""
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            
            self._embedding = OpenAIEmbeddings(
                model="text-embedding-3-small",
            )
            
            if not self.local_exp_pool_path:
                in_memory_collection = self.inner_collection_name
                self.vector_store = Chroma(
                    collection_name=in_memory_collection,
                    embedding_function=self._embedding,
                )
                return
            
            # 仅支持从 .pkl 快照加载（与 RawMemory 持久化格式一致）
            if self.local_exp_pool_path.endswith(".pkl"):
                import pickle
                with open(self.local_exp_pool_path, 'rb') as f:
                    snapshot = pickle.load(f)
                    if len(snapshot.get("ids", [])) == 0:
                        in_memory_collection = self.inner_collection_name
                        self.vector_store = Chroma(
                            collection_name=in_memory_collection,
                            embedding_function=self._embedding,
                        )
                        return
                with open(self.local_exp_pool_path, 'rb') as f:
                    snapshot = pickle.load(f)
                    mem_client = chromadb.Client()
                    mem_collection = mem_client.create_collection(self.inner_collection_name)
                    mem_collection.add(
                        ids=snapshot["ids"],
                        embeddings=snapshot["embeddings"],
                        documents=snapshot["documents"],
                        metadatas=snapshot["metadatas"],
                    )
                    self.vector_store = Chroma(
                        client=mem_client,
                        collection_name=self.inner_collection_name,
                        embedding_function=self._embedding,
                    )
                    init_doc_cnt = len(self.vector_store.get(include=["documents"]).get("documents", []))
                    print(f"[StrategyMemory] Loaded {init_doc_cnt} strategies from '{self.local_exp_pool_path}'")
                    return
        except ImportError as e:
            print("[StrategyMemory] Warning: langchain not installed. Memory will not work.")
            print(f"[StrategyMemory] Error: {e}")
            self.vector_store = None
    
    def _format_trajectory(self, experience: ExperienceOutput | APIExperienceOutput) -> str:
        """
        生成简短的轨迹（Observation/Action 对），用于策略提炼和检索展示。
        """
        trajectory_str = ""
        conversation = experience.conversation
        include_thought = self.include_thought
        # 跳过开头的 system/assistant prompt
        start_idx = 2 if len(conversation) > 2 else 0
        if start_idx < len(conversation):
            msg0 = conversation[start_idx]
            if isinstance(msg0, dict):
                content0 = msg0.get("content", msg0.get("value", ""))
            else:
                content0 = getattr(msg0, "content", getattr(msg0, "value", ""))
            if isinstance(content0, str) and "Below are past task experiences" in content0:
                start_idx += 1
        
        step_num = 1
        for i in range(start_idx, len(conversation)):
            msg = conversation[i]
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
            else:
                role = getattr(msg, "role", getattr(msg, "from", ""))
                content = getattr(msg, "content", getattr(msg, "value", ""))
            
            if role in ["user", "human"]:
                trajectory_str += f"Observation #{step_num}: {content}\n"
            elif role in ["assistant", "gpt"]:
                if 'Thought:' in content and 'Action:' in content:
                    # it means the ReAct format contains thought + action
                    if include_thought:
                        thought = content.split('Thought:')[1].split('Action:')[0].strip()
                        trajectory_str += f"Thought #{step_num}: {thought}\n"
                    content = content.split('Action:')[1].strip()
                elif 'Action:' in content:
                    content = content.split('Action:')[1].strip()
                trajectory_str += f"Action #{step_num}: {content}\n"
                step_num += 1
        
        return trajectory_str.strip()
    
    def _build_strategy_text(self, task_query: str, trajectory: str) -> str:
        """
        使用 LLM (dspy) 将任务描述与轨迹片段压缩为可复用策略；若 dspy 不可用则回退到规则式摘要。
        """
        class StrategySignature(dspy.Signature):
            """
            Convert a short trajectory into a reusable meta-strategy.

            The output should NOT retell what happened.
            Instead, extract general rules that would help another agent
            solve similar tasks in the future (meta-level guidelines).
            """

            task_description = dspy.InputField(
                desc=(
                    "Task description or query."
                )
            )
            trajectory = dspy.InputField(
                desc=(
                    "Observation/Action trace of the task execution."
                    "Use this only as evidence for patterns, not to copy verbatim."
                )
            )

            strategy = dspy.OutputField(
                desc=(
                    "A numbered list of 5-10 rules that describe a strategy for solving similar tasks."
                    " Each rule must be ONE sentence and separated by a new line character."
                    " Focus on the pattern or strategy of the task execution, not on specific entities from the trajectory or the task query."
                    " The numbered rules should be reusable across similar tasks."
                )
            )
        
        if self._strategy_predictor is None:
            self._strategy_predictor = dspy.Predict(StrategySignature)
        
        with dspy.context(lm=self.default_lm):
            result = self._strategy_predictor(
                task_description=task_query,
                trajectory=trajectory,
            )
        strategy_text = result.strategy.strip()
        filtered = []
        for line in strategy_text.splitlines():
            stripped = line.strip()
            if re.match(r"^\d+[\.\)]\s", stripped):
                filtered.append(stripped)
        if filtered:
            strategy_text = "\n".join(filtered)
        # 控制长度，避免过长策略污染存储
        strategy_text = strategy_text[: self.max_strategy_chars]
        return strategy_text
    
    def store(self, experience: ExperienceOutput | APIExperienceOutput, task_query: str, task_idx: int):
        """存储策略摘要到向量数据库"""
        if not self._storage_enabled or self.vector_store is None:
            return
        
        reward = experience.reward
        if self.success_only and reward <= 0:
            return
        
        try:
            from langchain_core.documents import Document
            
            trajectory = self._format_trajectory(experience)
            strategy_text = self._build_strategy_text(task_query, trajectory)
            result = "success" if reward > 0 else "failure"
            
            doc = Document(
                page_content=strategy_text,
                metadata={
                    "task_index": task_idx,
                    "task_query": task_query,
                    "reward": reward,
                    "result": result,
                    "strategy": strategy_text,
                    "trajectory": trajectory,
                },
            )
            self.vector_store.add_documents([doc])
            total = len(self.vector_store.get().get("ids", []))
            print(f"[StrategyMemory] Stored 1 strategy. Total strategies: {total}")
            # if random.random() < 0.3:
            print(f"[StrategyMemory] Strategy: \n{strategy_text}\n\n")
        except Exception as e:
            print(f"[StrategyMemory] Failed to store strategy: {e}")
    
    def retrieve(self, task_query: str, k: int = 3) -> List[MemoryItem]:
        """检索相似策略"""
        if self.vector_store is None:
            return []
        
        size_of_memory = len(self.vector_store.get().get("ids", []))
        if size_of_memory < self.k_retrieval:
            return []
        
        try:
            filter_dict = {"reward": 1} if self.success_only else {}
            documents = self.vector_store.similarity_search(
                task_query,
                k=min(self.k_retrieval, k),
                filter=filter_dict if filter_dict else None,
            )
            
            memories = []
            for doc in documents:
                task_query_retrieved = doc.metadata.get('task_query', doc.metadata.get('task', ''))
                strategy = doc.metadata.get("strategy", doc.page_content)
                result = doc.metadata.get('result', '')
                if not result:
                    reward = doc.metadata.get('reward', 0)
                    result = "success" if reward > 0 else "failure"
                exp_str = f"Task: {task_query_retrieved}\nStrategy:\n{strategy}\nResult: {result}"
                memories.append(MemoryItem(content=exp_str, metadata=doc.metadata))
            return memories
        except Exception as e:
            print(f"[StrategyMemory] Retrieval failed: {e}")
            return []
    
    def persist(self, path: str):
        """持久化策略向量库到 .pkl（格式与 RawMemory 一致）"""
        if self.vector_store is None:
            return
        assert path.endswith(".pkl"), "Only .pkl files are supported for persistence"
        import pickle
        try:
            snapshot = self.vector_store.get(include=["embeddings", "documents", "metadatas"])
            assert "embeddings" in snapshot and "documents" in snapshot and "metadatas" in snapshot
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)
            print(f"[StrategyMemory] Persisted {len(snapshot['documents'])} strategies to '{path}'")
        except Exception as e:
            print(f"[StrategyMemory] Failed to persist: {e}")

