"""
AIMEEnvServer
"""

from typing import Optional
import threading
import json
import re
import os
import datasets

file_path = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotInitializedError(Exception):
    pass


# AIME数据划分：64个题作为test，其他作为train
# 假设AIME2024有30题，AIME2025有30题，总共60题
# 我们选择前64个作为test（如果总数超过64，则取前64；如果少于64，则全部作为test）
TEST_SIZE = 64

# 数据路径配置（可以通过环境变量覆盖）
aime_data_path = os.environ.get(
    "AIME_DATA_PATH",
    os.path.join(file_path, "..", "data"),
)


def extract_answer(text: str) -> Optional[str]:
    """
    从文本中提取答案。
    AIME答案通常是0-999之间的整数。
    尝试提取最后一个数字作为答案。
    """
    # 尝试提取<answer>标签中的内容
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
        # 提取数字
        numbers = re.findall(r'\d+', answer_text)
        if numbers:
            return numbers[-1]  # 返回最后一个数字
    
    # 如果没有<answer>标签，尝试提取文本末尾的数字
    numbers = re.findall(r'\b\d{1,3}\b', text)  # 匹配1-3位数字（AIME答案范围）
    if numbers:
        return numbers[-1]
    
    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    # 移除前导零（但保留单个0）
    answer = answer.strip()
    if answer.startswith('0') and len(answer) > 1:
        answer = answer.lstrip('0')
        if not answer:  # 如果全部是0，返回"0"
            answer = "0"
    return answer


def check_answer(prediction: str, ground_truth: str) -> bool:
    """检查答案是否正确"""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    return pred_normalized == gt_normalized


class AIMEEnvServer:
    """
    AIME数学推理环境服务器
    Single-turn任务：每次step都会返回done=True
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        self.ls = []
        self._lock = threading.Lock()

        # 加载AIME数据集
        # 假设数据格式：每个数据项包含 'problem' 和 'answer' 字段
        # 数据可以从HuggingFace datasets或本地文件加载
        
        # 尝试从本地文件加载
        train_data_path = os.path.join(aime_data_path, "train.jsonl")
        test_data_path = os.path.join(aime_data_path, "test.jsonl")
        
        if os.path.exists(train_data_path) and os.path.exists(test_data_path):
            # 从本地JSONL文件加载
            train_data = []
            test_data = []
            
            with open(train_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line))
            
            self.dataset = {
                "test": test_data,
                "train": train_data,
            }
        else:
            # 尝试从HuggingFace datasets加载
            try:
                # 尝试加载AIME2024和AIME2025数据集
                aime2024 = datasets.load_dataset("Maxwell-Jia/AIME_2024", split="train")
                aime2025 = None
                try:
                    aime2025 = datasets.load_dataset("selectdataset/AIME_2025", split="train")
                except:
                    logger.warning("Could not load AIME2025 dataset, using only AIME2024")
                
                # 合并数据
                all_data = []
                for item in aime2024:
                    # 标准化数据格式
                    data_item = {
                        "problem": item.get("Problem", item.get("problem", "")),
                        "answer": str(item.get("Answer", item.get("answer", ""))),
                    }
                    all_data.append(data_item)
                
                if aime2025:
                    for item in aime2025:
                        data_item = {
                            "problem": item.get("problem", ""),
                            "answer": str(item.get("answer", "")),
                        }
                        all_data.append(data_item)
                
                # 划分train和test
                # 选择前TEST_SIZE个作为test，其他作为train
                test_data = all_data[:TEST_SIZE] if len(all_data) >= TEST_SIZE else all_data
                train_data = all_data[TEST_SIZE:] if len(all_data) > TEST_SIZE else []
                
                self.dataset = {
                    "test": test_data,
                    "train": train_data,
                }
                
                logger.info(f"Loaded {len(test_data)} test problems and {len(train_data)} train problems")
            except Exception as e:
                logger.error(f"Failed to load AIME dataset: {e}")
                # 创建空数据集作为fallback
                self.dataset = {
                    "test": [],
                    "train": [],
                }

    def create(self, item_id: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1

        self.env[env_idx] = self._fetch_data(
            item_id
        )  # redundancy fetch to prevent NoneType Error
        self.ls.append(env_idx)

        return env_idx

    def step(self, env_idx, response: str):
        """
        Perform a step in the environment with the given action.
        对于single-turn任务，第一次step就返回done=True
        
        Input:
            env_idx: the index of the environment
            response: agent的响应（应该包含答案）
        Output:
            observation: the observation after taking the action
            reward: the reward received after taking the action (1 if correct, 0 if incorrect)
            done: always True for single-turn task
            info: additional information (not used here)
        """
        self._check_env_idx(env_idx)
        
        # 提取答案
        extracted_answer = extract_answer(response)
        ground_truth = str(self.env[env_idx]["answer"])
        
        # 计算reward
        if extracted_answer is None:
            reward = 0.0
            observation = "Your response does not contain a valid answer. Please provide your final answer as a number (0-999)."
        else:
            if check_answer(extracted_answer, ground_truth):
                reward = 1.0
                observation = f"Correct! The answer is {ground_truth}."
            else:
                reward = 0.0
                observation = f"Incorrect. Your answer was {extracted_answer}, but the correct answer is {ground_truth}."
        
        # Single-turn任务：总是返回done=True
        done = True
        
        return observation, reward, done, None

    def observation(self, env_idx):
        """返回当前问题的观察"""
        self._check_env_idx(env_idx)
        problem = self.env[env_idx]["problem"]
        # 格式化问题提示
        user_prompt = f"""Solve the following AIME math problem. Provide your reasoning and then give your final answer as a number between 0 and 999 (inclusive). 

Problem: {problem.strip()}

Please reason step by step and provide your final answer."""
        return user_prompt

    def reset(self, env_idx, item_id: Optional[int] = None):
        """重置环境到指定的问题"""
        self._check_env_idx(env_idx)
        self.env[env_idx] = self._fetch_data(item_id)

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _fetch_data(self, item_id: int):
        """
        从数据集中获取指定item_id的数据
        item_id是全局索引：
        - [0, TEST_SIZE): test数据
        - [TEST_SIZE, TEST_SIZE + len(train)): train数据
        """
        total_test = len(self.dataset["test"])
        total_train = len(self.dataset["train"])
        
        if item_id < total_test:
            # test数据：直接使用item_id作为索引
            return self.dataset["test"][item_id]
        elif item_id < total_test + total_train:
            # train数据：需要减去test数据的偏移量
            return self.dataset["train"][item_id - total_test]
        else:
            raise ValueError(
                f"Item id {item_id} is out of range. "
                f"Test: [0, {total_test}), Train: [{total_test}, {total_test + total_train})"
            )
    
    def __del__(self):
        for idx in self.ls:
            if idx in self.env:
                del self.env[idx]
                print(f"-------Env {idx} closed--------")
    
    def close(self, id):
        try:
            self._check_env_idx(id)
            self.ls.remove(id)
            del self.env[id]
            print(f"-------Env {id} closed--------")
            return True
        except Exception as e:
            print(f"Error closing env {id}: {e}")
            return False

aime_env_server = AIMEEnvServer()
