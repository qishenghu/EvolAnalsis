"""
GPQA-Diamond EnvServer for single-choice QA
"""
from typing import Optional, List, Dict, Any
import threading
import re
import os
import logging
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class NotInitializedError(Exception):
    pass

LETTER_LIST = ["A", "B", "C", "D"]

# 默认分片（gpqa只有test split），可通过 GPQA_SPLIT 覆盖
DEFAULT_SPLIT = os.environ.get("GPQA_SPLIT", "test")


def extract_answer(text: str) -> Optional[str]:
    """提取答案字母，优先 <answer> 标签，其次文本中的 A-D"""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1).strip() if match else text.strip()

    letter_match = re.search(r"\b([A-D])\b", candidate, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    return None


def check_answer(pred: Optional[str], gt_letter: str) -> bool:
    return pred is not None and pred.upper() == gt_letter.upper()


class GPQAEnvServer:
    """单轮选择题环境：单次回答，done=True（无效答案时允许重试）"""

    def __init__(self) -> None:
        self._max_id = 0
        self.env: Dict[int, Dict[str, Any]] = {}
        self.ls: List[int] = []
        self._lock = threading.Lock()

        try:
            ds = load_dataset("fingertap/GPQA-Diamond", split=DEFAULT_SPLIT)
            self.dataset = list(ds)
            logger.info(f"Loaded GPQA-Diamond split='{DEFAULT_SPLIT}' samples: {len(self.dataset)}")
        except Exception as e:
            logger.error(f"Failed to load GPQA-Diamond dataset: {e}")
            self.dataset = []

    def create(self, item_id: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1
        self.env[env_idx] = self._fetch_data(item_id)
        self.ls.append(env_idx)
        return env_idx

    def step(self, env_idx, response: str):
        self._check_env_idx(env_idx)
        sample = self.env[env_idx]

        pred_letter = extract_answer(response)
        gt_letter = sample["answer"].strip().upper()

        if pred_letter is None:
            reward = 0.0
            observation = "Unable to identify the answer. Please output the option letter (A/B/C/D) wrapped in <answer>...</answer>, such as <answer>A</answer>, <answer>B</answer>, <answer>C</answer> or <answer>D</answer>."
            done = False
        else:
            if check_answer(pred_letter, gt_letter):
                reward = 1.0
                observation = f"Correct! The correct option is {gt_letter}."
            else:
                reward = 0.0
                observation = f"Incorrect. Your answer is {pred_letter}, but the correct answer is {gt_letter}."
            done = True
        return observation, reward, done, None

    def observation(self, env_idx):
        self._check_env_idx(env_idx)
        sample = self.env[env_idx]
        user_prompt = (
            f"Question:\n{sample['question']}\n\n"
            "You must reason step by step inside <think>...</think>."
            " Then output only the final option letter (A/B/C/D) inside <answer>...</answer>."
        )
        return user_prompt

    def reset(self, env_idx, item_id: Optional[int] = None):
        self._check_env_idx(env_idx)
        self.env[env_idx] = self._fetch_data(item_id if item_id is not None else 0)

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _fetch_data(self, item_id: int):
        if not self.dataset:
            raise ValueError("GPQA-Diamond dataset is empty")
        if item_id < 0 or item_id >= len(self.dataset):
            raise ValueError(f"Item id {item_id} is out of range. Total: {len(self.dataset)}")
        return self.dataset[item_id]

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

gpqa_env_server = GPQAEnvServer()
