"""
MathBenchEnvServer for single-choice math reasoning
"""

from typing import Optional, List, Dict, Any
import threading
import json
import re
import os
import logging

file_path = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class NotInitializedError(Exception):
    pass

# 默认数据路径（用户可通过 MATHBENCH_DATA_PATH 覆盖）
DEFAULT_DATA_PATH = os.environ.get(
    "MATHBENCH_DATA_PATH",
    "/home/qisheng/agent/EvolCL/mathbench_v1/college_mix/single_choice_en_shuffled.jsonl",
)

LETTER_LIST = ["A", "B", "C", "D"]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(path):
        logger.error(f"MathBench data file not found: {path}")
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Invalid json line skipped: {line[:100]}")
                continue
    return data


def extract_answer(text: str, options: List[str]) -> Optional[str]:
    """从模型输出中提取选项字母或文本"""
    # 优先从<answer>标签中提取
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text.strip()

    # 查找字母
    letter_match = re.search(r"\b([A-D])\b", candidate, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()

    # 匹配选项文本
    candidate_lower = candidate.lower()
    for idx, opt in enumerate(options):
        if opt.lower() in candidate_lower:
            return LETTER_LIST[idx]

    return None


def check_answer(pred: Optional[str], gt_letter: str) -> bool:
    if pred is None:
        return False
    return pred.upper() == gt_letter.upper()


class MathBenchEnvServer:
    """单轮选择题环境：每个 step 直接结束"""

    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}
        self.ls = []
        self._lock = threading.Lock()

        # 加载数据
        self.dataset = _read_jsonl(DEFAULT_DATA_PATH)
        if not self.dataset:
            logger.warning("MathBench dataset is empty; please check path or file.")
        else:
            logger.info(f"Loaded MathBench samples: {len(self.dataset)}")

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

        pred_letter = extract_answer(response, sample["options"])
        gt_letter = sample["answer"].strip().upper()

        if pred_letter is None:
            reward = 0.0
            observation = "Unable to identify the answer, please provide the option letter wrapped in <answer>...</answer>, such as <answer>A</answer>, <answer>B</answer>, <answer>C</answer> or <answer>D</answer>."
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
        opts = "\n".join([f"{LETTER_LIST[i]}. {opt}" for i, opt in enumerate(sample["options"][:len(LETTER_LIST)])])
        user_prompt = (
            f"Question: {sample['question']}\n\nOptions:\n{opts}\n\nYou must reason step by step inside <think>...</think> to solve the question. Then, output only the final chosen option letter (A/B/C/D), wrapped inside <answer>...</answer>."
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
            raise ValueError("MathBench dataset is empty")
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

mathbench_env_server = MathBenchEnvServer()
