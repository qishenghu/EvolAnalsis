from typing import Any, Mapping, Dict, Optional

import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput

class GPQAEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are solving a single-choice question."
                    "You must reason step by step inside <think>...</think> to solve the question. "
                    "Then, in the last line of your response should be of the following format: '<answer>[LETTER]</answer>' (without quotes) where [LETTER] is one of A, B, C or D."
                ),
            }
        ),
        ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        is_eval: bool = True,
        test_data_start_index: int = -1,
        *args,
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.is_eval = is_eval
        self.test_data_start_index = test_data_start_index
        assert self.test_data_start_index >= 0, "test_data_start_index must be >= 0"
        self.id = 0
        data = {"id": 0}
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        question = self._get("observation")
        return question

    def step(self, action: str) -> StepOutput:
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        actual_id = self.test_data_start_index + id
        response = self._post("reset", {"id": actual_id})
        return response

    def close(self):
        response = self._post("close", {})
        return response

class GPQATask(BaseTask):
    env_client_cls = GPQAEnvClient
    env_name = "GPQA"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
