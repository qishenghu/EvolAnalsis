from typing import Any, Dict, List

from loguru import logger
from omegaconf import DictConfig
from openai.types.chat.chat_completion import ChatCompletion
from verl import DataProto
from verl.workers.rollout.chat_scheduler import CompletionCallback

class TokenAndProb:
    def __init__(self, t):
        # print(t)
        # ChatCompletionTokenLogprob(token='token_id:73594', bytes=[96, 96, 96], logprob=-1.9073468138230965e-06, top_logprobs=[])
        self.token_id = int(t.token.split('token_id:')[-1])
        self.logprob = t.logprob
        try:
            self.decoded_string = bytes(t.bytes).decode('utf-8')
        except:
            self.decoded_string = '<cannot decode>' + str(t.bytes)


class SimpleCompletionCallback(CompletionCallback):
    def __init__(self, config: DictConfig, scheduler: "ChatCompletionScheduler"):
        super().__init__(config, scheduler)
        logger.info("=" * 10 + "SimpleCompletionCallback is inited~" + "=" * 10)

    @staticmethod
    def _extract_tokens(choice) -> List[TokenAndProb]:
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None:
            return []
        content = getattr(logprobs, "content", None)
        if not content:
            return []
        return [TokenAndProb(token) for token in content]

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        choice = completions.choices[0]
        message = choice.message.model_dump(exclude_unset=True, exclude_none=True)
        message["role"] = message.get("role", "assistant")
        if "content" not in message:
            message["content"] = ""

        finish_reason = getattr(choice, "finish_reason", None)
        if message["content"] == "" or finish_reason != "stop":
            logger.warning(str(finish_reason))
            logger.bind(bad_case=True).error('empty content or non-stop finish reason')
            logger.bind(bad_case=True).error(str(choice))
            if message["content"] == "":
                # Preserve the assistant turn so the caller never falls back
                # to the previous user message when stop handling is unusual.
                message["content"] = "im_end"

        t = {
            "role": message["role"],
            "request_id": completions.id,
            "content": message["content"],
        }
        tokens = self._extract_tokens(choice)
        if tokens:
            t["tokens"] = tokens
        messages.append(t)

    def postprocess(self, batch: DataProto, batch_conversations: List[List[Dict[str, str]]], n: int) -> DataProto:
        """Post process batch data.

        Args:
            batch: Batch input messages from RLHFDataset.
            batch_conversations: List of messages including raw prompt, assistant response, tool response.
                Note that `len(batch_conversations) == len(batch) * n`, e.g n=2,
                batch_conversations=[messages_0_0, messages_0_1, messages_1_0, messages_1_1, ...]
            n: How many chat completion choices to generate for each input message.

        Returns:
            Batch data, should include ["prompts", "responses", "response_mask", "input_ids", "attention_mask", "position_ids"].
        """
        raise NotImplementedError
