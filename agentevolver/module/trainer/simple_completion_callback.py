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

    @staticmethod
    def _finish_reason_value(value: Any) -> str | None:
        """Return the wire value while tolerating enum-like SDK objects."""
        if value is None:
            return None
        return str(getattr(value, "value", value))

    def _should_use_react_tags(self) -> bool:
        env_params = getattr(self.config.env_service, "env_params", None)
        if env_params is None:
            return False
        return str(getattr(env_params, "action_format", "react") or "react").lower() == "react_tags"

    def _maybe_close_action_tag(self, content: str, stop_reason: Any) -> str:
        if not self._should_use_react_tags():
            return content
        if stop_reason != "</action>":
            return content
        if "<action>" not in content or "</action>" in content:
            return content
        had_trailing_newline = content.endswith("\n")
        content = content.rstrip()
        if had_trailing_newline:
            return f"{content}</action>"
        return f"{content}\n</action>"

    async def __call__(self, messages: List[Dict[str, str]], completions: ChatCompletion, info: Dict[str, Any]):
        choice = completions.choices[0]
        message = choice.message.model_dump(exclude_unset=True, exclude_none=True)
        message["role"] = message.get("role", "assistant")
        if "content" not in message:
            message["content"] = ""

        finish_reason = self._finish_reason_value(
            getattr(choice, "finish_reason", None)
        )
        # ``length`` is a completed rollout event, not a transport error.  Keep
        # the partial token stream exactly as returned and let the environment
        # score the (normally malformed/incomplete) action as a failure.
        truncated_by_length = finish_reason == "length"
        stop_reason = getattr(choice, "stop_reason", None)
        original_content = message["content"]
        tokens = self._extract_tokens(choice)
        if original_content == "" or not tokens:
            raise RuntimeError(
                "rollout returned an empty completion or omitted per-token IDs/logprobs; "
                "refusing to fabricate a trainable assistant turn"
            )
        message["content"] = self._maybe_close_action_tag(message["content"], stop_reason)
        if finish_reason != "stop":
            logger.warning(str(finish_reason))
            logger.bind(bad_case=True).error('non-stop finish reason')
            logger.bind(bad_case=True).error(str(choice))

        t = {
            "role": message["role"],
            "request_id": completions.id,
            "content": message["content"],
            # Content before deterministic environment-facing stop-tag repair.
            # This text and ``tokens`` describe the same sampled event.
            "sampled_content": original_content,
            "finish_reason": finish_reason,
            "truncated_by_length": truncated_by_length,
            "stop_reason": stop_reason,
        }
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
