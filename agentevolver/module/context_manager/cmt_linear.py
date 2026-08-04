import copy
import uuid
import json
import re
import hashlib
import torch
from typing import List, Union
from agentevolver.schema.trajectory import Sample, Reward
from agentevolver.schema.trajectory import Sample, Trajectory
from agentevolver.utils.compute_madness import repetition_penalty_reward_scalar
from agentevolver.module.context_manager.cmt_base import ExtendedMessage, ContextManagerBase, extract_assistant_header_tokens, auto_tokenize_message
from agentevolver.module.context_manager.cmt_base import chat_template_ids, find_sublist_indices, replace_token_ids
from agentevolver.module.context_manager.context_policy import (
    ContextBuildResult,
    DecisionSnapshot,
    StructuredContextPolicy,
)
from best_logger import register_logger, print_listofdict, print_dict, print_nested, NestedJsonItem, SeqItem
from agentevolver.module.exp_manager.exp_manager import ExperienceWorker, TrajExpConfig




class Linear_CMT(Trajectory, ContextManagerBase):
    """
    A linear context manager template that handles the conversation flow between LLM and environment.
    This class manages the context window, tokenization, and message history in a linear fashion.

    Attributes:
        config: Configuration object containing environment and model settings
        tokenizer: Tokenizer instance for processing text
        full_context (List[ExtendedMessage]): List of all messages in the conversation
        current_context_status (str): Current status of the context
        max_seq_length (int): Maximum sequence length for the context window
        max_env_output_length (int): Maximum length for environment outputs
        terminal_rewards_dict (dict): Dictionary storing terminal rewards
    """

    def __init__(self, config, tokenizer):
        """
        Initializes the Linear_CMT class with the provided configuration and tokenizer.

        Args:
            config: Configuration object containing environment and model settings.
            tokenizer: Tokenizer instance for processing text.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.full_context: List[ExtendedMessage] = []  # ⭐ Initialize the list to store all messages in the conversation
        self.current_context_status = ""
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len: int = self.config.actor_rollout_ref.rollout.max_model_len
        self.max_seq_length: int = max_model_len - max_response_length  # ⭐ Calculate the maximum sequence length for the context window
        self.max_env_output_length: int = self.config.actor_rollout_ref.rollout.max_env_len
        self.sliding_window_size: int = getattr(
            self.config.actor_rollout_ref.rollout, "sliding_window_size", -1
        )
        self.context_policy = StructuredContextPolicy(
            tokenizer, self.config.actor_rollout_ref.rollout
        )
        self.decision_snapshots: List[DecisionSnapshot] = []
        self._pending_context_result: ContextBuildResult | None = None
        # Qwen3.5 thinking mode: strip <think> blocks from history assistant turns
        # (mirrors the chat template's history branch) so that token_arr matches
        # the prompts the model actually sees in later turns.
        # Reasoning-context budget in tokens: 0 strips every history <think>,
        # -1 keeps them all, N>0 keeps the most recent N reasoning tokens.
        # The legacy boolean maps onto the two endpoints.
        budget = getattr(
            self.config.actor_rollout_ref.rollout, "reasoning_context_budget", None
        )
        if budget is None:
            budget = 0 if bool(getattr(
                self.config.actor_rollout_ref.rollout, "strip_think_in_history", False
            )) else -1
        self.reasoning_context_budget: int = int(budget)
        # kept for the call sites that only ask "is any history think dropped?"
        self.strip_think_in_history: bool = self.reasoning_context_budget >= 0
        self.blackout_token_combo = extract_assistant_header_tokens(tokenizer)
        self.generated_token_cnt = 0

        self.terminal_rewards_dict = {}
        self.discarded = False
        self.is_terminated = False
        self.reward: Union[Reward, None] = None
        self.context_time_cost = 0
        self.tag: str = ""
        self.task_id: str = ""
        # self.task_train_exp_mode: str = ""
        self.current_batch_success_rate:float = -1.0
        self.llm_output_mistakes = {}
        # self.experiences = []

        # log_prob_max_token_len_per_gpu: int = self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
        # ref_log_prob_max_token_len_per_gpu: int = self.config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
        # actor_ppo_max_token_len_per_gpu: int = self.config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        # critic_ppo_max_token_len_per_gpu: int = self.config.critic.ppo_max_token_len_per_gpu
        # assert log_prob_max_token_len_per_gpu >= max_model_len
        # assert critic_ppo_max_token_len_per_gpu >= max_model_len
        # assert actor_ppo_max_token_len_per_gpu >= max_model_len
        # assert ref_log_prob_max_token_len_per_gpu >= max_model_len
        assert self.config.data.max_prompt_length + self.config.data.max_response_length <= max_model_len  # ⭐ Ensure the sum of prompt and response lengths does not exceed the maximum model length


    def prepare_previous_context(self, mod='future'):
        """
        Prepare the input context for a future LLM call.

        Args:
            mod (str, optional): The mode to format the context. Defaults to 'future'.
                                 - 'future': Uses `content_for_future` for each message.
                                 - 'raw': Uses `content` for each message.

        Returns:
            list: Array of message dictionaries containing role and content, formatted for LLM input.

        Raises:
            ValueError: If an unknown mode is provided.
        """
        if mod == 'future':
            message_arr = [
                {"role": c.role, "content": c.content_for_future}  # ⭐ Format message with content_for_future
                for c in self.full_context
            ]
            return message_arr

        elif mod == 'raw':
            message_arr = [
                {"role": c.role, "content": c.content}  # ⭐ Format message with content
                for c in self.full_context
            ]
            return message_arr

        else:
            raise ValueError(f"Unknown mod {mod} in prepare_previous_context, only support 'future' and 'raw'")


    def check_context_token_num_safe(self, messages: List[dict]):
        """
        Checks if the total number of tokens in the prepared context messages is within a safe limit.

        Args:
            messages (List[dict]): A list of message dictionaries to be checked.

        Returns:
            bool: True if the total number of tokens is less than the maximum allowed sequence length, False otherwise.
        """
        # ``messages`` is the exact model-facing context returned by
        # prepare_next_llm_context.  The legacy implementation accidentally
        # replaced it with the raw, uncompressed event log, so compression could
        # never actually satisfy this safety gate.
        seq_length = len(
            chat_template_ids(
                self.tokenizer, messages, add_generation_prompt=True
            )
        )
        hard_limit = self.max_seq_length
        if self.context_policy.enabled:
            hard_limit = min(hard_limit, self.context_policy.max_prompt_tokens)
        return seq_length <= hard_limit


    def get_inc(self, text_frag_from, text_frag_to):
        """
        Get the incremental token array from text_frag_from to text_frag_to.

        Args:
            text_frag_from (str): The starting text fragment.
            text_frag_to (str): The ending text fragment.

        Returns:
            Tuple[List[int], str]: A tuple containing the list of incremental token IDs and a message with token length details.
        """
        tokenizer_output = self.tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids

        tokenizer_output = self.tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # ⭐ Get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg

    def remove_last_context(self):
        """
        Removes the last message from the full context if it is not authored by the language model.

        This method checks the author of the last message in the `full_context` list. If the author is not "llm",
        the last message is removed from the context. This is useful for managing the conversation history and
        ensuring that only relevant messages are kept in the context.

        Returns:
            None
        """
        if len(self.full_context) > 0:  # ⭐ Check if there are any messages in the context
            if self.full_context[-1].author != "llm":  # ⭐ Ensure the last message is not from the language model
                self.full_context.pop(-1)  # ⭐ Remove the last message from the context

    def remove_last_non_llm_msg(self, ext_msg_list:List[ExtendedMessage]):
        """
        Removes the last message from the list if it is not authored by the language model (llm).

        Args:
            ext_msg_list (List[ExtendedMessage]): The list of ExtendedMessage objects representing the conversation history.

        Returns:
            List[ExtendedMessage]: The updated list of ExtendedMessage objects with the last non-llm message removed if applicable.
        """
        if len(ext_msg_list) > 0:
            if ext_msg_list[-1].author != "llm":
                ext_msg_list.pop(-1)  # ⭐ Remove the last message if it is not from the llm
        return ext_msg_list



    @property
    def steps(self):
        """
        Returns the prepared previous context with the mode set to 'future'.

        Returns:
            dict: The prepared previous context.
        """
        return self.prepare_previous_context(mod='future')  # ⭐ Get the prepared context in 'future' mode

    def json(self):
        """
        Converts the prepared previous context (with the mode set to 'future') into a JSON-formatted string.

        Returns:
            str: A JSON-formatted string of the prepared previous context.
        """
        return json.dumps(self.prepare_previous_context(mod='future'), ensure_ascii=False, indent=2)  # ⭐ Convert the context to a JSON string

    def prepare_next_llm_context(self):
        """
        Prepares the context for the next LLM (Language Model) interaction.

        This function calls `prepare_previous_context` with the 'future' mode to set up the context.

        Returns:
            The result of the `prepare_previous_context` function call.
        """
        if self.context_policy.enabled:
            result = self.context_policy.build(self.full_context)
            self._pending_context_result = result
            return copy.deepcopy(result.messages)
        self._pending_context_result = None
        return self.prepare_previous_context(mod='future')  # ⭐ Prepares the context for the next LLM interaction


    def save_init_input(self, init_input_arr:list, add_nothink: bool=False):
        """
        Save and process the initial input messages to the context.

        Args:
            init_input_arr (list): Array of initial input messages to be processed
                                  Each message should be a dict with 'role' and 'content'
            add_nothink (bool, optional): If True, appends "/no_think" to the last message's content. Defaults to False.

        Note:
            - Initializes the context with the provided messages
            - Computes token arrays for each message
            - Validates that the context is empty before saving
        """
        # save basic
        assert len(self.full_context) == 0, "full_context should be empty when saving init input"
        for index, llm_msg in enumerate(init_input_arr):
            if (index == len(init_input_arr) - 1):
                if add_nothink:
                    llm_msg['content'] += "\n/no_think"
            ext_msg = ExtendedMessage(
                author="initialization",
                role=llm_msg['role'],
                content=llm_msg['content'],
                token_generator="manual",
                tokenizer=self.tokenizer,
            )
            self.full_context += [ext_msg]  # ⭐ Adds the extended message to the full context

        # Initialization messages are all loss-masked.  Store their one exact
        # full-template rendering as a single block instead of trying to infer
        # per-message increments.  Thinking templates can rewrite a former
        # final assistant turn when another message is appended, so progressive
        # prefix subtraction is not generally valid (and the old ``+=`` logic
        # duplicated every preceding prefix for 3+ initialization messages).
        try:
            initial_ids = chat_template_ids(
                self.tokenizer,
                init_input_arr,
                add_generation_prompt=False,
            )
        except Exception:
            initial_text = "".join(
                f"<|im_start|>{m['role']}\n{str(m['content']).strip()}<|im_end|>\n"
                for m in init_input_arr
            )
            initial_ids = self.tokenizer(
                initial_text, return_tensors="pt", padding=False
            )["input_ids"][0].tolist()
        for ext_msg in self.full_context:
            ext_msg.token_arr = []
        self.full_context[-1].token_arr = list(initial_ids)
        self.full_context[-1]._info = (
            f"exact initialization block token length: {len(initial_ids)}"
        )
        return

    def influence_extra_reward(self, llm_output):
        """
        Evaluates the LLM output for repetition and applies a penalty reward.
        The penalty is logged if non-zero, and the minimum penalty value is stored in the mistakes dictionary.

        Args:
            llm_output (dict): The output from the language model, expected to contain a 'content' key.

        Returns:
            None
        """
        this_msg_repetition_penalty_reward = repetition_penalty_reward_scalar(completion=llm_output['content'])  # ⭐ Calculate the repetition penalty reward
        if this_msg_repetition_penalty_reward != 0:
            print_dict({
                "reason": "repetition_penalty_reward",
                "content": llm_output['content'],
                "score": this_msg_repetition_penalty_reward,
            })
        if 'repetition_penalty_reward' not in self.llm_output_mistakes:
            self.llm_output_mistakes['repetition_penalty_reward'] = 0
        self.llm_output_mistakes['repetition_penalty_reward'] = min(this_msg_repetition_penalty_reward, self.llm_output_mistakes['repetition_penalty_reward'])  # ⭐ Update the mistakes dictionary with the minimum penalty

    @staticmethod
    def _finish_reason_value(value) -> str | None:
        """Normalize SDK enum-like values without inventing a reason."""
        if value is None:
            return None
        return str(getattr(value, "value", value))

    @classmethod
    def _completion_finish_contract(cls, raw_llm_output) -> tuple[str | None, bool]:
        """Read the completion termination contract with legacy fallbacks."""
        finish_reason = cls._finish_reason_value(
            raw_llm_output.get("finish_reason")
        )
        truncated_by_length = bool(
            raw_llm_output.get("truncated_by_length", False)
        ) or finish_reason == "length"
        return finish_reason, truncated_by_length

    def _record_decision_finish(self, raw_llm_output) -> None:
        """Attach an ordered, trajectory-level audit trail for every decision."""
        finish_reason, truncated_by_length = self._completion_finish_contract(
            raw_llm_output
        )
        finish_reasons = list(
            self.metadata.get("decision_finish_reasons") or []
        )
        length_flags = list(
            self.metadata.get("decision_truncated_by_length") or []
        )
        finish_reasons.append(finish_reason)
        length_flags.append(truncated_by_length)
        self.metadata["decision_finish_reasons"] = finish_reasons
        self.metadata["decision_truncated_by_length"] = length_flags
        self.metadata["decision_count"] = len(finish_reasons)
        self.metadata["length_truncated_decision_count"] = sum(length_flags)
        self.metadata["has_length_truncated_decision"] = any(length_flags)

    def _capture_decision_snapshot(self, input_msg_ref, raw_llm_output) -> None:
        """Freeze the exact prompt and sampled token stream for one action.

        Context compression is allowed to change what a *later* action sees. It
        is not allowed to retroactively change the prefix of this completion.
        Snapshot training therefore requires the token IDs returned by the
        rollout server; re-tokenizing repaired/normalized text would fabricate
        actions that were never sampled by the behavior policy.
        """
        if not self.context_policy.snapshot_training:
            return

        token_objects = raw_llm_output.get("tokens")
        if not token_objects:
            raise RuntimeError(
                "snapshot training requires raw rollout token IDs; the completion "
                "callback returned none"
            )
        completion_ids = [int(token.token_id) for token in token_objects]
        if not completion_ids:
            raise RuntimeError("snapshot training received an empty completion")

        prompt_messages = copy.deepcopy(input_msg_ref)
        prompt_ids = chat_template_ids(
            self.tokenizer, prompt_messages, add_generation_prompt=True
        )
        context_stats = {}
        raw_prompt_hash = self.context_policy.ids_hash(prompt_ids)
        if self._pending_context_result is not None:
            pending_ids = self._pending_context_result.prompt_token_ids
            if prompt_ids != pending_ids:
                raise RuntimeError(
                    "context changed between prepare_next_llm_context and sampling"
                )
            context_stats = dict(self._pending_context_result.stats)
            raw_prompt_hash = self._pending_context_result.raw_prompt_hash

        log_probs = []
        for token in token_objects:
            value = getattr(token, "logprob", None)
            if value is None:
                log_probs = []
                break
            log_probs.append(float(value))
        if self.context_policy.snapshot_training and len(log_probs) != len(completion_ids):
            raise RuntimeError(
                "snapshot training requires one rollout log-probability per "
                f"sampled token; got {len(log_probs)} for {len(completion_ids)}"
            )

        finish_reason, truncated_by_length = self._completion_finish_contract(
            raw_llm_output
        )
        self.decision_snapshots.append(
            DecisionSnapshot(
                step_index=len(self.decision_snapshots),
                prompt_messages=prompt_messages,
                prompt_token_ids=prompt_ids,
                completion_token_ids=completion_ids,
                completion_log_probs=log_probs,
                prompt_hash=self.context_policy.ids_hash(prompt_ids),
                raw_prompt_hash=raw_prompt_hash,
                template_hash=self.context_policy.template_hash,
                assistant_content=str(
                    raw_llm_output.get(
                        "sampled_content", raw_llm_output.get("content", "")
                    )
                ),
                finish_reason=finish_reason,
                truncated_by_length=truncated_by_length,
                stop_reason=raw_llm_output.get("stop_reason"),
                context_stats=context_stats,
            )
        )
        self._pending_context_result = None

    def save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=True):
        """
        Save the output from the LLM to the full context.

        Args:
            llm_output (dict): The output from the LLM containing 'role', 'content', and 'tokens'.
            input_msg_ref: Reference to the input messages for token increment calculation.
            auto_register_full_context (bool): Whether to register the output in the full context.

        Returns:
            ExtendedMessage: The processed and extended message object.

        Note:
            - Processes the LLM output and adds it to the conversation history.
            - Handles token processing and generation prompt management.
            - Ensures proper tokenization and context maintenance.
        """
        # save basic
        assert isinstance(llm_output, dict)
        raw_llm_output = dict(llm_output)
        llm_output = dict(llm_output)
        original_content = llm_output.get("content", "")
        llm_output['content'] = self._normalize_llm_output_content(llm_output)
        finish_reason, truncated_by_length = self._completion_finish_contract(
            raw_llm_output
        )
        if (
            'tokens' in llm_output
            and llm_output['content'] != original_content
            and not truncated_by_length
        ):
            llm_output.pop('tokens', None)
        token_generator = "manual" if 'tokens' in llm_output else "auto"
        ext_msg = ExtendedMessage(
            author="llm",
            role=llm_output['role'],
            content=llm_output['content'],
            token_generator=token_generator,
            tokenizer=self.tokenizer,
        )  # ⭐ Create an ExtendedMessage object with LLM output details
        if auto_register_full_context:
            self.full_context += [ext_msg]  # ⭐ Add the ExtendedMessage to the full context if auto_register_full_context is True
            self._capture_decision_snapshot(input_msg_ref, raw_llm_output)
            self._record_decision_finish(raw_llm_output)

        # check mistakes
        if auto_register_full_context:
            self.influence_extra_reward(llm_output)  # ⭐ Influence extra reward based on LLM output

        # generate token
        def get_token_inc_from_vllm_response(input_msg_ref) -> List[int]:
            generation_prompt_token, msg = self.get_inc(
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=False),
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False, add_generation_prompt=True),
            )
            # completion_token_arr will contain generation_prompt header
            completion_token_arr, msg2 = self.get_inc(
                # ...  <|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref, tokenize=False),
                # ...  <|im_end|><|im_start|>...<|im_end|>
                self.tokenizer.apply_chat_template(input_msg_ref + [ {"role": llm_output['role'],  "content": llm_output['content']} ], tokenize=False),
            )
            vllm_output_raw_token = [t.token_id for t in llm_output['tokens']]
            self.generated_token_cnt += len(vllm_output_raw_token)  # ⭐ Increment the generated token count
            if truncated_by_length:
                # A length completion has no sampled EOS. Preserve the exact
                # prefix + raw token stream instead of inheriting the rendered
                # chat placeholder's synthetic <|im_end|> token.
                return generation_prompt_token + vllm_output_raw_token
            if find_sublist_indices(completion_token_arr, generation_prompt_token) < 0:
                # Thinking templates re-normalize empty reasoning to '<think>\n\n</think>'
                # so the generation prompt ('...<think>\n') is not a subsequence of the
                # re-rendered message; rebuild from the raw vllm tokens instead.
                raw = list(vllm_output_raw_token)
                if raw and raw[-1] == self.tokenizer.eos_token_id:
                    raw = raw[:-1]
                eos_index = find_sublist_indices(completion_token_arr, [self.tokenizer.eos_token_id], reverse=True)
                tail = completion_token_arr[eos_index:] if eos_index >= 0 else [self.tokenizer.eos_token_id]
                return generation_prompt_token + raw + tail
            final_token_arr = replace_token_ids(place_holder=completion_token_arr, replace_with=vllm_output_raw_token, begin=generation_prompt_token, end=[self.tokenizer.eos_token_id])
            return final_token_arr

        if token_generator == "manual":
            token_arr_method2 = get_token_inc_from_vllm_response(input_msg_ref)  # ⭐ Generate token increments using the VLLM response
            ext_msg.token_arr = token_arr_method2  # ⭐ Assign the generated token array to the ExtendedMessage
        return ext_msg

    def _normalize_llm_output_content(self, llm_output):
        content = llm_output.get("content", "")
        # The generation prompt ends with '<think>\n', so the sampled content
        # carries the reasoning and its closing tag but not the opening one.
        # Store the canonical form instead: history turns are re-rendered
        # verbatim (thinkraw), and without this they would show a bare reasoning
        # block ending in '</think>' — teaching the model that a turn may open
        # with reasoning, after which it emits a second '<think>' at generation
        # time and stalls (observed as 63/64 malformed turns).
        if self.reasoning_context_budget != 0 and not content.lstrip().startswith("<think>"):
            # Applies to every sampled turn, including ones truncated before
            # '</think>': the tag is what marks this text as a sampled turn later,
            # so the training tokens can be rebuilt with the same forced prefix the
            # generation prompt used. History turns are think-stripped afterwards
            # and never carry it.
            content = "<think>\n" + content.lstrip("\n")
            llm_output["content"] = content
        if self._get_action_format() != "react_tags":
            return content
        if llm_output.get("stop_reason") != "</action>":
            return content
        # thinking mode: judge only the post-<think> part, so '<action>'/'</action>'
        # mentioned inside the reasoning block cannot trigger/suppress the repair
        post = content.split("</think>")[-1]
        if "<action>" not in post or "</action>" in post:
            return content
        had_trailing_newline = content.endswith("\n")
        content = content.rstrip()
        if had_trailing_newline:
            return f"{content}\n</action>"
        return f"{content}\n</action>"


    def save_llm_output_do_not_register_full_context(self, llm_output, input_msg_ref):
        """
        Saves the LLM output to the context without registering the full context.

        Args:
            llm_output: The output from the language model.
            input_msg_ref: Reference to the input message.

        Returns:
            The result of saving the LLM output with the specified options.
        """
        return Linear_CMT.save_llm_output(self, llm_output, input_msg_ref, auto_register_full_context=False)  # ⭐ Save LLM output without full context registration


    def save_env_output(self, env_output:dict, input_msg_ref:List[dict]=None, add_nothink=False):
        """
        Save and process environment output to the context.

        Args:
            env_output (dict): Environment output containing 'content'
            input_msg_ref (List[dict], optional): Reference messages for token calculation
            add_nothink (bool, optional): Whether to append '/no_think' to the content

        Note:
            - Clips environment output if it exceeds max_env_output_length
            - Processes the output as a user message in the conversation
            - Computes and stores token arrays for the environment response
        """
        assert isinstance(env_output, dict)
        if ('content' not in env_output) and ('error' in env_output):
            env_output['content'] = f"[Error from environment: {env_output['error']}]"
        elif ('content' not in env_output) or (not env_output['content']):
            env_output['content'] = '[No content provided by the environment]'
        if add_nothink:
            env_output['content'] += " /no_think"
        ext_msg = ExtendedMessage(
            author="env",
            role="user",
            content=env_output['content'],
            clip=True,
            clip_token_limit=self.max_env_output_length,
            token_generator="auto",
            tokenizer=self.tokenizer,
        )  # ⭐ Create an ExtendedMessage object with the environment content
        self.full_context += [ext_msg]  # ⭐ Add the ExtendedMessage to the full context
        if not self.context_policy.enabled:
            # Legacy transcript training mutates the future view in place.  The
            # snapshot path instead derives every prompt from immutable raw
            # messages, so retroactive mutation is both unnecessary and unsafe.
            self._compress_old_context()
            self._strip_history_think()
        return

    # ------------------------------------------------------------------
    # Sliding-window context compression (EPO / agl-envs style)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_action_line(content: str) -> str:
        """Extract the action from an LLM output that uses
        ``Thought:\\n...\\nAction:\\n<action>`` or ``Action: <action>`` format.
        """
        lines = content.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.lower().startswith("action:"):
                continue
            after_colon = stripped[len("action:"):].strip()
            if after_colon:
                return f"Action: {after_colon}"
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line:
                    return f"Action: {next_line}"
            return "Action: (empty)"
        return content.split("\n")[-1].strip() if content.strip() else "(no action)"

    @staticmethod
    def _extract_action_tag(content: str) -> str | None:
        """Extract an action tag from a react_tags LLM output.

        Only accepts a well-formed ``<action>...</action>`` block or an
        open ``<action>...`` block truncated by the stop sequence.
        """
        if not isinstance(content, str):
            return None
        match = re.search(r"<action>(.*?)</action>", content, flags=re.IGNORECASE | re.DOTALL)
        if match:
            action = match.group(1).strip()
            if action:
                return f"<action>\n{action}\n</action>"
        open_tag_match = re.search(r"<action>(.*)$", content, flags=re.IGNORECASE | re.DOTALL)
        if open_tag_match:
            action = open_tag_match.group(1).strip()
            if action:
                return f"<action>\n{action}\n</action>"
        return None

    def _get_action_format(self) -> str:
        env_params = getattr(self.config.env_service, "env_params", None)
        if env_params is None:
            return "react"
        return str(getattr(env_params, "action_format", "react") or "react").lower()

    def _compress_llm_message(self, content: str) -> str:
        # thinking mode: extract the action from the post-<think> part only,
        # so tags mentioned inside the reasoning block are never picked up
        # (a no-op when the model emits no <think> at all)
        content = content.split("</think>")[-1]
        if self._get_action_format() == "react_tags":
            extracted = self._extract_action_tag(content)
            return extracted if extracted is not None else content
        return self._extract_action_line(content)

    @staticmethod
    def _extract_last_webshop_action(content: str) -> str | None:
        """Extract the last well-formed WebShop action from free-form output."""
        if not isinstance(content, str):
            return None
        text = content.strip()
        if not text:
            return None

        matches = list(
            re.finditer(r"(search|click)\s*\[([^\]\n]+)\]", text, flags=re.IGNORECASE)
        )
        if not matches:
            return None

        last_match = matches[-1]
        action_type = last_match.group(1).lower()
        action_arg = last_match.group(2).strip()
        if not action_arg:
            return None
        return f"{action_type}[{action_arg}]"

    @staticmethod
    def _strip_action_hints(content: str) -> str:
        """Remove action hint blocks from an env observation, keeping only
        the actual observation text.  Recognises markers produced by
        ``sciworld_env._get_action_hints`` as well as WebShop action lists
        (``You can use:``, ``Clickable elements:``).
        """
        lines = content.split("\n")
        kept: list[str] = []
        hint_prefixes = (
            "available actions:",
            "obj candidates:",
            "obj must be replaced",
            "you can use:",
            "clickable elements:",
        )
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(hint_prefixes):
                break
            kept.append(line)
        result = "\n".join(kept).strip()
        return result if result else content

    def _retoken(self, msg: 'ExtendedMessage', new_content: str, mark_compressed: bool = True) -> None:
        """Update *msg* in-place: set ``_content_for_future`` and regenerate
        ``token_arr`` so that rollout and training stay consistent."""
        msg._content_for_future = new_content
        if mark_compressed:
            msg._sw_compressed = True
        msg.token_arr = auto_tokenize_message(self.tokenizer, msg.role, new_content)

    # ------------------------------------------------------------------
    # Thinking-mode history <think> stripping (Qwen3.5 hybrid thinking)
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_think(content: str) -> str:
        """Return the post-``</think>`` part of *content*, byte-identical to the
        Qwen3.5 chat template's history branch
        (``content.split('</think>')[-1].lstrip('\\n')`` after ``|trim``)."""
        if "</think>" not in content:
            return content
        return content.split("</think>")[-1].lstrip("\n").strip()

    @staticmethod
    def _think_span(content: str) -> str:
        """The ``<think>…</think>`` part of *content* ('' when absent)."""
        if "</think>" not in content:
            return ""
        return content.split("</think>")[0]

    def _strip_history_think(self) -> int:
        """Apply the reasoning-context budget to history assistant turns.

        Walking back from the most recent history turn, each turn's own
        ``<think>`` block is kept while the running total of reasoning tokens
        stays within ``reasoning_context_budget``; everything older is stripped.
        The current (last) assistant turn always keeps its think: its action
        tokens were sampled conditioned on it, so training must score them under
        the same conditional.

        Budget semantics: ``0`` strips all history think, ``-1`` keeps all,
        ``N>0`` keeps the most recent N reasoning tokens. The same rule renders
        the rollout prompt (served with the ``-thinkraw`` template, which emits
        history turns verbatim) and the training sequence, and is applied to
        teacher trajectories through the same call site, so student and teacher
        stay format-symmetric.

        Stripping is deferred, not one-shot: a turn inside the window today
        falls out of it as newer reasoning accumulates, so messages are
        re-examined until their think is actually gone. Sliding-window
        compressed messages (``_sw_compressed``) are already action-only.

        Returns:
            int: Number of messages whose content actually changed.
        """
        budget = self.reasoning_context_budget
        if budget < 0:  # keep-all
            return 0
        n_stripped = 0
        assistant_indices = [
            i for i, c in enumerate(self.full_context) if c.role == "assistant"
        ]
        used = 0
        # newest history turn first; the final assistant turn is never touched
        for idx in reversed(assistant_indices[:-1]):
            msg = self.full_context[idx]
            if getattr(msg, "_sw_compressed", False):
                continue
            content = msg.content_for_future
            think = self._think_span(content)
            if not think:
                continue  # already stripped in an earlier pass
            if budget > 0 and used + len(self.tokenizer.encode(think, add_special_tokens=False)) <= budget:
                used += len(self.tokenizer.encode(think, add_special_tokens=False))
                continue  # inside the window: keep this turn's reasoning
            stripped = self._strip_think(content)
            if stripped != content:
                self._retoken(msg, stripped, mark_compressed=False)
                n_stripped += 1
        return n_stripped

    def _compress_old_context(self):
        """Compress old turns outside the sliding window.

        For each interaction turn (LLM reply + env observation) older than
        the most recent ``sliding_window_size`` turns:

        * **LLM (assistant) messages** – replaced by the ``Action: …`` line
          only (Thought is dropped).
        * **Env (user) messages** – ``Available actions`` and ``OBJ
          candidates`` hints are stripped, keeping only the observation text.

        Both ``_content_for_future`` (rollout) and ``token_arr`` (training)
        are updated so that the two paths stay consistent.

        Set ``sliding_window_size`` to -1 to disable (default).
        """
        if self.sliding_window_size < 0:
            return

        llm_indices = [
            i for i, c in enumerate(self.full_context) if c.author == "llm"
        ]
        if len(llm_indices) <= self.sliding_window_size:
            return

        cutoff = (
            llm_indices[-self.sliding_window_size]
            if self.sliding_window_size > 0
            else len(self.full_context)
        )

        for idx in range(cutoff):
            msg = self.full_context[idx]
            if getattr(msg, "_sw_compressed", False):
                continue
            if msg.author == "initialization":
                continue

            if msg.author == "llm":
                self._retoken(msg, self._compress_llm_message(msg.content))
            elif msg.author == "env":
                stripped = self._strip_action_hints(msg.content)
                stripped = self._truncate_history_observation(stripped)
                if len(stripped) < len(msg._content_for_future):
                    self._retoken(msg, stripped)

    def _truncate_history_observation(self, content: str) -> str:
        """Cap an out-of-window observation at ``history_obs_max_chars``.

        Environments differ in where their tokens sit: ALFWorld repeats an action
        list every turn (stripping the hint block already removes ~62%), while
        WebShop's cost is the product-listing body itself, which hint stripping
        barely touches (measured 1.6% vs 43.1% with truncation). Disabled by
        default (-1); set per environment.

        Only the model-facing copy shrinks — the State Channel keys its progress
        map off `ExtendedMessage.content`, which `_retoken` leaves untouched.
        """
        limit = getattr(self.config.actor_rollout_ref.rollout, "history_obs_max_chars", -1)
        if limit is None or limit < 0 or len(content) <= limit:
            return content
        return content[:limit].rstrip() + " …"

    def to_role_content(self, ext_msg_array: List[ExtendedMessage]) -> List[dict]:
        """
        Converts a list of ExtendedMessage objects into a list of dictionaries with 'role' and 'content' keys.

        Args:
            ext_msg_array (List[ExtendedMessage]): A list of ExtendedMessage objects.

        Returns:
            List[dict]: A list of dictionaries, each containing 'role' and 'content' keys.
        """
        return [{"role": ext_msg.role, "content": ext_msg.content_for_future} for ext_msg in ext_msg_array]  # ⭐ Convert each ExtendedMessage to a dictionary

    def to_role_content_raw(self, ext_msg_array: List[ExtendedMessage]) -> List[dict]:
        """Same turns, but with the original (pre-compression) content.

        `_retoken` only rewrites `_content_for_future` and `token_arr`; `content`
        still holds what the environment actually returned. The State Channel
        hashes those strings, so it reads this view and stays correct under any
        context-compression scheme.
        """
        return [{"role": m.role, "content": m.content} for m in ext_msg_array]

    def prepare_world_interaction(self) -> str:
        """
        Process the latest model content before environment interaction.

        Returns:
            str: Processed content, with code extracted from markdown code blocks if present
                 or the raw content if no code blocks are found

        Note:
            - Extracts Python code from markdown code blocks (```python```)
            - Returns the raw content if no valid code blocks are found
        """
        latest_content = self.full_context[-1].content
        if "</think>" in latest_content:
            # thinking mode: the environment only sees the post-<think> part;
            # a response with '</think>' but no valid action in the remainder
            # falls through to the environment's existing format-error path
            latest_content = latest_content.split("</think>")[-1].lstrip("\n")
        env_type = str(getattr(self.config.env_service, "env_type", "") or "").lower()
        env_params = getattr(self.config.env_service, "env_params", None)
        enable_action_sanitizer = True
        if env_params is not None:
            enable_action_sanitizer = bool(getattr(env_params, "enable_action_sanitizer", True))

        if env_type == "webshop" and self._get_action_format() == "react_tags":
            return latest_content

        if env_type == "webshop" and enable_action_sanitizer:
            sanitized_action = self._extract_last_webshop_action(latest_content)
            if sanitized_action is not None:
                return sanitized_action
        return latest_content

    def filter_context_via_author(self, author: str) -> List[ExtendedMessage]:
        """
        Filters the full context to include only messages from a specific author and returns a deep copy of the filtered list.

        Args:
            author (str): The name of the author whose messages are to be included in the result.

        Returns:
            List[ExtendedMessage]: A deep copy of the list containing only the messages from the specified author.
        """
        return copy.deepcopy([ c for c in self.full_context if c.author == author ])  # ⭐ Filter and create a deep copy of the context

    def filter_context_via_authors(self, authors: str) -> List[ExtendedMessage]:
        """
        Filters the full context of messages, returning only those authored by the specified authors.

        Args:
            authors (str): A string of author names, separated by commas, indicating which authors' messages to include.

        Returns:
            List[ExtendedMessage]: A list of ExtendedMessage objects from the full context that match the specified authors.
        """
        return copy.deepcopy([ c for c in self.full_context if c.author in authors ])  # ⭐ Filter and deep copy the relevant messages

    def _select_decision_snapshot(self) -> DecisionSnapshot:
        if not self.decision_snapshots:
            raise RuntimeError(
                "snapshot training is enabled but the trajectory contains no "
                "captured decisions"
            )
        mode = self.context_policy.snapshot_selection
        if mode == "first":
            return self.decision_snapshots[0]
        if mode == "last":
            return self.decision_snapshots[-1]

        # One token-weighted turn per trajectory is a compute-light unbiased
        # estimator of the trajectory token-mean objective. Use a content hash
        # rather than process-global RNG so retry/order changes are reproducible.
        total = sum(len(s.completion_token_ids) for s in self.decision_snapshots)
        if total <= 0:
            raise RuntimeError("all decision snapshots have empty completions")
        digest = hashlib.sha256()
        digest.update(str(self.context_policy.snapshot_selection_seed).encode("utf-8"))
        digest.update(str(self.task_id).encode("utf-8"))
        digest.update(str(self.data_id).encode("utf-8"))
        digest.update(str(self.rollout_id).encode("utf-8"))
        target = int.from_bytes(digest.digest()[:8], "big") % total
        cursor = 0
        for snapshot in self.decision_snapshots:
            cursor += len(snapshot.completion_token_ids)
            if target < cursor:
                return snapshot
        return self.decision_snapshots[-1]

    def _group_tokenize_decision_snapshot(self) -> List[Sample]:
        """Build one actor sample from an immutable per-action snapshot."""
        snapshot = self._select_decision_snapshot()
        prompt_ids = list(snapshot.prompt_token_ids)
        response_ids = list(snapshot.completion_token_ids)

        max_prompt = int(self.config.data.max_prompt_length)
        max_response = int(self.config.data.max_response_length)
        max_model = max_prompt + max_response
        if len(prompt_ids) > max_prompt:
            raise RuntimeError(
                "decision snapshot prompt exceeds the actor limit: "
                f"{len(prompt_ids)} > {max_prompt}; compress before rollout"
            )
        if len(response_ids) > max_response:
            raise RuntimeError(
                "decision snapshot completion exceeds the actor limit: "
                f"{len(response_ids)} > {max_response}; never silently truncate "
                "a sampled completion"
            )
        if len(prompt_ids) + len(response_ids) > max_model:
            raise RuntimeError(
                "decision snapshot exceeds max model length despite individual "
                "limits"
            )

        input_ids = prompt_ids + response_ids
        attention_mask = [1] * len(input_ids)
        from verl.utils.model import compute_position_id_with_mask

        position_ids = compute_position_id_with_mask(
            torch.tensor(attention_mask)
        ).tolist()
        prompt_position_ids = position_ids[: len(prompt_ids)]
        response_position_ids = position_ids[len(prompt_ids) :]
        prompt_loss_mask = [0] * len(prompt_ids)
        response_loss_mask = [1] * len(response_ids)

        self.metadata["selected_decision_step"] = snapshot.step_index
        self.metadata["context_stats"] = dict(snapshot.context_stats)
        self.metadata["rollout_log_probs"] = list(snapshot.completion_log_probs)
        self.metadata["snapshot_training"] = True
        self.metadata["prompt_hash"] = snapshot.prompt_hash
        self.metadata["raw_prompt_hash"] = snapshot.raw_prompt_hash
        self.metadata["template_hash"] = snapshot.template_hash
        self.metadata["selected_finish_reason"] = snapshot.finish_reason
        self.metadata["selected_truncated_by_length"] = bool(
            snapshot.truncated_by_length
        )

        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=snapshot.step_index,
            messages=copy.deepcopy(snapshot.prompt_messages)
            + [
                {
                    "role": "assistant",
                    "content": snapshot.assistant_content,
                }
            ],
            # Reward shaping/state matching must continue to see the complete,
            # uncompressed environment trajectory.
            messages_raw=self.to_role_content_raw(self.full_context),
            input_ids=input_ids,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=[1] * len(prompt_ids),
            response_attention_mask=[1] * len(response_ids),
            loss_mask=prompt_loss_mask + response_loss_mask,
            prompt_loss_mask=prompt_loss_mask,
            response_loss_mask=response_loss_mask,
            position_ids=position_ids,
            prompt_position_ids=prompt_position_ids,
            response_position_ids=response_position_ids,
            reward_scores=self.reward.model_dump(),
            max_prompt_len=max_prompt,
            max_response_len=max_response,
            max_model_len=max_model,
        )
        return [sample]

    def group_tokenize(self):
        """
        Tokenizes the full context into a format suitable for input to a language model, creating a sample with necessary attributes like input IDs, attention masks, and position IDs.

        Returns:
            List[Sample]: An array containing a single Sample object, representing the tokenized context.
        """
        # A post-hoc compressed transcript cannot represent the different
        # contexts used at each sampling step. Snapshot mode trains exactly one
        # sampled span under its immutable behavior-policy prefix.
        if self.context_policy.snapshot_training:
            return self._group_tokenize_decision_snapshot()

        # assert self.latest_llm_interaction_socket is None, "unprocessed message buffer! forget to call `save_llm_output` after `prepare_next_llm_context`?"
        sample_arr = []
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        sample = Sample(
            data_id=self.data_id,
            rollout_id=self.rollout_id,
            task_id=self.task_id,
            minor_index_id=0,
            messages=self.to_role_content(ext_steps),
            messages_raw=self.to_role_content_raw(ext_steps),
            input_ids=cmt_tokenized["input_ids"],
            prompt_ids=cmt_tokenized["prompt_ids"],
            response_ids=cmt_tokenized["response_ids"],
            attention_mask=cmt_tokenized["attention_mask"],
            prompt_attention_mask=cmt_tokenized["prompt_attention_mask"],
            response_attention_mask=cmt_tokenized["response_attention_mask"],
            loss_mask=cmt_tokenized["loss_mask"],
            prompt_loss_mask=cmt_tokenized["prompt_loss_mask"],
            response_loss_mask=cmt_tokenized["response_loss_mask"],
            position_ids=cmt_tokenized["position_ids"],
            prompt_position_ids=cmt_tokenized["prompt_position_ids"],
            response_position_ids=cmt_tokenized["response_position_ids"],
            reward_scores=self.reward.model_dump(), # reward is duplicated in each sample
            max_prompt_len=self.config.data.max_prompt_length,
            max_response_len=self.config.data.max_response_length,
            max_model_len=self.config.data.max_response_length + self.config.data.max_prompt_length,
        )
        sample.truncate_output_ids()  # ⭐ Ensure the output IDs are within the allowed length
        sample_arr += [sample]
        return sample_arr


    def group_render_token_log(self):
        ext_steps=self.full_context
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]
        return {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }


    def generate_log(self, task_id):
        """
        Generates and prints a log for the specified task, including detailed information about the tokenized steps,
        input IDs, loss mask colors, and rewards. The log is formatted into a nested JSON structure and printed.

        Args:
            task_id (str): The ID of the task for which the log is being generated.

        Returns:
            None
        """
        nested_items_print_buffer = {}
        ext_steps=self.full_context  # ⭐ Retrieve the full context for the task
        cmt_tokenized = self.tokenize_steps(ext_steps=ext_steps)  # ⭐ Tokenize the extended steps
        text_arr = [self.tokenizer.decode(t) for t in cmt_tokenized["input_ids"]]  # ⭐ Decode the tokenized input IDs to text
        input_id_arr = [str(t) for t in cmt_tokenized["input_ids"]]  # ⭐ Convert input IDs to strings
        loss_mask_color_arr = ["#09ABCF" if mask==1 else "#D98510" for mask in cmt_tokenized["loss_mask"]]  # ⭐ Generate color array based on loss mask
        buffer = {
            "text_arr": text_arr,
            "input_id_arr": input_id_arr,
            "loss_mask_color_arr": loss_mask_color_arr,
        }
        len_prompt_ids = len(cmt_tokenized["prompt_ids"])  # ⭐ Calculate the length of prompt IDs
        len_response_ids = len(cmt_tokenized["response_ids"])  # ⭐ Calculate the length of response IDs
        len_input_ids = len(cmt_tokenized["input_ids"])  # ⭐ Calculate the length of input IDs
        reward = self.reward.outcome  # ⭐ Get the reward outcome
        task_outcome = str(self.reward.success_rate)  # ⭐ Get the task success rate as a string
        final_reward = self.reward_patch(self.reward).outcome  # ⭐ Get the final reward after applying the reward patch
        selectors = [task_id, task_outcome]
        nested_items_print_buffer[f".".join(selectors)] = NestedJsonItem(
            item_id=f"item",
            outcome=task_outcome,
            len_prompt_ids=len_prompt_ids,
            len_response_ids=len_response_ids,
            len_input_ids=len_input_ids,
            reward=f"{float(reward):.3f}",
            final_reward=final_reward,
            content=SeqItem(
                text = buffer['text_arr'],  # text
                title = buffer['text_arr'], # mouse hover
                count = buffer['input_id_arr'], # highlight text
                color = buffer['loss_mask_color_arr']   # color
            )
        )
        print_nested(nested_items_print_buffer,  # ⭐ Print the nested JSON buffer
            main_content="This is the main content of the nested JSON",
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="rollout",
            narrow=False,
            attach="Copy Sample Message"
        )
        print_listofdict(  # ⭐ Print the list of dictionaries
            self.steps,
            header=f"Training task {task_id} (Final Reward {final_reward})",
            mod="conversation",
            narrow=False,
        )

    def reward_patch(self, reward):
        """
        Creates a deep copy of the provided reward and may modify it based on internal state.

        Args:
            reward (object): The reward object to be patched, which must have an 'outcome' attribute.

        Returns:
            object: The possibly modified deep copy of the original reward.
        """
        _reward = copy.deepcopy(reward)  # ⭐ Create a deep copy of the reward to avoid modifying the original
        # if self.compute_madness() < 0: _reward.outcome = -1.0
        return _reward


    def compute_madness(self) -> float:
        """
        Evaluates the 'madness' of the model's output based on the proportion of certain types of mistakes (e.g., special tokens, repeated characters, non-ASCII characters).

        Returns:
            float: -1.0 if any mistake proportion is below the threshold, otherwise 0.0.
        """
        threshold = -0.01
        for k, v in self.llm_output_mistakes.items():
            if v < threshold: return -1.0  # ⭐ Check if any mistake proportion is below the threshold
        return 0.0


    def tokenize_steps(self, ext_steps: List[ExtendedMessage], debug=False) -> dict:
        """
        Tokenizes the given extended messages, processes them to separate prompts and responses, and prepares the data for model training.
        It also handles the extraction and discarding of experience information if needed.

        Args:
            ext_steps (List[ExtendedMessage]): A list of ExtendedMessage objects representing the conversation context.
            debug (bool, optional): A flag to enable debugging. Defaults to False.

        Returns:
            dict: A dictionary containing tokenized and processed data for model training.
        """
        from verl.utils.model import compute_position_id_with_mask
        ext_steps = self.remove_last_non_llm_msg(copy.deepcopy(ext_steps))  # ⭐ Remove the last non-LLM message

        # ⭐ 检查是否为 experience replay 数据
        is_experience_replay = self.metadata.get("is_experience_replay", False)

        exp_worker = ExperienceWorker(self.config)
        for i, ext_msg in enumerate(ext_steps):
            experience, new_content = exp_worker.manage_training_context(ext_msg.content_for_future, self.metadata)
            if experience:
                ext_steps[i] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=new_content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    uuid=ext_msg.uuid,
                )

        # mapping
        input_ids = []
        attention_mask = []
        loss_mask = []
        split_prompt_reponse_index = -1
        for ext_msg in ext_steps:
            # find split index, this have to be done before input_ids += ext_msg.token_arr
            # ⭐ Experience Replay: 对于 off-policy 数据，所有 LLM 消息都应该参与 loss 计算
            # 使用 exp_mask 区分 on-policy 和 off-policy，而不是让 loss_mask=0
            if (split_prompt_reponse_index == -1) and (ext_msg.need_training or (is_experience_replay and ext_msg.role == "assistant")):
                split_prompt_reponse_index = len(input_ids)
                # 对于 experience replay，允许 author 为 "llm(do_not_train)"
                if not is_experience_replay:
                    assert ext_msg.author == 'llm', "The first message after initialization should be from LLM, not from env or user"
            input_ids += ext_msg.token_arr
            attention_mask += [1] * len(ext_msg.token_arr)
            # ⭐ Experience Replay: 对于 off-policy 数据，LLM 消息的 loss_mask 应该为 1（参与 loss 计算）
            # 使用 exp_mask 来区分 on/off-policy，而不是用 loss_mask=0
            if is_experience_replay and ext_msg.role == "assistant":
                # Off-policy LLM 消息：参与 loss，但与 on-policy 用同一套 blackout
                # 规则（否则 teacher 序列会多留模板 token，污染 DR3 的 log-prob 特征）
                msg_loss_mask = ext_msg.get_loss_mask(
                    blackout_token_combo=self.blackout_token_combo, force_training=True
                )
            else:
                msg_loss_mask = ext_msg.get_loss_mask(blackout_token_combo=self.blackout_token_combo)
            loss_mask += msg_loss_mask

        # ⭐ 如果是 experience replay 数据且没有找到 split_prompt_reponse_index
        # 说明没有 LLM 消息（异常情况），设置为第一个位置
        if is_experience_replay and split_prompt_reponse_index == -1:
            split_prompt_reponse_index = 0  # 所有内容都是 response（用于计算 loss）

        assert split_prompt_reponse_index != -1, "split_prompt_reponse_index should not be -1, at least one message should be in the context"
        position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()  # ⭐ Compute position IDs with mask

        # separate prompt and response
        prompt_ids =            input_ids[:split_prompt_reponse_index]
        prompt_attention_mask = attention_mask[:split_prompt_reponse_index]
        prompt_position_ids =   position_ids[:split_prompt_reponse_index]
        prompt_loss_mask =      loss_mask[:split_prompt_reponse_index]

        response_ids =              input_ids[split_prompt_reponse_index:]
        response_attention_mask =   attention_mask[split_prompt_reponse_index:]
        response_position_ids =     position_ids[split_prompt_reponse_index:]
        response_loss_mask =        loss_mask[split_prompt_reponse_index:]

        cmt_tokenized = {}
        cmt_tokenized["input_ids"] = input_ids
        cmt_tokenized["prompt_ids"] = prompt_ids
        cmt_tokenized["response_ids"] = response_ids
        cmt_tokenized["attention_mask"] = attention_mask
        cmt_tokenized["prompt_attention_mask"] = prompt_attention_mask
        cmt_tokenized["response_attention_mask"] = response_attention_mask
        cmt_tokenized["loss_mask"] = loss_mask
        cmt_tokenized["prompt_loss_mask"] = prompt_loss_mask
        cmt_tokenized["response_loss_mask"] = response_loss_mask
        cmt_tokenized["position_ids"] = position_ids
        cmt_tokenized["prompt_position_ids"] = prompt_position_ids
        cmt_tokenized["response_position_ids"] = response_position_ids

        return cmt_tokenized
