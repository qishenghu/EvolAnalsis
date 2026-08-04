from agentevolver.schema.trajectory import Reward, Trajectory
from typing import List, Dict
import uuid as uuid_gen
from loguru import logger


def chat_template_ids(tokenizer, messages, **kwargs) -> List[int]:
    """apply_chat_template(tokenize=True) as a plain list of ids.

    transformers 5.x returns a BatchEncoding here where 4.x returned a list, so
    every length comparison and slice downstream needs the ids unwrapped.
    """
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if hasattr(out, "input_ids") or isinstance(out, dict):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):  # batched shape
        out = out[0]
    return list(out)


THINKING_MODE_NONE = "none"
THINKING_MODE_PROMPT_GUIDED = "prompt_guided"     # Qwen 2.5: prompt-based <think> hints
THINKING_MODE_NATIVE_QWEN3 = "native_qwen3"       # Qwen 3: native thinking with /no_think control
THINKING_MODE_NATIVE_QWEN35 = "native_qwen35"     # Qwen 3.5: native thinking, no /think soft switches

VALID_THINKING_MODES = {THINKING_MODE_NONE, THINKING_MODE_PROMPT_GUIDED,
                        THINKING_MODE_NATIVE_QWEN3, THINKING_MODE_NATIVE_QWEN35}


def resolve_thinking_mode(config) -> str:
    """
    Resolve the thinking mode from config, supporting both the new `thinking_mode` field
    and the legacy `use_qwen3` / `force_think` fields for backward compatibility.

    Priority:
        1. config.actor_rollout_ref.rollout.thinking_mode (new unified field)
        2. Legacy fallback: use_qwen3=true -> "native_qwen3", force_think=true -> "prompt_guided", else -> "none"

    Returns:
        One of: "none", "prompt_guided", "native_qwen3"
    """
    rollout_cfg = config.actor_rollout_ref.rollout

    # New unified config takes priority
    thinking_mode = getattr(rollout_cfg, "thinking_mode", None)
    if thinking_mode is not None:
        if thinking_mode not in VALID_THINKING_MODES:
            raise ValueError(
                f"Invalid thinking_mode='{thinking_mode}'. "
                f"Must be one of: {sorted(VALID_THINKING_MODES)}"
            )
        return thinking_mode

    # Legacy backward compatibility
    use_qwen3 = getattr(rollout_cfg, "use_qwen3", False)
    force_think = getattr(rollout_cfg, "force_think", False)

    if use_qwen3:
        return THINKING_MODE_NATIVE_QWEN3
    elif force_think:
        return THINKING_MODE_PROMPT_GUIDED
    else:
        return THINKING_MODE_NONE


def _is_thinking_chatml_template(tokenizer) -> bool:
    """
    True for chatml-style templates with native <think> handling (Qwen3 / Qwen3.5).
    These templates raise TemplateError on assistant-only or system-only message
    lists and re-render assistant turns with an injected <think> block, so
    incremental tokenization must avoid dummy-prefix round-trips for them.
    """
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    return ("<|im_start|>" in chat_template) and ("</think>" in chat_template)


def _is_sampled_turn(text_content: str) -> bool:
    """True when this assistant text still carries its reasoning block.

    History turns have their think stripped by the reasoning-context budget and
    must render verbatim; sampled turns (closed or not) must be rebuilt with the
    forced '<think>' prefix.
    """
    return ("</think>" in text_content) or text_content.startswith("<think>")


def auto_tokenize_message(tokenizer, role: str, content: str) -> List[int]:
    """
    Tokenize a single history message incrementally (chat-template formatting
    included), without access to the full conversation prefix.

    - Thinking chatml templates (Qwen3.5): assistant/system messages are rendered
      literally as '<|im_start|>{role}\\n{content}<|im_end|>\\n', byte-identical to
      the template's history branch. A dummy-prefix round-trip would either raise
      TemplateError (assistant-only prefix) or inject an empty <think> block
      (user-prefix dummy renders the appended assistant as a final turn).
    - Other templates: incremental tokenization against a [user] dummy prefix
      (user->assistant and user->user appends are prefix-consistent; renders the
      same increments as the legacy assistant-only dummy on Qwen2.5/Llama).
    """
    if _is_thinking_chatml_template(tokenizer) and role in ("assistant", "system"):
        # mirror the template's `render_content(...)|trim` on the literal path
        text_content = str(content).strip()
        if role == "assistant" and _is_sampled_turn(text_content):
            # The generation prompt forces '<|im_start|>assistant\n<think>\n', so a
            # sampled turn's training tokens MUST start with that exact prefix or we
            # score it under a context the policy never saw. Turns that fail to close
            # </think> used to fall through to the literal branch, silently dropping
            # the two forced tokens: the loss mask then trained the first content
            # token, teaching the model to emit its own '<think>' after the prompt
            # already opened one — which makes closing even less likely. Measured on
            # alfworld_qwen35_4b_grpo_v3: prefix mismatch 27% -> 99.5% while reward
            # went 0.50 -> 0.00, crossing 50% exactly at the collapse step.
            if "</think>" in text_content:
                reasoning = (
                    text_content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
                )
                post = text_content.split("</think>")[-1].lstrip("\n")
                text = f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{post}<|im_end|>\n"
            else:
                # unclosed think: keep the body verbatim, only restore the prefix
                body = text_content.split("<think>")[-1].lstrip("\n")
                text = f"<|im_start|>assistant\n<think>\n{body}<|im_end|>\n"
        else:
            text = f"<|im_start|>{role}\n{text_content}<|im_end|>\n"
        return tokenizer(text, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    dummy_msg = [{"role": "user", "content": "dummy text"}]
    text_from = tokenizer.apply_chat_template(dummy_msg, tokenize=False)
    text_to = tokenizer.apply_chat_template(
        dummy_msg + [{"role": role, "content": content}], tokenize=False
    )
    tokens_from = tokenizer(text_from, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    tokens_to = tokenizer(text_to, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    return tokens_to[len(tokens_from):]


def extract_assistant_header_tokens(tokenizer) -> List[int]:
    """
    Dynamically extract the assistant role header tokens from the tokenizer's chat template.
    Works for both Qwen 2.5 (<|im_start|>assistant\\n) and Qwen 3 (same format).
    This replaces the hardcoded tokenizer.encode("<|im_start|>assistant\\n").

    The generation prompt is preferred: it is the exact forced (non-generated)
    prefix of every LLM turn's token_arr (see `save_llm_output`), which is what
    the loss-mask blackout must match. For thinking templates (Qwen3/Qwen3.5)
    the re-rendered history header differs from the generation prompt, so the
    render-based extraction below is only a fallback.
    """
    try:
        no_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=False
        )
        with_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=True
        )
        if len(with_gen) > len(no_gen) and with_gen[:len(no_gen)] == no_gen:
            return with_gen[len(no_gen):]

        user_only = [{"role": "user", "content": "test"}]
        user_tokens = chat_template_ids(tokenizer, user_only, add_generation_prompt=False
        )

        with_assistant = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "x"}]
        full_tokens = chat_template_ids(tokenizer, with_assistant, add_generation_prompt=False
        )

        # Find 'x' token position to isolate the assistant header
        x_tokens = tokenizer.encode("x", add_special_tokens=False)
        user_len = len(user_tokens)

        # The assistant header is between the end of user-only tokens and the 'x' content
        for i in range(user_len, len(full_tokens) - len(x_tokens) + 1):
            if full_tokens[i:i + len(x_tokens)] == x_tokens:
                header = full_tokens[user_len:i]
                if header:
                    return header
                break

        # Fallback: use apply_chat_template with generation prompt to extract header
        logger.warning("Dynamic assistant header extraction found empty header, using generation prompt fallback")
        no_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=False
        )
        with_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=True
        )
        if len(with_gen) > len(no_gen):
            return with_gen[len(no_gen):]
        raise ValueError("Cannot extract assistant header tokens from tokenizer")
    except Exception as e:
        logger.warning(f"Failed to dynamically extract assistant header tokens: {e}, using generation prompt fallback")
        no_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=False
        )
        with_gen = chat_template_ids(tokenizer, [{"role": "user", "content": "test"}], add_generation_prompt=True
        )
        if len(with_gen) > len(no_gen):
            return with_gen[len(no_gen):]
        raise ValueError(f"Cannot extract assistant header tokens from tokenizer: {e}")


class ContextManagerBase:
    """
    Base class for context management, providing a framework for handling and processing messages, including tokenization, content clipping, and loss masking.

    Methods:
        - save_init_input: Saves the initial input.
        - prepare_next_llm_context: Prepares the next context for the LLM.
        - check_context_token_num_safe: Checks if the number of tokens in the context is safe.
        - prepare_world_interaction: Prepares the interaction with the world.
        - save_llm_output: Saves the output from the LLM.
        - save_env_output: Saves the output from the environment.
        - remove_last_context: Removes the last context.
        - generate_log: Generates a log.
        - group_tokenize: Tokenizes a group of messages.
    """

    def save_init_input(self, init_input_arr: List):
        """
        Saves the initial input array. This method should be implemented by subclasses.

        Args:
            init_input_arr (List): The initial input array to be saved.

        Raises:
            NotImplementedError: This method needs to be implemented by a subclass.
        """
        raise NotImplementedError

    def prepare_next_llm_context(self, **kwargs) -> List:
        """
        Prepares the next context for the LLM. This method should be implemented by subclasses.

        Args:
            **kwargs: Arbitrary keyword arguments that may be required for context preparation.

        Returns:
            List: The prepared context for the LLM.

        Raises:
            NotImplementedError: This method needs to be implemented by a subclass.
        """
        raise NotImplementedError

    def prepare_world_interaction(self, **kwargs) -> str:
        """
        Prepares the world interaction. This method should be overridden by subclasses to provide specific implementation.

        Args:
            **kwargs: Arbitrary keyword arguments that can be used by subclasses to customize the preparation of the world interaction.

        Returns:
            str: A string indicating the result or status of the preparation.
        """
        raise NotImplementedError

    def save_llm_output(self, llm_output, **kwargs):
        """
        Saves the output from a language model. This method must be implemented by subclasses.

        Args:
            llm_output: The output generated by the language model.
            **kwargs: Additional keyword arguments that may be used by subclasses.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def save_env_output(self, env_output, **kwargs):
        """
        Saves the output of the environment. This method should be implemented by subclasses.

        Args:
            env_output: The output from the environment that needs to be saved.
            **kwargs: Additional keyword arguments that may be required for saving the output.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError

    def group_tokenize(self):
        """
        Placeholder method for tokenizing a group of trajectories. This method should be implemented in subclasses.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError



class ExtendedMessage:

    def __init__(
            self,
            author,
            role="assistant",
            content="",
            token_arr=[],
            token_begin_index=-1,
            token_end_index=-1,
            clip=False,
            clip_token_limit=8192,
            tokenizer=None,
            token_generator="manual",
            build_from_uuid="",
            uuid=None,
        ):
        """
        Initializes an ExtendedMessage object with various attributes related to the message and its tokens.

        Args:
            author (str): The author of the message.
            role (str, optional): The role of the message sender. Defaults to "assistant".
            content (str, optional): The content of the message. Defaults to "".
            token_arr (list, optional): An array of tokens. Defaults to [].
            token_begin_index (int, optional): The starting index of the tokens. Defaults to -1.
            token_end_index (int, optional): The ending index of the tokens. Defaults to -1.
            clip (bool, optional): Whether to clip the content. Defaults to False.
            clip_token_limit (int, optional): The maximum number of tokens allowed when clipping. Defaults to 8192.
            tokenizer (Tokenizer, optional): The tokenizer to use for tokenization. Defaults to None.
            token_generator (str, optional): The method used to generate tokens. Defaults to "manual".
            build_from_uuid (str, optional): The UUID from which the message is built. Defaults to "".
            uuid (str, optional): The unique identifier for the message. Defaults to None.
        """
        self.author = author
        self.role = role
        self.content = content
        self.token_arr = token_arr
        self.token_begin_index = token_begin_index
        self.token_end_index = token_end_index
        # use property to ensure content is safe before use
        self._content_for_future = ""
        self._info = ""
        self.clip = clip
        if uuid is None:
            self.uuid = uuid_gen.uuid4().hex
        else:
            self.uuid = uuid
        self.build_from_uuid = build_from_uuid

        if not clip:
            self.generate_content_for_future(tokenizer=None, clip=False)
        else:
            self.generate_content_for_future(tokenizer=tokenizer, clip=True, clip_token_limit=clip_token_limit)  # ⭐ Generates future content with or without clipping
        self.eos_token_id = tokenizer.eos_token_id
        if token_generator == 'auto':
            self.token_arr = auto_tokenize_message(
                tokenizer, self.role, self.content_for_future
            )  # ⭐ Automatically generates tokens for the message (template-aware, see auto_tokenize_message)


    @property
    def content_for_future(self):
        """
        Property that returns the content for future use. If the content is empty, it sets a default value.

        Returns:
            str: The content for future use.
        """
        if self._content_for_future == "":
            self._content_for_future = "(Empty Content)"
            # raise ValueError("content_for_future is not set, or previous llm output is empty!")
        return self._content_for_future  # ⭐ Ensures that the content is not empty


    @property
    def need_training(self):
        """
        Property that determines whether the message needs training based on the author.

        Returns:
            bool: True if the message needs training, False otherwise.
        """
        NEED_TRAIN_AUTHORS = ["llm"]
        NON_TRAIN_AUTHORS = ["env", "initialization", "user", "memory", "llm(do_not_train)", "system"]
        assert (self.author in NEED_TRAIN_AUTHORS) or (self.author in NON_TRAIN_AUTHORS) or (self.author.endswith('(discard)')), f"author {self.author} is not identified"
        return (self.author in NEED_TRAIN_AUTHORS)  # ⭐ Determines if the message needs training based on the author


    def generate_content_for_future(self, tokenizer, clip, clip_token_limit=-1):
        """
        Generates a version of the content that is suitable for future use, potentially clipping it to fit within a specified token limit.

        Args:
            tokenizer (Tokenizer): The tokenizer used to count the number of tokens in the content.
            clip (bool): A flag indicating whether to clip the content if it exceeds the token limit.
            clip_token_limit (int, optional): The maximum number of tokens allowed. Defaults to -1, which means no clipping.

        Returns:
            None: The result is stored in the `_content_for_future` attribute.
        """
        _content: str = self.content
        if clip:
            assert clip_token_limit > 0, "clip_token_limit must be set when clip is True"
            n_token = len(tokenizer(_content, return_tensors="pt", padding=False)["input_ids"][0])  # ⭐ Count the number of tokens in the content
            if n_token > clip_token_limit:
                n_char = len(_content)  # 10,000
                eps = 100   # token
                preserve_percent = (clip_token_limit - eps) / n_token  # 3900 / 8000
                n_char_to_preserve = int(n_char * preserve_percent)
                _content = _content[:n_char_to_preserve] + "... truncate ..."  # ⭐ Truncate the content and add an ellipsis
        self._content_for_future = _content


    def get_loss_mask(self, blackout_token_combo, force_training: bool = False):
        """
        Generates a loss mask for the token array, blacking out specific token combinations and everything after the EOS token.

        Args:
            blackout_token_combo (list): A list of token IDs that should be blacked out in the first encounter.
            force_training (bool): Treat the message as trainable even when
                ``need_training`` is False. Used for replayed teacher turns,
                which must carry loss but need the *same* header/EOS blackout as
                on-policy turns — otherwise teacher sequences keep a handful of
                near-zero-logprob template tokens that on-policy ones drop, which
                shifts exactly the log-prob statistics DR3 estimates its density
                ratio from.

        Returns:
            list: A binary mask where 1 indicates the token contributes to the loss, and 0 indicates it does not.
        """
        def blackout_specific_token_ids_first_encounter(mask, arr, token_ids):
            """
            Blackouts the first occurrence of a specific sequence of token IDs in the mask.

            Args:
                mask (list): The initial mask.
                arr (list): The token array.
                token_ids (list): The sequence of token IDs to black out.

            Returns:
                list: The updated mask.
            """
            index = find_sublist_indices(arr, token_ids, reverse=False)
            if index >= 0:
                for i in range(index, index+len(token_ids)): mask[i] = 0  # ⭐ Blackout the specific token IDs
            else:
                # Thinking templates (Qwen3.5): history assistant turns are re-rendered
                # without the '<think>\n' part of the generation prompt, so the full
                # combo is absent. Blackout the longest combo prefix matching at the
                # start of the message ('<|im_start|>assistant\n') instead.
                k = 0
                while k < len(token_ids) and k < len(arr) and arr[k] == token_ids[k]:
                    k += 1
                for i in range(k): mask[i] = 0
            return mask

        def blackout_everything_after_eos_but_keep_eos(mask, token_arr, eos_token_id):
            """
            Blackouts everything after the EOS token, but keeps the EOS token itself.

            Args:
                mask (list): The initial mask.
                token_arr (list): The token array.
                eos_token_id (int): The EOS token ID.

            Returns:
                list: The updated mask.
            """
            eos_position = token_arr.index(eos_token_id) if eos_token_id in token_arr else -1
            if eos_position != -1:
                for i in range(eos_position + 1, len(mask)):
                    mask[i] = 0  # ⭐ Blackout everything after the EOS token
            return mask

        if self.need_training or force_training:
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(msg_token_mask, self.token_arr, blackout_token_combo)
            msg_token_mask = blackout_everything_after_eos_but_keep_eos(mask=msg_token_mask, token_arr=self.token_arr, eos_token_id=self.eos_token_id)
            return msg_token_mask
        else:
            msg_token_mask = [0] * len(self.token_arr)
            return msg_token_mask

    def get_inc_simple(self, text_frag_from, text_frag_to, tokenizer):
        """
        Get the incremental token array from `text_frag_from` to `text_frag_to`.

        Args:
            text_frag_from (str): The original text fragment.
            text_frag_to (str): The target text fragment.
            tokenizer: The tokenizer used to convert text into tokens.

        Returns:
            tuple: A tuple containing the list of incremental token IDs and a message string.
        """
        tokenizer_output = tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids  # ⭐ Accumulate the token IDs from the original text

        tokenizer_output = tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        input_id_increment = input_ids[len(token_ids_acc):]  # ⭐ Get the new tokens added in this step
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg


def find_sublist_indices(large_list, small_list, reverse=False):
    """
    Finds the starting index of the first occurrence of `small_list` in `large_list`.
    If `reverse` is True, the search is conducted from the end of `large_list`.

    Args:
        large_list (list): The list in which to search for the `small_list`.
        small_list (list): The sublist to search for within `large_list`.
        reverse (bool, optional): If True, the search starts from the end of `large_list`. Defaults to False.

    Returns:
        int: The starting index of the first occurrence of `small_list` in `large_list`, or -1 if not found.
    """
    small_len = len(small_list)
    if reverse:
        for i in reversed(range(len(large_list) - small_len + 1)):
            if large_list[i: i+small_len] == small_list:  # ⭐ Check if the current slice matches the small_list
                return i
    for i in range(len(large_list) - small_len + 1):
        if large_list[i: i+small_len] == small_list:  # ⭐ Check if the current slice matches the small_list
            return i
    return -1


def replace_token_ids(place_holder, replace_with, begin, end):
    """
    Replaces a segment of `place_holder` identified by `begin` and `end` markers with `replace_with`.
    Ensures that `begin` and `end` tokens are not included in the final result if they were part of `replace_with`.

    Args:
        place_holder (list): The original list where the replacement will occur.
        replace_with (list): The list to be inserted in place of the segment between `begin` and `end`.
        begin (list): The marker indicating the start of the segment to be replaced.
        end (list): The marker indicating the end of the segment to be replaced.

    Returns:
        list: The modified list after the replacement.
    """
    _begin_index = find_sublist_indices(place_holder, begin) + len(begin)  # ⭐ Find the index right after the `begin` marker
    _end_index = find_sublist_indices(place_holder, end, reverse=True)  # ⭐ Find the index at the `end` marker

    if replace_with[-len(end):] == end:  # remove end token
        replace_with = replace_with[:-len(end)]
    if replace_with[:len(begin)] == begin:  # remove begin token
        replace_with = replace_with[len(begin):]

    final = place_holder[:_begin_index] + replace_with + place_holder[_end_index:]  # ⭐ Construct the final list with the replacement
    return final
