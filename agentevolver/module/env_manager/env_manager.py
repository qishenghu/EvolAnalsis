import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Tuple
from typing import Optional, Union

import numpy as np
import torch
import random
import re
import os
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import (pad_sequence_to_length)

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.module.env_manager.env_worker import EnvWorker
from agentevolver.module.trainer.ae_async_llm_server_manager import BaAsyncLLMServerManager
from agentevolver.module.task_manager.rewards import grader_manager
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory, Sample
from agentevolver.utils.step_parser import parse_response_ids_to_steps
# do not delete this line
from agentevolver.module.task_manager.rewards import LlmAsJudgeRewardCalculator,LlmAsJudgeRewardCalculatorWithGT,LlmAsJudgeBinaryRewardCalculator,LlmAsJudgeBinaryRewardCalculatorWithGT,EnvGrader, AvgBinaryGTJudge, AvgLlmJudge
from beast_logger import register_logger
from agentevolver.module.exp_manager.exp_manager import TaskExpConfig, TrajExpConfig


def _sample_uid(sample: Sample) -> str:
    """Return the UID used by GRPO (group_ids is built from data_id)."""
    if not hasattr(sample, "data_id"):
        raise RuntimeError("sample is missing data_id required for GRPO grouping")
    return str(sample.data_id)


def _align_samples_by_complete_uid_groups(
    samples: List[Sample],
    world_size: int,
    *,
    expected_group_size: Optional[int] = None,
    require_divisible: bool = True,
    batch_label: str = "batch",
) -> List[Sample]:
    """Align a batch without ever splitting a GRPO UID group.

    expected_group_size is intentionally supplied only for a pure on-policy
    GRPO training batch. Replay/teacher mixtures can have a different per-UID
    composition, so they use the same whole-group alignment but do not inherit
    the pure-GRPO cardinality assertion.

    When alignment is needed, dynamic programming selects the largest number
    of samples whose complete UID groups sum to a multiple of world_size.
    Ties are resolved by the groups' first-appearance order. We deliberately do
    not pad/resample: duplicating an on-policy rollout would change the GRPO
    group statistics while pretending to provide another independent sample.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if expected_group_size is not None and expected_group_size <= 0:
        raise ValueError(
            f"expected_group_size must be positive, got {expected_group_size}"
        )
    if not samples:
        if require_divisible:
            raise RuntimeError(f"{batch_label} contains no samples")
        return []

    # Plain dict preserves first-appearance order on all supported Pythons.
    uid_to_samples: Dict[str, List[Sample]] = {}
    for sample in samples:
        uid_to_samples.setdefault(_sample_uid(sample), []).append(sample)

    if expected_group_size is not None:
        malformed = [
            (uid, len(group))
            for uid, group in uid_to_samples.items()
            if len(group) != expected_group_size
        ]
        if malformed:
            details = ", ".join(
                f"uid={uid!r}:count={count}" for uid, count in malformed
            )
            raise RuntimeError(
                "pure on-policy GRPO group integrity violation before optimizer: "
                f"each UID must contain exactly rollout.n={expected_group_size} "
                f"samples; {details}"
            )

    # Validation never reaches an optimizer and must retain every requested
    # task. Its final partial DP batch is legitimate; do not silently turn an
    # N-task evaluation into floor(N/world_size)*world_size tasks merely because
    # a divisible subset happens to exist.
    if not require_divisible:
        return list(samples)

    if len(samples) % world_size == 0:
        return list(samples)

    group_items = list(uid_to_samples.items())

    # remainder -> (kept sample count, tuple(group indices)). Retaining only the
    # best candidate for each remainder is sufficient because future group
    # sizes depend on neither the path nor the already-selected UIDs.
    states: Dict[int, Tuple[int, Tuple[int, ...]]] = {0: (0, ())}
    for group_index, (_uid, group) in enumerate(group_items):
        group_size = len(group)
        next_states = dict(states)
        for kept_count, kept_indices in list(states.values()):
            candidate_count = kept_count + group_size
            candidate_indices = kept_indices + (group_index,)
            remainder = candidate_count % world_size
            incumbent = next_states.get(remainder)
            if (
                incumbent is None
                or candidate_count > incumbent[0]
                or (
                    candidate_count == incumbent[0]
                    and candidate_indices < incumbent[1]
                )
            ):
                next_states[remainder] = (candidate_count, candidate_indices)
        states = next_states

    kept_count, kept_indices = states[0]
    if kept_count == 0:
        message = (
            f"{batch_label} has {len(samples)} samples across "
            f"{len(group_items)} complete UID groups, but no non-empty complete-"
            f"group subset is divisible by world_size={world_size}"
        )
        raise RuntimeError(message + "; refusing to enter the optimizer")

    kept_uids = {group_items[index][0] for index in kept_indices}
    aligned = [sample for sample in samples if _sample_uid(sample) in kept_uids]
    if len(aligned) != kept_count or len(aligned) % world_size != 0:
        raise AssertionError(
            "internal complete-group alignment error: "
            f"selected={kept_count}, materialized={len(aligned)}, "
            f"world_size={world_size}"
        )

    kept_index_set = set(kept_indices)
    dropped_uids = [
        uid for index, (uid, _group) in enumerate(group_items)
        if index not in kept_index_set
    ]
    logger.warning(
        f"[{batch_label}] Deterministically trimmed {len(samples) - len(aligned)} "
        f"samples as {len(dropped_uids)} complete UID group(s) to align with "
        f"world_size={world_size}; dropped_uids={dropped_uids}"
    )
    return aligned


_FINISH_REASON_CODE = {
    None: -1,
    "stop": 0,
    "length": 1,
}


def _finish_reason_code(reason) -> int:
    """Compact tensor representation; reserve 2 for known nonstandard values."""
    if reason is not None:
        reason = str(getattr(reason, "value", reason))
    return _FINISH_REASON_CODE.get(reason, 2)


def init_logger(experiment_name):
    """
    Initializes the logger with the given experiment name and sets up the logging environment.

    Args:
        experiment_name (str): The name of the experiment for which the logger is being initialized.
    """
    if 'BEST_LOGGER_INIT' in os.environ: return  # Prevents re-initialization in ray environment
    os.environ['BEST_LOGGER_INIT'] = '1'
    os.environ['BEST_LOGGER_WEB_SERVICE_URL'] = "http://127.0.0.1:8181/"
    from datetime import datetime
    final_log_path = os.path.join( "experiments", experiment_name, "trace_rollout", datetime.now().strftime("%Y_%m_%d_%H_%M"))
    non_console_mods = ["conversation", "rollout", "token_clip", "bad_case", "env_clip"]
    register_logger(mods=["evaluation", "exception"], non_console_mods=non_console_mods, auto_clean_mods=[], base_log_path=final_log_path, debug=False)
    print('Run `beast_logger_go` and click the url to inspect rollout logs. Continue in 5 seconds')
    time.sleep(2.5)



class ParallelEnvManager(object):
    """
    Manages a parallel environment for running multiple tasks, handling retries, logging, and using a language model to generate responses, ultimately returning the trajectories of the tasks.
    """
    def __init__(self, config: DictConfig, async_rollout_manager: BaAsyncLLMServerManager, max_parallel: int,
                 max_llm_retries: int = 3, **kwargs):
        """
        Initializes the ParallelEnvManager with the provided configuration and settings.

        Args:
            config (DictConfig): Configuration dictionary containing all necessary parameters.
            async_rollout_manager (BaAsyncLLMServerManager): Manager for asynchronous rollouts.
            max_parallel (int): Maximum number of parallel tasks that can be run.
            max_llm_retries (int, optional): Maximum number of retries for the language model. Defaults to 3.
            **kwargs: Additional keyword arguments.
        """
        init_logger(experiment_name=config.trainer.experiment_name)  # ⭐ Initializes the logger
        super().__init__(**kwargs)

        self.config: DictConfig = config
        self.async_rollout_manager: BaAsyncLLMServerManager = async_rollout_manager
        self.max_parallel: int = max_parallel
        self.max_llm_retries: int = max_llm_retries

        self.rollout_n = config.actor_rollout_ref.rollout.n
        self.model_name = self.async_rollout_manager.chat_scheduler.model_name
        self.tokenizer = self.async_rollout_manager.chat_scheduler.completion_callback.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.rollout_config = config.actor_rollout_ref.rollout

        # self.experience_template = config.hybrid_experience_training.experience_template
        self.llm_mode = "local" # use fsdp worker ("local") or use foreign server ("remote")
        self.current_token = 0
        self.current_token_count_time = time.time()


    def get_llm_chat_fn(self, sampling_params: dict = None) -> callable:
        """
        Returns a callable function for chatting with a language model. The returned function is either
        `llm_chat` or `llm_chat_remote` based on the `llm_mode` attribute.

        Args:
            sampling_params (dict, optional): Default sampling parameters for the chat function.

        Returns:
            callable: A function to chat with the language model.
        """

        def llm_chat(messages: List[Dict[str, str]],
                     custom_sampling_params: dict = None,
                     request_id: str = None) -> dict:
            """
            Sends messages to the language model and returns the assistant's response. This function handles
            retries and updates the sampling parameters.

            Args:
                messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "value" keys.
                custom_sampling_params (dict, optional): Custom sampling parameters to override the default ones.
                request_id (str, optional): A unique identifier for the request.

            Returns:
                dict: The last message in the input messages, which is expected to be the assistant's response.
            """
            # TODO: sending sampling_params to rollout server
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ Update sampling parameters

            input_messages = copy.deepcopy(messages)
            weighted_addresses = self.async_rollout_manager.chat_scheduler.weighted_addresses
            # logger.info(f"weighted_addresses={weighted_addresses}")
            for i in range(self.max_llm_retries):
                try:
                    self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                       sampling_params=updated_sampling_params,
                                                                       request_id=request_id)  # ⭐ Submit chat completions
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            last_message = input_messages[-1]
            if last_message.get("role") != "assistant":
                raise RuntimeError(
                    "Chat completion callback did not append an assistant message. "
                    f"last_role={last_message.get('role')}, sampling_params={updated_sampling_params}"
                )
            return last_message

        def llm_chat_remote(messages: List[Dict[str, str]],
                           custom_sampling_params: dict = None,
                           request_id: str = None) -> dict:
            """
            Sends messages to the remote language model and returns the assistant's response. This function
            handles retries and updates the sampling parameters.

            Args:
                messages (List[Dict[str, str]]): A list of message dictionaries with "role" and "value" keys.
                custom_sampling_params (dict, optional): Custom sampling parameters to override the default ones.
                request_id (str, optional): A unique identifier for the request.

            Returns:
                dict: The last message in the output messages, which is expected to be the assistant's response.
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)
            updated_sampling_params.update({"logprobs": 1, "return_tokens_as_token_ids": True})  # ⭐ Update sampling parameters
            input_messages = copy.deepcopy(messages)
            for i in range(self.max_llm_retries):
                try:
                    output_message = self.async_rollout_manager.submit_chat_completions(messages=input_messages,
                                                                                         sampling_params=updated_sampling_params,
                                                                                         request_id=request_id)  # ⭐ Submit chat completions
                    break
                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(2**i)
            last_message = output_message[-1]
            if last_message.get("role") != "assistant":
                raise RuntimeError(
                    "Remote chat completion did not return an assistant message. "
                    f"last_role={last_message.get('role')}, sampling_params={updated_sampling_params}"
                )
            return last_message

        if self.llm_mode == "remote":
            return llm_chat_remote
        else:
            return llm_chat

    def _sampling_params_for_mode(
        self, mode: Literal["sample", "validate"]
    ) -> dict:
        """Build the exact per-request sampling contract for one rollout mode."""
        sampling_params = dict(
            n=1,
            max_completion_tokens=self.rollout_config.response_length,
            temperature=self.rollout_config.temperature,
            top_p=self.rollout_config.top_p,
        )
        stop_sequences = self._cfg_to_list(
            self.rollout_config.get("stop_sequences", None)
        )
        if stop_sequences:
            sampling_params["stop"] = stop_sequences

        if mode == "validate":
            val_kwargs = self.rollout_config.val_kwargs
            sampling_params["temperature"] = val_kwargs.temperature
            sampling_params["top_k"] = val_kwargs.top_k
            sampling_params["top_p"] = val_kwargs.top_p
            # vLLM accepts an OpenAI-compatible per-request seed.  Keep it in
            # the explicit validation request even for temperature-zero runs,
            # so a future sampling-policy change cannot silently drop it.
            validation_seed = val_kwargs.get("seed", None)
            if validation_seed is not None:
                sampling_params["seed"] = int(validation_seed)
            val_stop_sequences = self._cfg_to_list(
                val_kwargs.get("stop_sequences", None)
            )
            sampling_params["stop"] = (
                val_stop_sequences if val_stop_sequences else stop_sequences
            )

        return sampling_params

    @staticmethod
    def _cfg_to_list(value):
        if value is None:
            return None
        if isinstance(value, ListConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], (list, tuple, ListConfig)):
                return ParallelEnvManager._cfg_to_list(value[0])
            return value
        return [value]

    def step_status_printer(self, tmux):
        """
        Prints the current status of the steps in the parallel environment, including the number of threads in different step ranges and the token generation rate.

        Args:
            tmux (dict): A dictionary containing the 'step' and 'token' information for the current state of the environment.
        """
        # Initialize a counter to keep track of the number of threads in each step range
        step_counter = {}

        # Calculate the total tokens and the time elapsed since the last count
        current_token = sum(tmux['token'])
        current_time = time.time()
        delta_token = current_token - self.current_token
        delta_time = current_time - self.current_token_count_time
        self.current_token = current_token
        self.current_token_count_time = current_time
        token_gen_per_sec_str = f"{delta_token/delta_time:.2f} tokens/s" if delta_time > 0 else "N/A"

        # Categorize the steps into bins and count the number of threads in each bin
        for step in tmux['step']:
            if step == -1:
                step_counter[(-1, 'terminated')] = step_counter.get((-1, 'terminated'), 0) + 1
                continue
            else:
                start = (step // 5) * 5
                end = start + 5
                step_counter[(start, end)] = step_counter.get((start, end), 0) + 1

        # Sort the step counter by the start value of each bin
        step_counter = dict(sorted(step_counter.items(), key=lambda x: x[0][0]))  # ⭐ Sort the step counter to ensure the output is in ascending order

        # Prepare the print buffer with the formatted step ranges and thread counts
        print_buf = []
        for (start, end), count in step_counter.items():
            if start != -1:
                print_buf += [f"[{start}-{end}]:{count} threads"]
        for (start, end), count in step_counter.items():
            if start == -1:
                print_buf += [f"[finished]:{count} threads"]

        # Print the rollout progress with the token generation rate and the step ranges
        print(f"Rollout progress ({token_gen_per_sec_str}): " + "  //  ".join(print_buf))



    def rollout_env_worker(self, task: Task, traj_exp_config: TrajExpConfig, data_id: str, rollout_id: str, mode: Literal["sample", "validate"],
                           thread_index: int, tmux: dict, stop:list, **kwargs) -> Trajectory:
        """
        Processes a single task in a thread-safe way, handling retries and exceptions.

        Args:
            task (Task): The task to be processed.
            traj_exp_config (TrajExpConfig): Experience Configuration for the trajectory.
            data_id (str): The ID of the data.
            rollout_id (str): The ID of the rollout.
            mode (Literal["sample", "validate"]): The mode of operation, either 'sample' or 'validate'.
            thread_index (int): The index of the thread.
            tmux (dict): TMUX configuration.
            stop (list): List of stop conditions.
            **kwargs: Additional keyword arguments.

        Returns:
            Trajectory: The trajectory generated from the task execution.
        """
        max_retry = 4
        for retry in range(max_retry):
            try:

                # TODO add try exception
                sampling_params = self._sampling_params_for_mode(mode)

                llm_chat_fn = self.get_llm_chat_fn(sampling_params)
                reward_caculator=grader_manager.get_calculator(task.evaluator, task=task)
                agent_flow: BaseAgentFlow = AgentFlow(
                    reward_calculator=reward_caculator,
                    llm_chat_fn=llm_chat_fn,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    **kwargs
                )

                env_worker = EnvWorker(task=task, thread_index=thread_index, config=self.config, tokenizer=self.tokenizer)
                trajectory: Trajectory = env_worker.execute(data_id=data_id, rollout_id=rollout_id, traj_exp_config=traj_exp_config, agent_flow=agent_flow, tmux=tmux, stop=stop) # ⭐ Execute the task and generate the trajectory
                if trajectory.metadata is None:
                    trajectory.metadata = {}
                # Validation usually uses val_kwargs.n=1, whereas training uses
                # rollout.n. Preserve the mode so only optimizer-bound pure
                # GRPO batches receive the strict rollout.n cardinality check.
                trajectory.metadata["rollout_mode"] = mode
                return trajectory

            except Exception as e:
                if retry < max_retry - 1:
                    logger.bind(exception=True).exception(f"rollout_env_worker error: {e.args}, retrying {retry + 1}/{max_retry}")
                    time.sleep(2 ** retry)
                else:
                    logger.bind(exception=True).exception(f"rollout_env_worker failed after {max_retry} retries: {e.args}")
                    raise e

    def rollout(
        self, 
        tasks: List[Task], 
        task_exp_configs: List[TaskExpConfig], 
        mode: Literal["sample", "validate"], 
        epoch: str,
        n_rollout_override: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        Executes a list of tasks in a parallel environment using a thread pool, with automatic retries for failed tasks.

        Args:
            tasks (List[Task]): A list of tasks to be processed.
            task_exp_configs (List[TaskExpConfig]): A list of experience configurations corresponding to each task.
            mode (Literal["sample", "validate"]): The mode of operation, either 'sample' or 'validate'.
            epoch (str): The current epoch identifier, used for logging and progress bar.
            n_rollout_override (Optional[int]): Override the number of rollouts per task (for LUFFY-style mixing).
                If provided, this value will be used instead of the default rollout_n.

        Returns:
            List[Trajectory]: A sorted list of Trajectory objects representing the results of the successfully completed tasks.
        """
        traj_cmt_array = []
        # ⭐ LUFFY: 支持 n_rollout_override 参数
        if n_rollout_override is not None:
            base_rollout_n = n_rollout_override
        else:
            base_rollout_n = self.rollout_config.val_kwargs.n if mode == "validate" else self.rollout_n
        future_to_params: Dict[Future, Tuple[Task, TrajExpConfig, str, str, str, int, dict, list[bool]]] = {}
        max_trajectory_resubmits = int(
            self.rollout_config.get("max_trajectory_resubmits", 2)
        )
        if max_trajectory_resubmits < 0:
            raise ValueError("max_trajectory_resubmits must be non-negative")
        resubmit_counts: Dict[Tuple[str, str], int] = {}

        # ⭐ Experience Replay: 计算每个 task 的实际 on-policy rollout 数量
        # 如果 task 有 n_offpolicy_trajectories，则减少相应的 on-policy rollout
        task_rollout_counts = []
        total_rollouts = 0
        for task in tasks:
            n_offpolicy = 0
            if hasattr(task, 'metadata') and task.metadata:
                n_offpolicy = task.metadata.get("n_offpolicy_trajectories", 0)
            # on-policy rollout 数量 = base_rollout_n - n_offpolicy
            task_rollout_n = max(1, base_rollout_n - n_offpolicy)  # 至少保留 1 次 on-policy rollout
            task_rollout_counts.append(task_rollout_n)
            total_rollouts += task_rollout_n
        
        tmux = {
            'step': [0 for _ in range(total_rollouts)],
            'token': [0 for _ in range(total_rollouts)],
        }
        stop = [False for _ in range(total_rollouts)]

        thread_index = 0
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # 2. submit: submit all tasks to the thread pool
            for data_id, (task, task_exp_config) in enumerate(zip(tasks, task_exp_configs)):
                task_rollout_n = task_rollout_counts[data_id]
                for rollout_id in range(task_rollout_n):
                    add_exp = task_exp_config.add_exp[rollout_id] if rollout_id < len(task_exp_config.add_exp) else False
                    train_mode = task_exp_config.train_mode
                    traj_exp_config = TrajExpConfig(
                        add_exp=add_exp, train_mode=train_mode, task_id=task.task_id, data_id=data_id, rollout_id=rollout_id, mode=mode)

                    params = (task, traj_exp_config, str(data_id), str(rollout_id), mode, thread_index, tmux,stop)
                    future = executor.submit(self.rollout_env_worker, *params)
                    future_to_params[future] = params
                    thread_index += 1

            total_rollouts = len(future_to_params)
            pbar = tqdm(total=total_rollouts, desc=f"Epoch {epoch}: Collecting rollouts")

            # 3. wait for all tasks to complete
            while future_to_params:
                # if any future is done, process it
                for future in as_completed(future_to_params):
                    # get the corresponding params, and remove it from the dict
                    params = future_to_params.pop(future)
                    self.step_status_printer(tmux) # cc: i don't know what this is

                    # 4. get the results and handle errors
                    try:
                        result = future.result()  # ⭐ Retrieve the result from the completed future

                        # if the result has metadata error, try to recover
                        if 'error' in result.metadata:
                            error_msg = result.metadata['error']
                            retry_key = (params[2], params[3])
                            resubmit_counts[retry_key] = resubmit_counts.get(retry_key, 0) + 1
                            if resubmit_counts[retry_key] > max_trajectory_resubmits:
                                raise RuntimeError(
                                    "trajectory retry budget exhausted for "
                                    f"data_id={params[2]} rollout_id={params[3]}: {error_msg}"
                                )
                            logger.warning(f"Task {params[1]}-{params[2]} failed with metadata error: {error_msg}. Retrying... \n Task: {params[0]}")
                            # as most errors are internet error or quota, we wait before resubmit it
                            time.sleep(30)
                            # resubmit and reset tmux and stop
                            thread_index=params[5]
                            for k in tmux: tmux[k][thread_index] = 0
                            stop[thread_index]=False
                            new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                            future_to_params[new_future] = params
                            continue

                        # 5. if the task is successful, add it to the result list
                        result.metadata["expected_group_rollouts"] = int(
                            task_rollout_counts[int(params[2])]
                        )
                        traj_cmt_array.append(result)
                        pbar.update(1) # update progress bar when success

                    except Exception as e:
                        # handle the uncaught exception
                        retry_key = (params[2], params[3])
                        resubmit_counts[retry_key] = resubmit_counts.get(retry_key, 0) + 1
                        if resubmit_counts[retry_key] > max_trajectory_resubmits:
                            raise RuntimeError(
                                "trajectory retry budget exhausted for "
                                f"data_id={params[2]} rollout_id={params[3]}"
                            ) from e
                        logger.error(
                            f"Task {params[1]}-{params[2]} raised an exception: {e}. "
                            f"Retrying {resubmit_counts[retry_key]}/{max_trajectory_resubmits}... \n Task: {params[0]}"
                        )
                        # resubmit, and reset tmux and stop
                        thread_index=params[5]
                        for k in tmux: tmux[k][thread_index] = 0
                        stop[thread_index]=False
                        new_future = executor.submit(self.rollout_env_worker, *params) # type: ignore
                        future_to_params[new_future] = params
            pbar.close()

        task_success_rate = np.mean([cmt.reward.success_rate for cmt in traj_cmt_array])
        for cmt in traj_cmt_array:
            cmt.current_batch_success_rate = np.mean(task_success_rate)

        # keep trajectory sorted
        traj_cmt_array = sorted(traj_cmt_array, key=lambda x: (int(x.data_id), int(x.rollout_id)))
        return traj_cmt_array


    # TODO: define an extra class for trajectory-dataproto converting.
    def to_dataproto(
        self, cmt_array, *, optimizer_batch: Optional[bool] = None
    ) -> DataProto:
        """
        Converts a list of trajectories into a DataProto object.

        Args:
            cmt_array (list): A list of trajectories that need to be converted.

        Returns:
            DataProto: The resulting DataProto object after conversion.
        """
        # Step 1: Convert trajectories to samples: tokenizing
        samples = self.trajectories_to_samples(
            cmt_array, optimizer_batch=optimizer_batch
        )  # ⭐ Tokenize the trajectories to create samples

        # Step 2: Convert samples to DataProto: padding
        dataproto = self.samples_to_dataproto(samples)  # ⭐ Pad the samples and convert them to DataProto

        return dataproto

    def _align_teacher_log_probs(
        self,
        teacher_log_probs: List[float],
        response_loss_mask: List[int],
        response_length: int,
        response_ids: Optional[List[int]] = None,
        teacher_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        将 Teacher 的 log_probs 对齐到 tokenized response 序列。
        
        ⭐ 对齐策略（两种模式）：
        
        模式 1（精确对齐，推荐）：如果提供了 response_ids 和 teacher_token_ids
        - 使用 token_ids 匹配，找到 teacher token 在 response 中的精确位置
        - 这样可以正确跳过 chat template 添加的特殊标记
        
        模式 2（简单对齐）：如果没有提供 token_ids
        - 按顺序填充到 loss_mask=1 的位置
        - 可能会因为 chat template 特殊标记导致位置偏移
        
        ⭐ Chat Template 问题说明：
        - Qwen 的 chat template 在每条 assistant 消息前后添加特殊标记：
          <|im_start|>assistant\n[实际内容]<|im_end|>
        - 采集时的 log_probs 只包含 [实际内容] 部分
        - 训练时 tokenize 后 loss_mask=1 包含了整个消息（含特殊标记）
        - 所以需要跳过特殊标记，只对 [实际内容] 部分对齐
        
        Args:
            teacher_log_probs: Teacher 生成的 log_probs 列表（只有 LLM 生成的 token）
            response_loss_mask: Response 部分的 loss_mask（1=LLM, 0=非 LLM）
            response_length: Response 序列长度
            response_ids: Response 部分的 token ids（用于精确对齐）
            teacher_token_ids: Teacher 生成的 token ids（用于精确对齐）
            
        Returns:
            torch.Tensor: 对齐后的 log_probs，shape [response_length]
        """
        # 创建全零的对齐后 log_probs
        aligned_log_probs = torch.zeros(response_length, dtype=torch.float32)
        
        if not teacher_log_probs:
            return aligned_log_probs
        
        # 找到所有 LLM 响应位置（loss_mask=1）
        llm_positions = []
        for i, mask_val in enumerate(response_loss_mask):
            if mask_val == 1:
                llm_positions.append(i)
        
        # ⭐ 模式 1：精确对齐（使用 token_ids）
        if response_ids is not None and teacher_token_ids is not None and len(teacher_token_ids) > 0:
            # 使用滑动窗口匹配 teacher_token_ids 在 response_ids 中的位置
            filled_count = self._align_by_token_ids(
                aligned_log_probs=aligned_log_probs,
                teacher_log_probs=teacher_log_probs,
                teacher_token_ids=teacher_token_ids,
                response_ids=response_ids,
                llm_positions=llm_positions,
            )
            
            if filled_count > 0:
                coverage = filled_count / len(teacher_log_probs) if teacher_log_probs else 0
                if coverage < 0.9:
                    logger.warning(
                        f"[Teacher Log Prob Alignment] Token-based alignment only matched "
                        f"{filled_count}/{len(teacher_log_probs)} tokens ({coverage:.1%}). "
                        f"Falling back to sequential alignment for remaining."
                    )
                else:
                    logger.debug(
                        f"[Teacher Log Prob Alignment] Token-based alignment: "
                        f"{filled_count}/{len(teacher_log_probs)} tokens matched ({coverage:.1%})."
                    )
                return aligned_log_probs
        
        # ⭐ 模式 2：简单对齐（按顺序填充）
        # 为了处理 chat template 问题，我们从后往前填充
        # 因为 chat template 主要在消息开头添加标记（<|im_start|>assistant\n）
        # 而结尾的 <|im_end|> 通常在 loss_mask=0 的区域（env response 后）
        num_to_fill = min(len(teacher_log_probs), len(llm_positions))
        
        # 从后往前填充（末尾对齐）
        for i in range(num_to_fill):
            teacher_idx = len(teacher_log_probs) - 1 - i
            llm_idx = len(llm_positions) - 1 - i
            pos = llm_positions[llm_idx]
            aligned_log_probs[pos] = teacher_log_probs[teacher_idx]
        
        # 如果 teacher_log_probs 比 LLM 位置多，记录警告
        if len(teacher_log_probs) > len(llm_positions):
            logger.warning(
                f"[Teacher Log Prob Alignment] More teacher log_probs ({len(teacher_log_probs)}) "
                f"than LLM positions ({len(llm_positions)}). Extra log_probs will be discarded."
            )
        elif len(teacher_log_probs) < len(llm_positions):
            unfilled = len(llm_positions) - len(teacher_log_probs)
            logger.debug(
                f"[Teacher Log Prob Alignment] Fewer teacher log_probs ({len(teacher_log_probs)}) "
                f"than LLM positions ({len(llm_positions)}). {unfilled} leading positions "
                f"(likely chat template tokens) will be 0."
            )
        
        return aligned_log_probs
    
    def _align_by_token_ids(
        self,
        aligned_log_probs: torch.Tensor,
        teacher_log_probs: List[float],
        teacher_token_ids: List[int],
        response_ids: List[int],
        llm_positions: List[int],
    ) -> int:
        """
        使用 token_ids 进行精确对齐。
        
        找到 teacher_token_ids 在 response_ids 中的位置，并填充对应的 log_probs。
        
        Returns:
            int: 成功匹配的 token 数量
        """
        if not teacher_token_ids or not response_ids:
            return 0
        
        filled_count = 0
        teacher_idx = 0
        
        # 遍历 llm_positions，尝试匹配 teacher_token_ids
        for pos in llm_positions:
            if teacher_idx >= len(teacher_token_ids):
                break
            
            if pos < len(response_ids) and response_ids[pos] == teacher_token_ids[teacher_idx]:
                # 匹配成功
                if teacher_idx < len(teacher_log_probs):
                    aligned_log_probs[pos] = teacher_log_probs[teacher_idx]
                    filled_count += 1
                teacher_idx += 1
        
        return filled_count

    def convert_offpolicy_to_cmt(
        self,
        offpolicy_trajectories: List[Trajectory],
        config: DictConfig,
        tokenizer,
        task_id_to_data_id: Optional[Dict[str, int]] = None
    ) -> List:
        """
        将 off-policy Trajectory 转换为 Linear_CMT 对象，以便使用相同的 tokenize 流程。
        
        Args:
            offpolicy_trajectories: Off-policy trajectory 列表
            config: 配置对象
            tokenizer: Tokenizer 实例
            task_id_to_data_id: task_id 到 data_id 的映射。
                ⭐ ExGRPO 设计：同一个 task 的所有 rollouts（on-policy 和 off-policy）应该共享同一个 data_id
                这个映射确保 off-policy trajectory 使用与对应 on-policy trajectory 相同的 data_id
            
        Returns:
            List[Linear_CMT]: 转换后的 Linear_CMT 对象列表
        """
        from agentevolver.module.context_manager.cmt_linear import Linear_CMT
        from agentevolver.module.context_manager.cmt_base import ExtendedMessage
        
        cmt_array = []
        
        for traj_index, traj in enumerate(offpolicy_trajectories):
            # 创建 Linear_CMT 对象
            cmt = Linear_CMT(config, tokenizer)
            
            # 设置基本信息
            task_id = traj.task_id if hasattr(traj, 'task_id') else traj.metadata.get("task_id", "unknown")
            
            # ⭐ ExGRPO 设计：使用 task_id_to_data_id 映射获取 data_id
            # 确保 off-policy trajectory 使用与对应 on-policy trajectory 相同的 data_id
            # 这样 GRPO 计算 advantage 时，同一个 task 的所有 rollouts 会在同一个分组中
            if task_id_to_data_id is not None and task_id in task_id_to_data_id:
                data_id = task_id_to_data_id[task_id]
            else:
                # 回退：使用 task_id 作为 data_id
                try:
                    data_id = int(task_id)
                except (ValueError, TypeError):
                    data_id = hash(task_id) % 100000
            
            cmt.data_id = str(data_id)
            cmt.rollout_id = str(traj.metadata.get("rollout_id", traj_index))
            cmt.task_id = task_id
            cmt.query = traj.query if hasattr(traj, 'query') else traj.metadata.get("query", "")
            cmt.reward = traj.reward
            cmt.is_terminated = traj.is_terminated if hasattr(traj, 'is_terminated') else True
            
            # 将 messages 转换为 ExtendedMessage 并填充 full_context
            # ⭐ Experience Replay: 对于 off-policy 数据，所有 LLM 消息的 author 保持为 "llm"
            # 这样 loss_mask 不会为 0，LLM 消息会参与 off-policy loss 计算
            # 使用 exp_mask=1 来区分 on-policy 和 off-policy，而不是用 loss_mask=0
            steps = traj.steps if hasattr(traj, 'steps') else []
            for msg in steps:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    author = msg.get("author", role)
                    
                    # ⭐ Experience Replay: LLM 消息保持 author="llm"，用于计算 off-policy loss
                    # 使用 exp_mask 区分 on/off-policy，而不是让 loss_mask=0
                    if role == "assistant":
                        author = "llm"  # 保持为 "llm"，loss_mask=1，参与 off-policy loss 计算
                    elif role == "user":
                        author = "user"
                    else:
                        author = role
                    
                    ext_msg = ExtendedMessage(
                        author=author,
                        role=role,
                        content=msg.get("content", ""),
                        token_generator='auto',
                        tokenizer=tokenizer,
                    )
                    cmt.full_context.append(ext_msg)
            
            # Apply the same sliding-window compression as on-policy rollouts
            # so that DR3 discriminator sees a consistent format.
            cmt._compress_old_context()
            # Thinking mode: strip history <think> blocks exactly like on-policy
            # rollouts, so the DR3 discriminator cannot separate teacher vs
            # student on formatting alone (no-op unless strip_think_in_history).
            n_think_stripped = cmt._strip_history_think()

            # 标记为 experience replay
            cmt.metadata["is_experience_replay"] = True
            cmt.metadata["old_log_probs"] = traj.metadata.get("old_log_probs")
            cmt.metadata["policy_version"] = traj.metadata.get("policy_version")
            cmt.metadata["entropy"] = traj.metadata.get("entropy")
            cmt.metadata["task_id"] = task_id
            # Preserve completion termination provenance when an online
            # trajectory enters replay. A historical 10K length failure must
            # never become an unobservable "unknown" sample.
            for termination_key in (
                "decision_finish_reasons",
                "decision_truncated_by_length",
                "decision_count",
                "length_truncated_decision_count",
                "has_length_truncated_decision",
                "selected_finish_reason",
                "selected_truncated_by_length",
                "length_truncation_terminated",
                "length_truncation_step",
            ):
                if termination_key in traj.metadata:
                    cmt.metadata[termination_key] = copy.deepcopy(
                        traj.metadata[termination_key]
                    )
            
            # ⭐ Teacher Experience: 传递 teacher 特有的 metadata
            cmt.metadata["is_teacher"] = traj.metadata.get("is_teacher", False)
            cmt.metadata["has_log_prob"] = traj.metadata.get("has_log_prob", False)
            # Teacher log_probs 可能在 log_probs 或 log_probs_per_turn 字段
            if traj.metadata.get("is_teacher", False):
                # 优先使用累积的 log_probs，便于对齐
                cmt.metadata["teacher_log_probs"] = traj.metadata.get("log_probs")
                cmt.metadata["teacher_log_probs_per_turn"] = traj.metadata.get("log_probs_per_turn")
                
                # ⭐ 提取累积的 token_ids（用于精确对齐）
                log_probs_per_turn = traj.metadata.get("log_probs_per_turn", [])
                if log_probs_per_turn:
                    accumulated_token_ids = []
                    for turn_info in log_probs_per_turn:
                        if isinstance(turn_info, dict) and "token_ids" in turn_info:
                            accumulated_token_ids.extend(turn_info["token_ids"])
                    if accumulated_token_ids:
                        cmt.metadata["teacher_token_ids"] = accumulated_token_ids

                # ⭐ Thinking teachers: stripping history <think> tokens breaks the
                # per-token correspondence between the recorded log_probs/token_ids
                # and the retokenized trajectory. The react_tags 72B teacher has no
                # think blocks (unaffected), and the DR3 path needs no teacher
                # log-probs — but per-token alignment is NOT yet supported here.
                if n_think_stripped > 0 and (
                    traj.metadata.get("log_probs") or traj.metadata.get("log_probs_per_turn")
                ):
                    logger.warning(
                        f"[convert_offpolicy_to_cmt] strip_think_in_history removed <think> blocks "
                        f"from {n_think_stripped} history turn(s) of a teacher trajectory that carries "
                        f"log_probs; per-token log-prob alignment for thinking teachers is not yet "
                        f"supported and the aligned values will be unreliable (task_id={task_id})."
                    )

            cmt_array.append(cmt)
        
        return cmt_array


    def get_extra(self, cmt):
        """
        Extracts and returns extra information from the comment's metadata.

        Args:
            cmt (object): The comment object containing metadata.

        Returns:
            dict: A dictionary with keys 'add_exp', 'task_train_expmode', 'experience_list', 
                  'is_experience_replay', and 'old_log_probs' corresponding to their respective 
                  values in the comment's metadata.
        """
        finish_reasons = list(
            cmt.metadata.get("decision_finish_reasons") or []
        )
        finish_reasons = [
            None
            if reason is None
            else str(getattr(reason, "value", reason))
            for reason in finish_reasons
        ]
        length_flags = [
            bool(value)
            for value in (
                cmt.metadata.get("decision_truncated_by_length") or []
            )
        ]
        if len(length_flags) < len(finish_reasons):
            length_flags.extend(
                reason == "length"
                for reason in finish_reasons[len(length_flags) :]
            )

        selected_step = cmt.metadata.get("selected_decision_step")
        try:
            selected_index = int(selected_step)
        except (TypeError, ValueError):
            selected_index = -1

        if "selected_finish_reason" in cmt.metadata:
            selected_finish_reason = cmt.metadata.get("selected_finish_reason")
            if selected_finish_reason is not None:
                selected_finish_reason = str(
                    getattr(selected_finish_reason, "value", selected_finish_reason)
                )
        elif 0 <= selected_index < len(finish_reasons):
            selected_finish_reason = finish_reasons[selected_index]
        elif finish_reasons:
            # Legacy transcript samples train over the full episode rather than
            # one snapshot. The last decision is the terminal completion.
            selected_finish_reason = finish_reasons[-1]
        else:
            selected_finish_reason = None

        if "selected_truncated_by_length" in cmt.metadata:
            selected_truncated_by_length = bool(
                cmt.metadata.get("selected_truncated_by_length")
            )
        elif 0 <= selected_index < len(length_flags):
            selected_truncated_by_length = length_flags[selected_index]
        elif length_flags:
            selected_truncated_by_length = length_flags[-1]
        else:
            selected_truncated_by_length = selected_finish_reason == "length"

        decision_count = int(
            cmt.metadata.get("decision_count", len(finish_reasons)) or 0
        )
        length_truncated_decision_count = int(
            cmt.metadata.get(
                "length_truncated_decision_count", sum(length_flags)
            )
            or 0
        )

        decision_context_stats = [
            dict(snapshot.context_stats)
            for snapshot in getattr(cmt, "decision_snapshots", [])
        ]
        decision_audit = []
        for snapshot in getattr(cmt, "decision_snapshots", []):
            stop_reason = snapshot.stop_reason
            if stop_reason is not None:
                stop_reason = str(getattr(stop_reason, "value", stop_reason))
            decision_audit.append(
                {
                    "step_index": int(snapshot.step_index),
                    "prompt_tokens": len(snapshot.prompt_token_ids),
                    "prompt_hash": snapshot.prompt_hash,
                    "raw_prompt_hash": snapshot.raw_prompt_hash,
                    "completion_tokens": len(snapshot.completion_token_ids),
                    "completion_hash": cmt.context_policy.ids_hash(
                        snapshot.completion_token_ids
                    ),
                    "template_hash": snapshot.template_hash,
                    "context_stats": dict(snapshot.context_stats),
                    "finish_reason": snapshot.finish_reason,
                    "truncated_by_length": bool(
                        snapshot.truncated_by_length
                    ),
                    "stop_reason": stop_reason,
                }
            )
        # Legacy/non-snapshot trajectories may still expose one selected
        # context record. Keep it auditable without pretending that it covers
        # every environment decision.
        if not decision_context_stats:
            selected_context_stats = cmt.metadata.get("context_stats")
            if isinstance(selected_context_stats, dict) and selected_context_stats:
                decision_context_stats = [dict(selected_context_stats)]

        extras = {
            "add_exp": cmt.metadata.get("add_exp", None),  # ⭐ Retrieves the 'add_exp' value from metadata
            "task_train_expmode": cmt.metadata.get("task_train_exp_mode", None),  # ⭐ Retrieves the 'task_train_exp_mode' value from metadata
            "experience_list": cmt.metadata.get("experience_list", []),  # ⭐ Retrieves the 'experience' list from metadata
            "is_experience_replay": cmt.metadata.get("is_experience_replay", False),  # ⭐ Experience Replay: 标记是否为 off-policy 数据
            "old_log_probs": cmt.metadata.get("old_log_probs"),  # ⭐ Experience Replay: 历史策略的 log_prob
            "recorded_response_mask": cmt.metadata.get("response_mask"),  # ⭐ Multi-turn: 历史轨迹的 response_mask，用于对齐
            # ⭐ Experience Replay: 轨迹陈旧度与质量诊断（用于分析 endo replay 的有效性/风险）
            "policy_version": cmt.metadata.get("policy_version"),
            "entropy": cmt.metadata.get("entropy"),
            "task_id": cmt.task_id,  # ⭐ 保存 task_id 用于后续处理
            "rollout_id": cmt.rollout_id,
            # Trusted low-cardinality task telemetry source. This comes from
            # the environment reset response, not query-text inference.
            "env_type": cmt.metadata.get("env_type"),
            "environment_task_type": cmt.metadata.get(
                "environment_task_type"
            ),
            # ⭐ Teacher Experience: 传递 teacher 特有字段
            "is_teacher": cmt.metadata.get("is_teacher", False),
            "has_log_prob": cmt.metadata.get("has_log_prob", False),
            "teacher_log_probs": cmt.metadata.get("teacher_log_probs"),  # 累积的 log_probs
            "teacher_log_probs_per_turn": cmt.metadata.get("teacher_log_probs_per_turn"),  # 分轮的 log_probs
            "teacher_token_ids": cmt.metadata.get("teacher_token_ids"),  # 累积的 token_ids（用于精确对齐）
            # Exact per-decision rollout/training contract.
            "snapshot_training": cmt.metadata.get("snapshot_training", False),
            "selected_decision_step": cmt.metadata.get("selected_decision_step"),
            "context_stats": cmt.metadata.get("context_stats"),
            "decision_context_stats": decision_context_stats,
            "decision_audit": decision_audit,
            "rollout_log_probs": cmt.metadata.get("rollout_log_probs"),
            "prompt_hash": cmt.metadata.get("prompt_hash"),
            "raw_prompt_hash": cmt.metadata.get("raw_prompt_hash"),
            "template_hash": cmt.metadata.get("template_hash"),
            "rollout_mode": cmt.metadata.get("rollout_mode"),
            "expected_group_rollouts": cmt.metadata.get(
                "expected_group_rollouts"
            ),
            # Completion termination contract. Ordered lists preserve every
            # environment decision; scalar fields refer to the selected
            # snapshot (or the terminal decision in legacy transcript mode).
            "finish_reason": selected_finish_reason,
            "truncated_by_length": selected_truncated_by_length,
            "decision_finish_reasons": finish_reasons,
            "decision_truncated_by_length": length_flags,
            "decision_count": decision_count,
            "length_truncated_decision_count": (
                length_truncated_decision_count
            ),
            "has_length_truncated_decision": bool(
                cmt.metadata.get(
                    "has_length_truncated_decision",
                    length_truncated_decision_count > 0,
                )
            ),
            "episode_end_reason": cmt.metadata.get("episode_end_reason"),
            "context_overflow_step": cmt.metadata.get(
                "context_overflow_step"
            ),
            "context_overflow_prompt_tokens": cmt.metadata.get(
                "context_overflow_prompt_tokens"
            ),
        }
        return extras


    def trajectories_to_samples(
        self,
        cmt_array: List,
        *,
        optimizer_batch: Optional[bool] = None,
    ) -> List[Sample]:
        """
        Converts a list of trajectories into a list of samples, ensuring the number of samples is divisible by the total number of GPUs across all nodes.

        Args:
            cmt_array (List): A list of trajectories to be converted into samples.

        Returns:
            List[Sample]: A list of samples with extras added and adjusted to be divisible by the world size.
        """
        # Step 1: Conversion
        sample_arr_final = []
        for cmt in cmt_array:
            # cc: message returned by the new env will be tagged as initialization, with no loss-mask
            sample_arr = cmt.group_tokenize()  # ⭐ Tokenize the trajectory into samples
            # group_tokenize may attach exact-decision metadata, so extract
            # extras only after the immutable training sample has been chosen.
            extras = self.get_extra(cmt)
            for sample in sample_arr:
                sample.extras = extras  # ⭐ Add extra information to each sample
            sample_arr_final += sample_arr

        # Step 2: Align optimizer-bound batches using complete GRPO UID groups.
        # Random sample-level deletion corrupts GRPO whenever it removes only
        # part of a UID group. Validation may remain non-divisible when no
        # complete-group alignment exists: it never reaches the optimizer, and
        # val_kwargs.n commonly differs from rollout.n.
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        rollout_modes = {
            sample.extras.get("rollout_mode")
            for sample in sample_arr_final
            if sample.extras.get("rollout_mode") is not None
        }
        inferred_optimizer_batch = "sample" in rollout_modes
        is_optimizer_batch = (
            inferred_optimizer_batch
            if optimizer_batch is None
            else bool(optimizer_batch)
        )
        has_offpolicy_or_teacher = any(
            sample.extras.get("is_experience_replay", False)
            or sample.extras.get("is_teacher", False)
            for sample in sample_arr_final
        )
        adv_estimator = str(
            OmegaConf.select(self.config, "algorithm.adv_estimator", default="")
        ).lower()
        is_pure_onpolicy_grpo = (
            is_optimizer_batch
            and adv_estimator.endswith("grpo")
            and not has_offpolicy_or_teacher
            and rollout_modes == {"sample"}
        )

        expected_snapshot_group_size = None
        if is_pure_onpolicy_grpo:
            uid_to_samples: Dict[str, List[Sample]] = {}
            for sample in sample_arr_final:
                uid_to_samples.setdefault(_sample_uid(sample), []).append(sample)
            snapshot_only = True
            expected_sizes = set()
            malformed_rollout_groups = []
            for uid, group in uid_to_samples.items():
                declared = {
                    int(
                        sample.extras.get("expected_group_rollouts")
                        or self.rollout_n
                    )
                    for sample in group
                }
                if len(declared) != 1:
                    raise RuntimeError(
                        "inconsistent expected rollout cardinality inside UID "
                        f"group {uid!r}: {sorted(declared)}"
                    )
                expected_rollouts = next(iter(declared))
                expected_sizes.add(expected_rollouts)
                unique_rollouts = {str(sample.rollout_id) for sample in group}
                if len(unique_rollouts) != expected_rollouts:
                    malformed_rollout_groups.append(
                        (uid, len(unique_rollouts), expected_rollouts)
                    )
                snapshot_only = snapshot_only and all(
                    bool(sample.extras.get("snapshot_training", False))
                    for sample in group
                )
            if malformed_rollout_groups:
                details = ", ".join(
                    f"uid={uid!r}:unique_rollouts={actual},expected={expected}"
                    for uid, actual, expected in malformed_rollout_groups
                )
                raise RuntimeError(
                    "pure on-policy GRPO rollout integrity violation before "
                    f"optimizer: {details}"
                )
            # Exact decision snapshots produce one trainable Sample per
            # rollout, so sample cardinality must also match. Legacy context
            # managers may emit multiple decision samples per rollout; their
            # rollout-level integrity was checked above without a false reject.
            if snapshot_only and len(expected_sizes) == 1:
                expected_snapshot_group_size = next(iter(expected_sizes))

        return _align_samples_by_complete_uid_groups(
            sample_arr_final,
            world_size,
            expected_group_size=expected_snapshot_group_size,
            require_divisible=is_optimizer_batch,
            batch_label=(
                "pure on-policy GRPO training batch"
                if is_pure_onpolicy_grpo
                else "mixed/replay or validation batch"
            ),
        )

    def samples_to_dataproto(self, samples: list[Sample]) -> DataProto:
        """
        Converts a list of Sample objects into a DataProto object, batching and padding the data.

        Args:
            samples (list[Sample]): A list of Sample objects to be converted.

        Returns:
            DataProto: A DataProto object containing the batched and padded data.
        """
        # Initialize lists to store batched data
        step_ids_list  = []
        steps_texts_list = []
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        prompt_exp_mask_list, response_exp_mask_list = [], []  # List of binary masks indicating whether to consider off_clip_high for each sample in the batch
        messages = []
        messages_raw = []
        reward_scores = []
        task_ids = []
        rollout_ids = []
        extras = [] # List of dictionaries containing supplementary data for each trajectory, including "add_exp", "task_train_expmode", "experience_list"
        k_text_list = []
        for sample in samples:
            # Validate that all fields have the same length
            assert len(sample.input_ids) == len(sample.attention_mask) == len(sample.position_ids) == len(
                sample.loss_mask), f"Sample {sample.request_id} has mismatched lengths: " \
                                f"{len(sample.input_ids)=}, {len(sample.attention_mask)=}, " \
                                f"{len(sample.position_ids)=}, {len(sample.loss_mask)=}"

            task_ids.append(sample.task_id)
            rollout_ids.append(sample.rollout_id)
            # Discard samples with prompt length exceeding limit
            if len(sample.prompt_ids) > self.config.data.max_prompt_length:
                raise RuntimeError(
                    f"Sample {getattr(sample, 'request_id', 'unknown')} has prompt_ids length {len(sample.prompt_ids)} "
                    f"greater than max_prompt_length {self.config.data.max_prompt_length}."
                )

            if len(sample.response_ids) > self.config.data.max_response_length:
                raise RuntimeError(
                    f"Sample {getattr(sample, 'request_id', 'unknown')} has response_ids "
                    f"length {len(sample.response_ids)} greater than max_response_length "
                    f"{self.config.data.max_response_length}; refusing to attach a "
                    "trajectory reward to a silently truncated prefix"
                )

            # ------------- shuchang 0714: append step_ids and steps_texts ------------
            resp_ids = sample.response_ids
            # shuchang: 0809
            # FIXME: Solve the issue of misaligned step IDs, use the unified step parsing function parse_response_ids_to_steps
            resp_ids = sample.response_ids
            parse_result = parse_response_ids_to_steps(resp_ids, self.tokenizer) # ⭐ Parse the response IDs into step IDs and texts

            step_ids_list.append(torch.tensor(parse_result.step_ids, dtype=torch.long))
            # generate steps_texts (for semantic evaluation)
            steps_texts_list.append([
                {"action": s["action_text"], "observation": s["observation_text"]}
                for s in parse_result.steps
            ])

            # Append tensors to respective lists
            assert len(sample.prompt_ids) != 0
            assert len(sample.response_ids) != 0
            prompt_ids.append(torch.tensor(sample.prompt_ids, dtype=torch.int))
            response_ids.append(torch.tensor(sample.response_ids, dtype=torch.int))

            prompt_attention_mask.append(torch.tensor(sample.prompt_attention_mask, dtype=torch.int))
            response_attention_mask.append(torch.tensor(sample.response_attention_mask, dtype=torch.int))

            prompt_position_ids.append(torch.tensor(sample.prompt_position_ids, dtype=torch.int))
            response_position_ids.append(torch.tensor(sample.response_position_ids, dtype=torch.int))

            prompt_loss_mask.append(torch.tensor(sample.prompt_loss_mask, dtype=torch.int))
            response_loss_mask.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))

            messages.append({"messages": sample.messages})
            messages_raw.append({"messages": getattr(sample, "messages_raw", None) or sample.messages})
            reward_scores.append(sample.reward_scores)
            extras.append(sample.extras)

            # Create experience mask: 
            # 1. Experience Replay (is_experience_replay=True): 只对 LLM 响应位置设置为 1（基于 loss_mask）
            # 2. Experience-guided (add_exp=True, task_train_expmode="discard"): prompt 和 response 都标记为 1
            # 3. 纯 On-policy: 全为 0
            # ⭐ 优先级是 experience replay > experience-guided > on-policy
            # ⭐ Multi-turn 关键：使用 loss_mask 来创建 exp_mask，确保只对 LLM 响应位置设置为 1
            if sample.extras.get("is_experience_replay", False):
                # Experience Replay: 只对 LLM 响应位置（loss_mask=1）设置 exp_mask=1
                # 这样 Environment 响应不会被标记为 off-policy
                prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
                # ⭐ 使用 response_loss_mask 而不是全 1，确保只标记 LLM 响应
                response_exp_mask_list.append(torch.tensor(sample.response_loss_mask, dtype=torch.int))
            elif sample.extras.get("add_exp", False) and sample.extras.get("task_train_expmode", None)=="discard":
                # Experience-guided: prompt 和 response 都标记为 1
                prompt_exp_mask_list.append(torch.ones(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.ones(len(sample.response_loss_mask), dtype=torch.int))
            else:
                # On-policy without experience: 全为 0
                prompt_exp_mask_list.append(torch.zeros(len(sample.prompt_loss_mask), dtype=torch.int))
                response_exp_mask_list.append(torch.zeros(len(sample.response_loss_mask), dtype=torch.int))



        max_prompt_length_this_batch = max([p.shape[-1] for p in prompt_ids])
        assert max_prompt_length_this_batch <= self.config.data.max_prompt_length
        max_response_length_this_batch = max([p.shape[-1] for p in response_ids])
        assert max_response_length_this_batch <= self.config.data.max_response_length

        # Batch and pad sequences
        # ------------- shuchang 0714: pad step_ids and steps_texts ------------
        step_ids_pad = pad_sequence(
            step_ids_list, batch_first=True, padding_value=-1
        )

        step_ids_pad = pad_sequence_to_length(
            step_ids_pad, self.config.data.max_response_length, -1
        )  # ⭐ Pad step IDs to the maximum response length
        # ------------- shuchang 0714: pad step_ids and steps_texts ------------


        prompt_ids =            pad_sequence(prompt_ids, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        prompt_attention_mask = pad_sequence(prompt_attention_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_position_ids =   pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        prompt_loss_mask =      pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        prompt_exp_mask_list =  pad_sequence(prompt_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")

        prompt_ids =            pad_sequence_to_length(prompt_ids, max_prompt_length_this_batch, self.pad_token_id, left_pad=True)
        prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_position_ids =   pad_sequence_to_length(prompt_position_ids, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_loss_mask =      pad_sequence_to_length(prompt_loss_mask, max_prompt_length_this_batch, 0, left_pad=True)
        prompt_exp_mask_list =  pad_sequence_to_length(prompt_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)

        response_ids =            pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        response_loss_mask =      pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        # response_exp_mask_list =  pad_sequence(response_exp_mask_list, batch_first=True, padding_value=0, padding_side="left")
        response_exp_mask_list  = pad_sequence(response_exp_mask_list,  batch_first=True, padding_value=0)        # shuchang debug: Remove padding_side="left"


        response_ids =            pad_sequence_to_length(response_ids, max_response_length_this_batch, self.pad_token_id)  # ⭐ Pad response IDs to the maximum response length
        response_attention_mask = pad_sequence_to_length(response_attention_mask, max_response_length_this_batch, 0)  # ⭐ Pad response attention mask to the maximum response length
        response_loss_mask =      pad_sequence_to_length(response_loss_mask, max_response_length_this_batch, 0)  # ⭐ Pad response loss mask to the maximum response length
        # response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_prompt_length_this_batch, 0, left_pad=True)  # ⭐ Pad response experience mask list to the maximum prompt length
        response_exp_mask_list =  pad_sequence_to_length(response_exp_mask_list, max_response_length_this_batch, 0) # shuchang debug: Should pad to max_response_length_this_batch, not the previous max_prompt_length_this_batch

        delta_position_id = torch.arange(1, response_ids.size(1) + 1, device=response_ids.device).unsqueeze(0).repeat(len(samples), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id  # ⭐ Calculate the position IDs for the response

        # Concatenate prompt and response tensors
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)  # ⭐ Concatenate prompt and response IDs
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)  # ⭐ Concatenate prompt and response attention masks
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)  # ⭐ Concatenate prompt and response position IDs
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)  # ⭐ Concatenate prompt and response loss masks
        # shuchang: construct group_id
        group_ids = torch.tensor([int(s.data_id) for s in samples], dtype=torch.long)  # ⭐ Construct group IDs from sample data
        # Validate masks have same shape
        exp_mask = torch.cat((prompt_exp_mask_list, response_exp_mask_list), dim=-1)  # ⭐ Concatenate prompt and response experience masks

        assert exp_mask.shape == loss_mask.shape, f"Shape mismatch: {exp_mask.shape} vs {loss_mask.shape}"

        # ⭐ Experience Replay: 处理 recorded_old_log_probs
        # 对于 self-generated experience: 直接使用保存的 old_log_probs（已对齐）
        # 对于 teacher experience: 需要将 teacher_log_probs 对齐到 response_loss_mask
        recorded_old_log_probs_list = []
        teacher_mask_list = []  # ⭐ 新增：标记哪些样本是 teacher 轨迹
        
        for sample in samples:
            is_teacher = sample.extras.get("is_teacher", False)
            has_log_prob = sample.extras.get("has_log_prob", False)
            response_length = len(sample.response_ids)
            
            if is_teacher and has_log_prob:
                # ⭐ Teacher Experience: 对齐 teacher_log_probs 到 response_loss_mask
                teacher_log_probs = sample.extras.get("teacher_log_probs")
                teacher_token_ids = sample.extras.get("teacher_token_ids")  # 用于精确对齐
                if teacher_log_probs is not None:
                    aligned_log_probs = self._align_teacher_log_probs(
                        teacher_log_probs=teacher_log_probs,
                        response_loss_mask=sample.response_loss_mask,
                        response_length=response_length,
                        response_ids=sample.response_ids,  # 用于精确对齐
                        teacher_token_ids=teacher_token_ids,  # 用于精确对齐
                    )
                    recorded_old_log_probs_list.append(aligned_log_probs)
                else:
                    # Teacher 有标记 has_log_prob 但没有实际数据，创建零向量
                    recorded_old_log_probs_list.append(
                        torch.zeros(response_length, dtype=torch.float32)
                    )
                # 标记为 teacher（用于 loss 计算）
                teacher_mask_list.append(torch.ones(response_length, dtype=torch.int))
            elif is_teacher:
                # ⭐ Teacher Experience 但没有 log_prob：使用 LUFFY 模式，无需 old_log_probs
                recorded_old_log_probs_list.append(
                    torch.zeros(response_length, dtype=torch.float32)
                )
                teacher_mask_list.append(torch.ones(response_length, dtype=torch.int))
            else:
                # ⭐ Self-generated Experience: 直接使用保存的 old_log_probs
                old_log_probs = sample.extras.get("old_log_probs")
                if old_log_probs is not None:
                    # 转换为 tensor 并对齐长度
                    if isinstance(old_log_probs, (list, np.ndarray)):
                        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
                    # 对齐到 response_length
                    if len(old_log_probs) > response_length:
                        old_log_probs = old_log_probs[:response_length]
                    elif len(old_log_probs) < response_length:
                        old_log_probs = torch.cat([
                            old_log_probs,
                            torch.zeros(response_length - len(old_log_probs), dtype=torch.float32)
                        ])
                    recorded_old_log_probs_list.append(old_log_probs)
                else:
                    # 如果没有记录，创建零向量（后续会重新计算）
                    recorded_old_log_probs_list.append(
                        torch.zeros(response_length, dtype=torch.float32)
                    )
                # 非 teacher
                teacher_mask_list.append(torch.zeros(response_length, dtype=torch.int))
        
        # Pad recorded_old_log_probs 到 max_response_length
        recorded_old_log_probs = pad_sequence(recorded_old_log_probs_list, batch_first=True, padding_value=0.0)
        recorded_old_log_probs = pad_sequence_to_length(
            recorded_old_log_probs, max_response_length_this_batch, 0.0
        )
        
        # ⭐ Teacher Experience: Pad teacher_mask 到 max_response_length
        # teacher_mask 标记哪些样本是 teacher 轨迹，用于 het_compute_teacher_aware_loss
        teacher_mask = pad_sequence(teacher_mask_list, batch_first=True, padding_value=0)
        teacher_mask = pad_sequence_to_length(teacher_mask, max_response_length_this_batch, 0)
        # 需要拼接 prompt 部分（prompt 部分 teacher_mask 全为 0）
        prompt_teacher_mask = torch.zeros_like(prompt_ids, dtype=torch.int)
        teacher_mask_full = torch.cat((prompt_teacher_mask, teacher_mask), dim=-1)

        # Preserve the behavior-policy log-probabilities returned by vLLM.  The
        # actor-side recomputation remains useful, but only as an identity/drift
        # check; it is not evidence that the sampled conditional was correct.
        rollout_log_probs_list = []
        rollout_log_probs_mask_list = []
        has_rollout_log_probs = False
        for sample in samples:
            raw_log_probs = sample.extras.get("rollout_log_probs")
            response_length = len(sample.response_ids)
            if raw_log_probs is None:
                if sample.extras.get("snapshot_training", False):
                    raise RuntimeError(
                        "snapshot sample is missing rollout log-probabilities"
                    )
                rollout_log_probs_list.append(
                    torch.zeros(response_length, dtype=torch.float32)
                )
                rollout_log_probs_mask_list.append(
                    torch.zeros(response_length, dtype=torch.int)
                )
                continue
            raw_log_probs = list(raw_log_probs)
            if len(raw_log_probs) != response_length:
                raise RuntimeError(
                    "rollout log-probability/token mismatch: "
                    f"{len(raw_log_probs)} != {response_length}"
                )
            has_rollout_log_probs = True
            rollout_log_probs_list.append(
                torch.tensor(raw_log_probs, dtype=torch.float32)
            )
            rollout_log_probs_mask_list.append(
                torch.tensor(sample.response_loss_mask, dtype=torch.int)
            )

        rollout_log_probs = pad_sequence(
            rollout_log_probs_list, batch_first=True, padding_value=0.0
        )
        rollout_log_probs = pad_sequence_to_length(
            rollout_log_probs, max_response_length_this_batch, 0.0
        )
        rollout_log_probs_mask = pad_sequence(
            rollout_log_probs_mask_list, batch_first=True, padding_value=0
        )
        rollout_log_probs_mask = pad_sequence_to_length(
            rollout_log_probs_mask, max_response_length_this_batch, 0
        )

        # Completion termination is sample-level state. Keeping the selected
        # values as tensors makes them survive padding, validation dispatch and
        # balance/reorder; the ordered per-decision audit trail remains in the
        # non-tensor batch and ``extras``.
        finish_reasons = []
        truncated_by_length = []
        decision_counts = []
        length_truncated_decision_counts = []
        has_length_truncated_decision = []
        decision_finish_reasons = []
        decision_truncated_by_length = []
        for sample in samples:
            sample_extras = sample.extras or {}
            reason = sample_extras.get("finish_reason")
            if reason is not None:
                reason = str(getattr(reason, "value", reason))
            is_length = bool(
                sample_extras.get("truncated_by_length", False)
            ) or reason == "length"
            all_reasons = list(
                sample_extras.get("decision_finish_reasons") or []
            )
            all_reasons = [
                None
                if item is None
                else str(getattr(item, "value", item))
                for item in all_reasons
            ]
            all_length_flags = [
                bool(item)
                for item in (
                    sample_extras.get("decision_truncated_by_length") or []
                )
            ]
            if len(all_length_flags) < len(all_reasons):
                all_length_flags.extend(
                    item == "length"
                    for item in all_reasons[len(all_length_flags) :]
                )
            decision_count = int(
                sample_extras.get("decision_count", len(all_reasons)) or 0
            )
            length_count = int(
                sample_extras.get(
                    "length_truncated_decision_count",
                    sum(all_length_flags),
                )
                or 0
            )

            finish_reasons.append(reason)
            truncated_by_length.append(is_length)
            decision_counts.append(decision_count)
            length_truncated_decision_counts.append(length_count)
            has_length_truncated_decision.append(
                bool(
                    sample_extras.get(
                        "has_length_truncated_decision", length_count > 0
                    )
                )
            )
            decision_finish_reasons.append(all_reasons)
            decision_truncated_by_length.append(all_length_flags)

        # Construct the batch using TensorDict
        batch_fields = {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "exp_mask": exp_mask,        # add exp_mask by ANNI
                "step_ids": step_ids_pad,
                "group_ids": group_ids,   # ★ add groupid
                "recorded_old_log_probs": recorded_old_log_probs,  # ⭐ Experience Replay: 历史策略的 old_log_probs
                "teacher_mask": teacher_mask_full,  # ⭐ Teacher Experience: 标记 teacher 轨迹位置
                "finish_reason_code": torch.tensor(
                    [_finish_reason_code(reason) for reason in finish_reasons],
                    dtype=torch.long,
                ),
                "truncated_by_length": torch.tensor(
                    truncated_by_length, dtype=torch.bool
                ),
                "decision_count": torch.tensor(
                    decision_counts, dtype=torch.long
                ),
                "length_truncated_decision_count": torch.tensor(
                    length_truncated_decision_counts, dtype=torch.long
                ),
                "has_length_truncated_decision": torch.tensor(
                    has_length_truncated_decision, dtype=torch.bool
                ),
        }
        if has_rollout_log_probs:
            batch_fields["rollout_log_probs"] = rollout_log_probs
            batch_fields["rollout_log_probs_mask"] = rollout_log_probs_mask

        context_stats = [sample.extras.get("context_stats") for sample in samples]
        if any(isinstance(stats, dict) for stats in context_stats):
            for stat_name in (
                "raw_prompt_tokens",
                "managed_prompt_tokens",
                "compressed_turns",
                "dropped_turns",
                "clipped_observations",
            ):
                batch_fields[f"context_{stat_name}"] = torch.tensor(
                    [
                        int(stats.get(stat_name, -1))
                        if isinstance(stats, dict)
                        else -1
                        for stats in context_stats
                    ],
                    dtype=torch.long,
                )
            batch_fields["context_selected_decision_step"] = torch.tensor(
                [
                    int(sample.extras.get("selected_decision_step", -1))
                    for sample in samples
                ],
                dtype=torch.long,
            )

        # Trajectory-wide context audit. Snapshot training deliberately uses
        # one immutable decision for the loss, but validation must still prove
        # whether management activated at any environment decision rather than
        # only at the selected snapshot.
        context_decision_counts = []
        context_active_decision_counts = []
        context_max_raw_prompt_tokens = []
        context_max_managed_prompt_tokens = []
        context_total_dropped_turns = []
        context_total_clipped_observations = []
        decision_context_stats_batch = []
        for sample in samples:
            raw_stats = list(
                (sample.extras or {}).get("decision_context_stats") or []
            )
            normalized_stats = [
                dict(stats) for stats in raw_stats if isinstance(stats, dict)
            ]
            valid_stats = [
                stats
                for stats in normalized_stats
                if int(stats.get("raw_prompt_tokens", -1)) >= 0
                and int(stats.get("managed_prompt_tokens", -1)) >= 0
            ]
            active_count = sum(
                int(
                    int(stats.get("raw_prompt_tokens", -1))
                    != int(stats.get("managed_prompt_tokens", -1))
                    or int(stats.get("compressed_turns", 0)) > 0
                    or int(stats.get("dropped_turns", 0)) > 0
                    or int(stats.get("clipped_observations", 0)) > 0
                )
                for stats in valid_stats
            )
            context_decision_counts.append(len(valid_stats))
            context_active_decision_counts.append(active_count)
            context_max_raw_prompt_tokens.append(
                max(
                    (int(stats["raw_prompt_tokens"]) for stats in valid_stats),
                    default=-1,
                )
            )
            context_max_managed_prompt_tokens.append(
                max(
                    (
                        int(stats["managed_prompt_tokens"])
                        for stats in valid_stats
                    ),
                    default=-1,
                )
            )
            context_total_dropped_turns.append(
                sum(int(stats.get("dropped_turns", 0)) for stats in valid_stats)
            )
            context_total_clipped_observations.append(
                sum(
                    int(stats.get("clipped_observations", 0))
                    for stats in valid_stats
                )
            )
            decision_context_stats_batch.append(normalized_stats)

        batch_fields.update(
            {
                "context_decision_count": torch.tensor(
                    context_decision_counts, dtype=torch.long
                ),
                "context_active_decision_count": torch.tensor(
                    context_active_decision_counts, dtype=torch.long
                ),
                "context_any_management_active": torch.tensor(
                    [count > 0 for count in context_active_decision_counts],
                    dtype=torch.bool,
                ),
                "context_max_raw_prompt_tokens": torch.tensor(
                    context_max_raw_prompt_tokens, dtype=torch.long
                ),
                "context_max_managed_prompt_tokens": torch.tensor(
                    context_max_managed_prompt_tokens, dtype=torch.long
                ),
                "context_total_dropped_turns": torch.tensor(
                    context_total_dropped_turns, dtype=torch.long
                ),
                "context_total_clipped_observations": torch.tensor(
                    context_total_clipped_observations, dtype=torch.long
                ),
            }
        )

        batch = TensorDict(batch_fields, batch_size=len(samples))

        decision_finish_reasons_array = np.empty(len(samples), dtype=object)
        decision_finish_reasons_array[:] = decision_finish_reasons
        decision_truncated_by_length_array = np.empty(
            len(samples), dtype=object
        )
        decision_truncated_by_length_array[:] = (
            decision_truncated_by_length
        )
        decision_context_stats_array = np.empty(len(samples), dtype=object)
        decision_context_stats_array[:] = decision_context_stats_batch

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "task_ids": np.array(task_ids),
                "rollout_ids": np.array(rollout_ids),
                "messages": np.array(messages),
                # pre-compression view; the State Channel hashes these
                "messages_raw": np.array(messages_raw),
                "reward_scores": np.array(reward_scores),
                "extras": np.array(extras),
                "steps": np.array(steps_texts_list, dtype=object),
                "finish_reasons": np.array(finish_reasons, dtype=object),
                "decision_finish_reasons": decision_finish_reasons_array,
                "decision_truncated_by_length": (
                    decision_truncated_by_length_array
                ),
                "decision_context_stats": decision_context_stats_array,
            }
        )
