import time
import os

import requests
from loguru import logger

from agentevolver.client.em_client import EMClient
from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.base_agent_flow import BaseAgentFlow
from agentevolver.utils.utils import convert_tool_to_user_message
from agentevolver.schema.trajectory import Reward, Trajectory
from best_logger import register_logger, print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear import Linear_CMT, ExtendedMessage
from agentevolver.module.context_manager.cmt_linear_think import LinearThinkCMT
from agentevolver.module.context_manager.cmt_base import (
    THINKING_MODE_NATIVE_QWEN3,
    chat_template_ids,
    resolve_thinking_mode,
)
from agentevolver.module.context_manager.cmt_context_clip import SelfContextClipCMT
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator
from typing import Any, Callable, Dict, List, Union, Optional
import threading
from agentevolver.module.exp_manager.exp_manager import TrajExpConfig, ExperienceWorker


log_generate_lock = threading.Lock()


def _is_env_infrastructure_error(error: Exception) -> bool:
    """Return whether an env.step failure must use the retry/fail path.

    A model-authored malformed action is a policy event and belongs in the
    sampled trajectory.  Transport failures and HTTP 5xx responses are not
    policy events: converting them into a zero-reward trajectory would silently
    train on infrastructure noise, so they must escape to the existing worker
    retry and trajectory-resubmit layers.
    """
    if isinstance(
        error,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    ):
        return True
    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response
        status_code = getattr(response, "status_code", None)
        return status_code is not None and 500 <= int(status_code) < 600
    return False


class AgentFlow(BaseAgentFlow):

    def __init__(
        self,
        reward_calculator: Optional[RewardCalculator] = None,
        decision_observer: Optional[Callable[..., None]] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the AgentFlow class.

        Args:
            reward_calculator (Optional[RewardCalculator]): An optional reward calculator object.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        super().__init__(**kwargs)  # ⭐ Call the constructor of the base class
        self._reward_calculator = reward_calculator
        # Optional, synchronous audit hook used by API-teacher collection.  It
        # observes the exact managed prompt and provider result before either is
        # normalized or written into the trajectory.  Production training does
        # not provide the hook, so its rollout path is unchanged.
        self._decision_observer = decision_observer
        # self._enable_context_generator=self.config.experience_maker.enable_context_generator

        self.instruction_template_ids = self.tokenizer.encode("user\n")  # ⭐ Encode the user instruction template
        self.response_template_ids = self.tokenizer.encode("assistant\n")  # ⭐ Encode the assistant response template
        # self.em_client = EMClient(base_url=self.config.experience_maker.base_url)  # ⭐ Initialize the EMClient
        self.sparse = self.config.actor_rollout_ref.rollout.sparse  # add sparse by ANNI 0723
        self.sciworld_success_threshold: float = float(
            getattr(self.config.actor_rollout_ref.rollout, "sciworld_success_threshold", 0)
        )
        self.cmt: Union[Linear_CMT, LinearThinkCMT] = None
        self.console_debug_mode: bool = self.config.actor_rollout_ref.rollout.debug_llm_io
        self.exp_worker = ExperienceWorker(config=self.config)


    def execute(self, context_manager, init_messages: List[dict], env: EnvClient, instance_id: str, tmux, stop, thread_index, task_id, traj_exp_config,data_id="", rollout_id="", query="", **kwargs) -> Linear_CMT:
        """
        Executes the interaction between the AI agent and the environment, managing the context, experience generation, and reward calculation.

        Args:
            context_manager (ContextManager): The context manager for the current task.
            init_messages (List[dict]): Initial messages for the task.
            env (EnvClient): The environment client.
            instance_id (str): The ID of the instance.
            tmux (dict): TMUX dictionary for tracking steps and tokens.
            stop (list): A list indicating whether to stop the thread.
            thread_index (int): The index of the current thread.
            task_id (str): The ID of the task.
            traj_exp_config (TrajExpConfig): Experience Configuration for the trajectory.
            data_id (str, optional): The ID of the data. Defaults to "".
            rollout_id (str, optional): The ID of the rollout. Defaults to "".
            query (str, optional): The query string. Defaults to "".
            **kwargs: Additional keyword arguments.

        Returns:
            Linear_CMT: The context manager after the execution.
        """
        self.cmt = context_manager
        # ⭐ CATALYST 提示臂:把教师 think 摘要注入首条 user 消息末尾。位置与
        # 试点采集器 HintedAgentFlow.execute 同序(manage_rollout_context /
        # save_init_input 之前)、同目标消息、同模板字节(sha pin 校验)。
        # hint 为 None(默认)时零行为——连 deepcopy 都不做(字节等价纪律)。
        catalyst_hint_text = getattr(traj_exp_config, "catalyst_hint_text", None)
        if catalyst_hint_text:
            from agentevolver.module.exp_manager.catalyst import (
                hint_sha256,
                inject_hint_into_init_messages,
            )
            init_messages = inject_hint_into_init_messages(
                init_messages, catalyst_hint_text
            )
            self.cmt.metadata["catalyst_arm"] = "hint"
            self.cmt.metadata["catalyst_hint_sha256"] = hint_sha256(
                catalyst_hint_text
            )
            # v3.1:hint 臂 critic 基线透传(None=未启用,零行为)
            _hint_vhat = getattr(traj_exp_config, "catalyst_hint_vhat", None)
            if _hint_vhat is not None:
                self.cmt.metadata["catalyst_hint_vhat"] = float(_hint_vhat)
        # ⭐ CATALYST v2 entry 臂(设计 §2,镜像试点 TakeoverAgentFlow 三步):
        # ① 逐步重放教师前 k 个 action 推进 env(失败抛
        #    CatalystEntryReplayError,worker 降级为裸臂重试);
        # ② save_init_input 之后把 k 对 (assistant, live env 观测) seed 进
        #    full_context——v6 起 assistant 侧带教师 think(可见性由渲染域
        #    决定:strip 域模板剥离,近窗域最近 m 轮保留);seed 消息不产生
        #    decision snapshot,故拿不到 loss(零模仿不变式,结构保证);
        # ③ 学生步数预算收缩为 max_steps − k。
        catalyst_entry_seed_pairs = None
        catalyst_entry_payload = getattr(
            traj_exp_config, "catalyst_entry_plan", None
        )
        if catalyst_entry_payload:
            from agentevolver.module.exp_manager.catalyst_entry import (
                EntryPlan,
                replay_teacher_prefix,
            )
            entry_plan = EntryPlan.from_payload(catalyst_entry_payload)
            catalyst_entry_seed_pairs, entry_divergence = (
                replay_teacher_prefix(
                    env, instance_id, entry_plan, self.tokenizer
                )
            )
            self.max_steps = int(self.max_steps) - int(entry_plan.k_steps)
            if self.max_steps < 1:
                raise RuntimeError(
                    f"[CATALYST] entry k={entry_plan.k_steps} leaves no "
                    "student budget; entry book / max_steps misconfigured"
                )
            # v5:若本槽同时带 hint(前面已注入)则为 rescue 复合槽
            self.cmt.metadata["catalyst_arm"] = (
                "rescue" if catalyst_hint_text else "entry"
            )
            self.cmt.metadata["catalyst_entry_rung"] = int(entry_plan.rung)
            self.cmt.metadata["catalyst_entry_frac"] = float(entry_plan.frac)
            self.cmt.metadata["catalyst_entry_k"] = int(entry_plan.k_steps)
            self.cmt.metadata["catalyst_entry_divergence"] = int(
                entry_divergence
            )
            # v3 课程 critic 基线(plan 时冻结,经 extras 透传给优势覆写)
            self.cmt.metadata["catalyst_entry_vhat"] = float(
                catalyst_entry_payload.get("vhat", 0.0) or 0.0
            )
            self.cmt.metadata["catalyst_entry_source"] = str(
                catalyst_entry_payload.get("source", "teacher")
            )
        if getattr(traj_exp_config, "catalyst_entry_degraded", False):
            # 重放失败后的降级重试:本 rollout 是干净的裸臂(新 env 实例),
            # 打标供治理端计数(catalyst/entry_degraded)。
            self.cmt.metadata["catalyst_entry_degraded"] = True
        # ⭐ CATALYST v4:所有臂(含裸)透传 plan 时冻结的 V̂(统一优势先验
        # 与在线校准探针共用;None=非 v4,零行为)。
        _v4_m = getattr(traj_exp_config, "catalyst_v4_m", None)
        if _v4_m is not None:
            self.cmt.metadata["catalyst_v4_m"] = float(_v4_m)
        # Qwen 3 suppressed thinking in history via the /no_think soft switch.
        # Qwen 3.5 dropped those switches, so appending it there is dead text that
        # also contradicts the generation prompt (which opens with '<think>'):
        # the model emits malformed turns and the env rejects the action.
        thinking_mode = resolve_thinking_mode(self.config)
        add_nothink = (thinking_mode == THINKING_MODE_NATIVE_QWEN3)

        # 1. 🚀 Initialize messages
        traj_exp_config.query = query
        init_messages, traj_exp_config = self.exp_worker.manage_rollout_context(
                init_messages=init_messages,
                traj_exp_config=traj_exp_config
                )
        self.cmt.metadata["task_train_mode"] = traj_exp_config.train_mode
        self.cmt.metadata["add_exp"] = traj_exp_config.add_exp
        self.cmt.metadata["experience_list"] = traj_exp_config.experience_list
        # init_messages, metadata = self.add_experience(init_messages, task_id, data_id, rollout_id, query, add_exp)  # ⭐ Initialize messages and metadata
        # self.cmt.metadata = metadata
        self.cmt.save_init_input(init_messages, add_nothink)
        # ⭐ CATALYST v2 entry 臂 ②:seed 已发生历史(与试点 _seeded_save_init
        # 逐字同构:author=llm/env、env 侧带 clip 上限;渲染由
        # StructuredContextPolicy 走正常军规链路)。seed 不经 save_llm_output,
        # 不产生 decision snapshot → 结构性拿不到 loss。
        if catalyst_entry_seed_pairs:
            for _assistant_content, _user_content in catalyst_entry_seed_pairs:
                self.cmt.full_context.append(
                    ExtendedMessage(
                        author="llm",
                        role="assistant",
                        content=str(_assistant_content),
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
                )
                self.cmt.full_context.append(
                    ExtendedMessage(
                        author="env",
                        role="user",
                        content=str(_user_content),
                        clip=True,
                        clip_token_limit=self.cmt.max_env_output_length,
                        token_generator="auto",
                        tokenizer=self.tokenizer,
                    )
                )

        request_id: str = ""
        err_in_generating=False
        err_in_env = False
        malformed_action = False
        length_truncation_terminated = False
        for act_step in range(self.max_steps):
            # 2. 🔄 Update thread progress
            tmux['step'][thread_index] = act_step
            if (stop is not None) and stop[thread_index]: # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                self.cmt.metadata["episode_end_reason"] = "external_stop_discarded"
                break

            # 3. ⏮️ get previous steps
            try:
                step_input_message_arr = self.cmt.prepare_next_llm_context()  # ⭐ Prepare the next LLM context
            except Exception as e:
                print_listofdict(self.cmt.to_role_content(self.cmt.full_context), mod='exception', header="Before Crash")
                raise e

            # 4. ⚠️ check token overflow
            is_safe: bool = self.cmt.check_context_token_num_safe(step_input_message_arr)  # ⭐ Check if the context token count is safe
            if not is_safe:
                logger.warning(f"Token overflow detected at step {act_step}. Current token count exceeds the limit.")
                self.cmt.is_terminated = False # trajectory not finished.
                self.cmt.metadata["episode_end_reason"] = "context_overflow"
                self.cmt.metadata["context_overflow_step"] = act_step
                self.cmt.metadata["context_overflow_prompt_tokens"] = len(
                    chat_template_ids(
                        self.tokenizer,
                        step_input_message_arr,
                        add_generation_prompt=True,
                    )
                )
                break

            # 5. 🤖 call llm
            llm_output = self.llm_chat_fn(step_input_message_arr, request_id=request_id)  # ⭐ Call the LLM to generate the next response
            if llm_output.get("role") != "assistant":
                raise RuntimeError(
                    f"Expected assistant response from llm_chat_fn, got role={llm_output.get('role')}"
                )
            if self._decision_observer is not None:
                self._decision_observer(
                    step_index=act_step,
                    prompt_messages=step_input_message_arr,
                    context_result=getattr(
                        self.cmt, "_pending_context_result", None
                    ),
                    llm_output=llm_output,
                )
            if (stop is not None) and stop[thread_index]:  # Check if the thread should stop (because other threads have completed, making this thread useless)
                self.cmt.discarded = True
                self.cmt.metadata["episode_end_reason"] = "external_stop_discarded"
                break

            # 6. 💾 save llm output
            self.cmt.save_llm_output(llm_output, input_msg_ref=step_input_message_arr)  # ⭐ Save the LLM output
            tmux['token'][thread_index] += self.cmt.generated_token_cnt

            # A max-token completion contains no model-authorized terminal
            # action boundary. Preserve the exact 10K sampled stream for
            # training/audit, but never execute partial text in the environment
            # or let a later turn turn that failed decision into positive
            # trajectory reward.
            finish_reason = llm_output.get("finish_reason")
            if finish_reason is not None:
                finish_reason = str(
                    getattr(finish_reason, "value", finish_reason)
                )
            if bool(llm_output.get("truncated_by_length", False)) or finish_reason == "length":
                length_truncation_terminated = True
                self.cmt.metadata["length_truncation_terminated"] = True
                self.cmt.metadata["length_truncation_step"] = act_step
                self.cmt.metadata["episode_end_reason"] = "length_truncation"
                self.cmt.is_terminated = True
                logger.warning(
                    "terminating trajectory at generation length limit: "
                    f"step={act_step}, sampled_tokens={len(llm_output.get('tokens') or [])}"
                )
                break

            # 7. 🌍 world interaction
            world_interaction = self.cmt.prepare_world_interaction()
            if isinstance(world_interaction, str) and not world_interaction.strip():
                # The sampled completion itself is non-empty (and is already
                # preserved above), but Qwen may close </think> without emitting
                # an environment-facing action.  This is a policy failure, not a
                # transport failure: retain it once as an on-policy negative
                # sample and never introduce survivor bias by resampling it.
                malformed_action = True
                self.cmt.metadata["malformed_action"] = True
                self.cmt.metadata["malformed_action_step"] = act_step
                self.cmt.metadata["episode_end_reason"] = "malformed_action"
                logger.warning(
                    "terminating trajectory for empty environment-facing action: "
                    f"step={act_step}"
                )
                state = {
                    "content": (
                        "Malformed action: the model produced no environment-facing "
                        "content after reasoning."
                    ),
                    "role": "user",
                }
                env_output = {
                    "reward": 0,
                    "is_terminated": True,
                    "state": state,
                }
            else:
                try:
                    env_output = env.step(instance_id, {"content": world_interaction, "role": "assistant"})  # ⭐ Interact with the environment
                    assert len(env_output['state'])==1
                    env_output["state"] = env_output["state"][0]
                    if env_output["state"]["role"] == "tool":
                        env_output["state"] = convert_tool_to_user_message(env_output["state"], self.tokenizer, format="qwen")
                    if self.console_debug_mode:
                        print_listofdict(
                            step_input_message_arr +
                            [{'role': 'llm_latest', 'content': llm_output['content']}] +
                            [{'role': 'env',        'content': env_output["state"]['content']}]
                        , mod='c')
                except Exception as e:
                    if _is_env_infrastructure_error(e):
                        raise
                    logger.bind(exception=True).exception(f"call env.step error with {e}")
                    err_in_env = True
                    self.cmt.is_terminated = False # trajectory not finished.
                    state = {"content": str(e), "role": "user"}
                    env_output = {
                        "reward": 0,
                        "is_terminated": True,
                        "state": state,
                    }

            # 8. 📥 save environment output
            state = env_output["state"]
            state.pop('tool_calls', None)
            self.cmt.save_env_output(state, input_msg_ref=step_input_message_arr, add_nothink=add_nothink)  # ⭐ Save the environment output

            # 9. 🔚 determine if the episode is terminated
            self.cmt.is_terminated = env_output["is_terminated"]
            if self.cmt.is_terminated or err_in_env or malformed_action:
                if malformed_action:
                    self.cmt.metadata["episode_end_reason"] = "malformed_action"
                else:
                    self.cmt.metadata["episode_end_reason"] = (
                        "env_error" if err_in_env else "env_terminated"
                    )
                break

        if "episode_end_reason" not in self.cmt.metadata:
            self.cmt.metadata["episode_end_reason"] = "max_steps"

        tmux['step'][thread_index] = -1

        if malformed_action:
            score = 0.0
            reason = (
                "Model produced no environment-facing action after reasoning; "
                "forced failure reward=0."
            )
        elif self._reward_calculator is not None:
            grader_res = self._reward_calculator.calculate_reward(self.cmt, env, instance_id)  # ⭐ Calculate the reward using the reward calculator
            score = grader_res["score"] 
            reason = grader_res["reason"] or "No reason provided."
        else:
            score = env.evaluate(instance_id, params={
                "sparse": self.sparse,
                "sciworld_success_threshold": self.sciworld_success_threshold,
            })  # ⭐ Evaluate the score from the environment
            reason = "Outcome 1 = success, 0 = failure."

        # ---------------------------
        # Success definition & SciWorld outcome override
        # ---------------------------
        # For SciWorld we compute outcome (score) AND success_rate from the
        # same env_info + threshold, so they are guaranteed consistent
        # regardless of which grader produced the initial `score` above.
        #   threshold=0  → EPO-style `won` (done AND score > 0)
        #   threshold=70 → agl-envs-style strict criterion
        success_rate = 0.0
        if not err_in_env and not malformed_action:
            try:
                env_info = env.get_tools_info(instance_id, messages={}, params={})
            except Exception:
                env_info = None

            env_id = env_info.get("env_id") if isinstance(env_info, dict) else None
            if env_id == "sciworld":
                done_flag = bool(env_info.get("done", self.cmt.is_terminated)) if isinstance(env_info, dict) else bool(self.cmt.is_terminated)
                raw_score_val = 0.0
                if isinstance(env_info, dict):
                    try:
                        raw_score_val = float(env_info.get("score", 0.0) or 0.0)
                    except Exception:
                        raw_score_val = 0.0
                is_success = done_flag and raw_score_val > self.sciworld_success_threshold
                success_rate = 1.0 if is_success else 0.0
                score = success_rate
            else:
                if isinstance(env_info, dict) and "success_rate" in env_info:
                    try:
                        success_rate = float(env_info.get("success_rate", 0.0) or 0.0)
                    except Exception:
                        success_rate = 1.0 if float(score) >= 1.0 else 0.0
                else:
                    success_rate = 1.0 if float(score) >= 1.0 else 0.0

        if length_truncation_terminated:
            score = float(
                self.config.actor_rollout_ref.rollout.get(
                    "length_truncation_reward", 0.0
                )
            )
            success_rate = 0.0
            reason = (
                "Generation reached the per-decision token limit before a "
                f"complete action; forced failure reward={score}."
            )

        if malformed_action:
            # Do not rely on a potentially stale environment score and do not
            # allow a custom reward patch to turn a malformed policy event into
            # success.
            score = 0.0
            success_rate = 0.0

        self.cmt.reward = Reward(outcome=score, success_rate=success_rate, madness=self.cmt.compute_madness(), description=reason)  # ⭐ Set the reward for the context
        self.cmt.reward = self.cmt.reward_patch(self.cmt.reward)
        if malformed_action:
            self.cmt.reward.outcome = 0.0
            self.cmt.reward.success_rate = 0.0
        self.cmt.remove_last_context()

        with log_generate_lock:
            self.cmt.generate_log(task_id=task_id)  # ⭐ Generate the log for the task


        return self.cmt
