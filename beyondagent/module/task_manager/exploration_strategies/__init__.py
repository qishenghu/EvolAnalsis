import abc
import copy
import time
from typing import Callable, NotRequired, Optional, Sequence, TypedDict, Unpack

from loguru import logger

from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager.base import LlmClient, TaskObjectiveRetrieval
from beyondagent.module.task_manager.explorer import Explorer
from beyondagent.module.task_manager.prompts.prompt_explore import get_agent_interaction_system_prompt
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory


class ExploreStrategy(abc.ABC):
    """The abstract class of exploration strategy used in Task Manager for task generation.
    
    It provides nescessary contexts.
    """
    def _inject_deps(self,old_retrival: TaskObjectiveRetrieval,llm_client: LlmClient):
        self._old_retrival = old_retrival
        # TODO: where should I init the llm client
        self._llm_client=llm_client
    
    @property
    def llm_client(self):
        if not hasattr(self,"_llm_client"):
            raise AttributeError("llm_client is not injected")
        return self._llm_client
    
    @property
    def old_retrival(self) -> TaskObjectiveRetrieval:
        if not hasattr(self, "_old_retrival"):
            raise AttributeError("old_retrival is not injected")
        return self._old_retrival
    
    @abc.abstractmethod
    def explore(
        self, task: Task, data_id: str, rollout_id: str
    ) -> list[Trajectory]:
        """Explore the env.
        """
        pass


class LlmRandomSamplingExploreStrategyProps(TypedDict):
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    

class LlmRandomSamplingExploreStrategy(ExploreStrategy):
    def __init__(self, env_service_url: str,max_llm_retries: int, max_explore_step: int,* , tokenizer, config,**kwargs: Unpack[LlmRandomSamplingExploreStrategyProps]):
        self._max_llm_retries = max_llm_retries
        self._max_explore_step = max_explore_step
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer
        self._config = config
        
        self._exploration_llm_temperature=kwargs.get("exploration_llm_temperature", 1.0)
        self._exploration_llm_top_p=kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k=kwargs.get("exploration_llm_top_k", 1)
        
    
    def explore(self, task: Task, data_id: str, rollout_id: str) -> list[Trajectory]:
        env_worker = Explorer(
            env_type=task.env_type,
            task_id=task.task_id,
            instance_id=None,
            env_service_url=self._env_service_url,
        )
        llm_chat_fn = self._get_llm_chat_fn(
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        agent_flow: BaseAgentFlow = AgentFlow(
            enable_context_generator=False,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # TODO(cc): this is ugly

        old_objectives = self.old_retrival.retrieve_objectives(task)

        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            system_prompt=get_agent_interaction_system_prompt(task, old_objectives),
            agent_flow=agent_flow,
        )

        return [traj]
    
    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res = None
            for i in range(self._max_llm_retries):
                try:
                    res = self.llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            assert res is not None, f"LLM client failed to chat"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat