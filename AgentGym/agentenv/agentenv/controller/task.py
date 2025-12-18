from typing import Any, Callable, Mapping, Optional, Sequence

from transformers import GenerationConfig

from . import Agent, APIAgent, BaseEnvClient
from .types import ConversationMessage, APIConversationMessage, ExperienceOutput, APIExperienceOutput
from .memory import BaseMemory, NullMemory
import copy

class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
        is_eval: bool = True,
        test_data_start_index: int = 2420,
        memory: Optional[BaseMemory] = None,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
            memory (Optional[BaseMemory]): Optional memory module for experience storage and retrieval.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        assert 'is_eval' in client_args, "is_eval must be provided in client_args"
        assert 'test_data_start_index' in client_args, "test_data_start_index must be provided in client_args"
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])
        self.memory = memory if memory is not None else NullMemory()

    def _generate_experience_one(
        self,
        agent: Agent | APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        if isinstance(agent, Agent):
            tokenizer = agent.tokenizer
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            conversation_tokenized = agent.chat_template.tokenize_conversation(
                conversation, tokenizer, add_generation_prompt=True
            )
        elif isinstance(agent, APIAgent):
            conversation = [APIConversationMessage({"role": "user", "content": client.conversation_start[0]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "assistant", "content": client.conversation_start[1]["value"], "reasoning_content": None}),
                            APIConversationMessage({"role": "user", "content": state, "reasoning_content": None})]
        else:
            raise NotImplementedError
        rounds = 0

        while not done:
            if isinstance(agent, Agent):
                input_length = len(conversation_tokenized["input_ids"])
                # if input_length exceeds max_length, break
                if input_length >= (generation_config.max_length or 4096):
                    break
                try:
                    generated_tokens = agent.generate(
                        [conversation_tokenized["input_ids"]], generation_config
                    )[0]
                except Exception as e:  # pylint: disable=W0718:broad-exception-caught
                    print(e)
                    break  # break if generate method raises exceptions

                if generated_tokens[-1] != tokenizer.eos_token_id:
                    generated_tokens += [tokenizer.eos_token_id]

                generated_text = tokenizer.decode(generated_tokens)
                conversation_tokenized["text"] += f" {generated_text}"
                conversation_tokenized["input_ids"] += generated_tokens
                conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

                generated_text = generated_text[
                    : -len(tokenizer.eos_token)
                ]  # not endswith eos_token
                conversation.append(
                    ConversationMessage(
                        {"from": "gpt", "loss": True, "value": generated_text}
                    )
                )
            elif isinstance(agent, APIAgent):
                ### Use `state = client.observe()` to replace the latest state message in conversation
                ### Because client.observe() will provide the latest available actions
                ### If we only use the step_output.state as the last env observation, it does not include available actions
                cur_state = client.observe()
                temp_conversation = conversation.copy()
                assert temp_conversation[-1]['role'] == 'user', "The latest message in conversation must be a user message"
                temp_conversation[-1]['content'] = cur_state # Replace the latest state message in conversation
                print(f"Observation: {cur_state}")


                generated_text, generated_reasoning_text = agent.generate(temp_conversation, 
                                                                        temperature=generation_config.temperature if generation_config is not None else None
                                                                        )
                conversation.append(
                    APIConversationMessage(
                        {"role": "assistant", "content": generated_text, "reasoning_content": generated_reasoning_text}
                    )
                )
            else:
                raise NotImplementedError

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            if isinstance(agent, Agent):
                env_message = ConversationMessage(
                    {"from": "human", "loss": None, "value": state}
                )
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message, tokenizer, add_generation_prompt=True
                )

                conversation.append(env_message)
                conversation_tokenized["text"] += env_message_tokenized["text"]
                conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
                conversation_tokenized["action_mask"] += env_message_tokenized[
                    "action_mask"
                ]
            elif isinstance(agent, APIAgent):
                conversation.append(
                    APIConversationMessage(
                        {"role": "user", "content": state, "reasoning_content": None}
                    )
                )
            else:
                raise NotImplementedError

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        if isinstance(agent, Agent):
            return ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text=conversation_tokenized["text"],
                seq_ids=conversation_tokenized["input_ids"],
                attention_mask=[1] * len(conversation_tokenized["input_ids"]),
                action_mask=conversation_tokenized["action_mask"],
            )
        elif isinstance(agent, APIAgent):
            return APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
        else:
            raise NotImplementedError

    def _generate_experience_batch(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        client = self.clients[0]
        result = [
            self._generate_experience_one(
                agent=agent,
                client=client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
            )
            for idx in idxs
        ]
        return result

    def generate_experience(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        return self._generate_experience_batch(
            agent=agent,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
    
    def _extract_task_query(self, client: BaseEnvClient, initial_state: str) -> str:
        """
        从client和initial_state中提取task query
        
        Args:
            client: 环境客户端
            initial_state: 初始状态字符串
            
        Returns:
            任务查询字符串
        """
        # 尝试从conversation_start中提取任务描述
        # if hasattr(client, 'conversation_start') and client.conversation_start:
        #     first_msg = client.conversation_start[0]
        #     if isinstance(first_msg, dict):
        #         task_desc = first_msg.get("value", "")
        #     else:
        #         task_desc = getattr(first_msg, "value", "")
        #     # 如果conversation_start包含任务描述，使用它
        #     if task_desc and len(task_desc) > 50:  # 假设任务描述较长
        #         return task_desc
        
        # 否则使用initial_state作为task query
        return initial_state
    
    def _inject_memories_to_conversation(
        self,
        conversation: list,
        retrieved_memories: list,
        is_api_agent: bool = False
    ) -> list:
        """
        将检索到的记忆注入到conversation中
        
        记忆会被插入在第一个实际任务消息（state）之前，这样agent可以在看到任务状态之前先看到相关经验。
        
        Args:
            conversation: 原始conversation列表
            retrieved_memories: 检索到的MemoryItem列表
            is_api_agent: 是否为APIAgent
            
        Returns:
            注入记忆后的conversation
        """
        if not retrieved_memories:
            return conversation
        
        # 构建记忆字符串
        memory_str = "\n\n---\n\n".join([item.content for item in retrieved_memories])
        memory_prompt = (
                "Below are some experiences that may be relevant:\n\n"
                f"{memory_str}\n\n"
                "---\n\n"
                "You may draw on these experiences as references where appropriate. Please proceed to address the following task:\n"
            )
        print(f"[Inject Memories to Conversation] Memory Prompt: {memory_prompt}")
        
        # 找到第一个实际任务消息的位置
        # conversation_start通常有2条消息（system prompt和assistant确认）
        # 第一个实际任务消息通常是第3条（index 2），也就是state消息
        # 我们想在state之前插入记忆，所以插入位置是最后一个conversation_start消息之后
        
        if is_api_agent:
            # APIAgent: [system, assistant, state, ...]
            # 插入在state之前，即index 2
            insert_idx = 2
            memory_msg = APIConversationMessage({
                "role": "user",
                "content": memory_prompt,
                "reasoning_content": None
            })
        else:
            # Agent: [system, assistant, state, ...]
            # 插入在state之前，即index 2
            insert_idx = 2 if len(conversation) > 2 else len(conversation)
            memory_msg = ConversationMessage({
                "from": "human",
                "loss": None,
                "value": memory_prompt
            })
        
        conversation.insert(insert_idx, memory_msg)
        return conversation
    
    def _generate_experience_one_with_memory(
        self,
        agent: Agent | APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        enable_memory_storage: bool = True,
    ) -> ExperienceOutput | APIExperienceOutput:
        """
        生成experience，支持Memory的存储和检索
        
        这个方法与_generate_experience_one类似，但添加了：
        1. 在开始前检索相似经验
        2. 将检索到的经验注入conversation
        3. 生成后存储经验到Memory
        
        Args:
            agent: Agent或APIAgent实例
            client: 环境客户端
            idx: 任务索引
            generation_config: 生成配置
            max_rounds: 最大轮数
            
        Returns:
            ExperienceOutput或APIExperienceOutput
        """
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        
        # 1. 提取task query用于检索
        task_query = self._extract_task_query(client, state)
        
        # 2. 检索相似经验
        retrieved_memories = []
        if self.memory is not None:
            if enable_memory_storage:
                self.memory.enable_storage()
            else:
                self.memory.disable_storage()

            retrieved_memories = self.memory.retrieve(task_query, k=self.memory.k_retrieval if hasattr(self.memory, 'k_retrieval') else 3)
            if retrieved_memories:
                print(f"[BaseTask] Retrieved {len(retrieved_memories)} memories for task {idx}")
        
        # 3. 构建conversation
        is_api_agent = isinstance(agent, APIAgent)
        if isinstance(agent, Agent):
            tokenizer = agent.tokenizer
            conversation = list(client.conversation_start)
            conversation.append(
                ConversationMessage({"from": "human", "loss": None, "value": state})
            )
            # 注入记忆
            if len(retrieved_memories) > 0:
                conversation = self._inject_memories_to_conversation(
                    conversation, retrieved_memories, is_api_agent=False
                )
            conversation_tokenized = agent.chat_template.tokenize_conversation(
                conversation, tokenizer, add_generation_prompt=True
            )
        elif isinstance(agent, APIAgent):
            conversation = [
                APIConversationMessage({
                    "role": "user",
                    "content": client.conversation_start[0]["value"],
                    "reasoning_content": None
                }),
                APIConversationMessage({
                    "role": "assistant",
                    "content": client.conversation_start[1]["value"],
                    "reasoning_content": None
                }),
                APIConversationMessage({
                    "role": "user",
                    "content": state,
                    "reasoning_content": None
                })
            ]
            # 注入记忆
            if len(retrieved_memories) > 0:
                conversation = self._inject_memories_to_conversation(
                    conversation, retrieved_memories, is_api_agent=True
                )
        else:
            raise NotImplementedError
        
        rounds = 0
        
        # 4. 执行任务循环（与原始方法相同）
        while not done:
            if isinstance(agent, Agent):
                input_length = len(conversation_tokenized["input_ids"])
                if input_length >= (generation_config.max_length or 4096):
                    break
                try:
                    generated_tokens = agent.generate(
                        [conversation_tokenized["input_ids"]], generation_config
                    )[0]
                except Exception as e:
                    print(e)
                    break
                
                if generated_tokens[-1] != tokenizer.eos_token_id:
                    generated_tokens += [tokenizer.eos_token_id]
                
                generated_text = tokenizer.decode(generated_tokens)
                conversation_tokenized["text"] += f" {generated_text}"
                conversation_tokenized["input_ids"] += generated_tokens
                conversation_tokenized["action_mask"] += [1] * len(generated_tokens)
                
                generated_text = generated_text[: -len(tokenizer.eos_token)]
                conversation.append(
                    ConversationMessage(
                        {"from": "gpt", "loss": True, "value": generated_text}
                    )
                )
            elif isinstance(agent, APIAgent):
                cur_state = client.observe()
                # deep copy
                temp_conversation = copy.deepcopy(conversation)
                assert temp_conversation[-1]['role'] == 'user', "The latest message in conversation must be a user message"
                temp_conversation[-1]['content'] = cur_state # Replace the latest state message in conversation
                print(f"[Observation]: {cur_state}")

                generated_text, generated_reasoning_text = agent.generate(temp_conversation, 
                                                                        temperature=generation_config.temperature if generation_config is not None else None
                                                                        )
                del temp_conversation

                conversation.append(
                    APIConversationMessage({
                        "role": "assistant",
                        "content": generated_text,
                        "reasoning_content": generated_reasoning_text
                    })
                )
            else:
                raise NotImplementedError
            
            print(f"[Generated Text]: {generated_text}")
            
            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )

            
            if isinstance(agent, Agent):
                env_message = ConversationMessage({
                    "from": "human",
                    "loss": None,
                    "value": state
                })
                env_message_tokenized = agent.chat_template.tokenize_conversation_one(
                    env_message, tokenizer, add_generation_prompt=True
                )
                
                conversation.append(env_message)
                conversation_tokenized["text"] += env_message_tokenized["text"]
                conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
                conversation_tokenized["action_mask"] += env_message_tokenized["action_mask"]
            elif isinstance(agent, APIAgent):
                conversation.append(
                    APIConversationMessage({
                        "role": "user",
                        "content": state,
                        "reasoning_content": None
                    })
                )
            else:
                raise NotImplementedError
            
            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break
        
        # 5. 创建experience output
        if isinstance(agent, Agent):
            experience_output = ExperienceOutput(
                conversation=conversation,
                reward=reward,
                text=conversation_tokenized["text"],
                seq_ids=conversation_tokenized["input_ids"],
                attention_mask=[1] * len(conversation_tokenized["input_ids"]),
                action_mask=conversation_tokenized["action_mask"],
            )
        elif isinstance(agent, APIAgent):
            experience_output = APIExperienceOutput(
                conversation=conversation,
                reward=reward,
            )
        else:
            raise NotImplementedError
        
        # 6. 存储经验到Memory
        if self.memory is not None and self.memory._storage_enabled:
            self.memory.store(experience_output, task_query, idx)
        
        return experience_output
    
    def generate_experience_with_memory(
        self,
        agent: Agent | APIAgent,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        enable_memory_storage: bool = True,
    ) -> list[ExperienceOutput | APIExperienceOutput]:
        """
        生成experience，使用Memory支持版本
        
        Args:
            agent: Agent或APIAgent实例
            idxs: 任务索引或索引列表
            generation_config: 生成配置
            max_rounds: 最大轮数
            
        Returns:
            ExperienceOutput或APIExperienceOutput列表
        """
        if isinstance(idxs, int):
            idxs = [idxs]
        
        client = self.clients[0]
        result = [
            self._generate_experience_one_with_memory(
                agent=agent,
                client=client,
                idx=idx,
                generation_config=generation_config,
                max_rounds=max_rounds,
                enable_memory_storage=enable_memory_storage,
            )
            for idx in idxs
        ]
        return result
