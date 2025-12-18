"""
Memory使用示例

展示如何在AgentGym中使用Memory模块进行经验存储和检索
"""
from agentenv.controller import APIAgent, AlfWorldTask
from agentenv.controller.memory import RawMemory, NullMemory


def example_basic_usage():
    """基本使用示例"""
    # 1. 创建Memory（从空的经验池开始）
    memory = RawMemory(
        local_exp_pool_path="",  # 空路径表示从空开始
        collection_name="exp_pool",
        success_only=True,  # 只存储成功的经验
        k_retrieval=1,  # 检索1个最相似的经验
    )
    
    # 2. 创建Task，传入Memory
    task = AlfWorldTask(
        client_args={
            "env_server_base": "http://localhost:3000",
            "data_len": 100,
        },
        memory=memory,  # 传入Memory
    )
    
    # 3. 创建Agent
    agent = APIAgent(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",
        model="qwen-plus",
    )
    
    # 4. 使用Memory支持的方法生成experience
    experiences = task.generate_experience_with_memory(
        agent=agent,
        idxs=[0, 1, 2],  # 任务索引
    )
    
    print(f"Generated {len(experiences)} experiences")
    for i, exp in enumerate(experiences):
        print(f"Experience {i}: reward={exp.reward}")


def example_with_existing_pool():
    """使用已有经验池的示例"""
    # 从已有的jsonl文件加载经验池
    memory = RawMemory(
        local_exp_pool_path="./exp_pool_snapshot.jsonl",
        success_only=True,
        k_retrieval=2,  # 检索2个最相似的经验
    )
    
    task = AlfWorldTask(
        client_args={"env_server_base": "http://localhost:3000", "data_len": 100},
        memory=memory,
    )
    
    agent = APIAgent(api_key="your-api-key", base_url="http://localhost:8000/v1", model="qwen-plus")
    
    # 生成experience时会自动检索相似经验
    experiences = task.generate_experience_with_memory(agent=agent, idxs=[10, 11, 12])


def example_evaluation_mode():
    """评估模式：只检索，不存储"""
    # 加载已有经验池
    memory = RawMemory(
        local_exp_pool_path="./exp_pool_trained.jsonl",
        success_only=True,
        k_retrieval=1,
    )
    
    # 禁用存储（评估时不需要存储新经验）
    memory.disable_storage()
    
    task = AlfWorldTask(
        client_args={"env_server_base": "http://localhost:3000", "data_len": 100},
        memory=memory,
    )
    
    agent = APIAgent(api_key="your-api-key", base_url="http://localhost:8000/v1", model="qwen-plus")
    
    # 评估测试集
    test_indices = list(range(50, 100))
    results = task.generate_experience_with_memory(agent=agent, idxs=test_indices)
    
    # 计算成功率
    success_count = sum(1 for exp in results if exp.reward > 0)
    success_rate = success_count / len(results)
    print(f"Success rate: {success_rate:.2%}")


def example_training_loop():
    """训练循环示例"""
    # 创建Memory
    memory = RawMemory(
        local_exp_pool_path="",  # 从空开始
        success_only=True,
        k_retrieval=1,
    )
    
    task = AlfWorldTask(
        client_args={"env_server_base": "http://localhost:3000", "data_len": 100},
        memory=memory,
    )
    
    agent = APIAgent(api_key="your-api-key", base_url="http://localhost:8000/v1", model="qwen-plus")
    
    # 训练循环
    train_indices = list(range(50))
    for step, idx in enumerate(train_indices):
        experience = task.generate_experience_with_memory(agent=agent, idxs=idx)
        
        # 每10步保存一次经验池
        if (step + 1) % 10 == 0:
            memory.persist(f"./exp_pool_step_{step+1}.jsonl")
            print(f"Saved experience pool at step {step+1}")


def example_baseline():
    """Baseline示例：不使用Memory"""
    # 使用NullMemory（或不传memory参数）
    task = AlfWorldTask(
        client_args={"env_server_base": "http://localhost:3000", "data_len": 100},
        memory=NullMemory(),  # 或者不传memory参数
    )
    
    agent = APIAgent(api_key="your-api-key", base_url="http://localhost:8000/v1", model="qwen-plus")
    
    # 使用原有方法（不使用Memory）
    experiences = task.generate_experience(agent=agent, idxs=[0, 1, 2])
    
    # 或者使用Memory方法但传入NullMemory（效果相同）
    experiences = task.generate_experience_with_memory(agent=agent, idxs=[0, 1, 2])


if __name__ == "__main__":
    print("Memory使用示例")
    print("=" * 50)
    
    # 运行基本示例
    # example_basic_usage()
    
    # 运行其他示例
    # example_with_existing_pool()
    # example_evaluation_mode()
    # example_training_loop()
    # example_baseline()

