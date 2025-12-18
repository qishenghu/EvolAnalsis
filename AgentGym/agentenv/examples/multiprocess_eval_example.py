"""
多进程评估示例

展示如何使用Evaluator的多进程功能进行并行评估
"""
from agentenv.controller import Evaluator, APIAgent, AlfWorldTask
from agentenv.controller.memory import RawMemory
from transformers import GenerationConfig


def example_basic_multiprocess_eval():
    """基本多进程评估示例"""
    # 创建Agent和Task
    agent = APIAgent(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",
        model="qwen-plus",
    )
    
    task = AlfWorldTask(
        client_args={
            "env_server_base": "http://localhost:3000",
            "data_len": 100,
        },
    )
    
    evaluator = Evaluator(agent=agent, tasks=[task])
    
    # 多进程评估
    test_indices = list(range(50, 100))
    result = evaluator.eval_multiprocess(
        idxs=test_indices,
        max_rounds=30,
        num_processes=4,  # 使用4个进程
    )
    
    print(f"Score: {result.score}")
    print(f"Success rate: {result.success}")
    print(f"Total experiences: {len(result.experiences)}")


def example_multiprocess_with_memory():
    """使用Memory的多进程评估示例"""
    # 创建Memory
    memory = RawMemory(
        local_exp_pool_path="./exp_pool.jsonl",  # 加载已有经验池
        success_only=True,
        k_retrieval=1,
    )
    
    # 创建Agent和Task（传入Memory）
    agent = APIAgent(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",
        model="qwen-plus",
    )
    
    task = AlfWorldTask(
        client_args={
            "env_server_base": "http://localhost:3000",
            "data_len": 100,
        },
        memory=memory,
    )
    
    evaluator = Evaluator(agent=agent, tasks=[task])
    
    # 多进程评估（使用Memory，但不存储新经验）
    test_indices = list(range(50, 100))
    result = evaluator.eval_multiprocess(
        idxs=test_indices,
        max_rounds=30,
        use_memory=True,  # 使用Memory检索
        enable_memory_storage=False,  # 评估时不存储
        num_processes=4,
    )
    
    print(f"Score: {result.score}")
    print(f"Success rate: {result.success}")


def example_multiprocess_generate_experience():
    """多进程生成experience示例"""
    agent = APIAgent(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",
        model="qwen-plus",
    )
    
    task = AlfWorldTask(
        client_args={
            "env_server_base": "http://localhost:3000",
            "data_len": 100,
        },
    )
    
    evaluator = Evaluator(agent=agent, tasks=[task])
    
    # 多进程生成experience
    train_indices = list(range(50))
    experiences = evaluator.generate_experience_multiprocess(
        idxs=train_indices,
        max_rounds=30,
        num_processes=8,  # 使用8个进程加速
    )
    
    print(f"Generated {len(experiences)} experiences")
    success_count = sum(1 for exp in experiences if exp.reward > 0)
    print(f"Success count: {success_count}")


def example_compare_single_vs_multiprocess():
    """对比单进程和多进程性能"""
    import time
    
    agent = APIAgent(
        api_key="your-api-key",
        base_url="http://localhost:8000/v1",
        model="qwen-plus",
    )
    
    task = AlfWorldTask(
        client_args={
            "env_server_base": "http://localhost:3000",
            "data_len": 100,
        },
    )
    
    evaluator = Evaluator(agent=agent, tasks=[task])
    test_indices = list(range(20))  # 测试20个任务
    
    # 单进程
    start_time = time.time()
    result_single = evaluator.eval(idxs=test_indices, max_rounds=30)
    single_time = time.time() - start_time
    
    # 多进程（4个进程）
    start_time = time.time()
    result_multi = evaluator.eval_multiprocess(
        idxs=test_indices,
        max_rounds=30,
        num_processes=4,
    )
    multi_time = time.time() - start_time
    
    print(f"Single process: {single_time:.2f}s, Score: {result_single.score:.4f}")
    print(f"Multi process (4): {multi_time:.2f}s, Score: {result_multi.score:.4f}")
    print(f"Speedup: {single_time / multi_time:.2f}x")


if __name__ == "__main__":
    print("多进程评估示例")
    print("=" * 50)
    
    # 运行示例
    # example_basic_multiprocess_eval()
    # example_multiprocess_with_memory()
    # example_multiprocess_generate_experience()
    # example_compare_single_vs_multiprocess()

