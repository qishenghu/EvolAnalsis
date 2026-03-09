# WebShop 环境集成

## 概述

WebShop 是一个模拟在线购物的交互式环境。Agent 需要根据用户的购物指令，通过搜索、浏览、筛选商品，最终购买符合要求的商品。

## 环境特点

- **真实的购物场景**：模拟 Amazon 风格的电商网站
- **复杂的用户需求**：涉及颜色、尺寸、价格、功能等多种属性
- **丰富的商品库**：包含数千种商品

## 快速开始

### 1. 安装 AgentGym WebShop 服务器

```bash
cd AgentGym/agentenv-webshop
conda env create -n agentenv-webshop -f environment.yml
conda activate agentenv-webshop
bash ./setup.sh
```

### 2. 启动 WebShop 服务器

```bash
webshop --host 0.0.0.0 --port 36003
```

### 3. 启动环境服务

```bash
cd env_service/launch_script
bash webshop.sh
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `WEBSHOP_SERVER_URL` | `http://127.0.0.1:36003` | AgentGym WebShop 服务器地址 |

## 任务 ID 格式

- `task_id` 对应 `session_id`
- 支持两种格式：纯数字（如 `5238`）或带前缀格式（如 `webshop_5238`）
- 若未显式指定 `session_id`，则由 AgentGym WebShop server 随机采样任务

## 数据划分

- `train` 使用 `env_service/environments/webshop/webshop_train.json`
- `val` / `dev` / `test` 使用 `env_service/environments/webshop/webshop_test.json`
- 若本地缺少 `webshop_train.json`，环境会回退为从默认 `1000` 商品数据动态推导 train goal 索引，并排除 test task
- 如需覆盖默认 train 候选池，也可在 `get_query_list(..., params={"total_sessions": N})` 中显式传入 `total_sessions`

## 动作格式

WebShop 支持两种动作类型：

1. **搜索**：`search[query]`
   - 例：`search[red dress under 50 dollars]`

2. **点击**：`click[element]`
   - 例：`click[Buy Now]`
   - 例：`click[size: large]`

## 购物流程示例

1. 收到指令："Find me a red dress under $50"
2. `search[red dress]`
3. `click[B09XXXXX]`（点击商品）
4. `click[color: red]`（选择颜色）
5. `click[size: medium]`（选择尺寸）
6. `click[Buy Now]`（购买）

## API 参考

### `get_init_state(params)`

初始化环境，返回购物指令和初始页面。

### `step(action, params)`

执行动作，返回页面更新和奖励。

### `evaluate()`

返回购物任务完成度（0.0 - 1.0），基于：
- 是否购买了商品
- 商品与需求的匹配度
- 价格是否在预算内

## 注意事项

1. WebShop 服务器启动时间较长，需要加载商品数据库
2. 确保有足够的内存（建议 8GB+）
3. 首次启动可能需要下载数据
4. `env_service` 关闭实例时会调用 WebShop server 的删除接口，避免训练过程中远端 env 实例持续堆积

