import json
from typing import Optional, Sequence, Tuple

from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory


AGENT_SUMMARIZE_SYSTEM_PROMPT = """
You are a *Real-World Task Discovery Expert*. Your specialty is to analyze an agent's API exploration history and discover realistic, user-centered problems that could be solved using the same interaction patterns.

========================  YOUR JOB  ========================
1. Analyze the agent's interaction sequence to understand what capabilities were discovered.
2. **Think like a real user**: What practical, everyday problems would naturally require these exact capabilities?
3. Abstract each discovered capability pattern into a **realistic user scenario**, a **natural user query**, and the **action sequence** that solves it.

=====================  ABSTRACTION RULES  ==================
**Focus on User Intent, Not Technical Exploration:**
- Transform technical API exploration into genuine user needs
- Ask: "What real-world problem would make someone naturally use these APIs in this order?"
- Prioritize common, relatable scenarios over edge cases

**User-Centered Thinking:**
- Generate queries that sound like something a real person would ask
- Focus on outcomes and goals, not API mechanics
- Use natural language that reflects user intent, not technical documentation

**Specificity and Verifiability:**
- Every query must have clear, measurable success criteria
- Include specific details: amounts, dates, names, quantities, thresholds, etc.
- The query should be precise enough that someone can definitively judge if it was answered correctly
- Avoid vague terms like "check", "review", "ensure" without specific targets

**Practical Scenarios:**
- Every task should solve a concrete problem someone might actually face
- The solution should feel intuitive and goal-oriented
- Avoid purely exploratory or informational queries unless they serve a clear purpose

**Technical Accuracy:**
- The action sequence must still be technically correct and executable
- All actions must be available in the current environment
- Maintain the minimal, complete sequence from the original exploration

========================  OUTPUT FORMAT  ===================
For every realistic task you identify, output exactly one block:

<task>
{
  "query": "[A natural user request that someone would actually make]",
  "confidence": "[0.0 - 1.0, your confidence this represents a real user need]",
  "action_sequence": "[The minimal technical sequence that accomplishes the user's goal]"
}
</task>

===========================  EXAMPLES  ======================

**POOR (Vague/Unverifiable):**
```
"query": "I need to check my Venmo balance to make sure I have enough funds for the weekend's grocery shopping"
```

**GOOD (Specific/Verifiable):**
```
"query": "Do I have at least $150 in my Venmo account for this weekend's grocery shopping?"
```

---

**POOR (Unclear Success Criteria):**
```
"query": "I need to review my work meetings and presentations to prepare for the upcoming team conference"
```

**GOOD (Clear Target):**
```
"query": "Show me all my meetings and presentations from the past month that mentioned 'Q4 budget' or 'financial review'"
```

---

**POOR (Missing Details):**
```
"query": "I need to check my credit card details to make an online payment"
```

**GOOD (Specific Information Need):**
```
"query": "What is my available credit limit and the full card number for my main credit card?"
```

========================  EXAMPLE OUTPUT  ===================

EXAMPLE 1
<task>
{
  "query": "What is the exact admin password for the deployment server account named 'supervisor'?",
  "confidence": 0.9,
  "action_sequence": "# step0\nprint(apis.api_docs.show_app_descriptions())\n# step1\nprint(apis.api_docs.show_api_descriptions(app_name='supervisor'))\n# step2\nprint(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))\n# step3\nprint(apis.supervisor.show_account_passwords())\npasswords = apis.supervisor.show_account_passwords()"
}
</task>

EXAMPLE 2
<task>
{
  "query": "Are the files 'config.yml', 'database.conf', and 'security.key' still present in the /home/admin directory?",
  "confidence": 1.0,
  "action_sequence": "# step0\ncd /home/admin\n # step1\nls ."
}
</task>

EXAMPLE 3
<task>
{
  "query": "Find red women's heels under $100 that can be delivered by next Friday",
  "confidence": 1.0,
  "action_sequence": "# step0\n[click('https://www.taobao.com')]\n # step1\n[search('red shoes')]"
}
</task>

========================  KEY PRINCIPLES  ===================
1. **Human-First**: Always start with "What would make a real person do this?"
2. **Context Matters**: Provide realistic scenarios that justify the action sequence
3. **Precise Language**: Use specific, measurable terms instead of vague descriptors
4. **Verifiable Outcomes**: Every query should have clear success/failure criteria
5. **Concrete Details**: Include numbers, names, dates, amounts, thresholds, or other specific requirements
6. **Actionable Clarity**: Someone should be able to definitively judge if the query was answered correctly

========================  SPECIFICITY CHECKLIST  =============
Before finalizing each query, ask:
- ✅ **What exactly** does the user want to know/achieve?
- ✅ **How much, how many, which ones** - are quantities/targets specified?
- ✅ **When, where, who** - are relevant constraints included?
- ✅ **How would I know** if this query was successfully answered?
- ✅ **Can someone else** judge if the response is correct without additional context?
"""


def _get_action_observation_pair(traj: Trajectory) -> list[tuple[str, str]]:
    res = []
    for idx, step in enumerate(traj.steps):
        assert "role" in step, "steps must have role field"
        if step["role"] == "assistant" and idx + 1 < len(traj.steps):
            next_step = traj.steps[idx + 1]
            # As there is no standard for environments, we do not know whether it will response as user or tool.
            if next_step["role"] == "tool":
                # get observation from tool message
                observation = next_step["content"]
            elif next_step["role"] == "user":
                # get observation from user message
                observation = next_step["content"]
            else:
                continue
            res.append((step["content"], observation))

    return res


def get_task_summarize_prompt(
    trajectories: Sequence[Trajectory],
    old_objectives: Sequence[TaskObjective],
    len_history: int = 2,
) -> tuple[str, str]:
    """获取任务摘要 prompt"""
    x = ""
    idx = 0
    for traj in trajectories:
        pairs = _get_action_observation_pair(traj)
        for k, v in enumerate(pairs):
            histories = pairs[max(0, k - len_history) : k]
            x += f"## Record {idx}\n"
            x += f"### History\n"
            for history in histories:
                x += f"{history[0]}\n->\n{history[1]}\n\n"
            x += f"### Action\n{v[0]}\n"
            x += f"### Observation\n{v[1]}\n"
            x += f"### Reward: {traj.reward.outcome}\n{traj.reward.description}\n"
            idx += 1

    objectives: list[str] = []
    for ob in old_objectives:
        if isinstance(ob.objective, str):
            objectives.append(ob.objective)
        else:
            objectives.extend(ob.objective)

    user_prompt = f"""Please analyze the following agent interaction sequence and abstract specific tasks from it:

{x}

# Old Objectives
You have already explored the following objectives:

{objectives}

Please avoid repeating these objectives.

# Now Start

Please identify the specific tasks the agent is attempting to complete in these interactions, and abstract them into clear task descriptions and queries following the specified format.
"""

    return AGENT_SUMMARIZE_SYSTEM_PROMPT, user_prompt


def parse_tasks_from_response(task: Task, response: str) -> list[TaskObjective]:
    """从响应中解析任务列表"""
    task = task.copy()

    tasks: list[TaskObjective] = []
    try:
        import re

        # 找到所有<task>标签中的内容
        task_matches = re.findall(r"<task>(.*?)</task>", response, re.DOTALL)

        for task_content in task_matches:
            t = json.loads(task_content)

            # 检查必要字段
            if (
                "query" not in t
                or "confidence" not in t
                or "action_sequence" not in t
            ):
                continue
            task.query = t["query"]
            tasks.append(
                TaskObjective(
                    task=task,
                    confidence=t["confidence"],
                    ground_truth=t["action_sequence"],
                    reward=None,
                )
            )

    except Exception as e:
        print(f"Error parsing tasks: {e}")

    return tasks
