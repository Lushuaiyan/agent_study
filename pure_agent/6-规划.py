from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal


# 定义需要的数据结构
class Step(BaseModel):
    step_id: int
    description: str    # 步骤的详细描述
    status: Literal["pending", "in_progress", "completed", "failed"]
    # 该步骤的状态: 待做的, 正在做的, 完成的, 失败的
    tool_name: str | None = None    # 可能需要使用到的工具
    dependencies: List[int] = []    # 依赖哪些上一步的step_id

class Plan(BaseModel):
    goal: str   # 最终目标
    steps: List[Step]   # 实现目标需要的步骤

# 编写提示词
PLANNING_SYSTEM_PROMPT = """
# 角色与任务
你是一个顶级的任务规划专家, 请将用户的目标拆解成具体的, 可执行的步骤
# 输出规范
你必须严格按照JSON Schema格式输出, 不要包含任何无关的格式与内容
Schema: {
    "goal": "目标",
    "steps": [
        {
            "step_id": 1,
            "description": "步骤的详细描述",
            "status": Literal["pending", "in_progress", "completed", "failed"],
            "tool_name": "需要使用的工具" 或 None,
            "dependencies": List[int] = []
        },
    ]
}
步骤之间如果有先后顺序, 在dependencies中指明需要依赖的step_id, 注意不要生成环形结构
"""

# 定义规划类
class Planner:
    def __init__(self, client: OpenAI, model: str = "deepseek-v4-pro", tools: list = []):
        self.client = client
        self.model = model
        self.tools = tools
        self.plan: Plan = None

    # 生成规划
    def create_plan(self, user_goal: str)->Plan:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_goal}
            ],
            stream=False,
            tools = self.tools,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}
        )
        json_str = resp.choices[0].message.content
        # 解析成Plan对象
        plan = Plan.model_validate_json(json_str)
        return plan
    
    # 找到下一步需要做的步骤
    def get_next_actionable_step(self)-> Step | None:
        if self.paln is None:
            return None
        
        # 生成step_id -> Step的映射, 方便查询依赖步骤的状态
        step_map = {step.step_id for step in self.plan.steps}

        for step in self.plan.steps:
            if step.status != "pending":
                continue
            # 检查所有依赖项是否都完成了
            all_deps_completed = all(
                dep_id in step_map and step_map[dep_id].status == "completed"
                for dep_id in step.dependencies
            )
            if all_deps_completed:
                return step
        return None
    
    # 每个step的执行流程, 并返回观察结果
    def execute_step(self, step: Step, context: str)->str:
        pass
    # 最终的调用流程
    def run(self, goal: str):
        self.plan = self.create_plan(goal)
        while True:
            step = self.get_next_actionable_step()
            if not step:
                break
            step.status = "in_progress"
            observation = self.execute_step(step, self.get_context_for_step(step))
            step.status = "completed" if "成功" in observation else "failed"
            if step.status == "failed":
                # 调用LLM重新规划, 生成剩余步骤
                self.plan = self.replan(goal, self.plan, step, observation)
            
        return self.summarize_results()
