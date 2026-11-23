"""submit 工具：模型申请结束任务。

关键设计（模型与确定性代码的边界）：
调用 submit 不等于任务成功！AgentLoop 检测到 submit 后会进入 VERIFYING 状态，
由确定性 Verifier（如 pytest）做最终裁决。测试不通过则打回继续修复。
"""

from __future__ import annotations

from pydantic import Field

from mca.tools.base import Tool, ToolContext, ToolParams


class SubmitParams(ToolParams):
    summary: str = Field(min_length=1, description="简要说明定位到的根因与所做的修改")


class SubmitTool(Tool):
    name = "submit"
    description = "当你认为所有问题已经修复且测试通过时调用此工具提交结果。系统随后会运行确定性验证。"
    params_model = SubmitParams
    readonly = False

    async def run(self, params: SubmitParams, ctx: ToolContext) -> str:
        return f"已收到提交申请（摘要: {params.summary[:200]}），正在进入确定性验证阶段。"
