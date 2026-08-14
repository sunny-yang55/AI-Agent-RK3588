"""
Agent Response Generator

负责:
1. 用户目标理解
2. Tool结果整理
3. Memory上下文注入
4. LLM生成最终回复
"""

import logging
import platform
from datetime import datetime

from tools.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)


class ResponseGenerator:

    def __init__(self, llm=None):

        if llm:
            self.llm = llm
        else:
            self.llm = LLMAdapter()

    def _build_context(self):

        return {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": platform.system(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        }

    def generate(
        self,
        goal,
        tool_results,
        memory=None,
        request_id=None,
        conversation_memory=None,
        on_token=None,
    ):
        runtime = self._build_context()

        prompt = f"""
# Role

你是 AI-Agent 智能助手。

你不是简单聊天机器人，而是一个具备：
- 任务规划能力
- 工具调用能力
- 记忆能力
的智能体。

# 当前运行环境
当前时间:
{runtime["time"]}
操作系统:
{runtime["system"]}
系统信息:
{runtime["platform"]}
Python版本:
{runtime["python"]}

# 用户请求
{goal}
# 工具执行结果
{tool_results}
# Agent长期记忆
{memory}
# 当前对话上下文
当前关注实体:
{
conversation_memory.get_focus()
if conversation_memory
else "无"
}
最近对话:
{
conversation_memory.get_history()
if conversation_memory
else "无"
}

# 回复要求

1. 使用中文回答。
2. 直接回答用户问题。
3. 不要说：
“我收到你的请求”
“正在处理中”
4. 如果知道答案，直接给答案。
5. 如果信息不足，明确说明。
6. 当前是语音交互：普通问答最多120个汉字，优先使用2到3个完整句子。
7. 优先回答最重要的信息；用户明确要求详细说明时才展开。
8. 必须以完整句子结束，禁止在专有名词、逗号、冒号或半句话处截断。
9. 不要为了凑数量罗列过多项目，景点或例子最多给出4个。
现在生成最终回复:
"""
        """测试memory输出
        print("================ CONTEXT DEBUG ================")

        if conversation_memory:
            print("Focus:", conversation_memory.get_focus())

            print("History:", conversation_memory.get_history())

        print("==============================================")
        """
        try:

            response = self.llm.chat(
                prompt,
                request_id=request_id,
                on_token=on_token,
            )

            return response

        except Exception as e:

            print("==============================")
            print("RESPONDER REAL ERROR")
            print(type(e))
            print(str(e))

            import traceback

            traceback.print_exc()

            print("==============================")

            return "抱歉，我暂时无法生成回答。"
