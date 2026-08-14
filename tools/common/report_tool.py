from tools.llm.adapter import LLMAdapter


class ReportTool:

    def __init__(self):
        self.llm = LLMAdapter()


    def generate_report(self, goal, results):

        prompt = f"""
你是一个AI Agent。

任务:
{goal}

执行结果:
{results}


请生成简洁的中文分析报告。

要求:
1. 总结完成情况
2. 说明关键发现
3. 给出下一步建议
"""

        return self.llm.chat(prompt)