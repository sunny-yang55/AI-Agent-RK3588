from tools.llm.adapter import LLMAdapter


class ResponseGenerator:

    def __init__(self):

        self.llm = LLMAdapter()

    def generate_response(self, goal: str):

        response = self.llm.chat(goal)

        # ==========================
        # Voice mode response limit
        # ==========================

        if len(response) > 50:
            response = response[:500] + "\n\n更多详细信息请查看文字回答。"
        return response
