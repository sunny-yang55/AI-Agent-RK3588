from collections import deque


class ConversationMemory:

    def __init__(self, max_history=10):

        # 当前对话关注实体
        self.current_focus = None

        # 当前实体类型
        self.current_category = None

        # 最近对话历史
        self.history = deque(maxlen=max_history)

        # 对话轮次
        self.turn = 0

    # -------------------------
    # Entity
    # -------------------------

    def update_entity(self, entity, category=None):

        if entity:

            self.current_focus = entity

        if category:

            self.current_category = category

    def get_focus(self):

        return self.current_focus

    # -------------------------
    # History
    # -------------------------

    def add_user(self, text):

        self.history.append({"role": "user", "content": text})

        self.turn += 1

    def add_assistant(self, text):

        self.history.append({"role": "assistant", "content": text})

    def get_history(self):

        return list(self.history)

    # -------------------------
    # Rewrite
    # -------------------------

    def rewrite(self, text):

        if not text:

            return text

        if self.current_focus is None:

            return text

        pronouns = [
            "它",
            "它的",
            "他",
            "他的",
            "她",
            "她的",
            "这家公司",
            "那个公司",
            "该公司",
            "这个企业",
            "该企业",
            "这个学校",
            "那个学校",
            "该校",
        ]

        result = text

        for p in pronouns:

            result = result.replace(p, self.current_focus)

        return result

    # -------------------------
    # Debug
    # -------------------------

    def dump(self):

        print("========== Memory ==========")

        print("Focus :", self.current_focus)

        print("Category :", self.current_category)

        print("Turn :", self.turn)
        print()

        for item in self.history:

            print(item)

        print("============================")
