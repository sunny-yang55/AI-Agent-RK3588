import voice_ui as ui


class ContextResolver:

    # Only genuinely elliptical follow-ups inherit the previous entity.
    # A sentence that introduces an explicit topic after these verbs is a new
    # request even if the entity is not present in entity_alias.json.
    NEW_TOPIC_PREFIXES = (
        "介绍一下", "介绍下", "介绍", "讲一下", "讲讲", "说一下", "说说",
        "了解一下", "查一下", "搜索", "查询",
    )

    ELLIPTICAL_PREFIXES = (
        "它", "他的", "她的", "这个", "那个", "该", "其",
        "有哪些", "有什么", "多少", "在哪里", "在哪", "怎么样", "如何",
        "专业有哪些", "学院有哪些", "地址是", "位置是",
    )

    def resolve(self, text, memory):

        if not text:
            return text

        # ==========================
        # Step 1. 获取当前实体
        # ==========================

        focus = memory.get_focus()

        if not focus:
            return text

        # ==========================
        # Step 2. 已经包含实体
        # ==========================

        if focus in text:
            return text

        compact = text.strip("。！？!?，,；;：: ")

        # “介绍一下安徽合肥/长鑫存储” already names a fresh topic. Never
        # prepend the old focus. Pronoun-based forms such as “介绍一下它” are
        # rewritten by ConversationMemory before reaching this resolver.
        for prefix in self.NEW_TOPIC_PREFIXES:
            if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
                return text

        # ==========================
        # Step 3. 闲聊过滤
        # ==========================

        ignore_patterns = [
            "你好",
            "您好",
            "嗨",
            "谢谢",
            "感谢",
            "辛苦了",
            "再见",
            "拜拜",
            "好的",
            "嗯",
            "哦",
        ]

        for p in ignore_patterns:

            if p in text:

                return text

        # ==========================
        # Step 4. 判断是否需要主体
        # ==========================

        need_context_patterns = [
            # 地点
            "在哪里",
            "在哪",
            "地址",
            "位置",
            "地点",
            "总部",
            "校区",
            # 时间
            "成立",
            "创办",
            "多久",
            "年份",
            "时间",
            # 信息
            "专业",
            "学院",
            "学校",
            "公司",
            "企业",
            # 属性
            "优势",
            "特色",
            "怎么样",
            "如何",
            "情况",
            # 省略主语的追问
            "有哪些",
            "多少",
        ]

        need_context = any(compact.startswith(p) for p in self.ELLIPTICAL_PREFIXES)

        for p in need_context_patterns:

            if compact == p or compact.startswith(p) or len(compact) <= 8 and p in compact:

                need_context = True

                break

        if not need_context:

            return text

        # ==========================
        # Step 5. 主语补全
        # ==========================

        resolved = f"{focus}{text}"

        ui.debug(f"[Context Resolver V3] {text} -> {resolved}")

        return resolved
