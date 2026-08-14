import logging
import re
import time

logger = logging.getLogger(__name__)


class MemoryCompressor:

    def __init__(self, max_episode=50):

        self.max_episode = max_episode

    def should_compress(self, episodic):

        size = len(episodic)

        logger.debug("[Memory] Checking compression")

        logger.debug(f"[Memory] Episodic size: {size}")

        return size > self.max_episode

    def compress(self, episodes):

        logger.debug("[Memory] Compressing...")

        facts = []

        topics = []

        for ep in episodes[-100:]:

            text = str(ep)

            # 去除无意义标签

            text = text.replace("用户:", "")

            text = text.replace("AI-Agent:", "")

            # ---------
            # 提取实体
            # ---------

            entities = re.findall(r"[\u4e00-\u9fa5]{4,15}", text)

            for e in entities:

                if e not in facts and len(e) <= 15:

                    facts.append(e)

            # ---------
            # 保存主题
            # ---------

            if "goal" in text:

                topics.append(text[:80])

        summary = {
            "created": time.time(),
            "episode_count": len(episodes),
            "facts": facts[:30],
            "topics": topics[-10:],
        }

        logger.debug("[Memory] Compression finished")

        return summary
