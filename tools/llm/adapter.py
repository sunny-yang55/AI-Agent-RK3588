"""
AI-Agent LLM Adapter
统一模型调用接口,支持：
- Qwen
- DeepSeek
- Kimi
- OpenAI Compatible API
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


class LLMAdapter:

    def __init__(self):

        # 项目根目录
        self.root = Path(__file__).resolve().parents[2]

        # 加载配置
        # ==============================
        # Load environment configuration
        # ==============================

        env_name = os.getenv("AI_AGENT_ENV", ".env.qwen")

        env_path = self.root / "config" / env_name

        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        load_dotenv(env_path)

        # print("Load env:", env_path)

        self.provider = os.getenv("LLM_PROVIDER", "qwen")

        self.base_url = os.getenv("LLM_BASE_URL")

        self.api_key = os.getenv("LLM_API_KEY")

        self.model = os.getenv("LLM_MODEL")
        """测试
        print("==============================")
        print("DEBUG ENV")
        print("==============================")
        print("DEBUG ENV")
        print("BASE_URL =", repr(self.base_url))
        print("MODEL =", repr(self.model))
        if self.api_key:
            print("API_KEY =", repr(self.api_key[:10]))
        else:
            print("API_KEY = None")

        print("==============================")
        print("==============================")
        """
        if not self.base_url:
            raise RuntimeError("LLM_BASE_URL not configured")

        if not self.api_key:
            raise RuntimeError("LLM_API_KEY not configured")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _messages(self, message):
        return [
                {
                    "role": "system",
                    "content": """
你是AI-Agent智能助手。
你的输入可能来自语音识别系统（ASR），
用户输入可能存在少量语音识别错误。
处理规则：
1. 优先理解用户真实意图，而不是机械匹配文字。
2. 对以下情况进行语义纠正：
- 同音字错误
- 近音词错误
- 公司名称错误
- 人名错误
- 地名错误
- 专业术语错误

例如：
- 长兴存储 → 长鑫存储（如果上下文涉及安徽合肥、DRAM、芯片）
- 科大讯飞 → 不要错误改成其他公司

3. 如果存在明显正确候选：
直接按照正确意图回答。

4. 如果存在多个可能：
先指出可能的情况，再询问用户。

5. 不要因为名称不存在立即回答“不存在”，先进行合理猜测，然后再请求用户重复一遍。

6. 不要输出：
- “你提供的数据有误”
- “你的文本无法识别”
- “AudioData为空”
- “samples=None”

用户只关心任务结果。

7. 保持专业助手风格：
- 简洁
- 清晰
- 结构化
- 不使用角色扮演语言。

8. 当前是语音交互场景：
- 默认先给结论，最多120个汉字，优先使用2到3个完整句子；
- 用户明确要求详细说明时再展开；
- 避免冗长背景、重复总结和不必要的项目符号。
- 必须自然收尾，绝不能在逗号、冒号、专有名词或半句话处停止。
- 接近长度限制时主动结束当前句，不要继续开启新的要点。
- 景点、专业或其他例子最多列举4个；不要用重复总结凑长度。
""",
                },
                {"role": "user", "content": message},
            ]

    def chat(self, message, request_id=None, on_token=None):
        """Generate one response, optionally forwarding streamed text chunks."""
        kwargs = {
            "model": self.model,
            "messages": self._messages(message),
        }
        max_tokens = int(os.getenv("AI_AGENT_MAX_RESPONSE_TOKENS", "240"))
        if max_tokens > 0:
            kwargs["max_tokens"] = max_tokens

        if on_token is None:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        started = time.perf_counter()
        first_token_s = None
        pieces = []
        response = self.client.chat.completions.create(**kwargs, stream=True)
        for chunk in response:
            content = chunk.choices[0].delta.content if chunk.choices else None
            if not content:
                continue
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            pieces.append(content)
            on_token(content)

        return "".join(pieces)
