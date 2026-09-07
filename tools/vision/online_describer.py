"""Online multimodal scene description with a safe local fallback boundary.

This module is intentionally independent from camera ownership.  The camera
process supplies a fresh JPEG only when a user asks a visual question; this
module then sends that single image to an OpenAI-compatible multimodal API.
It must never be used as the source of robot-grasp coordinates.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Allows source-only diagnostics before runtime setup.
    def load_dotenv(*_args, **_kwargs):
        return False


def should_use_online_scene_description(query: str) -> bool:
    """Use the cloud model for broad scene questions, not precise locating."""
    from .session import is_broad_scene_query

    return is_broad_scene_query(query)


class OnlineVisionDescriber:
    """One-image, fact-bounded multimodal description client.

    ``VISION_LLM_*`` values take priority over the existing text ``LLM_*``
    values.  Keeping the visual model separate lets deployments select a
    multimodal model without changing the conversation model.
    """

    def __init__(self, *, client: Any | None = None, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self._load_environment()
        self.enabled = os.getenv("VISION_LLM_ENABLED", "0").strip().lower() in {
            "1", "true", "yes",
        }
        self.base_url = os.getenv("VISION_LLM_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.api_key = os.getenv("VISION_LLM_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = os.getenv("VISION_LLM_MODEL", "").strip()
        self.timeout = max(3.0, float(os.getenv("VISION_LLM_TIMEOUT", "20")))
        self._client = client

    def _load_environment(self) -> None:
        # Keep the same default as LLMAdapter.  The RK3588 deployment stores
        # its active credentials in config/.env.qwen; .env.rk3588 is only an
        # example file and must never be assumed to exist.
        env_name = os.getenv("AI_AGENT_ENV", ".env.qwen")
        env_path = self.root / "config" / env_name
        if env_path.is_file():
            load_dotenv(env_path, override=False)

    @property
    def is_available(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key and self.model)

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        missing = [
            name
            for name, value in (
                ("VISION_LLM_BASE_URL", self.base_url),
                ("VISION_LLM_API_KEY", self.api_key),
                ("VISION_LLM_MODEL", self.model),
            )
            if not value
        ]
        return "ready" if not missing else "missing " + ", ".join(missing)

    def describe(self, image_jpeg: bytes, query: str, facts: dict[str, Any]) -> str:
        """Return a concise Chinese scene reply based on one current image."""
        if not self.is_available:
            raise RuntimeError(f"online vision is not configured: {self.status}")
        if not image_jpeg:
            raise ValueError("camera returned an empty image")
        image_data = base64.b64encode(image_jpeg).decode("ascii")
        payload = json.dumps(self._safe_facts(facts), ensure_ascii=False)
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._instructions()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"用户的问题：{query}\n本地视觉事实：{payload}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "low"},
                        },
                    ],
                },
            ],
            max_tokens=180,
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        if not answer or not answer.strip():
            raise RuntimeError("online vision returned an empty answer")
        return answer.strip()

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _safe_facts(facts: dict[str, Any]) -> dict[str, Any]:
        """Keep evidence small and never promote candidate shape as a fact."""
        objects = []
        for item in facts.get("workbench_objects", []):
            objects.append(
                {
                    "color": item.get("color_zh") or item.get("color"),
                    "center_pixel": item.get("center_pixel"),
                    "center_roi": item.get("center_roi"),
                }
            )
        return {
            "local_summary": facts.get("summary", ""),
            "workbench_objects": objects,
            "generic_detections": [
                {"label": item.get("label_zh") or item.get("label"), "confidence": item.get("confidence")}
                for item in facts.get("detections", [])
            ],
        }

    @staticmethod
    def _instructions() -> str:
        return (
            "你是小安的视觉描述模块，用简洁自然的中文回答语音用户，最多三句。"
            "依据当前图片和本地视觉事实回答；本地事实优先于猜测。"
            "可以描述可见物品、颜色和大致关系；看不清就明确说不确定，绝不编造。"
            "当用户询问桌面或画面有什么时，先说二到四个最显著的物品，尽量包含颜色和类别；"
            "再简短补充工作台上的彩色物块。若物块形状仅凭图片无法确认，使用“看起来像”而非断言。"
            "不要声称已经执行机械臂动作，也不要生成抓取坐标。"
            "对于广义场景盘点，可在最后自然地问一次用户是否需要定位某个物块或描述某件物品；"
            "对于具体的是非问题，直接回答，不要追加机械式追问。"
        )
