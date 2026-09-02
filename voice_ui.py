"""Small dependency-free user interface for the RK3588 voice runtime."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import unicodedata
from pathlib import Path
from datetime import datetime


USER_MODE = os.getenv("AI_AGENT_UI_MODE", "user").strip().lower() != "debug"
LOG_PATH = Path(__file__).resolve().parent / "logs" / "voice-debug.log"
COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def _width() -> int:
    return max(54, min(78, shutil.get_terminal_size((72, 24)).columns - 2))


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def debug(message: str) -> None:
    if not USER_MODE:
        print(message, flush=True)
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}\n")
    except OSError:
        pass


def banner() -> None:
    width = _width()
    title = "小安 · RK3588 智能语音助手"
    subtitle = "语音交互 v1.4  ·  持续对话模式"
    tips = "唤醒：包含“小安”  │  打断：说“停/停止”  │  退出：说“再见”"
    print()
    print(_c("36", "╭" + "─" * (width - 2) + "╮"))
    for line in (title, subtitle, "", tips):
        pad = max(0, width - 4 - _display_width(line))
        print(_c("36", "│ ") + _c("1;97", line) + " " * pad + _c("36", " │"))
    print(_c("36", "╰" + "─" * (width - 2) + "╯"))
    print(_c("90", "  状态信息会持续显示；详细诊断写入 logs/voice-debug.log"))
    print()


def state(message: str, symbol: str = "●") -> None:
    colors = {"✓": "32", "!": "33", "■": "31", "▶": "35", "◌": "36", "●": "36"}
    print(f"{_c(colors.get(symbol, '36'), symbol)} {_c('90', message)}", flush=True)


def user(text: str) -> None:
    print()
    print(_c("34", "┌─ 你"))
    for line in textwrap.wrap(text, width=_width() - 4) or [""]:
        print(f"{_c('34', '│')} {line}")


def assistant(text: str) -> None:
    print(_c("35", "┌─ 小安"))
    for line in textwrap.wrap(text.strip(), width=_width() - 4) or [""]:
        print(f"{_c('35', '│')} {line}")


def assistant_start() -> None:
    print("\n" + _c("35", "小安："), end="", flush=True)


def completed() -> None:
    print(_c("32", "└─ ✓ 回答完成") + "\n", flush=True)


def metric(message: str) -> None:
    print(_c("90", f"   ⏱ {message}"), flush=True)


def _replace_line(message: str) -> None:
    print(f"\r\033[2K{message}", end="", flush=True)


def recognition_start() -> None:
    _replace_line("🎤 检测到讲话，正在识别…")


def recognition_partial(text: str) -> None:
    _replace_line(f"◌ 正在识别… {text}")


def recognition_finalize() -> None:
    _replace_line("◌ 正在进行精准校正…")


def recognition_result(text: str) -> None:
    print(f"\r\033[2K✓ 识别结果：{text}", flush=True)
