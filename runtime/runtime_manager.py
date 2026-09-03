import asyncio
import os
import time

from agent import Agent
from runtime.input_adapter import InputAdapter
from runtime.streaming_speech import StreamingSpeechPlayer
from runtime.vision_control import VisionVoiceController
import voice_ui as ui
from speech.filter import ConfidenceFilter, NoiseFilter
from tools.common.context_resolver import ContextResolver
from tools.common.conversation_memory import ConversationMemory
from tools.common.entity_resolver import EntityResolver
from tools.llm.adapter import LLMAdapter
from tools.speech import SpeechManager
from tools.vision.process_service import ProcessVisionService


class RuntimeManager:

    def __init__(self, agent):

        self.agent = agent

        ui.debug("[系统] 初始化语音模块...")

        self.speech = SpeechManager()

        self.vision = VisionVoiceController(
            ProcessVisionService(),
            self.speech.speak,
        )

        ui.debug("[系统] AI核心加载完成")

        self.llm = LLMAdapter()

        # print("[Runtime] Ready")

        self.entity_resolver = EntityResolver()

        self.memory = ConversationMemory()

        self.context_resolver = ContextResolver()

        self.noise_filter = NoiseFilter()

        self.confidence_filter = ConfidenceFilter()

    def start_voice_loop(self):

        if ui.USER_MODE:
            ui.banner()
        else:
            print("==============================\n AI-Agent Voice Runtime \n “再见”，“退出”，退出语音 \n==============================")

        technical_failures = 0
        max_failures = max(1, int(os.getenv("AI_AGENT_MAX_AUDIO_FAILURES", "3")))

        while True:

            result = self.speech.listen()

            if not result.success:

                if result.woke:
                    # Online TTS acknowledgement delays the next microphone
                    # opening by several seconds. Default to an immediate UI
                    # acknowledgement; it can still be enabled explicitly.
                    if os.getenv("AI_AGENT_WAKE_ACK_TTS", "0").lower() in {
                        "1", "true", "yes"
                    }:
                        self.speech.speak("我在，请说。", allow_interrupt=False)
                    elif ui.USER_MODE:
                        ui.assistant_start()
                        print("我在，请说。")
                    continue

                if result.error:
                    technical_failures += 1
                    ui.debug(
                        f"[Runtime] Audio failure {technical_failures}/{max_failures}: "
                        f"{result.error}"
                    )
                    if technical_failures >= max_failures:
                        print("[Runtime] 连续录音失败，已停止语音循环。请检查麦克风后重新启动。")
                        break
                    time.sleep(1.0)
                else:
                    technical_failures = 0
                    ui.debug("[Runtime] Empty speech event, retry")

                continue

            technical_failures = 0

            if result.exit:

                ui.debug("[Runtime] Exit command detected")

                self.speech.speak("好的，再见！", allow_interrupt=False)

                if ui.USER_MODE:
                    ui.state("语音助手已退出", "✓")

                break

            # ==========================
            # Speech Result
            # ==========================

            raw_text = result.text.strip()

            if self.vision.handle(raw_text):
                continue

            # 测试
            # print("[ASR Raw]", repr(raw_text))

            text = self.noise_filter.clean(raw_text)

            # Preserve what the user actually said for the UI. Entity,
            # memory and context rewriting below are internal LLM inputs.
            display_text = text

            # 测试
            # print("[ASR Clean]", repr(text))
            # ==========================
            # Step 3.1 Noise Filter
            # ==========================

            if self.noise_filter.is_noise(text):

                ui.debug(f"[Runtime] Noise ignored: {text!r}")

                continue

            # ==========================
            # Step 3.2 Confidence Filter
            # ==========================

            confidence = self.confidence_filter.check(text)

            # 测试
            # print("[Confidence]", confidence)

            if not confidence:
                ui.debug(f"[Runtime] Low confidence: {text!r}")

                self.speech.speak("没有听清，请再说一次")

                continue

            # ==========================
            # Step 3.3 Entity Resolve
            # ==========================

            resolved_text, entity = self.entity_resolver.resolve(text)

            # 保存当前对话实体

            if entity:

                self.memory.update_entity(entity)

            # ==========================
            # Memory Rewrite
            # ==========================

            resolved_text = self.memory.rewrite(resolved_text)

            # 测试
            # print("[Memory Rewrite]", resolved_text)

            # ==========================
            # Context Resolver
            # 主语补全
            # ==========================

            resolved_text = self.context_resolver.resolve(resolved_text, self.memory)

            # 测试
            # print("[Context Resolve]", resolved_text)

            result.text = resolved_text

            # --------------------------
            # Conversation Rewrite
            # --------------------------

            # 测试
            # print("[Memory Rewrite]", resolved_text)

            result.text = resolved_text

            # 测试
            # print("[Runtime] Final text:", resolved_text)

            # 保存用户历史

            self.memory.add_user(result.text)

            event = InputAdapter.from_speech(result)

            if ui.USER_MODE:
                ui.user(display_text)
                ui.state("正在思考…", "◌")
            else:
                print("\n" + "=" * 60 + "\n User:\n" + str(event.data) + "\n" + "=" * 60)

            response_started = time.perf_counter()
            first_token_s = None
            streamed = False
            stream_tts = None

            stream_enabled = os.getenv(
                "AI_AGENT_STREAM_RESPONSE", "1"
            ).strip().lower() not in {"0", "false", "no"}
            # Start speaking complete semantic sentences while later LLM text
            # is still arriving. This avoids waiting for the whole answer.
            tts_mode = os.getenv("AI_AGENT_TTS_MODE", "whole").strip().lower()
            stream_tts_enabled = tts_mode == "sentence" and os.getenv(
                "AI_AGENT_STREAM_TTS", "0"
            ).strip().lower() not in {"0", "false", "no"}

            if stream_enabled and stream_tts_enabled:
                stream_tts = StreamingSpeechPlayer(
                    lambda sentence: self.speech.speak(sentence, display=False)
                )

            def show_response_chunk(chunk):
                nonlocal first_token_s, streamed
                if not streamed:
                    first_token_s = time.perf_counter() - response_started
                    if ui.USER_MODE:
                        ui.state("正在生成回答…", "◌")
                    else:
                        print("\nAI-Agent:")
                    streamed = True
                if not ui.USER_MODE:
                    print(chunk, end="", flush=True)
                if stream_tts is not None:
                    stream_tts.feed(chunk)

            stream_callback = None
            if stream_enabled:
                stream_callback = show_response_chunk

            try:
                response = asyncio.run(
                    self.agent.run(
                        event,
                        context={
                            "conversation_memory": self.memory,
                            "response_stream_callback": stream_callback,
                        },
                    )
                )
            finally:
                if stream_tts is not None:
                    # Close the input without waiting for synthesis/playback.
                    # This keeps the LLM token callback completely independent
                    # from the serial background audio worker.
                    stream_tts.finish(wait=False)

            if isinstance(response, dict):

                text = response.get("response", "")

            else:

                text = str(response)

            self.memory.add_assistant(text)

            total_s = time.perf_counter() - response_started
            if streamed:
                if ui.USER_MODE:
                    ui.assistant(text)
                    ui.metric(
                        f"首字 {first_token_s:.2f}s · 完整回答 {total_s:.2f}s · 正在准备语音"
                    )
                else:
                    print()
                ui.debug(
                    f"[性能] LLM首字: {first_token_s:.2f}s | "
                    f"完整回复: {total_s:.2f}s"
                )
            else:
                ui.debug(f"[性能] 完整回复: {total_s:.2f}s")

            if stream_tts is None or stream_tts.submitted == 0:
                control = self.speech.speak(text, display=not streamed)
                if control == "exit":
                    print("[Runtime] 播报期间收到退出命令")
                    break
                if control in {"stop", "sleep"}:
                    ui.state("已停止当前播报", "■") if ui.USER_MODE else print("[Runtime] 当前播报已停止")
                elif ui.USER_MODE:
                    ui.completed()
            else:
                # Avoid recording the loudspeaker, but only after LLM timing
                # has been printed. Audio remains ordered on one worker.
                stream_tts.wait()
                if ui.USER_MODE:
                    ui.completed()

        self.vision.close()
