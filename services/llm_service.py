"""大模型服务与模拟回答。

使用标准库调用 OpenAI 兼容的 /chat/completions 接口，避免在第一阶段增加
额外 SDK 依赖。没有 API Key 时返回明确标注的模拟回答。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import Settings, settings as default_settings
from services.prompt_service import build_rag_system_prompt


class LLMConfigurationError(RuntimeError):
    """大模型配置缺失或无效。"""


@dataclass(frozen=True)
class LLMResponse:
    """一次问答结果。"""

    answer: str
    mode: str
    success: bool = True
    message: str = ""

    @property
    def is_simulated(self) -> bool:
        return self.mode == "模拟回答"


class LLMService:
    """封装真实接口调用和离线模拟回答。"""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings

    @property
    def is_configured(self) -> bool:
        return self.config.llm_configured

    def status_text(self) -> str:
        """返回不包含密钥的状态说明。"""

        if self.is_configured:
            return f"已配置模型：{self.config.openai_model}"
        return "未配置 API Key，当前仅可使用模拟回答"

    def answer(
        self,
        question: str,
        *,
        allow_simulation: bool = True,
        evidence_context: str = "",
        system_prompt: str = "",
    ) -> LLMResponse:
        """回答问题；配置存在时优先调用真实模型。

        allow_simulation=False 且没有密钥时，返回清晰失败信息而不是崩溃。
        """

        clean_question = question.strip()
        if not clean_question:
            return LLMResponse(
                answer="",
                mode="输入校验",
                success=False,
                message="问题不能为空，请输入具体的体育科研问题。",
            )

        if not self.is_configured:
            if allow_simulation:
                return LLMResponse(
                    answer=self._simulated_answer(clean_question),
                    mode="模拟回答",
                    message="未检测到 OPENAI_API_KEY，已自动使用本地模拟模板。",
                )
            return LLMResponse(
                answer="",
                mode="配置缺失",
                success=False,
                message=(
                    "未检测到 OPENAI_API_KEY。请把 .env.example 复制为 .env，"
                    "填写密钥后重启 Streamlit；当前不会调用真实模型。"
                ),
            )

        try:
            if system_prompt.strip():
                resolved_system_prompt = system_prompt.strip()
            elif evidence_context.strip():
                resolved_system_prompt = build_rag_system_prompt(evidence_context)
            else:
                resolved_system_prompt = self._default_system_prompt()
            answer = self._call_chat_completions(
                clean_question,
                resolved_system_prompt,
            )
            return LLMResponse(answer=answer, mode="大模型回答")
        except LLMConfigurationError as exc:
            if allow_simulation:
                return LLMResponse(
                    answer=self._simulated_answer(clean_question),
                    mode="模拟回答",
                    message=f"大模型调用未完成，已回退到模拟模式：{exc}",
                )
            return LLMResponse(
                answer="",
                mode="调用失败",
                success=False,
                message=str(exc),
            )

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """调用兼容 Chat Completions 的接口并返回完整 assistant message。

        该方法是 Function Calling 的底层接口。调用方负责传入标准 OpenAI
        兼容消息数组，并根据返回的 ``tool_calls`` 执行对应工具。
        """

        if not self.is_configured:
            raise LLMConfigurationError(
                "未配置大模型 API Key，无法执行 Function Calling。"
            )
        if not messages:
            raise LLMConfigurationError("消息列表不能为空。")

        payload: dict[str, Any] = {
            "model": self.config.openai_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        data = self._request_chat_payload(payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMConfigurationError(
                "接口返回内容缺少 choices。"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMConfigurationError(
                "接口返回的 choice 格式不正确。"
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMConfigurationError(
                "接口返回内容缺少 assistant message。"
            )
        return message

    def _call_chat_completions(
        self,
        question: str,
        system_prompt: str,
    ) -> str:
        payload = {
            "model": self.config.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.2,
        }
        data = self._request_chat_payload(payload)

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMConfigurationError(
                "接口返回内容不是预期的 Chat Completions 格式。"
            ) from exc

        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            ).strip()
        else:
            text = str(content or "").strip()

        if not text:
            raise LLMConfigurationError("大模型返回了空内容。")
        return text

    def _request_chat_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """发送 Chat Completions 请求并返回经过基础校验的 JSON。"""

        endpoint = _build_chat_endpoint(self.config.openai_base_url)
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.config.llm_timeout_seconds
            ) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_text = _safe_read_error(exc)
            raise LLMConfigurationError(
                f"接口返回 HTTP {exc.code}。{error_text}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMConfigurationError(f"无法连接大模型接口：{exc}") from exc

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(
                "接口返回内容不是有效 JSON。"
            ) from exc
        if not isinstance(data, dict):
            raise LLMConfigurationError(
                "接口返回内容不是预期的 Chat Completions 格式。"
            )
        return data

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "你是体育科研方法助手。请用中文给出结构化、审慎的回答；"
            "不要编造文献、作者、数据或统计结果。需要证据时明确提示用户"
            "检索并核验真实数据库来源，并说明回答的局限性。"
        )

    @staticmethod
    def _simulated_answer(question: str) -> str:
        """生成不依赖外部接口的教学型模拟回答。"""

        return (
            "> **模拟内容提示：** 当前回答由本地规则模板生成，未调用真实大模型，"
            "不代表真实文献结论。\n\n"
            f"**问题：** {question}\n\n"
            "### 1. 先界定核心概念\n"
            "明确研究对象、研究场景、干预或暴露因素、结局指标及其测量方式。"
            "如果关键词含义不唯一，应先查阅权威教材和数据库词表。\n\n"
            "### 2. 建立证据检索路径\n"
            "建议组合“研究对象 AND 核心变量 AND 结局指标”检索 CNKI、"
            "Web of Science、PubMed 或 SPORTDiscus。当前工具不会提供虚构论文，"
            "请记录真实检索日期并阅读原文。\n\n"
            "### 3. 把问题变成可检验设计\n"
            "优先说明研究问题、假设、样本来源、变量定义、统计计划和局限性。"
            "若涉及人体实验，应提前完成伦理审查与知情同意。\n\n"
            "### 4. 结论边界\n"
            "在真实数据和文献核验前，只能提出研究思路，不能给出效果大小、"
            "显著性结论或文献引用。"
        )


def _build_chat_endpoint(base_url: str) -> str:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        raise LLMConfigurationError("OPENAI_BASE_URL 不能为空。")
    if clean_url.endswith("/chat/completions"):
        return clean_url
    return f"{clean_url}/chat/completions"


def _safe_read_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "请检查 API Key、模型名称和接口地址。"
    return raw[:500] if raw else "请检查 API Key、模型名称和接口地址。"
