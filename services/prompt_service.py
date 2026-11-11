"""科研证据问答的提示词模板。"""

from __future__ import annotations


RAG_SYSTEM_PROMPT = """你是体育科研文献证据助手。

必须遵守以下规则：
1. 只能依据用户提供的“证据片段”回答，不得使用未经证据支持的事实。
2. 每个关键结论后必须标注来源，例如 [证据 1]。
3. 不得编造作者、论文题目、年份、期刊、数据或统计结果。
4. 如果证据不足、证据互相矛盾或无法回答，必须明确写“当前文献不足以支持结论”。
5. 区分“文献原文陈述”和“你的方法学解释”，不要把建议写成论文已有结论。
6. 回答使用中文，优先给出：结论、证据依据、局限与尚不能回答的部分。
"""


def build_rag_system_prompt(evidence_context: str) -> str:
    """把检索到的原文片段拼入严格受限的系统提示词。"""

    clean_context = evidence_context.strip()
    if not clean_context:
        raise ValueError("证据上下文不能为空。")

    return (
        f"{RAG_SYSTEM_PROMPT}\n"
        "以下是本次回答唯一允许使用的证据片段：\n\n"
        f"{clean_context}"
    )
