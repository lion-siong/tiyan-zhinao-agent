"""Semantic Scholar 公开检索与用户 PDF 本地 RAG 的 Streamlit 页面。"""

from __future__ import annotations

import json
from collections.abc import Sequence

import streamlit as st

from agents.literature_research_agent import (
    LiteratureResearchAgent,
    LiteratureResearchResult,
)
from config import settings
from services.llm_service import LLMService
from services.rag_service import EvidenceAnswer
from services.user_paper_rag import (
    UserPaperRAGError,
    UserPaperRAGService,
    UserPaperRecord,
)
from services.vector_service import VectorStoreError


@st.cache_resource
def get_literature_research_agent() -> LiteratureResearchAgent:
    """复用 Function Calling Agent 和模型服务。"""

    return LiteratureResearchAgent(settings, LLMService(settings))


@st.cache_resource
def get_user_paper_rag_service() -> UserPaperRAGService:
    """复用隔离的用户论文 Chroma 服务。"""

    return UserPaperRAGService(settings)


def render_literature_research() -> None:
    """渲染公开文献检索与个人论文 RAG 两个独立工作区。"""

    st.markdown(
        '<div class="section-label">LITERATURE INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.title("智能文献研究")
    st.write(
        "用 Function Calling 检索公开论文元数据，或只在用户主动上传的 "
        "PDF 原文中做本地 RAG 问答。两种数据来源严格隔离。"
    )

    workspace = st.segmented_control(
        "文献研究方式",
        ("公开文献检索", "个人论文 RAG"),
        default="公开文献检索",
        key="literature_research_workspace",
    )
    if workspace is None:
        workspace = "公开文献检索"

    if workspace == "公开文献检索":
        _render_public_search()
    else:
        _render_user_paper_rag()


def _render_public_search() -> None:
    """渲染 Semantic Scholar/arXiv 公开元数据检索与综述。"""

    st.info(
        "公开来源仅限 Semantic Scholar 和 arXiv 官方接口：只读取题名、"
        "作者、年份、摘要、DOI 与开放链接，不爬取知网，也不自动下载全文。"
    )
    if settings.llm_configured:
        st.caption(
            f"模型：{settings.openai_model}。Agent 会先优化检索词，"
            "再调用 semantic_scholar_search 获取真实结果。"
        )
    else:
        st.warning(
            "未配置大模型 API Key：仍会执行真实公开检索，但不会模拟 "
            "Function Calling，结果会明确标注为规则整理。"
        )

    with st.form("public_literature_search_form"):
        research_question = st.text_area(
            "研究问题",
            placeholder=(
                "例如：大学生体育锻炼行为与体质健康水平之间有什么关系？"
            ),
            height=96,
            key="public_literature_question",
        )
        limit = st.slider(
            "返回论文条数",
            min_value=1,
            max_value=settings.semantic_scholar_max_results,
            value=min(8, settings.semantic_scholar_max_results),
        )
        submitted = st.form_submit_button(
            "检索并整理文献",
            type="primary",
            icon=":material/search:",
        )

    if submitted:
        if not research_question.strip():
            st.error("请输入具体研究问题。")
        else:
            try:
                with st.spinner(
                    "正在优化检索词并读取公开论文元数据..."
                ):
                    result = get_literature_research_agent().run(
                        research_question,
                        limit=limit,
                    )
            except Exception as exc:
                st.error(f"文献研究流程未完成：{exc}")
            else:
                st.session_state["public_literature_result"] = result

    result = st.session_state.get("public_literature_result")
    if isinstance(result, LiteratureResearchResult):
        _render_public_result(result)


def _render_public_result(result: LiteratureResearchResult) -> None:
    """展示可追溯的公开检索结果、综述与 GB/T 7714 引用。"""

    st.divider()
    if not result.success:
        st.error(result.message or "本轮文献检索未完成。")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("公开来源", result.source or "未返回")
    metric_columns[1].metric("真实元数据", len(result.papers))
    metric_columns[2].metric("运行模式", result.mode)

    st.caption(f"原始研究问题：{result.original_query}")
    st.caption(f"实际检索式：{result.optimized_query or '未生成'}")
    if result.message:
        st.info(result.message)
    if result.warnings:
        with st.expander(f"检索提示（{len(result.warnings)} 条）"):
            for warning in result.warnings:
                st.warning(warning)

    st.subheader("文献综述与研究现状")
    if result.review.strip():
        st.markdown(result.review)
    else:
        st.info("当前没有可展示的综述内容。")

    if result.papers:
        st.subheader("论文元数据")
        for index, paper in enumerate(result.papers, start=1):
            with st.expander(
                f"[P{index}] {paper.title}",
                expanded=index == 1,
            ):
                st.markdown(
                    f"**作者：** {paper.authors_text}  \n"
                    f"**年份：** {paper.year or '信息缺失'}  \n"
                    f"**来源：** {paper.venue or '信息缺失'}  \n"
                    f"**DOI：** {paper.doi or '信息缺失'}"
                )
                abstract = paper.abstract.strip()
                st.write(
                    abstract
                    if abstract
                    else "公开来源未提供摘要，不能据此概括研究结论。"
                )
                with st.container(horizontal=True):
                    if paper.doi:
                        st.link_button(
                            "打开 DOI",
                            f"https://doi.org/{paper.doi}",
                            icon=":material/open_in_new:",
                        )
                    if paper.pdf_url:
                        st.link_button(
                            "开放 PDF",
                            paper.pdf_url,
                            icon=":material/picture_as_pdf:",
                        )
                    if paper.url:
                        st.link_button(
                            "来源页面",
                            paper.url,
                            icon=":material/link:",
                        )

    references = result.references
    if references:
        st.subheader("GB/T 7714 引用（需人工核验）")
        st.code("\n\n".join(references), language=None)
        st.caption(
            "格式化器只使用来源返回的字段；缺失信息会明确写出，"
            "正式提交前仍需核对卷、期、页码和文献类型。"
        )

    payload = json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        indent=2,
    )
    st.download_button(
        "下载本轮文献研究 JSON",
        payload,
        file_name="public_literature_result.json",
        mime="application/json",
        icon=":material/download:",
        width="stretch",
    )


def _render_user_paper_rag() -> None:
    """渲染仅处理用户上传 PDF 的本地 Chroma RAG 工作区。"""

    st.info(
        "版权与隐私边界：此工作区只读取你主动上传的 PDF，并在本机 "
        "data/user_papers/ 中按内容哈希隔离保存；公开检索元数据不会写入这里。"
    )
    service = get_user_paper_rag_service()
    _render_user_paper_upload(service)

    records = service.list_papers()
    if not records:
        st.caption("尚未导入用户论文。请先上传可复制文本的 PDF。")
        return

    _render_user_paper_library(service, records)

    paper_keys = tuple(record.paper_key for record in records)
    records_by_key = {record.paper_key: record for record in records}
    selected_keys = st.multiselect(
        "选择参与问答的用户论文",
        options=paper_keys,
        default=paper_keys,
        format_func=lambda key: records_by_key[key].file_name,
        key="user_paper_question_scope",
    )
    if not selected_keys:
        st.caption("请至少选择一篇论文后再提问。")
        return

    if settings.llm_configured:
        answer_mode = st.segmented_control(
            "回答模式",
            ("原文证据摘录", "大模型基于原文回答"),
            default="原文证据摘录",
            key="user_paper_answer_mode",
        )
        use_llm = answer_mode == "大模型基于原文回答"
    else:
        use_llm = False
        st.caption("未配置 API Key，当前只展示检索到的原文证据。")

    with st.form("user_paper_rag_question_form"):
        question = st.text_area(
            "基于所选论文提问",
            placeholder=(
                "例如：这些论文使用了哪些研究设计？样本和测量指标有何差异？"
            ),
            height=100,
            key="user_paper_rag_question",
        )
        top_k = st.slider(
            "检索证据片段数",
            min_value=1,
            max_value=10,
            value=min(settings.rag_top_k, 10),
        )
        retrieval_mode = st.segmented_control(
            "检索方式",
            ("混合检索", "语义检索", "BM25 检索"),
            default="混合检索",
            key="user_paper_retrieval_mode",
        )
        asked = st.form_submit_button(
            "检索并回答",
            type="primary",
            icon=":material/manage_search:",
        )

    if asked:
        if not question.strip():
            st.error("问题不能为空。")
        else:
            mode_map = {
                "混合检索": "hybrid",
                "语义检索": "semantic",
                "BM25 检索": "bm25",
            }
            try:
                with st.spinner("正在用户论文原文中检索证据..."):
                    answer = service.answer(
                        question,
                        selected_keys,
                        top_k=top_k,
                        use_llm=use_llm,
                        retrieval_mode=mode_map[
                            retrieval_mode or "混合检索"
                        ],
                    )
            except (UserPaperRAGError, VectorStoreError, OSError) as exc:
                st.error(f"用户论文问答未完成：{exc}")
            else:
                st.session_state["user_paper_rag_answer"] = answer

    answer = st.session_state.get("user_paper_rag_answer")
    if isinstance(answer, EvidenceAnswer):
        _render_user_paper_answer(answer)


def _render_user_paper_upload(service: UserPaperRAGService) -> None:
    """处理多篇 PDF 上传、内容去重和错误提示。"""

    uploaded_files = st.file_uploader(
        "上传用户自己的 PDF 论文",
        type=["pdf"],
        accept_multiple_files=True,
        key="user_paper_pdf_uploader",
    )
    if not uploaded_files:
        return
    if len(uploaded_files) > settings.rag_max_documents:
        st.error(
            f"一次最多导入 {settings.rag_max_documents} 篇 PDF，"
            "请减少文件数量。"
        )
        return

    st.caption(
        "　".join(uploaded_file.name for uploaded_file in uploaded_files)
    )
    if st.button(
        "解析并建立本地索引",
        type="primary",
        icon=":material/library_add:",
        key="user_paper_ingest_button",
    ):
        messages: list[tuple[str, str]] = []
        created_count = 0
        with st.spinner(
            "正在解析 PDF、按页切分并同步 Chroma；"
            "首次生成向量可能需要数分钟..."
        ):
            for uploaded_file in uploaded_files:
                try:
                    result = service.ingest_pdf(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        sync_vector=False,
                    )
                except (UserPaperRAGError, OSError, ValueError) as exc:
                    messages.append(("error", f"{uploaded_file.name}：{exc}"))
                    continue
                if result.created:
                    created_count += 1
                    messages.append(
                        (
                            "success",
                            f"{result.record.file_name} 已导入："
                            f"{result.record.chunk_count} 个分页证据片段。",
                        )
                    )
                else:
                    messages.append(
                        ("info", f"{uploaded_file.name}：{result.warning}")
                    )
                if result.warning and result.created:
                    messages.append(("warning", result.warning))
            if created_count:
                try:
                    sync_result = service.sync_index()
                except (VectorStoreError, ValueError, OSError) as exc:
                    messages.append(
                        (
                            "warning",
                            "PDF 已保存，但 Chroma 暂不可用；"
                            f"问答会回退 BM25：{exc}",
                        )
                    )
                else:
                    messages.append(
                        (
                            "info",
                            "用户论文 Chroma 已同步："
                            f"{sync_result.total_chunks} 个片段。",
                        )
                    )
        st.session_state["user_paper_ingest_messages"] = messages

    messages = st.session_state.pop("user_paper_ingest_messages", [])
    for level, message in messages:
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        elif level == "success":
            st.success(message)
        else:
            st.info(message)


def _render_user_paper_library(
    service: UserPaperRAGService,
    records: Sequence[UserPaperRecord],
) -> None:
    """展示用户论文列表、Chroma 状态和删除入口。"""

    st.subheader("我的用户论文库")
    rows = [
        {
            "文件名": record.file_name,
            "页数": record.page_count,
            "证据片段": record.chunk_count,
            "导入时间": record.created_at,
        }
        for record in records
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    expected_chunks = sum(record.chunk_count for record in records)
    try:
        status = service.vector_store.status(
            expected_chunks=expected_chunks
        )
    except VectorStoreError as exc:
        st.warning(f"Chroma 状态不可用，将回退 BM25：{exc}")
    else:
        if status.synchronized:
            st.success(
                f"Chroma 已同步：{status.stored_chunks} 个用户论文片段。",
                icon=":material/database:",
            )
        else:
            st.warning(
                f"{status.message} 仍可使用 BM25 关键词检索。",
                icon=":material/database_off:",
            )
        if st.button(
            "重建用户论文 Chroma 索引",
            icon=":material/sync:",
            key="user_paper_rebuild_vector",
        ):
            try:
                with st.spinner("正在同步用户论文向量..."):
                    sync_result = service.sync_index()
            except (VectorStoreError, ValueError, OSError) as exc:
                st.error(f"Chroma 同步失败：{exc}")
            else:
                st.success(
                    f"索引同步完成：{sync_result.total_chunks} 个片段。"
                )

    records_by_key = {record.paper_key: record for record in records}
    selected_key = st.selectbox(
        "选择要删除的用户论文",
        options=tuple(records_by_key),
        format_func=lambda key: records_by_key[key].file_name,
        key="user_paper_delete_scope",
    )
    confirm_delete = st.checkbox(
        "确认删除本地 PDF、分页证据和对应 Chroma 向量",
        key="user_paper_delete_confirm",
    )
    if st.button(
        "删除选中的用户论文",
        icon=":material/delete:",
        disabled=not confirm_delete,
        key="user_paper_delete_button",
    ):
        try:
            service.remove_paper(selected_key)
        except UserPaperRAGError as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("user_paper_rag_answer", None)
            st.session_state["user_paper_delete_notice"] = (
                f"{records_by_key[selected_key].file_name} 已删除。"
            )
            st.rerun()

    delete_notice = st.session_state.pop("user_paper_delete_notice", "")
    if delete_notice:
        st.success(delete_notice)


def _render_user_paper_answer(answer: EvidenceAnswer) -> None:
    """展示回答、检索方式和带文件页码的原文证据。"""

    st.divider()
    st.subheader("基于用户 PDF 的回答")
    if answer.success:
        st.markdown(answer.answer)
    else:
        st.error(answer.message)
    st.caption(f"回答模式：{answer.mode}")
    if answer.success and answer.message:
        st.info(answer.message)

    if not answer.evidence:
        return
    st.subheader("原文证据")
    for index, item in enumerate(answer.evidence, start=1):
        with st.expander(
            f"证据 {index} · {item.chunk.citation}",
            expanded=index == 1,
        ):
            sources = "、".join(item.retrieval_sources) or "BM25"
            st.caption(
                f"检索来源：{sources}　|　"
                f"BM25：{item.bm25_score:.3f}　|　"
                f"语义：{item.semantic_score:.3f}"
            )
            st.write(item.chunk.text)
