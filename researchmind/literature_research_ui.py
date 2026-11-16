"""Semantic Scholar 公开检索与用户 PDF 本地 RAG 的 Streamlit 页面。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import streamlit as st

from agents.literature_research_agent import (
    LiteratureResearchAgent,
    LiteratureResearchResult,
)
from config import settings
from researchmind.adolescent_library import create_adolescent_workflow
from researchmind.analysis_status_ui import schedule_for_domain
from researchmind.literature_import import import_candidates
from researchmind.literature_retrieval import MultiSourceRetriever
from services.llm_service import LLMService
from services.rag_service import EvidenceAnswer
from services.user_paper_rag import (
    UserPaperRAGError,
    UserPaperRAGService,
    UserPaperRecord,
)
from services.vector_service import VectorStoreError


@st.cache_resource
def get_multi_source_retriever() -> MultiSourceRetriever:
    """复用统一多源检索编排器（含进程内缓存与限速）。"""

    return MultiSourceRetriever(settings)


@st.cache_resource
def get_literature_research_agent() -> LiteratureResearchAgent:
    """复用 Function Calling Agent、模型服务与统一多源检索器。"""

    return LiteratureResearchAgent(
        settings,
        LLMService(settings),
        retriever=get_multi_source_retriever(),
    )


@st.cache_resource
def get_user_paper_rag_service() -> UserPaperRAGService:
    """复用隔离的用户论文 Chroma 服务。"""

    return UserPaperRAGService(settings)


@st.cache_resource
def get_public_import_workflow():
    """复用可写的青少年专题工作流，供公开检索结果导入论文库。"""

    return create_adolescent_workflow(settings)


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
    """渲染五类公开学术来源的元数据检索、失败隔离与综述。"""

    st.info(
        "公开来源按 Semantic Scholar → arXiv → OpenAlex → Crossref → PubMed "
        "顺序聚合检索：只读取题名、作者、年份、摘要、DOI 与开放链接，"
        "不爬取知网，也不自动下载全文。"
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
                "例如：青少年体育锻炼干预与体质健康水平之间有什么关系？"
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


def _render_source_board(report: dict[str, Any]) -> None:
    """渲染各数据源的状态、返回数量、耗时与失败建议看板。"""

    sources = report.get("sources") if report else []
    if not sources:
        return
    label = {
        "success": "成功",
        "empty": "无结果",
        "failed": "失败",
        "disabled": "未启用",
    }
    rows = [
        {
            "数据源": source.get("source", ""),
            "状态": label.get(source.get("status"), source.get("status", "")),
            "返回数": source.get("returned", 0),
            "耗时(ms)": source.get("elapsed_ms", 0),
            "尝试次数": source.get("attempts", 0),
            "失败原因 / 建议": source.get("error") or source.get("suggestion") or "—",
        }
        for source in sources
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_public_result(result: LiteratureResearchResult) -> None:
    """展示可追溯的公开检索结果、综述与 GB/T 7714 引用。"""

    st.divider()
    report = result.report or {}
    all_failed = bool(report.get("all_failed")) if report else not result.papers

    if not result.success or all_failed:
        st.error(
            result.message
            or "本轮所有公开数据源均未返回可核验论文，未生成综述或虚构引用。"
        )
        _render_source_board(report)
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "去重后论文", report.get("total_unique", len(result.papers))
    )
    metric_columns[1].metric("可核验 DOI", report.get("verified_doi_count", 0))
    metric_columns[2].metric("含摘要", report.get("abstract_count", 0))
    metric_columns[3].metric("开放全文", report.get("fulltext_count", 0))

    st.caption(f"原始研究问题：{result.original_query}")
    st.caption(f"实际检索式：{result.optimized_query or '未生成'}")
    expanded = report.get("expanded_queries") or ()
    if len(expanded) > 1:
        st.caption(f"同义词扩展：{'、'.join(expanded[1:])}")

    _render_source_board(report)

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
            abstract = paper.abstract.strip()
            content_level = "含摘要" if abstract else "仅题录（无摘要）"
            access_level = "开放全文" if paper.pdf_url.strip() else "未获取全文"
            doi_level = "DOI 可核验" if paper.doi.strip() else "DOI 缺失，需人工核验"
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
                st.caption(f"{doi_level} · {content_level} · {access_level}")
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

        _render_import_panel(result)

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


def _render_import_panel(result: LiteratureResearchResult) -> None:
    """把勾选的候选论文导入专题论文库，并调度增量派生分析。"""

    candidates = tuple(result.candidates or ())
    if not candidates:
        return

    st.subheader("导入专题论文库")
    st.caption(
        "仅导入你勾选的论文；不符合专题方向的记录会被相关性判断跳过。"
        "导入后可在论文库自主删除，删除会同步清理分析缓存、索引与图谱关联。"
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = st.multiselect(
        "选择要导入的论文",
        options=list(by_id),
        format_func=lambda cid: by_id[cid].title,
        key="public_import_select",
    )
    if not st.button(
        "导入所选到专题论文库",
        type="primary",
        icon=":material/library_add:",
        disabled=not selected,
    ):
        return

    workflow = get_public_import_workflow()
    domain = workflow.ensure_default_domain()
    chosen = tuple(by_id[cid] for cid in selected)
    try:
        with st.spinner("正在写入论文库并校验记录..."):
            imported = import_candidates(
                workflow, domain, chosen, download_open_pdf=False
            )
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"论文导入失败：{exc}")
        return

    parts = [
        f"新导入 {imported.imported} 篇",
        f"补齐元数据 {imported.updated} 篇",
        f"重复跳过 {imported.skipped_duplicates} 篇",
        f"专题不相关 {imported.skipped_ineligible} 篇",
        f"失败 {imported.failed} 篇",
    ]
    st.success("；".join(parts))
    if imported.messages:
        with st.expander("导入明细"):
            for message in imported.messages:
                st.caption(message)
    if imported.changed:
        schedule_for_domain(workflow, domain)
        st.caption(
            "已标记专题分析待更新：分歧分析、知识图谱、综述与测评将基于"
            "新增论文增量重算，不会重新处理全部论文。"
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
