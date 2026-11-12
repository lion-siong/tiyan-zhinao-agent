"""ResearchMind 的 Streamlit 页面。

本模块只负责界面交互，论文检索、阅读、分析、组卷和评分均由
``researchmind`` 包中的工作流与 Agent 完成。所有耗时的完整分析都放在按钮
之后执行，页面打开时只读取已有结果和轻量计数。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import streamlit as st

from researchmind.agents import (
    FIELD_LABELS,
    chinese_evidence_option_text,
    chinese_only_excerpt,
)
from researchmind.collectors import (
    CollectionResult,
    LiteratureCollectionError,
    LiteratureCollector,
)
from researchmind.models import (
    AssessmentQuestion,
    CandidatePaper,
    DebateAnalysis,
    EvidenceRef,
    PaperAnalysis,
    PaperRecord,
    ResearchDomain,
    ResearchSynthesis,
    ScoreResult,
)
from researchmind.storage import ResearchMindStorageError
from researchmind.workflow import (
    DEFAULT_DEBATE_TOPIC,
    LearningCycleResult,
    LibrarySyncResult,
    PipelineResult,
    ResearchMindWorkflow,
    ResearchMindWorkflowError,
)
from researchmind.zotero_service import ZoteroError, ZoteroService
from tools.library_ingest import (
    IncomingPDF,
    IngestResult,
    ingest_pdf_files,
    remove_library_documents,
)


RESEARCHMIND_PAGES = (
    "学术领域管理",
    "文献自动采集",
    "我的论文库",
    "论文智能阅读",
    "学术观点对比",
    "学术分歧分析",
    "知识图谱",
    "AI 学习测评",
    "错题本",
    "学习成长报告",
)

_FIELD_ORDER = (
    "purpose",
    "population",
    "methods",
    "results",
    "conclusions",
    "limitations",
)


@st.cache_resource
def get_researchmind_workflow() -> ResearchMindWorkflow:
    """创建共享工作流；SQLite 连接仍由存储层按操作短连接管理。"""

    return ResearchMindWorkflow()


def render_researchmind_page(page: str) -> None:
    """根据导航名称渲染 ResearchMind 页面。"""

    workflow = get_researchmind_workflow()
    domain = _render_domain_selector(workflow)
    renderers = {
        "学术领域管理": _render_domain_management,
        "文献自动采集": _render_literature_collection,
        "我的论文库": _render_library,
        "论文智能阅读": _render_paper_reading,
        "学术观点对比": _render_viewpoint_comparison,
        "学术分歧分析": _render_debate_analysis,
        "知识图谱": _render_knowledge_graph,
        "AI 学习测评": _render_learning_assessment,
        "错题本": _render_wrong_answers,
        "学习成长报告": _render_learning_report,
    }
    renderer = renderers.get(page)
    if renderer is None:
        st.error(f"未知 ResearchMind 页面：{page}")
        return
    try:
        renderer(workflow, domain)
    except (
        ResearchMindStorageError,
        ResearchMindWorkflowError,
        LiteratureCollectionError,
        ZoteroError,
        ValueError,
        OSError,
    ) as exc:
        st.error(f"当前操作未完成：{exc}")
    except Exception as exc:
        st.error(f"页面暂时无法加载：{exc}")
        st.caption("数据不会被自动删除。请检查文献库、数据库或网络配置后重试。")


def _render_domain_selector(
    workflow: ResearchMindWorkflow,
) -> ResearchDomain:
    """显示当前领域选择器，并确保选中领域有可用论文库。"""

    default_domain = workflow.ensure_default_domain()
    domains = workflow.store.list_domains()
    domain_by_id = {domain.domain_id: domain for domain in domains}
    ids = tuple(domain_by_id)
    session_key = "researchmind_active_domain"
    active_id = st.session_state.get(session_key)
    if active_id not in domain_by_id:
        active_id = default_domain.domain_id
        st.session_state[session_key] = active_id

    selected_id = st.selectbox(
        "当前研究领域",
        options=ids,
        index=ids.index(active_id),
        format_func=lambda value: domain_by_id[value].name,
        key="researchmind_domain_selector",
        help="所有阅读、分析、题库和学习记录都会归入当前领域。",
    )
    st.session_state[session_key] = selected_id
    domain = domain_by_id[selected_id]

    if workflow.store.paper_count(domain.domain_id) == 0:
        with st.spinner("首次进入该领域，正在同步项目内置真实论文库..."):
            sync_result = workflow.sync_bundled_library(domain=domain)
        st.caption(sync_result.message)
        domain = workflow.store.get_domain(domain.domain_id) or domain
    return domain


def _render_domain_management(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "ACADEMIC DOMAIN",
        "学术领域管理",
        "用领域组织论文、知识体系、题库和学习记录。",
    )
    st.info(
        "项目内置真实论文库会在首次进入新领域时同步到该领域。"
        "每个领域使用独立的论文与证据标识，不会覆盖其他领域的数据。",
        icon=":material/hub:",
    )

    domains = workflow.store.list_domains()
    counts = workflow.store.counts(domain.domain_id)
    metrics = st.columns(4)
    metrics[0].metric("研究领域", f"{len(domains)} 个")
    metrics[1].metric("当前论文", f"{counts['papers']} 篇")
    metrics[2].metric("带页码证据", f"{counts['evidence']} 条")
    metrics[3].metric("已生成试题", f"{counts['questions']} 道")

    with st.container(border=True):
        st.subheader(domain.name)
        st.write(domain.description)
        st.caption("关键词：" + "、".join(domain.keywords))
        st.caption(f"领域 ID：{domain.domain_id}")

    with st.form("researchmind_create_domain_form"):
        st.subheader("新建研究领域")
        name = st.text_input(
            "领域名称",
            placeholder="例如：青少年运动技能学习与反馈",
        )
        description = st.text_area(
            "领域说明",
            placeholder="说明研究对象、核心问题和希望积累的证据范围。",
            height=90,
        )
        keyword_text = st.text_input(
            "检索关键词",
            placeholder="运动技能、反馈、青少年（使用顿号或逗号分隔）",
        )
        submitted = st.form_submit_button(
            "创建领域",
            type="primary",
            icon=":material/add_circle:",
        )
    if submitted:
        keywords = _split_keywords(keyword_text)
        if not name.strip():
            st.error("领域名称不能为空。")
        elif not keywords:
            st.error("请至少填写一个检索关键词。")
        else:
            created = workflow.create_domain(
                name=name,
                description=description,
                keywords=keywords,
            )
            st.session_state["researchmind_active_domain"] = created.domain_id
            st.session_state["researchmind_domain_selector"] = created.domain_id
            st.rerun()

    st.subheader("全部领域")
    rows = []
    for item in domains:
        item_counts = workflow.store.counts(item.domain_id)
        rows.append(
            {
                "领域": item.name,
                "关键词": "、".join(item.keywords),
                "论文": item_counts["papers"],
                "证据": item_counts["evidence"],
                "试题": item_counts["questions"],
                "答题记录": item_counts["attempts"],
                "创建时间": _short_time(item.created_at),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="ResearchMind 学术领域清单",
    )

    with st.container(horizontal=True, vertical_alignment="bottom"):
        if st.button(
            "强制同步项目论文库",
            icon=":material/sync:",
            help="重新读取 manifest.json 和证据索引，不重新解析 PDF 全文。",
        ):
            with st.spinner("正在同步真实论文元数据与证据..."):
                result = workflow.sync_bundled_library(
                    domain=domain,
                    force=True,
                )
            _show_sync_result(result)
            st.rerun()


def _render_literature_collection(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "LITERATURE COLLECTOR",
        "文献自动采集",
        "从 OpenAlex 或 Crossref 检索真实元数据，只下载明确开放获取的 PDF。",
    )
    zotero = ZoteroService(workflow.config)
    zotero_status = zotero.status()
    if zotero_status.configured:
        st.success("Zotero Web API 已配置，可同步真实元数据或已取得的本地 PDF。")
    else:
        st.info(
            "Zotero 尚未配置。本地论文库仍可正常使用；"
            "如需同步，请在 .env 中填写 ZOTERO_API_KEY 和 ZOTERO_LIBRARY_ID。"
        )

    with st.form("researchmind_collection_form"):
        query = st.text_input(
            "检索主题",
            value="大学生体育锻炼行为 体质健康",
            placeholder="输入研究领域、关键词或英文检索词",
        )
        controls = st.columns(2)
        limit = controls[0].number_input(
            "检索数量",
            min_value=1,
            max_value=25,
            value=10,
            step=1,
        )
        date_from = controls[1].text_input(
            "起始日期",
            value=workflow.config.researchmind_date_from,
            help="格式：YYYY-MM-DD。默认检索近三年。",
        )
        submitted = st.form_submit_button(
            "检索真实学术元数据",
            type="primary",
            icon=":material/search:",
        )
    if submitted:
        if not query.strip():
            st.error("检索主题不能为空。")
        else:
            with st.spinner("正在访问 OpenAlex / Crossref..."):
                result = LiteratureCollector(workflow.config).search(
                    query,
                    limit=int(limit),
                    date_from=date_from,
                )
            st.session_state["researchmind_collection_result"] = result

    result = st.session_state.get("researchmind_collection_result")
    if not isinstance(result, CollectionResult):
        st.caption(
            "检索结果不会自动写入本地论文库。请先检查题名、作者、DOI 和来源，"
            "再决定是否下载开放 PDF 或同步到 Zotero。"
        )
        return

    st.divider()
    metric_columns = st.columns(4)
    metric_columns[0].metric("来源", result.source)
    metric_columns[1].metric("真实元数据", f"{len(result.candidates)} 篇")
    metric_columns[2].metric("开放 PDF", f"{sum(c.has_open_pdf for c in result.candidates)} 篇")
    metric_columns[3].metric("起始日期", result.date_from)
    for warning in result.warnings:
        st.warning(warning)
    if not result.candidates:
        st.info("来源没有返回可核验论文，未写入任何记录，也没有生成替代引用。")
        return

    rows = [
        {
            "候选 ID": item.candidate_id,
            "题名": item.title,
            "作者": item.authors_text if hasattr(item, "authors_text") else "、".join(item.authors),
            "年份": item.year,
            "期刊": item.journal,
            "DOI": item.doi,
            "来源": item.source,
            "开放 PDF": "有" if item.has_open_pdf else "仅元数据",
        }
        for item in result.candidates
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="学术来源返回的真实论文元数据",
    )
    st.caption(
        "“仅元数据”表示尚未获得可确认的开放 PDF；系统不会绕过付费墙，"
        "也不会把候选元数据冒充为已解析全文。"
    )

    candidate_by_id = {item.candidate_id: item for item in result.candidates}
    selected_id = st.selectbox(
        "选择要处理的论文",
        options=tuple(candidate_by_id),
        format_func=lambda value: candidate_by_id[value].title,
        key="researchmind_candidate_selector",
    )
    candidate = candidate_by_id[selected_id]
    _render_candidate_detail(candidate)

    action_columns = st.columns(2)
    with action_columns[0]:
        if st.button(
            "下载开放 PDF 并写入项目论文库",
            type="primary",
            width="stretch",
            icon=":material/download:",
            disabled=not candidate.has_open_pdf,
        ):
            _download_and_ingest_candidate(
                workflow,
                domain,
                candidate,
            )
    with action_columns[1]:
        if st.button(
            "仅同步元数据到 Zotero",
            width="stretch",
            icon=":material/library_add:",
            disabled=not zotero_status.configured,
        ):
            try:
                with st.spinner("正在把真实元数据写入 Zotero..."):
                    item_key = zotero.add_candidate(candidate)
            except ZoteroError as exc:
                st.error(f"Zotero 同步失败：{exc}")
            else:
                st.success(f"已写入 Zotero，条目 key：{item_key}")
                st.caption("Zotero 同步的是当前候选元数据，不代表项目已取得全文。")


def _render_library(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "MY LIBRARY",
        "我的论文库",
        "查看当前领域真实论文、证据数量、开放来源和本地 PDF 状态。",
    )
    delete_notice = st.session_state.pop(
        "researchmind_paper_delete_notice",
        "",
    )
    if delete_notice:
        st.success(delete_notice)
    papers = workflow.store.list_papers(domain.domain_id)
    evidence_count = workflow.store.evidence_count(domain.domain_id)
    zotero = ZoteroService(workflow.config)
    zotero_status = zotero.status()
    metrics = st.columns(4)
    metrics[0].metric("论文", f"{len(papers)} 篇")
    metrics[1].metric("证据片段", f"{evidence_count} 条")
    metrics[2].metric(
        "PDF 可用",
        f"{sum(Path(paper.pdf_path).is_file() for paper in papers)} 篇",
    )
    metrics[3].metric(
        "年份覆盖",
        _year_range(papers),
    )

    if not papers:
        st.warning(
            "当前领域还没有论文。请通过“文献自动采集”导入开放论文，"
            "或使用其他已授权的 PDF 建立论文库。"
        )
        if st.button("刷新论文库状态", icon=":material/sync:"):
            result = workflow.sync_bundled_library(domain=domain, force=True)
            _show_sync_result(result)
            st.rerun()
        return

    filter_columns = st.columns([2, 1, 1])
    search = filter_columns[0].text_input(
        "筛选论文",
        placeholder="输入题名、作者、期刊或 DOI",
    )
    years = tuple(sorted({paper.year for paper in papers}, reverse=True))
    selected_year = filter_columns[1].selectbox(
        "年份",
        options=("全部", *years),
    )
    selected_access = filter_columns[2].selectbox(
        "PDF 状态",
        options=("全部", "本地 PDF 可用", "仅元数据"),
    )

    lowered = search.strip().lower()
    filtered = []
    for paper in papers:
        haystack = " ".join(
            (
                paper.title,
                paper.authors_text,
                paper.journal,
                paper.doi,
                " ".join(paper.keywords),
            )
        ).lower()
        if lowered and lowered not in haystack:
            continue
        if selected_year != "全部" and paper.year != selected_year:
            continue
        exists = Path(paper.pdf_path).is_file()
        if selected_access == "本地 PDF 可用" and not exists:
            continue
        if selected_access == "仅元数据" and exists:
            continue
        filtered.append(paper)

    rows = [
        {
            "题名": paper.title,
            "作者": paper.authors_text,
            "年份": paper.year,
            "期刊": paper.journal,
            "DOI": paper.doi,
            "关键词": "、".join(paper.keywords),
            "PDF": "可用" if Path(paper.pdf_path).is_file() else "缺失",
        }
        for paper in filtered
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="当前研究领域的真实论文清单",
    )
    st.download_button(
        "下载论文清单 CSV",
        data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
        file_name="researchmind_papers.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    st.subheader("论文详情")
    paper_by_id = {paper.paper_id: paper for paper in filtered}
    if paper_by_id:
        selected_paper_id = st.selectbox(
            "选择论文",
            options=tuple(paper_by_id),
            format_func=lambda value: paper_by_id[value].title,
            key="researchmind_library_paper",
        )
        selected_paper = paper_by_id[selected_paper_id]
        _render_paper_metadata(selected_paper)
        pdf_available = Path(selected_paper.pdf_path).is_file()
        if st.button(
            "同步元数据与 PDF 到 Zotero",
            icon=":material/library_add:",
            disabled=not zotero_status.configured or not pdf_available,
            help="仅在 Zotero 已配置且本地 PDF 可读取时启用。",
        ):
            try:
                with st.spinner("正在上传论文元数据与 PDF 到 Zotero..."):
                    parent_key, attachment_key = zotero.add_paper_with_pdf(
                        selected_paper
                    )
            except ZoteroError as exc:
                st.error(f"Zotero 同步失败：{exc}")
            else:
                st.success(
                    "已上传到 Zotero。"
                    f"父条目 key：{parent_key}；PDF 附件 key：{attachment_key}"
                )
        if not zotero_status.configured:
            st.caption(
                "Zotero 尚未配置。请在 .env 中填写 ZOTERO_API_KEY 和 "
                "ZOTERO_LIBRARY_ID；本地论文库不受影响。"
            )
        elif not pdf_available:
            st.caption("该论文只有元数据，本地 PDF 缺失，因此不能上传附件。")
    else:
        st.info("当前筛选条件下没有论文。")

    if papers:
        _render_library_removal(workflow, domain, papers)

    if st.button(
        "检查并同步最新论文库",
        icon=":material/sync:",
        help="同步 manifest.json 和证据索引中的新增论文。",
    ):
        result = workflow.sync_bundled_library(domain=domain)
        _show_sync_result(result)
        if not result.used_cached_sync:
            st.rerun()


def _render_library_removal(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    papers: tuple[PaperRecord, ...],
) -> None:
    """提供带双重确认的项目论文库永久删除入口。"""

    with st.expander("删除论文", icon=":material/delete_forever:"):
        st.warning(
            "删除会永久移除本地 PDF、证据索引、研究框架和当前领域的"
            "相关分析结果，并影响其他使用同一项目论文库的研究领域。"
        )
        paper_by_id = {paper.paper_id: paper for paper in papers}
        selected_ids = tuple(
            st.multiselect(
                "选择要删除的论文",
                options=tuple(paper_by_id),
                format_func=lambda value: paper_by_id[value].title,
                key="researchmind_library_delete_scope",
            )
        )
        confirmed = st.checkbox(
            "我确认永久删除所选论文及其本地文件",
            key="researchmind_library_delete_confirm",
        )
        if st.button(
            "永久删除所选论文",
            icon=":material/delete_forever:",
            disabled=not selected_ids or not confirmed,
            key="researchmind_library_delete_button",
        ):
            _delete_library_papers(workflow, domain, selected_ids)


def _delete_library_papers(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    paper_ids: tuple[str, ...],
) -> None:
    """删除论文资产、重建派生索引，并清理当前领域的失效结果。"""

    selected_papers = tuple(
        paper
        for paper_id in paper_ids
        if (paper := workflow.store.get_paper(paper_id)) is not None
    )
    if not selected_papers:
        st.error("未找到要删除的论文，请刷新页面后重试。")
        return

    try:
        with st.spinner("正在删除 PDF、重建证据索引和研究框架..."):
            removal = remove_library_documents(
                [Path(paper.pdf_path).name for paper in selected_papers],
                workflow.knowledge_base_dir,
            )
            workflow.store.invalidate_domain_derivations(
                domain.domain_id,
                paper_ids,
            )
            sync_result = workflow.sync_bundled_library(
                domain=domain,
                force=True,
            )
    except (KnowledgeBaseError, OSError, ResearchMindWorkflowError) as exc:
        st.error(f"论文删除未完成：{exc}")
        return

    removed_titles = "、".join(paper.title for paper in selected_papers)
    st.session_state["researchmind_paper_delete_notice"] = (
        f"已删除 {len(selected_papers)} 篇论文：{removed_titles}。"
        f"当前领域剩余 {sync_result.total_papers} 篇论文。"
    )
    if removal.vector_warning:
        st.warning(removal.vector_warning)
    for key in tuple(st.session_state):
        if (
            key.startswith("researchmind_analysis_")
            or key.startswith("researchmind_debate_")
            or key.startswith("researchmind_synthesis_")
            or key
            in {
                "researchmind_active_questions",
                "researchmind_learning_result",
            }
        ):
            st.session_state.pop(key, None)
    st.rerun()


def _render_paper_reading(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "PAPER READER",
        "论文智能阅读",
        "从真实 PDF 原文提取研究目的、对象、方法、结果、结论和局限。",
    )
    st.warning(
        "当前使用可解释的本地规则定位原文。缺失字段会显示“证据不足”，"
        "不会由系统补写研究结论或实验数据。",
        icon=":material/fact_check:",
    )
    papers = workflow.store.list_papers(domain.domain_id)
    if not papers:
        st.info("当前领域没有论文，请先在“我的论文库”同步项目内置论文库。")
        return

    selected_id = st.selectbox(
        "选择论文",
        options=tuple(paper.paper_id for paper in papers),
        format_func=lambda value: next(
            paper.title for paper in papers if paper.paper_id == value
        ),
        key="researchmind_reader_paper",
    )
    paper = next(paper for paper in papers if paper.paper_id == selected_id)
    _render_paper_metadata(paper)

    analysis = workflow.store.get_analysis(selected_id)
    if analysis is not None:
        st.success("已读取该论文的结构化结果。")
        _render_analysis(analysis)
    else:
        st.info("该论文尚未执行结构化阅读。")

    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "解析当前论文",
            type="primary",
            icon=":material/document_scanner:",
        ):
            with st.spinner("正在从已建证据索引中定位六个研究字段..."):
                analysis = workflow.reader.read_paper(selected_id)
            st.session_state[f"researchmind_analysis_{selected_id}"] = analysis
            st.rerun()
        if st.button(
            "解析当前领域全部论文",
            icon=":material/playlist_play:",
            help="逐篇运行本地规则阅读，不会调用大模型。",
        ):
            with st.spinner("正在解析当前领域的全部论文..."):
                analyses = workflow.reader.read_domain(domain.domain_id)
            st.success(f"已完成 {len(analyses)} 篇论文的结构化阅读。")
            st.rerun()

    st.subheader("领域阅读概览")
    overview = []
    for item in papers:
        item_analysis = workflow.store.get_analysis(item.paper_id)
        overview.append(
            {
                "论文": item.title,
                "状态": item_analysis.status if item_analysis else "尚未阅读",
                "已定位字段": (
                    f"{item_analysis.filled_field_count}/6"
                    if item_analysis
                    else "-"
                ),
                "阅读方式": (
                    item_analysis.extraction_mode
                    if item_analysis
                    else "待处理"
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(overview),
        width="stretch",
        hide_index=True,
        alt="领域内论文结构化阅读状态",
    )


def _render_viewpoint_comparison(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "ACADEMIC DEBATE",
        "学术观点对比",
        "比较论文观点、样本、研究设计和结论边界，观点必须回指原文页码。",
    )
    st.caption(
        "相关性证据不能自动升级为因果关系。系统不会把有限论文中的相同方向"
        "直接写成普遍学术共识。"
    )
    topic = st.text_input(
        "争议或比较主题",
        value=_default_topic(domain),
        key="researchmind_viewpoint_topic",
    )
    if st.button(
        "运行观点对比",
        type="primary",
        icon=":material/compare_arrows:",
    ):
        if not topic.strip():
            st.error("比较主题不能为空。")
        else:
            with st.spinner("正在定位不同论文的可核验观点..."):
                debate = workflow.debate_analyzer.analyze(
                    domain.domain_id,
                    topic,
                )
            st.session_state[f"researchmind_debate_{domain.domain_id}"] = debate
            st.success("观点对比已完成。")

    debate = st.session_state.get(
        f"researchmind_debate_{domain.domain_id}"
    )
    if debate is None:
        debate = workflow.store.latest_debate(domain.domain_id)
    if debate is None:
        st.info("尚未生成观点对比。点击上方按钮后开始分析。")
        return
    _render_debate_summary(workflow, debate)


def _render_debate_analysis(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "DISAGREEMENT MAP",
        "学术分歧分析",
        "区分观点分歧、方法差异和证据不足，并明确研究空白。",
    )
    topic = st.text_input(
        "分析主题",
        value=_default_topic(domain),
        key="researchmind_debate_topic",
    )
    if st.button(
        "重新分析学术分歧",
        type="primary",
        icon=":material/account_tree:",
    ):
        if not topic.strip():
            st.error("分析主题不能为空。")
        else:
            with st.spinner("正在比较论据、方法和证据强度..."):
                debate = workflow.debate_analyzer.analyze(
                    domain.domain_id,
                    topic,
                )
            st.session_state[f"researchmind_debate_{domain.domain_id}"] = debate
            st.rerun()

    debate = st.session_state.get(
        f"researchmind_debate_{domain.domain_id}"
    )
    if debate is None:
        debate = workflow.store.latest_debate(domain.domain_id)
    if debate is None:
        st.info("当前领域还没有分歧分析结果。")
        return

    columns = st.columns(4)
    columns[0].metric("争议主题", debate.topic)
    columns[1].metric("可追溯观点", f"{len(debate.viewpoints)} 个")
    columns[2].metric("证据强度", debate.evidence_strength)
    columns[3].metric(
        "涉及论文",
        f"{len({paper_id for view in debate.viewpoints for paper_id in view.supporting_papers})} 篇",
    )
    with st.container(border=True):
        st.markdown("**共识边界**")
        st.write(debate.consensus_summary)
        st.markdown("**分歧判断**")
        st.write(debate.disagreement_summary)

    left, right = st.columns(2)
    with left:
        st.subheader("可能原因")
        _render_bullet_list(debate.possible_reasons)
        st.subheader("研究局限")
        _render_bullet_list(debate.limitations)
    with right:
        st.subheader("研究空白")
        _render_bullet_list(debate.research_gaps)
        st.subheader("证据边界")
        st.warning(debate.evidence_strength, icon=":material/balance:")

    st.subheader("结构化观点与证据")
    _render_viewpoints(workflow, debate)
    st.download_button(
        "下载学术分歧分析 JSON",
        data=json.dumps(
            debate.to_dict(),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        file_name="academic_debate.json",
        mime="application/json",
        icon=":material/download:",
    )


def _render_knowledge_graph(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "KNOWLEDGE GRAPH",
        "知识图谱",
        "把概念、论文、研究维度与争议关系组织成可追溯的领域知识体系。",
    )
    synthesis = workflow.store.get_latest_synthesis(domain.domain_id)
    if st.button(
        "更新知识体系与图谱",
        type="primary",
        icon=":material/hub:",
        help="依次执行结构化阅读、观点分析和知识体系归纳。",
    ):
        with st.spinner("正在构建领域知识体系，这可能需要十几秒..."):
            workflow.reader.read_domain(domain.domain_id)
            debate = workflow.store.latest_debate(domain.domain_id)
            if debate is None:
                debate = workflow.debate_analyzer.analyze(
                    domain.domain_id,
                    _default_topic(domain),
                )
            synthesis = workflow.synthesizer.synthesize(
                domain.domain_id,
                debate=debate,
            )
        st.session_state[f"researchmind_synthesis_{domain.domain_id}"] = synthesis
        st.success("知识体系与图谱已更新。")

    synthesis = st.session_state.get(
        f"researchmind_synthesis_{domain.domain_id}"
    ) or synthesis
    if synthesis is None:
        st.info("当前领域还没有知识体系。点击上方按钮开始构建。")
        return

    metrics = st.columns(4)
    metrics[0].metric("知识节点", f"{len(synthesis.nodes)} 个")
    metrics[1].metric("关系边", f"{len(synthesis.edges)} 条")
    metrics[2].metric("来源论文", f"{len(synthesis.source_paper_ids)} 篇")
    metrics[3].metric("生成方式", synthesis.mode)
    st.caption(f"生成时间：{_short_time(synthesis.generated_at)}")

    _render_graphviz_or_table(synthesis)

    tabs = st.tabs(
        (
            "概念与框架",
            "共识与分歧",
            "局限与空白",
            "创新方向",
        )
    )
    with tabs[0]:
        st.markdown("**核心概念**")
        _render_bullet_list(synthesis.concepts)
        st.markdown("**理论框架链**")
        if synthesis.theory_framework:
            st.markdown(
                " → ".join(
                    f"**{item}**" for item in synthesis.theory_framework
                )
            )
        else:
            st.info("当前证据不足以构建完整理论框架链。")
    with tabs[1]:
        st.markdown("**学术共识边界**")
        _render_bullet_list(synthesis.consensus)
        st.markdown("**学术分歧**")
        _render_bullet_list(synthesis.disagreements)
    with tabs[2]:
        st.markdown("**研究局限**")
        _render_bullet_list(synthesis.limitations)
        st.markdown("**研究空白**")
        _render_bullet_list(synthesis.research_gaps)
    with tabs[3]:
        _render_bullet_list(synthesis.innovation_directions)

    st.download_button(
        "下载知识体系 JSON",
        data=json.dumps(
            synthesis.to_dict(),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        file_name="research_knowledge_synthesis.json",
        mime="application/json",
        icon=":material/download:",
    )


def _render_learning_assessment(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "LEARNING ASSESSMENT",
        "AI 学习测评",
        "根据论文证据自动组卷，客观题自动评分，主观题按评分细则批改。",
    )
    if not workflow.store.list_papers(domain.domain_id):
        st.info("当前领域没有论文，无法生成有来源的试题。")
        return

    with st.form("researchmind_generate_questions_form"):
        st.subheader("自动组卷")
        question_types = st.multiselect(
            "题型",
            options=(
                "单项选择题",
                "多项选择题",
                "判断题",
                "简答题",
                "论述题",
                "文献对比分析题",
            ),
            default=(
                "单项选择题",
                "多项选择题",
                "判断题",
                "简答题",
                "文献对比分析题",
            ),
        )
        topic = st.text_input(
            "对比主题",
            value=_default_topic(domain),
        )
        generate = st.form_submit_button(
            "生成新试卷",
            type="primary",
            icon=":material/quiz:",
        )
    if generate:
        if not question_types:
            st.error("请至少选择一种题型。")
        else:
            with st.spinner("正在根据题库论文与原文证据生成试题..."):
                questions = workflow.assessment.generate_question_set(
                    domain.domain_id,
                    question_types=question_types,
                    debate_topic=topic,
                )
            st.session_state["researchmind_active_questions"] = tuple(
                question.question_id for question in questions
            )
            st.session_state["researchmind_assessment_topic"] = topic
            st.session_state.pop("researchmind_learning_result", None)
            st.success(f"已生成 {len(questions)} 道有来源的试题。")

    questions = _active_questions(workflow, domain)
    if not questions:
        st.info("尚未生成试卷。可选择题型后点击“生成新试卷”。")
        return

    attempts = workflow.store.list_attempts(domain.domain_id, limit=100)
    answers: dict[str, str] = {}
    with st.form("researchmind_answer_sheet"):
        st.subheader("答题")
        for number, question in enumerate(questions, start=1):
            with st.container(border=True):
                st.markdown(f"**第 {number} 题 · {question.question_type}**")
                st.write(question.prompt)
                st.caption(
                    f"难度：{question.difficulty}　|　"
                    f"知识点：{'、'.join(question.knowledge_tags)}　|　"
                    f"满分：{question.max_score:g}"
                )
                answer = _render_question_input(question, number)
                answers[question.question_id] = answer
                with st.expander("查看本题评分标准"):
                    _render_question_rubric(question)
        submitted = st.form_submit_button(
            "提交并批改",
            type="primary",
            icon=":material/grading:",
        )

    if submitted:
        with st.spinner("正在按评分细则批改并更新能力掌握度..."):
            result = workflow.grade_learning_cycle(
                domain.domain_id,
                answers=answers,
                topic=st.session_state.get(
                    "researchmind_assessment_topic",
                    _default_topic(domain),
                ),
                generate_next=True,
            )
        st.session_state["researchmind_learning_result"] = result

    result = st.session_state.get("researchmind_learning_result")
    if isinstance(result, LearningCycleResult):
        _render_learning_cycle_result(workflow, result)

    if attempts:
        st.caption(
            f"当前领域累计答题记录 {len(attempts)} 条。"
            "每次提交都会更新掌握度、错题本和下一次组卷。"
        )


def _render_wrong_answers(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "WRONG ANSWER BOOK",
        "错题本",
        "查看低于 60% 的答题记录、标准答案、原文证据和评分细则缺分原因。",
    )
    wrong_attempts = workflow.store.list_wrong_attempts(domain.domain_id)
    all_attempts = workflow.store.list_attempts(domain.domain_id)
    metrics = st.columns(4)
    metrics[0].metric("累计答题", f"{len(all_attempts)} 条")
    metrics[1].metric("错题记录", f"{len(wrong_attempts)} 条")
    metrics[2].metric(
        "涉及知识点",
        f"{len({tag for item in wrong_attempts for tag in item.knowledge_tags})} 个",
    )
    metrics[3].metric(
        "待复习题型",
        f"{len({item.question_type for item in wrong_attempts})} 类",
    )
    if not wrong_attempts:
        st.success("目前没有低于 60% 的答题记录。", icon=":material/task_alt:")
        return

    question_type = st.selectbox(
        "筛选题型",
        options=("全部", *sorted({item.question_type for item in wrong_attempts})),
    )
    filtered = tuple(
        item
        for item in wrong_attempts
        if question_type == "全部" or item.question_type == question_type
    )
    rows = [
        {
            "时间": _short_time(item.created_at),
            "题型": item.question_type,
            "题目": item.prompt,
            "得分": f"{item.score:g}/{item.max_score:g}",
            "知识点": "、".join(item.knowledge_tags),
            "批改方式": item.mode,
        }
        for item in filtered
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="ResearchMind 错题记录",
    )

    for index, attempt in enumerate(filtered, start=1):
        question = workflow.store.get_question(attempt.question_id)
        with st.expander(
            f"{index:02d}　{attempt.question_type}　"
            f"{attempt.score:g}/{attempt.max_score:g} 分"
        ):
            st.markdown(f"**题目：** {attempt.prompt}")
            st.markdown("**我的答案：**")
            st.write(
                _format_question_answer(question, attempt.user_answer)
                if question is not None
                else (attempt.user_answer or "（未作答）")
            )
            st.markdown("**批改反馈：**")
            st.write(attempt.feedback)
            if question is None:
                st.warning("原始题目记录缺失，无法展示标准答案和来源。")
                continue
            _render_question_sources(question)

    recommendation = workflow.tutor.recommend(domain.domain_id)
    st.subheader("错题复习建议")
    st.write(recommendation.summary)
    _render_recommendation(workflow, recommendation)


def _render_learning_report(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "LEARNING REPORT",
        "学习成长报告",
        "展示五个能力维度的掌握度、答题趋势和下一轮个性化学习安排。",
    )
    mastery = workflow.store.ensure_mastery(domain.domain_id)
    attempts = workflow.store.list_attempts(domain.domain_id)
    wrong_attempts = workflow.store.list_wrong_attempts(domain.domain_id)
    total_score = sum(item.score for item in attempts)
    total_max = sum(item.max_score for item in attempts)
    overall = 100 * total_score / total_max if total_max else 0.0
    metrics = st.columns(4)
    metrics[0].metric("累计答题", f"{len(attempts)} 条")
    metrics[1].metric("总体得分率", f"{overall:.1f}%")
    metrics[2].metric("错题记录", f"{len(wrong_attempts)} 条")
    metrics[3].metric(
        "平均掌握度",
        f"{sum(item.score for item in mastery) / len(mastery):.1f}%",
    )

    mastery_frame = pd.DataFrame(
        {
            "能力维度": [item.dimension for item in mastery],
            "掌握度": [item.score for item in mastery],
            "作答次数": [item.attempts_count for item in mastery],
        }
    ).set_index("能力维度")
    st.subheader("五项能力掌握度")
    st.bar_chart(
        mastery_frame[["掌握度"]],
        height=300,
        alt="ResearchMind 五项科研能力掌握度柱状图",
    )
    st.dataframe(
        mastery_frame.reset_index(),
        width="stretch",
        hide_index=True,
        alt="ResearchMind 能力掌握度明细",
    )
    st.caption(
        "初始能力权重为概念记忆 15%、文献理解 20%、对比分析 25%、"
        "批判性思维 20%、学术应用 20%；后续得分按答题记录动态更新。"
    )

    if attempts:
        history_rows = [
            {
                "时间": _short_time(item.created_at),
                "得分率": round(
                    100 * item.score / item.max_score
                    if item.max_score
                    else 0.0,
                    2,
                ),
                "题型": item.question_type,
            }
            for item in reversed(attempts)
        ]
        st.subheader("答题趋势")
        history = pd.DataFrame(history_rows)
        history["序号"] = range(1, len(history) + 1)
        st.line_chart(
            history.set_index("序号")[["得分率"]],
            height=260,
            alt="ResearchMind 历次答题得分率折线图",
        )
    else:
        st.info("还没有答题记录。完成一次 AI 学习测评后会生成真实成长趋势。")

    recommendation = workflow.tutor.recommend(domain.domain_id)
    st.subheader("个性化学习建议")
    st.write(recommendation.summary)
    _render_recommendation(workflow, recommendation)

    if st.button(
        "刷新并生成下一轮学习安排",
        type="primary",
        icon=":material/auto_awesome:",
    ):
        with st.spinner("正在根据薄弱知识点匹配论文与复习计划..."):
            recommendation = workflow.tutor.recommend(domain.domain_id)
            next_questions = workflow.assessment.generate_question_set(
                domain.domain_id,
                question_types=_question_types_for_review(recommendation),
                debate_topic=_default_topic(domain),
            )
            recommendation = workflow.tutor.recommend(
                domain.domain_id,
                next_question_ids=tuple(
                    item.question_id for item in next_questions
                ),
            )
        st.session_state["researchmind_active_questions"] = tuple(
            item.question_id for item in next_questions
        )
        st.session_state["researchmind_assessment_topic"] = _default_topic(
            domain
        )
        st.success(
            f"已生成下一轮 {len(next_questions)} 道针对性试题，"
            "可前往“AI 学习测评”继续作答。"
        )
        _render_recommendation(workflow, recommendation)


def _render_debate_summary(
    workflow: ResearchMindWorkflow,
    debate: DebateAnalysis,
) -> None:
    metrics = st.columns(4)
    metrics[0].metric("主题", debate.topic)
    metrics[1].metric("观点", f"{len(debate.viewpoints)} 个")
    metrics[2].metric("证据强度", debate.evidence_strength)
    metrics[3].metric(
        "证据页码",
        f"{sum(len(item.evidence) for item in debate.viewpoints)} 条",
    )
    with st.container(border=True):
        st.markdown("**共识边界**")
        st.write(debate.consensus_summary)
        st.markdown("**分歧判断**")
        st.write(debate.disagreement_summary)
    _render_viewpoints(workflow, debate)
    with st.expander("查看方法差异、局限与研究空白"):
        _render_bullet_list(debate.possible_reasons)
        _render_bullet_list(debate.limitations)
        _render_bullet_list(debate.research_gaps)


def _render_viewpoints(
    workflow: ResearchMindWorkflow,
    debate: DebateAnalysis,
) -> None:
    for number, viewpoint in enumerate(debate.viewpoints, start=1):
        with st.container(border=True):
            st.markdown(f"**观点 {number}　{viewpoint.stance}**")
            st.write(viewpoint.claim)
            paper_titles = _paper_titles(
                workflow,
                viewpoint.supporting_papers,
            )
            st.caption("关联论文：" + "、".join(paper_titles))
            _render_evidence(viewpoint.evidence)


def _render_analysis(analysis: PaperAnalysis) -> None:
    st.caption(analysis.status)
    for field in _FIELD_ORDER:
        label = FIELD_LABELS.get(field, field)
        evidence = analysis.field_evidence(field)
        st.markdown(f"### {label}")
        if not evidence:
            st.info("证据不足：当前原文中没有定位到该字段。")
            continue
        _render_evidence(evidence)


def _render_evidence(evidence: Iterable[EvidenceRef]) -> None:
    for item in evidence:
        with st.container(border=True):
            st.markdown(f"**{item.document_title}**")
            st.caption(
                f"{item.file_name}　|　第 {item.page_number} 页　|　"
                f"证据 ID：{item.chunk_id}"
            )
            st.write(item.quote)


def _render_question_input(
    question: AssessmentQuestion,
    number: int,
) -> str:
    key = f"researchmind_answer_{question.question_id}_{number}"
    if question.question_type == "单项选择题":
        labels = _question_option_labels(question)
        selected = st.radio(
            "选择一个答案",
            options=labels,
            index=None,
            key=key,
        )
        if selected is None:
            return ""
        return _letter(labels.index(selected))
    if question.question_type == "多项选择题":
        labels = _question_option_labels(question)
        selected = st.multiselect(
            "选择所有正确选项",
            options=labels,
            key=key,
        )
        indexes = sorted({labels.index(item) for item in selected})
        return ",".join(_letter(index) for index in indexes)
    if question.question_type == "判断题":
        return st.radio(
            "判断",
            options=question.options,
            index=None,
            horizontal=True,
            key=key,
        ) or ""
    return st.text_area(
        "回答",
        height=130 if question.question_type == "简答题" else 190,
        placeholder="写出核心观点，并尽量对应论文题名、页码和原文证据。",
        key=key,
    )


def _question_option_labels(question: AssessmentQuestion) -> tuple[str, ...]:
    """用中文序号展示选项，选中项通过列表索引回写内部答案。"""

    return tuple(
        f"{_choice_marker(index)}、{option}"
        for index, option in enumerate(_display_question_options(question))
    )


def _display_question_options(
    question: AssessmentQuestion,
) -> tuple[str, ...]:
    """兼容旧题库，确保界面选项有中文且不包含网址和英文。"""

    display: list[str] = []
    seen: set[str] = set()
    for index, raw_option in enumerate(question.options):
        cleaned = chinese_only_excerpt(raw_option, max_chars=160)
        if not cleaned:
            evidence = _matching_option_evidence(question, raw_option)
            if evidence is not None:
                cleaned = chinese_evidence_option_text(
                    evidence,
                    designated=(
                        evidence.document_id
                        in set(question.source_paper_ids)
                    ),
                    ordinal=index + 1,
                )
            else:
                cleaned = (
                    "可定位原文证据"
                    f"（第{_chinese_ordinal(index + 1)}项）"
                )
        if cleaned in seen:
            cleaned = f"{cleaned}（第{_chinese_ordinal(index + 1)}项）"
        seen.add(cleaned)
        display.append(cleaned)
    return tuple(display)


def _matching_option_evidence(
    question: AssessmentQuestion,
    raw_option: str,
) -> EvidenceRef | None:
    needle = re.sub(r"\s+", " ", raw_option).strip().rstrip("…").lower()
    if len(needle) < 8:
        return None
    for evidence in question.evidence:
        quote = re.sub(r"\s+", " ", evidence.quote).strip().lower()
        if needle in quote or needle[:80] in quote:
            return evidence
    return None


def _chinese_ordinal(value: int) -> str:
    numerals = "零一二三四五六七八九十"
    if 0 <= value <= 10:
        return numerals[value]
    return str(value)


def _choice_marker(index: int) -> str:
    markers = "甲乙丙丁戊己庚辛"
    return markers[index] if 0 <= index < len(markers) else str(index + 1)


def _format_question_answer(
    question: AssessmentQuestion,
    answer: str,
) -> str:
    """把内部答案字母转换为用户能核对的中文选项内容。"""

    clean_answer = str(answer or "").strip()
    if not clean_answer:
        return "（未作答）"
    if question.question_type not in {"单项选择题", "多项选择题"}:
        return clean_answer

    letters = re.findall(r"[A-H]", clean_answer.upper())
    if not letters:
        return clean_answer
    options = _display_question_options(question)
    parts: list[str] = []
    for letter in dict.fromkeys(letters):
        index = ord(letter) - ord("A")
        if 0 <= index < len(options):
            parts.append(
                f"{_choice_marker(index)}、{options[index]}"
            )
    return "；".join(parts) if parts else clean_answer


def _render_question_rubric(question: AssessmentQuestion) -> None:
    st.markdown("**评分标准**")
    rubric_rows = [
        {
            "细则": item.label,
            "分值": item.points,
            "要求": item.description,
        }
        for item in question.rubric
    ]
    st.dataframe(
        pd.DataFrame(rubric_rows),
        width="stretch",
        hide_index=True,
        alt="题目评分细则",
    )


def _render_question_sources(question: AssessmentQuestion) -> None:
    st.markdown("**标准答案**")
    st.write(
        _format_question_answer(
            question,
            question.standard_answer,
        )
    )
    _render_question_rubric(question)
    st.markdown("**来源证据**")
    _render_evidence(question.evidence)


def _render_learning_cycle_result(
    workflow: ResearchMindWorkflow,
    result: LearningCycleResult,
) -> None:
    st.divider()
    st.subheader("本轮批改结果")
    st.success(result.summary)
    for question, score in result.scores:
        with st.container(border=True):
            st.markdown(f"**{question.question_type}　{score.score:g}/{score.max_score:g}**")
            st.write(question.prompt)
            st.caption(f"批改方式：{score.mode}")
            st.markdown("**你的答案**")
            st.write(
                _format_question_answer(
                    question,
                    str(getattr(score, "user_answer", "")),
                )
            )
            st.markdown("**标准答案**")
            st.write(
                _format_question_answer(
                    question,
                    question.standard_answer,
                )
            )
            if score.correct is True:
                st.success("本题判定：正确，已按标准答案获得满分。")
            elif score.correct is False:
                st.error("本题判定：错误，未获得客观题分数。")
            else:
                st.caption("主观题已按下列评分细则和内容要点计分。")
            st.write(score.feedback)
            criteria_rows = [
                {
                    "评分细则": item.label,
                    "得分": item.score,
                    "满分": item.max_score,
                    "依据": item.reason,
                }
                for item in score.criteria
            ]
            st.dataframe(
                pd.DataFrame(criteria_rows),
                width="stretch",
                hide_index=True,
                alt="本题评分细则得分依据",
            )
            with st.expander("查看本题原文证据与页码"):
                _render_evidence(question.evidence)

    mastery_frame = pd.DataFrame(
        {
            "能力维度": list(result.mastery),
            "掌握度": list(result.mastery.values()),
        }
    ).set_index("能力维度")
    st.subheader("更新后的掌握度")
    st.bar_chart(
        mastery_frame,
        height=230,
        alt="本轮答题后的五项能力掌握度",
    )
    st.subheader("个性化复习与下一轮安排")
    st.write(result.recommendation.summary)
    _render_recommendation(workflow, result.recommendation)
    st.info(
        f"已生成下一轮 {len(result.next_questions)} 道试题。"
        "可在“学习成长报告”查看变化，或在“AI 学习测评”继续学习。"
    )


def _render_recommendation(workflow: ResearchMindWorkflow, recommendation: Any) -> None:
    if recommendation.weak_dimensions:
        st.warning(
            "优先提升：" + "、".join(recommendation.weak_dimensions),
            icon=":material/trending_up:",
        )
    if recommendation.weak_tags:
        st.caption("薄弱知识点：" + "、".join(recommendation.weak_tags))
    with st.container(border=True):
        st.markdown("**推荐论文**")
        papers = []
        for paper_id in recommendation.recommended_paper_ids:
            paper = workflow.store.get_paper(paper_id)
            if paper is not None:
                papers.append(paper)
        if papers:
            for paper in papers:
                st.markdown(f"- {paper.title}（{paper.year}）")
                if paper.doi:
                    st.caption(f"DOI：{paper.doi}")
        else:
            st.caption("当前记录不足以匹配具体论文，先完成一轮测评可获得推荐。")
        st.markdown("**复习计划**")
        _render_bullet_list(recommendation.review_plan)


def _render_paper_metadata(paper: PaperRecord) -> None:
    with st.container(border=True):
        st.subheader(paper.title)
        st.write(paper.citation)
        st.caption(
            f"本地 PDF：{paper.pdf_path}　|　"
            f"文件状态：{'可用' if Path(paper.pdf_path).is_file() else '缺失'}"
        )
        st.caption(f"来源核验：{paper.verification_status}")
        if paper.source_url:
            st.link_button(
                "打开出版方或元数据页面",
                url=paper.source_url,
                icon=":material/open_in_new:",
            )


def _render_candidate_detail(candidate: CandidatePaper) -> None:
    with st.container(border=True):
        st.subheader(candidate.title)
        st.write("、".join(candidate.authors) or "作者信息未返回")
        st.caption(
            f"{candidate.year}　|　{candidate.journal or '期刊信息未返回'}　|　"
            f"来源：{candidate.source}"
        )
        if candidate.doi:
            st.caption(f"DOI：{candidate.doi}")
        if candidate.source_url:
            st.link_button(
                "打开来源页面",
                url=candidate.source_url,
                icon=":material/open_in_new:",
            )
        if candidate.abstract:
            with st.expander("查看来源返回的摘要"):
                st.write(candidate.abstract)


def _download_and_ingest_candidate(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    candidate: CandidatePaper,
) -> None:
    try:
        with st.spinner("正在下载并验证开放获取 PDF..."):
            pdf_bytes = LiteratureCollector(
                workflow.config
            ).download_open_pdf(candidate)
            result = ingest_pdf_files(
                [
                    IncomingPDF(
                        file_name=_safe_pdf_name(candidate),
                        data=pdf_bytes,
                    )
                ],
                workflow.config.knowledge_base_dir,
            )
    except (LiteratureCollectionError, ValueError, OSError) as exc:
        st.error(f"开放 PDF 未导入：{exc}")
        return

    if result.imported_count:
        with st.spinner("正在同步新论文到 ResearchMind..."):
            sync_result = workflow.sync_bundled_library(
                domain=domain,
                force=True,
            )
        _render_ingest_result(result)
        _show_sync_result(sync_result)
        st.cache_resource.clear()
        st.cache_data.clear()
    else:
        _render_ingest_result(result)


def _render_ingest_result(result: IngestResult) -> None:
    if result.imported:
        st.success(
            "已导入：" + "、".join(item.title for item in result.imported)
        )
    if result.duplicates:
        st.info("重复文件未再次导入：" + "；".join(result.duplicates))
    if result.skipped:
        st.warning("未导入：" + "；".join(result.skipped))
    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)


def _show_sync_result(result: LibrarySyncResult) -> None:
    if result.used_cached_sync:
        st.info(result.message)
    else:
        st.success(result.message)
    st.caption(
        f"SQLite 中当前共有 {result.total_papers} 篇论文、"
        f"{result.total_evidence} 条带页码证据。"
    )


def _render_graphviz_or_table(synthesis: ResearchSynthesis) -> None:
    dot = _synthesis_to_dot(synthesis)
    try:
        st.graphviz_chart(
            dot,
            width="stretch",
            height=780,
            alt="ResearchMind 领域概念、论文与证据关系图",
        )
    except Exception:
        st.info(
            "图形渲染组件暂不可用，已回退为节点表和关系表。"
            "可安装 graphviz 后重新打开页面。"
        )
        node_rows = [
            {
                "节点": node.label,
                "类型": node.node_type,
                "说明": node.description,
                "证据数": node.evidence_count,
            }
            for node in synthesis.nodes
        ]
        st.dataframe(
            pd.DataFrame(node_rows),
            width="stretch",
            hide_index=True,
            alt="知识图谱节点表",
        )
        labels = {node.node_id: node.label for node in synthesis.nodes}
        edge_rows = [
            {
                "起点": labels.get(edge.source_id, edge.source_id),
                "关系": edge.relation,
                "终点": labels.get(edge.target_id, edge.target_id),
                "证据数": edge.evidence_count,
            }
            for edge in synthesis.edges
        ]
        st.dataframe(
            pd.DataFrame(edge_rows),
            width="stretch",
            hide_index=True,
            alt="知识图谱关系表",
        )


def _synthesis_to_dot(synthesis: ResearchSynthesis) -> str:
    colors = {
        "领域": "#0f766e",
        "概念": "#25636f",
        "论文": "#7c5b24",
        "研究设计": "#4f5f86",
        "争议": "#9b4b3f",
    }
    lines = [
        "digraph ResearchMind {",
        "graph [rankdir=LR, bgcolor=\"transparent\", pad=0.35, nodesep=0.55, ranksep=0.9];",
        "node [shape=box, style=\"rounded,filled\", fontname=\"Microsoft YaHei\", fontsize=14, margin=\"0.18,0.1\", color=\"#d9e2e2\"];",
        "edge [color=\"#82989c\", arrowsize=0.8, fontname=\"Microsoft YaHei\", fontsize=11];",
    ]
    for node in synthesis.nodes:
        color = colors.get(node.node_type, "#45656b")
        label = f"{node.label}\n{node.node_type} · {node.evidence_count} 条证据"
        lines.append(
            f"{_dot_quote(node.node_id)} "
            f"[label={_dot_quote(label)}, fillcolor=\"#f7fbfa\", fontcolor={_dot_quote(color)}];"
        )
    node_ids = {node.node_id for node in synthesis.nodes}
    for edge in synthesis.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            continue
        lines.append(
            f"{_dot_quote(edge.source_id)} -> {_dot_quote(edge.target_id)} "
            f"[label={_dot_quote(edge.relation)}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _dot_quote(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return f'"{escaped}"'


def _active_questions(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> tuple[AssessmentQuestion, ...]:
    ids = st.session_state.get("researchmind_active_questions", ())
    selected = []
    for question_id in ids:
        question = workflow.store.get_question(str(question_id))
        if question is not None and question.domain_id == domain.domain_id:
            selected.append(question)
    if selected:
        return tuple(selected)
    return workflow.store.list_questions(domain.domain_id, limit=6)


def _paper_titles(
    workflow: ResearchMindWorkflow,
    paper_ids: Iterable[str],
) -> tuple[str, ...]:
    titles: list[str] = []
    for paper_id in paper_ids:
        paper = workflow.store.get_paper(paper_id)
        titles.append(paper.title if paper else f"未找到论文记录：{paper_id}")
    return tuple(titles)


def _question_types_for_review(recommendation: Any) -> tuple[str, ...]:
    weak = set(recommendation.weak_dimensions)
    types: list[str] = []
    if "概念记忆" in weak:
        types.extend(("单项选择题", "判断题"))
    if "文献理解" in weak:
        types.extend(("多项选择题", "简答题"))
    if "对比分析" in weak:
        types.append("文献对比分析题")
    if "批判性思维" in weak:
        types.extend(("论述题", "文献对比分析题"))
    if "学术应用" in weak:
        types.extend(("论述题", "简答题"))
    if not types:
        types.extend(("多项选择题", "文献对比分析题"))
    return tuple(dict.fromkeys(types))


def _default_topic(domain: ResearchDomain) -> str:
    if domain.keywords:
        return "与".join(domain.keywords[:2]) + "的关系"
    return DEFAULT_DEBATE_TOPIC


def _split_keywords(value: str) -> tuple[str, ...]:
    normalized = value.replace("，", ",").replace("、", ",")
    return tuple(
        dict.fromkeys(
            item.strip() for item in normalized.split(",") if item.strip()
        )
    )


def _year_range(papers: Iterable[PaperRecord]) -> str:
    years = [int(paper.year) for paper in papers if paper.year.isdigit()]
    if not years:
        return "未知"
    first, last = min(years), max(years)
    return str(first) if first == last else f"{first}-{last}"


def _short_time(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    return parsed.strftime("%Y-%m-%d %H:%M")


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def _safe_pdf_name(candidate: CandidatePaper) -> str:
    stem = "".join(
        character if character.isalnum() else "-"
        for character in candidate.title[:80]
    ).strip("-")
    stem = stem or candidate.candidate_id
    return f"{stem[:80]}.pdf"


def _render_bullet_list(values: Iterable[str]) -> None:
    items = tuple(value for value in values if str(value).strip())
    if not items:
        st.caption("证据不足，暂无可展示内容。")
        return
    for value in items:
        st.markdown(f"- {value}")


def _page_heading(label: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.write(description)


__all__ = [
    "RESEARCHMIND_PAGES",
    "get_researchmind_workflow",
    "render_researchmind_page",
]
