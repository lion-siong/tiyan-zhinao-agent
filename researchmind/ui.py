"""ResearchMind 的 Streamlit 页面。

本模块只负责界面交互，论文检索、阅读、分析、组卷和评分均由
``researchmind`` 包中的工作流与 Agent 完成。所有耗时的完整分析都放在按钮
之后执行，页面打开时只读取已有结果和轻量计数。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import streamlit as st

from researchmind.adolescent_library import (
    assess_candidates,
    create_adolescent_workflow,
    create_library_backup,
    create_legacy_workflow,
    list_library_backups,
    restore_library_backup,
)
from researchmind.adolescent_topic import (
    CORE_DOMAIN_NAME,
    INSUFFICIENT_EVIDENCE_TEXT,
    assess_topic_relevance,
)
from researchmind.analysis_status_ui import (
    clear_analysis_session_state,
    render_analysis_update_status,
    schedule_for_domain,
)
from researchmind.library_state import schedule_analysis_refresh
from researchmind.agents import (
    AssessmentGenerationError,
    FIELD_LABELS,
    chinese_evidence_option_text,
    chinese_only_excerpt,
)
from researchmind.collectors import (
    CollectionResult,
    LITERATURE_SOURCES,
    LiteratureCollectionError,
    LiteratureCollector,
)
from researchmind.literature_import import (
    ImportResult,
    import_candidates,
    library_status_for_candidates,
    permanently_delete_papers,
    restore_papers,
    soft_delete_papers,
)
from researchmind.models import (
    ASSESSMENT_MODULES,
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
from researchmind.knowledge_map_ui import (
    render_knowledge_map_page,
    render_knowledge_summary_page,
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
from researchmind.zotero_local_service import (
    ZOTERO_DOMAIN_NAME,
    ZoteroLocalError,
    probe_zotero_local,
    sync_zotero_local_library,
)
from tools.knowledge_base import KnowledgeBaseError
from tools.library_ingest import (
    IncomingPDF,
    IngestResult,
    rebuild_library,
    ingest_pdf_files,
)
from tools.pdf_metadata import extract_pdf_metadata
from tools.pdf_reader import extract_pdf_text

LOGGER = logging.getLogger(__name__)


RESEARCHMIND_PAGES = (
    "学术领域管理",
    "文献自动采集",
    "我的论文库",
    "论文智能阅读",
    "学术观点对比",
    "学术分歧分析",
    "知识图谱",
    "知识图谱总结",
    "AI 学习测评",
    "错题本",
    "学习成长报告",
)

RESEARCHMIND_PAGE_ALIASES = {
    "科研知识地图": "知识图谱",
    "科研知识地图总结": "知识图谱总结",
}

LIBRARY_MODE_ADOLESCENT = "adolescent"
LIBRARY_MODE_LEGACY = "legacy"
LIBRARY_MODE_LABELS = {
    LIBRARY_MODE_ADOLESCENT: "青少年专题库",
    LIBRARY_MODE_LEGACY: "旧库只读",
}
LEGACY_READ_ONLY_PAGES = {
    "学术领域管理",
    "我的论文库",
    "知识图谱",
    "知识图谱总结",
}

_FIELD_ORDER = (
    "purpose",
    "population",
    "methods",
    "interventions",
    "comparators",
    "measures",
    "results",
    "statistics",
    "conclusions",
    "limitations",
)


@st.cache_resource
def get_researchmind_workflow(
    library_mode: str = LIBRARY_MODE_ADOLESCENT,
) -> ResearchMindWorkflow:
    """按页面模式创建共享工作流，专题库与旧库绝不混用路径。"""

    if library_mode == LIBRARY_MODE_LEGACY:
        return create_legacy_workflow()
    if library_mode != LIBRARY_MODE_ADOLESCENT:
        raise ValueError(f"未知论文库模式：{library_mode}")
    return create_adolescent_workflow()


def render_researchmind_page(page: str) -> None:
    """根据导航名称渲染 ResearchMind 页面。"""

    page = RESEARCHMIND_PAGE_ALIASES.get(page, page)
    library_mode = _render_library_mode_selector()
    try:
        workflow = get_researchmind_workflow(library_mode)
    except (ResearchMindStorageError, ResearchMindWorkflowError, OSError, ValueError) as exc:
        st.error(f"当前论文库暂时无法打开：{exc}")
        if library_mode == LIBRARY_MODE_LEGACY:
            st.caption(
                "旧库只要不存在或不完整就会被拒绝访问，系统不会自动创建、"
                "覆盖或迁移旧数据。请先检查原数据库和备份。"
            )
        return

    if workflow.read_only and page not in LEGACY_READ_ONLY_PAGES:
        st.info(
            "旧库现在是只读回看模式。为避免在旧数据上产生新分析、题库或删除"
            "操作，此页暂不开放；请切换到“青少年专题库”继续完整工作流。",
            icon=":material/lock:",
        )
        return

    domain = _render_domain_selector(workflow)
    renderers = {
        "学术领域管理": _render_domain_management,
        "文献自动采集": _render_literature_collection,
        "我的论文库": _render_library,
        "论文智能阅读": _render_paper_reading,
        "学术观点对比": _render_viewpoint_comparison,
        "学术分歧分析": _render_debate_analysis,
        "知识图谱": render_knowledge_map_page,
        "知识图谱总结": render_knowledge_summary_page,
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
        LOGGER.exception("ResearchMind page failed: %s", page)
        st.error(f"页面暂时无法加载：{exc}")
        st.caption("数据不会被自动删除。请检查文献库、数据库或网络配置后重试。")


def _render_library_mode_selector() -> str:
    """显示紧凑的论文库模式切换，旧库只能作为只读备份入口。"""

    current = st.session_state.get(
        "researchmind_library_mode",
        LIBRARY_MODE_ADOLESCENT,
    )
    if current not in LIBRARY_MODE_LABELS:
        current = LIBRARY_MODE_ADOLESCENT
    selected = st.segmented_control(
        "论文库模式",
        options=tuple(LIBRARY_MODE_LABELS),
        format_func=lambda value: LIBRARY_MODE_LABELS[value],
        default=current,
        key="researchmind_library_mode_selector",
        help=(
            "青少年专题库用于当前研究与写入；旧库只读用于回看和恢复前核对，"
            "不会复制或覆盖原有文件。"
        ),
    )
    mode = str(selected or current)
    st.session_state["researchmind_library_mode"] = mode
    if mode == LIBRARY_MODE_LEGACY:
        st.caption(
            "旧库只读：不自动同步、不新建领域、不导入或删除论文。"
        )
    else:
        st.caption(
            f"专题方向：{CORE_DOMAIN_NAME}。不符合研究对象的论文不会进入核心图谱。"
        )
    return mode


def _render_domain_selector(
    workflow: ResearchMindWorkflow,
) -> ResearchDomain:
    """显示当前领域选择器，并确保选中领域有可用论文库。"""

    if workflow.read_only:
        domains = workflow.store.list_domains()
        if not domains:
            raise ResearchMindWorkflowError(
                "旧论文库中没有可读取的研究领域，只读模式不会创建新领域。"
            )
        default_domain = domains[0]
    else:
        default_domain = workflow.ensure_default_domain()
        domains = workflow.store.list_domains()
    domain_by_id = {domain.domain_id: domain for domain in domains}
    ids = tuple(domain_by_id)
    session_key = (
        "researchmind_active_domain_legacy"
        if workflow.read_only
        else "researchmind_active_domain"
    )
    active_id = st.session_state.get(session_key)
    preferred_id = (
        None
        if workflow.read_only
        else workflow.store.get_metadata(
            "researchmind_preferred_domain"
        )
    )
    preferred_seen_key = "researchmind_preferred_domain_seen"
    preferred_seen = st.session_state.get(preferred_seen_key)
    if (
        active_id not in domain_by_id
        or (
            preferred_id in domain_by_id
            and preferred_id != preferred_seen
        )
    ):
        active_id = (
            preferred_id
            if preferred_id in domain_by_id
            else default_domain.domain_id
        )
        st.session_state[session_key] = active_id
        st.session_state[preferred_seen_key] = active_id

    # selectbox 实例化前同步选择器初值；widget key 不能在实例化之后再写。
    st.session_state["researchmind_domain_selector"] = active_id

    selected_id = st.selectbox(
        "当前研究领域",
        options=ids,
        format_func=lambda value: domain_by_id[value].name,
        key="researchmind_domain_selector",
        help="所有阅读、分析、题库和学习记录都会归入当前领域。",
    )
    st.session_state[session_key] = selected_id
    if not workflow.read_only:
        workflow.store.set_metadata(
            "researchmind_preferred_domain",
            selected_id,
        )
        st.session_state[preferred_seen_key] = selected_id
    domain = domain_by_id[selected_id]

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
    if workflow.read_only:
        st.info(
            "当前为旧库只读模式。页面只读取已有元数据、分析结果和证据，"
            "不会同步、新建、覆盖或删除任何旧数据。",
            icon=":material/lock:",
        )
    elif workflow.domain_name == CORE_DOMAIN_NAME:
        st.info(
            "当前专题库只服务“青少年体质健康与体育锻炼干预研究”。"
            "新论文导入前会做本地相关度预检，不符合研究对象的文献不会进入核心图谱。",
            icon=":material/verified:",
        )
    else:
        st.info(
            "论文与证据均来自独立论文库；不会自动从旧库复制内容。",
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

    if workflow.read_only:
        st.caption("只读模式已禁用新建领域入口。")
    elif workflow.domain_name == CORE_DOMAIN_NAME:
        st.caption("专题版固定为唯一核心研究方向，已禁用新建领域入口。")
    else:
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

    if not workflow.read_only:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            if st.button(
                "同步当前专题库",
                icon=":material/sync:",
                help="重新读取专题库 manifest.json 和证据索引，不解析其他论文库。",
            ):
                with st.spinner("正在同步专题论文元数据与证据..."):
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
    operation_notice = st.session_state.pop(
        "researchmind_collection_notice",
        "",
    )
    if operation_notice:
        st.success(operation_notice)

    with st.container(border=True):
        st.subheader("Zotero 本地文库")
        try:
            zotero_local = probe_zotero_local()
        except ZoteroLocalError as exc:
            st.error(str(exc))
        else:
            st.caption(
                f"已连接 {zotero_local['base_url']}，"
                f"当前活动题录 {zotero_local['active_items']} 篇。"
                "同步会写入独立的“Zotero 青少年体育文献库”领域，"
                "不覆盖现有专题库。"
            )
            if st.button(
                "同步 Zotero 本地活动论文",
                icon=":material/sync_alt:",
                key="researchmind_sync_zotero_local",
            ):
                try:
                    with st.spinner(
                        "正在读取 Zotero 题录与 PDF，并建立页级证据..."
                    ):
                        sync_result = sync_zotero_local_library(workflow)
                        schedule_for_domain(
                            workflow,
                            sync_result.domain,
                            force=True,
                        )
                except (ZoteroLocalError, OSError, ValueError) as exc:
                    st.error(f"Zotero 本地库同步失败：{exc}")
                else:
                    st.session_state["researchmind_active_domain"] = (
                        sync_result.domain.domain_id
                    )
                    st.session_state["researchmind_collection_notice"] = (
                        f"{sync_result.message} 已切换到"
                        f"“{ZOTERO_DOMAIN_NAME}”，并启动后台分析。"
                    )
                    st.rerun()

    current_year = datetime.now().year
    with st.form("researchmind_collection_form"):
        query = st.text_input(
            "研究主题",
            value="青少年体质健康 体育锻炼干预",
            placeholder="例如：青少年体育锻炼、运动干预与睡眠质量",
        )
        keywords = st.text_input(
            "附加关键词（可选）",
            placeholder="例如：随机对照试验、运动强度、心肺适能",
        )
        filter_columns = st.columns([1, 1, 1, 1])
        year_from = filter_columns[0].number_input(
            "起始年份",
            min_value=1900,
            max_value=current_year,
            value=max(2021, current_year - 5),
            step=1,
        )
        year_to = filter_columns[1].number_input(
            "结束年份",
            min_value=1900,
            max_value=current_year,
            value=current_year,
            step=1,
        )
        source = filter_columns[2].selectbox(
            "文献来源",
            options=LITERATURE_SOURCES,
            index=0,
        )
        limit = filter_columns[3].number_input(
            "返回数量",
            min_value=1,
            max_value=25,
            value=10,
            step=1,
        )
        try_open_pdf = st.checkbox(
            "尝试下载合法开放获取 PDF",
            value=False,
            help="仅下载 OpenAlex 明确标记开放获取且提供 PDF 地址的文件。",
        )
        submitted = st.form_submit_button(
            "开始检索",
            type="primary",
            icon=":material/search:",
        )

    if submitted:
        clean_query = " ".join(
            part.strip()
            for part in (query, keywords)
            if part.strip()
        )
        if not clean_query:
            st.error("研究主题不能为空。")
        elif int(year_from) > int(year_to):
            st.error("起始年份不能晚于结束年份。")
        else:
            try:
                with st.spinner("正在访问 OpenAlex / Crossref..."):
                    result = LiteratureCollector(workflow.config).search(
                        clean_query,
                        limit=int(limit),
                        date_from=f"{int(year_from)}-01-01",
                        date_to=f"{int(year_to)}-12-31",
                        source=source,
                    )
            except LiteratureCollectionError as exc:
                st.error(f"文献检索失败：{exc}")
            else:
                st.session_state["researchmind_collection_result"] = result
                st.session_state["researchmind_collection_selection"] = []

    result = st.session_state.get("researchmind_collection_result")
    if not isinstance(result, CollectionResult):
        st.caption(
            "检索只获取公开元数据。选择论文后可导入当前专题论文库；"
            "全文仅来自合法开放 PDF 或你上传的本地文件。"
        )
        return

    st.divider()
    statuses = library_status_for_candidates(
        workflow.store,
        domain.domain_id,
        result.candidates,
    )
    imported_count = sum(
        status.status == "已导入" for status in statuses.values()
    )
    deleted_count = sum(
        status.status == "回收站" for status in statuses.values()
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric("来源", result.source)
    metric_columns[1].metric("真实元数据", f"{len(result.candidates)} 篇")
    metric_columns[2].metric("开放 PDF", f"{sum(c.has_open_pdf for c in result.candidates)} 篇")
    metric_columns[3].metric("已导入", f"{imported_count} 篇")
    metric_columns[4].metric("回收站", f"{deleted_count} 篇")
    for warning in result.warnings:
        st.warning(warning)
    if not result.candidates:
        st.info("来源没有返回可核验论文，未写入任何记录，也没有生成替代引用。")
        return

    rows = []
    for candidate in result.candidates:
        relevance = assess_topic_relevance(
            title=candidate.title,
            abstract=candidate.abstract,
            keywords=candidate.keywords,
        )
        rows.append(
            {
                "题名": candidate.title,
                "作者": candidate.authors_text,
                "年份": candidate.year,
                "期刊": candidate.journal,
                "DOI": candidate.doi,
                "来源": candidate.source,
                "开放 PDF": "有" if candidate.has_open_pdf else "仅元数据",
                "专题相关度": relevance.display_status,
                "论文库状态": statuses[candidate.candidate_id].status,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="学术来源返回的真实论文元数据及论文库状态",
    )

    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in result.candidates
    }
    candidate_ids = tuple(candidate_by_id)
    selection_key = "researchmind_collection_selection"
    select_columns = st.columns([1, 1, 3])
    if select_columns[0].button(
        "全选",
        icon=":material/select_all:",
        key="researchmind_collection_select_all",
    ):
        st.session_state[selection_key] = list(candidate_ids)
        st.rerun()
    if select_columns[1].button(
        "取消选择",
        icon=":material/deselect:",
        key="researchmind_collection_clear",
    ):
        st.session_state[selection_key] = []
        st.rerun()
    selected_ids = tuple(
        st.multiselect(
            "选择要导入的论文",
            options=candidate_ids,
            format_func=lambda value: (
                f"{candidate_by_id[value].title}"
                f"（{statuses[value].status}）"
            ),
            key=selection_key,
        )
    )
    select_columns[2].caption(
        f"已选择 {len(selected_ids)} 篇；重复论文不会生成第二条记录。"
    )

    st.subheader("检索结果")
    for candidate in result.candidates:
        status = statuses[candidate.candidate_id]
        relevance = assess_topic_relevance(
            title=candidate.title,
            abstract=candidate.abstract,
            keywords=candidate.keywords,
        )
        with st.expander(
            f"{candidate.title}｜{candidate.year}｜{status.status}",
            icon=":material/article:",
        ):
            _render_candidate_detail(candidate)
            detail_columns = st.columns(3)
            detail_columns[0].caption(f"来源：{candidate.source}")
            detail_columns[1].caption(
                f"专题相关度：{relevance.display_status}"
            )
            detail_columns[2].caption(
                "开放 PDF："
                + ("可尝试下载" if candidate.has_open_pdf else "仅元数据")
            )
            if st.button(
                (
                    "从回收站恢复并导入"
                    if status.status == "回收站"
                    else "导入论文库"
                ),
                type="primary" if status.status == "未导入" else "secondary",
                icon=":material/library_add:",
                disabled=status.status == "已导入",
                key=f"researchmind_import_candidate_{candidate.candidate_id}",
            ):
                _import_collection_candidates(
                    workflow,
                    domain,
                    (candidate,),
                    download_open_pdf=bool(try_open_pdf),
                )

    st.divider()
    action_columns = st.columns([1, 2])
    with action_columns[0]:
        if st.button(
            f"批量导入已选 {len(selected_ids)} 篇",
            type="primary",
            icon=":material/library_add:",
            width="stretch",
            disabled=not selected_ids,
            key="researchmind_collection_batch_import",
        ):
            _import_collection_candidates(
                workflow,
                domain,
                tuple(candidate_by_id[item] for item in selected_ids),
                download_open_pdf=bool(try_open_pdf),
            )
    action_columns[1].caption(
        "“已导入”状态直接查询当前 SQLite；点击导入后系统会真实写入并校验记录数。"
    )


def _import_collection_candidates(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    candidates: tuple[CandidatePaper, ...],
    *,
    download_open_pdf: bool,
) -> None:
    """执行真实导入，并把数据库返回的统计结果交给下一轮页面展示。"""

    try:
        with st.spinner("正在写入论文库并校验数据库记录..."):
            result = import_candidates(
                workflow,
                domain,
                candidates,
                download_open_pdf=download_open_pdf,
            )
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"论文导入失败：{exc}")
        return

    notice_parts = [
        f"新导入 {result.imported} 篇",
        f"补齐元数据 {result.updated} 篇",
        f"从回收站恢复 {result.restored} 篇",
        f"重复跳过 {result.skipped_duplicates} 篇",
        f"专题不相关 {result.skipped_ineligible} 篇",
        f"失败 {result.failed} 篇",
    ]
    if download_open_pdf:
        notice_parts.append(f"开放 PDF 入库 {result.downloaded_pdfs} 篇")
    detail = "；".join(notice_parts)
    if result.messages:
        detail += "。处理明细：" + "；".join(result.messages[:5])
        if len(result.messages) > 5:
            detail += f"；另有 {len(result.messages) - 5} 条明细"
    if result.changed:
        try:
            schedule_for_domain(workflow, domain)
        except Exception as exc:
            st.error(
                f"论文已导入，但后台分析启动失败：{type(exc).__name__}: {exc}"
            )
        else:
            detail += "；已启动后台知识图谱与学术分歧更新。"
            clear_analysis_session_state(domain.domain_id)
    st.session_state["researchmind_collection_notice"] = detail
    st.rerun()


def _render_library(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    _page_heading(
        "MY LIBRARY",
        "我的论文库",
        "检索、解析、筛选和维护当前专题的真实论文与证据。",
    )
    operation_notice = st.session_state.pop(
        "researchmind_library_operation_notice",
        "",
    )
    if operation_notice:
        st.success(operation_notice)
    delete_notice = st.session_state.pop(
        "researchmind_paper_delete_notice",
        "",
    )
    if delete_notice:
        st.success(delete_notice)
    papers = workflow.store.list_papers(domain.domain_id)
    deleted_papers = workflow.store.list_deleted_papers(domain.domain_id)
    evidence_count = workflow.store.evidence_count(domain.domain_id)
    zotero = ZoteroService(workflow.config)
    zotero_status = zotero.status()
    metrics = st.columns(5)
    metrics[0].metric("论文", f"{len(papers)} 篇")
    metrics[1].metric(
        "已解析",
        f"{sum(paper.parse_status == '已解析' for paper in papers)} 篇",
    )
    metrics[2].metric(
        "待分析",
        f"{sum(paper.analysis_status in {'未分析', '待分析'} for paper in papers)} 篇",
    )
    metrics[3].metric(
        "已分析",
        f"{sum(paper.analysis_status == '已分析' for paper in papers)} 篇",
    )
    metrics[4].metric("回收站", f"{len(deleted_papers)} 篇")
    st.caption(
        f"证据片段 {evidence_count} 条；"
        f"本地 PDF {sum(Path(paper.pdf_path).is_file() for paper in papers)} 篇；"
        f"年份覆盖 {_year_range(papers)}。"
    )

    if not workflow.read_only:
        _render_library_import(workflow, domain)

    if not papers and not deleted_papers:
        st.warning(
            "当前领域还没有论文。请通过“文献自动采集”导入开放论文，"
            "或上传你有权使用的 PDF 建立专题论文库。"
            "检索题录不等同于已获取全文。"
        )
        if not workflow.read_only:
            _render_library_backup_management(workflow)
        return

    filtered: list[PaperRecord] = []
    if papers:
        filter_columns = st.columns([2.2, 1.2, 0.9, 1, 1])
        search = filter_columns[0].text_input(
            "筛选论文",
            placeholder="题名、作者、期刊或 DOI",
        )
        categories = tuple(
            sorted({paper.category for paper in papers if paper.category})
        )
        selected_category = filter_columns[1].selectbox(
            "分类",
            options=("全部", *categories),
        )
        years = tuple(sorted({paper.year for paper in papers}, reverse=True))
        selected_year = filter_columns[2].selectbox(
            "年份",
            options=("全部", *years),
        )
        selected_pdf = filter_columns[3].selectbox(
            "PDF 状态",
            options=("全部", "本地 PDF 可用", "PDF 缺失"),
        )
        selected_analysis = filter_columns[4].selectbox(
            "分析状态",
            options=("全部", "未分析", "待分析", "已分析", "待补充全文"),
        )
        second_filters = st.columns([1, 1, 2])
        selected_parse = second_filters[0].selectbox(
            "解析状态",
            options=("全部", "仅元数据", "已解析"),
        )
        sort_order = second_filters[1].selectbox(
            "排序",
            options=(
                "导入时间（最新）",
                "导入时间（最早）",
                "发表年份（最新）",
                "题名",
            ),
        )
        second_filters[2].caption(
            "删除采用两阶段流程：先移入回收站，永久删除前自动备份。"
        )

        lowered = search.strip().lower()
        for paper in papers:
            haystack = " ".join(
                (
                    paper.title,
                    paper.authors_text,
                    paper.journal,
                    paper.doi,
                    paper.source,
                    paper.category,
                    " ".join(paper.keywords),
                )
            ).lower()
            if lowered and lowered not in haystack:
                continue
            if (
                selected_category != "全部"
                and paper.category != selected_category
            ):
                continue
            if selected_year != "全部" and paper.year != selected_year:
                continue
            has_pdf = Path(paper.pdf_path).is_file()
            if selected_pdf == "本地 PDF 可用" and not has_pdf:
                continue
            if selected_pdf == "PDF 缺失" and has_pdf:
                continue
            if (
                selected_parse != "全部"
                and paper.parse_status != selected_parse
            ):
                continue
            if (
                selected_analysis != "全部"
                and paper.analysis_status != selected_analysis
            ):
                continue
            filtered.append(paper)

        if sort_order == "导入时间（最新）":
            filtered.sort(key=lambda item: item.imported_at, reverse=True)
        elif sort_order == "导入时间（最早）":
            filtered.sort(key=lambda item: item.imported_at)
        elif sort_order == "发表年份（最新）":
            filtered.sort(key=lambda item: item.year, reverse=True)
        else:
            filtered.sort(key=lambda item: item.title.casefold())

        rows = [
            {
                "题名": paper.title,
                "作者": paper.authors_text,
                "年份": paper.year,
                "期刊": paper.journal,
                "DOI": paper.doi,
                "来源": paper.source,
                "分类": paper.category,
                "导入时间": _display_timestamp(paper.imported_at),
                "PDF": "可用" if Path(paper.pdf_path).is_file() else "缺失",
                "解析状态": paper.parse_status,
                "分析状态": paper.analysis_status,
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
            evidence_available = bool(
                workflow.store.list_evidence(selected_paper.paper_id, limit=1)
            )
            if not evidence_available:
                st.warning(
                    "当前论文仅有公开元数据，需导入合法获取的 PDF 后，"
                    "才能进行全文阅读、证据提取和深度科研分析。",
                    icon=":material/picture_as_pdf:",
                )
            action_columns = st.columns([1, 1, 1, 1])
            if pdf_available:
                try:
                    pdf_bytes = Path(selected_paper.pdf_path).read_bytes()
                except OSError as exc:
                    action_columns[0].button(
                        "下载本地 PDF",
                        disabled=True,
                        help=f"PDF 读取失败：{exc}",
                    )
                else:
                    action_columns[0].download_button(
                        "下载本地 PDF",
                        data=pdf_bytes,
                        file_name=Path(selected_paper.pdf_path).name,
                        mime="application/pdf",
                        icon=":material/download:",
                    )
            if action_columns[1].button(
                "执行结构化阅读",
                icon=":material/document_scanner:",
                disabled=not evidence_available,
                help=(
                    "从已建正文证据中提取研究字段。"
                    if evidence_available
                    else "需先导入并解析合法获取的 PDF。"
                ),
            ):
                with st.spinner("正在从真实证据中定位结构字段..."):
                    workflow.reader.read_paper(selected_paper.paper_id)
                schedule_for_domain(workflow, domain, force=True)
                st.rerun()
            if action_columns[2].button(
                "同步到 Zotero",
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
                        f"父条目 key：{parent_key}；"
                        f"PDF 附件 key：{attachment_key}"
                    )
            if action_columns[3].button(
                "删除当前论文",
                icon=":material/delete:",
                disabled=workflow.read_only,
            ):
                st.session_state["researchmind_delete_confirm_ids"] = (
                    selected_paper.paper_id,
                )
                st.session_state["researchmind_delete_confirm_mode"] = "soft"
                st.rerun()
            if not zotero_status.configured:
                st.caption(
                    "Zotero 尚未配置，本地论文库不受影响。"
                    "请通过 .env 配置 ZOTERO_API_KEY 和 ZOTERO_LIBRARY_ID。"
                )

            analysis = workflow.store.get_analysis(selected_paper.paper_id)
            if analysis is not None:
                with st.expander(
                    "查看当前结构化阅读结果",
                    on_change="ignore",
                ):
                    _render_analysis(analysis)
            if pdf_available:
                pdf_expander = st.expander(
                    "在页面内查看 PDF",
                    icon=":material/picture_as_pdf:",
                    on_change="rerun",
                )
                if pdf_expander.open:
                    with pdf_expander:
                        st.pdf(
                            selected_paper.pdf_path,
                            height=620,
                            alt=f"{selected_paper.title} 本地 PDF",
                        )
        else:
            st.info("当前筛选条件下没有论文。")

    if not workflow.read_only:
        _render_library_removal(
            workflow,
            domain,
            tuple(filtered or papers),
        )

    if workflow.read_only:
        st.caption("旧库只读模式已禁用上传、删除、重新索引和恢复操作。")
        return

    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "同步最新专题论文库",
            icon=":material/sync:",
            help="同步专题库 manifest.json 和证据索引中的新增论文。",
        ):
            result = workflow.sync_bundled_library(domain=domain)
            _show_sync_result(result)
            if not result.used_cached_sync:
                st.rerun()
        if st.button(
            "重新索引专题库",
            icon=":material/refresh:",
            help="重建证据索引、研究框架和向量库，不替换原始 PDF。",
        ):
            with st.spinner("正在重建专题证据索引和向量库..."):
                rebuild = rebuild_library(workflow.knowledge_base_dir)
                sync_result = workflow.sync_bundled_library(
                    domain=domain,
                    force=True,
                )
            st.success(
                f"已重建 {len(rebuild.bundle.documents)} 篇论文、"
                f"{len(rebuild.bundle.chunks)} 条证据。"
            )
            if rebuild.vector_warning:
                st.warning(rebuild.vector_warning)
            _show_sync_result(sync_result)
            schedule_for_domain(workflow, domain, force=True)
            st.rerun()

    _render_library_backup_management(workflow)


def _display_timestamp(value: str) -> str:
    """把 ISO 时间压缩成适合表格阅读的展示文本。"""

    clean = str(value).strip()
    if not clean:
        return "未知"
    return clean[:16].replace("T", " ")


def _paper_workflow_status(
    paper: PaperRecord,
    analysis: PaperAnalysis | None,
) -> str:
    """从数据库状态和真实资产推导论文流程状态。"""

    if paper.analysis_status == "已分析":
        return "已完成分析"
    if paper.parse_status == "已解析":
        return "已解析"
    if Path(paper.pdf_path).is_file():
        return "已获取全文"
    return "检索到"


def _render_library_import(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    """提供多 PDF 上传、相关度预检、去重和批量入库入口。"""

    with st.expander(
        "批量导入论文 PDF",
        icon=":material/upload_file:",
        on_change="ignore",
    ):
        st.caption(
            "仅导入你有权使用的本地 PDF。系统先读取题名、DOI、前几页原文并做"
            "青少年专题相关度判断；范围外、重复或无可提取文本的文件不会进入核心库。"
        )
        uploaded_files = st.file_uploader(
            "选择 PDF 文件",
            type=("pdf",),
            accept_multiple_files=True,
            key=f"researchmind_pdf_batch_{domain.domain_id}",
            help="支持一次选择多篇；每篇最多 20 MB，文件不会上传到第三方服务。",
        )
        if not uploaded_files:
            st.caption("选择文件后会先显示预检结果，不会立即写入论文库。")
            return

        try:
            preflight = _preflight_pdf_uploads(
                workflow,
                uploaded_files,
                domain,
            )
        except (KnowledgeBaseError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"PDF 预检未完成：{exc}")
            return

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "文件": item["file_name"],
                        "题名": item["title"],
                        "年份": item["year"],
                        "DOI": item["doi"],
                        "专题相关度": item["relevance"],
                        "预检结论": item["status"],
                        "PDF SHA-256": item["sha256"][:16] + "…",
                    }
                    for item in preflight
                ]
            ),
            width="stretch",
            hide_index=True,
            alt="批量 PDF 专题相关度和去重预检结果",
        )
        eligible = [item for item in preflight if item["eligible"]]
        st.caption(
            f"可导入 {len(eligible)} 篇；其余文件保留在预检表中，不会写入专题库。"
        )
        if st.button(
            f"导入 {len(eligible)} 篇通过预检的论文",
            type="primary",
            icon=":material/library_add:",
            disabled=not eligible,
            key=f"researchmind_pdf_batch_import_{domain.domain_id}",
        ):
            try:
                backup = create_library_backup(
                    workflow,
                    reason="批量导入前自动备份",
                )
                result = ingest_pdf_files(
                    [
                        IncomingPDF(
                            file_name=item["file_name"],
                            data=item["data"],
                        )
                        for item in eligible
                    ],
                    workflow.knowledge_base_dir,
                )
                sync_result = workflow.sync_bundled_library(
                    domain=domain,
                    force=True,
                )
            except (KnowledgeBaseError, OSError, ValueError) as exc:
                st.error(f"批量导入未完成：{exc}")
                return

            warning_text = (
                f"；向量提示：{result.vector_warning}"
                if result.vector_warning
                else ""
            )
            st.session_state["researchmind_library_operation_notice"] = (
                f"已导入 {result.imported_count} 篇论文，"
                f"专题库当前 {sync_result.total_papers} 篇。"
                f"导入前备份：{backup.path.name}{warning_text}"
            )
            schedule_for_domain(workflow, domain, force=True)
            st.rerun()


def _preflight_pdf_uploads(
    workflow: ResearchMindWorkflow,
    uploaded_files: Iterable[Any],
    domain: ResearchDomain,
) -> tuple[dict[str, Any], ...]:
    """本地预检 PDF 元数据、可提取文本、专题相关度和重复项。"""

    manifest_path = workflow.knowledge_base_dir / "manifest.json"
    manifest_hashes: set[str] = set()
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_data = {}
        for item in manifest_data.get("documents", []):
            if isinstance(item, dict):
                value = str(item.get("pdf_sha256", "")).strip().lower()
                if value:
                    manifest_hashes.add(value)

    existing_papers = workflow.store.list_papers(domain.domain_id)
    known_hashes = set(manifest_hashes)
    known_dois = {
        paper.doi.strip().lower()
        for paper in existing_papers
        if paper.doi.strip() and paper.doi != "待补充"
    }
    known_titles = {
        _normalized_paper_title(paper.title)
        for paper in existing_papers
        if paper.title.strip()
    }
    results: list[dict[str, Any]] = []

    for uploaded in uploaded_files:
        file_name = Path(getattr(uploaded, "name", "未命名.pdf")).name
        data = uploaded.getvalue()
        item: dict[str, Any] = {
            "file_name": file_name,
            "data": data,
            "title": file_name,
            "year": "待核对",
            "doi": "待核对",
            "sha256": "",
            "relevance": INSUFFICIENT_EVIDENCE_TEXT,
            "status": "无法读取",
            "eligible": False,
        }
        results.append(item)
        if not data:
            item["status"] = "空文件"
            continue

        try:
            metadata = extract_pdf_metadata(data, file_name)
        except (OSError, ValueError, RuntimeError) as exc:
            item["status"] = f"元数据无法识别：{exc}"
            continue

        extraction = extract_pdf_text(
            data,
            max_pages=min(metadata.page_count, 5),
            max_chars=14_000,
        )
        relevance = assess_topic_relevance(
            title=metadata.title,
            keywords=metadata.keywords,
            full_text=extraction.text,
        )
        normalized_title = _normalized_paper_title(metadata.title)
        doi = metadata.doi.strip().lower()
        item.update(
            {
                "title": metadata.title,
                "year": metadata.year,
                "doi": metadata.doi,
                "sha256": metadata.pdf_sha256,
                "relevance": relevance.display_status,
            }
        )

        duplicate_reason = ""
        if metadata.pdf_sha256 in known_hashes:
            duplicate_reason = "重复：PDF SHA-256 已存在"
        elif doi and doi != "待补充" and doi in known_dois:
            duplicate_reason = "重复：DOI 已存在"
        elif normalized_title and normalized_title in known_titles:
            duplicate_reason = "重复：标准化标题已存在"

        if duplicate_reason:
            item["status"] = duplicate_reason
            continue
        if not extraction.success or not extraction.pages:
            item["status"] = "未通过：PDF 没有可索引文本"
            continue
        if not relevance.is_core_relevant:
            item["status"] = f"待人工审核：{relevance.status}，不进入核心库"
            continue

        item["status"] = "通过预检，可导入"
        item["eligible"] = True
        known_hashes.add(metadata.pdf_sha256)
        if doi and doi != "待补充":
            known_dois.add(doi)
        if normalized_title:
            known_titles.add(normalized_title)

    return tuple(results)


def _normalized_paper_title(title: str) -> str:
    """统一题名中的空白、大小写和常见标点，用于本地重复检测。"""

    return re.sub(r"[\W_]+", "", title).casefold()


def _render_library_backup_management(
    workflow: ResearchMindWorkflow,
) -> None:
    """显示专题库备份，并在用户明确确认后恢复完整快照。"""

    backups = list_library_backups(workflow)
    with st.expander(
        "专题库备份与恢复",
        icon=":material/settings_backup_restore:",
        on_change="ignore",
    ):
        st.caption(
            "删除和批量导入前会自动创建 ZIP 备份。恢复前系统会再次备份当前状态，"
            "因此误操作仍可回退。"
        )
        if not backups:
            st.info("当前还没有专题库备份。")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "备份文件": item.path.name,
                        "时间": item.created_at,
                        "原因": item.reason,
                        "大小": f"{item.size_bytes / 1024 / 1024:.2f} MB",
                    }
                    for item in backups
                ]
            ),
            width="stretch",
            hide_index=True,
            alt="专题论文库备份清单",
        )
        selected_name = st.selectbox(
            "选择要恢复的备份",
            options=tuple(item.path.name for item in backups),
            key=f"researchmind_backup_select_{workflow.store.db_path}",
        )
        selected = next(
            item for item in backups if item.path.name == selected_name
        )
        confirmed = st.checkbox(
            "我确认用该备份覆盖当前专题库状态",
            key=f"researchmind_backup_confirm_{workflow.store.db_path}",
        )
        if st.button(
            "恢复所选备份",
            icon=":material/restore:",
            disabled=not confirmed,
            key=f"researchmind_backup_restore_{workflow.store.db_path}",
        ):
            try:
                safety_backup = restore_library_backup(
                    workflow,
                    selected.path,
                )
            except (OSError, ValueError, KnowledgeBaseError) as exc:
                st.error(f"专题库恢复未完成：{exc}")
                return
            st.session_state["researchmind_library_operation_notice"] = (
                f"已恢复备份 {selected.path.name}；"
                f"恢复前状态已保存为 {safety_backup.path.name}。"
            )
            schedule_for_domain(workflow, domain, force=True)
            st.rerun()


def _render_library_removal(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    papers: tuple[PaperRecord, ...],
) -> None:
    """提供批量软删除、回收站恢复和永久删除入口。"""

    deleted_papers = workflow.store.list_deleted_papers(domain.domain_id)
    with st.container(border=True):
        st.subheader("论文管理")
        st.caption(
            "移入回收站不会删除 PDF 和证据，可随时恢复；"
            "永久删除前会自动备份专题库，并把历史报告标记为来源失效。"
        )
        if papers:
            paper_by_id = {paper.paper_id: paper for paper in papers}
            selected_ids = tuple(
                st.multiselect(
                    "选择要移入回收站的论文",
                    options=tuple(paper_by_id),
                    format_func=lambda value: paper_by_id[value].title,
                    key=f"researchmind_library_delete_scope_{domain.domain_id}",
                )
            )
            if st.button(
                (
                    f"将选中的 {len(selected_ids)} 篇移入回收站"
                    if selected_ids
                    else "移入回收站"
                ),
                icon=":material/delete:",
                disabled=not selected_ids,
                key=f"researchmind_library_delete_button_{domain.domain_id}",
            ):
                st.session_state["researchmind_delete_confirm_ids"] = (
                    selected_ids
                )
                st.session_state["researchmind_delete_confirm_mode"] = "soft"
                st.rerun()
        else:
            st.caption("当前筛选条件下没有可删除的论文。")

        recycle_bin = st.expander(
            f"回收站（{len(deleted_papers)} 篇）",
            icon=":material/recycling:",
            on_change="rerun",
        )
        if not recycle_bin.open:
            return
        with recycle_bin:
            if not deleted_papers:
                st.info("回收站为空。")
                return
            deleted_by_id = {
                paper.paper_id: paper for paper in deleted_papers
            }
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "题名": paper.title,
                            "作者": paper.authors_text,
                            "年份": paper.year,
                            "DOI": paper.doi,
                            "删除时间": _display_timestamp(
                                paper.deleted_at
                            ),
                            "PDF": (
                                "保留"
                                if Path(paper.pdf_path).is_file()
                                else "无本地文件"
                            ),
                        }
                        for paper in deleted_papers
                    ]
                ),
                width="stretch",
                hide_index=True,
                alt="回收站中的论文清单",
            )
            recycle_ids = tuple(
                st.multiselect(
                    "选择回收站论文",
                    options=tuple(deleted_by_id),
                    format_func=lambda value: deleted_by_id[value].title,
                    key=f"researchmind_recycle_scope_{domain.domain_id}",
                )
            )
            restore_column, delete_column, _ = st.columns([1, 1, 2])
            if restore_column.button(
                f"恢复 {len(recycle_ids)} 篇" if recycle_ids else "恢复论文",
                icon=":material/restore:",
                disabled=not recycle_ids,
                key=f"researchmind_restore_button_{domain.domain_id}",
            ):
                try:
                    restored = restore_papers(
                        workflow.store,
                        domain.domain_id,
                        recycle_ids,
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"论文恢复失败：{exc}")
                else:
                    st.session_state["researchmind_paper_delete_notice"] = (
                        f"已从回收站恢复 {len(restored)} 篇论文。"
                    )
                    schedule_for_domain(workflow, domain, force=True)
                    st.rerun()
            if delete_column.button(
                (
                    f"永久删除 {len(recycle_ids)} 篇"
                    if recycle_ids
                    else "永久删除"
                ),
                icon=":material/delete_forever:",
                disabled=not recycle_ids,
                key=f"researchmind_permanent_delete_button_{domain.domain_id}",
            ):
                st.session_state["researchmind_delete_confirm_ids"] = (
                    recycle_ids
                )
                st.session_state["researchmind_delete_confirm_mode"] = (
                    "permanent"
                )
                st.rerun()

    pending_ids = tuple(
        st.session_state.get("researchmind_delete_confirm_ids", ())
    )
    pending_mode = st.session_state.get(
        "researchmind_delete_confirm_mode",
        "",
    )
    if pending_ids and pending_mode == "soft":
        _confirm_soft_delete(workflow, domain, pending_ids)
    elif pending_ids and pending_mode == "permanent":
        _confirm_permanent_delete(workflow, domain, pending_ids)


def _clear_invalidated_library_state(domain_id: str | None = None) -> None:
    """清理删除或恢复后可能引用旧证据的页面会话结果。"""

    clear_analysis_session_state(domain_id)
    st.session_state.pop("researchmind_active_questions", None)
    st.session_state.pop("researchmind_learning_result", None)


@st.dialog(
    "确认移入回收站",
    width="small",
    icon=":material/delete:",
)
def _confirm_soft_delete(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    paper_ids: tuple[str, ...],
) -> None:
    """确认后软删除论文，保留 PDF、证据和恢复能力。"""

    papers = tuple(
        paper
        for paper_id in paper_ids
        if (
            paper := workflow.store.get_paper(
                paper_id,
                include_deleted=True,
            )
        )
        is not None
    )
    if not papers:
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
        st.error("未找到要删除的论文，请刷新页面后重试。")
        return
    st.write(
        "确定删除选中的论文吗？删除后将从当前论文库隐藏，"
        "但会保留在回收站，可恢复。"
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "题名": paper.title,
                    "年份": paper.year,
                    "DOI": paper.doi,
                }
                for paper in papers
            ]
        ),
        width="stretch",
        hide_index=True,
        alt="待移入回收站的论文",
    )
    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button(
        f"确认移入回收站（{len(papers)} 篇）",
        type="primary",
        icon=":material/delete:",
        width="stretch",
    ):
        try:
            result = soft_delete_papers(
                workflow.store,
                domain.domain_id,
                (paper.paper_id for paper in papers),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"移入回收站失败：{exc}")
            return
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
        st.session_state["researchmind_paper_delete_notice"] = (
            f"已将 {len(result.paper_ids)} 篇论文移入回收站；"
            f"失效分析 {sum(result.invalidated.values())} 项。"
        )
        _clear_invalidated_library_state(domain.domain_id)
        schedule_for_domain(workflow, domain, force=True)
        st.rerun()
    if cancel_column.button(
        "取消",
        icon=":material/close:",
        width="stretch",
    ):
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
        st.rerun()


@st.dialog(
    "确认永久删除",
    width="small",
    icon=":material/delete_forever:",
)
def _confirm_permanent_delete(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    paper_ids: tuple[str, ...],
) -> None:
    """二次确认后永久删除论文记录和独占 PDF。"""

    papers = tuple(
        paper
        for paper_id in paper_ids
        if (
            paper := workflow.store.get_paper(
                paper_id,
                include_deleted=True,
            )
        )
        is not None
        and paper.is_deleted
    )
    if not papers:
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
        st.error("所选论文不在回收站中，请刷新页面后重试。")
        return
    st.warning(
        "永久删除后无法从回收站恢复。系统会先自动备份专题库，"
        "历史报告的来源会标记为已删除。",
        icon=":material/warning:",
    )
    confirmed = st.checkbox(
        f"我确认永久删除选中的 {len(papers)} 篇论文",
        key="researchmind_permanent_delete_confirm",
    )
    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button(
        "确认永久删除",
        type="primary",
        icon=":material/delete_forever:",
        disabled=not confirmed,
        width="stretch",
    ):
        try:
            result = permanently_delete_papers(
                workflow,
                domain,
                (paper.paper_id for paper in papers),
            )
        except (
            KnowledgeBaseError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            st.error(f"永久删除失败：{exc}")
            return
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
        st.session_state["researchmind_paper_delete_notice"] = (
            f"已永久删除 {len(result.paper_ids)} 篇论文；"
            f"自动备份：{result.backup.path.name}。"
        )
        _clear_invalidated_library_state(domain.domain_id)
        schedule_for_domain(workflow, domain, force=True)
        st.rerun()
    if cancel_column.button(
        "取消",
        icon=":material/close:",
        width="stretch",
        key="researchmind_permanent_delete_cancel",
    ):
        st.session_state.pop("researchmind_delete_confirm_ids", None)
        st.session_state.pop("researchmind_delete_confirm_mode", None)
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
    evidence_by_paper = {
        item.paper_id: bool(
            workflow.store.list_evidence(item.paper_id, limit=1)
        )
        for item in papers
    }
    has_full_text = evidence_by_paper[selected_id]
    readable_paper_count = sum(evidence_by_paper.values())

    analysis = workflow.store.get_analysis(selected_id)
    if not has_full_text:
        st.warning(
            "当前论文仅有公开元数据，需导入合法获取的 PDF 后才能进行全文阅读"
            "和证据提取。摘要不会被当作论文全文。",
            icon=":material/picture_as_pdf:",
        )
    if analysis is not None:
        if has_full_text:
            st.success("已读取该论文的结构化结果。")
        else:
            st.info(
                "数据库中存在历史阅读记录，但当前论文没有可检索正文；"
                "该记录不是全文分析结果。",
                icon=":material/history:",
            )
        _render_analysis(analysis)
    else:
        st.info("该论文尚未执行结构化阅读。")

    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button(
            "解析当前论文",
            type="primary",
            icon=":material/document_scanner:",
            disabled=not has_full_text,
            help=(
                "从已建正文证据中定位六个研究字段。"
                if has_full_text
                else "需先导入并解析合法获取的 PDF。"
            ),
        ):
            with st.spinner("正在从已建证据索引中定位六个研究字段..."):
                analysis = workflow.reader.read_paper(selected_id)
            st.session_state[f"researchmind_analysis_{selected_id}"] = analysis
            st.rerun()
        if st.button(
            f"解析有全文的论文（{readable_paper_count} 篇）",
            icon=":material/playlist_play:",
            help="逐篇运行本地规则阅读，不会调用大模型。",
            disabled=readable_paper_count == 0,
        ):
            readable_papers = tuple(
                item for item in papers if evidence_by_paper[item.paper_id]
            )
            with st.spinner("正在解析已获取全文的论文..."):
                analyses = tuple(
                    workflow.reader.read_paper(item.paper_id)
                    for item in readable_papers
                )
            skipped = len(papers) - len(readable_papers)
            st.success(f"已完成 {len(analyses)} 篇论文的结构化阅读。")
            if skipped:
                st.info(
                    f"另有 {skipped} 篇仅有题录，已跳过全文解析，"
                    "不会被标记为已解析。"
                )
            st.rerun()

    st.subheader("领域阅读概览")
    overview = []
    for item in papers:
        item_analysis = workflow.store.get_analysis(item.paper_id)
        overview.append(
            {
                "论文": item.title,
                "全文状态": (
                    "可阅读"
                    if evidence_by_paper[item.paper_id]
                    else "仅题录"
                ),
                "状态": (
                    (
                        "基于摘要分析"
                        if not evidence_by_paper[item.paper_id]
                        and any(
                            evidence.source_type == "摘要"
                            for evidence in item_analysis.all_evidence
                        )
                        else item_analysis.status
                    )
                    if item_analysis
                    else (
                        "需补充全文"
                        if not evidence_by_paper[item.paper_id]
                        else "尚未阅读"
                    )
                ),
                "已定位字段": (
                    f"{item_analysis.filled_field_count}/10"
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
    analysis_state = render_analysis_update_status(
        workflow,
        domain,
        topic.strip() or workflow.default_debate_topic,
        page_key="viewpoint-compare",
        action_label="重新分析观点与分歧",
        auto_schedule=(
            topic.strip() == workflow.default_debate_topic
        ),
    )
    if st.button(
        "运行观点对比",
        type="primary",
        icon=":material/compare_arrows:",
    ):
        if not topic.strip():
            st.error("比较主题不能为空。")
        else:
            schedule_analysis_refresh(
                workflow,
                domain,
                topic,
                force=True,
            )
            clear_analysis_session_state(domain.domain_id)
            st.rerun()

    debate = workflow.store.latest_debate(domain.domain_id)
    if not analysis_state.is_current:
        st.info(
            "论文库版本已变化，旧观点结果已停止展示；后台分析完成后本页会自动更新。",
            icon=":material/hourglass_top:",
        )
        return
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
    analysis_state = render_analysis_update_status(
        workflow,
        domain,
        topic.strip() or workflow.default_debate_topic,
        page_key="debate-analysis",
        action_label="重新分析学术分歧",
        auto_schedule=(
            topic.strip() == workflow.default_debate_topic
        ),
    )
    if st.button(
        "重新分析学术分歧",
        type="primary",
        icon=":material/account_tree:",
    ):
        if not topic.strip():
            st.error("分析主题不能为空。")
        else:
            schedule_analysis_refresh(
                workflow,
                domain,
                topic,
                force=True,
            )
            clear_analysis_session_state(domain.domain_id)
            st.rerun()

    debate = workflow.store.latest_debate(domain.domain_id)
    if not analysis_state.is_current:
        st.info(
            "当前论文库版本的分析正在后台生成；旧版本结论已冻结，不再展示。",
            icon=":material/hourglass_top:",
        )
        return
    if debate is None:
        st.info("当前领域还没有分歧分析结果。")
        return

    columns = st.columns(5)
    columns[0].metric("争议主题", debate.topic)
    columns[1].metric("可追溯观点", f"{len(debate.viewpoints)} 个")
    columns[2].metric("证据强度", debate.evidence_strength)
    columns[3].metric(
        "涉及论文",
        f"{len({paper_id for view in debate.viewpoints for paper_id in view.supporting_papers})} 篇",
    )
    columns[4].metric(
        "分析论文",
        f"{len(debate.considered_paper_ids)} 篇",
    )
    st.caption(
        f"{debate.analysis_mode} · {debate.coverage_note or '覆盖范围未记录'}"
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
    """兼容旧调用点；统一转发到版本感知的知识图谱页面。"""

    render_knowledge_map_page(workflow, domain)


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

    existing_questions = workflow.store.list_questions(
        domain.domain_id,
        limit=100,
    )
    attempts = workflow.store.list_attempts(domain.domain_id, limit=100)
    mastery = workflow.store.ensure_mastery(domain.domain_id)
    overview = st.columns(4)
    overview[0].metric(
        "活动论文",
        f"{workflow.store.paper_count(domain.domain_id)} 篇",
    )
    overview[1].metric("已有试题", f"{len(existing_questions)} 道")
    overview[2].metric("累计答题", f"{len(attempts)} 条")
    overview[3].metric(
        "平均掌握度",
        f"{sum(item.score for item in mastery) / len(mastery):.1f}%",
    )

    with st.form("researchmind_generate_questions_form"):
        st.subheader("自动组卷")
        assessment_modules = st.multiselect(
            "测评模块",
            options=ASSESSMENT_MODULES,
            default=ASSESSMENT_MODULES,
            help=(
                "基础知识理解会自动生成单选、多选、判断和简答题；"
                "其余模块分别生成深度理解、文献对比、学术思辨和个性化补学题。"
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
        if not assessment_modules:
            st.error("请至少选择一个测评模块。")
        else:
            try:
                with st.spinner("正在根据题库论文与原文证据生成试题..."):
                    questions = workflow.assessment.generate_question_set(
                        domain.domain_id,
                        assessment_modules=assessment_modules,
                        debate_topic=topic,
                    )
            except AssessmentGenerationError as error:
                LOGGER.warning(
                    "Assessment generation failed code=%s domain=%s: %s",
                    error.code,
                    domain.domain_id,
                    error,
                )
                st.error(
                    error.user_message,
                    title=f"试卷生成失败 · {error.code}",
                    icon=":material/error:",
                )
                st.info(
                    error.suggestion,
                    title="建议操作",
                    icon=":material/build:",
                )
            except (ResearchMindStorageError, OSError) as error:
                LOGGER.exception(
                    "Assessment storage access failed for domain %s",
                    domain.domain_id,
                )
                st.error(
                    f"论文数据库或本地文件读取失败：{error}",
                    title="数据连接失败",
                    icon=":material/database_off:",
                )
                st.info(
                    "请检查数据库文件是否存在、磁盘空间和读写权限后重试。",
                    icon=":material/build:",
                )
            else:
                st.session_state["researchmind_active_questions"] = tuple(
                    question.question_id for question in questions
                )
                st.session_state["researchmind_assessment_topic"] = topic
                st.session_state.pop("researchmind_learning_result", None)
                st.success(f"已生成 {len(questions)} 道有来源的试题。")
                missing_types = getattr(
                    workflow.assessment,
                    "last_missing_question_types",
                    (),
                )
                if missing_types:
                    st.warning(
                        "以下题型因证据或结构字段不足未生成："
                        + "、".join(missing_types)
                        + "。已生成的其他题型仍保留真实来源。",
                        title="试卷数量不足",
                        icon=":material/quiz:",
                    )
                missing_modules = getattr(
                    workflow.assessment,
                    "last_missing_assessment_modules",
                    (),
                )
                if missing_modules:
                    st.warning(
                        "以下测评模块因证据或结构字段不足未生成："
                        + "、".join(missing_modules)
                        + "。",
                        title="模块生成不完整",
                        icon=":material/extension_off:",
                    )
                if missing_modules:
                    st.info(
                        f"请求 {len(assessment_modules)} 个测评模块，"
                        f"实际生成 {len(questions)} 道题。"
                        "系统不会为补足数量而生成无原文依据的题目。",
                        icon=":material/fact_check:",
                    )
                field_gaps = getattr(
                    workflow.assessment,
                    "last_field_gap_counts",
                    {},
                )
                optional_gaps = tuple(
                    f"{FIELD_LABELS[field]}缺少 {count} 篇"
                    for field, count in field_gaps.items()
                    if field in {
                        "interventions",
                        "comparators",
                        "measures",
                        "statistics",
                    }
                    and count
                )
                if optional_gaps:
                    st.caption(
                        "非必需字段覆盖提示："
                        + "；".join(optional_gaps)
                        + "。缺失字段只会影响依赖该字段的题目，不会使整份试卷失效。"
                    )
                critical_gaps = tuple(
                    f"{FIELD_LABELS[field]}缺少 {count} 篇"
                    for field, count in field_gaps.items()
                    if field in {"methods", "results", "conclusions"}
                    and count
                )
                if critical_gaps:
                    st.warning(
                        "部分论文缺少组卷关键字段："
                        + "；".join(critical_gaps)
                        + "。系统已仅使用具备原文证据的论文和字段。",
                        title="论文结构字段不完整",
                        icon=":material/data_alert:",
                    )
                unparsed_titles = getattr(
                    workflow.assessment,
                    "last_unparsed_titles",
                    (),
                )
                parse_errors = getattr(
                    workflow.assessment,
                    "last_parse_errors",
                    (),
                )
                if unparsed_titles:
                    st.warning(
                        f"{len(unparsed_titles)} 篇论文没有可用于出题的解析证据，"
                        "已跳过："
                        + "、".join(unparsed_titles[:5])
                        + ("…" if len(unparsed_titles) > 5 else ""),
                        title="部分论文未参与组卷",
                        icon=":material/document_scanner:",
                    )
                if parse_errors:
                    st.error(
                        "部分论文解析失败，已跳过且未用模拟内容替代："
                        + "；".join(parse_errors[:3]),
                        title="论文解析异常",
                        icon=":material/error:",
                    )

    questions = _active_questions(workflow, domain)
    if not questions:
        st.info("尚未生成试卷。可选择测评模块后点击“生成新试卷”。")
        return

    module_questions: dict[str, list[AssessmentQuestion]] = {
        module: [] for module in ASSESSMENT_MODULES
    }
    for question in questions:
        module_questions.setdefault(question.assessment_module, []).append(question)
    available_modules = tuple(
        module for module in ASSESSMENT_MODULES if module_questions.get(module)
    )
    selected_module = st.segmented_control(
        "当前测评模块",
        options=available_modules,
        default=available_modules[0],
        key=f"researchmind-assessment-module-{domain.domain_id}",
    )
    if selected_module is None:
        return
    questions = tuple(module_questions[selected_module])
    module_descriptions = {
        "基础知识理解": "研究背景、目的、对象、方法、变量、结论和局限。",
        "文献深度理解": "研究设计合理性、样本代表性、结果支持范围与证据边界。",
        "文献对比分析": "比较对象、方法、样本、干预、测量、结果和结论一致性。",
        "学术思辨训练": "分析结论差异、证据强弱、偏倚、混杂与研究改进。",
        "个性化错题与补学": "根据错题标签和已测练习表现生成针对性训练。",
    }
    st.caption(module_descriptions.get(selected_module, ""))

    answers: dict[str, str] = {}
    with st.form(f"researchmind_answer_sheet::{selected_module}"):
        st.subheader("答题")
        for number, question in enumerate(questions, start=1):
            with st.container(border=True):
                st.markdown(
                    f"**第 {number} 题 · {question.assessment_module} · "
                    f"{question.question_type}**"
                )
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
                assessment_modules=(
                    selected_module,
                    "个性化错题与补学",
                ),
            )
        st.session_state["researchmind_learning_result"] = result
        if result.next_questions:
            st.session_state["researchmind_active_questions"] = tuple(
                question.question_id for question in result.next_questions
            )

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
            _render_question_sources(question, workflow)

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
        "展示已记录题目的练习表现、答题趋势和下一轮学习安排；未作答维度显示“未测”。",
    )
    mastery = workflow.store.ensure_mastery(domain.domain_id)
    attempts = workflow.store.list_attempts(domain.domain_id)
    wrong_attempts = workflow.store.list_wrong_attempts(domain.domain_id)
    measured = tuple(item for item in mastery if item.attempts_count > 0)
    total_score = sum(item.score for item in attempts)
    total_max = sum(item.max_score for item in attempts)
    overall = 100 * total_score / total_max if total_max else 0.0
    metrics = st.columns(4)
    metrics[0].metric("累计答题", f"{len(attempts)} 条")
    metrics[1].metric("总体得分率", f"{overall:.1f}%")
    metrics[2].metric("错题记录", f"{len(wrong_attempts)} 条")
    metrics[3].metric(
        "已测维度平均练习表现",
        (
            f"{sum(item.score for item in measured) / len(measured):.1f}%"
            if measured
            else "未测"
        ),
    )

    mastery_frame = pd.DataFrame(
        {
            "能力维度": [item.dimension for item in mastery],
            "练习表现": [
                item.score if item.attempts_count > 0 else None
                for item in mastery
            ],
            "作答次数": [item.attempts_count for item in mastery],
            "状态": [
                "已测" if item.attempts_count > 0 else "未测"
                for item in mastery
            ],
        }
    ).set_index("能力维度")
    st.subheader("五项科研练习表现（未校准）")
    if measured:
        st.bar_chart(
            mastery_frame.loc[
                mastery_frame["状态"] == "已测",
                ["练习表现"],
            ],
            height=300,
            alt="ResearchMind 已测科研练习表现柱状图",
        )
    else:
        st.info("尚无答题记录，五个维度均为“未测”，不生成 0 分能力画像。")
    st.dataframe(
        mastery_frame.reset_index(),
        width="stretch",
        hide_index=True,
        alt="ResearchMind 科研练习表现明细",
    )
    st.caption(
        "“练习表现”是已记录题目得分率的算术平均值，不是经过量表校准的能力等级；"
        "没有答题记录的维度不参与薄弱项判断。"
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
    metrics = st.columns(5)
    metrics[0].metric("主题", debate.topic)
    metrics[1].metric("观点", f"{len(debate.viewpoints)} 个")
    metrics[2].metric("证据强度", debate.evidence_strength)
    metrics[3].metric(
        "证据页码",
        f"{sum(len(item.evidence) for item in debate.viewpoints)} 条",
    )
    metrics[4].metric(
        "纳入论文",
        f"{len(debate.considered_paper_ids)} 篇",
    )
    st.caption(
        f"{debate.analysis_mode} · {debate.coverage_note or '覆盖范围未记录'}"
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
            st.caption(
                f"主张来源：{viewpoint.claim_origin}；"
                f"立场判断：{viewpoint.stance_origin}"
            )
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
                f"{item.file_name}　|　{item.source_type}　|　"
                f"{item.location_label}　|　"
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


def _render_question_sources(
    question: AssessmentQuestion,
    workflow: ResearchMindWorkflow | None = None,
) -> None:
    st.markdown("**标准答案**")
    st.write(
        _format_question_answer(
            question,
            question.standard_answer,
        )
    )
    st.markdown("**详细解析**")
    if question.explanation.strip():
        st.write(question.explanation)
    else:
        st.info(
            "该题来自历史题库，尚未保存独立解析。可依据下方标准答案、"
            "评分细则和原文证据核对，不使用题目之外的信息补写。",
            icon=":material/history_edu:",
        )
    _render_question_rubric(question)
    if workflow is not None:
        source_rows = []
        for paper_id in question.source_paper_ids:
            paper = workflow.store.get_paper(paper_id)
            if paper is None:
                source_rows.append(
                    {
                        "论文": f"已删除或不可用：{paper_id}",
                        "作者": "",
                        "年份": "",
                        "DOI": "",
                    }
                )
                continue
            source_rows.append(
                {
                    "论文": paper.title,
                    "作者": paper.authors_text,
                    "年份": paper.year,
                    "DOI": paper.doi or "未提供",
                }
            )
        if source_rows:
            st.markdown("**对应论文来源**")
            st.dataframe(
                pd.DataFrame(source_rows),
                width="stretch",
                hide_index=True,
                alt="本题对应论文来源",
            )
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
            st.markdown(
                f"**{question.assessment_module} · {question.question_type}　"
                f"{score.score:g}/{score.max_score:g}**"
            )
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
            with st.expander("查看详细解析、论文来源与原文证据"):
                _render_question_sources(question, workflow)

    mastery_frame = pd.DataFrame(
        {
            "维度": list(result.mastery),
            "练习表现": list(result.mastery.values()),
        }
    ).set_index("维度")
    st.subheader("更新后的科研练习表现（未校准）")
    st.bar_chart(
        mastery_frame,
        height=230,
        alt="本轮答题后的五项科研练习表现",
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
                workflow.knowledge_base_dir,
            )
    except (
        KnowledgeBaseError,
        LiteratureCollectionError,
        ValueError,
        OSError,
    ) as exc:
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
        schedule_for_domain(workflow, domain, force=True)
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
        if (
            question is not None
            and question.domain_id == domain.domain_id
            and _question_sources_active(workflow, question)
        ):
            selected.append(question)
    if selected:
        return tuple(selected)
    active = tuple(
        question
        for question in workflow.store.list_questions(
            domain.domain_id,
            limit=50,
        )
        if _question_sources_active(workflow, question)
    )
    return active[:6]


def _question_sources_active(
    workflow: ResearchMindWorkflow,
    question: AssessmentQuestion,
) -> bool:
    """题目及其所有选项证据必须仍关联活动论文。"""

    source_ids = set(question.source_paper_ids)
    source_ids.update(
        evidence.document_id for evidence in question.evidence
    )
    return bool(source_ids) and all(
        workflow.store.get_paper(paper_id) is not None
        for paper_id in source_ids
    )


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
        types.extend(("多项选择题", "简答题", "深度理解题"))
    if "对比分析" in weak:
        types.append("文献对比分析题")
    if "批判性思维" in weak:
        types.extend(("论述题", "学术思辨题", "文献对比分析题"))
    if "学术应用" in weak:
        types.extend(("论述题", "简答题"))
    if not types:
        types.extend(("多项选择题", "深度理解题", "文献对比分析题"))
    types.append("个性化补学题")
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
    "RESEARCHMIND_PAGE_ALIASES",
    "RESEARCHMIND_PAGES",
    "get_researchmind_workflow",
    "render_researchmind_page",
]
