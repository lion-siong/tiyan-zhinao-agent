"""科研工作台的研究简报、证据检索与页面交互。

工作台负责把用户输入、规则规划、检索策略、真实论文证据和已有数据分析
结果组织到同一个可检查的页面中。它不会自动生成学术结论，也不会把尚未
执行的步骤标记为已完成。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from agents.planner import ResearchPlan, ResearchPlannerAgent, SIMULATION_NOTICE
from researchmind.models import EvidenceRef
from researchmind.workflow import (
    ResearchMindWorkflow,
    ResearchMindWorkflowError,
)
from services.rag_service import LocalEvidenceIndex
from tools.document_chunker import EvidenceChunk
from tools.literature_search import SearchStrategy, build_search_strategy


@dataclass(frozen=True)
class WorkbenchBrief:
    """工作台根据用户输入形成的研究简报。"""

    task: str
    population: str
    method: str
    research_question: str
    plan: ResearchPlan
    strategy: SearchStrategy
    design_template: tuple[tuple[str, str], ...]
    evidence_query: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "population": self.population,
            "method": self.method,
            "research_question": self.research_question,
            "plan": self.plan.to_dict(),
            "search_strategy": self.strategy.to_dict(),
            "design_template": [
                {"label": label, "content": content}
                for label, content in self.design_template
            ],
            "evidence_query": self.evidence_query,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class WorkbenchEvidenceHit:
    """一条来自真实论文 PDF 的工作台证据。"""

    document_id: str
    document_title: str
    file_name: str
    page_number: int
    quote: str
    score: float

    @property
    def citation(self) -> str:
        return (
            f"{self.document_title}（{self.file_name}），"
            f"第 {self.page_number} 页"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "quote": self.quote,
            "score": round(self.score, 6),
        }


@dataclass(frozen=True)
class WorkbenchStage:
    """工作台中的一个真实工作阶段。"""

    title: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }


def build_workbench_brief(
    task: str,
    *,
    population: str = "",
    method: str = "",
) -> WorkbenchBrief:
    """根据真实用户输入生成研究简报和检索策略。

    这里的规划与设计内容来自规则模板，不代表已经检索或验证了学术事实。
    """

    clean_task = task.strip()
    if not clean_task:
        raise ValueError("研究任务不能为空。")

    clean_population = population.strip()
    clean_method = method.strip()
    subject = clean_population or "目标研究对象"
    if clean_method:
        research_question = (
            f"在{subject}中，{clean_method}与“{clean_task}”相关结果之间"
            "存在怎样的关系？"
        )
    else:
        research_question = (
            f"在{subject}中，“{clean_task}”的现状、相关因素及证据边界是什么？"
        )

    plan = ResearchPlannerAgent().create_plan(clean_task)
    strategy = build_search_strategy(
        clean_task,
        clean_population,
        clean_method,
    )
    design_template = (
        (
            "研究对象",
            f"暂定对象为{subject}。纳入与排除标准必须结合研究条件确定，"
            "不得在未收集数据前预设样本量或结果。",
        ),
        (
            "核心变量",
            "根据研究问题列出暴露或干预、结局指标及可能的混杂因素。"
            "变量名称和测量方式应优先参考可核验文献。",
        ),
        (
            "研究设计",
            "先判断横断面调查、纵向观察、准实验或随机对照设计是否可行。"
            "相关关系不能直接解释为因果关系。",
        ),
        (
            "测量与数据",
            "记录量表、设备、测试流程、缺失值处理计划与伦理审查要求。"
            "真实数据必须在收集或获得授权后再上传分析。",
        ),
        (
            "分析与核验",
            "先做描述统计和数据质量检查，再选择与研究设计匹配的统计方法。"
            "所有数字结果必须来自真实数据文件。",
        ),
    )
    evidence_terms = tuple(
        dict.fromkeys(
            item
            for item in (clean_task, clean_population, clean_method)
            if item
        )
    )
    return WorkbenchBrief(
        task=clean_task,
        population=clean_population,
        method=clean_method,
        research_question=research_question,
        plan=plan,
        strategy=strategy,
        design_template=design_template,
        evidence_query=" ".join(evidence_terms),
        generated_at=_now(),
    )


def search_domain_evidence(
    evidence: Iterable[EvidenceRef],
    query: str,
    *,
    top_k: int = 5,
) -> tuple[WorkbenchEvidenceHit, ...]:
    """使用现有 BM25 索引检索领域内真实原文，并保留论文与页码。"""

    clean_query = query.strip()
    if not clean_query:
        raise ValueError("证据检索问题不能为空。")
    if not 1 <= top_k <= 10:
        raise ValueError("证据数量必须在 1 到 10 之间。")

    references = tuple(evidence)
    if not references:
        return ()

    reference_by_chunk = {item.chunk_id: item for item in references}
    chunks = tuple(
        EvidenceChunk(
            chunk_id=item.chunk_id,
            document_name=item.file_name,
            page_number=item.page_number,
            text=item.quote,
        )
        for item in references
    )
    results = LocalEvidenceIndex(chunks).search(clean_query, top_k=top_k)
    hits: list[WorkbenchEvidenceHit] = []
    for result in results:
        reference = reference_by_chunk.get(result.chunk.chunk_id)
        if reference is None:
            continue
        hits.append(
            WorkbenchEvidenceHit(
                document_id=reference.document_id,
                document_title=reference.document_title,
                file_name=reference.file_name,
                page_number=reference.page_number,
                quote=reference.quote,
                score=result.score,
            )
        )
    return tuple(hits)


def build_workbench_stages(
    brief: WorkbenchBrief,
    *,
    evidence_hits: tuple[WorkbenchEvidenceHit, ...] = (),
    evidence_searched: bool = False,
    data_summary: Mapping[str, Any] | None = None,
) -> tuple[WorkbenchStage, ...]:
    """根据当前真实产物生成阶段状态，不虚报未执行的工作。"""

    if evidence_hits:
        evidence_status = f"已命中 {len(evidence_hits)} 条可追溯证据"
        evidence_detail = (
            "证据来自当前领域已登记的真实 PDF，可查看文件名和页码。"
        )
    elif evidence_searched:
        evidence_status = "证据不足"
        evidence_detail = (
            "本次检索没有命中原文。应调整关键词或补充真实 PDF，"
            "不能据此推断文献中不存在相关研究。"
        )
    else:
        evidence_status = "待检索"
        evidence_detail = "运行工作台证据检索后才会显示真实原文命中。"

    if data_summary:
        file_name = str(data_summary.get("file_name", "已上传文件"))
        rows = int(data_summary.get("rows", 0))
        columns = int(data_summary.get("columns", 0))
        data_status = "已有真实数据分析结果"
        data_detail = f"{file_name}：{rows} 行、{columns} 列。"
    else:
        data_status = "待操作"
        data_detail = (
            "请到“数据分析”页上传 CSV 或 XLSX；工作台不会生成替代数据。"
        )

    return (
        WorkbenchStage(
            "01 研究问题定义",
            "已定义",
            brief.research_question,
        ),
        WorkbenchStage(
            "02 文献检索策略",
            "已生成",
            brief.strategy.chinese_query,
        ),
        WorkbenchStage(
            "03 真实文献与证据",
            evidence_status,
            evidence_detail,
        ),
        WorkbenchStage(
            "04 研究设计模板",
            "模板已生成，待人工确认",
            "模板只提供检查项，不替代导师审核、伦理审查和预注册。",
        ),
        WorkbenchStage(
            "05 真实数据分析",
            data_status,
            data_detail,
        ),
        WorkbenchStage(
            "06 科研报告",
            "待撰写",
            "工作台可导出工作底稿，但不会自动补写文献结论或统计结果。",
        ),
    )


def build_workbench_export(
    brief: WorkbenchBrief,
    *,
    evidence_hits: tuple[WorkbenchEvidenceHit, ...] = (),
    evidence_query: str = "",
    evidence_searched: bool = False,
    data_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """生成可下载的 JSON 工作底稿，范围仅限已产生的真实内容。"""

    safe_data_summary = dict(data_summary) if data_summary else None
    return {
        "schema_version": 1,
        "notice": (
            "这是科研工作底稿，不是学术论文或综述。规划与设计内容来自"
            "规则模板；证据命中来自已登记 PDF；数据结果仅来自用户上传文件。"
        ),
        "exported_at": _now(),
        "brief": brief.to_dict(),
        "evidence_search": {
            "query": evidence_query.strip(),
            "searched": evidence_searched,
            "hits": [item.to_dict() for item in evidence_hits],
        },
        "data_analysis": safe_data_summary,
        "stages": [
            item.to_dict()
            for item in build_workbench_stages(
                brief,
                evidence_hits=evidence_hits,
                evidence_searched=evidence_searched,
                data_summary=safe_data_summary,
            )
        ],
    }


def render_workbench(
    workflow: ResearchMindWorkflow,
    request_navigation: Callable[[str], None],
) -> None:
    """渲染可操作的科研工作台页面。"""

    st.markdown(
        '<div class="section-label">RESEARCH WORKSPACE</div>',
        unsafe_allow_html=True,
    )
    st.title("科研工作台", icon=":material/space_dashboard:")
    st.caption(
        "研究问题、检索策略、真实证据、研究设计和数据分析，在一个"
        "可检查的工作流中完成。"
    )
    st.warning(
        "规则规划与设计模板为模拟建议；引用、统计结果和最终结论"
        "必须来自真实文献与数据核验。",
        icon=":material/fact_check:",
    )

    try:
        workflow.ensure_default_domain()
        domains = workflow.store.list_domains()
        domain_by_id = {domain.domain_id: domain for domain in domains}
        active_id = st.session_state.get("researchmind_active_domain")
        if active_id not in domain_by_id:
            active_id = domains[0].domain_id
        selected_id = st.selectbox(
            "当前研究领域",
            options=tuple(domain_by_id),
            index=tuple(domain_by_id).index(active_id),
            format_func=lambda value: domain_by_id[value].name,
            key="workbench_domain_selector",
            help="工作台会读取该领域已同步的真实论文和证据。",
        )
        st.session_state["researchmind_active_domain"] = selected_id
        domain = workflow.ensure_ready(selected_id)
        counts = workflow.store.counts(domain.domain_id)
    except (ResearchMindWorkflowError, ValueError, OSError) as exc:
        st.error(f"研究领域暂时无法加载：{exc}")
        return

    st.caption(
        f"真实论文 {counts['papers']} 篇　·　"
        f"带页码证据 {counts['evidence']} 条　·　"
        f"已生成试题 {counts['questions']} 道　·　"
        f"答题记录 {counts['attempts']} 条　·　"
        f"大模型 {'已配置' if workflow.config.llm_configured else '未配置'}"
    )

    existing_brief = st.session_state.get("workbench_brief")
    if not isinstance(existing_brief, WorkbenchBrief):
        existing_brief = None
    prefill = str(st.session_state.pop("workbench_prefill", "") or "")
    if "workbench_task_input" not in st.session_state:
        st.session_state["workbench_task_input"] = (
            prefill or (existing_brief.task if existing_brief else "")
        )
    if "workbench_population_input" not in st.session_state:
        st.session_state["workbench_population_input"] = (
            existing_brief.population if existing_brief else ""
        )
    if "workbench_method_input" not in st.session_state:
        st.session_state["workbench_method_input"] = (
            existing_brief.method if existing_brief else ""
        )

    with st.expander(
        "设置研究任务",
        expanded=existing_brief is None,
        icon=":material/edit_note:",
    ):
        with st.form("workbench_brief_form"):
            task = st.text_area(
                "研究任务",
                key="workbench_task_input",
                height=100,
                placeholder=(
                    "例如：研究青少年体育锻炼干预与体质健康水平的关系，"
                    "并制定可执行的数据分析计划。"
                ),
            )
            input_columns = st.columns(2)
            population = input_columns[0].text_input(
                "研究对象",
                key="workbench_population_input",
                placeholder="例如：12-18 岁在校青少年",
            )
            method = input_columns[1].text_input(
                "方法或干预",
                key="workbench_method_input",
                placeholder="例如：问卷调查、体育锻炼行为量表",
            )
            submitted = st.form_submit_button(
                "生成研究简报与检索策略",
                type="primary",
                icon=":material/route:",
            )

        if submitted:
            try:
                brief = build_workbench_brief(
                    task,
                    population=population,
                    method=method,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["workbench_brief"] = brief
                st.session_state["workbench_evidence_hits"] = ()
                st.session_state["workbench_evidence_searched"] = False
                st.session_state["workbench_evidence_query"] = (
                    brief.evidence_query
                )
                st.session_state.pop(
                    "workbench_evidence_query_input",
                    None,
                )
                existing_brief = brief
                st.rerun()

    brief = st.session_state.get("workbench_brief")
    if not isinstance(brief, WorkbenchBrief):
        st.info(
            "先填写研究任务并生成研究简报，工作台会逐步展开后续操作。"
        )
        _render_workbench_actions(request_navigation)
        return

    raw_data_summary = st.session_state.get("workbench_data_summary")
    data_summary = (
        dict(raw_data_summary)
        if isinstance(raw_data_summary, Mapping)
        else None
    )

    tabs = st.tabs(
        (
            "研究规划",
            "证据与阶段",
            "研究设计",
            "导出与下一步",
        )
    )

    with tabs[0]:
        st.subheader("研究简报")
        st.info(brief.research_question)
        st.caption(
            f"任务类型：{brief.plan.task_type}　|　"
            f"规划方式：{brief.plan.mode}　|　"
            f"生成时间：{_short_time(brief.generated_at)}"
        )

        with st.expander("查看规则规划步骤", expanded=False):
            for step in brief.plan.steps:
                st.markdown(f"**{step.order:02d}　{step.title}**")
                st.write(step.description)
                st.caption(
                    f"工具：{step.tool_name}　|　"
                    f"预期结果：{step.expected_output}　|　状态：{step.status}"
                )
            st.caption(SIMULATION_NOTICE)

        st.subheader("检索策略")
        strategy_columns = st.columns(2)
        with strategy_columns[0]:
            st.markdown("**中文数据库检索式草稿**")
            st.code(brief.strategy.chinese_query, language="text")
        with strategy_columns[1]:
            st.markdown("**英文检索结构提示**")
            st.code(brief.strategy.english_query_hint, language="text")
        st.caption("建议数据库：" + "、".join(brief.strategy.databases))

    with tabs[1]:
        st.subheader("检索真实原文")
        st.caption(
            f"当前领域有 {counts['evidence']} 条带页码证据。"
            "检索结果只展示真实 PDF 原文片段。"
        )
        if counts["evidence"] == 0:
            st.warning(
                "当前青少年专题库还没有可追溯的 PDF 页码证据。"
                "证据不足，暂不能判断；请先在“文献自动采集”或"
                "“我的论文库”导入真实且相关的论文。",
                icon=":material/library_books:",
            )
        if "workbench_evidence_query_input" not in st.session_state:
            st.session_state["workbench_evidence_query_input"] = (
                brief.evidence_query
            )
        evidence_columns = st.columns(
            [4, 1],
            vertical_alignment="bottom",
        )
        evidence_columns[0].text_input(
            "证据检索问题",
            key="workbench_evidence_query_input",
            placeholder="可修改为更具体的问题或关键词",
        )
        top_k = evidence_columns[1].number_input(
            "证据数量",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

        if st.button(
            "检索当前领域的真实证据",
            key="workbench_search_evidence",
            icon=":material/search:",
            width="stretch",
        ):
            query = str(
                st.session_state.get("workbench_evidence_query_input", "")
            )
            with st.spinner("正在检索已登记论文的真实原文..."):
                try:
                    hits = search_domain_evidence(
                        workflow.store.list_domain_evidence(
                            domain.domain_id
                        ),
                        query,
                        top_k=int(top_k),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["workbench_evidence_hits"] = hits
                    st.session_state["workbench_evidence_searched"] = True
                    st.session_state["workbench_evidence_query"] = (
                        query.strip()
                    )

        evidence_hits = st.session_state.get("workbench_evidence_hits", ())
        if not isinstance(evidence_hits, tuple) or not all(
            isinstance(item, WorkbenchEvidenceHit) for item in evidence_hits
        ):
            evidence_hits = ()
        evidence_searched = bool(
            st.session_state.get("workbench_evidence_searched", False)
        )

        if evidence_hits:
            st.markdown("**真实证据命中**")
            for index, hit in enumerate(evidence_hits, start=1):
                with st.expander(
                    f"证据 {index} · {hit.citation}",
                    expanded=index == 1,
                ):
                    st.caption(f"匹配分数：{hit.score:.3f}")
                    st.write(hit.quote)
            st.caption(
                "以上是检索命中的原文片段，不代表系统已经判断其能回答"
                "研究问题。"
            )
        elif evidence_searched:
            st.warning(
                "本次检索没有命中可追溯原文。证据不足，暂不能判断；"
                "请调整关键词，或导入包含相关内容的真实 PDF。",
                icon=":material/search_off:",
            )
        else:
            st.info(
                "点击“检索当前领域的真实证据”查看与当前问题相关的"
                "原文片段。"
            )

        st.subheader("研究阶段")
        current_stages = build_workbench_stages(
            brief,
            evidence_hits=evidence_hits,
            evidence_searched=evidence_searched,
            data_summary=data_summary,
        )
        stage_table = pd.DataFrame(
            [
                {
                    "阶段": item.title,
                    "当前状态": item.status,
                    "说明": item.detail,
                }
                for item in current_stages
            ]
        )
        st.dataframe(
            stage_table,
            width="stretch",
            hide_index=True,
            alt="科研工作台各阶段真实状态",
        )

    with tabs[2]:
        st.subheader("真实数据分析")
        if data_summary:
            data_metrics = st.columns(4)
            data_metrics[0].metric(
                "文件",
                str(data_summary.get("file_name", "-")),
            )
            data_metrics[1].metric("行数", data_summary.get("rows", 0))
            data_metrics[2].metric("列数", data_summary.get("columns", 0))
            data_metrics[3].metric(
                "缺失单元格",
                data_summary.get("missing_cells", 0),
            )
            st.caption(
                "该摘要来自本会话最近一次成功读取的真实数据文件。"
                "图表与完整统计请进入“数据分析”页查看。"
            )
        else:
            st.info(
                "当前会话还没有真实数据分析结果。请到“数据分析”页上传"
                " CSV 或 XLSX 文件。"
            )

        st.subheader("研究设计检查模板")
        st.caption(
            "以下均为模板提示，需结合真实文献、导师意见和伦理要求修改。"
        )
        for label, content in brief.design_template:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.write(content)

    with tabs[3]:
        st.subheader("导出科研工作底稿")
        st.write(
            "导出的 JSON 包含研究问题、规则规划、检索策略、真实证据命中"
            "和当前数据摘要，不包含自动撰写的论文结论。"
        )
        evidence_query = str(
            st.session_state.get(
                "workbench_evidence_query",
                brief.evidence_query,
            )
        )
        evidence_hits = st.session_state.get("workbench_evidence_hits", ())
        if not isinstance(evidence_hits, tuple):
            evidence_hits = ()
        evidence_searched = bool(
            st.session_state.get("workbench_evidence_searched", False)
        )
        export_data = build_workbench_export(
            brief,
            evidence_hits=evidence_hits,
            evidence_query=evidence_query,
            evidence_searched=evidence_searched,
            data_summary=data_summary,
        )
        st.download_button(
            "下载科研工作底稿 JSON",
            data=json.dumps(
                export_data,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="research_workbench_draft.json",
            mime="application/json",
            icon=":material/download:",
            help="工作底稿只包含用户输入、规则模板、真实证据命中和已有数据摘要。",
            width="stretch",
        )
        st.space("small")
        _render_workbench_actions(request_navigation)


def _render_workbench_actions(
    request_navigation: Callable[[str], None],
) -> None:
    """提供进入现有真实功能页面的快捷入口。"""

    st.subheader("继续执行")
    actions = (
        ("文献自动采集", ":material/travel_explore:"),
        ("论文智能阅读", ":material/document_scanner:"),
        ("文献证据问答", ":material/manage_search:"),
        ("数据分析", ":material/table_chart:"),
    )
    columns = st.columns(2)
    for index, (page, icon) in enumerate(actions):
        with columns[index % 2]:
            if st.button(
                page,
                key=f"workbench_action_{index}",
                icon=icon,
                width="stretch",
            ):
                request_navigation(page)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _short_time(value: str) -> str:
    """把 ISO 时间显示为适合界面的本地时间文本。"""

    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


__all__ = [
    "WorkbenchBrief",
    "WorkbenchEvidenceHit",
    "WorkbenchStage",
    "build_workbench_brief",
    "build_workbench_export",
    "build_workbench_stages",
    "render_workbench",
    "search_domain_evidence",
]
