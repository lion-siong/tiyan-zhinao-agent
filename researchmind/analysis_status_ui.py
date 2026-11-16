"""知识图谱与学术分歧页面的共享更新状态组件。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

from researchmind.library_state import (
    AnalysisState,
    get_analysis_state,
    schedule_analysis_refresh,
)
from researchmind.models import ResearchDomain
from researchmind.workflow import ResearchMindWorkflow


def clear_analysis_session_state(domain_id: str | None = None) -> None:
    """只清理科研分析相关的会话结果，不动导航和用户输入。"""

    suffixes = (domain_id,) if domain_id else ()
    for key in tuple(st.session_state):
        if not (
            key.startswith("researchmind_analysis_")
            or key.startswith("researchmind_debate_")
            or key.startswith("researchmind_synthesis_")
            or key.startswith("analysis_ui_state_")
        ):
            continue
        if suffixes and all(suffix not in key for suffix in suffixes):
            continue
        st.session_state.pop(key, None)


def schedule_for_domain(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    *,
    force: bool = False,
) -> None:
    """使用领域默认主题启动后台派生分析。"""

    schedule_analysis_refresh(
        workflow,
        domain,
        workflow.default_debate_topic,
        force=force,
    )
    clear_analysis_session_state(domain.domain_id)


@st.fragment(run_every="3s")
def render_analysis_update_status(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    topic: str,
    *,
    page_key: str,
    action_label: str,
    auto_schedule: bool = True,
) -> AnalysisState:
    """展示版本状态，并在必要时自动启动一次后台重算。"""

    state = get_analysis_state(workflow.store, domain.domain_id, topic)
    if workflow.read_only:
        st.info(
            "旧库为只读模式，不启动派生分析更新。",
            icon=":material/lock:",
        )
        return state
    should_schedule = (
        workflow.read_only is False
        and state.paper_count > 0
        and state.status != "失败"
        and auto_schedule
    )
    if should_schedule and not state.is_current:
        try:
            schedule_analysis_refresh(workflow, domain, topic)
            state = get_analysis_state(workflow.store, domain.domain_id, topic)
        except Exception as error:
            st.error(
                f"无法启动后台分析：{type(error).__name__}: {error}",
                title="分析启动失败",
            )

    columns = st.columns([1, 1.2, 1.2, 1.3, 0.8])
    columns[0].metric("当前论文", f"{state.paper_count} 篇")
    columns[1].metric("科研库版本", state.version)
    columns[2].metric(
        "最近分析",
        _display_time(state.latest_analysis_at),
    )
    columns[3].metric(
        "最新论文",
        "已纳入" if state.latest_paper_included else "尚未纳入",
    )
    columns[4].metric("状态", state.status)

    if state.status == "失败" and state.error:
        st.error(
            state.error,
            title="科研分析失败",
            icon=":material/error:",
        )
    elif state.status == "分析中":
        st.info(
            "后台正在重新阅读当前论文库并更新知识图谱与学术分歧。",
            title="分析进行中",
            icon=":material/progress_activity:",
        )
    elif not state.is_current:
        st.warning(
            "论文库已变化，当前页面会等待新分析完成后再展示结论，"
            "不会继续显示与当前版本不一致的旧结果。",
            title="派生分析待更新",
            icon=":material/schedule:",
        )

    if st.button(
        action_label,
        key=f"analysis-force-{page_key}-{domain.domain_id}",
        icon=":material/refresh:",
        help="后台重新阅读全部活动论文，并更新学术分歧与知识图谱。",
    ):
        try:
            schedule_analysis_refresh(
                workflow,
                domain,
                topic,
                force=True,
            )
            clear_analysis_session_state(domain.domain_id)
            st.success("已启动后台分析。")
        except Exception as error:
            st.error(f"重新分析失败：{type(error).__name__}: {error}")

    marker_key = f"analysis_ui_state_{page_key}_{domain.domain_id}"
    previous_status = st.session_state.get(marker_key)
    st.session_state[marker_key] = state.status
    if previous_status == "分析中" and state.is_current:
        st.rerun()
    return state


def _display_time(value: str) -> str:
    if not value:
        return "暂无"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone().strftime("%m-%d %H:%M")
    except ValueError:
        return value[:16]


__all__ = [
    "clear_analysis_session_state",
    "render_analysis_update_status",
    "schedule_for_domain",
]
