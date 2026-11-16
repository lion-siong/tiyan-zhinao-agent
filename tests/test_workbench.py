"""科研工作台的真实证据、阶段状态与页面导航测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from researchmind.models import ResearchDomain
from researchmind.storage import ResearchMindStore
from researchmind.workbench import (
    build_workbench_brief,
    build_workbench_export,
    build_workbench_stages,
    search_domain_evidence,
)
from researchmind.workflow import ResearchMindWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
TOPIC = "大学生体育锻炼行为与体质健康水平的关系研究"


@pytest.fixture(scope="module")
def workbench_workflow(
    tmp_path_factory,
) -> tuple[ResearchMindWorkflow, ResearchDomain]:
    """在临时数据库中准备真实论文和带页码证据。"""

    root = tmp_path_factory.mktemp("workbench")
    workflow = ResearchMindWorkflow(
        store=ResearchMindStore(root / "researchmind.sqlite3"),
        config=Settings.from_env({"OPENAI_API_KEY": ""}),
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )
    domain = workflow.ensure_ready()
    return workflow, domain


def test_workbench_brief_rejects_empty_task() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        build_workbench_brief("   ")


def test_workbench_brief_builds_plan_and_search_strategy() -> None:
    brief = build_workbench_brief(
        TOPIC,
        population="普通高校在校大学生",
        method="问卷调查与体质测试",
    )

    assert brief.task == TOPIC
    assert brief.plan.steps
    assert "AND" in brief.strategy.chinese_query
    assert brief.strategy.databases
    assert brief.design_template
    assert "普通高校在校大学生" in brief.research_question
    assert brief.evidence_query


def test_workbench_evidence_has_real_file_and_page_source(
    workbench_workflow,
) -> None:
    workflow, domain = workbench_workflow
    papers = workflow.store.list_papers(domain.domain_id)
    paper_file_names = {Path(paper.pdf_path).name for paper in papers}

    hits = search_domain_evidence(
        workflow.store.list_domain_evidence(domain.domain_id),
        "大学生 体育锻炼 体质健康",
        top_k=5,
    )

    assert hits
    assert any("体育锻炼" in hit.quote for hit in hits)
    assert all(hit.document_id for hit in hits)
    assert all(hit.document_title for hit in hits)
    assert all(hit.file_name in paper_file_names for hit in hits)
    assert all(hit.page_number > 0 for hit in hits)
    assert all(hit.quote.strip() for hit in hits)
    assert all("第 " in hit.citation and "页" in hit.citation for hit in hits)


def test_workbench_does_not_mark_missing_evidence_as_completed() -> None:
    brief = build_workbench_brief(TOPIC)
    stages = build_workbench_stages(
        brief,
        evidence_hits=(),
        evidence_searched=True,
        data_summary=None,
    )
    stages_by_title = {stage.title: stage for stage in stages}

    assert stages_by_title["03 真实文献与证据"].status == "证据不足"
    assert stages_by_title["05 真实数据分析"].status == "待操作"
    assert stages_by_title["06 科研报告"].status == "待撰写"
    assert all("已完成" not in stage.status for stage in stages)


def test_workbench_export_is_a_draft_not_a_generated_review() -> None:
    brief = build_workbench_brief(TOPIC)
    export_data = build_workbench_export(
        brief,
        evidence_hits=(),
        evidence_query=TOPIC,
        evidence_searched=True,
        data_summary=None,
    )

    assert export_data["schema_version"] == 1
    assert "不是学术论文或综述" in export_data["notice"]
    assert export_data["evidence_search"]["hits"] == []
    assert export_data["data_analysis"] is None
    assert export_data["brief"]["plan"]["steps"]


def test_home_module_cards_only_target_existing_pages() -> None:
    from app import MODULE_CARDS, PAGES

    assert "科研工作台" in PAGES
    assert all(card[3] in PAGES for card in MODULE_CARDS)


def test_home_card_opens_workbench() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=60).run()

    test_app.button(key="module_card_0").click().run()

    assert not test_app.exception
    assert test_app.session_state["navigation"] == "科研工作台"


def test_workbench_page_accepts_home_prefill_and_searches_safely() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=60).run()
    prefill = "青少年体育锻炼干预与体质健康水平的关系研究"
    test_app.session_state["workbench_prefill"] = prefill
    test_app.session_state["navigation"] = "科研工作台"
    test_app.run()

    assert not test_app.exception
    assert test_app.text_area(key="workbench_task_input").value == prefill
    test_app.form_submit_button[0].click().run()

    assert not test_app.exception
    test_app.button(key="workbench_search_evidence").click().run()

    assert not test_app.exception
    assert test_app.session_state["workbench_evidence_searched"] is True
    hits = test_app.session_state["workbench_evidence_hits"]
    assert isinstance(hits, tuple)
    if hits:
        assert all(hit.quote.strip() for hit in hits)
        assert all(hit.citation.strip() for hit in hits)
    else:
        assert any(
            "证据不足，暂不能判断" in warning.value
            for warning in test_app.warning
        )
