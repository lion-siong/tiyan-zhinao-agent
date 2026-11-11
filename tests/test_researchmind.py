"""ResearchMind 真实论文库、Agent 工作流与学习闭环测试。"""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings
from researchmind.collectors import (
    LiteratureCollectionError,
    LiteratureCollector,
    _search_crossref,
)
from researchmind.models import (
    QUESTION_TYPES,
    CandidatePaper,
    EvidenceRef,
    PaperRecord,
)
from researchmind.storage import BASELINE_MASTERY, ResearchMindStore
from researchmind.ui import (
    RESEARCHMIND_PAGES,
    _display_question_options,
    _download_and_ingest_candidate,
    _format_question_answer,
    _question_option_labels,
)
from researchmind.workflow import (
    DEFAULT_DEBATE_TOPIC,
    ResearchMindWorkflow,
)
from researchmind.zotero_service import ZoteroError, ZoteroService
from tools.knowledge_base import load_bundled_knowledge_base
from tools.library_ingest import IngestResult, sync_pdf_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


@pytest.fixture(scope="module")
def no_key_config() -> Settings:
    """测试环境不读取真实密钥，主观题固定走本地评分细则。"""

    return Settings.from_env(
        {
            "OPENAI_API_KEY": "",
            "RESEARCHMIND_DATE_FROM": "2023-01-01",
        }
    )


@pytest.fixture(scope="module")
def pipeline_bundle(tmp_path_factory, no_key_config):
    """在临时数据库中执行一次真实论文库完整分析。"""

    root = tmp_path_factory.mktemp("researchmind-pipeline")
    store = ResearchMindStore(root / "researchmind.sqlite3")
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )
    domain = workflow.ensure_ready()
    pipeline = workflow.run_analysis_pipeline(
        domain.domain_id,
        topic=DEFAULT_DEBATE_TOPIC,
    )
    return workflow, domain, pipeline


def test_sqlite_persistence_and_real_library_sync(
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    database_path = tmp_path / "researchmind.sqlite3"
    store = ResearchMindStore(database_path)
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )

    domain = workflow.ensure_ready()
    sync_result = workflow.sync_bundled_library(domain=domain)
    bundle = load_bundled_knowledge_base(KNOWLEDGE_BASE_DIR)
    evidence = store.list_domain_evidence(domain.domain_id)

    assert database_path.is_file()
    assert len(bundle.documents) == 12
    assert sync_result.document_count == len(bundle.documents)
    assert store.paper_count(domain.domain_id) == len(bundle.documents)
    assert store.evidence_count(domain.domain_id) == len(bundle.chunks)
    assert len(evidence) >= 100
    assert all(item.page_number > 0 for item in evidence)
    assert all(item.quote.strip() for item in evidence)
    assert all(Path(paper.pdf_path).is_file() for paper in store.list_papers(domain.domain_id))

    reopened = ResearchMindStore(database_path)
    assert reopened.get_domain(domain.domain_id) == domain
    assert reopened.paper_count(domain.domain_id) == store.paper_count(domain.domain_id)
    assert reopened.evidence_count(domain.domain_id) == store.evidence_count(
        domain.domain_id
    )


def test_custom_domain_ids_do_not_conflict(
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    store = ResearchMindStore(tmp_path / "researchmind.sqlite3")
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )
    default_domain = workflow.ensure_default_domain()
    custom_domain = workflow.create_domain(
        name="大学生运动技能学习",
        description="测试多个领域共享同一批真实 PDF 时的稳定标识。",
        keywords=("运动技能", "大学生", "学习"),
    )

    workflow.sync_bundled_library(domain=default_domain)
    workflow.sync_bundled_library(domain=custom_domain)

    default_paper_ids = {
        paper.paper_id for paper in store.list_papers(default_domain.domain_id)
    }
    custom_paper_ids = {
        paper.paper_id for paper in store.list_papers(custom_domain.domain_id)
    }
    default_chunk_ids = {
        item.chunk_id
        for item in store.list_domain_evidence(default_domain.domain_id)
    }
    custom_chunk_ids = {
        item.chunk_id
        for item in store.list_domain_evidence(custom_domain.domain_id)
    }

    assert default_paper_ids.isdisjoint(custom_paper_ids)
    assert default_chunk_ids.isdisjoint(custom_chunk_ids)
    assert all(item.startswith(f"{custom_domain.domain_id}-paper-") for item in custom_paper_ids)
    assert all(item.startswith(f"{custom_domain.domain_id}-chunk-") for item in custom_chunk_ids)


def test_new_pdf_directory_entry_syncs_into_researchmind(
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    """复制进 PDF 目录的真实论文可以被扫描并同步到 ResearchMind。"""

    knowledge_dir = tmp_path / "knowledge_base"
    pdf_dir = knowledge_dir / "pdfs"
    pdf_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "library_name": "ResearchMind 自动同步测试库",
        "description": "只包含一篇真实 PDF，用于验证目录扫描与 SQLite 同步。",
        "documents": [],
    }
    (knowledge_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_pdf = KNOWLEDGE_BASE_DIR / "pdfs" / "ass_2401349.pdf"
    (pdf_dir / source_pdf.name).write_bytes(source_pdf.read_bytes())

    ingest_result = sync_pdf_directory(
        knowledge_dir,
        auto_vector_sync=False,
    )
    assert ingest_result.imported_count == 1

    store = ResearchMindStore(tmp_path / "researchmind.sqlite3")
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=knowledge_dir,
    )
    domain = workflow.ensure_default_domain()
    stale_paper = PaperRecord(
        paper_id="stale-paper",
        domain_id=domain.domain_id,
        title="已从论文清单移除的论文",
        authors=("测试作者",),
        year="2025",
        journal="测试期刊",
        doi="10.0000/stale",
        source_url="https://example.com/stale",
        abstract="",
        keywords=("过期数据",),
        pdf_path=str(pdf_dir / "stale.pdf"),
        source="同步回归测试",
        verification_status="仅用于测试",
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.upsert_paper(stale_paper)
    store.replace_evidence(
        stale_paper.paper_id,
        (
            EvidenceRef(
                chunk_id="stale-chunk",
                document_id=stale_paper.paper_id,
                document_title=stale_paper.title,
                file_name="stale.pdf",
                page_number=1,
                quote="这条证据应随过期论文一起删除。",
            ),
        ),
    )
    sync_result = workflow.sync_bundled_library(domain=domain, force=True)

    papers = store.list_papers(domain.domain_id)
    assert sync_result.papers_removed == 1
    assert sync_result.total_papers == 1
    assert len(papers) == 1
    assert papers[0].doi == "10.12677/ass.2026.157601"
    assert store.get_paper(stale_paper.paper_id) is None
    assert store.evidence_count(domain.domain_id) > 0


def test_crossref_pdf_link_is_metadata_only(
    monkeypatch: pytest.MonkeyPatch,
    no_key_config: Settings,
) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/example",
                    "title": ["A Crossref article"],
                    "author": [{"given": "Test", "family": "Author"}],
                    "published": {"date-parts": [[2024]]},
                    "container-title": ["Example Journal"],
                    "URL": "https://doi.org/10.1234/example",
                    "link": [
                        {
                            "content-type": "application/pdf",
                            "URL": "https://example.com/article.pdf",
                        }
                    ],
                }
            ]
        }
    }
    monkeypatch.setattr(
        "researchmind.collectors._read_json",
        lambda *args, **kwargs: payload,
    )

    candidates = _search_crossref(
        "example",
        limit=1,
        date_from="2023-01-01",
        timeout=10,
    )

    assert len(candidates) == 1
    assert candidates[0].pdf_url == "https://example.com/article.pdf"
    assert candidates[0].is_open_access is False
    assert candidates[0].has_open_pdf is False
    with pytest.raises(LiteratureCollectionError, match="没有可确认"):
        LiteratureCollector(no_key_config).download_open_pdf(candidates[0])


def test_candidate_download_forwards_data_to_ingest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    import researchmind.ui as researchmind_ui

    store = ResearchMindStore(tmp_path / "researchmind.sqlite3")
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=tmp_path / "knowledge_base",
    )
    domain = workflow.ensure_default_domain()
    candidate = CandidatePaper(
        candidate_id="candidate-1",
        title="开放获取候选论文",
        authors=("测试作者",),
        year="2025",
        journal="测试期刊",
        doi="10.1234/open",
        source_url="https://doi.org/10.1234/open",
        abstract="",
        keywords=("开放获取",),
        source="OpenAlex",
        pdf_url="https://example.com/open.pdf",
        is_open_access=True,
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        researchmind_ui.LiteratureCollector,
        "download_open_pdf",
        lambda self, paper: b"%PDF-1.7 test",
    )
    monkeypatch.setattr(
        researchmind_ui.st,
        "spinner",
        lambda *args, **kwargs: nullcontext(),
    )

    def fake_ingest(files, base_dir):
        captured["files"] = files
        captured["base_dir"] = base_dir
        return IngestResult(
            imported=(),
            duplicates=(),
            skipped=(),
            warnings=(),
            framework=None,
        )

    monkeypatch.setattr(researchmind_ui, "ingest_pdf_files", fake_ingest)

    _download_and_ingest_candidate(workflow, domain, candidate)

    files = captured["files"]
    assert len(files) == 1
    assert files[0].file_name.endswith(".pdf")
    assert files[0].data == b"%PDF-1.7 test"


def test_six_agents_produce_traceable_analysis(pipeline_bundle) -> None:
    workflow, domain, pipeline = pipeline_bundle

    assert len(pipeline.analyses) == 12
    assert len(pipeline.debate.viewpoints) > 0
    assert len(pipeline.synthesis.nodes) > 0
    assert len(pipeline.synthesis.source_paper_ids) == 12
    assert workflow.store.paper_count(domain.domain_id) == 12

    for viewpoint in pipeline.debate.viewpoints:
        assert viewpoint.supporting_papers
        assert viewpoint.evidence
        assert viewpoint.page_numbers
        assert all(item.page_number > 0 for item in viewpoint.evidence)
        assert all(item.document_id in viewpoint.supporting_papers for item in viewpoint.evidence)

    assert pipeline.debate.evidence_strength.startswith("待评估")
    assert (
        "证据不足" in pipeline.debate.consensus_summary
        or "系统综述核验" in pipeline.debate.consensus_summary
    )
    assert pipeline.synthesis.consensus
    assert pipeline.synthesis.research_gaps


def test_all_question_types_include_sources_rubrics_and_tags(
    pipeline_bundle,
) -> None:
    _, _, pipeline = pipeline_bundle
    questions_by_type = {
        question.question_type: question for question in pipeline.questions
    }

    assert set(questions_by_type) == set(QUESTION_TYPES)
    for question in pipeline.questions:
        assert question.source_paper_ids
        assert question.evidence
        assert all(item.page_number > 0 for item in question.evidence)
        assert question.rubric
        assert question.max_score > 0
        assert question.knowledge_tags
        assert question.difficulty
        assert question.standard_answer.strip()

    for question_type in ("论述题", "文献对比分析题"):
        standard_answer = questions_by_type[question_type].standard_answer
        assert "比较框架" in standard_answer
        assert "跨论文比较与异同" in standard_answer
        assert "方法异同" in standard_answer
        assert "统计分析比较" in standard_answer
        assert "相关与因果" in standard_answer
        assert "证据不足" in standard_answer
        assert "学术共识" in standard_answer


def test_choice_options_are_chinese_without_urls_or_misalignment(
    pipeline_bundle,
) -> None:
    _, _, pipeline = pipeline_bundle
    questions_by_type = {
        question.question_type: question for question in pipeline.questions
    }

    for question_type in ("单项选择题", "多项选择题"):
        question = questions_by_type[question_type]
        options = _display_question_options(question)
        labels = _question_option_labels(question)

        assert len(options) == len(question.options)
        assert len(labels) == len(options)
        assert len(set(labels)) == len(labels)
        for option in options:
            assert option.strip()
            assert re.search(r"[\u4e00-\u9fff]", option)
            assert not re.search(r"[A-Za-z]", option)
            assert "http://" not in option.lower()
            assert "https://" not in option.lower()
            assert "www." not in option.lower()
            assert "doi" not in option.lower()

    legacy = replace(
        questions_by_type["单项选择题"],
        options=(
            "https://doi.org/10.1234/example English evidence",
            "本题指定论文第 2 页：含研究方法线索",
        ),
    )
    legacy_options = _display_question_options(legacy)
    assert all(re.search(r"[\u4e00-\u9fff]", item) for item in legacy_options)
    assert all(not re.search(r"[A-Za-z]", item) for item in legacy_options)


def test_objective_and_subjective_scoring_use_explicit_rules(
    pipeline_bundle,
    no_key_config: Settings,
) -> None:
    workflow, _, pipeline = pipeline_bundle
    questions_by_type = {
        question.question_type: question for question in pipeline.questions
    }

    for question_type in ("单项选择题", "多项选择题", "判断题"):
        question = questions_by_type[question_type]
        correct = workflow.assessment.score(question, question.standard_answer)
        wrong = workflow.assessment.score(question, "错误答案")
        assert correct.correct is True
        assert correct.score == question.max_score
        assert correct.user_answer == question.standard_answer
        assert wrong.correct is False
        assert wrong.score == 0.0
        assert wrong.user_answer == "错误答案"

    for question_type in ("单项选择题", "多项选择题"):
        question = questions_by_type[question_type]
        standard_text = _format_question_answer(
            question,
            question.standard_answer,
        )
        assert standard_text != "（未作答）"
        assert re.search(r"[甲乙丙丁]", standard_text)
        assert not re.search(r"[A-H]", standard_text)

    assert no_key_config.llm_configured is False
    short_answer = questions_by_type["简答题"]
    subjective = workflow.assessment.score(
        short_answer,
        short_answer.standard_answer,
    )
    assert subjective.mode == "本地评分细则（非大模型）"
    assert subjective.criteria
    assert all(item.reason for item in subjective.criteria)
    assert "语言流畅" in subjective.feedback


def test_learning_cycle_updates_mastery_wrong_book_and_next_round(
    pipeline_bundle,
) -> None:
    workflow, domain, pipeline = pipeline_bundle
    before_wrong = len(workflow.store.list_wrong_attempts(domain.domain_id))
    answers = {question.question_id: "" for question in pipeline.questions}

    result = workflow.grade_learning_cycle(
        domain.domain_id,
        answers=answers,
        generate_next=True,
    )

    assert len(result.scores) == len(pipeline.questions)
    assert set(result.mastery) == set(BASELINE_MASTERY)
    assert result.next_questions
    assert result.recommendation.weak_dimensions
    assert result.recommendation.recommended_paper_ids
    assert result.recommendation.review_plan
    assert len(workflow.store.list_wrong_attempts(domain.domain_id)) > before_wrong
    assert any(item.score == 0.0 for item in workflow.store.list_attempts(domain.domain_id))


def test_acceptance_workflow_runs_from_real_theme(
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    store = ResearchMindStore(tmp_path / "acceptance.sqlite3")
    workflow = ResearchMindWorkflow(
        store=store,
        config=no_key_config,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )

    pipeline, learning = workflow.run_end_to_end_acceptance(
        topic=DEFAULT_DEBATE_TOPIC,
    )

    assert len(pipeline.analyses) == 12
    assert set(question.question_type for question in pipeline.questions) == set(
        QUESTION_TYPES
    )
    assert learning.scores
    assert learning.mastery
    assert learning.recommendation.review_plan
    assert learning.next_questions


def test_zotero_missing_configuration_has_clear_error(
    pipeline_bundle,
) -> None:
    workflow, domain, _ = pipeline_bundle
    service = ZoteroService(workflow.config)
    paper = workflow.store.list_papers(domain.domain_id)[0]

    assert service.status().configured is False
    with pytest.raises(ZoteroError, match="ZOTERO_API_KEY"):
        service.add_paper(paper)
    with pytest.raises(ZoteroError, match="ZOTERO_API_KEY"):
        service.add_paper_with_pdf(paper)


def test_literature_collector_rejects_empty_query_without_network() -> None:
    collector = LiteratureCollector(Settings.from_env({"OPENAI_API_KEY": ""}))

    with pytest.raises(LiteratureCollectionError, match="不能为空"):
        collector.search("   ")


def test_all_researchmind_pages_load_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=60).run()

    for page in RESEARCHMIND_PAGES:
        test_app.session_state["navigation"] = page
        test_app.run()
        assert not test_app.exception, f"页面运行失败：{page}"
