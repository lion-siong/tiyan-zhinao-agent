"""ResearchMind 真实论文库、Agent 工作流与学习闭环测试。"""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config import Settings
from researchmind.collectors import (
    LiteratureCollectionError,
    LiteratureCollector,
    _search_crossref,
)
from researchmind.agents import (
    AssessmentGenerationError,
    FIELD_LABELS,
    PersonalTutorAgent,
)
from researchmind.models import (
    ASSESSMENT_MODULES,
    QUESTION_TYPES,
    AssessmentQuestion,
    CandidatePaper,
    EvidenceRef,
    PaperAnalysis,
    PaperRecord,
    ResearchDomain,
    RubricCriterion,
)
from researchmind.storage import BASELINE_MASTERY, ResearchMindStore
from researchmind.ui import (
    RESEARCHMIND_PAGES,
    _display_question_options,
    _download_and_ingest_candidate,
    _format_question_answer,
    _question_option_labels,
    _render_learning_assessment,
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


def test_paper_deletion_invalidates_only_affected_learning_data(
    tmp_path: Path,
) -> None:
    """删除论文后，相关旧题失效，无关题目与答题记录继续保留。"""

    store = ResearchMindStore(tmp_path / "researchmind.sqlite3")
    domain = ResearchDomain.create(
        name="论文删除一致性测试",
        description="验证删除论文后派生学习数据的清理范围。",
        keywords=("测试",),
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.upsert_domain(domain)

    deleted_paper = PaperRecord(
        paper_id="deleted-paper",
        domain_id=domain.domain_id,
        title="待删除论文",
        authors=("测试作者",),
        year="2025",
        journal="测试期刊",
        doi="10.0000/deleted",
        source_url="",
        abstract="",
        keywords=("测试",),
        pdf_path=str(tmp_path / "deleted.pdf"),
        source="测试",
        verification_status="测试",
        created_at="2026-01-01T00:00:00+00:00",
    )
    kept_paper = replace(
        deleted_paper,
        paper_id="kept-paper",
        title="保留论文",
        doi="10.0000/kept",
    )
    store.upsert_paper(deleted_paper)
    store.upsert_paper(kept_paper)

    def question(
        question_id: str,
        paper_id: str,
    ) -> AssessmentQuestion:
        return AssessmentQuestion(
            question_id=question_id,
            domain_id=domain.domain_id,
            question_type="单项选择题",
            prompt=f"关于 {paper_id} 的测试题",
            options=("中文选项甲", "中文选项乙"),
            standard_answer="A",
            rubric=(
                RubricCriterion(
                    label="答案正确",
                    description="选择原文支持的正确选项",
                    points=10,
                ),
            ),
            knowledge_tags=("测试知识点",),
            difficulty="基础",
            source_paper_ids=(paper_id,),
            evidence=(
                EvidenceRef(
                    chunk_id=f"{question_id}-evidence",
                    document_id=paper_id,
                    document_title=paper_id,
                    file_name=f"{paper_id}.pdf",
                    page_number=1,
                    quote="用于验证论文删除后的派生数据清理。",
                ),
            ),
            created_at="2026-01-01T00:00:00+00:00",
        )

    affected = question("affected-question", deleted_paper.paper_id)
    unaffected = question("unaffected-question", kept_paper.paper_id)
    store.save_question(affected)
    store.save_question(unaffected)
    store.record_attempt(
        domain_id=domain.domain_id,
        question=affected,
        user_answer="A",
        score=10,
        feedback="正确",
        mode="客观题自动评分",
    )
    store.record_attempt(
        domain_id=domain.domain_id,
        question=unaffected,
        user_answer="A",
        score=10,
        feedback="正确",
        mode="客观题自动评分",
    )

    cleanup = store.invalidate_domain_derivations(
        domain.domain_id,
        (deleted_paper.paper_id,),
    )

    remaining_questions = store.list_questions(domain.domain_id)
    remaining_attempts = store.list_attempts(domain.domain_id)
    assert cleanup["questions"] == 1
    assert cleanup["attempts"] == 1
    assert [item.question_id for item in remaining_questions] == [
        unaffected.question_id
    ]
    assert [item.question_id for item in remaining_attempts] == [
        unaffected.question_id
    ]


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

    strength_text = pipeline.debate.evidence_strength
    assert strength_text.startswith(("自动初筛：", "待评估："))
    assert "非正式 GRADE 评价" in strength_text
    assert (
        "证据不足" in pipeline.debate.consensus_summary
        or "系统综述核验" in pipeline.debate.consensus_summary
    )
    assert pipeline.synthesis.consensus
    assert pipeline.synthesis.research_gaps


def test_all_question_types_include_sources_rubrics_and_tags(
    pipeline_bundle,
) -> None:
    workflow, _, pipeline = pipeline_bundle
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
        assert question.explanation.strip()

    persisted = workflow.store.get_question(
        pipeline.questions[0].question_id
    )
    assert persisted is not None
    assert persisted.explanation.strip()

    for question_type in ("论述题", "文献对比分析题"):
        standard_answer = questions_by_type[question_type].standard_answer
        assert "比较框架" in standard_answer
        assert "跨论文比较与异同" in standard_answer
        assert "方法异同" in standard_answer
        assert "统计分析比较" in standard_answer
        assert "相关与因果" in standard_answer
        assert "证据不足" in standard_answer
        assert "学术共识" in standard_answer


def test_assessment_field_labels_cover_new_analysis_schema() -> None:
    analysis = PaperAnalysis(paper_id="schema-probe")
    assert {
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
    }.issubset(FIELD_LABELS)
    assert analysis.interventions == ()


def test_assessment_generates_all_five_modules_from_real_evidence(
    pipeline_bundle,
) -> None:
    workflow, domain, _ = pipeline_bundle

    questions = workflow.assessment.generate_question_set(
        domain.domain_id,
        assessment_modules=ASSESSMENT_MODULES,
        debate_topic=DEFAULT_DEBATE_TOPIC,
    )
    by_module = {
        module: tuple(
            question
            for question in questions
            if question.assessment_module == module
        )
        for module in ASSESSMENT_MODULES
    }

    assert all(by_module.values())
    assert {
        question.question_type for question in by_module["基础知识理解"]
    } == {"单项选择题", "多项选择题", "判断题", "简答题"}
    assert {
        question.question_type for question in by_module["文献深度理解"]
    } == {"深度理解题"}
    assert {
        question.question_type for question in by_module["文献对比分析"]
    } == {"文献对比分析题"}
    assert {
        question.question_type for question in by_module["学术思辨训练"]
    } == {"学术思辨题"}
    assert {
        question.question_type
        for question in by_module["个性化错题与补学"]
    } == {"个性化补学题"}

    comparison = by_module["文献对比分析"][0]
    assert len(set(comparison.source_paper_ids)) >= 2
    assert len(
        {item.document_id for item in comparison.evidence}
    ) >= 2
    assert [item.label for item in comparison.rubric] == [
        "核心观点准确性",
        "文献证据引用能力",
        "多篇论文对比能力",
        "学术逻辑与推理能力",
        "批判性思维能力",
    ]
    assert all(
        server_question.assessment_module == question.assessment_module
        for question in questions
        if (server_question := workflow.store.get_question(
            question.question_id
        )) is not None
    )


def test_subjective_scoring_uses_evidence_not_answer_length(
    pipeline_bundle,
) -> None:
    workflow, domain, _ = pipeline_bundle
    comparison = workflow.assessment.generate_question_set(
        domain.domain_id,
        assessment_modules=("文献对比分析",),
        debate_topic=DEFAULT_DEBATE_TOPIC,
    )[0]

    evidence_based = workflow.assessment.score(
        comparison,
        comparison.standard_answer,
    )
    long_unsupported = workflow.assessment.score(
        comparison,
        "研究有意义并且结论很重要。" * 300,
    )

    assert evidence_based.score > long_unsupported.score
    assert long_unsupported.score <= 2.0
    critical = next(
        item
        for item in evidence_based.criteria
        if item.label == "批判性思维能力"
    )
    assert critical.score == critical.max_score
    assert "篇幅" in evidence_based.feedback
    assert all(
        item.label
        for item in evidence_based.criteria
    )


def test_personalized_review_uses_recent_wrong_answer_tags(
    pipeline_bundle,
) -> None:
    workflow, domain, pipeline = pipeline_bundle
    objective = next(
        question
        for question in pipeline.questions
        if question.is_objective
    )
    workflow.tutor.grade_and_record(objective, "未选择任何正确选项")
    wrong_tags = {
        tag
        for attempt in workflow.store.list_wrong_attempts(domain.domain_id)
        for tag in attempt.knowledge_tags
    }

    personalized = workflow.assessment.generate_question_set(
        domain.domain_id,
        assessment_modules=("个性化错题与补学",),
        debate_topic=DEFAULT_DEBATE_TOPIC,
    )[0]

    assert personalized.question_type == "个性化补学题"
    assert "你的近期薄弱点为" in personalized.prompt
    assert sorted(
        wrong_tags.intersection(personalized.knowledge_tags)
    ) == sorted(wrong_tags)


def test_learning_assessment_page_renders_and_switches_modules(
    pipeline_bundle,
) -> None:
    workflow, domain, _ = pipeline_bundle

    def script(renderer, test_workflow, test_domain) -> None:
        renderer(test_workflow, test_domain)

    app = AppTest.from_function(
        script,
        default_timeout=30,
        args=(
            _render_learning_assessment,
            workflow,
            domain,
        ),
    ).run()

    assert not app.exception
    assert app.multiselect[0].value == list(ASSESSMENT_MODULES)
    app.form_submit_button[0].click().run(timeout=30)

    assert not app.exception
    assert any(
        "已生成" in item.value and "道有来源的试题" in item.value
        for item in app.success
    )
    module_control = app.segmented_control(
        key=f"researchmind-assessment-module-{domain.domain_id}"
    )
    assert tuple(module_control.options) == ASSESSMENT_MODULES
    module_control.select("文献对比分析").run(timeout=10)

    assert not app.exception
    assert any(
        "比较《" in item.value
        for item in app.markdown
    )


def test_empty_library_assessment_returns_actionable_error(
    tmp_path: Path,
    no_key_config: Settings,
) -> None:
    workflow = ResearchMindWorkflow(
        store=ResearchMindStore(tmp_path / "empty.sqlite3"),
        config=no_key_config,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
    )
    domain = workflow.ensure_default_domain()

    with pytest.raises(AssessmentGenerationError) as error:
        workflow.assessment.generate_question_set(domain.domain_id)

    assert error.value.code == "empty_library"
    assert "没有活动论文" in error.value.user_message
    assert "导入论文" in error.value.suggestion


def test_model_scoring_failure_is_explicitly_reported(
    pipeline_bundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _, pipeline = pipeline_bundle
    question = next(
        item
        for item in pipeline.questions
        if item.question_type == "简答题"
    )

    class _ConfiguredFailureLLM:
        is_configured = True

    monkeypatch.setattr(
        workflow.assessment,
        "llm_service",
        _ConfiguredFailureLLM(),
    )

    def fail_scoring(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("模型请求超时")

    monkeypatch.setattr(
        workflow.assessment,
        "_score_with_llm",
        fail_scoring,
    )
    result = workflow.assessment.score(question, "仅依据原文作答。")

    assert "模型调用失败" in result.mode
    assert "TimeoutError" in result.feedback
    assert "模型请求超时" in result.feedback


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


def test_practice_performance_is_unmeasured_until_a_real_attempt(
    tmp_path: Path,
) -> None:
    store = ResearchMindStore(tmp_path / "practice-performance.sqlite3")
    domain = ResearchDomain.create(
        "练习表现测试",
        "自动化测试夹具。",
        ("青少年", "体育锻炼"),
        "2026-11-16T00:00:00+00:00",
    )
    store.upsert_domain(domain)
    evidence = EvidenceRef(
        chunk_id="practice-evidence",
        document_id="practice-paper",
        document_title="Automated Test Fixture",
        file_name="fixture.pdf",
        page_number=1,
        quote="Automated test fixture evidence for practice scoring.",
        source_type="全文",
        section_hint="result",
    )
    question = AssessmentQuestion(
        question_id="practice-question",
        domain_id=domain.domain_id,
        question_type="个性化补学题",
        prompt="自动化测试练习题。",
        options=(),
        standard_answer="测试答案",
        rubric=(
            RubricCriterion(
                label="测试维度",
                description="自动化测试评分。",
                points=10.0,
            ),
        ),
        knowledge_tags=("练习表现",),
        difficulty="基础",
        source_paper_ids=("practice-paper",),
        evidence=(evidence,),
        created_at="2026-11-16T00:00:00+00:00",
    )
    store.save_question(question)
    tutor = PersonalTutorAgent(store)

    initial = store.ensure_mastery(domain.domain_id)
    initial_recommendation = tutor.recommend(domain.domain_id)
    assert all(item.attempts_count == 0 for item in initial)
    assert all(item.score == 0.0 for item in initial)
    assert initial_recommendation.weak_dimensions == ()
    assert "未测" in initial_recommendation.summary

    store.record_attempt(
        domain_id=domain.domain_id,
        question=question,
        user_answer="测试答案",
        score=10.0,
        feedback="自动化测试满分记录。",
        mode="自动化测试",
    )
    tutor.update_mastery(domain.domain_id)
    measured = store.ensure_mastery(domain.domain_id)

    assert all(item.attempts_count == 1 for item in measured)
    assert all(item.score == 100.0 for item in measured)


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
