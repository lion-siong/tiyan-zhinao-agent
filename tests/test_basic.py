"""第一阶段核心功能测试。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from agents.planner import ResearchPlannerAgent
from config import Settings
from services.llm_service import LLMService
from services.rag_service import (
    EvidenceRAGService,
    HybridEvidenceIndex,
    LocalEvidenceIndex,
)
from services.vector_service import (
    EvidenceVectorStore,
    VectorSearchResult,
    VectorStoreError,
)
from tools.data_analysis import (
    DataAnalysisError,
    analyze_dataframe,
    create_numeric_histogram,
    load_dataframe,
)
from tools.document_chunker import EvidenceChunk, chunk_pdf_pages
from tools.knowledge_base import load_bundled_knowledge_base
from tools.library_ingest import (
    IncomingPDF,
    ingest_pdf_files,
    remove_library_documents,
    sync_pdf_directory,
)
from tools.literature_framework import load_literature_framework
from tools.literature_search import build_search_strategy
from tools.pdf_metadata import extract_pdf_metadata
from tools.pdf_reader import PDFPageText, extract_pdf_text


def test_settings_reads_key_without_exposing_it() -> None:
    config = Settings.from_env(
        {
            "OPENAI_API_KEY": "test-secret-key",
            "OPENAI_BASE_URL": "https://example.com/v1",
            "OPENAI_MODEL": "test-model",
        }
    )
    assert config.llm_configured is True
    assert config.openai_model == "test-model"
    assert "test-secret-key" not in repr(config)


def test_settings_reads_local_vector_configuration() -> None:
    config = Settings.from_env(
        {
            "EMBEDDING_MODEL": "local-test-model",
            "CHROMA_COLLECTION_NAME": "test_chunks",
            "RAG_RETRIEVAL_MODE": "semantic",
        }
    )

    assert config.embedding_model == "local-test-model"
    assert config.chroma_collection_name == "test_chunks"
    assert config.rag_retrieval_mode == "semantic"


def test_settings_reads_researchmind_database_override(tmp_path: Path) -> None:
    database_path = tmp_path / "public-demo.sqlite3"
    config = Settings.from_env(
        {"RESEARCHMIND_DB_PATH": str(database_path)}
    )

    assert config.researchmind_db_path == database_path.resolve()


def test_planner_creates_steps_and_detects_data_task() -> None:
    agent = ResearchPlannerAgent()
    plan = agent.create_plan("分析学生问卷数据和体质测试成绩")
    assert plan.task_type == "数据分析型"
    assert len(plan.steps) >= 5
    assert plan.steps[0].order == 1
    assert "数据分析" in " ".join(step.tool_name for step in plan.steps)


def test_planner_rejects_empty_task() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        ResearchPlannerAgent().create_plan("   ")


def test_topic_templates_have_required_sections() -> None:
    suggestions = ResearchPlannerAgent().generate_topic_suggestions(
        "体育课堂即时反馈", count=3
    )
    assert len(suggestions) == 3
    assert all(suggestion.title for suggestion in suggestions)
    assert all(suggestion.research_question for suggestion in suggestions)
    assert all("需" in suggestion.research_subject for suggestion in suggestions)


def test_literature_strategy_does_not_claim_to_return_papers() -> None:
    strategy = build_search_strategy("体育教学 学习投入", "大学生", "问卷调查")
    assert "AND" in strategy.chinese_query
    assert any("CNKI" in database for database in strategy.databases)
    assert any("不返回真实文献" in note for note in strategy.notes)


def test_csv_loading_and_numeric_summary() -> None:
    csv_bytes = "姓名,成绩,训练时长\n张三,80,10\n李四,90,\n王五,100,20\n".encode(
        "utf-8-sig"
    )
    dataframe = load_dataframe(csv_bytes, "sample.csv")
    result = analyze_dataframe(dataframe)

    assert result.rows == 3
    assert result.columns == 3
    assert result.numeric_summary.loc[
        result.numeric_summary["字段"] == "成绩", "均值"
    ].iloc[0] == pytest.approx(90.0)
    assert result.missing_summary.loc[
        result.missing_summary["字段"] == "训练时长", "缺失数量"
    ].iloc[0] == 1


def test_xlsx_loading() -> None:
    buffer = io.BytesIO()
    pd.DataFrame({"成绩": [80, 90], "组别": ["A", "B"]}).to_excel(
        buffer, index=False, engine="openpyxl"
    )
    dataframe = load_dataframe(buffer.getvalue(), "sample.xlsx")
    assert dataframe.shape == (2, 2)
    assert dataframe["成绩"].mean() == pytest.approx(85.0)


def test_data_loading_rejects_unsupported_extension() -> None:
    with pytest.raises(DataAnalysisError, match="仅支持"):
        load_dataframe(b"anything", "sample.txt")


def test_histogram_requires_numeric_column() -> None:
    dataframe = pd.DataFrame({"姓名": ["甲", "乙"]})
    with pytest.raises(DataAnalysisError, match="不是数值列"):
        create_numeric_histogram(dataframe, "姓名")


def test_histogram_returns_figure_for_numeric_column() -> None:
    dataframe = pd.DataFrame({"成绩": [80, 85, 90, 95]})
    figure = create_numeric_histogram(dataframe, "成绩", bins=4)
    assert figure.axes
    assert figure.axes[0].get_ylabel() == "频数"


def test_pdf_reader_handles_empty_and_corrupt_files() -> None:
    empty_result = extract_pdf_text(b"")
    corrupt_result = extract_pdf_text(b"not-a-pdf")

    assert empty_result.success is False
    assert "为空" in empty_result.message
    assert corrupt_result.success is False


def test_pdf_reader_extracts_text_from_valid_pdf() -> None:
    fitz = pytest.importorskip("pymupdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Sports research test document.")
    pdf_bytes = document.tobytes()
    document.close()

    result = extract_pdf_text(pdf_bytes)
    assert result.success is True
    assert result.page_count == 1
    assert "Sports research" in result.text
    assert result.pages[0].page_number == 1


def test_llm_service_falls_back_to_clearly_labelled_simulation() -> None:
    service = LLMService(Settings.from_env({"OPENAI_API_KEY": ""}))
    response = service.answer("如何设计体育课堂观察研究？")

    assert response.success is True
    assert response.is_simulated is True
    assert "模拟内容提示" in response.answer


def test_llm_service_reports_missing_key_when_simulation_disabled() -> None:
    service = LLMService(Settings.from_env({"OPENAI_API_KEY": ""}))
    response = service.answer("测试问题", allow_simulation=False)

    assert response.success is False
    assert "OPENAI_API_KEY" in response.message


def test_document_chunker_preserves_page_numbers() -> None:
    pages = (
        PDFPageText(
            page_number=2,
            text="研究对象为大学生运动员。采用最大摄氧量评价有氧能力。",
        ),
        PDFPageText(
            page_number=5,
            text="训练干预持续八周，并在干预结束后再次测量最大摄氧量。",
        ),
    )
    chunks = chunk_pdf_pages("测试文献.pdf", pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 2
    assert chunks[1].page_number == 5
    assert all(chunk.document_name == "测试文献.pdf" for chunk in chunks)


def test_local_evidence_index_retrieves_matching_page() -> None:
    pages = (
        PDFPageText(page_number=1, text="本文介绍体育课堂观察的基本流程。"),
        PDFPageText(
            page_number=4,
            text="研究使用最大摄氧量测试评估大学生运动员的有氧能力。",
        ),
    )
    index = LocalEvidenceIndex(chunk_pdf_pages("证据测试.pdf", pages))
    results = index.search("最大摄氧量如何评估有氧能力", top_k=2)

    assert results
    assert results[0].chunk.page_number == 4
    assert "最大摄氧量" in results[0].chunk.text


class _FixedVectorEncoder:
    """测试用确定性编码器，不加载真实模型。"""

    def encode_documents(self, texts) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        if "有氧" in text or "摄氧量" in text:
            return [1.0, 0.0, 0.0]
        if "力量" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_chroma_vector_store_persists_and_syncs_incrementally(
    tmp_path: Path,
) -> None:
    chunks = (
        EvidenceChunk("chunk-a", "paper.pdf", 1, "有氧训练和最大摄氧量。"),
        EvidenceChunk("chunk-b", "paper.pdf", 2, "力量训练方法。"),
    )
    store = EvidenceVectorStore(
        tmp_path,
        model_name="fake-e5",
        collection_name="test_chunks",
        cache_dir=tmp_path / "models",
        encoder=_FixedVectorEncoder(),
    )

    first = store.sync_chunks(chunks)
    second = store.sync_chunks(chunks)
    results = store.search("最大摄氧量", top_k=2)

    assert first.added_chunks == 2
    assert first.encoded_chunks == 2
    assert second.added_chunks == 0
    assert second.unchanged_chunks == 2
    assert second.encoded_chunks == 0
    assert results[0].chunk_id == "chunk-a"
    store.close()

    reopened = EvidenceVectorStore(
        tmp_path,
        model_name="fake-e5",
        collection_name="test_chunks",
        cache_dir=tmp_path / "models",
        encoder=_FixedVectorEncoder(),
    )
    status = reopened.status(expected_chunks=2)
    assert status.available is True
    assert status.synchronized is True
    assert status.stored_chunks == 2
    reopened.close()


def test_chroma_vector_store_reopens_with_more_than_hnsw_threshold(
    tmp_path: Path,
) -> None:
    """回归 Chroma 默认阈值触发压缩后无法重新打开的问题。"""

    chunks = tuple(
        EvidenceChunk(
            f"chunk-{index}",
            "paper.pdf",
            index + 1,
            f"有氧训练证据片段 {index}。",
        )
        for index in range(1005)
    )
    store = EvidenceVectorStore(
        tmp_path,
        model_name="fake-e5",
        collection_name="large_test_chunks",
        cache_dir=tmp_path / "models",
        encoder=_FixedVectorEncoder(),
    )

    result = store.sync_chunks(chunks)
    assert result.total_chunks == 1005
    store.close()

    reopened = EvidenceVectorStore(
        tmp_path,
        model_name="fake-e5",
        collection_name="large_test_chunks",
        cache_dir=tmp_path / "models",
        encoder=_FixedVectorEncoder(),
    )
    status = reopened.status(expected_chunks=1005)
    results = reopened.search("有氧训练", top_k=3)

    assert status.available is True
    assert status.synchronized is True
    assert status.stored_chunks == 1005
    assert results
    reopened.close()


class _CorruptHnswCollection:
    """模拟打开集合后读取 HNSW 段失败。"""

    def get(self, *, include):
        raise RuntimeError(
            "Error constructing hnsw segment reader: "
            "Error creating hnsw segment reader: "
            "Error loading hnsw index"
        )


class _MemoryCollection:
    """测试用内存集合，不依赖真实 Chroma。"""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, dict]] = {}

    def get(self, *, include):
        ids = list(self.records)
        return {
            "ids": ids,
            "documents": [self.records[item][0] for item in ids],
            "metadatas": [self.records[item][1] for item in ids],
        }

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
            self.records[chunk_id] = (document, metadata)

    def delete(self, *, ids) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.records)


class _RecoveringVectorStore(EvidenceVectorStore):
    """第一次打开返回损坏集合，重建后返回可写集合。"""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            tmp_path,
            model_name="fake-e5",
            collection_name="test_chunks",
            cache_dir=tmp_path / "models",
            encoder=_FixedVectorEncoder(),
        )
        self.corrupt_collection = _CorruptHnswCollection()
        self.healthy_collection = _MemoryCollection()
        self.open_attempts = 0

    def _get_collection(self):
        self.open_attempts += 1
        if self.open_attempts == 1:
            return self.corrupt_collection
        return self.healthy_collection


def test_corrupt_hnsw_index_is_isolated_and_rebuilt(
    tmp_path: Path,
) -> None:
    vector_dir = tmp_path / "chroma"
    vector_dir.mkdir()
    sentinel = vector_dir / "broken-index.bin"
    sentinel.write_bytes(b"damaged index")
    source_pdf = tmp_path / "paper.pdf"
    source_pdf.write_bytes(b"original source must remain")

    store = _RecoveringVectorStore(tmp_path)
    chunks = (
        EvidenceChunk("chunk-a", "paper.pdf", 1, "有氧训练和最大摄氧量。"),
        EvidenceChunk("chunk-b", "paper.pdf", 2, "力量训练方法。"),
    )

    result = store.sync_chunks(chunks)

    assert result.total_chunks == 2
    assert result.added_chunks == 2
    assert result.recovery_backup is not None
    assert result.recovery_backup.is_dir()
    assert (result.recovery_backup / sentinel.name).read_bytes() == b"damaged index"
    assert store.open_attempts == 2
    assert store.healthy_collection.count() == 2
    assert source_pdf.read_bytes() == b"original source must remain"
    store.close()


class _SemanticOnlyStore:
    def search(self, question: str, *, top_k: int):
        return (VectorSearchResult("chunk-semantic", 0.91),)


class _FailingVectorStore:
    def search(self, question: str, *, top_k: int):
        raise VectorStoreError("测试模型不可用")


def test_hybrid_index_merges_semantic_and_keyword_results() -> None:
    chunks = (
        EvidenceChunk("chunk-keyword", "a.pdf", 1, "最大摄氧量测试。"),
        EvidenceChunk("chunk-semantic", "b.pdf", 2, "心肺耐力评估。"),
    )
    index = HybridEvidenceIndex(chunks, vector_store=_SemanticOnlyStore())
    results = index.search(
        "最大摄氧量",
        top_k=2,
        mode="hybrid",
    )

    assert results[0].chunk.chunk_id == "chunk-semantic"
    assert "BM25" in results[1].retrieval_sources
    assert "语义" in results[0].retrieval_sources


def test_hybrid_index_falls_back_when_semantic_search_fails() -> None:
    chunks = (
        EvidenceChunk("chunk-a", "paper.pdf", 1, "最大摄氧量测试。"),
    )
    index = HybridEvidenceIndex(chunks, vector_store=_FailingVectorStore())
    results = index.search("最大摄氧量", top_k=1, mode="semantic")

    assert results[0].chunk.chunk_id == "chunk-a"
    assert "已自动回退 BM25" in index.last_warning


def test_evidence_rag_marks_local_excerpt_as_simulation() -> None:
    pages = (
        PDFPageText(
            page_number=3,
            text="本研究采用随机对照试验，主要结局指标是最大摄氧量。",
        ),
    )
    index = LocalEvidenceIndex(chunk_pdf_pages("方法研究.pdf", pages))
    service = EvidenceRAGService(
        index,
        LLMService(Settings.from_env({"OPENAI_API_KEY": ""})),
    )
    answer = service.answer_question("研究设计和主要结局指标是什么？")

    assert answer.success is True
    assert answer.mode == "证据摘录（模拟）"
    assert "模拟内容提示" in answer.answer
    assert answer.evidence[0].chunk.page_number == 3


def test_evidence_rag_refuses_answer_without_supporting_text() -> None:
    pages = (
        PDFPageText(
            page_number=1,
            text="本文讨论最大摄氧量与耐力训练的关系。",
        ),
    )
    index = LocalEvidenceIndex(chunk_pdf_pages("耐力研究.pdf", pages))
    service = EvidenceRAGService(
        index,
        LLMService(Settings.from_env({"OPENAI_API_KEY": ""})),
    )
    answer = service.answer_question("量子计算芯片采用什么制造工艺？")

    assert answer.mode == "证据不足"
    assert "不足以支持结论" in answer.answer
    assert answer.evidence == ()


def test_bundled_knowledge_base_has_verified_metadata() -> None:
    bundle = load_bundled_knowledge_base()
    documents_by_file = {
        document.file_name: document for document in bundle.documents
    }

    expected_dois = {
        "ass_2401349.pdf": "10.12677/ass.2026.157601",
        "aps_3081402.pdf": "10.12677/aps.2026.143067",
        "huang-2023-social-ecological.pdf": "10.3390/jcm12030777",
        "sheng-2023-peer-support.pdf": "10.3389/fpsyg.2023.1103109",
        "zhang-2024-health-beliefs.pdf": "10.1186/s40359-024-01558-3",
        "wang-2024-blended-pe.pdf": "10.1186/s12889-024-20469-x",
        "niu-2024-taichi-fitness.pdf": "10.1371/journal.pone.0297117",
        "zhang-2025-predictors-pa.pdf": "10.3389/fpubh.2025.1699606",
        "sun-2025-aerobic-fitness.pdf": "10.1186/s13102-025-01149-8",
        "gao-2025-exercise-self-efficacy.pdf": "10.3389/fpsyg.2025.1428365",
        "xiong-2025-social-support.pdf": "10.3389/fpsyg.2025.1601079",
        "liang-2026-app-pe.pdf": "10.1371/journal.pone.0345759",
    }

    assert len(bundle.documents) == 12
    assert len(bundle.chunks) >= 100
    assert set(documents_by_file) == set(expected_dois)
    assert all(
        documents_by_file[file_name].doi == doi
        for file_name, doi in expected_dois.items()
    )
    assert documents_by_file["ass_2401349.pdf"].doi == "10.12677/ass.2026.157601"
    assert documents_by_file["aps_3081402.pdf"].doi == "10.12677/aps.2026.143067"
    assert documents_by_file["ass_2401349.pdf"].page_count == 14
    assert documents_by_file["aps_3081402.pdf"].page_count == 8
    assert all(
        "2023" <= document.year <= "2026"
        for document in bundle.documents
    )
    assert all(
        1 <= chunk.page_number
        <= documents_by_file[chunk.document_name].page_count
        for chunk in bundle.chunks
    )


def test_bundled_corpus_retrieves_both_papers_with_page_citations() -> None:
    bundle = load_bundled_knowledge_base()
    index = LocalEvidenceIndex(bundle.chunks)

    exercise_results = index.search(
        "体育锻炼 影响因素 个人 家庭 学校 社会",
        top_k=3,
    )
    fitness_results = index.search(
        "山东省 高校 体质健康 问卷 调查",
        top_k=3,
    )

    assert exercise_results[0].chunk.document_name == "ass_2401349.pdf"
    assert fitness_results[0].chunk.document_name == "aps_3081402.pdf"
    assert "第 1 页" in fitness_results[0].chunk.citation


def test_pdf_metadata_recognizes_both_bundled_papers() -> None:
    pdf_dir = Path(__file__).resolve().parent.parent / "knowledge_base" / "pdfs"
    exercise = extract_pdf_metadata(
        (pdf_dir / "ass_2401349.pdf").read_bytes(),
        "ass_2401349.pdf",
    )
    fitness = extract_pdf_metadata(
        (pdf_dir / "aps_3081402.pdf").read_bytes(),
        "aps_3081402.pdf",
    )

    assert exercise.title == "大学生体育锻炼行为的影响因素研究"
    assert exercise.authors == ("夏慧琴",)
    assert exercise.journal == "社会科学前沿"
    assert exercise.pages == "533-546"
    assert exercise.doi == "10.12677/ass.2026.157601"
    assert exercise.page_count == 14

    assert fitness.title == "大学生体质健康影响因素与提升对策研究"
    assert fitness.authors == ("彭玲", "王俊")
    assert fitness.journal == "体育科学进展"
    assert fitness.pages == "464-471"
    assert fitness.doi == "10.12677/aps.2026.143067"
    assert fitness.page_count == 8
    assert not exercise.warnings
    assert not fitness.warnings


def test_literature_framework_uses_all_papers_and_valid_pages() -> None:
    bundle = load_bundled_knowledge_base()
    framework = load_literature_framework()
    page_counts = {
        document.file_name: document.page_count for document in bundle.documents
    }
    documents_in_framework = {
        item.file_name
        for dimension in framework.dimensions
        for item in dimension.evidence
    }

    assert framework.documents_count == 12
    assert len(framework.dimensions) == 9
    assert list(framework.logic_chain) == [
        dimension.title for dimension in framework.dimensions
    ]
    assert framework.evidence_count >= 50
    assert documents_in_framework == set(page_counts)

    for dimension in framework.dimensions:
        assert dimension.evidence
        assert all(
            1 <= item.page_number <= page_counts[item.file_name]
            for item in dimension.evidence
        )
        assert all(item.chunk_id and item.excerpt for item in dimension.evidence)


def test_pdf_directory_sync_builds_index_and_framework(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge_base"
    pdf_dir = knowledge_dir / "pdfs"
    (knowledge_dir / "index").mkdir(parents=True)
    pdf_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "library_name": "自动化导入测试库",
        "description": "用于验证 PDF 自动入库和框架更新。",
        "documents": [],
    }
    (knowledge_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_pdf = (
        Path(__file__).resolve().parent.parent
        / "knowledge_base"
        / "pdfs"
        / "ass_2401349.pdf"
    )
    pdf_bytes = source_pdf.read_bytes()
    (pdf_dir / source_pdf.name).write_bytes(pdf_bytes)

    first_result = sync_pdf_directory(
        knowledge_dir,
        auto_vector_sync=False,
    )
    assert first_result.imported_count == 1
    assert first_result.framework.documents_count == 1
    assert (knowledge_dir / "index" / "evidence_chunks.json").exists()
    assert (knowledge_dir / "framework.json").exists()

    bundle = load_bundled_knowledge_base(knowledge_dir)
    framework = load_literature_framework(knowledge_dir)
    assert len(bundle.documents) == 1
    assert bundle.documents[0].doi == "10.12677/ass.2026.157601"
    assert framework.documents_count == 1
    assert framework.evidence_count > 0

    duplicate_result = ingest_pdf_files(
        [IncomingPDF(source_pdf.name, pdf_bytes)],
        knowledge_dir,
        auto_vector_sync=False,
    )
    assert duplicate_result.imported_count == 0
    assert duplicate_result.duplicates
    assert len(load_bundled_knowledge_base(knowledge_dir).documents) == 1


def test_library_removal_updates_manifest_index_and_framework(
    tmp_path: Path,
) -> None:
    """删除论文后不能留下清单、索引、框架或 PDF 残片。"""

    knowledge_dir = tmp_path / "knowledge_base"
    pdf_dir = knowledge_dir / "pdfs"
    (knowledge_dir / "index").mkdir(parents=True)
    pdf_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "library_name": "论文删除测试库",
        "description": "验证论文删除和派生索引重建。",
        "documents": [],
    }
    (knowledge_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_pdf = (
        Path(__file__).resolve().parent.parent
        / "knowledge_base"
        / "pdfs"
        / "ass_2401349.pdf"
    )
    ingest_result = ingest_pdf_files(
        [IncomingPDF(source_pdf.name, source_pdf.read_bytes())],
        knowledge_dir,
        auto_vector_sync=False,
    )
    imported = ingest_result.imported[0]

    removal = remove_library_documents(
        [imported.file_name],
        knowledge_dir,
        auto_vector_sync=False,
    )

    assert [item.file_name for item in removal.removed] == [imported.file_name]
    assert load_bundled_knowledge_base(knowledge_dir).documents == ()
    assert load_literature_framework(knowledge_dir).documents_count == 0
    assert not (pdf_dir / imported.file_name).exists()
    updated_manifest = json.loads(
        (knowledge_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert updated_manifest["documents"] == []
    index = json.loads(
        (knowledge_dir / "index" / "evidence_chunks.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["chunks"] == []


def test_light_and_dark_palettes_keep_required_contrast() -> None:
    """两套主题的关键文字和主按钮应满足 WCAG AA 对比度。"""

    from app import THEME_PALETTES

    required_tokens = {
        "background",
        "surface",
        "text",
        "muted",
        "primary",
        "primary_text",
        "border",
    }

    def relative_luminance(color: str) -> float:
        channels = [
            int(color[index : index + 2], 16) / 255
            for index in (1, 3, 5)
        ]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return (
            0.2126 * linear[0]
            + 0.7152 * linear[1]
            + 0.0722 * linear[2]
        )

    def contrast_ratio(foreground: str, background: str) -> float:
        lighter, darker = sorted(
            (
                relative_luminance(foreground),
                relative_luminance(background),
            ),
            reverse=True,
        )
        return (lighter + 0.05) / (darker + 0.05)

    assert set(THEME_PALETTES) == {"light", "dark"}
    for palette in THEME_PALETTES.values():
        assert required_tokens <= palette.keys()
        assert contrast_ratio(palette["text"], palette["background"]) >= 4.5
        assert contrast_ratio(palette["muted"], palette["background"]) >= 4.5
        assert (
            contrast_ratio(palette["primary_text"], palette["primary"])
            >= 4.5
        )

    assert (
        THEME_PALETTES["light"]["background"]
        != THEME_PALETTES["dark"]["background"]
    )


def test_sidebar_theme_switcher_offers_all_modes() -> None:
    """侧边栏主题按钮应覆盖浅色、深色和跟随系统。"""

    from app import THEME_SWITCHER_HTML

    assert 'data-theme-choice="Light"' in THEME_SWITCHER_HTML
    assert 'data-theme-choice="Dark"' in THEME_SWITCHER_HTML
    assert 'data-theme-choice="System"' in THEME_SWITCHER_HTML
    assert "stActiveTheme-/-v2" in THEME_SWITCHER_HTML


def test_page_assistant_context_covers_every_page() -> None:
    """每个功能页都应有专属智能问答上下文。"""

    from app import (
        PAGE_ASSISTANT_CONTEXTS,
        PAGES,
        build_page_assistant_system_prompt,
    )

    assert set(PAGES) <= PAGE_ASSISTANT_CONTEXTS.keys()
    prompt = build_page_assistant_system_prompt("数据分析")
    assert "数据分析" in prompt
    assert "不得编造论文" in prompt
    assert "证据不足" in prompt


def test_all_streamlit_pages_load_without_exception() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    test_app = AppTest.from_file(app_path, default_timeout=30).run()
    pages = (
        "首页",
        "智能科研问答",
        "科研选题",
        "文献阅读",
        "文献证据问答",
        "文献框架",
        "数据分析",
        "系统状态",
    )
    for page in pages:
        test_app.session_state["navigation"] = page
        test_app.run()
        assert not test_app.exception, f"页面运行失败：{page}"
