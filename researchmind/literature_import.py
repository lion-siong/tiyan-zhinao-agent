"""文献检索结果入库、去重、回收站和安全永久删除。

本模块把候选元数据与现有论文库连接起来。检索接口返回的内容只视为候选，
只有经过专题相关度判断并成功写入 SQLite 后，界面才显示为“已导入”。
开放 PDF 仍通过现有 PDF 导入器处理，不绕过付费墙或登录限制。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from researchmind.adolescent_library import (
    LibraryBackup,
    create_library_backup,
)
from researchmind.adolescent_topic import (
    THEME_BY_ID,
    assess_topic_relevance,
)
from researchmind.collectors import LiteratureCollectionError, LiteratureCollector
from researchmind.models import (
    CandidatePaper,
    PaperRecord,
    ResearchDomain,
)
from researchmind.storage import ResearchMindStore
from researchmind.workflow import (
    DEFAULT_DOMAIN_NAME,
    ResearchMindWorkflow,
)
from tools.knowledge_base import KnowledgeBaseError
from tools.library_ingest import (
    IncomingPDF,
    ingest_pdf_files,
    remove_library_documents,
)


@dataclass(frozen=True)
class CandidateLibraryStatus:
    """一个候选论文在当前论文库中的真实状态。"""

    candidate_id: str
    status: str
    paper_id: str = ""
    title: str = ""

    @property
    def is_imported(self) -> bool:
        return self.status in {"已导入", "回收站"}

    @property
    def is_deleted(self) -> bool:
        return self.status == "回收站"


@dataclass(frozen=True)
class ImportResult:
    """一次单篇或批量导入的可核验结果。"""

    imported: int = 0
    updated: int = 0
    skipped_duplicates: int = 0
    skipped_ineligible: int = 0
    failed: int = 0
    restored: int = 0
    downloaded_pdfs: int = 0
    paper_ids: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()

    @property
    def processed(self) -> int:
        return (
            self.imported
            + self.updated
            + self.skipped_duplicates
            + self.skipped_ineligible
            + self.failed
            + self.restored
        )

    @property
    def changed(self) -> bool:
        return bool(
            self.imported
            or self.updated
            or self.restored
            or self.downloaded_pdfs
        )


@dataclass(frozen=True)
class DeletionResult:
    """软删除或恢复的可核验结果。"""

    paper_ids: tuple[str, ...]
    invalidated: dict[str, int]


@dataclass(frozen=True)
class PermanentDeletionResult:
    """永久删除结果，保留备份和删除审计。"""

    paper_ids: tuple[str, ...]
    backup: LibraryBackup
    removed_pdf_files: tuple[str, ...]
    preserved_shared_files: tuple[str, ...]
    invalidated: dict[str, int]


def normalize_doi(value: str) -> str:
    """规范化 DOI，仅用于本地去重，不改写展示值。"""

    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.rstrip(".,;，；。").casefold()


def normalize_title(value: str) -> str:
    """规范化题名，忽略全半角、大小写、空白和常见标点差异。"""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def library_status_for_candidates(
    store: ResearchMindStore,
    domain_id: str,
    candidates: Iterable[CandidatePaper],
) -> dict[str, CandidateLibraryStatus]:
    """返回每个候选论文是否已导入或位于回收站。"""

    statuses: dict[str, CandidateLibraryStatus] = {}
    for candidate in candidates:
        existing = store.find_paper_by_identifier(
            domain_id,
            doi=candidate.doi,
            title=candidate.title,
            include_deleted=True,
        )
        if existing is None:
            status = CandidateLibraryStatus(
                candidate_id=candidate.candidate_id,
                status="未导入",
                title=candidate.title,
            )
        else:
            status = CandidateLibraryStatus(
                candidate_id=candidate.candidate_id,
                status="回收站" if existing.is_deleted else "已导入",
                paper_id=existing.paper_id,
                title=existing.title,
            )
        statuses[candidate.candidate_id] = status
    return statuses


def import_candidates(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    candidates: Iterable[CandidatePaper],
    *,
    download_open_pdf: bool = False,
    collector: LiteratureCollector | None = None,
) -> ImportResult:
    """把候选论文真实写入当前专题库，并返回逐项处理统计。"""

    if workflow.read_only:
        raise ValueError("旧库为只读模式，不能导入新论文。")
    unique_candidates = _deduplicate_candidates(tuple(candidates))
    if not unique_candidates:
        return ImportResult(messages=("没有可处理的候选论文。",))

    messages: list[str] = []
    downloaded_pdfs = 0
    downloaded_manifest_ids: dict[str, str] = {}
    relevance_by_id = {
        candidate.candidate_id: assess_topic_relevance(
            title=candidate.title,
            abstract=candidate.abstract,
            keywords=candidate.keywords,
        )
        for candidate in unique_candidates
    }

    if download_open_pdf:
        pdf_client = collector or LiteratureCollector(workflow.config)
        incoming: list[IncomingPDF] = []
        incoming_by_name: dict[str, CandidatePaper] = {}
        for candidate in unique_candidates:
            if (
                not candidate.has_open_pdf
                or not relevance_by_id[candidate.candidate_id].is_core_relevant
            ):
                continue
            existing = workflow.store.find_paper_by_identifier(
                domain.domain_id,
                doi=candidate.doi,
                title=candidate.title,
                include_deleted=True,
            )
            if (
                existing is not None
                and not existing.is_deleted
                and Path(existing.pdf_path).is_file()
            ):
                continue
            try:
                data = pdf_client.download_open_pdf(candidate)
            except LiteratureCollectionError as exc:
                messages.append(
                    f"{candidate.title}：开放 PDF 下载失败，将保留元数据导入。"
                    f"{exc}"
                )
                continue
            file_name = _safe_pdf_name(candidate)
            incoming.append(IncomingPDF(file_name=file_name, data=data))
            incoming_by_name[file_name] = candidate

        if incoming:
            imported_by_candidate: dict[str, str] = {}
            try:
                ingest_result = ingest_pdf_files(
                    incoming,
                    workflow.knowledge_base_dir,
                )
                for imported in ingest_result.imported:
                    original_name = (
                        imported.original_file_name or imported.file_name
                    )
                    candidate = incoming_by_name.get(original_name)
                    if candidate is not None:
                        imported_by_candidate[candidate.candidate_id] = (
                            imported.document_id
                        )
                downloaded_pdfs = len(imported_by_candidate)
                for warning in ingest_result.warnings:
                    messages.append(f"PDF 导入提示：{warning}")
                for duplicate in ingest_result.duplicates:
                    messages.append(f"PDF 去重：{duplicate}")
                for skipped in ingest_result.skipped:
                    messages.append(f"PDF 未导入：{skipped}")
            except (KnowledgeBaseError, OSError, ValueError) as exc:
                messages.append(
                    f"开放 PDF 批量导入失败，元数据仍可继续入库：{exc}"
                )

            if imported_by_candidate:
                try:
                    workflow.sync_bundled_library(
                        domain=domain,
                        force=True,
                    )
                except Exception as exc:
                    messages.append(
                        "PDF 已写入专题文件库，但 SQLite 同步失败；"
                        f"请稍后在论文库点击“同步最新专题论文库”。{exc}"
                    )
                else:
                    for candidate_id, document_id in (
                        imported_by_candidate.items()
                    ):
                        candidate = next(
                            item
                            for item in unique_candidates
                            if item.candidate_id == candidate_id
                        )
                        paper = workflow.store.find_paper_by_identifier(
                            domain.domain_id,
                            doi=candidate.doi,
                            title=candidate.title,
                            include_deleted=True,
                        )
                        if paper is None:
                            manifest_id = _manifest_paper_id(
                                domain,
                                document_id,
                            )
                            paper = workflow.store.get_paper(
                                manifest_id,
                                include_deleted=True,
                            )
                        if paper is not None and Path(paper.pdf_path).is_file():
                            downloaded_manifest_ids[candidate_id] = (
                                paper.paper_id
                            )
                        else:
                            messages.append(
                                f"{candidate.title}：PDF 已写入清单，"
                                "但 SQLite 中未能确认对应全文记录。"
                            )

    stats = {
        "imported": 0,
        "updated": 0,
        "skipped_duplicates": 0,
        "skipped_ineligible": 0,
        "failed": 0,
        "restored": 0,
    }
    affected_ids: list[str] = []

    for candidate in unique_candidates:
        relevance = relevance_by_id[candidate.candidate_id]
        if not relevance.is_core_relevant:
            stats["skipped_ineligible"] += 1
            messages.append(
                f"{candidate.title}：专题判断为“{relevance.status}”，"
                "未写入核心论文库。"
            )
            continue
        try:
            outcome, paper_id = _import_one_candidate(
                workflow.store,
                domain,
                candidate,
                relevance_category=_category_for_candidate(candidate),
                preferred_paper_id=downloaded_manifest_ids.get(
                    candidate.candidate_id
                ),
            )
        except Exception as exc:
            stats["failed"] += 1
            messages.append(f"{candidate.title}：导入失败。{exc}")
            continue

        stats[outcome] += 1
        if paper_id:
            affected_ids.append(paper_id)
        messages.append(f"{candidate.title}：{_outcome_text(outcome)}。")

    return ImportResult(
        **stats,
        downloaded_pdfs=downloaded_pdfs,
        paper_ids=tuple(dict.fromkeys(affected_ids)),
        messages=tuple(messages),
    )


def soft_delete_papers(
    store: ResearchMindStore,
    domain_id: str,
    paper_ids: Iterable[str],
) -> DeletionResult:
    """把论文移入回收站，并同步失效依赖这些论文的分析结果。"""

    deleted_ids = store.soft_delete_papers(domain_id, paper_ids)
    invalidated = store.invalidate_domain_derivations(
        domain_id,
        deleted_ids,
    )
    return DeletionResult(
        paper_ids=deleted_ids,
        invalidated=invalidated,
    )


def restore_papers(
    store: ResearchMindStore,
    domain_id: str,
    paper_ids: Iterable[str],
) -> tuple[str, ...]:
    """从回收站恢复论文，不自动恢复已经失效的旧分析结果。"""

    return store.restore_papers(domain_id, paper_ids)


def permanently_delete_papers(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    paper_ids: Iterable[str],
    *,
    reason: str = "用户在回收站永久删除",
) -> PermanentDeletionResult:
    """备份、记录审计、清理独占 PDF，再永久删除数据库记录。"""

    if workflow.read_only:
        raise ValueError("旧库为只读模式，不能永久删除论文。")
    selected_ids = tuple(
        dict.fromkeys(str(item) for item in paper_ids if str(item).strip())
    )
    all_papers = workflow.store.list_papers(
        domain.domain_id,
        include_deleted=True,
    )
    by_id = {paper.paper_id: paper for paper in all_papers}
    selected = tuple(
        by_id[paper_id]
        for paper_id in selected_ids
        if paper_id in by_id
    )
    if not selected:
        raise ValueError("未找到要永久删除的论文。")
    if any(not paper.is_deleted for paper in selected):
        raise ValueError("永久删除前必须先将论文移入回收站。")

    backup = create_library_backup(
        workflow,
        reason="永久删除前自动备份",
    )
    workflow.store.save_tombstones(selected, reason=reason)
    invalidated = workflow.store.invalidate_domain_derivations(
        domain.domain_id,
        (paper.paper_id for paper in selected),
    )

    selected_id_set = {paper.paper_id for paper in selected}
    other_paths = {
        str(Path(paper.pdf_path).resolve())
        for paper in all_papers
        if paper.paper_id not in selected_id_set and paper.pdf_path.strip()
    }
    pdf_root = (workflow.knowledge_base_dir / "pdfs").resolve()
    removable_names: list[str] = []
    preserved_shared: list[str] = []
    for paper in selected:
        if not paper.pdf_path.strip():
            continue
        path = Path(paper.pdf_path).resolve()
        if path.parent != pdf_root:
            continue
        if str(path) in other_paths:
            preserved_shared.append(path.name)
        else:
            removable_names.append(path.name)

    if removable_names:
        try:
            remove_library_documents(
                removable_names,
                workflow.knowledge_base_dir,
                allow_missing=True,
            )
        except (KnowledgeBaseError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"PDF 清理失败，数据库记录未永久删除：{exc}"
            ) from exc

    deleted_ids = workflow.store.permanently_delete_papers(selected_ids)
    return PermanentDeletionResult(
        paper_ids=deleted_ids,
        backup=backup,
        removed_pdf_files=tuple(removable_names),
        preserved_shared_files=tuple(preserved_shared),
        invalidated=invalidated,
    )


def _import_one_candidate(
    store: ResearchMindStore,
    domain: ResearchDomain,
    candidate: CandidatePaper,
    *,
    relevance_category: str,
    preferred_paper_id: str | None = None,
) -> tuple[str, str]:
    """写入一篇候选，返回结果类型和论文 ID。"""

    existing = store.find_paper_by_identifier(
        domain.domain_id,
        doi=candidate.doi,
        title=candidate.title,
        include_deleted=True,
    )
    if existing is None and preferred_paper_id:
        existing = store.get_paper(
            preferred_paper_id,
            include_deleted=True,
        )
    if existing is not None:
        has_full_text = bool(
            store.list_evidence(existing.paper_id, limit=1)
        )
        merged = _merge_missing_metadata(
            existing,
            candidate,
            category=relevance_category,
            has_full_text=has_full_text,
        )
        changed = merged != existing
        if changed:
            store.upsert_paper(merged)
        if not has_full_text and (
            existing.parse_status == "已解析"
            or existing.analysis_status == "已分析"
        ):
            # 兼容早期版本留下的状态：没有正文证据时不能继续标为已解析。
            store.update_paper_status(
                existing.paper_id,
                parse_status="仅元数据",
                analysis_status=(
                    "待补充全文"
                    if existing.analysis_status == "已分析"
                    else existing.analysis_status
                ),
            )
        restored = False
        if existing.is_deleted:
            restored_ids = store.restore_papers(
                domain.domain_id,
                (existing.paper_id,),
            )
            restored = existing.paper_id in restored_ids
        if restored and changed:
            return "restored", existing.paper_id
        if restored:
            return "restored", existing.paper_id
        if changed:
            return "updated", existing.paper_id
        return "skipped_duplicates", existing.paper_id

    now = _now()
    paper_id = _paper_id_for_candidate(domain.domain_id, candidate)
    paper = PaperRecord(
        paper_id=paper_id,
        domain_id=domain.domain_id,
        title=candidate.title.strip(),
        authors=tuple(candidate.authors),
        year=candidate.year.strip() or "待核对",
        journal=candidate.journal.strip() or "待核对",
        doi=candidate.doi.strip(),
        source_url=candidate.source_url.strip(),
        abstract=candidate.abstract.strip(),
        keywords=tuple(candidate.keywords),
        pdf_path="",
        source=candidate.source,
        verification_status=(
            f"仅元数据，由 {candidate.source} 公开接口返回；"
            "尚未获取并核验全文。"
        ),
        created_at=now,
        imported_at=now,
        category=relevance_category,
        parse_status="仅元数据",
        analysis_status="未分析",
        normalized_title=normalize_title(candidate.title),
        normalization_version="v1",
        import_origin=candidate.source,
        is_manifest_managed=False,
    )
    store.upsert_paper(paper)
    return "imported", paper_id


def _merge_missing_metadata(
    existing: PaperRecord,
    candidate: CandidatePaper,
    *,
    category: str,
    has_full_text: bool,
) -> PaperRecord:
    """只补空字段，不覆盖已有全文、证据和核验状态。"""

    verification = existing.verification_status
    if not Path(existing.pdf_path).is_file() and not verification.strip():
        verification = (
            f"仅元数据，由 {candidate.source} 公开接口返回；"
            "尚未获取并核验全文。"
        )
    updated = replace(
        existing,
        title=existing.title.strip() or candidate.title.strip(),
        authors=existing.authors or tuple(candidate.authors),
        year=(
            candidate.year.strip()
            if (
                existing.year.strip() in {"", "待核对"}
                and candidate.year.strip()
            )
            else existing.year
        ),
        journal=(
            candidate.journal.strip()
            if (
                existing.journal.strip() in {"", "待核对"}
                and candidate.journal.strip()
            )
            else existing.journal
        ),
        doi=existing.doi.strip() or candidate.doi.strip(),
        source_url=existing.source_url.strip() or candidate.source_url.strip(),
        abstract=existing.abstract.strip() or candidate.abstract.strip(),
        keywords=existing.keywords or tuple(candidate.keywords),
        pdf_path=existing.pdf_path,
        source=existing.source.strip() or candidate.source,
        verification_status=verification,
        category=(
            category
            if existing.category.strip() in {"", "待分类"}
            else existing.category
        ),
        import_origin=existing.import_origin.strip() or candidate.source,
        normalized_title=(
            existing.normalized_title.strip()
            or normalize_title(existing.title or candidate.title)
        ),
        parse_status="已解析" if has_full_text else "仅元数据",
        analysis_status=(
            existing.analysis_status
            if has_full_text or existing.analysis_status != "已分析"
            else "待补充全文"
        ),
    )
    return updated


def _category_for_candidate(candidate: CandidatePaper) -> str:
    relevance = assess_topic_relevance(
        title=candidate.title,
        abstract=candidate.abstract,
        keywords=candidate.keywords,
    )
    if relevance.matched_theme_ids:
        return THEME_BY_ID[relevance.matched_theme_ids[0]].label
    return "待分类"


def _deduplicate_candidates(
    candidates: tuple[CandidatePaper, ...],
) -> tuple[CandidatePaper, ...]:
    output: list[CandidatePaper] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    for candidate in candidates:
        doi_key = normalize_doi(candidate.doi)
        title_key = normalize_title(candidate.title)
        if not doi_key and not title_key:
            continue
        if (
            (doi_key and doi_key in seen_dois)
            or (title_key and title_key in seen_titles)
        ):
            continue
        if doi_key:
            seen_dois.add(doi_key)
        if title_key:
            seen_titles.add(title_key)
        output.append(candidate)
    return tuple(output)


def _paper_id_for_candidate(
    domain_id: str,
    candidate: CandidatePaper,
) -> str:
    identifier = normalize_doi(candidate.doi) or normalize_title(candidate.title)
    digest = hashlib.sha1(
        f"{domain_id}|{identifier}".encode("utf-8")
    ).hexdigest()[:20]
    return f"paper-import-{digest}"


def _manifest_paper_id(
    domain: ResearchDomain,
    document_id: str,
) -> str:
    """按工作流现有规则计算清单论文 ID，用于关联刚导入的 PDF。"""

    if domain.name == DEFAULT_DOMAIN_NAME:
        return document_id
    return f"{domain.domain_id}-paper-{document_id}"


def _safe_pdf_name(candidate: CandidatePaper) -> str:
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", candidate.title)[:80]
    return f"{candidate.candidate_id}-{stem or 'paper'}.pdf"


def _outcome_text(outcome: str) -> str:
    return {
        "imported": "已导入",
        "updated": "已补齐已有记录的空缺元数据",
        "restored": "已从回收站恢复并重新导入",
        "skipped_duplicates": "论文库中已有完整记录，未重复写入",
    }.get(outcome, outcome)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CandidateLibraryStatus",
    "DeletionResult",
    "ImportResult",
    "PermanentDeletionResult",
    "import_candidates",
    "library_status_for_candidates",
    "normalize_doi",
    "normalize_title",
    "permanently_delete_papers",
    "restore_papers",
    "soft_delete_papers",
]
