"""从 Zotero Desktop Local API 同步当前活动论文到 ResearchMind。

该服务只读取 Zotero 活动文库，不删除或修改 Zotero。同步目标是独立的
ResearchMind 领域，避免覆盖用户已有的本地论文库。
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from researchmind.adolescent_topic import assess_topic_relevance
from researchmind.library_state import mark_analysis_completed
from researchmind.models import EvidenceRef, PaperRecord, ResearchDomain
from researchmind.workflow import ResearchMindWorkflow
from tools.document_chunker import chunk_pdf_pages
from tools.pdf_reader import extract_pdf_text


ZOTERO_LOCAL_BASE_URL = "http://127.0.0.1:23119/api/users/0"
ZOTERO_DOMAIN_NAME = "Zotero 青少年体育文献库"
ZOTERO_DOMAIN_DESCRIPTION = (
    "通过 Zotero Desktop Local API 读取的当前活动青少年体育、体质健康、"
    "学校干预、心理健康和影响因素论文库。"
)
ZOTERO_DOMAIN_KEYWORDS = (
    "青少年",
    "体质健康",
    "体育锻炼",
    "学校体育",
    "心理健康",
)


class ZoteroLocalError(RuntimeError):
    """Zotero Local API 不可用或返回异常。"""


@dataclass(frozen=True)
class ZoteroLocalSyncResult:
    domain: ResearchDomain
    active_items: int
    imported: int
    updated: int
    metadata_only: int
    pdf_parsed: int
    evidence_written: int
    deactivated: int

    @property
    def message(self) -> str:
        return (
            f"Zotero 活动论文 {self.active_items} 篇；"
            f"新增 {self.imported} 篇、更新 {self.updated} 篇、"
            f"仅题录 {self.metadata_only} 篇；"
            f"成功解析 PDF {self.pdf_parsed} 篇，"
            f"写入页级证据 {self.evidence_written} 条；"
            f"不再活动 {self.deactivated} 篇。"
        )


def probe_zotero_local(
    base_url: str = ZOTERO_LOCAL_BASE_URL,
) -> dict[str, Any]:
    """读取本地用户文库的顶层条目，不执行任何写入。"""

    url = f"{base_url.rstrip('/')}/items/top?limit=100&include=data"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.load(response)
    except Exception as error:
        raise ZoteroLocalError(
            f"无法读取 Zotero Local API：{type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, list):
        raise ZoteroLocalError("Zotero Local API 返回格式不正确。")
    return {
        "ok": True,
        "active_items": sum(
            item.get("data", {}).get("itemType") == "journalArticle"
            for item in payload
        ),
        "base_url": base_url.rstrip("/"),
    }


def sync_zotero_local_library(
    workflow: ResearchMindWorkflow,
    *,
    base_url: str = ZOTERO_LOCAL_BASE_URL,
) -> ZoteroLocalSyncResult:
    """把 Zotero 当前活动论文同步到独立 ResearchMind 领域。"""

    if workflow.read_only:
        raise ValueError("只读论文库不能执行 Zotero 同步。")
    domain = _ensure_zotero_domain(workflow)
    items = _get_json(
        f"{base_url.rstrip('/')}/items/top?limit=100&include=data"
    )
    articles = [
        item
        for item in items
        if item.get("data", {}).get("itemType") == "journalArticle"
    ]
    library_collections = {
        collection["key"]: collection["data"]["name"]
        for collection in _get_json(
            f"{base_url.rstrip('/')}/collections?limit=100&include=data"
        )
    }
    imported = 0
    updated = 0
    metadata_only = 0
    pdf_parsed = 0
    evidence_written = 0
    seen_ids: set[str] = set()

    for item in articles:
        data = item["data"]
        title = str(data.get("title") or "").strip()
        abstract = str(data.get("abstractNote") or "").strip()
        keywords = tuple(
            str(tag.get("tag", "")).strip()
            for tag in data.get("tags", [])
            if str(tag.get("tag", "")).strip()
        )
        relevance = assess_topic_relevance(
            title=title,
            abstract=abstract,
            keywords=keywords,
        )
        doi = str(data.get("DOI") or "").strip()
        existing = workflow.store.find_paper_by_identifier(
            domain.domain_id,
            doi=doi,
            title=title,
            include_deleted=True,
        )
        paper_id = (
            existing.paper_id
            if existing is not None
            else f"zotero-{item['key']}"
        )
        seen_ids.add(paper_id)
        children = _get_json(
            f"{base_url.rstrip('/')}/items/{item['key']}/children?limit=100"
        )
        pdf_attachment = next(
            (
                child
                for child in children
                if child.get("data", {}).get("itemType") == "attachment"
                and child.get("data", {}).get("contentType") == "application/pdf"
                and child.get("links", {}).get("enclosure", {}).get("href")
            ),
            None,
        )
        pdf_path = ""
        evidence: tuple[EvidenceRef, ...] = ()
        if pdf_attachment is not None:
            pdf_path = str(
                _local_file_path(
                    pdf_attachment["links"]["enclosure"]["href"]
                )
            )
            if Path(pdf_path).is_file():
                extraction = extract_pdf_text(
                    Path(pdf_path).read_bytes(),
                    max_pages=300,
                    max_chars=1_000_000,
                )
                if extraction.success and extraction.pages:
                    chunks = chunk_pdf_pages(
                        Path(pdf_path).name,
                        extraction.pages,
                    )
                    evidence = tuple(
                        EvidenceRef(
                            chunk_id=f"{paper_id}-{chunk.chunk_id}",
                            document_id=paper_id,
                            document_title=title,
                            file_name=Path(pdf_path).name,
                            page_number=chunk.page_number,
                            quote=chunk.text,
                        )
                        for chunk in chunks
                    )
                    pdf_parsed += 1
            if not evidence:
                metadata_only += 1
        else:
            metadata_only += 1

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        authors = tuple(
            " ".join(
                part
                for part in (
                    str(creator.get("firstName", "")).strip(),
                    str(creator.get("lastName", "")).strip(),
                )
                if part
            )
            or str(creator.get("name", "")).strip()
            for creator in data.get("creators", [])
            if (
                str(creator.get("lastName", "")).strip()
                or str(creator.get("name", "")).strip()
            )
        )
        category_names = [
            library_collections.get(key, "")
            for key in data.get("collections", [])
        ]
        paper = PaperRecord(
            paper_id=paper_id,
            domain_id=domain.domain_id,
            title=title,
            authors=authors,
            year=str(data.get("date") or "待核对"),
            journal=str(data.get("publicationTitle") or "待核对"),
            doi=doi,
            source_url=str(data.get("url") or ""),
            abstract=abstract,
            keywords=keywords,
            pdf_path=pdf_path,
            source="Zotero Local API",
            verification_status=(
                f"Zotero 活动题录；专题规则判断：{relevance.status}；"
                + (
                    "PDF 已完成页级文本解析。"
                    if evidence
                    else "暂无可用可解析 PDF。"
                )
            ),
            created_at=existing.created_at if existing else now,
            imported_at=existing.imported_at if existing else now,
            category=next(
                (name for name in category_names if name),
                "待分类",
            ),
            parse_status="已解析" if evidence else "仅元数据",
            analysis_status="待分析" if evidence else "未分析",
            normalized_title="",
            normalization_version="v1",
            import_origin="Zotero Local API",
            is_manifest_managed=False,
        )
        workflow.store.upsert_paper(paper)
        workflow.store.replace_evidence(paper_id, evidence)
        evidence_written += len(evidence)
        if existing is None:
            imported += 1
        else:
            updated += 1

    current_zotero_papers = {
        paper.paper_id: paper
        for paper in workflow.store.list_papers(domain.domain_id)
        if paper.source == "Zotero Local API"
    }
    stale_ids = tuple(
        paper_id
        for paper_id in current_zotero_papers
        if paper_id not in seen_ids
    )
    if stale_ids:
        workflow.store.soft_delete_papers(domain.domain_id, stale_ids)
    workflow.store.mark_domain_analysis_stale(
        domain.domain_id,
        reason="Zotero 本地论文库发生同步更新",
        affected_paper_ids=tuple(seen_ids),
    )
    workflow.store.set_metadata(
        "researchmind_preferred_domain",
        domain.domain_id,
    )
    return ZoteroLocalSyncResult(
        domain=domain,
        active_items=len(articles),
        imported=imported,
        updated=updated,
        metadata_only=metadata_only,
        pdf_parsed=pdf_parsed,
        evidence_written=evidence_written,
        deactivated=len(stale_ids),
    )


def _ensure_zotero_domain(workflow: ResearchMindWorkflow) -> ResearchDomain:
    for domain in workflow.store.list_domains():
        if domain.name == ZOTERO_DOMAIN_NAME:
            return domain
    return workflow.create_domain(
        name=ZOTERO_DOMAIN_NAME,
        description=ZOTERO_DOMAIN_DESCRIPTION,
        keywords=ZOTERO_DOMAIN_KEYWORDS,
    )


def _get_json(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except Exception as error:
        raise ZoteroLocalError(
            f"Zotero Local API 请求失败：{type(error).__name__}: {error}"
        ) from error


def _local_file_path(href: str) -> Path:
    parsed = urllib.parse.urlparse(href)
    return Path(urllib.request.url2pathname(parsed.path))


__all__ = [
    "ZOTERO_DOMAIN_NAME",
    "ZoteroLocalError",
    "ZoteroLocalSyncResult",
    "probe_zotero_local",
    "sync_zotero_local_library",
]
