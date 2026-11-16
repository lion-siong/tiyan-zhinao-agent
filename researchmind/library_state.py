"""论文库版本指纹与派生科研分析的统一状态管理。

论文库是唯一事实源。本模块不生成论文，只根据当前活动论文、证据和分析状态
计算指纹，并通过 SQLite ``app_metadata`` 持久化知识图谱/学术分歧的更新状态。
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from researchmind.evidence_quality import ANALYSIS_ENGINE_VERSION
from researchmind.knowledge_map import KNOWLEDGE_MAP_ENGINE_VERSION
from researchmind.models import ResearchDomain
from researchmind.storage import ResearchMindStore

if TYPE_CHECKING:
    from researchmind.workflow import ResearchMindWorkflow


ANALYSIS_STATUSES = ("待更新", "分析中", "已更新", "失败")
_STATUS_PREFIX = "analysis_status:"
_FINGERPRINT_PREFIX = "analysis_fingerprint:"
_TOPIC_PREFIX = "analysis_topic:"
_COMPLETED_PREFIX = "analysis_completed_at:"
_ERROR_PREFIX = "analysis_error:"
_SCOPE_PREFIX = "analysis_paper_ids:"
_STARTED_PREFIX = "analysis_started_at:"

_JOB_LOCK = threading.Lock()
_ACTIVE_JOBS: dict[tuple[str, str, str], Future[None]] = {}
_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="researchmind-analysis",
)


@dataclass(frozen=True)
class AnalysisState:
    """页面可展示的派生分析状态。"""

    domain_id: str
    topic: str
    status: str
    library_fingerprint: str
    analysis_fingerprint: str
    paper_count: int
    analyzed_paper_ids: tuple[str, ...]
    latest_analysis_at: str
    error: str
    is_current: bool
    latest_paper_included: bool

    @property
    def version(self) -> str:
        return self.library_fingerprint[:12]


def library_fingerprint(
    store: ResearchMindStore,
    domain_id: str,
) -> str:
    """根据活动论文元数据和页级证据生成稳定 SHA-256 指纹。"""

    digest = hashlib.sha256()
    papers = store.list_papers(domain_id)
    for paper in sorted(papers, key=lambda item: item.paper_id):
        metadata = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "journal": paper.journal,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "keywords": paper.keywords,
            "category": paper.category,
        }
        digest.update(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for evidence in store.list_evidence(paper.paper_id):
            digest.update(
                json.dumps(
                    {
                        "chunk_id": evidence.chunk_id,
                        "page": evidence.page_number,
                        "quote": evidence.quote,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    digest.update(
        json.dumps(
            {"analysis_engine_version": ANALYSIS_ENGINE_VERSION},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def graph_fingerprint(
    store: ResearchMindStore,
    domain_id: str,
) -> str:
    """论文库源数据、结构化阅读和有效分析结果的组合指纹。"""

    digest = hashlib.sha256(
        library_fingerprint(store, domain_id).encode("utf-8")
    )
    for paper in sorted(
        store.list_papers(domain_id),
        key=lambda item: item.paper_id,
    ):
        analysis = store.get_analysis(paper.paper_id)
        digest.update(
            json.dumps(
                analysis.to_dict() if analysis else None,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    debate = store.latest_debate(domain_id)
    synthesis = store.get_latest_synthesis(domain_id)
    digest.update(
        json.dumps(
            {
                "debate": debate.to_dict() if debate else None,
                "synthesis": synthesis.to_dict() if synthesis else None,
                "knowledge_map_engine_version": KNOWLEDGE_MAP_ENGINE_VERSION,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def get_analysis_state(
    store: ResearchMindStore,
    domain_id: str,
    topic: str,
) -> AnalysisState:
    """读取数据库中的持久化分析状态并与当前论文库指纹比较。"""

    current_fingerprint = library_fingerprint(store, domain_id)
    stored_fingerprint = (
        store.get_metadata(f"{_FINGERPRINT_PREFIX}{domain_id}") or ""
    )
    stored_topic = store.get_metadata(f"{_TOPIC_PREFIX}{domain_id}") or ""
    raw_status = (
        store.get_metadata(f"{_STATUS_PREFIX}{domain_id}") or "待更新"
    )
    paper_ids_raw = (
        store.get_metadata(f"{_SCOPE_PREFIX}{domain_id}") or "[]"
    )
    try:
        analyzed_ids = tuple(json.loads(paper_ids_raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        analyzed_ids = ()
    active_ids = tuple(
        paper.paper_id for paper in store.list_papers(domain_id)
    )
    is_current = bool(
        stored_fingerprint
        and stored_fingerprint == current_fingerprint
        and stored_topic == topic
        and store.latest_debate(domain_id) is not None
        and store.get_latest_synthesis(domain_id) is not None
    )
    if raw_status == "分析中" and analysis_job_running(store, domain_id):
        status = "分析中"
    elif is_current:
        status = "已更新"
    else:
        status = raw_status
    if status not in ANALYSIS_STATUSES:
        status = "待更新"
    if status == "已更新" and not is_current:
        status = "待更新"
    latest = (
        store.get_metadata(f"{_COMPLETED_PREFIX}{domain_id}")
        or store.get_metadata(f"{_STARTED_PREFIX}{domain_id}")
        or ""
    )
    return AnalysisState(
        domain_id=domain_id,
        topic=topic,
        status=status,
        library_fingerprint=current_fingerprint,
        analysis_fingerprint=stored_fingerprint,
        paper_count=len(active_ids),
        analyzed_paper_ids=analyzed_ids,
        latest_analysis_at=latest,
        error=store.get_metadata(f"{_ERROR_PREFIX}{domain_id}") or "",
        is_current=is_current,
        latest_paper_included=is_current
        and set(analyzed_ids) == set(active_ids),
    )


def mark_analysis_running(
    store: ResearchMindStore,
    domain_id: str,
    topic: str,
) -> None:
    store.set_metadata(f"{_STATUS_PREFIX}{domain_id}", "分析中")
    store.set_metadata(f"{_TOPIC_PREFIX}{domain_id}", topic)
    store.set_metadata(f"{_ERROR_PREFIX}{domain_id}", "")
    store.set_metadata(
        f"{_STARTED_PREFIX}{domain_id}",
        _now_iso(),
    )


def mark_analysis_failed(
    store: ResearchMindStore,
    domain_id: str,
    topic: str,
    error: str,
) -> None:
    store.set_metadata(f"{_STATUS_PREFIX}{domain_id}", "失败")
    store.set_metadata(f"{_TOPIC_PREFIX}{domain_id}", topic)
    store.set_metadata(f"{_ERROR_PREFIX}{domain_id}", str(error)[:1000])


def mark_analysis_completed(
    store: ResearchMindStore,
    domain_id: str,
    topic: str,
    fingerprint: str,
) -> None:
    paper_ids = tuple(
        paper.paper_id for paper in store.list_papers(domain_id)
    )
    store.set_metadata_many(
        {
            f"{_STATUS_PREFIX}{domain_id}": "已更新",
            f"{_TOPIC_PREFIX}{domain_id}": topic,
            f"{_FINGERPRINT_PREFIX}{domain_id}": fingerprint,
            f"{_ERROR_PREFIX}{domain_id}": "",
            f"analysis_invalidation_reason:{domain_id}": "",
            f"{_SCOPE_PREFIX}{domain_id}": json.dumps(
                list(paper_ids),
                ensure_ascii=False,
            ),
            f"{_COMPLETED_PREFIX}{domain_id}": _now_iso(),
        }
    )


def refresh_domain_analysis_now(
    workflow: "ResearchMindWorkflow",
    domain: ResearchDomain,
    topic: str,
) -> AnalysisState:
    """同步重建全部论文阅读、学术分歧和知识体系。"""

    workflow._ensure_writable()
    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("分析主题不能为空。")
    if workflow.store.paper_count(domain.domain_id) == 0:
        raise ValueError("当前领域没有可分析论文。")
    fingerprint = library_fingerprint(workflow.store, domain.domain_id)
    mark_analysis_running(workflow.store, domain.domain_id, clean_topic)
    try:
        workflow.reader.read_domain(domain.domain_id)
        debate = workflow.debate_analyzer.analyze(
            domain.domain_id,
            clean_topic,
        )
        workflow.synthesizer.synthesize(
            domain.domain_id,
            debate=debate,
        )
        mark_analysis_completed(
            workflow.store,
            domain.domain_id,
            clean_topic,
            fingerprint,
        )
    except Exception as error:
        mark_analysis_failed(
            workflow.store,
            domain.domain_id,
            clean_topic,
            f"{type(error).__name__}: {error}",
        )
        raise
    return get_analysis_state(workflow.store, domain.domain_id, clean_topic)


def schedule_analysis_refresh(
    workflow: "ResearchMindWorkflow",
    domain: ResearchDomain,
    topic: str,
    *,
    force: bool = False,
) -> bool:
    """在后台排队一次分析；同一论文库版本和主题不会重复提交。"""

    clean_topic = topic.strip()
    if not clean_topic:
        raise ValueError("分析主题不能为空。")
    state = get_analysis_state(workflow.store, domain.domain_id, clean_topic)
    if state.is_current and not force:
        return False
    key = (
        str(workflow.store.db_path),
        domain.domain_id,
        state.library_fingerprint,
    )
    with _JOB_LOCK:
        running = _ACTIVE_JOBS.get(key)
        if running is not None and not running.done():
            return False
        mark_analysis_running(workflow.store, domain.domain_id, clean_topic)

        def job() -> None:
            try:
                refresh_domain_analysis_now(
                    workflow,
                    domain,
                    clean_topic,
                )
            finally:
                with _JOB_LOCK:
                    _ACTIVE_JOBS.pop(key, None)

        _ACTIVE_JOBS[key] = _EXECUTOR.submit(job)
    return True


def analysis_job_running(
    store: ResearchMindStore,
    domain_id: str,
) -> bool:
    with _JOB_LOCK:
        return any(
            key[0] == str(store.db_path)
            and key[1] == domain_id
            and not future.done()
            for key, future in _ACTIVE_JOBS.items()
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "ANALYSIS_STATUSES",
    "AnalysisState",
    "analysis_job_running",
    "get_analysis_state",
    "graph_fingerprint",
    "library_fingerprint",
    "mark_analysis_completed",
    "mark_analysis_failed",
    "mark_analysis_running",
    "refresh_domain_analysis_now",
    "schedule_analysis_refresh",
]
