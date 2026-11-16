#!/usr/bin/env python3
"""Real-data acceptance test for library -> graph/debate synchronization."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings
from researchmind.knowledge_map import build_knowledge_map
from researchmind.library_state import (
    get_analysis_state,
    refresh_domain_analysis_now,
)
from researchmind.literature_import import import_candidates, soft_delete_papers
from researchmind.models import CandidatePaper
from researchmind.storage import ResearchMindStore
from researchmind.workflow import ResearchMindWorkflow
from researchmind.zotero_local_service import sync_zotero_local_library


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "acceptance-linkage"
POOL_PATH = ROOT / "zotero" / "batch2" / "candidate-pool.json"
RESULT_PATH = ROOT / "reports" / "library-linkage-acceptance.json"
ADDITIONAL_DOIS = (
    "10.1177/08901171221113836",
    "10.3390/ijerph192215409",
    "10.1186/s12889-025-22170-z",
    "10.7717/peerj.17097",
    "10.3390/ijerph20021328",
)


def _config() -> Settings:
    return Settings.from_env(
        {
            "RESEARCHMIND_DB_PATH": str(RUN_DIR / "legacy.sqlite3"),
            "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                RUN_DIR / "researchmind.sqlite3"
            ),
            "ADOLESCENT_KNOWLEDGE_BASE_DIR": str(RUN_DIR / "knowledge-base"),
            "OPENAI_API_KEY": "",
        }
    )


def _workflow(config: Settings) -> ResearchMindWorkflow:
    return ResearchMindWorkflow(
        store=ResearchMindStore(config.adolescent_topic_db_path),
        config=config,
        knowledge_base_dir=config.adolescent_topic_knowledge_base_dir,
        auto_sync_bundled=False,
        default_debate_topic="青少年体育锻炼干预与体质健康水平的关系",
    )


def _candidate_from_pool(item: dict) -> CandidatePaper:
    return CandidatePaper(
        candidate_id=f"accept-{item['pmid']}",
        title=item["title"],
        authors=tuple(item["authors"]),
        year=item["year"],
        journal=item["journal"],
        doi=item["doi"],
        source_url=item["source_url"],
        abstract=item["abstract"],
        keywords=tuple(item.get("keywords", [])),
        source="PubMed / Europe PMC",
        pdf_url="",
        is_open_access=False,
    )


def _graph_count(workflow: ResearchMindWorkflow, domain) -> int:
    papers = workflow.store.list_papers(domain.domain_id)
    analyses = {
        paper.paper_id: workflow.store.get_analysis(paper.paper_id)
        for paper in papers
    }
    evidence = {
        paper.paper_id: workflow.store.list_evidence(paper.paper_id)
        for paper in papers
    }
    knowledge_map = build_knowledge_map(
        domain_name=domain.name,
        papers=papers,
        analyses=analyses,
        evidence_by_paper=evidence,
        debate=workflow.store.latest_debate(domain.domain_id),
        synthesis=workflow.store.get_latest_synthesis(domain.domain_id),
    )
    return len(knowledge_map.papers)


def main() -> int:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    config = _config()
    workflow = _workflow(config)

    sync = sync_zotero_local_library(workflow)
    domain = sync.domain
    baseline_count = workflow.store.paper_count(domain.domain_id)
    if baseline_count != 35:
        raise AssertionError(
            f"Expected 35 Zotero papers, got {baseline_count}: {sync.message}"
        )
    baseline_state = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    baseline_graph_count = _graph_count(workflow, domain)
    baseline_debate = workflow.store.latest_debate(domain.domain_id)
    if baseline_graph_count != 35 or baseline_debate is None:
        raise AssertionError("Baseline graph/debate did not include 35 papers")

    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    pool_by_doi = {item["doi"].casefold(): item for item in pool}
    additional = [
        _candidate_from_pool(pool_by_doi[doi.casefold()])
        for doi in ADDITIONAL_DOIS
    ]
    imported = import_candidates(workflow, domain, additional)
    count_40 = workflow.store.paper_count(domain.domain_id)
    if count_40 != 40:
        raise AssertionError(
            f"Expected 40 after adding 5, got {count_40}; "
            f"imported={imported.imported}, skipped={imported.skipped_ineligible}"
        )
    stale_after_add = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    if stale_after_add.status != "待更新":
        raise AssertionError("Analysis was not invalidated after adding papers")

    state_40 = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    debate_40 = workflow.store.latest_debate(domain.domain_id)
    synthesis_40 = workflow.store.get_latest_synthesis(domain.domain_id)
    graph_40 = _graph_count(workflow, domain)
    if (
        state_40.status != "已更新"
        or graph_40 != 40
        or debate_40 is None
        or synthesis_40 is None
        or len(debate_40.considered_paper_ids) != 40
        or len(synthesis_40.source_paper_ids) != 40
    ):
        raise AssertionError("Five new papers were not included in derived analysis")

    removed_id = imported.paper_ids[-1]
    deletion = soft_delete_papers(
        workflow.store,
        domain.domain_id,
        (removed_id,),
    )
    if deletion.paper_ids != (removed_id,):
        raise AssertionError("Expected one paper to be soft-deleted")
    count_39 = workflow.store.paper_count(domain.domain_id)
    stale_after_delete = get_analysis_state(
        workflow.store,
        domain.domain_id,
        workflow.default_debate_topic,
    )
    if count_39 != 39 or stale_after_delete.status != "待更新":
        raise AssertionError("Delete did not update library version/analysis state")

    state_39 = refresh_domain_analysis_now(
        workflow,
        domain,
        workflow.default_debate_topic,
    )
    synthesis_39 = workflow.store.get_latest_synthesis(domain.domain_id)
    graph_39 = _graph_count(workflow, domain)
    if (
        state_39.status != "已更新"
        or graph_39 != 39
        or synthesis_39 is None
        or removed_id in synthesis_39.source_paper_ids
    ):
        raise AssertionError("Deleted paper remains in graph/debate output")

    restarted_workflow = _workflow(config)
    restarted_domain = next(
        item
        for item in restarted_workflow.store.list_domains()
        if item.domain_id == domain.domain_id
    )
    restarted_state = get_analysis_state(
        restarted_workflow.store,
        restarted_domain.domain_id,
        restarted_workflow.default_debate_topic,
    )
    restarted_graph_count = _graph_count(
        restarted_workflow,
        restarted_domain,
    )
    if (
        restarted_workflow.store.paper_count(restarted_domain.domain_id) != 39
        or not restarted_state.is_current
        or restarted_graph_count != 39
    ):
        raise AssertionError("Restart consistency check failed")

    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "Zotero Local API + real second-batch candidate metadata",
        "zotero_sync": {
            "active_items": sync.active_items,
            "imported": sync.imported,
            "updated": sync.updated,
            "metadata_only": sync.metadata_only,
            "pdf_parsed": sync.pdf_parsed,
            "evidence_written": sync.evidence_written,
        },
        "baseline": {
            "papers": baseline_count,
            "graph_papers": baseline_graph_count,
            "debate_considered_papers": len(
                baseline_debate.considered_paper_ids
            ),
            "status": baseline_state.status,
            "version": baseline_state.version,
        },
        "after_add_5": {
            "papers": count_40,
            "graph_papers": graph_40,
            "debate_considered_papers": len(
                debate_40.considered_paper_ids
            ),
            "synthesis_source_papers": len(
                synthesis_40.source_paper_ids
            ),
            "status": state_40.status,
            "version": state_40.version,
            "added_dois": list(ADDITIONAL_DOIS),
        },
        "after_delete_1": {
            "papers": count_39,
            "graph_papers": graph_39,
            "deleted_paper_id": removed_id,
            "deleted_paper_excluded_from_synthesis": (
                removed_id not in synthesis_39.source_paper_ids
            ),
            "status": state_39.status,
            "version": state_39.version,
        },
        "after_restart": {
            "papers": 39,
            "graph_papers": restarted_graph_count,
            "status": restarted_state.status,
            "version": restarted_state.version,
        },
        "passed": True,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
