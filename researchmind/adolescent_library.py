"""青少年专题论文库的隔离、备份和恢复。

专题库使用独立 SQLite、PDF、证据索引和 Chroma 目录，启动时不会读取或复制
旧论文库中的论文。备份以 ZIP 形式保存，恢复前会先为当前状态再建一份备份。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

from config import Settings, settings as default_settings
from researchmind.adolescent_topic import (
    CORE_DOMAIN_DESCRIPTION,
    CORE_DOMAIN_KEYWORDS,
    CORE_DOMAIN_NAME,
    TopicRelevance,
    assess_topic_relevance,
)
from researchmind.models import CandidatePaper
from researchmind.storage import ResearchMindStore
from researchmind.workflow import ResearchMindWorkflow
from tools.knowledge_base import (
    INDEX_FILE_PATH,
    build_evidence_corpus,
)
from tools.literature_framework import build_and_save_literature_framework


@dataclass(frozen=True)
class LibraryBackup:
    """一份可恢复的专题库备份。"""

    path: Path
    created_at: str
    reason: str
    size_bytes: int


@dataclass(frozen=True)
class CandidateRelevance:
    """候选论文的来源信息与本地相关度判断。"""

    candidate: CandidatePaper
    relevance: TopicRelevance


def create_adolescent_workflow(
    config: Settings | None = None,
) -> ResearchMindWorkflow:
    """创建与旧论文库完全隔离的专题工作流。"""

    active_config = config or default_settings
    db_path = active_config.adolescent_topic_db_path.resolve()
    knowledge_base_dir = (
        active_config.adolescent_topic_knowledge_base_dir.resolve()
    )
    _assert_isolated_library_paths(active_config, db_path, knowledge_base_dir)
    scaffold_topic_library(knowledge_base_dir)
    return ResearchMindWorkflow(
        store=ResearchMindStore(db_path),
        config=active_config,
        knowledge_base_dir=knowledge_base_dir,
        domain_name=CORE_DOMAIN_NAME,
        domain_description=CORE_DOMAIN_DESCRIPTION,
        domain_keywords=CORE_DOMAIN_KEYWORDS,
        auto_sync_bundled=False,
        default_debate_topic="青少年体育锻炼干预与体质健康水平的关系",
    )


def create_legacy_workflow(
    config: Settings | None = None,
) -> ResearchMindWorkflow:
    """创建旧论文库只读回看工作流，不创建或更新任何旧库数据。"""

    active_config = config or default_settings
    return ResearchMindWorkflow(
        store=ResearchMindStore(
            active_config.legacy_db_path,
            read_only=True,
        ),
        config=active_config,
        knowledge_base_dir=active_config.legacy_knowledge_base_dir,
        auto_sync_bundled=False,
        read_only=True,
    )


def scaffold_topic_library(knowledge_base_dir: Path | str) -> Path:
    """创建空的专题清单、证据索引和框架文件，已有内容永不覆盖。"""

    root = Path(knowledge_base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "pdfs").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)
    (root / "chroma").mkdir(parents=True, exist_ok=True)
    (root / "backups").mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "schema_version": 1,
            "library_name": "青少年体质健康与体育锻炼干预专题论文库",
            "description": (
                "仅保存经用户授权导入并通过本地相关性预检的论文。"
                "该库与旧论文库相互隔离。"
            ),
            "documents": [],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    index_path = root / INDEX_FILE_PATH
    framework_path = root / "framework.json"
    if index_path.exists() and framework_path.exists():
        return root

    # 空库也会生成合法的空索引；只在索引或框架缺失时构建一次。
    bundle = build_evidence_corpus(root)
    if not framework_path.exists():
        build_and_save_literature_framework(bundle, root)
    return root


def assess_candidates(
    candidates: Iterable[CandidatePaper],
) -> tuple[CandidateRelevance, ...]:
    """为检索候选补充可解释的专题相关度。"""

    results: list[CandidateRelevance] = []
    for candidate in candidates:
        relevance = assess_topic_relevance(
            title=candidate.title,
            abstract=candidate.abstract,
            keywords=candidate.keywords,
        )
        results.append(CandidateRelevance(candidate, relevance))
    return tuple(results)


def create_library_backup(
    workflow: ResearchMindWorkflow,
    *,
    reason: str,
) -> LibraryBackup:
    """备份当前专题 SQLite 和论文库文件。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safe_reason = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in reason.strip()
    ).strip("-") or "backup"
    backup_dir = workflow.knowledge_base_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{timestamp}-{safe_reason}.zip"

    with tempfile.TemporaryDirectory(prefix="tiyan-backup-") as temp_name:
        temp_root = Path(temp_name)
        db_snapshot = temp_root / "researchmind.sqlite3"
        _sqlite_backup(workflow.store.db_path, db_snapshot)
        with zipfile.ZipFile(
            backup_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.write(db_snapshot, "database/researchmind.sqlite3")
            for path in _knowledge_base_files(workflow.knowledge_base_dir):
                archive.write(
                    path,
                    (
                        PurePosixPath("knowledge_base")
                        / path.relative_to(workflow.knowledge_base_dir)
                    ).as_posix(),
                )
    return LibraryBackup(
        path=backup_path,
        created_at=timestamp,
        reason=reason,
        size_bytes=backup_path.stat().st_size,
    )


def list_library_backups(
    workflow: ResearchMindWorkflow,
) -> tuple[LibraryBackup, ...]:
    """按时间倒序返回专题库备份。"""

    backup_dir = workflow.knowledge_base_dir / "backups"
    if not backup_dir.exists():
        return ()
    backups: list[LibraryBackup] = []
    for path in sorted(
        backup_dir.glob("*.zip"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        parts = path.stem.split("-", 1)
        created_at = parts[0]
        reason = parts[1] if len(parts) > 1 else "backup"
        backups.append(
            LibraryBackup(
                path=path,
                created_at=created_at,
                reason=reason,
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(backups)


def restore_library_backup(
    workflow: ResearchMindWorkflow,
    backup_path: Path | str,
) -> LibraryBackup:
    """恢复一份专题库备份，并在覆盖前自动备份当前状态。"""

    source = Path(backup_path).resolve()
    backup_root = (workflow.knowledge_base_dir / "backups").resolve()
    if source.parent != backup_root or not source.is_file():
        raise ValueError("只能恢复专题库 backups 目录中的有效 ZIP 备份。")
    if not zipfile.is_zipfile(source):
        raise ValueError("所选文件不是有效 ZIP 备份。")

    safety_backup = create_library_backup(
        workflow,
        reason="恢复前自动备份",
    )
    current_files = _knowledge_base_files(workflow.knowledge_base_dir)
    with tempfile.TemporaryDirectory(prefix="tiyan-restore-") as temp_name:
        temp_root = Path(temp_name)
        _safe_extract(source, temp_root)
        restored_db = temp_root / "database" / "researchmind.sqlite3"
        if not restored_db.is_file():
            raise ValueError("备份中缺少 SQLite 数据库。")
        _validate_sqlite(restored_db)

        restored_kb = temp_root / "knowledge_base"
        restored_files: dict[Path, Path] = {}
        if restored_kb.is_dir():
            for path in restored_kb.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(restored_kb)
                if relative.parts and relative.parts[0] == "backups":
                    continue
                restored_files[relative] = path

        # 先清理备份快照中不存在的当前文件，使恢复结果成为完整镜像，
        # 避免删除后恢复时留下 PDF、索引片段或旧向量侧车文件。
        for path in current_files:
            relative = path.relative_to(workflow.knowledge_base_dir)
            if relative in restored_files:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue

        shutil.copy2(restored_db, workflow.store.db_path)
        for relative, path in restored_files.items():
            destination = workflow.knowledge_base_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return safety_backup


def _assert_isolated_library_paths(
    config: Settings,
    db_path: Path,
    knowledge_base_dir: Path,
) -> None:
    legacy_db = config.legacy_db_path.resolve()
    legacy_kb = config.legacy_knowledge_base_dir.resolve()
    if db_path == legacy_db:
        raise ValueError("专题 SQLite 不能与旧论文库使用同一路径。")
    if knowledge_base_dir == legacy_kb:
        raise ValueError("专题论文库目录不能与旧论文库目录相同。")
    if _is_relative_to(db_path, legacy_kb) or _is_relative_to(
        knowledge_base_dir,
        legacy_db.parent,
    ):
        raise ValueError("专题库与旧论文库路径发生交叉，已停止初始化。")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sqlite_backup(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3.Connection 的上下文管理器只负责提交事务，不负责关闭文件句柄。
    # Windows 上未关闭的目标数据库会锁定临时备份，导致备份 ZIP 收尾失败。
    with closing(sqlite3.connect(source_path)) as source:
        with closing(sqlite3.connect(destination_path)) as destination:
            source.backup(destination)
            destination.commit()


def _validate_sqlite(path: Path) -> None:
    try:
        with closing(sqlite3.connect(path)) as connection:
            result = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"备份数据库无法读取：{exc}") from exc
    if not result or str(result[0]).lower() != "ok":
        raise ValueError("备份数据库完整性检查未通过。")


def _knowledge_base_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "backups" in path.relative_to(root).parts:
            continue
        files.append(path)
    return tuple(files)


def _safe_extract(source: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not _is_relative_to(target, root):
                raise ValueError("备份 ZIP 包含不安全路径，已停止恢复。")
        archive.extractall(destination)


__all__ = [
    "CandidateRelevance",
    "LibraryBackup",
    "assess_candidates",
    "create_adolescent_workflow",
    "create_legacy_workflow",
    "create_library_backup",
    "list_library_backups",
    "restore_library_backup",
    "scaffold_topic_library",
]
