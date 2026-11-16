"""ResearchMind 的 SQLite 持久化层。

SQLite 负责保存领域、论文、证据、结构化阅读、学术分歧、知识体系、题库、
答题记录和能力掌握度。它不依赖外部服务，适合竞赛现场和初学者本地调试。
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from config import settings
from researchmind.models import (
    AssessmentQuestion,
    AttemptRecord,
    DebateAnalysis,
    EvidenceRef,
    MasteryRecord,
    PaperAnalysis,
    PaperRecord,
    ResearchDomain,
    ResearchSynthesis,
    RubricCriterion,
)


DEFAULT_DB_PATH = (
    settings.researchmind_db_path
    or settings.data_dir / "researchmind.sqlite3"
)
BASELINE_MASTERY = {
    "概念记忆": 0.0,
    "文献理解": 0.0,
    "对比分析": 0.0,
    "批判性思维": 0.0,
    "学术应用": 0.0,
}


class ResearchMindStorageError(RuntimeError):
    """研究数据库无法初始化、读取或写入时抛出的可展示错误。"""


class ResearchMindStore:
    """研究数据与学习记录的本地 SQLite 存储。"""

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.db_path.is_file():
                raise ResearchMindStorageError(
                    f"只读研究数据库不存在：{self.db_path}"
                )
            self._validate_read_only_schema()
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """创建短连接并确保提交或回滚。"""

        connection: sqlite3.Connection | None = None
        try:
            if self.read_only:
                connection = sqlite3.connect(
                    f"{self.db_path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=20,
                )
            else:
                connection = sqlite3.connect(self.db_path, timeout=20)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if not self.read_only:
                connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            if not self.read_only:
                connection.commit()
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise ResearchMindStorageError(f"研究数据库操作失败：{exc}") from exc
        finally:
            if connection is not None:
                connection.close()

    def _validate_read_only_schema(self) -> None:
        """确认只读数据库已经初始化，不创建或迁移任何表结构。"""

        required_tables = {
            "research_domains",
            "papers",
            "paper_evidence",
            "paper_analyses",
            "debates",
            "syntheses",
            "questions",
            "attempts",
            "mastery",
            "app_metadata",
        }
        try:
            with self.connection() as connection:
                rows = connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchMindStorageError(
                f"只读研究数据库无法读取：{exc}"
            ) from exc
        existing = {str(row["name"]) for row in rows}
        missing = sorted(required_tables - existing)
        if missing:
            raise ResearchMindStorageError(
                "旧研究数据库结构不完整，缺少数据表："
                + "、".join(missing)
            )

    def initialize(self) -> None:
        """创建全部数据表；重复运行不会删除已有数据。"""

        schema = """
        CREATE TABLE IF NOT EXISTS research_domains (
            domain_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            year TEXT NOT NULL,
            journal TEXT NOT NULL,
            doi TEXT NOT NULL,
            source_url TEXT NOT NULL,
            abstract TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            source TEXT NOT NULL,
            verification_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            imported_at TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT '待分类',
            parse_status TEXT NOT NULL DEFAULT '仅元数据',
            analysis_status TEXT NOT NULL DEFAULT '未分析',
            is_deleted INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT NOT NULL DEFAULT '',
            normalized_title TEXT NOT NULL DEFAULT '',
            normalization_version TEXT NOT NULL DEFAULT 'v1',
            import_origin TEXT NOT NULL DEFAULT '',
            is_manifest_managed INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_papers_domain
            ON papers(domain_id);
        CREATE INDEX IF NOT EXISTS idx_papers_doi
            ON papers(doi);

        CREATE TABLE IF NOT EXISTS paper_evidence (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            page_number INTEGER NOT NULL CHECK(page_number > 0),
            quote TEXT NOT NULL,
            section_hint TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_paper
            ON paper_evidence(paper_id);
        CREATE INDEX IF NOT EXISTS idx_evidence_page
            ON paper_evidence(paper_id, page_number);

        CREATE TABLE IF NOT EXISTS paper_analyses (
            paper_id TEXT PRIMARY KEY,
            purpose_json TEXT NOT NULL,
            population_json TEXT NOT NULL,
            methods_json TEXT NOT NULL,
            interventions_json TEXT NOT NULL DEFAULT '[]',
            comparators_json TEXT NOT NULL DEFAULT '[]',
            measures_json TEXT NOT NULL DEFAULT '[]',
            results_json TEXT NOT NULL,
            statistics_json TEXT NOT NULL DEFAULT '[]',
            conclusions_json TEXT NOT NULL,
            limitations_json TEXT NOT NULL,
            extraction_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS debates (
            debate_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_valid INTEGER NOT NULL DEFAULT 1,
            invalidated_at TEXT NOT NULL DEFAULT '',
            invalidated_paper_ids_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_debates_domain
            ON debates(domain_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS syntheses (
            synthesis_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_valid INTEGER NOT NULL DEFAULT 1,
            invalidated_at TEXT NOT NULL DEFAULT '',
            invalidated_paper_ids_json TEXT NOT NULL DEFAULT '[]',
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_syntheses_domain
            ON syntheses(domain_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS questions (
            question_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            question_type TEXT NOT NULL,
            prompt TEXT NOT NULL,
            options_json TEXT NOT NULL,
            standard_answer TEXT NOT NULL,
            rubric_json TEXT NOT NULL,
            knowledge_tags_json TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            source_paper_ids_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            scoring_mode TEXT NOT NULL,
            explanation TEXT NOT NULL DEFAULT '',
            assessment_module TEXT NOT NULL DEFAULT '基础知识理解',
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_questions_domain
            ON questions(domain_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            question_id TEXT NOT NULL,
            user_answer TEXT NOT NULL,
            score REAL NOT NULL,
            max_score REAL NOT NULL,
            feedback TEXT NOT NULL,
            mode TEXT NOT NULL,
            knowledge_tags_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id),
            FOREIGN KEY (question_id) REFERENCES questions(question_id)
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_domain
            ON attempts(domain_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_attempts_question
            ON attempts(question_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS mastery (
            domain_id TEXT NOT NULL,
            dimension TEXT NOT NULL,
            score REAL NOT NULL,
            attempts_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (domain_id, dimension),
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );

        CREATE TABLE IF NOT EXISTS app_metadata (
            metadata_key TEXT PRIMARY KEY,
            metadata_value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_tombstones (
            tombstone_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            domain_id TEXT NOT NULL,
            title TEXT NOT NULL,
            authors_json TEXT NOT NULL,
            year TEXT NOT NULL,
            journal TEXT NOT NULL,
            doi TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            deleted_at TEXT NOT NULL,
            reason TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tombstones_domain
            ON paper_tombstones(domain_id, deleted_at DESC);
        """
        try:
            with self.connection() as connection:
                connection.executescript(schema)
                self._migrate_papers_schema(connection)
                self._migrate_derivation_schema(connection)
                self._migrate_analysis_schema(connection)
                self._migrate_question_schema(connection)
        except ResearchMindStorageError:
            raise
        except sqlite3.Error as exc:
            raise ResearchMindStorageError(
                f"无法初始化研究数据库：{exc}"
            ) from exc

    def _migrate_papers_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """为旧 papers 表补充新字段；只新增，不删除或覆盖已有数据。"""

        existing = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(papers)"
            ).fetchall()
        }
        additions = (
            ("imported_at", "TEXT NOT NULL DEFAULT ''"),
            ("category", "TEXT NOT NULL DEFAULT '待分类'"),
            ("parse_status", "TEXT NOT NULL DEFAULT '仅元数据'"),
            ("analysis_status", "TEXT NOT NULL DEFAULT '未分析'"),
            ("is_deleted", "INTEGER NOT NULL DEFAULT 0"),
            ("deleted_at", "TEXT NOT NULL DEFAULT ''"),
            ("normalized_title", "TEXT NOT NULL DEFAULT ''"),
            ("normalization_version", "TEXT NOT NULL DEFAULT 'v1'"),
            ("import_origin", "TEXT NOT NULL DEFAULT ''"),
            (
                "is_manifest_managed",
                "INTEGER NOT NULL DEFAULT 1",
            ),
        )
        for column, definition in additions:
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE papers ADD COLUMN {column} {definition}"
                )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_domain_deleted
            ON papers(domain_id, is_deleted)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_papers_normalized_title
            ON papers(domain_id, normalized_title)
            """
        )
        connection.execute(
            """
            UPDATE papers
            SET imported_at = created_at
            WHERE trim(imported_at) = ''
            """
        )
        rows = connection.execute(
            """
            SELECT paper_id, title
            FROM papers
            WHERE trim(normalized_title) = ''
            """
        ).fetchall()
        connection.executemany(
            """
            UPDATE papers
            SET normalized_title = ?, normalization_version = 'v1'
            WHERE paper_id = ?
            """,
            (
                (_normalize_title(str(row["title"])), str(row["paper_id"]))
                for row in rows
            ),
        )

    def _migrate_derivation_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """为历史分析增加失效标记，不删除已有报告。"""

        for table_name in ("debates", "syntheses"):
            existing = {
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
            additions = (
                ("is_valid", "INTEGER NOT NULL DEFAULT 1"),
                ("invalidated_at", "TEXT NOT NULL DEFAULT ''"),
                (
                    "invalidated_paper_ids_json",
                    "TEXT NOT NULL DEFAULT '[]'",
                ),
            )
            for column, definition in additions:
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN {column} {definition}"
                    )

    def _migrate_analysis_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """为论文结构化阅读增加证据字段；仅新增，不重写历史结果。"""

        existing = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(paper_analyses)"
            ).fetchall()
        }
        additions = (
            ("interventions_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("comparators_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("measures_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("statistics_json", "TEXT NOT NULL DEFAULT '[]'"),
        )
        for column, definition in additions:
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE paper_analyses "
                    f"ADD COLUMN {column} {definition}"
                )

    def _migrate_question_schema(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """为历史题目补充解析字段，不重写旧题内容。"""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(questions)"
            ).fetchall()
        }
        if "explanation" not in columns:
            connection.execute(
                "ALTER TABLE questions "
                "ADD COLUMN explanation TEXT NOT NULL DEFAULT ''"
            )
        if "assessment_module" not in columns:
            connection.execute(
                "ALTER TABLE questions "
                "ADD COLUMN assessment_module TEXT NOT NULL "
                "DEFAULT '基础知识理解'"
            )

    def upsert_domain(self, domain: ResearchDomain) -> None:
        """新增或更新研究领域。"""

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO research_domains (
                    domain_id, name, description, keywords_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(domain_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    keywords_json = excluded.keywords_json
                """,
                (
                    domain.domain_id,
                    domain.name,
                    domain.description,
                    _json(domain.keywords),
                    domain.created_at,
                ),
            )
            self._ensure_mastery_rows(connection, domain.domain_id)

    def list_domains(self) -> tuple[ResearchDomain, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM research_domains ORDER BY created_at, name"
            ).fetchall()
        return tuple(_row_to_domain(row) for row in rows)

    def get_domain(self, domain_id: str) -> ResearchDomain | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM research_domains WHERE domain_id = ?",
                (domain_id,),
            ).fetchone()
        return _row_to_domain(row) if row else None

    def get_default_domain(self) -> ResearchDomain | None:
        domains = self.list_domains()
        return domains[0] if domains else None

    def upsert_paper(self, paper: PaperRecord) -> None:
        """按论文 ID 更新元数据，不覆盖证据或意外恢复软删除论文。"""

        normalized_title = (
            paper.normalized_title.strip()
            or _normalize_title(paper.title)
        )
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    paper_id, domain_id, title, authors_json, year, journal,
                    doi, source_url, abstract, keywords_json, pdf_path,
                    source, verification_status, created_at, imported_at,
                    category, parse_status, analysis_status, is_deleted,
                    deleted_at, normalized_title, normalization_version,
                    import_origin, is_manifest_managed
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(paper_id) DO UPDATE SET
                    domain_id = excluded.domain_id,
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    year = excluded.year,
                    journal = excluded.journal,
                    doi = excluded.doi,
                    source_url = excluded.source_url,
                    abstract = excluded.abstract,
                    keywords_json = excluded.keywords_json,
                    pdf_path = excluded.pdf_path,
                    source = excluded.source,
                    verification_status = excluded.verification_status,
                    imported_at = CASE
                        WHEN trim(papers.imported_at) = ''
                        THEN excluded.imported_at
                        ELSE papers.imported_at
                    END,
                    category = CASE
                        WHEN trim(excluded.category) NOT IN ('', '待分类')
                        THEN excluded.category
                        ELSE papers.category
                    END,
                    parse_status = CASE
                        WHEN papers.parse_status = '已解析' THEN '已解析'
                        WHEN trim(excluded.parse_status) != ''
                        THEN excluded.parse_status
                        ELSE papers.parse_status
                    END,
                    analysis_status = CASE
                        WHEN papers.analysis_status = '已分析' THEN '已分析'
                        WHEN trim(excluded.analysis_status) NOT IN ('', '未分析')
                        THEN excluded.analysis_status
                        ELSE papers.analysis_status
                    END,
                    is_deleted = CASE
                        WHEN papers.is_deleted = 1 THEN 1
                        ELSE excluded.is_deleted
                    END,
                    deleted_at = CASE
                        WHEN papers.is_deleted = 1 THEN papers.deleted_at
                        ELSE excluded.deleted_at
                    END,
                    normalized_title = excluded.normalized_title,
                    normalization_version = excluded.normalization_version,
                    import_origin = CASE
                        WHEN excluded.import_origin != ''
                        THEN excluded.import_origin
                        ELSE papers.import_origin
                    END,
                    is_manifest_managed = MAX(
                        papers.is_manifest_managed,
                        excluded.is_manifest_managed
                    )
                """,
                (
                    paper.paper_id,
                    paper.domain_id,
                    paper.title,
                    _json(paper.authors),
                    paper.year,
                    paper.journal,
                    paper.doi,
                    paper.source_url,
                    paper.abstract,
                    _json(paper.keywords),
                    paper.pdf_path,
                    paper.source,
                    paper.verification_status,
                    paper.created_at,
                    paper.imported_at or paper.created_at,
                    paper.category or "待分类",
                    paper.parse_status or "仅元数据",
                    paper.analysis_status or "未分析",
                    int(paper.is_deleted),
                    paper.deleted_at,
                    normalized_title,
                    paper.normalization_version or "v1",
                    paper.import_origin,
                    int(paper.is_manifest_managed),
                ),
            )
            self._invalidate_analysis_with_connection(
                connection,
                paper.domain_id,
                (paper.paper_id,),
                reason="论文元数据发生变化",
            )

    def list_papers(
        self,
        domain_id: str,
        *,
        include_deleted: bool = False,
    ) -> tuple[PaperRecord, ...]:
        deleted_clause = self._paper_deleted_clause(
            include_deleted=include_deleted,
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM papers
                WHERE domain_id = ?
                  {deleted_clause}
                ORDER BY year DESC, title
                """,
                (domain_id,),
            ).fetchall()
        return tuple(_row_to_paper(row) for row in rows)

    def list_deleted_papers(
        self,
        domain_id: str,
    ) -> tuple[PaperRecord, ...]:
        """返回回收站论文，按删除时间倒序。"""

        if not self._has_paper_column("is_deleted"):
            return ()
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM papers
                WHERE domain_id = ? AND is_deleted = 1
                ORDER BY deleted_at DESC, title
                """,
                (domain_id,),
            ).fetchall()
        return tuple(_row_to_paper(row) for row in rows)

    def get_paper(
        self,
        paper_id: str,
        *,
        include_deleted: bool = False,
    ) -> PaperRecord | None:
        deleted_clause = self._paper_deleted_clause(
            include_deleted=include_deleted,
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM papers
                WHERE paper_id = ? {deleted_clause}
                """,
                (paper_id,),
            ).fetchone()
        return _row_to_paper(row) if row else None

    def paper_count(
        self,
        domain_id: str,
        *,
        include_deleted: bool = False,
    ) -> int:
        deleted_clause = self._paper_deleted_clause(
            include_deleted=include_deleted,
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM papers
                WHERE domain_id = ? {deleted_clause}
                """,
                (domain_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def manifest_paper_count(
        self,
        domain_id: str,
        *,
        include_deleted: bool = True,
    ) -> int:
        """统计由项目清单管理的论文，用于判断同步缓存是否有效。"""

        if not self._has_paper_column("is_manifest_managed"):
            return self.paper_count(
                domain_id,
                include_deleted=include_deleted,
            )
        deleted_clause = self._paper_deleted_clause(
            include_deleted=include_deleted,
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count FROM papers
                WHERE domain_id = ?
                  AND is_manifest_managed = 1
                  {deleted_clause}
                """,
                (domain_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def find_paper_by_identifier(
        self,
        domain_id: str,
        *,
        doi: str = "",
        title: str = "",
        include_deleted: bool = True,
    ) -> PaperRecord | None:
        """按规范化 DOI 优先、题名其次查找同领域论文。"""

        target_doi = _normalize_doi(doi)
        target_title = _normalize_title(title)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM papers WHERE domain_id = ?",
                (domain_id,),
            ).fetchall()
        papers = tuple(_row_to_paper(row) for row in rows)
        if not include_deleted:
            papers = tuple(item for item in papers if not item.is_deleted)
        if target_doi:
            for paper in papers:
                if _normalize_doi(paper.doi) == target_doi:
                    return paper
        if target_title:
            for paper in papers:
                stored = (
                    paper.normalized_title
                    or _normalize_title(paper.title)
                )
                if stored == target_title:
                    return paper
        return None

    def delete_papers_except(
        self,
        domain_id: str,
        keep_paper_ids: Iterable[str],
    ) -> int:
        """物理清理已离开 manifest 的论文，不触碰检索导入论文。

        该函数只服务项目内置论文清单同步。用户从检索页导入的纯元数据论文
        必须保留，因此仅删除 ``is_manifest_managed = 1`` 的过期记录。
        """

        keep_ids = {str(paper_id) for paper_id in keep_paper_ids}
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT paper_id FROM papers
                WHERE domain_id = ? AND is_manifest_managed = 1
                """,
                (domain_id,),
            ).fetchall()
            stale_ids = [
                str(row["paper_id"])
                for row in rows
                if str(row["paper_id"]) not in keep_ids
            ]
            if not stale_ids:
                return 0
            connection.executemany(
                "DELETE FROM papers WHERE paper_id = ?",
                ((paper_id,) for paper_id in stale_ids),
            )
            self._invalidate_analysis_with_connection(
                connection,
                domain_id,
                stale_ids,
                reason="论文清单移除了论文",
            )
        return len(stale_ids)

    def soft_delete_papers(
        self,
        domain_id: str,
        paper_ids: Iterable[str],
        *,
        deleted_at: str | None = None,
    ) -> tuple[str, ...]:
        """将论文移入回收站，保留元数据、证据和 PDF 以便恢复。"""

        ids = tuple(dict.fromkeys(str(item) for item in paper_ids if item))
        if not ids:
            return ()
        timestamp = deleted_at or _now()
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE papers
                SET is_deleted = 1, deleted_at = ?
                WHERE domain_id = ?
                  AND paper_id IN ({placeholders})
                  AND is_deleted = 0
                """,
                (timestamp, domain_id, *ids),
            )
            if cursor.rowcount <= 0:
                return ()
            rows = connection.execute(
                f"""
                SELECT paper_id FROM papers
                WHERE domain_id = ?
                  AND paper_id IN ({placeholders})
                  AND is_deleted = 1
                  AND deleted_at = ?
                """,
                (domain_id, *ids, timestamp),
            ).fetchall()
            self._invalidate_analysis_with_connection(
                connection,
                domain_id,
                (str(row["paper_id"]) for row in rows),
                reason="论文移入回收站",
            )
        return tuple(str(row["paper_id"]) for row in rows)

    def restore_papers(
        self,
        domain_id: str,
        paper_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """从回收站恢复论文。"""

        ids = tuple(dict.fromkeys(str(item) for item in paper_ids if item))
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            cursor = connection.execute(
                f"""
                UPDATE papers
                SET is_deleted = 0, deleted_at = ''
                WHERE domain_id = ?
                  AND paper_id IN ({placeholders})
                  AND is_deleted = 1
                """,
                (domain_id, *ids),
            )
            if cursor.rowcount <= 0:
                return ()
            rows = connection.execute(
                f"""
                SELECT paper_id FROM papers
                WHERE domain_id = ?
                  AND paper_id IN ({placeholders})
                  AND is_deleted = 0
                """,
                (domain_id, *ids),
            ).fetchall()
            self._invalidate_analysis_with_connection(
                connection,
                domain_id,
                (str(row["paper_id"]) for row in rows),
                reason="论文从回收站恢复",
            )
        return tuple(str(row["paper_id"]) for row in rows)

    def permanently_delete_papers(
        self,
        paper_ids: Iterable[str],
    ) -> tuple[str, ...]:
        """永久删除数据库记录；调用方必须先完成确认和备份。"""

        ids = tuple(dict.fromkeys(str(item) for item in paper_ids if item))
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            existing_rows = connection.execute(
                f"""
                SELECT paper_id, domain_id FROM papers
                WHERE paper_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            existing_ids = tuple(
                str(row["paper_id"]) for row in existing_rows
            )
            if not existing_ids:
                return ()
            domain_ids = {
                str(row["domain_id"]) for row in existing_rows
            }
            existing_placeholders = ",".join("?" for _ in existing_ids)
            connection.execute(
                f"""
                DELETE FROM papers
                WHERE paper_id IN ({existing_placeholders})
                """,
                existing_ids,
            )
            for domain_id in domain_ids:
                self._invalidate_analysis_with_connection(
                    connection,
                    domain_id,
                    existing_ids,
                    reason="论文被永久删除",
                )
        return existing_ids

    def save_tombstones(
        self,
        papers: Iterable[PaperRecord],
        *,
        reason: str,
        deleted_at: str | None = None,
    ) -> int:
        """永久删除前保存最小来源审计信息。"""

        timestamp = deleted_at or _now()
        items = tuple(papers)
        if not items:
            return 0
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO paper_tombstones (
                    tombstone_id, paper_id, domain_id, title, authors_json,
                    year, journal, doi, source, source_url, pdf_path,
                    deleted_at, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        _stable_id("tombstone", paper.paper_id, timestamp),
                        paper.paper_id,
                        paper.domain_id,
                        paper.title,
                        _json(paper.authors),
                        paper.year,
                        paper.journal,
                        paper.doi,
                        paper.source,
                        paper.source_url,
                        paper.pdf_path,
                        timestamp,
                        reason,
                    )
                    for paper in items
                ),
            )
        return len(items)

    def update_paper_status(
        self,
        paper_id: str,
        *,
        parse_status: str | None = None,
        analysis_status: str | None = None,
    ) -> None:
        """单独更新流程状态，避免 manifest 同步覆盖已有分析状态。"""

        updates: list[str] = []
        parameters: list[Any] = []
        if parse_status is not None:
            updates.append("parse_status = ?")
            parameters.append(str(parse_status).strip() or "仅元数据")
        if analysis_status is not None:
            updates.append("analysis_status = ?")
            parameters.append(str(analysis_status).strip() or "未分析")
        if not updates:
            return
        parameters.append(paper_id)
        with self.connection() as connection:
            connection.execute(
                f"""
                UPDATE papers SET {", ".join(updates)}
                WHERE paper_id = ?
                """,
                parameters,
            )

    def invalidate_domain_derivations(
        self,
        domain_id: str,
        deleted_paper_ids: Iterable[str],
    ) -> dict[str, int]:
        """标记领域分析失效，并清理明确依赖已删除论文的试题。

        论文库变化后，旧的观点比较和知识体系可能继续引用已删除论文。
        历史报告保留在数据库中供审计，但会被标记为失效，默认不再作为当前
        结论展示。题库只删除明确引用被删论文的题目，无关题目和其余学习
        记录继续保留。
        """

        deleted_ids = {
            str(paper_id)
            for paper_id in deleted_paper_ids
            if str(paper_id).strip()
        }
        if not deleted_ids:
            return {
                "debates": 0,
                "syntheses": 0,
                "questions": 0,
                "attempts": 0,
            }

        with self.connection() as connection:
            question_rows = connection.execute(
                """
                SELECT question_id, source_paper_ids_json
                FROM questions
                WHERE domain_id = ?
                """,
                (domain_id,),
            ).fetchall()
            affected_question_ids = []
            for row in question_rows:
                source_ids = set(_loads(str(row["source_paper_ids_json"]), []))
                if source_ids & deleted_ids:
                    affected_question_ids.append(str(row["question_id"]))

            deleted_attempts = 0
            if affected_question_ids:
                placeholders = ",".join("?" for _ in affected_question_ids)
                attempt_cursor = connection.execute(
                    f"""
                    DELETE FROM attempts
                    WHERE domain_id = ?
                      AND question_id IN ({placeholders})
                    """,
                    (domain_id, *affected_question_ids),
                )
                deleted_attempts = max(0, attempt_cursor.rowcount)

            deleted_questions = 0
            if affected_question_ids:
                placeholders = ",".join("?" for _ in affected_question_ids)
                question_cursor = connection.execute(
                    f"""
                    DELETE FROM questions
                    WHERE domain_id = ?
                      AND question_id IN ({placeholders})
                    """,
                    (domain_id, *affected_question_ids),
                )
                deleted_questions = max(0, question_cursor.rowcount)

            invalidated_at = _now()
            invalidated_debates = 0
            invalidated_syntheses = 0
            for table_name in ("debates", "syntheses"):
                rows = connection.execute(
                    f"""
                    SELECT rowid, invalidated_paper_ids_json
                    FROM {table_name}
                    WHERE domain_id = ? AND is_valid = 1
                    """,
                    (domain_id,),
                ).fetchall()
                for row in rows:
                    previous = {
                        str(item)
                        for item in _loads(
                            str(row["invalidated_paper_ids_json"]),
                            [],
                        )
                    }
                    connection.execute(
                        f"""
                        UPDATE {table_name}
                        SET is_valid = 0,
                            invalidated_at = ?,
                            invalidated_paper_ids_json = ?
                        WHERE rowid = ?
                        """,
                        (
                            invalidated_at,
                            _json(sorted(previous | deleted_ids)),
                            int(row["rowid"]),
                        ),
                    )
                if table_name == "debates":
                    invalidated_debates = len(rows)
                else:
                    invalidated_syntheses = len(rows)

        self.mark_domain_analysis_stale(
            domain_id,
            reason="论文删除导致派生分析失效",
            affected_paper_ids=deleted_ids,
        )
        return {
            "debates": invalidated_debates,
            "syntheses": invalidated_syntheses,
            "questions": deleted_questions,
            "attempts": deleted_attempts,
        }

    def replace_evidence(
        self,
        paper_id: str,
        evidence: tuple[EvidenceRef, ...],
    ) -> None:
        """以一次事务替换一篇论文的证据，避免新旧页码混杂。"""

        with self.connection() as connection:
            connection.execute(
                "DELETE FROM paper_evidence WHERE paper_id = ?",
                (paper_id,),
            )
            connection.executemany(
                """
                INSERT INTO paper_evidence (
                    chunk_id, paper_id, page_number, quote, section_hint
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        item.chunk_id,
                        paper_id,
                        item.page_number,
                        item.quote,
                        _section_hint(item.quote),
                    )
                    for item in evidence
                ),
            )
            paper_row = connection.execute(
                "SELECT domain_id FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if paper_row:
                self._invalidate_analysis_with_connection(
                    connection,
                    str(paper_row["domain_id"]),
                    (paper_id,),
                    reason="论文页级证据发生变化",
                )

    def list_evidence(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> tuple[EvidenceRef, ...]:
        deleted_clause = self._paper_deleted_clause(
            alias="p",
            include_deleted=include_deleted,
        )
        sql = """
            SELECT e.*, p.title AS document_title, p.pdf_path
            FROM paper_evidence e
            JOIN papers p ON p.paper_id = e.paper_id
            WHERE e.paper_id = ? {deleted_clause}
            ORDER BY e.page_number, e.chunk_id
        """.format(deleted_clause=deleted_clause)
        parameters: list[Any] = [paper_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, int(limit)))
        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_row_to_evidence(row) for row in rows)

    def list_domain_evidence(
        self,
        domain_id: str,
        *,
        limit: int | None = None,
        include_deleted: bool = False,
    ) -> tuple[EvidenceRef, ...]:
        deleted_clause = self._paper_deleted_clause(
            alias="p",
            include_deleted=include_deleted,
        )
        sql = """
            SELECT e.*, p.title AS document_title, p.pdf_path
            FROM paper_evidence e
            JOIN papers p ON p.paper_id = e.paper_id
            WHERE p.domain_id = ? {deleted_clause}
            ORDER BY p.year DESC, e.page_number, e.chunk_id
        """.format(deleted_clause=deleted_clause)
        parameters: list[Any] = [domain_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, int(limit)))
        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_row_to_evidence(row) for row in rows)

    def evidence_count(
        self,
        domain_id: str,
        *,
        include_deleted: bool = False,
    ) -> int:
        deleted_clause = self._paper_deleted_clause(
            alias="p",
            include_deleted=include_deleted,
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*) AS count
                FROM paper_evidence e
                JOIN papers p ON p.paper_id = e.paper_id
                WHERE p.domain_id = ? {deleted_clause}
                """,
                (domain_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def save_analysis(self, analysis: PaperAnalysis) -> None:
        with self.connection() as connection:
            evidence_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM paper_evidence
                WHERE paper_id = ?
                """,
                (analysis.paper_id,),
            ).fetchone()
            has_full_text = bool(evidence_row and int(evidence_row["count"]) > 0)
            has_abstract_evidence = any(
                item.source_type == "摘要"
                for item in analysis.all_evidence
            )
            parse_status = (
                "已解析"
                if has_full_text
                else "摘要已解析"
                if has_abstract_evidence
                else "仅元数据"
            )
            analysis_status = (
                "已分析"
                if has_full_text
                else "基于摘要已分析"
                if has_abstract_evidence
                else "待补充全文"
            )
            connection.execute(
                """
                INSERT INTO paper_analyses (
                    paper_id, purpose_json, population_json, methods_json,
                    interventions_json, comparators_json, measures_json,
                    results_json, statistics_json, conclusions_json,
                    limitations_json, extraction_mode, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    purpose_json = excluded.purpose_json,
                    population_json = excluded.population_json,
                    methods_json = excluded.methods_json,
                    interventions_json = excluded.interventions_json,
                    comparators_json = excluded.comparators_json,
                    measures_json = excluded.measures_json,
                    results_json = excluded.results_json,
                    statistics_json = excluded.statistics_json,
                    conclusions_json = excluded.conclusions_json,
                    limitations_json = excluded.limitations_json,
                    extraction_mode = excluded.extraction_mode,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    analysis.paper_id,
                    _json_evidence(analysis.purpose),
                    _json_evidence(analysis.population),
                    _json_evidence(analysis.methods),
                    _json_evidence(analysis.interventions),
                    _json_evidence(analysis.comparators),
                    _json_evidence(analysis.measures),
                    _json_evidence(analysis.results),
                    _json_evidence(analysis.statistics),
                    _json_evidence(analysis.conclusions),
                    _json_evidence(analysis.limitations),
                    analysis.extraction_mode,
                    analysis.status,
                    _now(),
                ),
            )
            connection.execute(
                """
                UPDATE papers
                SET parse_status = ?,
                    analysis_status = ?
                WHERE paper_id = ?
                """,
                (
                    parse_status,
                    analysis_status,
                    analysis.paper_id,
                ),
            )

    def get_analysis(self, paper_id: str) -> PaperAnalysis | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM paper_analyses WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        if row is None:
            return None
        return PaperAnalysis(
            paper_id=str(row["paper_id"]),
            purpose=_evidence_from_json(row["purpose_json"]),
            population=_evidence_from_json(row["population_json"]),
            methods=_evidence_from_json(row["methods_json"]),
            interventions=_evidence_from_json(
                row["interventions_json"]
                if "interventions_json" in row.keys()
                else "[]"
            ),
            comparators=_evidence_from_json(
                row["comparators_json"]
                if "comparators_json" in row.keys()
                else "[]"
            ),
            measures=_evidence_from_json(
                row["measures_json"]
                if "measures_json" in row.keys()
                else "[]"
            ),
            results=_evidence_from_json(row["results_json"]),
            statistics=_evidence_from_json(
                row["statistics_json"]
                if "statistics_json" in row.keys()
                else "[]"
            ),
            conclusions=_evidence_from_json(row["conclusions_json"]),
            limitations=_evidence_from_json(row["limitations_json"]),
            extraction_mode=str(row["extraction_mode"]),
            status=str(row["status"]),
        )

    def save_debate(self, domain_id: str, debate: DebateAnalysis) -> str:
        debate_id = _stable_id(
            "debate", domain_id, debate.topic, _now()
        )
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO debates (
                    debate_id, domain_id, topic, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    debate_id,
                    domain_id,
                    debate.topic,
                    _json(debate.to_dict()),
                    _now(),
                ),
            )
        return debate_id

    def list_debates(self, domain_id: str) -> tuple[DebateAnalysis, ...]:
        """返回当前仍有效的观点比较结果。"""

        valid_clause = (
            "AND is_valid = 1"
            if self._has_table_column("debates", "is_valid")
            else ""
        )
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT payload_json FROM debates
                WHERE domain_id = ? {valid_clause}
                ORDER BY created_at DESC
                """,
                (domain_id,),
            ).fetchall()
        return tuple(_debate_from_json(row["payload_json"]) for row in rows)

    def latest_debate(self, domain_id: str) -> DebateAnalysis | None:
        debates = self.list_debates(domain_id)
        return debates[0] if debates else None

    def save_synthesis(
        self,
        synthesis: ResearchSynthesis,
    ) -> str:
        synthesis_id = _stable_id(
            "synthesis", synthesis.domain_id, synthesis.generated_at
        )
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO syntheses (
                    synthesis_id, domain_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    synthesis_id,
                    synthesis.domain_id,
                    _json(synthesis.to_dict()),
                    synthesis.generated_at,
                ),
            )
        return synthesis_id

    def get_latest_synthesis(
        self,
        domain_id: str,
    ) -> ResearchSynthesis | None:
        valid_clause = (
            "AND is_valid = 1"
            if self._has_table_column("syntheses", "is_valid")
            else ""
        )
        with self.connection() as connection:
            row = connection.execute(
                f"""
                SELECT payload_json FROM syntheses
                WHERE domain_id = ? {valid_clause}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (domain_id,),
            ).fetchone()
        return _synthesis_from_json(row["payload_json"]) if row else None

    def save_question(self, question: AssessmentQuestion) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO questions (
                    question_id, domain_id, question_type, prompt,
                    options_json, standard_answer, rubric_json,
                    knowledge_tags_json, difficulty,
                    source_paper_ids_json, evidence_json, created_at,
                    scoring_mode, explanation, assessment_module
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question.question_id,
                    question.domain_id,
                    question.question_type,
                    question.prompt,
                    _json(question.options),
                    question.standard_answer,
                    _json(
                        [
                            {
                                "label": item.label,
                                "description": item.description,
                                "points": item.points,
                                "expected_keywords": list(
                                    item.expected_keywords
                                ),
                            }
                            for item in question.rubric
                        ]
                    ),
                    _json(question.knowledge_tags),
                    question.difficulty,
                    _json(question.source_paper_ids),
                    _json_evidence(question.evidence),
                    question.created_at,
                    question.scoring_mode,
                    question.explanation,
                    question.assessment_module,
                ),
            )

    def save_questions(
        self,
        questions: tuple[AssessmentQuestion, ...],
    ) -> None:
        for question in questions:
            self.save_question(question)

    def list_questions(
        self,
        domain_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[AssessmentQuestion, ...]:
        sql = """
            SELECT * FROM questions
            WHERE domain_id = ?
            ORDER BY created_at DESC, question_id
        """
        parameters: list[Any] = [domain_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, int(limit)))
        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_row_to_question(row) for row in rows)

    def get_question(self, question_id: str) -> AssessmentQuestion | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM questions WHERE question_id = ?",
                (question_id,),
            ).fetchone()
        return _row_to_question(row) if row else None

    def record_attempt(
        self,
        *,
        domain_id: str,
        question: AssessmentQuestion,
        user_answer: str,
        score: float,
        feedback: str,
        mode: str,
        created_at: str | None = None,
    ) -> AttemptRecord:
        timestamp = created_at or _now()
        attempt_id = _stable_id(
            "attempt",
            domain_id,
            question.question_id,
            timestamp,
            user_answer,
        )
        safe_score = max(0.0, min(float(score), question.max_score))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO attempts (
                    attempt_id, domain_id, question_id, user_answer, score,
                    max_score, feedback, mode, knowledge_tags_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    domain_id,
                    question.question_id,
                    user_answer,
                    safe_score,
                    question.max_score,
                    feedback,
                    mode,
                    _json(question.knowledge_tags),
                    timestamp,
                ),
            )
        return AttemptRecord(
            attempt_id=attempt_id,
            domain_id=domain_id,
            question_id=question.question_id,
            user_answer=user_answer,
            score=safe_score,
            max_score=question.max_score,
            feedback=feedback,
            mode=mode,
            knowledge_tags=question.knowledge_tags,
            created_at=timestamp,
            question_type=question.question_type,
            prompt=question.prompt,
        )

    def list_attempts(
        self,
        domain_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[AttemptRecord, ...]:
        sql = """
            SELECT a.*, q.question_type, q.prompt
            FROM attempts a
            JOIN questions q ON q.question_id = a.question_id
            WHERE a.domain_id = ?
            ORDER BY a.created_at DESC
        """
        parameters: list[Any] = [domain_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, int(limit)))
        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_row_to_attempt(row) for row in rows)

    def list_wrong_attempts(
        self,
        domain_id: str,
    ) -> tuple[AttemptRecord, ...]:
        """返回得分低于满分 60% 的题目，作为错题本。"""

        attempts = self.list_attempts(domain_id)
        return tuple(
            attempt
            for attempt in attempts
            if attempt.max_score > 0
            and attempt.score / attempt.max_score < 0.6
        )

    def ensure_mastery(self, domain_id: str) -> tuple[MasteryRecord, ...]:
        with self.connection() as connection:
            self._ensure_mastery_rows(connection, domain_id)
            rows = connection.execute(
                """
                SELECT * FROM mastery
                WHERE domain_id = ?
                ORDER BY rowid
                """,
                (domain_id,),
            ).fetchall()
        return tuple(_row_to_mastery(row) for row in rows)

    def set_mastery(
        self,
        domain_id: str,
        dimension: str,
        score: float,
        attempts_count: int,
    ) -> None:
        if dimension not in BASELINE_MASTERY:
            raise ValueError(f"未知能力维度：{dimension}")
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO mastery (
                    domain_id, dimension, score, attempts_count, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(domain_id, dimension) DO UPDATE SET
                    score = excluded.score,
                    attempts_count = excluded.attempts_count,
                    updated_at = excluded.updated_at
                """,
                (
                    domain_id,
                    dimension,
                    max(0.0, min(100.0, float(score))),
                    max(0, int(attempts_count)),
                    _now(),
                ),
            )

    def get_metadata(self, key: str) -> str | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT metadata_value FROM app_metadata WHERE metadata_key = ?",
                (key,),
            ).fetchone()
        return str(row["metadata_value"]) if row else None

    def set_metadata(self, key: str, value: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO app_metadata (
                    metadata_key, metadata_value, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(metadata_key) DO UPDATE SET
                    metadata_value = excluded.metadata_value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _now()),
            )

    def set_metadata_many(self, values: dict[str, str]) -> None:
        """在一个事务中原子更新多项运行状态。"""

        if not values:
            return
        timestamp = _now()
        with self.connection() as connection:
            connection.executemany(
                """
                INSERT INTO app_metadata (
                    metadata_key, metadata_value, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(metadata_key) DO UPDATE SET
                    metadata_value = excluded.metadata_value,
                    updated_at = excluded.updated_at
                """,
                (
                    (key, value, timestamp)
                    for key, value in values.items()
                ),
            )

    def mark_domain_analysis_stale(
        self,
        domain_id: str,
        *,
        reason: str,
        affected_paper_ids: Iterable[str] = (),
    ) -> None:
        """显式标记领域图谱和分歧为待更新。"""

        with self.connection() as connection:
            self._invalidate_analysis_with_connection(
                connection,
                domain_id,
                affected_paper_ids,
                reason=reason,
            )

    def _invalidate_analysis_with_connection(
        self,
        connection: sqlite3.Connection,
        domain_id: str,
        affected_paper_ids: Iterable[str],
        *,
        reason: str,
    ) -> None:
        """在论文写事务内使全部派生分析失效。"""

        affected = {
            str(paper_id)
            for paper_id in affected_paper_ids
            if str(paper_id).strip()
        }
        timestamp = _now()
        for table_name in ("debates", "syntheses"):
            rows = connection.execute(
                f"""
                SELECT rowid, invalidated_paper_ids_json
                FROM {table_name}
                WHERE domain_id = ? AND is_valid = 1
                """,
                (domain_id,),
            ).fetchall()
            for row in rows:
                previous = {
                    str(item)
                    for item in _loads(
                        str(row["invalidated_paper_ids_json"]),
                        [],
                    )
                }
                connection.execute(
                    f"""
                    UPDATE {table_name}
                    SET is_valid = 0,
                        invalidated_at = ?,
                        invalidated_paper_ids_json = ?
                    WHERE rowid = ?
                    """,
                    (
                        timestamp,
                        _json(sorted(previous | affected)),
                        int(row["rowid"]),
                    ),
                )
        metadata = {
            f"analysis_status:{domain_id}": "待更新",
            f"analysis_error:{domain_id}": "",
            f"analysis_invalidation_reason:{domain_id}": reason,
        }
        connection.executemany(
            """
            INSERT INTO app_metadata (
                metadata_key, metadata_value, updated_at
            ) VALUES (?, ?, ?)
            ON CONFLICT(metadata_key) DO UPDATE SET
                metadata_value = excluded.metadata_value,
                updated_at = excluded.updated_at
            """,
            (
                (key, value, timestamp)
                for key, value in metadata.items()
            ),
        )

    def counts(self, domain_id: str) -> dict[str, int]:
        """返回首页用于展示的真实记录数量。"""

        has_deleted = self._has_paper_column("is_deleted")
        deleted_clause = "AND is_deleted = 0" if has_deleted else ""
        evidence_deleted_clause = (
            "AND p.is_deleted = 0" if has_deleted else ""
        )
        with self.connection() as connection:
            papers = connection.execute(
                f"""
                SELECT COUNT(*) AS value FROM papers
                WHERE domain_id = ? {deleted_clause}
                """,
                (domain_id,),
            ).fetchone()
            if has_deleted:
                deleted_papers = connection.execute(
                    """
                    SELECT COUNT(*) AS value FROM papers
                    WHERE domain_id = ? AND is_deleted = 1
                    """,
                    (domain_id,),
                ).fetchone()
            else:
                deleted_papers = None
            evidence = connection.execute(
                f"""
                SELECT COUNT(*) AS value
                FROM paper_evidence e
                JOIN papers p ON p.paper_id = e.paper_id
                WHERE p.domain_id = ? {evidence_deleted_clause}
                """,
                (domain_id,),
            ).fetchone()
            questions = connection.execute(
                "SELECT COUNT(*) AS value FROM questions WHERE domain_id = ?",
                (domain_id,),
            ).fetchone()
            attempts = connection.execute(
                "SELECT COUNT(*) AS value FROM attempts WHERE domain_id = ?",
                (domain_id,),
            ).fetchone()
        return {
            "papers": int(papers["value"] if papers else 0),
            "deleted_papers": int(
                deleted_papers["value"] if deleted_papers else 0
            ),
            "evidence": int(evidence["value"] if evidence else 0),
            "questions": int(questions["value"] if questions else 0),
            "attempts": int(attempts["value"] if attempts else 0),
        }

    def _has_paper_column(self, column_name: str) -> bool:
        """兼容尚未迁移或只读的旧数据库，避免引用不存在字段。"""

        return self._has_table_column("papers", column_name)

    def _has_table_column(self, table_name: str, column_name: str) -> bool:
        """检查固定内部表的字段，供只读旧库兼容查询使用。"""

        allowed_tables = {"papers", "debates", "syntheses"}
        if table_name not in allowed_tables:
            raise ValueError(f"不支持检查数据表：{table_name}")
        with self.connection() as connection:
            columns = {
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({table_name})"
                ).fetchall()
            }
        return column_name in columns

    def _paper_deleted_clause(
        self,
        *,
        alias: str = "",
        include_deleted: bool = False,
    ) -> str:
        """按数据库实际结构生成软删除过滤条件。"""

        if include_deleted or not self._has_paper_column("is_deleted"):
            return ""
        prefix = f"{alias}." if alias else ""
        return f"AND {prefix}is_deleted = 0"

    @staticmethod
    def _ensure_mastery_rows(
        connection: sqlite3.Connection,
        domain_id: str,
    ) -> None:
        timestamp = _now()
        connection.executemany(
            """
            INSERT INTO mastery (
                domain_id, dimension, score, attempts_count, updated_at
            ) VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(domain_id, dimension) DO UPDATE SET
                score = CASE
                    WHEN mastery.attempts_count = 0 THEN excluded.score
                    ELSE mastery.score
                END,
                updated_at = CASE
                    WHEN mastery.attempts_count = 0 THEN excluded.updated_at
                    ELSE mastery.updated_at
                END
            """,
            (
                (domain_id, dimension, score, timestamp)
                for dimension, score in BASELINE_MASTERY.items()
            ),
        )


def _row_to_domain(row: sqlite3.Row) -> ResearchDomain:
    return ResearchDomain(
        domain_id=str(row["domain_id"]),
        name=str(row["name"]),
        description=str(row["description"]),
        keywords=tuple(_loads(row["keywords_json"], [])),
        created_at=str(row["created_at"]),
    )


def _row_to_paper(row: sqlite3.Row) -> PaperRecord:
    keys = set(row.keys())

    def value(name: str, default: Any) -> Any:
        return row[name] if name in keys else default

    return PaperRecord(
        paper_id=str(row["paper_id"]),
        domain_id=str(row["domain_id"]),
        title=str(row["title"]),
        authors=tuple(_loads(row["authors_json"], [])),
        year=str(row["year"]),
        journal=str(row["journal"]),
        doi=str(row["doi"]),
        source_url=str(row["source_url"]),
        abstract=str(row["abstract"]),
        keywords=tuple(_loads(row["keywords_json"], [])),
        pdf_path=str(row["pdf_path"]),
        source=str(row["source"]),
        verification_status=str(row["verification_status"]),
        created_at=str(row["created_at"]),
        imported_at=str(value("imported_at", row["created_at"])),
        category=str(value("category", "待分类")),
        parse_status=str(value("parse_status", "仅元数据")),
        analysis_status=str(value("analysis_status", "未分析")),
        is_deleted=bool(value("is_deleted", 0)),
        deleted_at=str(value("deleted_at", "")),
        normalized_title=str(value("normalized_title", "")),
        normalization_version=str(value("normalization_version", "v1")),
        import_origin=str(value("import_origin", "")),
        is_manifest_managed=bool(value("is_manifest_managed", 1)),
    )


def _row_to_evidence(row: sqlite3.Row) -> EvidenceRef:
    source_type = str(row["source_type"]) if "source_type" in row.keys() else "全文"
    section_hint = (
        str(row["section_hint"]) if "section_hint" in row.keys() else ""
    )
    return EvidenceRef(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["paper_id"]),
        document_title=str(row["document_title"]),
        file_name=Path(str(row["pdf_path"])).name,
        page_number=int(row["page_number"]),
        quote=str(row["quote"]),
        source_type=source_type or "全文",
        section_hint=section_hint,
    )


def _row_to_question(row: sqlite3.Row) -> AssessmentQuestion:
    rubric = tuple(
        RubricCriterion(
            label=str(item["label"]),
            description=str(item["description"]),
            points=float(item["points"]),
            expected_keywords=tuple(item.get("expected_keywords", [])),
        )
        for item in _loads(row["rubric_json"], [])
    )
    return AssessmentQuestion(
        question_id=str(row["question_id"]),
        domain_id=str(row["domain_id"]),
        question_type=str(row["question_type"]),
        prompt=str(row["prompt"]),
        options=tuple(_loads(row["options_json"], [])),
        standard_answer=str(row["standard_answer"]),
        rubric=rubric,
        knowledge_tags=tuple(_loads(row["knowledge_tags_json"], [])),
        difficulty=str(row["difficulty"]),
        source_paper_ids=tuple(_loads(row["source_paper_ids_json"], [])),
        evidence=_evidence_from_json(row["evidence_json"]),
        created_at=str(row["created_at"]),
        scoring_mode=str(row["scoring_mode"]),
        explanation=(
            str(row["explanation"])
            if "explanation" in row.keys()
            else ""
        ),
        assessment_module=(
            str(row["assessment_module"])
            if "assessment_module" in row.keys()
            else "基础知识理解"
        ),
    )


def _row_to_attempt(row: sqlite3.Row) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=str(row["attempt_id"]),
        domain_id=str(row["domain_id"]),
        question_id=str(row["question_id"]),
        user_answer=str(row["user_answer"]),
        score=float(row["score"]),
        max_score=float(row["max_score"]),
        feedback=str(row["feedback"]),
        mode=str(row["mode"]),
        knowledge_tags=tuple(_loads(row["knowledge_tags_json"], [])),
        created_at=str(row["created_at"]),
        question_type=str(row["question_type"]),
        prompt=str(row["prompt"]),
    )


def _row_to_mastery(row: sqlite3.Row) -> MasteryRecord:
    return MasteryRecord(
        domain_id=str(row["domain_id"]),
        dimension=str(row["dimension"]),
        score=float(row["score"]),
        attempts_count=int(row["attempts_count"]),
        updated_at=str(row["updated_at"]),
    )


def _debate_from_json(value: str) -> DebateAnalysis:
    data = _loads(value, {})
    from researchmind.models import (
        QuestionSynthesis,
        ResearchConflict,
        Viewpoint,
    )

    viewpoints = tuple(
        Viewpoint(
            claim=str(item["claim"]),
            stance=str(item.get("stance", "待评估")),
            supporting_papers=tuple(item["supporting_papers"]),
            evidence=tuple(
                EvidenceRef.from_dict(entry)
                for entry in item.get("evidence", [])
            ),
            claim_origin=str(item.get("claim_origin", "原文摘录")),
            stance_origin=str(
                item.get("stance_origin", "本地规则推断")
            ),
        )
        for item in data.get("viewpoints", [])
    )
    return DebateAnalysis(
        topic=str(data["topic"]),
        viewpoints=viewpoints,
        possible_reasons=tuple(data.get("possible_reasons", [])),
        limitations=tuple(data.get("limitations", [])),
        evidence_strength=str(
            data.get("evidence_strength", "待评估")
        ),
        research_gaps=tuple(data.get("research_gaps", [])),
        consensus_summary=str(
            data.get("consensus_summary", "证据不足，暂不能判定学术共识。")
        ),
        disagreement_summary=str(
            data.get("disagreement_summary", "当前证据不足以确认学术分歧。")
        ),
        analysis_mode=str(
            data.get("analysis_mode", "本地规则证据归纳")
        ),
        considered_paper_ids=tuple(
            data.get("considered_paper_ids", [])
        ),
        evidence_covered_paper_ids=tuple(
            data.get("evidence_covered_paper_ids", [])
        ),
        coverage_note=str(data.get("coverage_note", "")),
        question_syntheses=tuple(
            QuestionSynthesis(
                question_id=str(item["question_id"]),
                label=str(item["label"]),
                question=str(item["question"]),
                conclusion=str(item["conclusion"]),
                status=str(item["status"]),
                evidence_certainty=str(item["evidence_certainty"]),
                certainty_reason=str(item["certainty_reason"]),
                supporting_paper_ids=tuple(
                    item.get("supporting_paper_ids", [])
                ),
                opposing_paper_ids=tuple(
                    item.get("opposing_paper_ids", [])
                ),
                inconclusive_paper_ids=tuple(
                    item.get("inconclusive_paper_ids", [])
                ),
                representative_evidence=tuple(
                    EvidenceRef.from_dict(entry)
                    for entry in item.get("representative_evidence", [])
                ),
                outcome_measures=tuple(item.get("outcome_measures", [])),
                limitations=tuple(item.get("limitations", [])),
                analysis_time=str(item.get("analysis_time", "")),
                negative_paper_ids=tuple(
                    item.get("negative_paper_ids", [])
                ),
                null_paper_ids=tuple(item.get("null_paper_ids", [])),
                mixed_paper_ids=tuple(item.get("mixed_paper_ids", [])),
                review_status=str(
                    item.get("review_status", "未审核")
                ),
            )
            for item in data.get("question_syntheses", [])
        ),
        conflicts=tuple(
            ResearchConflict(
                conflict_id=str(item["conflict_id"]),
                question_id=str(item["question_id"]),
                question=str(item["question"]),
                viewpoint_a=str(item["viewpoint_a"]),
                paper_ids_a=tuple(item.get("paper_ids_a", [])),
                evidence_a=tuple(
                    EvidenceRef.from_dict(entry)
                    for entry in item.get("evidence_a", [])
                ),
                viewpoint_b=str(item["viewpoint_b"]),
                paper_ids_b=tuple(item.get("paper_ids_b", [])),
                evidence_b=tuple(
                    EvidenceRef.from_dict(entry)
                    for entry in item.get("evidence_b", [])
                ),
                divergence_reasons=tuple(
                    item.get("divergence_reasons", [])
                ),
                current_conclusion=str(item.get("current_conclusion", "")),
                unresolved_questions=tuple(
                    item.get("unresolved_questions", [])
                ),
                evidence_certainty=str(
                    item.get("evidence_certainty", "无法评估")
                ),
                analysis_time=str(item.get("analysis_time", "")),
                review_status=str(
                    item.get("review_status", "待审核候选")
                ),
                comparability_status=str(
                    item.get(
                        "comparability_status",
                        "未自动判定，需同结局与干预条件人工复核",
                    )
                ),
                significance_note=str(
                    item.get(
                        "significance_note",
                        "仅将正向与负向结果列为候选对立方向；"
                        "未达统计显著不视为方向相反",
                    )
                ),
            )
            for item in data.get("conflicts", [])
        ),
        analysis_time=str(data.get("analysis_time", "")),
        engine_version=str(data.get("engine_version", "")),
    )


def _synthesis_from_json(value: str) -> ResearchSynthesis:
    data = _loads(value, {})
    from researchmind.models import KnowledgeEdge, KnowledgeNode

    return ResearchSynthesis(
        domain_id=str(data["domain_id"]),
        concepts=tuple(data.get("concepts", [])),
        theory_framework=tuple(data.get("theory_framework", [])),
        consensus=tuple(data.get("consensus", [])),
        disagreements=tuple(data.get("disagreements", [])),
        limitations=tuple(data.get("limitations", [])),
        research_gaps=tuple(data.get("research_gaps", [])),
        innovation_directions=tuple(
            data.get("innovation_directions", [])
        ),
        source_paper_ids=tuple(data.get("source_paper_ids", [])),
        generated_at=str(data["generated_at"]),
        nodes=tuple(
            KnowledgeNode(
                node_id=str(item["node_id"]),
                label=str(item["label"]),
                node_type=str(item["node_type"]),
                description=str(item["description"]),
                evidence_count=int(item["evidence_count"]),
            )
            for item in data.get("nodes", [])
        ),
        edges=tuple(
            KnowledgeEdge(
                source_id=str(item["source_id"]),
                target_id=str(item["target_id"]),
                relation=str(item["relation"]),
                evidence_count=int(item["evidence_count"]),
            )
            for item in data.get("edges", [])
        ),
        mode=str(data.get("mode", "本地证据归纳")),
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_evidence(evidence: tuple[EvidenceRef, ...]) -> str:
    return _json([item.to_dict() for item in evidence])


def _evidence_from_json(value: str) -> tuple[EvidenceRef, ...]:
    return tuple(
        EvidenceRef.from_dict(item)
        for item in _loads(value, [])
    )


def _loads(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _stable_id(prefix: str, *parts: object) -> str:
    import hashlib

    digest = hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _section_hint(quote: str) -> str:
    """仅根据原文关键词给片段加轻量段落提示，不推断额外内容。"""

    lowered = quote.lower()
    hints = (
        ("方法", "方法"),
        ("研究对象", "对象"),
        ("结果", "结果"),
        ("结论", "结论"),
        ("局限", "局限"),
        ("purpose", "目的"),
        ("method", "method"),
        ("result", "result"),
        ("conclusion", "conclusion"),
        ("limitation", "limitation"),
    )
    for keyword, hint in hints:
        if keyword in lowered:
            return hint
    return ""


def _normalize_doi(value: str) -> str:
    """统一 DOI 大小写、URL 前缀和末尾标点，用于本地去重。"""

    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized.casefold().rstrip(".,;，。；")


def _normalize_title(value: str) -> str:
    """统一中英文题名的大小写、全半角、空白和标点。"""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\W_]+", "", normalized)
