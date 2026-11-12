"""ResearchMind 的 SQLite 持久化层。

SQLite 负责保存领域、论文、证据、结构化阅读、学术分歧、知识体系、题库、
答题记录和能力掌握度。它不依赖外部服务，适合竞赛现场和初学者本地调试。
"""

from __future__ import annotations

import json
import sqlite3
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
    "概念记忆": 15.0,
    "文献理解": 20.0,
    "对比分析": 25.0,
    "批判性思维": 20.0,
    "学术应用": 20.0,
}


class ResearchMindStorageError(RuntimeError):
    """研究数据库无法初始化、读取或写入时抛出的可展示错误。"""


class ResearchMindStore:
    """研究数据与学习记录的本地 SQLite 存储。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or DEFAULT_DB_PATH).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """创建短连接并确保提交或回滚。"""

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.db_path, timeout=20)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            if connection is not None:
                connection.rollback()
            raise ResearchMindStorageError(f"研究数据库操作失败：{exc}") from exc
        finally:
            if connection is not None:
                connection.close()

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
            results_json TEXT NOT NULL,
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
            FOREIGN KEY (domain_id) REFERENCES research_domains(domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_debates_domain
            ON debates(domain_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS syntheses (
            synthesis_id TEXT PRIMARY KEY,
            domain_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
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
        """
        try:
            with self.connection() as connection:
                connection.executescript(schema)
        except ResearchMindStorageError:
            raise
        except sqlite3.Error as exc:
            raise ResearchMindStorageError(
                f"无法初始化研究数据库：{exc}"
            ) from exc

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
        """按论文 ID 更新元数据，不覆盖已经存在的证据片段。"""

        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO papers (
                    paper_id, domain_id, title, authors_json, year, journal,
                    doi, source_url, abstract, keywords_json, pdf_path,
                    source, verification_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    verification_status = excluded.verification_status
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
                ),
            )

    def list_papers(self, domain_id: str) -> tuple[PaperRecord, ...]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM papers
                WHERE domain_id = ?
                ORDER BY year DESC, title
                """,
                (domain_id,),
            ).fetchall()
        return tuple(_row_to_paper(row) for row in rows)

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return _row_to_paper(row) if row else None

    def paper_count(self, domain_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM papers WHERE domain_id = ?",
                (domain_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def delete_papers_except(
        self,
        domain_id: str,
        keep_paper_ids: Iterable[str],
    ) -> int:
        """删除领域中已不在当前论文清单里的论文及其关联记录。"""

        keep_ids = {str(paper_id) for paper_id in keep_paper_ids}
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT paper_id FROM papers WHERE domain_id = ?",
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
        return len(stale_ids)

    def invalidate_domain_derivations(
        self,
        domain_id: str,
        deleted_paper_ids: Iterable[str],
    ) -> dict[str, int]:
        """删除已失效的领域分析、试题和关联答题记录。

        论文库变化后，旧的观点比较和知识体系可能继续引用已删除论文。
        因此清空该领域的派生分析；题库只删除明确引用被删论文的题目，
        无关题目和其余学习记录继续保留。
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

            debate_cursor = connection.execute(
                "DELETE FROM debates WHERE domain_id = ?",
                (domain_id,),
            )
            synthesis_cursor = connection.execute(
                "DELETE FROM syntheses WHERE domain_id = ?",
                (domain_id,),
            )

        return {
            "debates": max(0, debate_cursor.rowcount),
            "syntheses": max(0, synthesis_cursor.rowcount),
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

    def list_evidence(
        self,
        paper_id: str,
        *,
        limit: int | None = None,
    ) -> tuple[EvidenceRef, ...]:
        sql = """
            SELECT e.*, p.title AS document_title, p.pdf_path
            FROM paper_evidence e
            JOIN papers p ON p.paper_id = e.paper_id
            WHERE e.paper_id = ?
            ORDER BY e.page_number, e.chunk_id
        """
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
    ) -> tuple[EvidenceRef, ...]:
        sql = """
            SELECT e.*, p.title AS document_title, p.pdf_path
            FROM paper_evidence e
            JOIN papers p ON p.paper_id = e.paper_id
            WHERE p.domain_id = ?
            ORDER BY p.year DESC, e.page_number, e.chunk_id
        """
        parameters: list[Any] = [domain_id]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(max(1, int(limit)))
        with self.connection() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(_row_to_evidence(row) for row in rows)

    def evidence_count(self, domain_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM paper_evidence e
                JOIN papers p ON p.paper_id = e.paper_id
                WHERE p.domain_id = ?
                """,
                (domain_id,),
            ).fetchone()
        return int(row["count"]) if row else 0

    def save_analysis(self, analysis: PaperAnalysis) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO paper_analyses (
                    paper_id, purpose_json, population_json, methods_json,
                    results_json, conclusions_json, limitations_json,
                    extraction_mode, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    purpose_json = excluded.purpose_json,
                    population_json = excluded.population_json,
                    methods_json = excluded.methods_json,
                    results_json = excluded.results_json,
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
                    _json_evidence(analysis.results),
                    _json_evidence(analysis.conclusions),
                    _json_evidence(analysis.limitations),
                    analysis.extraction_mode,
                    analysis.status,
                    _now(),
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
            results=_evidence_from_json(row["results_json"]),
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
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM debates
                WHERE domain_id = ?
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
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM syntheses
                WHERE domain_id = ?
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
                    scoring_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def counts(self, domain_id: str) -> dict[str, int]:
        """返回首页用于展示的真实记录数量。"""

        with self.connection() as connection:
            papers = connection.execute(
                "SELECT COUNT(*) AS value FROM papers WHERE domain_id = ?",
                (domain_id,),
            ).fetchone()
            evidence = connection.execute(
                """
                SELECT COUNT(*) AS value
                FROM paper_evidence e
                JOIN papers p ON p.paper_id = e.paper_id
                WHERE p.domain_id = ?
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
            "evidence": int(evidence["value"] if evidence else 0),
            "questions": int(questions["value"] if questions else 0),
            "attempts": int(attempts["value"] if attempts else 0),
        }

    @staticmethod
    def _ensure_mastery_rows(
        connection: sqlite3.Connection,
        domain_id: str,
    ) -> None:
        timestamp = _now()
        connection.executemany(
            """
            INSERT OR IGNORE INTO mastery (
                domain_id, dimension, score, attempts_count, updated_at
            ) VALUES (?, ?, ?, 0, ?)
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
    )


def _row_to_evidence(row: sqlite3.Row) -> EvidenceRef:
    return EvidenceRef(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["paper_id"]),
        document_title=str(row["document_title"]),
        file_name=Path(str(row["pdf_path"])).name,
        page_number=int(row["page_number"]),
        quote=str(row["quote"]),
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
    from researchmind.models import Viewpoint

    viewpoints = tuple(
        Viewpoint(
            claim=str(item["claim"]),
            stance=str(item.get("stance", "待评估")),
            supporting_papers=tuple(item["supporting_papers"]),
            evidence=tuple(
                EvidenceRef.from_dict(entry)
                for entry in item.get("evidence", [])
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
