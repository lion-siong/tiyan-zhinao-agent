"""项目配置。

所有密钥都从环境变量或项目根目录的 .env 文件读取，不会写入代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """应用运行配置。"""

    app_name: str = "体研智脑 Agent"
    app_subtitle: str = "基于大语言模型与多智能体协作的体育科研全流程智能助手"
    project_root: Path = PROJECT_ROOT
    researchmind_db_path: Path | None = field(default=None, repr=False)
    openai_api_key: str = field(default="", repr=False)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = 60
    semantic_scholar_api_key: str = field(default="", repr=False)
    semantic_scholar_base_url: str = (
        "https://api.semanticscholar.org/graph/v1"
    )
    semantic_scholar_timeout_seconds: int = 25
    semantic_scholar_max_results: int = 20
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    max_pdf_size_mb: int = 20
    max_upload_size_mb: int = 30
    max_pdf_pages: int = 80
    max_pdf_chars: int = 40_000
    rag_chunk_chars: int = 700
    rag_chunk_overlap: int = 120
    rag_top_k: int = 5
    rag_max_documents: int = 5
    embedding_model: str = "intfloat/multilingual-e5-small"
    chroma_collection_name: str = "evidence_chunks"
    rag_retrieval_mode: str = "hybrid"
    researchmind_date_from: str = "2023-01-01"
    openalex_mailto: str = ""
    zotero_api_key: str = field(default="", repr=False)
    zotero_library_id: str = ""
    zotero_library_type: str = "user"
    zotero_base_url: str = "https://api.zotero.org"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """从映射对象或系统环境变量创建配置。

        传入 env 主要用于自动化测试；正常运行时不传参数即可。
        """

        values = env if env is not None else os.environ
        return cls(
            researchmind_db_path=_read_optional_path(
                values.get("RESEARCHMIND_DB_PATH")
            ),
            openai_api_key=values.get("OPENAI_API_KEY", "").strip(),
            openai_base_url=values.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).strip(),
            openai_model=values.get("OPENAI_MODEL", "gpt-4o-mini").strip(),
            llm_timeout_seconds=_read_positive_int(
                values.get("LLM_TIMEOUT_SECONDS"), default=60
            ),
            semantic_scholar_api_key=values.get(
                "SEMANTIC_SCHOLAR_API_KEY", ""
            ).strip(),
            semantic_scholar_base_url=values.get(
                "SEMANTIC_SCHOLAR_BASE_URL",
                "https://api.semanticscholar.org/graph/v1",
            ).strip()
            or "https://api.semanticscholar.org/graph/v1",
            semantic_scholar_timeout_seconds=_read_positive_int(
                values.get("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS"), default=25
            ),
            semantic_scholar_max_results=_read_positive_int(
                values.get("SEMANTIC_SCHOLAR_MAX_RESULTS"), default=20
            ),
            arxiv_api_url=values.get(
                "ARXIV_API_URL",
                "https://export.arxiv.org/api/query",
            ).strip()
            or "https://export.arxiv.org/api/query",
            max_pdf_size_mb=_read_positive_int(
                values.get("MAX_PDF_SIZE_MB"), default=20
            ),
            max_upload_size_mb=_read_positive_int(
                values.get("MAX_UPLOAD_SIZE_MB"), default=30
            ),
            max_pdf_pages=_read_positive_int(
                values.get("MAX_PDF_PAGES"), default=80
            ),
            max_pdf_chars=_read_positive_int(
                values.get("MAX_PDF_CHARS"), default=40_000
            ),
            rag_chunk_chars=_read_positive_int(
                values.get("RAG_CHUNK_CHARS"), default=700
            ),
            rag_chunk_overlap=_read_non_negative_int(
                values.get("RAG_CHUNK_OVERLAP"), default=120
            ),
            rag_top_k=_read_positive_int(
                values.get("RAG_TOP_K"), default=5
            ),
            rag_max_documents=_read_positive_int(
                values.get("RAG_MAX_DOCUMENTS"), default=5
            ),
            embedding_model=values.get(
                "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ).strip()
            or "intfloat/multilingual-e5-small",
            chroma_collection_name=values.get(
                "CHROMA_COLLECTION_NAME", "evidence_chunks"
            ).strip()
            or "evidence_chunks",
            rag_retrieval_mode=_read_retrieval_mode(
                values.get("RAG_RETRIEVAL_MODE"),
                default="hybrid",
            ),
            researchmind_date_from=values.get(
                "RESEARCHMIND_DATE_FROM", "2023-01-01"
            ).strip()
            or "2023-01-01",
            openalex_mailto=values.get("OPENALEX_MAILTO", "").strip(),
            zotero_api_key=values.get("ZOTERO_API_KEY", "").strip(),
            zotero_library_id=values.get("ZOTERO_LIBRARY_ID", "").strip(),
            zotero_library_type=values.get(
                "ZOTERO_LIBRARY_TYPE", "user"
            ).strip().lower()
            or "user",
            zotero_base_url=values.get(
                "ZOTERO_BASE_URL", "https://api.zotero.org"
            ).strip()
            or "https://api.zotero.org",
        )

    @property
    def llm_configured(self) -> bool:
        """是否已配置大模型密钥。"""

        return bool(self.openai_api_key)

    @property
    def semantic_scholar_configured(self) -> bool:
        """Semantic Scholar 可选密钥是否已配置。"""

        return bool(self.semantic_scholar_api_key)

    @property
    def zotero_configured(self) -> bool:
        """是否已配置 Zotero Web API。"""

        return bool(self.zotero_api_key and self.zotero_library_id)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.project_root / "knowledge_base"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.data_dir / "models"


def _read_positive_int(value: str | None, default: int) -> int:
    """读取正整数配置，非法值自动回退到默认值。"""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _read_non_negative_int(value: str | None, default: int) -> int:
    """读取非负整数配置，非法值自动回退到默认值。"""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _read_retrieval_mode(value: str | None, default: str) -> str:
    """读取检索模式，非法值自动回退为混合检索。"""

    allowed = {"hybrid", "semantic", "bm25"}
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _read_optional_path(value: str | None) -> Path | None:
    """读取可选的绝对或相对路径，空值保持未设置。"""

    normalized = str(value or "").strip()
    if not normalized:
        return None
    return Path(normalized).expanduser().resolve()


settings = Settings.from_env()
