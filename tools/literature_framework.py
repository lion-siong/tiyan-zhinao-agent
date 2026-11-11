"""从内置文献索引构建可持续扩展的研究框架。

框架使用固定科研维度加关键词检索，把每篇论文的真实原文片段归入相关维度。
这不是大模型生成的综述，也不会创造论文中没有出现的研究结论。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from services.rag_service import LocalEvidenceIndex
from tools.knowledge_base import (
    INDEX_FILE_PATH,
    BundledKnowledgeBase,
    KnowledgeBaseError,
    load_bundled_knowledge_base,
)


FRAMEWORK_FILE_NAME = "framework.json"
FRAMEWORK_SCHEMA_VERSION = 1
FRAMEWORK_NAME = "社会生态视角下大学生体育参与与体质健康研究框架"
FRAMEWORK_DESCRIPTION = (
    "从研究背景、研究对象、个体/家庭/学校/社会因素、研究方法、"
    "行为与健康结果、干预对策九个维度组织论文证据。"
)
FRAMEWORK_LOGIC_CHAIN = (
    "研究背景与问题",
    "研究对象与场景",
    "研究设计与方法",
    "个体因素",
    "家庭因素",
    "学校因素",
    "社会因素",
    "行为与健康结果",
    "干预对策",
)
MAX_EVIDENCE_PER_DOCUMENT = 2


@dataclass(frozen=True)
class FrameworkDimensionDefinition:
    """定义框架中的一个研究维度。"""

    key: str
    title: str
    description: str
    keywords: tuple[str, ...]


DEFAULT_DIMENSIONS = (
    FrameworkDimensionDefinition(
        key="background",
        title="研究背景与问题",
        description="论文关注的研究现状、现实问题与选题依据。",
        keywords=(
            "研究背景",
            "现状",
            "问题",
            "下滑",
            "严峻",
            "不足",
            "research background",
            "current situation",
            "challenge",
            "decline",
            "physical inactivity",
            "public health",
        ),
    ),
    FrameworkDimensionDefinition(
        key="population",
        title="研究对象与场景",
        description="论文调查或干预的人群、地区、学校与情境。",
        keywords=(
            "研究对象",
            "大学生",
            "高校",
            "学校",
            "地区",
            "调查对象",
            "college students",
            "university students",
            "undergraduate",
            "participants",
            "Chinese college",
        ),
    ),
    FrameworkDimensionDefinition(
        key="methods",
        title="研究设计与方法",
        description="论文采用的问卷、实验、测量、抽样与分析方法。",
        keywords=(
            "研究方法",
            "问卷调查",
            "问卷",
            "样本",
            "量表",
            "数据",
            "实验",
            "randomized controlled trial",
            "cross-sectional",
            "questionnaire",
            "survey",
            "sample",
            "measurement",
            "statistical analysis",
        ),
    ),
    FrameworkDimensionDefinition(
        key="individual",
        title="个体因素",
        description="与个人动机、认知、态度、习惯和自主性有关的因素。",
        keywords=(
            "个人因素",
            "个体",
            "锻炼动机",
            "运动兴趣",
            "生活习惯",
            "健康认知",
            "自主性",
            "self-efficacy",
            "motivation",
            "health belief",
            "health beliefs",
            "psychological capital",
            "exercise imagery",
            "attitude",
            "intention",
        ),
    ),
    FrameworkDimensionDefinition(
        key="family",
        title="家庭因素",
        description="父母支持、家庭氛围和家庭体育参与条件。",
        keywords=(
            "家庭因素",
            "家庭",
            "父母",
            "家长",
            "支持度",
            "运动氛围",
            "family",
            "parents",
            "parental support",
            "family support",
        ),
    ),
    FrameworkDimensionDefinition(
        key="school",
        title="学校因素",
        description="体育课程、教师、校园资源与人际环境。",
        keywords=(
            "学校因素",
            "学校",
            "高校",
            "体育课程",
            "教师",
            "人际环境",
            "体育资源",
            "physical education",
            "teacher",
            "campus",
            "school",
            "university",
            "course",
            "blended learning",
        ),
    ),
    FrameworkDimensionDefinition(
        key="society",
        title="社会因素",
        description="社会体育服务、公共设施、文化环境和舆论支持。",
        keywords=(
            "社会因素",
            "社会",
            "体育服务",
            "公共服务",
            "体育文化",
            "舆论",
            "social support",
            "social environment",
            "public services",
            "public facilities",
            "peer support",
            "social ecological",
        ),
    ),
    FrameworkDimensionDefinition(
        key="outcomes",
        title="行为与健康结果",
        description="体育锻炼行为、参与水平、体质健康和身体素质等结果。",
        keywords=(
            "体育锻炼行为",
            "运动参与",
            "体质健康",
            "身体素质",
            "健康水平",
            "锻炼行为",
            "physical activity",
            "physical exercise behavior",
            "exercise behavior",
            "physical fitness",
            "fitness",
            "health",
            "obesity",
            "weight loss",
        ),
    ),
    FrameworkDimensionDefinition(
        key="interventions",
        title="干预对策",
        description="论文提出的健康促进、协同育人和支持保障建议。",
        keywords=(
            "对策",
            "建议",
            "提升",
            "干预",
            "健康促进",
            "协同育人",
            "保障",
            "intervention",
            "promotion",
            "strategy",
            "recommendation",
            "program",
            "exercise training",
            "aerobic exercise",
            "tai chi",
        ),
    ),
)


@dataclass(frozen=True)
class FrameworkEvidence:
    """框架维度中的一条可追溯论文证据。"""

    document_id: str
    document_title: str
    file_name: str
    page_number: int
    chunk_id: str
    excerpt: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.document_title}，{self.file_name}，第 {self.page_number} 页"


@dataclass(frozen=True)
class FrameworkDimension:
    """一个框架维度及其当前文献证据。"""

    key: str
    title: str
    description: str
    keywords: tuple[str, ...]
    evidence: tuple[FrameworkEvidence, ...]

    @property
    def document_count(self) -> int:
        return len({item.document_id for item in self.evidence})


@dataclass(frozen=True)
class LiteratureFramework:
    """可持久化并随文献库自动更新的研究框架。"""

    name: str
    description: str
    logic_chain: tuple[str, ...]
    dimensions: tuple[FrameworkDimension, ...]
    documents_count: int
    built_at: str
    manifest_sha256: str
    index_sha256: str

    @property
    def evidence_count(self) -> int:
        return sum(len(dimension.evidence) for dimension in self.dimensions)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的普通字典。"""

        return {
            "schema_version": FRAMEWORK_SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "logic_chain": list(self.logic_chain),
            "documents_count": self.documents_count,
            "built_at": self.built_at,
            "manifest_sha256": self.manifest_sha256,
            "index_sha256": self.index_sha256,
            "dimensions": [
                {
                    "key": dimension.key,
                    "title": dimension.title,
                    "description": dimension.description,
                    "keywords": list(dimension.keywords),
                    "evidence": [
                        {
                            "document_id": item.document_id,
                            "document_title": item.document_title,
                            "file_name": item.file_name,
                            "page_number": item.page_number,
                            "chunk_id": item.chunk_id,
                            "excerpt": item.excerpt,
                            "score": item.score,
                        }
                        for item in dimension.evidence
                    ],
                }
                for dimension in self.dimensions
            ],
        }


def build_literature_framework(
    bundle: BundledKnowledgeBase,
    base_dir: Path | str | None = None,
) -> LiteratureFramework:
    """根据当前全部论文自动生成研究框架。"""

    root = _resolve_root(base_dir)
    documents_by_file = {
        document.file_name: document for document in bundle.documents
    }
    dimensions: list[FrameworkDimension] = []

    for definition in DEFAULT_DIMENSIONS:
        evidence: list[FrameworkEvidence] = []
        query = " ".join(definition.keywords)
        for document in bundle.documents:
            document_chunks = tuple(
                chunk
                for chunk in bundle.chunks
                if chunk.document_name == document.file_name
            )
            if not document_chunks:
                continue
            results = LocalEvidenceIndex(document_chunks).search(query, top_k=2)
            for result in results[:MAX_EVIDENCE_PER_DOCUMENT]:
                if result.score <= 0:
                    continue
                evidence.append(
                    FrameworkEvidence(
                        document_id=document.document_id,
                        document_title=document.title,
                        file_name=document.file_name,
                        page_number=result.chunk.page_number,
                        chunk_id=result.chunk.chunk_id,
                        excerpt=result.chunk.text,
                        score=round(result.score, 4),
                    )
                )
        dimensions.append(
            FrameworkDimension(
                key=definition.key,
                title=definition.title,
                description=definition.description,
                keywords=definition.keywords,
                evidence=tuple(evidence),
            )
        )

    manifest_path = root / "manifest.json"
    index_path = root / INDEX_FILE_PATH
    return LiteratureFramework(
        name=FRAMEWORK_NAME,
        description=FRAMEWORK_DESCRIPTION,
        logic_chain=FRAMEWORK_LOGIC_CHAIN,
        dimensions=tuple(dimensions),
        documents_count=len(documents_by_file),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        manifest_sha256=_sha256_file(manifest_path),
        index_sha256=_sha256_file(index_path),
    )


def build_and_save_literature_framework(
    bundle: BundledKnowledgeBase,
    base_dir: Path | str | None = None,
) -> LiteratureFramework:
    """生成框架并写入 knowledge_base/framework.json。"""

    root = _resolve_root(base_dir)
    framework = build_literature_framework(bundle, root)
    framework_path = root / FRAMEWORK_FILE_NAME
    temporary_path = framework_path.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(framework.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(framework_path)
    except OSError as exc:
        raise KnowledgeBaseError(f"研究框架无法保存：{exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return framework


def load_literature_framework(
    base_dir: Path | str | None = None,
) -> LiteratureFramework:
    """读取框架；文件缺失或与当前索引不一致时自动重建。"""

    root = _resolve_root(base_dir)
    bundle = load_bundled_knowledge_base(root)
    framework_path = root / FRAMEWORK_FILE_NAME
    manifest_path = root / "manifest.json"
    index_path = root / INDEX_FILE_PATH

    if framework_path.exists():
        try:
            framework = _read_framework(framework_path)
            if (
                framework.manifest_sha256 == _sha256_file(manifest_path)
                and framework.index_sha256 == _sha256_file(index_path)
                and framework.documents_count == len(bundle.documents)
            ):
                return framework
        except (KnowledgeBaseError, OSError, KeyError, TypeError, ValueError):
            pass

    return build_and_save_literature_framework(bundle, root)


def _read_framework(path: Path) -> LiteratureFramework:
    """读取并解析框架 JSON。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != FRAMEWORK_SCHEMA_VERSION:
        raise KnowledgeBaseError("研究框架版本不受支持。")

    dimensions: list[FrameworkDimension] = []
    for item in data["dimensions"]:
        evidence = tuple(
            FrameworkEvidence(
                document_id=str(entry["document_id"]),
                document_title=str(entry["document_title"]),
                file_name=str(entry["file_name"]),
                page_number=int(entry["page_number"]),
                chunk_id=str(entry["chunk_id"]),
                excerpt=str(entry["excerpt"]),
                score=float(entry["score"]),
            )
            for entry in item["evidence"]
        )
        dimensions.append(
            FrameworkDimension(
                key=str(item["key"]),
                title=str(item["title"]),
                description=str(item["description"]),
                keywords=tuple(str(value) for value in item["keywords"]),
                evidence=evidence,
            )
        )

    return LiteratureFramework(
        name=str(data["name"]),
        description=str(data["description"]),
        logic_chain=tuple(str(value) for value in data["logic_chain"]),
        dimensions=tuple(dimensions),
        documents_count=int(data["documents_count"]),
        built_at=str(data["built_at"]),
        manifest_sha256=str(data["manifest_sha256"]),
        index_sha256=str(data["index_sha256"]),
    )


def _resolve_root(base_dir: Path | str | None) -> Path:
    """返回框架所在的知识库目录。"""

    root = Path(base_dir or settings.knowledge_base_dir).resolve()
    if not root.exists():
        raise KnowledgeBaseError(f"知识库目录不存在：{root}")
    return root


def _sha256_file(path: Path) -> str:
    """分块计算文件哈希。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise KnowledgeBaseError(f"无法读取文件 {path}：{exc}") from exc
    return digest.hexdigest()
