"""参赛导出与 Git 工作区的敏感信息审计工具。

本工具只报告风险文件、规则名称和行号，不输出匹配到的密钥内容。
支持普通目录以及 ZIP/TAR 压缩包的文件名检查，不解压或执行压缩包内容。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SEVERITY_BLOCK = "block"
SEVERITY_REVIEW = "review"

BLOCK_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "secrets.toml",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "service-account*.json",
    "*.db",
    "*.db-*",
    "*.sqlite",
    "*.sqlite-*",
    "*.sqlite3",
    "*.sqlite3-*",
    "*.log",
)

BLOCK_DIR_PATTERNS = (
    ".tmp",
    ".pytest_tmp",
    "edge-profile*",
    "chrome-profile*",
    "firefox-profile*",
    "user_papers",
)

BLOCK_BROWSER_FILE_NAMES = {
    "cookies",
    "login data",
    "local state",
    "web data",
    "history",
    "network persistent state",
}

REVIEW_DIR_PATTERNS = (
    "chroma",
    "chroma.corrupt-*",
    "index",
    "pdfs",
)

ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz"}
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "node_modules",
    "models",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

CONTENT_RULES = (
    (
        "possible_platform_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    ),
    (
        "possible_secret_assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret)"
            r"\b\s*[:=]\s*([\"'])[A-Za-z0-9_./+\-]{16,}\1",
            re.IGNORECASE,
        ),
    ),
    (
        "possible_bearer_token",
        re.compile(
            r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._\-]{16,}",
            re.IGNORECASE,
        ),
    ),
)

MAX_CONTENT_SCAN_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class AuditFinding:
    """单条审计风险。只保存位置和规则，不保存真实敏感值。"""

    severity: str
    rule: str
    relative_path: str
    message: str
    line_number: int | None = None


@dataclass(frozen=True)
class AuditReport:
    """一次审计的汇总结果。"""

    target: str
    findings: tuple[AuditFinding, ...]

    @property
    def blocking_count(self) -> int:
        return sum(
            finding.severity == SEVERITY_BLOCK for finding in self.findings
        )

    @property
    def review_count(self) -> int:
        return sum(
            finding.severity == SEVERITY_REVIEW for finding in self.findings
        )

    @property
    def safe_for_export(self) -> bool:
        return self.blocking_count == 0


def audit_path(
    target: Path | str,
    *,
    scan_content: bool = True,
) -> AuditReport:
    """审计文件、目录、ZIP 或 TAR 包，返回不含密钥原文的结果。"""

    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"审计路径不存在：{path}")

    if path.is_file() and _is_supported_archive(path):
        findings = _audit_archive(path)
        return AuditReport(str(path), tuple(_deduplicate(findings)))

    if not path.is_dir():
        findings = _audit_file(path, path.parent, scan_content=scan_content)
        return AuditReport(str(path), tuple(findings))

    findings: list[AuditFinding] = []
    for current_root, directory_names, file_names in os.walk(path):
        current_path = Path(current_root)

        for directory_name in tuple(directory_names):
            directory_path = current_path / directory_name
            relative_path = _relative_posix(directory_path, path)
            finding = _audit_directory_name(relative_path, directory_name)
            if finding is not None:
                findings.append(finding)
                directory_names.remove(directory_name)
                continue
            if _is_skipped_directory(directory_name):
                directory_names.remove(directory_name)

        for file_name in file_names:
            file_path = current_path / file_name
            findings.extend(
                _audit_file(file_path, path, scan_content=scan_content)
            )

    return AuditReport(
        str(path),
        tuple(_deduplicate(findings)),
    )


def render_report(report: AuditReport) -> str:
    """生成适合终端查看的中文文本，不包含密钥内容。"""

    lines = [
        f"审计目标：{report.target}",
        f"阻断项：{report.blocking_count}；需人工复核：{report.review_count}",
    ]
    if not report.findings:
        lines.append("未发现敏感文件、隐私数据或疑似密钥。")
        return "\n".join(lines)

    for finding in report.findings:
        location = finding.relative_path
        if finding.line_number is not None:
            location = f"{location}:{finding.line_number}"
        lines.append(
            f"[{finding.severity.upper()}] {location} | "
            f"{finding.rule} | {finding.message}"
        )
    return "\n".join(lines)


def report_as_dict(report: AuditReport) -> dict[str, object]:
    """转换为便于 CI 或后续页面展示的 JSON 结构。"""

    return {
        "target": report.target,
        "safe_for_export": report.safe_for_export,
        "blocking_count": report.blocking_count,
        "review_count": report.review_count,
        "findings": [asdict(finding) for finding in report.findings],
    }


def _audit_directory_name(
    relative_path: str,
    directory_name: str,
) -> AuditFinding | None:
    normalized_name = directory_name.casefold()
    if any(
        fnmatch.fnmatch(normalized_name, pattern)
        for pattern in BLOCK_DIR_PATTERNS
    ):
        return AuditFinding(
            severity=SEVERITY_BLOCK,
            rule="sensitive_directory",
            relative_path=relative_path,
            message="可能包含 Cookie、登录凭据、用户上传或个人数据，禁止导出。",
        )
    if any(
        fnmatch.fnmatch(normalized_name, pattern)
        for pattern in REVIEW_DIR_PATTERNS
    ):
        return AuditFinding(
            severity=SEVERITY_REVIEW,
            rule="generated_or_document_directory",
            relative_path=relative_path,
            message="论文全文或派生索引需核对版权、隐私和可否公开再分发。",
        )
    return None


def _audit_file(
    path: Path,
    root: Path,
    *,
    scan_content: bool,
) -> list[AuditFinding]:
    relative_path = _relative_posix(path, root)
    normalized_name = path.name.casefold()
    findings: list[AuditFinding] = []

    if _is_sensitive_file_name(normalized_name):
        findings.append(
            AuditFinding(
                severity=SEVERITY_BLOCK,
                rule="sensitive_file",
                relative_path=relative_path,
                message="本地密钥、数据库、日志或凭据文件，禁止进入导出包和 Git。",
            )
        )
        return findings

    if _is_supported_archive(path):
        findings.append(
            AuditFinding(
                severity=SEVERITY_BLOCK,
                rule="archive_requires_audit",
                relative_path=relative_path,
                message="压缩包不能直接导出，请先解包并单独审计其内容。",
            )
        )
        return findings

    if path.suffix.casefold() == ".pdf":
        findings.append(
            AuditFinding(
                severity=SEVERITY_REVIEW,
                rule="pdf_requires_permission_review",
                relative_path=relative_path,
                message="论文全文可能受版权或个人隐私约束，公开前必须逐篇确认授权。",
            )
        )
        return findings

    if scan_content and _is_text_file(path):
        findings.extend(_scan_text_content(path, relative_path))
    return findings


def _scan_text_content(path: Path, relative_path: str) -> list[AuditFinding]:
    """扫描常见文本文件中的密钥模式，只返回行号和规则。"""

    try:
        data = path.read_bytes()[:MAX_CONTENT_SCAN_BYTES]
    except OSError:
        return [
            AuditFinding(
                severity=SEVERITY_REVIEW,
                rule="unreadable_text_file",
                relative_path=relative_path,
                message="文件无法读取，必须人工确认内容。",
            )
        ]
    return _scan_text_bytes(data, relative_path)


def _scan_text_bytes(data: bytes, relative_path: str) -> list[AuditFinding]:
    """扫描文本内容，只返回规则和行号，不保留匹配值。"""

    text = data.decode("utf-8", errors="ignore")
    findings: list[AuditFinding] = []
    for rule_name, pattern in CONTENT_RULES:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                findings.append(
                    AuditFinding(
                        severity=SEVERITY_BLOCK,
                        rule=rule_name,
                        relative_path=relative_path,
                        message="检测到疑似密钥格式，请确认、吊销并移除。",
                        line_number=line_number,
                    )
                )
    return findings


def _audit_archive(path: Path) -> Iterable[AuditFinding]:
    """读取压缩包成员名称和有限文本内容，不把成员解压到磁盘。"""

    try:
        if path.suffix.casefold() == ".zip":
            with zipfile.ZipFile(path) as archive:
                findings = list(
                    _audit_zip_members(path.name, archive)
                )
        else:
            with tarfile.open(path, mode="r:*") as archive:
                findings = list(
                    _audit_tar_members(path.name, archive)
                )
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        yield AuditFinding(
            severity=SEVERITY_BLOCK,
            rule="unreadable_archive",
            relative_path=path.name,
            message="压缩包无法读取或格式损坏，不能作为导出包。",
        )
        return

    yield from findings


def _audit_zip_members(
    archive_name: str,
    archive: zipfile.ZipFile,
) -> Iterable[AuditFinding]:
    """审计 ZIP 成员，并限制单个文本成员的读取大小。"""

    for info in archive.infolist():
        member_name = info.filename
        normalized_name = member_name.replace("\\", "/")
        relative_path = f"{archive_name}!/{normalized_name}"
        directory_finding, should_skip = _audit_archive_member_name(
            relative_path,
            normalized_name,
        )
        if directory_finding is not None:
            yield directory_finding
        if should_skip or info.is_dir():
            continue
        if not _is_text_file(Path(normalized_name)):
            continue
        try:
            with archive.open(info) as member:
                data = member.read(MAX_CONTENT_SCAN_BYTES)
        except (OSError, RuntimeError, zipfile.BadZipFile):
            yield AuditFinding(
                severity=SEVERITY_REVIEW,
                rule="unreadable_archive_member",
                relative_path=relative_path,
                message="压缩包成员无法读取，必须人工确认内容。",
            )
            continue
        yield from _scan_text_bytes(data, relative_path)


def _audit_tar_members(
    archive_name: str,
    archive: tarfile.TarFile,
) -> Iterable[AuditFinding]:
    """审计 TAR 成员，不提取文件到磁盘。"""

    for member in archive.getmembers():
        normalized_name = member.name.replace("\\", "/")
        relative_path = f"{archive_name}!/{normalized_name}"
        directory_finding, should_skip = _audit_archive_member_name(
            relative_path,
            normalized_name,
        )
        if directory_finding is not None:
            yield directory_finding
        if should_skip or not member.isfile():
            continue
        if not _is_text_file(Path(normalized_name)):
            continue
        try:
            member_file = archive.extractfile(member)
            if member_file is None:
                raise OSError("压缩包成员不是普通文件")
            data = member_file.read(MAX_CONTENT_SCAN_BYTES)
        except (OSError, tarfile.TarError):
            yield AuditFinding(
                severity=SEVERITY_REVIEW,
                rule="unreadable_archive_member",
                relative_path=relative_path,
                message="压缩包成员无法读取，必须人工确认内容。",
            )
            continue
        yield from _scan_text_bytes(data, relative_path)


def _audit_archive_member_name(
    relative_path: str,
    normalized_name: str,
) -> tuple[AuditFinding | None, bool]:
    """检查压缩包成员路径，返回目录级风险与是否停止深入检查。"""

    path_parts = tuple(
        part for part in normalized_name.split("/") if part
    )
    is_directory = normalized_name.endswith("/")
    directory_parts = path_parts if is_directory else path_parts[:-1]
    for index, part in enumerate(directory_parts):
        directory_path = "/".join(directory_parts[: index + 1])
        directory_finding = _audit_directory_name(
            f"{relative_path.split('!/', 1)[0]}!/{directory_path}",
            part,
        )
        if directory_finding is not None:
            return directory_finding, True

    if is_directory:
        return None, True

    file_name = path_parts[-1].casefold() if path_parts else ""
    if _is_sensitive_file_name(file_name):
        return (
            AuditFinding(
                severity=SEVERITY_BLOCK,
                rule="sensitive_archive_member",
                relative_path=relative_path,
                message="导出包中包含密钥、数据库、日志或凭据文件。",
            ),
            True,
        )

    if any(
        part.casefold().startswith(("edge-profile", "chrome-profile"))
        for part in path_parts
    ):
        return (
            AuditFinding(
                severity=SEVERITY_BLOCK,
                rule="browser_profile_in_archive",
                relative_path=relative_path,
                message="导出包中包含浏览器配置或登录资料。",
            ),
            True,
        )

    if file_name.endswith(".pdf"):
        return (
            AuditFinding(
                severity=SEVERITY_REVIEW,
                rule="pdf_requires_permission_review",
                relative_path=relative_path,
                message="论文全文可能受版权或个人隐私约束，公开前必须逐篇确认授权。",
            ),
            True,
        )

    return None, False


def _is_sensitive_file_name(normalized_name: str) -> bool:
    if normalized_name == ".env.example":
        return False
    if normalized_name in BLOCK_BROWSER_FILE_NAMES:
        return True
    return any(
        fnmatch.fnmatch(normalized_name, pattern)
        for pattern in BLOCK_FILE_PATTERNS
    )


def _is_skipped_directory(directory_name: str) -> bool:
    return directory_name.casefold() in SKIP_DIR_NAMES


def _is_text_file(path: Path) -> bool:
    if path.name.casefold() == ".env.example":
        return True
    if path.suffix.casefold() in TEXT_SUFFIXES:
        return True
    return path.name.startswith(".env")


def _is_supported_archive(path: Path) -> bool:
    normalized_name = path.name.casefold()
    return any(normalized_name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _deduplicate(findings: Iterable[AuditFinding]) -> list[AuditFinding]:
    return sorted(
        set(findings),
        key=lambda item: (
            item.relative_path,
            item.line_number or 0,
            item.rule,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计项目目录或参赛导出包中的敏感信息。"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="要审计的目录、文件、ZIP 或 TAR，默认当前目录。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出结果，报告中仍不包含密钥原文。",
    )
    parser.add_argument(
        "--no-content-scan",
        action="store_true",
        help="只检查文件名和目录名，不扫描文本内容。",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="存在需人工复核项时也返回失败退出码。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。阻断项返回 1，否则返回 0。"""

    arguments = _build_parser().parse_args(argv)
    try:
        report = audit_path(
            arguments.path,
            scan_content=not arguments.no_content_scan,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if arguments.json:
        print(
            json.dumps(
                report_as_dict(report),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_report(report))

    if report.blocking_count:
        return 1
    if arguments.fail_on_review and report.review_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
