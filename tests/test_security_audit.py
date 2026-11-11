"""敏感信息审计工具的回归测试。"""

from __future__ import annotations

import zipfile
from pathlib import Path

from tools.security_audit import (
    SEVERITY_BLOCK,
    SEVERITY_REVIEW,
    audit_path,
    render_report,
)


def test_security_audit_flags_sensitive_files_without_exposing_values(
    tmp_path: Path,
) -> None:
    fake_secret = "sk-" + "a" * 32
    (tmp_path / ".env").write_text(
        f"OPENAI_API_KEY={fake_secret}\n",
        encoding="utf-8",
    )
    (tmp_path / "researchmind.sqlite3").write_bytes(b"not-a-real-db")
    (tmp_path / "streamlit.err.log").write_text(
        "request failed",
        encoding="utf-8",
    )
    browser_dir = tmp_path / "edge-profile-local"
    browser_dir.mkdir()
    (browser_dir / "Cookies").write_bytes(b"local browser data")

    report = audit_path(tmp_path)
    output = render_report(report)

    assert report.safe_for_export is False
    assert report.blocking_count >= 4
    assert fake_secret not in output
    assert ".env" in output
    assert "researchmind.sqlite3" in output


def test_security_audit_accepts_safe_source_and_env_example(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env.example").write_text(
        "OPENAI_API_KEY=\nZOTERO_API_KEY=\n",
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text(
        "import os\napi_key = os.getenv('OPENAI_API_KEY')\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "密钥只保存在本机 .env 文件。\n",
        encoding="utf-8",
    )

    report = audit_path(tmp_path)

    assert report.findings == ()
    assert report.safe_for_export is True


def test_security_audit_does_not_flag_config_variable_reference(
    tmp_path: Path,
) -> None:
    (tmp_path / "client.py").write_text(
        "api_key=current_config.semantic_scholar_api_key\n",
        encoding="utf-8",
    )

    report = audit_path(tmp_path)

    assert report.findings == ()


def test_security_audit_marks_document_and_vector_directories_for_review(
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "knowledge_base" / "pdfs"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-placeholder")
    chroma_dir = tmp_path / "knowledge_base" / "chroma"
    chroma_dir.mkdir()

    report = audit_path(tmp_path)

    assert report.blocking_count == 0
    assert report.review_count == 2
    assert all(
        finding.severity == SEVERITY_REVIEW
        for finding in report.findings
    )


def test_security_audit_marks_standalone_pdf_for_review(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-placeholder")

    report = audit_path(tmp_path)

    assert report.blocking_count == 0
    assert report.review_count == 1
    assert report.findings[0].rule == "pdf_requires_permission_review"


def test_security_audit_blocks_pytest_temporary_directory(
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / ".pytest_tmp"
    temp_dir.mkdir()
    (temp_dir / "stdout.txt").write_text("temporary output", encoding="utf-8")

    report = audit_path(tmp_path)

    assert report.safe_for_export is False
    assert any(
        finding.rule == "sensitive_directory"
        and finding.relative_path == ".pytest_tmp"
        for finding in report.findings
    )


def test_security_audit_inspects_zip_member_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(".env", "OPENAI_API_KEY=not-a-real-value")
        archive.writestr("data/researchmind.sqlite3", "not-a-real-db")
        archive.writestr("app.py", "print('safe')\n")

    report = audit_path(archive_path)

    assert report.safe_for_export is False
    assert report.blocking_count == 2
    assert any(
        finding.rule == "sensitive_archive_member"
        and finding.severity == SEVERITY_BLOCK
        for finding in report.findings
    )


def test_security_audit_scans_zip_text_without_exposing_secret(
    tmp_path: Path,
) -> None:
    fake_secret = "sk-" + "b" * 32
    archive_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "services/example.py",
            f'API_KEY = "{fake_secret}"\n',
        )

    report = audit_path(archive_path)
    output = render_report(report)

    assert report.safe_for_export is False
    assert any(
        finding.rule == "possible_platform_key"
        for finding in report.findings
    )
    assert fake_secret not in output
