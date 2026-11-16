"""Run reproducibility checks inside an exported submission directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FILES = (
    "README.md",
    "app.py",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    ".env.example",
)
REQUIRED_DIRS = (
    "agents",
    "researchmind",
    "services",
    "tools",
    "tests",
    "docs",
    "knowledge_base",
)


def clean_runtime_artifacts(root: Path) -> list[str]:
    """Remove only known runtime directories created by the startup smoke test."""

    removed: list[str] = []
    exact_targets = (
        root / "data" / "user_papers",
        root / "knowledge_base" / "chroma",
        root / "knowledge_base_adolescent" / "backups",
        root / "knowledge_base_adolescent" / "chroma",
        root / "knowledge_base_adolescent" / "index",
        root / "knowledge_base_adolescent" / "pdfs",
    )
    for target in exact_targets:
        resolved = target.resolve()
        if root.resolve() not in resolved.parents:
            raise RuntimeError(f"拒绝清理项目根目录外路径：{resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
            removed.append(resolved.relative_to(root.resolve()).as_posix())
    for file_name in ("framework.json", "manifest.json"):
        target = (root / "knowledge_base_adolescent" / file_name).resolve()
        if root.resolve() not in target.parents:
            raise RuntimeError(f"拒绝清理项目根目录外文件：{target}")
        if target.is_file():
            target.unlink()
            removed.append(target.relative_to(root.resolve()).as_posix())
    return removed


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in {
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                "build",
                "backups",
            }
            for part in relative.parts
        ):
            continue
        if not path.is_file() or path.name in {
            "SUBMISSION_MANIFEST.json",
            "SUBMISSION_ACCEPTANCE.json",
        }:
            continue
        if path.suffix.casefold() in {".pyc", ".pyo", ".log", ".zip"}:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run_command(
    name: str,
    command: list[str],
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "return_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }


def run_startup_check(
    root: Path,
    environment: dict[str, str],
) -> dict[str, object]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(10)
    alive = process.poll() is None
    if alive:
        process.terminate()
    try:
        output, _ = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate(timeout=10)
    return {
        "name": "streamlit_startup",
        "command": command,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "process_alive_after_10_seconds": alive,
        "passed": alive,
        "stdout_tail": output[-12000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    missing = [
        name for name in REQUIRED_FILES if not (root / name).is_file()
    ] + [
        name for name in REQUIRED_DIRS if not (root / name).is_dir()
    ]
    if missing:
        raise SystemExit("提交包缺少必要文件或目录：" + "、".join(missing))

    source_hash = source_tree_sha256(root)
    with tempfile.TemporaryDirectory(prefix="tizhi-acceptance-") as temp:
        temp_root = Path(temp)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
                "RESEARCHMIND_DB_PATH": str(temp_root / "legacy.sqlite3"),
                "ADOLESCENT_RESEARCHMIND_DB_PATH": str(
                    temp_root / "adolescent.sqlite3"
                ),
                "ADOLESCENT_KNOWLEDGE_BASE_PATH": str(
                    temp_root / "knowledge_base"
                ),
            }
        )
        checks = [
            run_command(
                "core_imports",
                [
                    sys.executable,
                    "-c",
                    (
                        "import app, config, agents, researchmind.workflow, "
                        "researchmind.storage, services.rag_service, "
                        "tools.knowledge_base"
                    ),
                ],
                root,
                environment,
            ),
            run_command(
                "pytest",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(temp_root / "pytest"),
                    "--junitxml",
                    str(root / "SUBMISSION_TEST_RESULTS.xml"),
                ],
                root,
                environment,
            ),
            run_startup_check(root, environment),
        ]
    removed_runtime_artifacts = clean_runtime_artifacts(root)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "python": sys.version,
        "source_tree_sha256": source_hash,
        "required_files_present": True,
        "removed_runtime_artifacts": removed_runtime_artifacts,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "scope": (
            "在最终导出副本根目录执行核心模块导入、完整 pytest 和"
            "Streamlit 10 秒启动检查；运行数据库写入系统临时目录。"
        ),
        "not_covered": (
            "未调用真实大模型 API，未测试 OCR，未使用 541 页赛事样例，"
            "未执行人工科研判断金标准。"
        ),
    }
    output = root / "SUBMISSION_ACCEPTANCE.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
