"""Capture current runtime screenshots for the competition technical document."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence"
URL = "http://127.0.0.1:8502"
PAGES = (
    ("01-home", "首页"),
    ("02-workbench", "科研工作台"),
    ("03-literature-collection", "文献自动采集"),
    ("04-paper-library", "我的论文库"),
    ("05-academic-debate", "学术分歧分析"),
    ("06-knowledge-graph", "知识图谱"),
    ("07-learning-assessment", "AI 学习测评"),
    ("08-learning-report", "学习成长报告"),
    ("09-knowledge-summary", "知识图谱总结"),
    ("10-evidence-qa", "文献证据问答"),
)


def navigate(page, sidebar, label: str) -> None:
    combobox = sidebar.get_by_role("combobox").first
    combobox.click()
    page.wait_for_timeout(700)
    option = page.get_by_text(label, exact=True).last
    option.click()
    page.wait_for_timeout(5_000)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 920},
            device_scale_factor=1.5,
        )
        browser_errors: list[str] = []
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(6_000)
        sidebar = page.locator('[data-testid="stSidebar"]')

        pages: list[dict[str, object]] = []
        for prefix, label in PAGES:
            navigate(page, sidebar, label)
            body = page.locator("body").inner_text()
            screenshot = OUTPUT / f"{prefix}.png"
            page.screenshot(path=str(screenshot), full_page=False)
            pages.append(
                {
                    "page": label,
                    "screenshot": str(screenshot),
                    "has_page_load_error": "页面暂时无法加载" in body,
                    "has_operation_error": "当前操作未完成" in body,
                }
            )

        navigate(page, sidebar, "知识图谱")
        shell = page.locator(".kg-shell")
        if shell.count():
            shell.first.screenshot(
                path=str(OUTPUT / "06-knowledge-graph-component.png")
            )

        report = {
            "url": URL,
            "pages": pages,
            "browser_errors": browser_errors,
            "passed": all(
                not page_item["has_page_load_error"]
                and not page_item["has_operation_error"]
                for page_item in pages
            )
            and not browser_errors,
        }
        (OUTPUT / "capture-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
