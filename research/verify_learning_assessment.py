"""浏览器验收 AI 学习测评的真实组卷和页面错误状态。"""

from __future__ import annotations

import re
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = "http://127.0.0.1:8502"
OUTPUT = (
    Path(__file__).resolve().parent
    / "reports"
    / "ai-learning-assessment-page.png"
)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(6_000)

        sidebar = page.locator('[data-testid="stSidebar"]')
        combobox = sidebar.get_by_role("combobox")
        if combobox.count():
            combobox.first.click()
            page.wait_for_timeout(700)
        assessment_nav = page.get_by_text("AI 学习测评", exact=True)
        if assessment_nav.count():
            assessment_nav.first.click()
            page.wait_for_timeout(5_000)

        generate = page.get_by_role("button", name="生成新试卷")
        generate_clicked = False
        if generate.count():
            generate.first.click()
            generate_clicked = True
            page.wait_for_timeout(5_000)

        body_text = page.locator("body").inner_text()
        OUTPUT.write_bytes(page.screenshot(full_page=True))
        report = {
            "title": page.title(),
            "generate_clicked": generate_clicked,
            "body_has_page_title": "AI 学习测评" in body_text,
            "body_has_success": bool(
                re.search(r"已生成\s+\d+\s+道有来源的试题", body_text)
            ),
            "five_module_names_present": all(
                module in body_text
                for module in (
                    "基础知识理解",
                    "文献深度理解",
                    "文献对比分析",
                    "学术思辨训练",
                    "个性化错题与补学",
                )
            ),
            "raw_interventions_error": "'interventions'" in body_text,
            "page_load_error": "页面暂时无法加载" in body_text,
            "question_difficulty_labels": body_text.count("难度："),
            "browser_errors": errors,
            "screenshot": str(OUTPUT),
        }
        report["passed"] = all(
            (
                report["generate_clicked"],
                report["body_has_page_title"],
                report["body_has_success"],
                report["five_module_names_present"],
                not report["raw_interventions_error"],
                not report["page_load_error"],
                report["question_difficulty_labels"] >= 4,
                not errors,
            )
        )
        print(report)
        browser.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
