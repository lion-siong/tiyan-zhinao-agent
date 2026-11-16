"""Capture live model-call and failure-fallback screenshots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent


def navigate(page, sidebar, label: str) -> None:
    sidebar.get_by_role("combobox").first.click()
    page.wait_for_timeout(700)
    page.get_by_text(label, exact=True).last.click()
    page.wait_for_timeout(5_000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=("model", "fallback"), required=True)
    args = parser.parse_args()
    output = ROOT / (
        "11-model-call.png"
        if args.mode == "model"
        else "12-failure-fallback.png"
    )
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 980},
            device_scale_factor=1.5,
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(6_000)
        sidebar = page.locator('[data-testid="stSidebar"]')
        navigate(page, sidebar, "智能文献研究")

        question = (
            "青少年体育锻炼干预与心肺耐力之间是否存在一致的研究结论？"
        )
        textarea = page.locator("textarea").first
        textarea.fill(question)
        page.get_by_role("button", name="检索并整理文献").click()
        page.wait_for_timeout(2_000)
        page.get_by_text("运行模式", exact=True).wait_for(
            state="visible",
            timeout=120_000,
        )
        page.wait_for_timeout(2_000)
        if args.mode == "fallback":
            warning_toggle = page.get_by_text(
                re.compile(r"检索提示")
            )
            if warning_toggle.count():
                warning_toggle.first.scroll_into_view_if_needed()
                warning_toggle.first.click()
                page.wait_for_timeout(1_000)
        body = page.locator("body").inner_text()
        page.screenshot(path=str(output), full_page=False)
        report = {
            "url": args.url,
            "mode": args.mode,
            "question": question,
            "has_run_mode": "运行模式" in body,
            "has_review": "文献综述与研究现状" in body,
            "has_fallback_warning": "Function Calling 未完成" in body,
            "uses_function_calling": "Function Calling" in body,
            "uses_rule_mode": "规则" in body,
            "page_error": "页面暂时无法加载" in body,
            "browser_errors": errors,
            "screenshot": str(output),
        }
        target = ROOT / (
            "model-call-capture.json"
            if args.mode == "model"
            else "failure-fallback-capture.json"
        )
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
    if args.mode == "model":
        return 0 if report["has_run_mode"] and not report["page_error"] else 1
    return 0 if (
        report["has_run_mode"]
        and report["has_fallback_warning"]
        and not report["page_error"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
