"""用 Playwright 对本地知识图谱页面做浏览器冒烟测试。"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


URL = "http://127.0.0.1:8502"
OUTPUT = (
    Path(__file__).resolve().parent
    / "reports"
    / "knowledge-graph-five-layer-page.png"
)
COMPONENT_OUTPUT = (
    Path(__file__).resolve().parent
    / "reports"
    / "knowledge-graph-five-layer-component.png"
)
OVERVIEW_OUTPUT = (
    Path(__file__).resolve().parent
    / "reports"
    / "knowledge-graph-five-layer-overview.png"
)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        browser_errors: list[str] = []
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                browser_errors.append(f"console:{message.text}")
                if message.type == "error"
                and "metrics config" not in message.text
                else None
            ),
        )
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(8_000)

        sidebar = page.locator('[data-testid="stSidebar"]')
        sidebar_text = sidebar.inner_text()
        current_page = sidebar.get_by_text("首页", exact=True)
        combobox = sidebar.get_by_role("combobox")
        if combobox.count():
            combobox.first.click()
            page.wait_for_timeout(800)
        elif current_page.count():
            current_page.first.click()
            page.wait_for_timeout(800)
        graph_nav = page.get_by_text("知识图谱", exact=True)
        if graph_nav.count():
            graph_nav.first.click()
            page.wait_for_timeout(5_000)

        page.wait_for_selector(".kg-node", timeout=20_000)
        page.wait_for_timeout(500)
        body_text = page.locator("body").inner_text()
        shell_locator = page.locator(".kg-shell")
        if shell_locator.count():
            OVERVIEW_OUTPUT.write_bytes(shell_locator.first.screenshot())
        sidebar_text = sidebar.inner_text()
        graph_nodes = page.locator(".kg-node")
        initial_graph_nodes = graph_nodes.count()
        theme_node = page.locator(
            '.kg-node[aria-label="研究主题：体育锻炼干预"]'
        ).first
        theme_clicked = False
        theme_label = ""
        if theme_node.count():
            theme_label = theme_node.get_attribute("aria-label") or ""
            theme_node.dispatch_event("click")
            page.wait_for_timeout(1_000)
            theme_clicked = True
        expanded_graph_nodes = graph_nodes.count()
        expanded_node_texts = graph_nodes.all_text_contents()[:16]
        method_filter = page.locator("#kg-method")
        method_options = method_filter.locator("option").count()
        method_filter.select_option(index=1)
        page.wait_for_timeout(500)
        method_filtered_nodes = graph_nodes.count()
        method_filter.select_option("all")
        page.wait_for_timeout(500)
        variable_toggle = page.locator("#kg-show-variables")
        hidden_variable_nodes = expanded_graph_nodes
        if variable_toggle.count():
            variable_toggle.uncheck()
            page.wait_for_timeout(500)
            hidden_variable_nodes = graph_nodes.count()
            variable_toggle.check()
            page.wait_for_timeout(500)
        direction_node = page.locator(
            '.kg-node[aria-label="研究方向：有氧运动"]'
        ).first
        direction_clicked = False
        if direction_node.count():
            direction_node.dispatch_event("click")
            page.wait_for_timeout(800)
            direction_clicked = True
        direction_graph_nodes = graph_nodes.count()
        paper_toggle = page.locator("#kg-show-papers")
        hidden_paper_nodes = direction_graph_nodes
        if paper_toggle.count():
            paper_toggle.uncheck()
            page.wait_for_timeout(500)
            hidden_paper_nodes = graph_nodes.count()
            paper_toggle.check()
            page.wait_for_timeout(500)
        variable_node = page.locator(
            '.kg-node[aria-label="研究变量：最大摄氧量"]'
        ).first
        variable_clicked = False
        if variable_node.count():
            variable_node.dispatch_event("click")
            page.wait_for_timeout(800)
            variable_clicked = True
        variable_graph_nodes = graph_nodes.count()
        detail_visible = page.locator("#kg-detail").is_visible()
        page.locator("#kg-overview").click()
        page.wait_for_timeout(700)
        health_theme = page.locator(
            '.kg-node[aria-label="研究主题：青少年体质健康评价"]'
        ).first
        health_theme_clicked = False
        if health_theme.count():
            health_theme.dispatch_event("click")
            page.wait_for_timeout(700)
            health_theme_clicked = True
        body_direction = page.locator(
            '.kg-node[aria-label="研究方向：身体成分"]'
        ).first
        body_direction_clicked = False
        if body_direction.count():
            body_direction.dispatch_event("click")
            page.wait_for_timeout(700)
            body_direction_clicked = True
        synthesis_node = page.locator(
            '.kg-node[aria-label="学术争议：体育锻炼与身体成分"]'
        ).first
        synthesis_clicked = False
        synthesis_detail_title = ""
        if synthesis_node.count():
            synthesis_node.dispatch_event("click")
            page.wait_for_timeout(700)
            synthesis_clicked = True
            synthesis_detail_title = page.locator("#kg-detail-title").inner_text()
        network_mode = page.locator('button[data-mode="network"]')
        network_clicked = False
        if network_mode.count():
            network_mode.first.click()
            page.wait_for_timeout(1_000)
            network_clicked = True
        network_graph_nodes = graph_nodes.count()
        fullscreen_button = page.locator("#kg-fullscreen")
        fullscreen_clicked = False
        if fullscreen_button.count():
            fullscreen_button.click()
            page.wait_for_timeout(500)
            fullscreen_clicked = page.evaluate("Boolean(document.fullscreenElement)")
            if fullscreen_clicked:
                page.evaluate("document.exitFullscreen()")
                page.wait_for_timeout(500)
        component_info = page.locator("body").evaluate(
            """
            (root) => {
              const tags = []
              const visit = (node) => {
                for (const el of node.querySelectorAll("*")) {
                  const tag = el.tagName.toLowerCase()
                  if (tag.includes("custom") || tag === "iframe") tags.push(tag)
                  if (el.shadowRoot) visit(el.shadowRoot)
                }
              }
              visit(root)
              return {tags, hasKgShell: document.documentElement.innerHTML.includes("kg-shell")}
            }
            """
        )
        component_count = len(component_info["tags"])
        empty_message = page.get_by_text("没有匹配的图谱节点", exact=False)
        empty_visible = (
            empty_message.first.is_visible() if empty_message.count() else False
        )
        OUTPUT.write_bytes(page.screenshot(full_page=True))
        graph_title = page.get_by_text("图谱科研洞察", exact=False)
        if graph_title.count():
            graph_title.first.scroll_into_view_if_needed()
            page.mouse.wheel(0, -780)
            page.wait_for_timeout(1_000)
        COMPONENT_OUTPUT.write_bytes(page.screenshot(full_page=False))
        report = {
            "title": page.title(),
            "body_has_research_map_name": "知识图谱" in body_text,
            "body_has_insights": "图谱科研洞察" in body_text,
            "body_has_directions": "研究方向" in body_text,
            "body_has_variables": "研究变量" in body_text,
            "component_fallback_visible": "交互图谱组件暂不可用" in body_text,
            "component_count": component_count,
            "component_tags": component_info["tags"][:20],
            "has_kg_shell": component_info["hasKgShell"],
            "initial_graph_nodes": initial_graph_nodes,
            "theme_clicked": theme_clicked,
            "theme_label": theme_label,
            "expanded_graph_nodes": expanded_graph_nodes,
            "expanded_node_texts": expanded_node_texts,
                "method_filtered_nodes": method_filtered_nodes,
                "hidden_variable_nodes": hidden_variable_nodes,
            "direction_clicked": direction_clicked,
            "direction_graph_nodes": direction_graph_nodes,
                "hidden_paper_nodes": hidden_paper_nodes,
            "variable_clicked": variable_clicked,
            "variable_graph_nodes": variable_graph_nodes,
            "detail_visible": detail_visible,
                "health_theme_clicked": health_theme_clicked,
                "body_direction_clicked": body_direction_clicked,
                "synthesis_clicked": synthesis_clicked,
                "synthesis_detail_title": synthesis_detail_title,
            "network_clicked": network_clicked,
            "network_graph_nodes": network_graph_nodes,
                "method_options": method_options,
                "fullscreen_clicked": fullscreen_clicked,
            "combobox_count": combobox.count(),
            "visible_empty_message": empty_visible,
            "browser_errors": browser_errors[:20],
            "screenshot": str(OUTPUT),
            "component_screenshot": str(COMPONENT_OUTPUT),
            "overview_screenshot": str(OVERVIEW_OUTPUT),
        }
        report["passed"] = all(
            (
                report["body_has_insights"],
                report["body_has_research_map_name"],
                not report["component_fallback_visible"],
                initial_graph_nodes >= 7,
                theme_clicked and expanded_graph_nodes > initial_graph_nodes,
                hidden_variable_nodes < expanded_graph_nodes,
                direction_clicked and direction_graph_nodes > 0,
                hidden_paper_nodes < direction_graph_nodes,
                variable_clicked and variable_graph_nodes > direction_graph_nodes,
                detail_visible,
                health_theme_clicked and body_direction_clicked,
                synthesis_clicked and "体育锻炼与身体成分" in synthesis_detail_title,
                network_clicked and network_graph_nodes > 0,
                method_options > 1,
                0 < method_filtered_nodes <= expanded_graph_nodes,
                fullscreen_clicked,
                not browser_errors,
                OVERVIEW_OUTPUT.is_file(),
            )
        )
        print(report)
        browser.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
