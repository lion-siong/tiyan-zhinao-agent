"""ResearchMind 五层知识图谱与科研总结页面。

本模块只读取已入库论文、原文证据和现有 Agent 结果。图谱中的主题、内容、
论文和关系均由 :mod:`researchmind.knowledge_map` 生成，前端不补写任何学术
结论。当前实现不调用大模型，所有归纳都会明确标注为本地证据模式。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from researchmind.knowledge_map import (
    CORE_DIRECTION,
    CORE_DIRECTION_ID,
    INSUFFICIENT_EVIDENCE_TEXT,
    RETRIEVAL_SCOPE_NOTE,
    THEME_BY_ID,
    THEMES,
    KnowledgeFinding,
    KnowledgeMap,
    KnowledgeMapEdge,
    KnowledgeMapNode,
    KnowledgeMapPaper,
    build_knowledge_map,
    build_markdown_report,
)
from researchmind.knowledge_map_component import (
    render_knowledge_graph_component,
)
from researchmind.analysis_status_ui import (
    clear_analysis_session_state,
    render_analysis_update_status,
    schedule_for_domain,
)
from researchmind.library_state import graph_fingerprint
from researchmind.models import EvidenceRef, ResearchDomain
from researchmind.storage import ResearchMindStore
from researchmind.workflow import (
    DEFAULT_DEBATE_TOPIC,
    ResearchMindWorkflow,
)

LOGGER = logging.getLogger(__name__)


GRAPH_HTML = """
<section class="kg-shell" aria-label="五层知识图谱">
  <header class="kg-header">
    <div>
      <div class="kg-eyebrow">EVIDENCE KNOWLEDGE MAP</div>
  <h2>五层知识图谱</h2>
      <p>默认展示核心方向与研究主题；单击展开，双击查看论文与原文证据。</p>
    </div>
    <div class="kg-toolbar" role="toolbar" aria-label="知识地图工具">
      <label class="kg-search">
        <span>搜索</span>
        <input id="kg-search" type="search" placeholder="论文标题或研究主题" autocomplete="off" />
      </label>
      <label class="kg-select-label">
        <span>主题筛选</span>
        <select id="kg-theme-filter"></select>
      </label>
      <label class="kg-toggle">
        <input id="kg-relations" type="checkbox" />
        <span>论文关联</span>
      </label>
      <div class="kg-button-row">
        <button id="kg-overview" type="button" title="返回总览">总览</button>
        <button id="kg-collapse" type="button" title="折叠全部分支">折叠</button>
        <button id="kg-zoom-out" type="button" aria-label="缩小">−</button>
        <button id="kg-zoom-in" type="button" aria-label="放大">+</button>
      </div>
    </div>
  </header>
  <div class="kg-stage">
    <svg id="kg-svg" role="img" aria-label="核心领域、主题、变量与问题、论文证据及共识争议的五层关系图"></svg>
    <div id="kg-empty" class="kg-empty" hidden></div>
    <aside id="kg-detail" class="kg-detail" hidden aria-live="polite"></aside>
  </div>
  <footer class="kg-footer">
    <div class="kg-legend">
      <span><i class="dot dot-core"></i>核心方向</span>
      <span><i class="dot dot-theme"></i>研究主题</span>
      <span><i class="dot dot-content"></i>研究内容</span>
      <span><i class="dot dot-paper"></i>论文</span>
      <span><i class="line dashed"></i>有证据的关联</span>
    </div>
    <span id="kg-status" class="kg-status"></span>
  </footer>
</section>
"""


GRAPH_CSS = """
:host {
  color: var(--st-text-color, #182235);
  font-family: var(--st-font, "Microsoft YaHei", "Noto Sans SC", sans-serif);
}
* { box-sizing: border-box; }
.kg-shell {
  overflow: hidden;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 14px;
  background: var(--st-secondary-background-color, #ffffff);
}
.kg-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  padding: 20px 22px 18px;
  border-bottom: 1px solid var(--st-border-color, #d7e0ea);
  background:
    linear-gradient(90deg, rgba(36, 87, 214, 0.08), transparent 48%),
    var(--st-secondary-background-color, #ffffff);
}
.kg-eyebrow {
  margin-bottom: 5px;
  color: var(--st-blue-color, #2457d6);
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.12em;
}
.kg-header h2 {
  margin: 0;
  color: var(--st-text-color, #182235);
  font-size: 1.35rem;
  line-height: 1.35;
}
.kg-header p {
  margin: 7px 0 0;
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.84rem;
}
.kg-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 9px;
}
.kg-search,
.kg-select-label {
  display: grid;
  gap: 4px;
  min-width: 166px;
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.7rem;
  font-weight: 650;
}
.kg-select-label { min-width: 142px; }
.kg-search input,
.kg-select-label select {
  width: 100%;
  height: 36px;
  padding: 0 11px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 8px;
  outline: none;
  background: var(--st-background-color, #f6f8fb);
  color: var(--st-text-color, #182235);
  font: inherit;
  font-size: 0.8rem;
}
.kg-search input:focus,
.kg-select-label select:focus {
  border-color: var(--st-primary-color, #2457d6);
  box-shadow: 0 0 0 3px rgba(36, 87, 214, 0.12);
}
.kg-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 10px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 8px;
  background: var(--st-background-color, #f6f8fb);
  color: var(--st-text-color, #182235);
  font-size: 0.78rem;
  cursor: pointer;
}
.kg-toggle input { accent-color: var(--st-primary-color, #2457d6); }
.kg-button-row { display: inline-flex; gap: 6px; }
.kg-button-row button {
  height: 36px;
  min-width: 38px;
  padding: 0 11px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 8px;
  background: var(--st-background-color, #f6f8fb);
  color: var(--st-text-color, #182235);
  font: inherit;
  font-size: 0.78rem;
  font-weight: 650;
  cursor: pointer;
}
.kg-button-row button:hover,
.kg-button-row button:focus-visible {
  border-color: var(--st-primary-color, #2457d6);
  color: var(--st-primary-color, #2457d6);
}
.kg-stage {
  position: relative;
  min-height: 590px;
  background:
    linear-gradient(var(--st-grid-line, rgba(91, 107, 130, 0.06)) 1px, transparent 1px),
    linear-gradient(90deg, var(--st-grid-line, rgba(91, 107, 130, 0.06)) 1px, transparent 1px);
  background-size: 28px 28px;
}
#kg-svg {
  display: block;
  width: 100%;
  min-height: 590px;
  touch-action: none;
  cursor: grab;
}
#kg-svg.dragging { cursor: grabbing; }
.kg-edge {
  fill: none;
  stroke: rgba(91, 107, 130, 0.42);
  stroke-width: 1.35;
}
.kg-edge.hierarchy { stroke: rgba(91, 107, 130, 0.5); }
.kg-edge.theme_relation,
.kg-edge.paper_relation {
  stroke: rgba(165, 90, 10, 0.72);
  stroke-dasharray: 6 5;
  stroke-width: 1.45;
}
.kg-edge.is-active {
  stroke: var(--st-blue-color, #2457d6);
  stroke-width: 2.7;
  opacity: 1;
}
.kg-edge.is-muted { opacity: 0.32; }
.kg-node { cursor: pointer; }
.kg-node rect {
  stroke-width: 1.2;
  filter: drop-shadow(0 5px 8px rgba(15, 31, 54, 0.08));
  transition: filter 140ms ease, stroke-width 140ms ease;
}
.kg-node:hover rect,
.kg-node:focus rect {
  stroke-width: 2.2;
  filter: drop-shadow(0 7px 12px rgba(15, 31, 54, 0.14));
}
.kg-node.level-1 rect {
  fill: var(--st-blue-color, #2457d6);
  stroke: var(--st-blue-color, #2457d6);
}
.kg-node.level-2 rect {
  fill: color-mix(in srgb, var(--st-green-color, #147d64) 11%, var(--st-secondary-background-color, #ffffff));
  stroke: color-mix(in srgb, var(--st-green-color, #147d64) 55%, transparent);
}
.kg-node.level-3 rect {
  fill: var(--st-secondary-background-color, #ffffff);
  stroke: color-mix(in srgb, var(--st-gray-color, #5b6b82) 48%, transparent);
}
.kg-node.level-4 rect {
  fill: color-mix(in srgb, var(--st-orange-color, #a55a0a) 8%, var(--st-secondary-background-color, #ffffff));
  stroke: color-mix(in srgb, var(--st-orange-color, #a55a0a) 45%, transparent);
}
.kg-node.is-selected rect {
  stroke: var(--st-primary-color, #2457d6);
  stroke-width: 3;
}
.kg-node.is-muted { opacity: 0.42; }
.kg-node .kg-label {
  display: -webkit-box;
  overflow: hidden;
  color: var(--st-text-color, #182235);
  font-size: 12.5px;
  font-weight: 720;
  line-height: 1.35;
  text-overflow: ellipsis;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}
.kg-node.level-1 .kg-label { color: #ffffff; font-size: 15px; }
.kg-node .kg-meta {
  display: block;
  margin-top: 6px;
  color: var(--st-gray-color, #5b6b82);
  font-size: 10.5px;
  font-weight: 550;
  white-space: nowrap;
}
.kg-node.level-1 .kg-meta { color: rgba(255, 255, 255, 0.8); }
.kg-node .kg-expand {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  margin-right: 5px;
  border-radius: 50%;
  background: rgba(36, 87, 214, 0.1);
  color: var(--st-blue-color, #2457d6);
  font-size: 12px;
  font-weight: 800;
}
.kg-node.level-1 .kg-expand {
  background: rgba(255, 255, 255, 0.17);
  color: #ffffff;
}
.kg-detail {
  position: absolute;
  top: 18px;
  right: 18px;
  z-index: 4;
  display: grid;
  width: min(390px, calc(100% - 36px));
  max-height: calc(100% - 36px);
  overflow: auto;
  padding: 18px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 12px;
  background: var(--st-secondary-background-color, #ffffff);
  box-shadow: 0 18px 50px rgba(15, 31, 54, 0.2);
}
.kg-detail[hidden] { display: none; }
.kg-detail-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}
.kg-detail h3 {
  margin: 0;
  color: var(--st-text-color, #182235);
  font-size: 1rem;
  line-height: 1.45;
}
.kg-detail-close {
  flex: 0 0 auto;
  width: 30px;
  height: 30px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 8px;
  background: transparent;
  color: var(--st-text-color, #182235);
  cursor: pointer;
}
.kg-detail p {
  margin: 10px 0;
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.8rem;
  line-height: 1.65;
}
.kg-paper-card,
.kg-evidence-card {
  display: grid;
  gap: 5px;
  margin-top: 10px;
  padding: 11px 12px;
  border: 1px solid var(--st-border-color, #d7e0ea);
  border-radius: 9px;
  background: var(--st-background-color, #f6f8fb);
}
.kg-paper-card strong,
.kg-evidence-card strong {
  color: var(--st-text-color, #182235);
  font-size: 0.79rem;
  line-height: 1.45;
}
.kg-paper-card span,
.kg-evidence-card span {
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.72rem;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.kg-evidence-card blockquote {
  margin: 3px 0 0;
  padding-left: 9px;
  border-left: 3px solid color-mix(in srgb, var(--st-blue-color, #2457d6) 42%, transparent);
  color: var(--st-text-color, #182235);
  font-size: 0.74rem;
  line-height: 1.58;
}
.kg-empty {
  position: absolute;
  inset: 18px;
  place-items: center;
  padding: 24px;
  border: 1px dashed var(--st-border-color, #d7e0ea);
  border-radius: 12px;
  background: color-mix(in srgb, var(--st-secondary-background-color, #ffffff) 94%, transparent);
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.86rem;
  line-height: 1.7;
  text-align: center;
}
.kg-empty[hidden] { display: none; }
.kg-empty:not([hidden]) { display: grid; }
.kg-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 11px 20px;
  border-top: 1px solid var(--st-border-color, #d7e0ea);
  background: var(--st-background-color, #f6f8fb);
}
.kg-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 13px;
  color: var(--st-gray-color, #5b6b82);
  font-size: 0.69rem;
}
.kg-legend span { display: inline-flex; align-items: center; gap: 5px; }
.dot { width: 8px; height: 8px; border-radius: 50%; }
.dot-core { background: var(--st-blue-color, #2457d6); }
.dot-theme { background: var(--st-green-color, #147d64); }
.dot-content { background: var(--st-gray-color, #5b6b82); }
.dot-paper { background: var(--st-orange-color, #a55a0a); }
.line { display: inline-block; width: 20px; height: 0; border-top: 1px solid currentColor; }
.line.dashed { border-top-style: dashed; color: var(--st-orange-color, #a55a0a); }
.kg-status { flex: 0 0 auto; color: var(--st-gray-color, #5b6b82); font-size: 0.68rem; }
@media (max-width: 900px) {
  .kg-header { align-items: stretch; flex-direction: column; }
  .kg-toolbar { justify-content: flex-start; }
  .kg-search { flex: 1 1 210px; }
  .kg-footer { align-items: flex-start; flex-direction: column; }
}
"""


GRAPH_JS = """
export default function ({ data, parentElement }) {
  const shell = parentElement.querySelector(".kg-shell")
  if (!shell || !data) return

  const svg = parentElement.querySelector("#kg-svg")
  const empty = parentElement.querySelector("#kg-empty")
  const detail = parentElement.querySelector("#kg-detail")
  const status = parentElement.querySelector("#kg-status")
  const searchInput = parentElement.querySelector("#kg-search")
  const themeFilter = parentElement.querySelector("#kg-theme-filter")
  const relationToggle = parentElement.querySelector("#kg-relations")

  const NS = "http://www.w3.org/2000/svg"
  const nodes = Array.isArray(data.nodes) ? data.nodes : []
  const edges = Array.isArray(data.edges) ? data.edges : []
  const papers = Array.isArray(data.papers) ? data.papers : []
  const themes = Array.isArray(data.themes) ? data.themes : []
  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const paperById = new Map(papers.map((paper) => [paper.id, paper]))

  let expandedThemes = new Set()
  let expandedContent = new Set()
  let selectedId = data.core_id
  let zoom = 1
  let panX = 0
  let panY = 0
  let dragging = false
  let dragStart = null

  const viewport = document.createElementNS(NS, "g")
  viewport.setAttribute("id", "kg-viewport")
  svg.appendChild(viewport)

  const defs = document.createElementNS(NS, "defs")
  const marker = document.createElementNS(NS, "marker")
  marker.setAttribute("id", "kg-arrow")
  marker.setAttribute("viewBox", "0 0 10 10")
  marker.setAttribute("refX", "8")
  marker.setAttribute("refY", "5")
  marker.setAttribute("markerWidth", "5")
  marker.setAttribute("markerHeight", "5")
  marker.setAttribute("orient", "auto-start-reverse")
  const markerPath = document.createElementNS(NS, "path")
  markerPath.setAttribute("d", "M 0 0 L 10 5 L 0 10 z")
  markerPath.setAttribute("fill", "rgba(91, 107, 130, 0.68)")
  marker.appendChild(markerPath)
  defs.appendChild(marker)
  svg.prepend(defs)

  function makeOption(value, label) {
    const option = document.createElement("option")
    option.value = value
    option.textContent = label
    return option
  }

  themeFilter.appendChild(makeOption("all", "全部研究主题"))
  for (const theme of themes) {
    themeFilter.appendChild(makeOption(theme.id, `${theme.label}（${theme.paper_count}）`))
  }

  function ancestorsOf(node) {
    const result = []
    const queue = [...(node.parent_ids || [])]
    const seen = new Set()
    while (queue.length) {
      const id = queue.shift()
      if (seen.has(id)) continue
      seen.add(id)
      const parent = nodeById.get(id)
      if (!parent) continue
      result.push(parent)
      queue.push(...(parent.parent_ids || []))
    }
    return result
  }

  function pathIdsFor(nodeId) {
    const node = nodeById.get(nodeId)
    if (!node) return new Set()
    return new Set([nodeId, ...ancestorsOf(node).map((item) => item.id)])
  }

  function searchMatches(query) {
    const normalized = query.trim().toLocaleLowerCase("zh-CN")
    if (!normalized) return new Set()
    const ids = new Set()
    for (const node of nodes) {
      const haystack = `${node.label} ${node.description} ${node.type}`.toLocaleLowerCase("zh-CN")
      if (haystack.includes(normalized)) {
        ids.add(node.id)
        for (const ancestor of ancestorsOf(node)) ids.add(ancestor.id)
      }
    }
    for (const paper of papers) {
      const paperNode = nodeById.get(paper.node_id)
      const haystack = [
        paper.title,
        paper.authors,
        paper.year,
        paper.doi,
        paper.journal,
      ].join(" ").toLocaleLowerCase("zh-CN")
      if (!haystack.includes(normalized)) continue
      if (paperNode) {
        ids.add(paperNode.id)
        for (const ancestor of ancestorsOf(paperNode)) ids.add(ancestor.id)
      }
    }
    return ids
  }

  function visibleNodeIds() {
    const query = searchInput.value || ""
    const matches = searchMatches(query)
    const visible = new Set([data.core_id])
    const selectedTheme = themeFilter.value

    if (query.trim()) {
      return new Set(
        [...matches].filter((nodeId) => nodeById.has(nodeId)),
      )
    }

    for (const theme of themes) {
      if (selectedTheme !== "all" && theme.id !== selectedTheme) continue
      visible.add(theme.id)
      if (expandedThemes.has(theme.id)) {
        const contentNodes = nodes.filter((node) => node.level === 3 && (node.parent_ids || []).includes(theme.id))
        for (const content of contentNodes) {
          visible.add(content.id)
          if (expandedContent.has(content.id)) {
            for (const node of nodes) {
              if (node.level !== 4) continue
              if (!(node.parent_ids || []).includes(content.id)) continue
              visible.add(node.id)
            }
          }
        }
      }
    }
    return visible
  }

  function nodeSize(node) {
    if (node.level === 1) return { width: 220, height: 82 }
    if (node.level === 2) return { width: 224, height: 70 }
    if (node.level === 3) return { width: 182, height: 64 }
    return { width: 280, height: 72 }
  }

  function buildLayout(visible) {
    const positions = new Map()
    const visibleThemes = themes.filter((theme) => visible.has(theme.id))
    let cursorY = 62
    let maxRight = 1320
    const themeCenters = []

    for (const theme of visibleThemes) {
      const themeNode = nodeById.get(theme.id)
      if (!themeNode) continue
      const contents = nodes.filter(
        (node) => node.level === 3 && visible.has(node.id) && (node.parent_ids || []).includes(theme.id),
      )
      const groupTop = cursorY
      const contentCenters = []

      if (!contents.length) {
        const themeY = cursorY + 38
        positions.set(theme.id, { x: 420, y: themeY, ...nodeSize(themeNode) })
        cursorY += 94
        themeCenters.push(themeY)
        continue
      }

      for (const content of contents) {
        const contentPapers = nodes.filter(
          (node) =>
            node.level === 4 &&
            visible.has(node.id) &&
            (node.parent_ids || []).includes(content.id),
        )
        const contentY = cursorY + 34
        positions.set(content.id, { x: 720, y: contentY, ...nodeSize(content) })
        contentCenters.push(contentY)
        let paperCursor = contentY - 22
        for (const paperNode of contentPapers) {
          if (!positions.has(paperNode.id)) {
            positions.set(paperNode.id, {
              x: 1080,
              y: paperCursor,
              ...nodeSize(paperNode),
            })
          }
          paperCursor += 78
          maxRight = Math.max(maxRight, 1230)
        }
        cursorY += Math.max(88, contentPapers.length * 78 + 18)
      }

      const themeY = contentCenters.length
        ? contentCenters.reduce((sum, value) => sum + value, 0) / contentCenters.length
        : groupTop + 38
      positions.set(theme.id, { x: 420, y: themeY, ...nodeSize(themeNode) })
      themeCenters.push(themeY)
      cursorY += 18
    }

    const coreNode = nodeById.get(data.core_id)
    if (coreNode) {
      const coreY = themeCenters.length
        ? themeCenters.reduce((sum, value) => sum + value, 0) / themeCenters.length
        : 270
      positions.set(data.core_id, { x: 120, y: coreY, ...nodeSize(coreNode) })
    }
    return { positions, height: Math.max(640, cursorY + 48), width: maxRight }
  }

  function applyViewport() {
    viewport.setAttribute(
      "transform",
      `translate(${panX} ${panY}) scale(${zoom})`,
    )
  }

  function edgePath(source, target) {
    const x1 = source.x + source.width / 2
    const y1 = source.y
    const x2 = target.x - target.width / 2
    const y2 = target.y
    const control = Math.max(42, (x2 - x1) * 0.48)
    return `M ${x1} ${y1} C ${x1 + control} ${y1}, ${x2 - control} ${y2}, ${x2} ${y2}`
  }

  function relationSummary(node) {
    const paperCount = Number(node.paper_count || 0)
    const evidenceCount = Number(node.evidence_count || 0)
    const parts = [`${paperCount} 篇论文`, `${evidenceCount} 条证据`]
    if ((node.child_ids || []).length || (node.level === 2 && paperCount > 0)) {
      parts.push("单击展开")
    }
    return parts.join(" · ")
  }

  function renderNode(node, position, pathIds, selected) {
    const group = document.createElementNS(NS, "g")
    group.setAttribute("class", `kg-node level-${node.level}${selected ? " is-selected" : ""}`)
    group.setAttribute("transform", `translate(${position.x} ${position.y})`)
    group.setAttribute("tabindex", "0")
    group.setAttribute("role", "button")
    group.setAttribute("aria-label", `${node.type}：${node.label}`)
    group.dataset.nodeId = node.id

    const rect = document.createElementNS(NS, "rect")
    rect.setAttribute("x", String(-position.width / 2))
    rect.setAttribute("y", String(-position.height / 2))
    rect.setAttribute("width", String(position.width))
    rect.setAttribute("height", String(position.height))
    rect.setAttribute("rx", node.level === 1 ? "12" : "9")
    group.appendChild(rect)

    const foreign = document.createElementNS(NS, "foreignObject")
    foreign.setAttribute("x", String(-position.width / 2 + 12))
    foreign.setAttribute("y", String(-position.height / 2 + 8))
    foreign.setAttribute("width", String(position.width - 24))
    foreign.setAttribute("height", String(position.height - 16))

    const content = document.createElement("div")
    content.setAttribute("xmlns", "http://www.w3.org/1999/xhtml")
    content.style.height = "100%"
    content.style.display = "flex"
    content.style.flexDirection = "column"
    content.style.justifyContent = "center"
    content.style.pointerEvents = "none"

    const label = document.createElement("div")
    label.className = "kg-label"
    label.textContent = node.label
    content.appendChild(label)

    const meta = document.createElement("span")
    meta.className = "kg-meta"
    const hasChildren = (node.child_ids || []).length > 0
    const expanded =
      (node.level === 2 && expandedThemes.has(node.id)) ||
      (node.level === 3 && expandedContent.has(node.id))
    if (hasChildren) {
      const expand = document.createElement("i")
      expand.className = "kg-expand"
      expand.textContent = expanded ? "−" : "+"
      meta.appendChild(expand)
    }
    meta.appendChild(document.createTextNode(relationSummary(node)))
    content.appendChild(meta)
    foreign.appendChild(content)
    group.appendChild(foreign)

    group.onclick = (event) => {
      event.stopPropagation()
      selectedId = node.id
      if (node.level === 1) {
        expandedThemes.clear()
        expandedContent.clear()
      } else if (node.level === 2 && (node.child_ids || []).length) {
        if (expandedThemes.has(node.id)) expandedThemes.delete(node.id)
        else expandedThemes.add(node.id)
      } else if (node.level === 3 && (node.child_ids || []).length) {
        if (expandedContent.has(node.id)) expandedContent.delete(node.id)
        else expandedContent.add(node.id)
      }
      render()
    }
    group.ondblclick = (event) => {
      event.stopPropagation()
      showDetail(node.id)
    }
    group.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault()
        group.click()
      }
    }
    if (!pathIds.has(node.id) && selectedId && selectedId !== data.core_id) {
      group.classList.add("is-muted")
    }
    viewport.appendChild(group)
  }

  function render() {
    while (viewport.firstChild) viewport.removeChild(viewport.firstChild)
    const visible = visibleNodeIds()
    const layout = buildLayout(visible)
    const pathIds = pathIdsFor(selectedId)
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`)
    svg.setAttribute("height", String(Math.max(590, Math.min(760, layout.height))))

    const showRelations = relationToggle.checked
    for (const edge of edges) {
      if (!visible.has(edge.source) || !visible.has(edge.target)) continue
      if (edge.kind !== "hierarchy" && !showRelations) continue
      const source = layout.positions.get(edge.source)
      const target = layout.positions.get(edge.target)
      if (!source || !target) continue
      const path = document.createElementNS(NS, "path")
      const active =
        pathIds.has(edge.source) &&
        pathIds.has(edge.target) &&
        (selectedId === edge.source || selectedId === edge.target)
      path.setAttribute(
        "class",
        `kg-edge ${edge.kind}${active ? " is-active" : ""}`,
      )
      path.setAttribute("d", edgePath(source, target))
      path.setAttribute("marker-end", "url(#kg-arrow)")
      const title = document.createElementNS(NS, "title")
      title.textContent = `${edge.relation} · ${edge.evidence_count} 条证据`
      path.appendChild(title)
      viewport.appendChild(path)
    }

    for (const node of nodes) {
      const position = layout.positions.get(node.id)
      if (!position) continue
      renderNode(node, position, pathIds, node.id === selectedId)
    }
    applyViewport()
    const visiblePapers = [...visible].filter((id) => nodeById.get(id)?.level === 4).length
    status.textContent = visiblePapers
      ? `当前显示 ${visible.size} 个节点，其中 ${visiblePapers} 篇论文`
      : `当前显示 ${visible.size} 个节点 · 双击节点查看详情`
    empty.hidden = visible.size > 0
    if (!visible.size) {
      empty.textContent = "没有符合当前筛选条件的节点。"
    }
  }

  function clearNode(element) {
    while (element.firstChild) element.removeChild(element.firstChild)
  }

  function appendText(parent, tag, text, className = "") {
    const element = document.createElement(tag)
    if (className) element.className = className
    element.textContent = text
    parent.appendChild(element)
    return element
  }

  function showDetail(nodeId) {
    const node = nodeById.get(nodeId)
    if (!node) return
    clearNode(detail)
    detail.hidden = false

    const head = document.createElement("div")
    head.className = "kg-detail-head"
    appendText(head, "h3", node.label)
    const close = document.createElement("button")
    close.type = "button"
    close.className = "kg-detail-close"
    close.textContent = "×"
    close.setAttribute("aria-label", "关闭详情")
    close.onclick = () => {
      detail.hidden = true
    }
    head.appendChild(close)
    detail.appendChild(head)
    appendText(detail, "p", `${node.type} · ${relationSummary(node)}`)
    appendText(detail, "p", node.description || "暂无节点说明。")

    const relatedPapers = (node.paper_ids || [])
      .map((paperId) => paperById.get(paperId))
      .filter(Boolean)
    if (!relatedPapers.length) {
      appendText(
        detail,
        "p",
        "当前论文库中，该分支没有通过范围核验且可定位原文证据的论文。证据不足，待进一步验证。",
      )
    }
    for (const paper of relatedPapers) {
      const card = document.createElement("div")
      card.className = "kg-paper-card"
      appendText(card, "strong", paper.title)
      appendText(
        card,
        "span",
        `${paper.authors} · ${paper.year || "年份待核对"} · ${paper.journal || "期刊待核对"}`,
      )
      appendText(card, "span", `DOI：${paper.doi || "未提供"}`)
      appendText(card, "span", `来源：${paper.source || "未提供"}`)
      appendText(card, "span", `PDF：${paper.pdf_path || "未提供"}`)
      detail.appendChild(card)
    }

    const evidence = Array.isArray(node.evidence) ? node.evidence : []
    for (const item of evidence.slice(0, 5)) {
      const card = document.createElement("div")
      card.className = "kg-evidence-card"
      appendText(card, "strong", `${item.document_title} · 第 ${item.page_number} 页`)
      appendText(card, "span", item.file_name)
      const quote = document.createElement("blockquote")
      quote.textContent = item.quote
      card.appendChild(quote)
      detail.appendChild(card)
    }
    if (node.evidence_count > evidence.length && evidence.length) {
      appendText(
        detail,
        "p",
        `为保持页面流畅，此处展示前 ${evidence.length} 条代表性证据；完整证据可在“知识图谱总结”页面查看。`,
      )
    }
  }

  parentElement.querySelector("#kg-overview").onclick = () => {
    expandedThemes = new Set()
    expandedContent = new Set()
    selectedId = data.core_id
    themeFilter.value = "all"
    searchInput.value = ""
    zoom = 1
    panX = 0
    panY = 0
    detail.hidden = true
    render()
  }

  parentElement.querySelector("#kg-collapse").onclick = () => {
    expandedThemes = new Set()
    expandedContent = new Set()
    selectedId = data.core_id
    detail.hidden = true
    render()
  }

  parentElement.querySelector("#kg-zoom-in").onclick = () => {
    zoom = Math.min(2.2, zoom + 0.12)
    applyViewport()
  }

  parentElement.querySelector("#kg-zoom-out").onclick = () => {
    zoom = Math.max(0.55, zoom - 0.12)
    applyViewport()
  }

  searchInput.oninput = () => render()
  themeFilter.onchange = () => {
    expandedThemes.clear()
    expandedContent.clear()
    render()
  }
  relationToggle.onchange = () => render()

  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault()
      const delta = event.deltaY > 0 ? -0.08 : 0.08
      zoom = Math.max(0.55, Math.min(2.2, zoom + delta))
      applyViewport()
    },
    { passive: false },
  )

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest?.(".kg-node")) return
    dragging = true
    dragStart = { x: event.clientX, y: event.clientY, panX, panY }
    svg.classList.add("dragging")
    svg.setPointerCapture?.(event.pointerId)
  })
  svg.addEventListener("pointermove", (event) => {
    if (!dragging || !dragStart) return
    panX = dragStart.panX + (event.clientX - dragStart.x)
    panY = dragStart.panY + (event.clientY - dragStart.y)
    applyViewport()
  })
  svg.addEventListener("pointerup", () => {
    dragging = false
    dragStart = null
    svg.classList.remove("dragging")
  })
  svg.addEventListener("pointercancel", () => {
    dragging = false
    dragStart = null
    svg.classList.remove("dragging")
  })

  render()
  return () => {
    searchInput.oninput = null
    themeFilter.onchange = null
    relationToggle.onchange = null
  }
}
"""


_GRAPH_COMPONENT = st.components.v2.component(
    "researchmind_hierarchical_knowledge_map_v1",
    html=GRAPH_HTML,
    css=GRAPH_CSS,
    js=GRAPH_JS,
)


@st.cache_data(max_entries=24, show_spinner=False)
def _load_knowledge_map(
    db_path: str,
    domain_id: str,
    domain_name: str,
    library_version: str,
) -> KnowledgeMap:
    """从 SQLite 读取真实数据并构建图谱，论文库版本变化时自动换键。"""

    del library_version  # 只作为缓存键，数据本身由数据库读取。
    store = ResearchMindStore(db_path)
    papers = store.list_papers(domain_id)
    analyses = {
        paper.paper_id: store.get_analysis(paper.paper_id)
        for paper in papers
    }
    evidence = {
        paper.paper_id: store.list_evidence(paper.paper_id)
        for paper in papers
    }
    return build_knowledge_map(
        domain_name=domain_name,
        papers=papers,
        analyses=analyses,
        evidence_by_paper=evidence,
        debate=store.latest_debate(domain_id),
        synthesis=store.get_latest_synthesis(domain_id),
    )


def get_knowledge_map(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
    *,
    force_refresh: bool = False,
) -> KnowledgeMap:
    """返回当前领域的可追溯知识地图。"""

    if force_refresh:
        _load_knowledge_map.clear()
    db_path = Path(workflow.store.db_path).resolve()
    return _load_knowledge_map(
        str(db_path),
        domain.domain_id,
        domain.name,
        graph_fingerprint(workflow.store, domain.domain_id),
    )


def build_graph_payload(knowledge_map: KnowledgeMap) -> dict[str, Any]:
    """把图谱数据转换为前端可读结构。

    默认 ``nodes``、``edges`` 和 ``papers`` 只包含核心方向与六个主题。
    完整五层数据放在 ``all_*`` 字段中，由浏览器按用户展开动作逐步显示。
    """

    child_ids: dict[str, list[str]] = {}
    for node in knowledge_map.nodes:
        for parent_id in node.parent_ids:
            child_ids.setdefault(parent_id, []).append(node.node_id)

    theme_nodes = [
        node for node in knowledge_map.nodes if node.level == 2
    ]
    default_node_ids = {
        node.node_id for node in knowledge_map.nodes if node.level <= 2
    }
    all_node_payload = [
        _graph_node_payload(node, child_ids.get(node.node_id, ()))
        for node in knowledge_map.nodes
    ]
    default_node_payload = [
        item for item in all_node_payload if item["id"] in default_node_ids
    ]
    all_edge_payload = [
        _graph_edge_payload(edge) for edge in knowledge_map.edges
    ]
    default_edge_payload = [
        item
        for item in all_edge_payload
        if item["source"] in default_node_ids
        and item["target"] in default_node_ids
        and item["kind"] == "hierarchy"
    ]
    paper_payload: list[dict[str, Any]] = []
    for paper in knowledge_map.papers:
        if not paper.is_in_scope:
            continue
        paper_evidence = tuple(
            {
                item.chunk_id: item
                for item in (
                    *paper.evidence,
                    *((paper.metadata_evidence,) if paper.metadata_evidence else ()),
                    *paper.population,
                    *paper.methods,
                    *paper.interventions,
                    *paper.comparators,
                    *paper.measures,
                    *paper.results,
                    *paper.statistics,
                    *paper.conclusions,
                    *paper.limitations,
                )
            }.values()
        )
        paper_payload.append(
            {
                "id": paper.paper_id,
                "node_id": f"paper:{paper.paper_id}",
                "title": paper.title,
                "authors": paper.authors_text,
                "first_author": paper.first_author,
                "year": paper.year,
                "journal": paper.journal,
                "doi": paper.doi,
                "source_url": paper.source_url,
                "pdf_path": paper.pdf_path,
                "source": paper.source,
                "abstract": paper.abstract,
                "keywords": list(paper.keywords),
                "status": paper.analysis_status,
                "method_category": paper.method_category,
                "theme_ids": list(paper.theme_ids),
                "method_summary": paper.method_text,
                "conclusion_summary": paper.conclusion_text,
                "limitation_summary": paper.limitation_text,
                "methods": [
                    item.to_dict() for item in paper.methods[:3]
                ],
                "results": [
                    item.to_dict() for item in paper.results[:3]
                ],
                "conclusions": [
                    item.to_dict() for item in paper.conclusions[:3]
                ],
                "interventions": [
                    item.to_dict() for item in paper.interventions[:4]
                ],
                "comparators": [
                    item.to_dict() for item in paper.comparators[:3]
                ],
                "measures": [
                    item.to_dict() for item in paper.measures[:4]
                ],
                "statistics": [
                    item.to_dict() for item in paper.statistics[:4]
                ],
                "limitations": [
                    item.to_dict() for item in paper.limitations[:4]
                ],
                "evidence": [
                    item.to_dict() for item in paper_evidence[:6]
                ],
                "abstract_evidence": next(
                    (
                        item.to_dict()
                        for item in paper_evidence
                        if item.source_type == "摘要"
                    ),
                    None,
                ),
                "metadata_evidence": next(
                    (
                        item.to_dict()
                        for item in paper_evidence
                        if item.source_type == "元数据"
                    ),
                    None,
                ),
                "evidence_types": sorted(
                    {item.source_type for item in paper_evidence}
                ),
            }
        )
    statistic_by_id = {
        item.theme_id: item for item in knowledge_map.theme_statistics
    }
    return {
        "core_id": CORE_DIRECTION_ID,
        "core_direction": knowledge_map.core_direction,
        "nodes": default_node_payload,
        "edges": default_edge_payload,
        "papers": [],
        "all_nodes": all_node_payload,
        "all_edges": all_edge_payload,
        "all_papers": paper_payload,
        "themes": [
            {
                "id": node.node_id,
                "label": node.label,
                "description": node.description,
                "paper_count": node.paper_count,
                "evidence_count": node.evidence_count,
                "direction_count": statistic_by_id[node.node_id].direction_count,
                "variable_count": statistic_by_id[node.node_id].variable_count,
                "child_ids": child_ids.get(node.node_id, []),
                "method_summary": (
                    statistic_by_id[node.node_id].method_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
                "population_summary": (
                    statistic_by_id[node.node_id].population_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
                "measure_summary": (
                    statistic_by_id[node.node_id].measure_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
                "conclusion_summary": (
                    statistic_by_id[node.node_id].conclusion_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
                "controversy_summary": (
                    statistic_by_id[node.node_id].controversy_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
                "limitation_summary": (
                    statistic_by_id[node.node_id].limitation_summary
                    if node.node_id in statistic_by_id
                    else "证据不足，暂不能判断"
                ),
            }
            for node in theme_nodes
        ],
        "years": sorted(
            {
                paper.year
                for paper in knowledge_map.in_scope_papers
                if paper.year
            }
        ),
        "methods": sorted(
            {
                paper.method_category
                for paper in knowledge_map.in_scope_papers
            }
        ),
        "evidence_types": sorted(
            {
                item.source_type
                for paper in knowledge_map.in_scope_papers
                for item in (
                    *paper.evidence,
                    *((paper.metadata_evidence,) if paper.metadata_evidence else ()),
                    *paper.methods,
                    *paper.results,
                    *paper.conclusions,
                )
            }
        ),
        "graph_statistics": {
            "theme_count": knowledge_map.graph_statistics.theme_count,
            "direction_count": knowledge_map.graph_statistics.direction_count,
            "pending_direction_count": (
                knowledge_map.graph_statistics.pending_direction_count
            ),
            "variable_count": knowledge_map.graph_statistics.variable_count,
            "paper_node_count": knowledge_map.graph_statistics.paper_node_count,
            "research_question_count": (
                knowledge_map.graph_statistics.direction_count
            ),
            "evidence_node_count": (
                knowledge_map.graph_statistics.paper_node_count
            ),
            "consensus_node_count": (
                knowledge_map.graph_statistics.consensus_node_count
            ),
            "controversy_node_count": (
                knowledge_map.graph_statistics.controversy_node_count
            ),
            "research_gap_node_count": (
                knowledge_map.graph_statistics.research_gap_node_count
            ),
            "synthesis_node_count": (
                knowledge_map.graph_statistics.synthesis_node_count
            ),
            "hierarchy_relation_count": (
                knowledge_map.graph_statistics.hierarchy_relation_count
            ),
            "research_relation_count": (
                knowledge_map.graph_statistics.research_relation_count
            ),
            "typed_research_relation_count": (
                knowledge_map.graph_statistics.typed_research_relation_count
            ),
            "paper_variable_relation_count": (
                knowledge_map.graph_statistics.paper_variable_relation_count
            ),
            "question_variable_relation_count": (
                knowledge_map.graph_statistics.question_variable_relation_count
            ),
            "in_scope_paper_count": (
                knowledge_map.graph_statistics.in_scope_paper_count
            ),
            "pending_paper_count": (
                knowledge_map.graph_statistics.pending_paper_count
            ),
        },
    }


def _graph_node_payload(
    node: KnowledgeMapNode,
    child_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "id": node.node_id,
        "label": node.label,
        "type": node.node_type,
        "level": node.level,
        "description": node.description,
        "parent_ids": list(node.parent_ids),
        "child_ids": list(child_ids),
        "paper_ids": list(node.paper_ids),
        "paper_count": node.paper_count,
        "evidence_count": node.evidence_count,
        "evidence_types": list(node.evidence_types),
        "content_kind": node.content_kind,
        "variable_role": node.variable_role,
        "direction_ids": list(node.direction_ids),
        "level_label": {
            1: "核心研究领域",
            2: "核心研究主题",
            3: "研究变量与具体问题",
            4: "研究证据与发现",
            5: "学术共识与争议",
        }.get(node.level, ""),
        "evidence": [
            evidence.to_dict() for evidence in node.evidence[:3]
        ],
    }


def _graph_edge_payload(edge: KnowledgeMapEdge) -> dict[str, Any]:
    return {
        "id": edge.edge_id,
        "source": edge.source_id,
        "target": edge.target_id,
        "kind": edge.edge_kind,
        "relation": edge.relation,
        "paper_ids": list(edge.paper_ids),
        "evidence_count": edge.evidence_count,
        "relation_status": edge.relation_status,
        "relation_origin": edge.relation_origin,
        "evidence": [
            item.to_dict()
            for item in (
                edge.evidence[:3]
                if edge.edge_kind
                in {
                    "research_relation",
                    "synthesis_relation",
                    "evidence_synthesis",
                }
                else ()
            )
        ],
    }


def render_knowledge_map_page(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    """渲染简洁、可逐步展开的五层知识图谱。"""

    _page_heading(
        "EVIDENCE KNOWLEDGE MAP",
        "知识图谱",
        "先看研究方向与核心主题，再按分支展开研究内容和真实论文证据。",
    )
    state = render_analysis_update_status(
        workflow,
        domain,
        workflow.default_debate_topic,
        page_key="knowledge-map",
        action_label="重新生成知识图谱",
    )
    try:
        knowledge_map = get_knowledge_map(workflow, domain)
    except Exception as exc:
        LOGGER.exception(
            "Failed to build research knowledge map for domain %s",
            domain.domain_id,
        )
        st.error(
            f"知识图谱生成失败：{type(exc).__name__}: {exc}",
            title="地图数据读取失败",
            icon=":material/error:",
        )
        st.caption(
            "论文库数据未被修改。请先检查数据库连接、论文解析状态和日志。"
        )
        return
    _render_overview_metrics(knowledge_map)

    if not knowledge_map.in_scope_papers:
        st.warning(
            "当前论文库没有通过“青少年”研究对象原文核验的论文。"
            "图谱默认显示真实结构，但主题分支保持空白，没有用大学生论文或"
            "推断关系填充。请先检索并导入符合方向的论文。",
            icon=":material/filter_alt_off:",
            title="当前研究范围证据为空",
        )

    payload = build_graph_payload(knowledge_map)
    try:
        render_knowledge_graph_component(
            payload,
            key=f"researchmind-knowledge-map-{domain.domain_id}",
            height=1_020,
        )
    except Exception as exc:
        st.info(
            "交互知识地图组件暂不可用，已回退为主题节点表。"
            f"组件信息：{exc}"
        )
        _render_graph_fallback(knowledge_map)

    st.caption(
        "图谱只连接能够回指论文原文页码的关系。"
        "研究方向、变量和变量关系均要求可定位的全文、摘要或明确题录证据。"
        "关系颜色区分正向、负向、无显著和结果不一致；相关证据不会被自动"
        "解释为因果关系。"
    )
    _render_graph_insights(knowledge_map)
    _render_theme_details(knowledge_map)
    _render_graph_findings_summary(knowledge_map)
    _render_knowledge_maintenance(workflow, domain)


def render_knowledge_summary_page(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    """渲染独立的可视化科研分析报告与证据追溯页面。"""

    _page_heading(
        "RESEARCH SYNTHESIS",
        "知识图谱总结",
        "基于当前论文库原文证据生成研究主题、共识、争议、局限、空白与创新方向报告。",
    )
    state = render_analysis_update_status(
        workflow,
        domain,
        workflow.default_debate_topic,
        page_key="knowledge-summary",
        action_label="重新分析共识与争议",
    )
    try:
        knowledge_map = get_knowledge_map(workflow, domain)
    except Exception as exc:
        LOGGER.exception(
            "Failed to build research knowledge map summary for domain %s",
            domain.domain_id,
        )
        st.error(
            f"知识图谱总结生成失败：{type(exc).__name__}: {exc}",
            title="地图总结读取失败",
            icon=":material/error:",
        )
        return
    if not state.is_current:
        st.info(
            "当前版本正在后台分析；页面仅展示可从当前论文库直接重建的结构，"
            "不会复用已失效的旧分歧结论。",
            icon=":material/hourglass_top:",
        )
    _render_overview_metrics(knowledge_map)
    _render_question_synthesis_report(knowledge_map)

    mode_col, export_col = st.columns([3, 1], vertical_alignment="center")
    with mode_col:
        st.info(
            f"当前模式：{knowledge_map.mode}。"
            "本页没有把规则归纳伪装成大模型结论；"
            "证据不足的部分会明确标注。",
            icon=":material/verified:",
        )
    with export_col:
        report = build_markdown_report(knowledge_map)
        st.download_button(
            "导出 Markdown 报告",
            data=report.encode("utf-8"),
            file_name=f"体研智脑-知识图谱总结-{domain.name}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
            key=f"knowledge-report-download-{domain.domain_id}",
        )

    _render_report_overview(knowledge_map)
    _render_visual_report(knowledge_map)
    _render_findings_report(knowledge_map)
    _render_out_of_scope_audit(knowledge_map)
    _render_existing_synthesis(workflow, domain)
    st.caption(RETRIEVAL_SCOPE_NOTE)


def _render_overview_metrics(knowledge_map: KnowledgeMap) -> None:
    scope_evidence = len(
        {
            item.chunk_id
            for paper in knowledge_map.in_scope_papers
            for item in paper.evidence
        }
    )
    consensus_count = _supported_category_count(
        knowledge_map,
        "学术共识",
        "学术共识线索",
    )
    controversy_count = _supported_category_count(
        knowledge_map,
        "学术争议",
    )
    columns = st.columns([2.8, 1, 0.9, 1, 1])
    columns[0].metric(
        "核心方向",
        "青少年体质健康",
        help=CORE_DIRECTION,
    )
    columns[1].metric(
        "论文总数",
        f"{len(knowledge_map.papers)} 篇",
        help=f"范围内 {len(knowledge_map.in_scope_papers)} 篇；"
        f"范围外/待审核 {len(knowledge_map.out_of_scope_papers)} 篇。",
    )
    columns[2].metric("核心主题", f"{len(THEMES)} 个")
    columns[3].metric(
        "已识别共识",
        f"{consensus_count} 条",
        delta="仅计多篇证据支持" if consensus_count else "证据不足",
        delta_color="off",
    )
    columns[4].metric(
        "已识别争议",
        f"{controversy_count} 条",
        delta="需回到原文核验" if controversy_count else "证据不足",
        delta_color="off",
    )
    full_text_count = sum(bool(paper.evidence) for paper in knowledge_map.papers)
    abstract_analysis_count = sum(
        not paper.evidence
        and any(
            item.source_type == "摘要"
            for item in (
                *paper.population,
                *paper.methods,
                *paper.results,
                *paper.conclusions,
            )
        )
        for paper in knowledge_map.papers
    )
    evidence_metrics = st.columns(4)
    evidence_metrics[0].metric("全文证据论文", f"{full_text_count} 篇")
    evidence_metrics[1].metric("摘要级分析", f"{abstract_analysis_count} 篇")
    evidence_metrics[2].metric(
        "主题级证据综合",
        f"{len(knowledge_map.question_syntheses)} 个问题",
    )
    evidence_metrics[3].metric(
        "分析版本",
        knowledge_map.analysis_engine_version or "未记录",
    )
    graph_metrics = st.columns(5)
    graph_metrics[0].metric(
        "研究方向",
        knowledge_map.graph_statistics.direction_count,
        help=(
            f"另有 {knowledge_map.graph_statistics.pending_direction_count} "
            "个已确认主题但未命中具体方向词的待细分层。"
        ),
    )
    graph_metrics[1].metric(
        "研究变量",
        knowledge_map.graph_statistics.variable_count,
    )
    graph_metrics[2].metric(
        "论文证据节点",
        knowledge_map.graph_statistics.paper_node_count,
    )
    graph_metrics[3].metric(
        "变量关系",
        knowledge_map.graph_statistics.research_relation_count,
        delta=(
            f"明确方向 {knowledge_map.graph_statistics.typed_research_relation_count}"
        ),
        delta_color="off",
    )
    graph_metrics[4].metric(
        "待审核论文",
        knowledge_map.graph_statistics.pending_paper_count,
    )
    synthesis_metrics = st.columns(4)
    synthesis_metrics[0].metric(
        "研究问题",
        knowledge_map.graph_statistics.direction_count,
    )
    synthesis_metrics[1].metric(
        "学术共识",
        knowledge_map.graph_statistics.consensus_node_count,
    )
    synthesis_metrics[2].metric(
        "学术争议",
        knowledge_map.graph_statistics.controversy_node_count,
    )
    synthesis_metrics[3].metric(
        "尚待研究问题",
        knowledge_map.graph_statistics.research_gap_node_count,
    )
    st.caption(
        f"范围内可追溯证据 {scope_evidence} 条 · "
        f"范围外/待审核论文 {len(knowledge_map.out_of_scope_papers)} 篇 · "
        f"更新于 {_short_time(knowledge_map.generated_at)}"
    )


def _supported_category_count(
    knowledge_map: KnowledgeMap,
    *categories: str,
) -> int:
    allowed = set(categories)
    return sum(
        item.category in allowed for item in knowledge_map.supported_findings
    )


def _render_graph_fallback(knowledge_map: KnowledgeMap) -> None:
    rows = [
        {
            "层级": node.level,
            "节点": node.label,
            "类型": node.node_type,
            "论文数": node.paper_count,
            "证据数": node.evidence_count,
            "说明": node.description,
        }
        for node in knowledge_map.nodes
        if node.level <= 3
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="知识地图主题与研究方向回退表",
    )


def _render_graph_insights(knowledge_map: KnowledgeMap) -> None:
    st.subheader("图谱科研洞察")
    if not knowledge_map.graph_insights:
        st.info(
            "当前图谱尚未形成可计算的科研洞察。",
            icon=":material/query_stats:",
        )
        return
    category_order = (
        "研究热点",
        "新兴方向",
        "研究交叉",
        "研究空白",
        "研究趋势",
        "证据密度",
    )
    available = {
        item.category for item in knowledge_map.graph_insights
    }
    categories = tuple(
        category for category in category_order if category in available
    )
    selected = st.segmented_control(
        "洞察类型",
        options=categories,
        default=categories[0],
        key="knowledge-map-insight-category",
    )
    if selected is None:
        return
    items = tuple(
        item
        for item in knowledge_map.graph_insights
        if item.category == selected
    )
    st.caption(
        "热点、新兴方向、交叉和空白均由当前库论文数量、年份、证据覆盖"
        "和图谱关系自动计算；论文少不等于领域空白。"
    )
    for item in items:
        with st.container(border=True):
            st.markdown(f"**{item.title}**")
            st.write(item.statement)
            st.caption(f"{item.status} · {item.reliability}")
            if item.evidence:
                for index, evidence in enumerate(item.evidence[:3], start=1):
                    with st.expander(
                        f"原文证据 {index}：{evidence.source_type} · "
                        f"{evidence.location_label}",
                        icon=":material/fact_check:",
                    ):
                        st.text(evidence.quote)
            if item.limitations:
                st.caption("边界：" + "；".join(item.limitations))


def _render_theme_details(knowledge_map: KnowledgeMap) -> None:
    st.subheader("研究主题详情")
    theme_options = {
        item.theme_id: item
        for item in knowledge_map.theme_statistics
    }
    selected_theme = st.segmented_control(
        "选择研究主题",
        options=list(theme_options),
        format_func=lambda theme_id: theme_options[theme_id].label,
        default=next(iter(theme_options), None),
        key="knowledge-map-theme-detail",
    )
    if selected_theme is None:
        return
    statistic = theme_options[selected_theme]
    definition = THEME_BY_ID[selected_theme]
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.markdown(f"**{statistic.label}**")
            st.write(definition.description)
        with right:
            st.metric("范围内论文", statistic.paper_count)
            st.caption(
                f"研究方向 {statistic.direction_count} 个 · "
                f"变量 {statistic.variable_count} 个 · "
                f"可追溯证据 {statistic.evidence_count} 条"
            )
        papers = tuple(
            paper
            for paper in knowledge_map.in_scope_papers
            if selected_theme in paper.theme_ids
        )
        if not papers:
            st.warning(
                "当前库中没有该主题的青少年论文证据。"
                "不能据此推断整个学术领域没有相关研究。",
                icon=":material/search_off:",
            )
        else:
            evidence_rows = (
                ("研究方法", statistic.method_summary),
                ("研究对象", statistic.population_summary),
                ("常用评价指标", statistic.measure_summary),
                ("代表性研究结论", statistic.conclusion_summary),
                ("相关学术争议", statistic.controversy_summary),
                ("研究不足", statistic.limitation_summary),
            )
            summary_columns = st.columns(2)
            for index, (label, value) in enumerate(evidence_rows):
                with summary_columns[index % 2]:
                    st.markdown(f"**{label}**")
                    st.write(value or INSUFFICIENT_EVIDENCE_TEXT)

            direction_nodes = tuple(
                node
                for node in knowledge_map.nodes
                if node.level == 3
                and selected_theme in node.parent_ids
            )
            st.markdown("**研究方向与变量覆盖**")
            if direction_nodes:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "研究方向": node.label,
                                "类型": node.node_type,
                                "论文数": node.paper_count,
                                "证据数": node.evidence_count,
                                "证据类型": "、".join(node.evidence_types),
                            }
                            for node in direction_nodes
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                    alt=f"{statistic.label}研究方向覆盖表",
                )
            else:
                st.caption(INSUFFICIENT_EVIDENCE_TEXT)

            representative_ids = set(statistic.representative_paper_ids)
            representative_papers = tuple(
                paper for paper in papers
                if paper.paper_id in representative_ids
            )
            if representative_papers:
                st.markdown("**代表性论文与证据**")
                for paper in representative_papers:
                    with st.expander(
                        f"{paper.title}（{paper.year or '年份待核对'}）",
                        on_change="ignore",
                    ):
                        st.caption(
                            f"第一作者：{paper.first_author} · "
                            f"DOI：{paper.doi or '未提供'} · "
                            f"来源：{paper.source or '未提供'}"
                        )
                        paper_evidence = tuple(
                            item
                            for item in (
                                *paper.methods,
                                *paper.population,
                                *paper.measures,
                                *paper.conclusions,
                                *paper.limitations,
                            )
                            if item.chunk_id
                        )
                        if paper_evidence:
                            for evidence in paper_evidence[:6]:
                                st.markdown(
                                    f"- {evidence.location_label} · "
                                    f"{evidence.source_type} · "
                                    f"{evidence.file_name}"
                                )
                                st.caption(evidence.quote)
                        else:
                            st.caption(INSUFFICIENT_EVIDENCE_TEXT)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "论文": paper.title,
                            "作者": paper.authors_text,
                            "年份": paper.year,
                            "DOI": paper.doi or "未提供",
                            "证据数": len(paper.evidence),
                        }
                        for paper in papers
                    ]
                ),
                width="stretch",
                hide_index=True,
                alt=f"{statistic.label}主题论文清单",
            )


def _render_graph_findings_summary(knowledge_map: KnowledgeMap) -> None:
    st.subheader("证据摘要")
    consensus = _findings_for(
        knowledge_map,
        ("学术共识", "学术共识线索"),
        supported_only=False,
    )
    controversy = _findings_for(
        knowledge_map,
        ("学术争议",),
        supported_only=False,
    )
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**学术共识**")
        if consensus:
            for item in consensus[:2]:
                st.write(item.statement)
                st.caption(f"{item.status} · {item.reliability}")
        else:
            st.caption("证据不足，待进一步验证。")
    with right, st.container(border=True):
        st.markdown("**学术争议**")
        if controversy:
            for item in controversy[:2]:
                st.write(item.statement)
                st.caption(f"{item.status} · {item.reliability}")
        else:
            st.caption("证据不足，待进一步验证。")


def _render_knowledge_maintenance(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    with st.expander(
        "知识体系维护",
        icon=":material/settings:",
        on_change="ignore",
    ):
        if workflow.read_only:
            st.info(
                "当前为旧库只读模式，不会执行结构化阅读、学术分歧分析或"
                "知识体系归纳。请切换到青少年专题库后再更新分析结果。",
                icon=":material/lock:",
            )
            return
        st.caption(
            "更新会依次执行结构化阅读、学术分歧分析和知识体系归纳。"
            "该操作不会删除论文或向量数据。"
        )
        if st.button(
            "更新知识体系与图谱",
            icon=":material/sync:",
            key=f"knowledge-map-refresh-{domain.domain_id}",
        ):
            try:
                schedule_for_domain(workflow, domain, force=True)
            except Exception as error:
                st.error(
                    f"无法启动知识图谱更新：{type(error).__name__}: {error}"
                )
            else:
                clear_analysis_session_state(domain.domain_id)
                st.success("已在后台启动知识图谱更新。")
                st.rerun()


def _render_report_overview(knowledge_map: KnowledgeMap) -> None:
    st.subheader("研究领域总览")
    with st.container(border=True):
        st.markdown(f"### {knowledge_map.core_direction}")
        st.write(
            f"当前领域为“{knowledge_map.domain_name}”。"
            f"论文库共 {len(knowledge_map.papers)} 篇，其中 "
            f"{len(knowledge_map.in_scope_papers)} 篇通过青少年范围核验。"
        )
        if knowledge_map.in_scope_papers:
            hottest = max(
                knowledge_map.theme_statistics,
                key=lambda item: item.paper_count,
            )
            st.markdown(
                f"**当前库研究热点：**{hottest.label}"
                f"（{hottest.paper_count} 篇范围内论文）"
            )
        else:
            st.warning(
                "当前库没有范围内论文，暂不能识别真实研究热点。"
                "这里的空白只表示本库覆盖不足，不代表整个领域没有研究。",
                icon=":material/data_usage:",
            )
        rows = [
            {
                "研究主题": item.label,
                "范围内论文": item.paper_count,
                "可追溯证据": item.evidence_count,
            }
            for item in knowledge_map.theme_statistics
        ]
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            alt="研究主题论文与证据数量统计",
        )


def _render_visual_report(knowledge_map: KnowledgeMap) -> None:
    st.subheader("可视化科研报告")
    row_one = st.columns(2)
    with row_one[0], st.container(border=True):
        st.markdown("**研究主题分布**")
        if any(
            item.paper_count
            for item in knowledge_map.theme_statistics
        ):
            theme_df = pd.DataFrame(
                [
                    {"研究主题": item.label, "论文数": item.paper_count}
                    for item in knowledge_map.theme_statistics
                ]
            )
            st.bar_chart(
                theme_df.set_index("研究主题"),
                color="#2457D6",
                height=270,
                alt="各研究主题的范围内论文数量柱状图",
            )
        else:
            _empty_chart_placeholder(
                "当前库没有范围内论文，暂不能绘制真实研究主题分布。"
            )

    with row_one[1], st.container(border=True):
        st.markdown("**论文发表年份趋势**")
        if knowledge_map.year_distribution:
            year_df = pd.DataFrame(
                knowledge_map.year_distribution,
                columns=("年份", "论文数"),
            ).set_index("年份")
            st.line_chart(
                year_df,
                color="#147D64",
                height=270,
                alt="范围内论文发表年份趋势折线图",
            )
        else:
            _empty_chart_placeholder(
                "当前库没有可绘制的范围内论文年份。"
            )

    row_two = st.columns(2)
    with row_two[0], st.container(border=True):
        st.markdown("**研究方法分布**")
        if knowledge_map.method_distribution:
            method_df = pd.DataFrame(
                knowledge_map.method_distribution,
                columns=("研究方法", "论文数"),
            ).set_index("研究方法")
            st.bar_chart(
                method_df,
                color="#A55A0A",
                height=270,
                alt="论文研究方法分布柱状图",
            )
        else:
            _empty_chart_placeholder(
                "当前库尚未形成可核验的研究方法统计。"
            )

    with row_two[1], st.container(border=True):
        st.markdown("**学术观点对比**")
        if knowledge_map.stance_distribution:
            stance_df = pd.DataFrame(
                knowledge_map.stance_distribution,
                columns=("观点立场", "观点数"),
            ).set_index("观点立场")
            st.bar_chart(
                stance_df,
                color="#6B4FC8",
                horizontal=True,
                height=270,
                alt="学术观点立场分布条形图",
            )
        else:
            _empty_chart_placeholder(
                "没有形成可验证的相反观点，证据不足，待进一步验证。"
            )

    row_three = st.columns(2)
    with row_three[0], st.container(border=True):
        st.markdown("**学术争议关系**")
        controversy_items = _findings_for(
            knowledge_map,
            ("学术争议",),
            supported_only=False,
        )
        if controversy_items and any(item.evidence for item in controversy_items):
            for item in controversy_items:
                st.markdown(f"**{item.title}**")
                st.write(item.statement)
                st.caption(
                    f"支持论文 {len(set(item.paper_ids))} 篇 · "
                    f"证据 {len(item.evidence)} 条 · {item.reliability}"
                )
        else:
            _empty_chart_placeholder(
                "当前证据不足以绘制争议关系。"
                "这不表示领域不存在争议。"
            )

    with row_three[1], st.container(border=True):
        st.markdown("**主题与代表性论文对应**")
        correspondence = []
        for theme in knowledge_map.theme_statistics:
            theme_papers = [
                paper
                for paper in knowledge_map.in_scope_papers
                if theme.theme_id in paper.theme_ids
            ]
            correspondence.append(
                {
                    "研究主题": theme.label,
                    "论文数": theme.paper_count,
                    "代表性论文": (
                        theme_papers[0].title if theme_papers else "证据不足"
                    ),
                }
            )
        if knowledge_map.in_scope_papers:
            st.dataframe(
                pd.DataFrame(correspondence),
                width="stretch",
                hide_index=True,
                alt="研究主题与代表性论文对应表",
            )
        else:
            _empty_chart_placeholder(
                "当前库没有范围内论文，不能为研究主题指定代表性论文。"
            )


def _empty_chart_placeholder(message: str) -> None:
    st.info(
        message,
        icon=":material/query_stats:",
        width="stretch",
    )


def _render_question_synthesis_report(knowledge_map: KnowledgeMap) -> None:
    st.subheader("主题级共识与争议")
    st.caption(
        "每个研究问题独立比较论文级结果；证据确定性是自动初筛，"
        "不是正式 GRADE 专家评级。没有同题相反结果时不会生成争议。"
    )
    if not knowledge_map.question_syntheses:
        st.info(
            "当前分析记录尚未包含主题级证据综合，请使用上方重新分析入口更新。",
            icon=":material/hourglass_top:",
        )
        return
    paper_by_id = {
        paper.paper_id: paper for paper in knowledge_map.papers
    }
    for synthesis in knowledge_map.question_syntheses:
        with st.container(border=True):
            title_col, certainty_col = st.columns([4, 1])
            with title_col:
                st.markdown(f"**{synthesis.label}**")
                st.caption(synthesis.question)
            with certainty_col:
                st.metric("证据确定性", synthesis.evidence_certainty)
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "一致/正向",
                len(synthesis.supporting_paper_ids),
            )
            metric_cols[1].metric(
                "负向/无显著",
                len(synthesis.opposing_paper_ids),
            )
            metric_cols[2].metric(
                "方向未判定",
                len(synthesis.inconclusive_paper_ids),
            )
            metric_cols[3].metric(
                "状态",
                synthesis.status,
            )
            st.write(synthesis.conclusion)
            st.caption(
                f"{synthesis.certainty_reason} · "
                f"分析时间 {_short_time(synthesis.analysis_time)}"
            )
            if synthesis.outcome_measures:
                st.caption(
                    "问题相关指标：" + "、".join(synthesis.outcome_measures)
                )
            if synthesis.representative_evidence:
                for index, evidence in enumerate(
                    synthesis.representative_evidence,
                    start=1,
                ):
                    paper = paper_by_id.get(evidence.document_id)
                    with st.expander(
                        f"原文证据 {index}：{evidence.source_type} · "
                        f"{evidence.location_label} · "
                        f"{paper.title if paper else evidence.document_title}",
                        icon=":material/article:",
                    ):
                        if paper is not None:
                            st.caption(
                                f"{paper.authors_text} · {paper.year or '年份待核对'} · "
                                f"DOI：{paper.doi or '未提供'}"
                            )
                        st.text(evidence.quote)
            else:
                st.warning(
                    "该主题尚未提取到可定位的论文级结果证据。",
                    icon=":material/search_off:",
                )

    st.subheader("同题学术争议")
    if not knowledge_map.conflicts:
        st.info(
            "没有在同一研究问题内发现可核验的相反结果。"
            "未检测到争议不等于整个领域不存在争议。",
            icon=":material/balance:",
        )
        return
    for conflict in knowledge_map.conflicts:
        with st.container(border=True):
            st.markdown(f"**{conflict.question}**")
            left, right = st.columns(2)
            _render_conflict_side(
                left,
                "观点 A",
                conflict.viewpoint_a,
                conflict.paper_ids_a,
                conflict.evidence_a,
                paper_by_id,
            )
            _render_conflict_side(
                right,
                "观点 B",
                conflict.viewpoint_b,
                conflict.paper_ids_b,
                conflict.evidence_b,
                paper_by_id,
            )
            st.markdown("**主要分歧原因**")
            for reason in conflict.divergence_reasons:
                st.markdown(f"- {reason}")
            st.markdown("**当前能够得出的结论**")
            st.write(conflict.current_conclusion)
            st.markdown("**仍需解决的问题**")
            for question in conflict.unresolved_questions:
                st.markdown(f"- {question}")
            st.caption(
                f"证据确定性：{conflict.evidence_certainty} · "
                f"分析时间 {_short_time(conflict.analysis_time)}"
            )


def _render_conflict_side(
    container: Any,
    label: str,
    viewpoint: str,
    paper_ids: tuple[str, ...],
    evidence_items: tuple[Any, ...],
    paper_by_id: dict[str, KnowledgeMapPaper],
) -> None:
    with container:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.write(viewpoint)
            for paper_id in paper_ids:
                paper = paper_by_id.get(paper_id)
                if paper is not None:
                    st.markdown(
                        f"- {paper.title}（{paper.year or '年份待核对'}；"
                        f"DOI：{paper.doi or '未提供'}）"
                    )
            for evidence in _unique_evidence(evidence_items):
                with st.expander(
                    f"{evidence.source_type} · {evidence.location_label}",
                    icon=":material/fact_check:",
                ):
                    st.text(evidence.quote)


def _render_findings_report(knowledge_map: KnowledgeMap) -> None:
    st.subheader("可追溯科研结论")
    sections = (
        (
            "研究局限",
            ("研究局限",),
            "当前库没有可定位到原文的研究局限陈述。",
        ),
        (
            "研究局限与空白",
            ("研究空白",),
            "当前库覆盖不足，不能据此断言整个领域存在研究空白。",
        ),
        (
            "创新研究方向",
            ("创新方向",),
            "需要补充更多范围内论文后才能形成有证据支持的创新假设。",
        ),
    )
    paper_by_id = {
        paper.paper_id: paper for paper in knowledge_map.papers
    }
    for title, categories, empty_text in sections:
        items = _findings_for(
            knowledge_map,
            categories,
            supported_only=False,
        )
        st.markdown(f"### {title}")
        if not items:
            st.info(empty_text, icon=":material/manage_search:")
            continue
        for item in items:
            _render_finding_card(item, paper_by_id)


def _render_finding_card(
    item: KnowledgeFinding,
    paper_by_id: dict[str, KnowledgeMapPaper],
) -> None:
    with st.container(border=True):
        header, evidence_metric = st.columns([4, 1])
        with header:
            st.markdown(f"**{item.title}**")
            st.write(item.statement)
        with evidence_metric:
            st.metric("证据", len(item.evidence))
        st.caption(
            f"状态：{item.status} · 可靠性：{item.reliability}"
        )
        if not item.evidence:
            st.warning(
                "证据不足，待进一步验证。",
                icon=":material/report:",
            )
            return
        for index, evidence in enumerate(_unique_evidence(item.evidence), start=1):
            paper = paper_by_id.get(evidence.document_id)
            label = (
                f"查看证据 {index}：{evidence.document_title}"
                f" · 第 {evidence.page_number} 页"
            )
            with st.expander(label, icon=":material/article:"):
                if paper is not None:
                    st.markdown(f"**作者：**{paper.authors_text}")
                    st.markdown(f"**DOI：**{paper.doi or '未提供'}")
                    st.markdown(
                        f"**来源：**{paper.source or paper.source_url or '未提供'}"
                    )
                    st.markdown(
                        f"**本地 PDF：**{paper.pdf_path or '未提供'}"
                    )
                st.caption(
                    f"{evidence.document_title} · {evidence.source_type} · "
                    f"{evidence.location_label} · {evidence.file_name}"
                )
                st.text(evidence.quote)
        if item.limitations:
            st.caption("证据边界：" + "；".join(item.limitations))


def _findings_for(
    knowledge_map: KnowledgeMap,
    categories: Iterable[str],
    *,
    supported_only: bool,
) -> tuple[KnowledgeFinding, ...]:
    allowed = set(categories)
    findings = (
        knowledge_map.supported_findings
        if supported_only
        else knowledge_map.findings
    )
    return tuple(item for item in findings if item.category in allowed)


def _render_out_of_scope_audit(knowledge_map: KnowledgeMap) -> None:
    st.subheader("范围外/待审核论文")
    st.caption(
        "这些论文保留在项目论文库中，但不会作为青少年研究证据进入主题分支。"
    )
    if not knowledge_map.out_of_scope_papers:
        st.success("当前没有范围外/待审核论文。")
        return
    rows = [
        {
            "论文": paper.title,
            "作者": paper.authors_text,
            "年份": paper.year,
            "DOI": paper.doi or "未提供",
            "判定": paper.scope_reason,
        }
        for paper in knowledge_map.out_of_scope_papers
    ]
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        alt="范围外或待审核论文清单",
    )


def _render_existing_synthesis(
    workflow: ResearchMindWorkflow,
    domain: ResearchDomain,
) -> None:
    synthesis = workflow.store.get_latest_synthesis(domain.domain_id)
    if synthesis is None:
        return
    with st.expander(
        "查看既有知识体系结构化记录",
        icon=":material/account_tree:",
        on_change="ignore",
    ):
        st.caption(
            f"生成时间：{_short_time(synthesis.generated_at)} · "
            f"模式：{synthesis.mode}"
        )
        st.markdown("**核心概念**")
        _bullet_list(synthesis.concepts)
        st.markdown("**理论框架链**")
        _bullet_list(synthesis.theory_framework)
        st.markdown("**共识记录**")
        _bullet_list(synthesis.consensus)
        st.markdown("**分歧记录**")
        _bullet_list(synthesis.disagreements)
        st.markdown("**局限记录**")
        _bullet_list(synthesis.limitations)
        st.markdown("**空白记录**")
        _bullet_list(synthesis.research_gaps)
        st.markdown("**创新方向记录**")
        _bullet_list(synthesis.innovation_directions)
        st.download_button(
            "下载既有知识体系 JSON",
            data=json.dumps(
                synthesis.to_dict(),
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="research_knowledge_synthesis.json",
            mime="application/json",
            icon=":material/download:",
            key=f"synthesis-json-{domain.domain_id}",
        )


def _bullet_list(values: Iterable[str]) -> None:
    items = tuple(value for value in values if str(value).strip())
    if not items:
        st.caption("证据不足，暂无可展示内容。")
        return
    for item in items:
        st.markdown(f"- {item}")


def _page_heading(label: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-label">{label}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.write(description)


def _short_time(value: str) -> str:
    if not value:
        return "未知"
    return value.replace("T", " ")[:16]


def _default_debate_topic(domain: ResearchDomain) -> str:
    if domain.keywords:
        return "与".join(domain.keywords[:2]) + "的关系"
    return DEFAULT_DEBATE_TOPIC


def _unique_evidence(
    evidence: Iterable[EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    unique: dict[str, EvidenceRef] = {}
    for item in evidence:
        unique.setdefault(item.chunk_id, item)
    return tuple(unique.values())


__all__ = [
    "build_graph_payload",
    "get_knowledge_map",
    "render_knowledge_map_page",
    "render_knowledge_summary_page",
]
