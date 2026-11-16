"""Restore authorized full evidence text for the saved comparison experiment.

The historical question files intentionally stored only a 160-character
preview and SHA-256. The local ResearchMind database still contains the
authorized source excerpts, so this script restores the exact text by hash.
It does not call a model or alter the saved model responses.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "adolescent_fitness"
    / "researchmind.sqlite3"
)
ROOT = Path(__file__).resolve().parent / "controversy-comparison"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from researchmind.storage import ResearchMindStore


def normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    store = ResearchMindStore(DB_PATH, read_only=True)
    quotes_by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    quotes_by_paper: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for domain in store.list_domains():
        for paper in store.list_papers(domain.domain_id):
            if paper.abstract.strip():
                quotes_by_paper[paper.paper_id].append(
                    (f"abstract:{paper.paper_id}", paper.abstract)
                )
            for evidence in store.list_evidence(paper.paper_id):
                digest = hashlib.sha256(
                    evidence.quote.encode("utf-8")
                ).hexdigest()
                quotes_by_hash[digest].append(
                    (evidence.chunk_id, evidence.quote)
                )
                quotes_by_paper[paper.paper_id].append(
                    (evidence.chunk_id, evidence.quote)
                )

    checks: list[dict[str, object]] = []
    for path in sorted(ROOT.glob("question-*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        restored = 0
        missing: list[str] = []
        for item in saved.get("input_evidence", []):
            digest = str(item.get("quote_sha256", ""))
            matches = quotes_by_hash.get(digest, [])
            unique_quotes = {quote for _, quote in matches}
            match_method = "按SHA-256精确匹配"
            match_count = len(matches)
            if len(unique_quotes) == 1:
                quote = next(iter(unique_quotes))
            else:
                preview = normalize_for_match(
                    str(item.get("quote_preview", ""))
                )
                paper_quotes = quotes_by_paper.get(
                    str(item.get("paper_id", "")),
                    (),
                )
                candidate_quotes: set[str] = set()
                for prefix_length in (160, 100, 80, 60):
                    search_text = preview[:prefix_length]
                    candidate_quotes = {
                        quote
                        for _, quote in paper_quotes
                        if search_text
                        and search_text in normalize_for_match(quote)
                    }
                    if candidate_quotes:
                        break
                if not candidate_quotes:
                    missing.append(str(item.get("evidence_id", "")))
                    continue
                quote = max(candidate_quotes, key=len)
                match_method = (
                    "按论文ID与保存预览前"
                    f"{min(len(preview), prefix_length)}字符匹配，"
                    "存在嵌套片段时取最长片段"
                )
                match_count = len(candidate_quotes)
            item["quote_text"] = quote
            item["quote_length"] = len(quote)
            item["quote_text_status"] = (
                f"{match_method}，从本地授权证据库还原"
            )
            item["quote_match_count"] = match_count
            restored += 1
        path.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        checks.append(
            {
                "file": path.name,
                "input_evidence_count": len(saved.get("input_evidence", [])),
                "restored_full_text_count": restored,
                "missing_or_ambiguous_hashes": missing,
            }
        )

    output = {
        "method": (
            "优先按保存的 SHA-256 与当前本地授权证据库精确匹配；"
            "仅在哈希未命中时使用论文ID与保存预览的唯一匹配；"
            "仅用于核验历史模型输入，未调用模型，未改写模型回答。"
        ),
        "database": str(DB_PATH.relative_to(PROJECT_ROOT)),
        "question_count": len(checks),
        "all_full_text_restored": all(
            not item["missing_or_ambiguous_hashes"] for item in checks
        ),
        "checks": checks,
    }
    output_path = ROOT / "input-availability.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["all_full_text_restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
