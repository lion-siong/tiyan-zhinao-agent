"""命令行入口，用于快速检查规划器和模拟问答。

Streamlit 界面请使用：
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from agents.planner import ResearchPlannerAgent
from services.llm_service import LLMService


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="体研智脑 Agent 命令行工具")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--task", help="生成科研任务规划")
    action_group.add_argument("--topic", help="生成科研选题模板")
    action_group.add_argument("--ask", help="进行科研问答")
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="选题数量，范围 1-5，默认 3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行对应功能。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    agent = ResearchPlannerAgent()

    try:
        if args.task:
            plan = agent.create_plan(args.task)
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
        elif args.topic:
            suggestions = agent.generate_topic_suggestions(args.topic, args.count)
            payload = [suggestion.to_dict() for suggestion in suggestions]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        elif args.ask:
            response = LLMService().answer(args.ask, allow_simulation=True)
            print(f"[{response.mode}]")
            print(response.answer)
            if response.message:
                print(f"\n提示：{response.message}")
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
