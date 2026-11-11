"""主控 Agent 与科研任务规划器。

第一阶段只实现基于规则的规划，不自动执行复杂工具链。每个规划步骤都
预留了工具名称，便于后续替换为 LangGraph 节点或真实工具调用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SIMULATION_NOTICE = "以下内容由规则模板生成，属于模拟建议，需结合真实文献与数据人工核验。"


@dataclass(frozen=True)
class ResearchStep:
    """科研规划中的一个步骤。"""

    order: int
    title: str
    description: str
    tool_name: str
    expected_output: str
    status: str = "待执行"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchPlan:
    """科研任务的整体计划。"""

    task: str
    task_type: str
    goal: str
    steps: tuple[ResearchStep, ...]
    mode: str = "规则规划（非大模型）"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["steps"] = [step.to_dict() for step in self.steps]
        return result


@dataclass(frozen=True)
class TopicSuggestion:
    """一个候选研究选题模板。"""

    title: str
    background: str
    research_question: str
    research_subject: str
    research_value: str
    suggested_design: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ResearchPlannerAgent:
    """轻量级主控 Agent。

    当前职责是理解任务、判断任务类型、拆解步骤并返回工具路线。
    PDF、数据分析和问答服务由界面层按用户操作调用，不会在本类中偷偷执行。
    """

    TOOL_CATALOG = {
        "文献检索策略": "tools.literature_search",
        "PDF 文献阅读": "tools.pdf_reader",
        "表格数据分析": "tools.data_analysis",
        "智能科研问答": "services.llm_service",
        "科研报告生成": "future.report_generator",
    }

    def create_plan(self, task: str) -> ResearchPlan:
        """把用户的科研任务拆解为可执行步骤。"""

        clean_task = task.strip()
        if not clean_task:
            raise ValueError("科研任务不能为空。")

        task_type = self._detect_task_type(clean_task)
        steps = (
            ResearchStep(
                order=1,
                title="明确研究边界",
                description="整理研究目的、目标人群、时间范围和可获得的资源。",
                tool_name="主控 Agent",
                expected_output="一页研究任务说明",
            ),
            ResearchStep(
                order=2,
                title="制定文献检索策略",
                description="提取中英文关键词，组合研究对象、干预方式和结局指标。",
                tool_name="文献检索策略",
                expected_output="数据库建议与检索表达式",
            ),
            ResearchStep(
                order=3,
                title="阅读与整理证据",
                description="阅读真实 PDF 文献，记录研究设计、样本、方法和局限。",
                tool_name="PDF 文献阅读",
                expected_output="可追溯的文献阅读笔记",
            ),
            ResearchStep(
                order=4,
                title="设计研究方案",
                description="明确研究问题、对象、变量、测量方式与数据分析计划。",
                tool_name="主控 Agent",
                expected_output="结构化研究设计草案",
            ),
            ResearchStep(
                order=5,
                title="分析真实数据",
                description="上传经过核验的数据，完成描述统计、缺失值检查和基础图表。",
                tool_name="表格数据分析",
                expected_output="数据概览与统计结果",
            ),
            ResearchStep(
                order=6,
                title="撰写并核验报告",
                description="依据真实证据和统计结果撰写报告，逐项检查引用与结论边界。",
                tool_name="科研报告生成",
                expected_output="可人工复核的科研报告草稿",
            ),
        )

        return ResearchPlan(
            task=clean_task,
            task_type=task_type,
            goal=f"围绕“{clean_task}”形成问题清晰、证据可追溯、数据可复核的研究方案。",
            steps=steps,
        )

    def generate_topic_suggestions(
        self, direction: str, count: int = 3
    ) -> tuple[TopicSuggestion, ...]:
        """根据研究方向和固定模板生成候选选题。"""

        clean_direction = direction.strip()
        if not clean_direction:
            raise ValueError("研究方向不能为空。")
        if not 1 <= count <= 5:
            raise ValueError("建议数量必须在 1 到 5 之间。")

        candidates = (
            TopicSuggestion(
                title=f"{clean_direction}对大学生体育学习投入的影响研究",
                background=(
                    f"围绕“{clean_direction}”梳理体育教学情境中的相关变量，"
                    "通过真实文献界定概念，并说明研究缺口。"
                ),
                research_question=(
                    f"在高校体育课程中，{clean_direction}与大学生体育学习投入之间"
                    "存在怎样的关系？"
                ),
                research_subject="候选对象：普通高校在校大学生；实际范围需按研究条件确定。",
                research_value=(
                    "为体育教学设计提供可检验的问题框架，研究价值需在阅读文献后"
                    "结合真实证据重新论证。"
                ),
                suggested_design="建议先做横断面调查；如条件允许，可进一步设计对照干预。",
            ),
            TopicSuggestion(
                title=f"数字化环境下{clean_direction}教学模式的应用效果研究",
                background=(
                    f"关注“{clean_direction}”与数字化教学工具结合后的实施过程，"
                    "需要核实已有研究采用了哪些工具、人群和结局指标。"
                ),
                research_question=(
                    f"数字化支持下的{clean_direction}教学模式，对学生的技能学习"
                    "和课堂参与有何影响？"
                ),
                research_subject="候选对象：高校体育专业学生或公共体育课学生，需二选一。",
                research_value=(
                    "为教学模式选择提供阶段性证据；不能预先假定干预一定有效。"
                ),
                suggested_design="可采用准实验设计，设置前测、后测并说明分组方式。",
            ),
            TopicSuggestion(
                title=f"基于可穿戴设备的{clean_direction}训练负荷特征研究",
                background=(
                    f"以“{clean_direction}”训练过程为场景，说明可穿戴设备能够记录"
                    "哪些指标，以及现有研究尚未回答的问题。"
                ),
                research_question=(
                    f"在大学生{clean_direction}训练中，训练负荷与恢复指标的"
                    "时间变化特征是什么？"
                ),
                research_subject="候选对象：有规律训练经历的大学生；需明确纳入与排除标准。",
                research_value=(
                    "帮助形成训练监测方案，但设备和指标必须依据可靠来源选择。"
                ),
                suggested_design="建议采用重复测量设计，先完成设备信效度与伦理审查。",
            ),
            TopicSuggestion(
                title=f"{clean_direction}教学中教师反馈行为的观察研究",
                background=(
                    f"聚焦“{clean_direction}”课堂中的教师反馈，先通过真实课堂观察"
                    "和文献梳理确定可观察的行为类别。"
                ),
                research_question=(
                    f"在{clean_direction}课堂中，教师反馈行为具有哪些类型与分布特征？"
                ),
                research_subject="候选对象：高校体育课堂及任课教师，需取得课堂研究许可。",
                research_value="为改进课堂反馈提供描述性证据，不直接推断因果关系。",
                suggested_design="可采用结构化课堂观察，并评估观察者一致性。",
            ),
            TopicSuggestion(
                title=f"{clean_direction}相关体育科研热点与演进路径分析",
                background=(
                    f"围绕“{clean_direction}”检索真实数据库中的研究记录，"
                    "通过关键词共现和年份分布描述研究热点。"
                ),
                research_question=(
                    f"近十年{clean_direction}研究的关键词、研究对象和方法呈现"
                    "怎样的变化？"
                ),
                research_subject="研究对象为可核验的文献记录，不涉及编造文献条目。",
                research_value="帮助定位研究空白，但数据库范围和时间窗必须预先说明。",
                suggested_design="建议使用 CNKI 与 Web of Science 的可导出题录并记录检索日期。",
            ),
        )

        return candidates[:count]

    def list_tools(self) -> dict[str, str]:
        """返回当前规划器认知中的工具及模块位置。"""

        return dict(self.TOOL_CATALOG)

    @staticmethod
    def _detect_task_type(task: str) -> str:
        """根据关键词给出粗略任务类型，便于展示规划侧重点。"""

        lower_task = task.lower()
        if any(
            keyword in lower_task
            for keyword in ("数据", "excel", "csv", "统计", "问卷", "测量")
        ):
            return "数据分析型"
        if any(
            keyword in lower_task
            for keyword in ("实验", "干预", "对照", "训练方案")
        ):
            return "实验研究型"
        if any(
            keyword in lower_task
            for keyword in ("文献", "综述", "研究热点", "知识图谱")
        ):
            return "文献研究型"
        return "综合研究型"
