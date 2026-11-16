"""体育科研知识图谱的研究方向与变量分类词表。

词表只用于识别论文中明确出现的概念。调用方只有在全文或摘要中存在对应
文字证据时才创建节点和关系，词表本身不会生成任何研究结论。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableDefinition:
    """可由真实论文文本识别的变量。"""

    label: str
    role: str
    role_kind: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class DirectionDefinition:
    """一个研究主题下的具体研究方向。"""

    direction_id: str
    theme_id: str
    label: str
    description: str
    terms: tuple[str, ...]
    variables: tuple[VariableDefinition, ...]


def _variable(
    label: str,
    role: str,
    role_kind: str,
    terms: tuple[str, ...],
) -> VariableDefinition:
    return VariableDefinition(label, role, role_kind, terms)


EXERCISE = _variable(
    "体育锻炼",
    "自变量/暴露",
    "exposure",
    (
        "体育锻炼",
        "身体活动",
        "有氧运动",
        "运动干预",
        "运动训练",
        "physical activity",
        "exercise",
        "training",
    ),
)

SCHOOL_PE = _variable(
    "学校体育课程",
    "干预方式",
    "exposure",
    (
        "体育课程",
        "体育教学",
        "physical education",
        "school-based",
        "school intervention",
    ),
)

CARDIO = _variable(
    "心肺耐力",
    "因变量/结局",
    "outcome",
    (
        "心肺耐力",
        "心肺适能",
        "cardiorespiratory fitness",
        "aerobic capacity",
    ),
)

VO2 = _variable(
    "最大摄氧量",
    "测量指标",
    "outcome",
    ("最大摄氧量", "vo2", "vo₂", "maximal oxygen uptake"),
)

BODY_FAT = _variable(
    "体脂率",
    "测量指标",
    "outcome",
    ("体脂率", "体脂", "body fat", "fat mass"),
)

BMI = _variable(
    "BMI",
    "测量指标",
    "outcome",
    ("bmi", "body mass index", "体重指数"),
)

LEAN_MASS = _variable(
    "瘦体重",
    "测量指标",
    "outcome",
    ("瘦体重", "lean mass", "muscle mass"),
)

MUSCLE_STRENGTH = _variable(
    "肌肉力量",
    "因变量/结局",
    "outcome",
    ("肌肉力量", "muscle strength", "muscular strength", "strength performance"),
)

POWER = _variable(
    "爆发力",
    "因变量/结局",
    "outcome",
    ("爆发力", "explosive power", "lower-body power", "power output", "jump performance"),
)

SPEED = _variable(
    "速度表现",
    "因变量/结局",
    "outcome",
    ("速度表现", "sprint", "speed", "短跑"),
)

FLEXIBILITY = _variable(
    "柔韧性",
    "因变量/结局",
    "outcome",
    ("柔韧性", "flexibility", "range of motion", "sit-and-reach"),
)

BONE = _variable(
    "骨密度/骨量",
    "因变量/结局",
    "outcome",
    ("骨密度", "骨量", "bone mineral density", "bone mineral content"),
)

PHYSICAL_FITNESS = _variable(
    "体质健康/身体素质",
    "因变量/结局",
    "outcome",
    (
        "体质健康",
        "身体素质",
        "physical fitness",
        "health-related fitness",
        "fitness",
    ),
)

MENTAL_HEALTH = _variable(
    "心理健康",
    "因变量/结局",
    "outcome",
    ("心理健康", "mental health", "psychological health", "well-being", "wellbeing"),
)

DEPRESSION = _variable(
    "抑郁",
    "因变量/结局",
    "outcome",
    ("抑郁", "depression", "depressive"),
)

ANXIETY = _variable(
    "焦虑",
    "因变量/结局",
    "outcome",
    ("焦虑", "anxiety"),
)

SELF_ESTEEM = _variable(
    "自尊",
    "因变量/结局",
    "outcome",
    ("自尊", "self-esteem", "self esteem"),
)

MOTIVATION = _variable(
    "运动动机",
    "研究变量",
    "mediator",
    ("运动动机", "motivation", "exercise motivation"),
)

INTEREST = _variable(
    "运动兴趣",
    "研究变量",
    "mediator",
    ("运动兴趣", "exercise enjoyment", "enjoyment of physical activity", "interest in physical activity"),
)

PARTICIPATION = _variable(
    "体育参与",
    "因变量/行为结局",
    "outcome",
    ("体育参与", "运动参与", "sport participation", "physical activity participation"),
)

ADHERENCE = _variable(
    "运动依从性",
    "研究变量",
    "mediator",
    ("运动依从性", "依从性", "adherence", "intervention completion"),
)

SELF_EFFICACY = _variable(
    "自我效能",
    "研究变量",
    "mediator",
    ("自我效能", "self-efficacy", "self efficacy"),
)

BODY_IMAGE = _variable(
    "身体意象",
    "因变量/结局",
    "outcome",
    ("身体意象", "body image", "body image perception"),
)

ACADEMIC = _variable(
    "学业表现",
    "因变量/结局",
    "outcome",
    ("学业表现", "academic achievement", "cognitive performance", "educational outcomes"),
)

INTENSITY = _variable(
    "运动强度",
    "自变量/剂量",
    "exposure",
    ("运动强度", "exercise intensity", "high-intensity", "vigorous", "mvpa"),
)

FREQUENCY = _variable(
    "运动频率",
    "自变量/剂量",
    "exposure",
    ("运动频率", "exercise frequency", "frequency", "sessions per week", "days per week"),
)

DURATION = _variable(
    "干预周期",
    "研究设计变量",
    "exposure",
    ("干预周期", "duration", "weeks", "months", "program length"),
)

SESSION_TIME = _variable(
    "单次运动时间",
    "自变量/剂量",
    "exposure",
    ("单次运动时间", "session duration", "minutes per session", "min/session"),
)

MODE = _variable(
    "运动方式",
    "干预方式",
    "exposure",
    ("运动方式", "exercise mode", "training mode", "training type"),
)

AGE = _variable(
    "年龄",
    "调节变量",
    "moderator",
    ("年龄", "age", "aged", "years old"),
)

SEX = _variable(
    "性别",
    "调节变量",
    "moderator",
    ("性别", "sex", "gender", "boys", "girls"),
)

FAMILY = _variable(
    "家庭环境/支持",
    "自变量/影响因素",
    "exposure",
    ("家庭环境", "家庭支持", "parental", "family support", "family-based"),
)

SCHOOL_ENV = _variable(
    "学校环境",
    "自变量/影响因素",
    "exposure",
    ("学校环境", "school environment", "school support"),
)

SOCIOECONOMIC = _variable(
    "社会经济地位",
    "调节变量",
    "moderator",
    ("社会经济地位", "socioeconomic", "socio-economic status", "socioeconomic position"),
)

BUILT_ENV = _variable(
    "建成环境",
    "自变量/影响因素",
    "exposure",
    ("建成环境", "built environment", "neighborhood environment"),
)

SOCIAL_SUPPORT = _variable(
    "社会支持",
    "自变量/影响因素",
    "exposure",
    ("社会支持", "social support", "peer support"),
)

SEDENTARY = _variable(
    "久坐行为",
    "研究变量",
    "exposure",
    ("久坐行为", "sedentary behavior", "sedentary behaviour", "prolonged sitting"),
)

BASELINE = _variable(
    "运动基础/基线水平",
    "调节变量",
    "moderator",
    ("运动基础", "基线水平", "baseline", "training status", "maturity"),
)

INJURY = _variable(
    "运动损伤发生率",
    "安全性结局",
    "outcome",
    ("运动损伤", "injury incidence", "injury rate", "安全"),
)


RESEARCH_DIRECTIONS: tuple[DirectionDefinition, ...] = (
    DirectionDefinition(
        "direction:health:cardiorespiratory",
        "theme:health-assessment",
        "心肺耐力",
        "体育活动和运动训练与儿童青少年心肺适能、有氧能力的关系。",
        ("心肺耐力", "心肺适能", "cardiorespiratory fitness", "aerobic capacity", "vo2", "vo₂"),
        (EXERCISE, CARDIO, VO2),
    ),
    DirectionDefinition(
        "direction:health:body-composition",
        "theme:health-assessment",
        "身体成分",
        "体育锻炼与体脂、瘦体重、BMI 和肥胖指标。",
        ("身体成分", "body composition", "体脂", "body fat", "fat mass", "lean mass", "bmi"),
        (EXERCISE, BODY_FAT, BMI, LEAN_MASS),
    ),
    DirectionDefinition(
        "direction:health:muscular-fitness",
        "theme:health-assessment",
        "肌肉力量与爆发力",
        "力量、爆发力、跳跃和速度相关体适能。",
        ("肌肉力量", "muscle strength", "muscular fitness", "strength", "power", "jump", "sprint"),
        (EXERCISE, MUSCLE_STRENGTH, POWER, SPEED),
    ),
    DirectionDefinition(
        "direction:health:flexibility",
        "theme:health-assessment",
        "身体柔韧性",
        "柔韧性和关节活动范围相关证据。",
        ("柔韧性", "flexibility", "range of motion", "sit-and-reach"),
        (EXERCISE, FLEXIBILITY),
    ),
    DirectionDefinition(
        "direction:health:bone",
        "theme:health-assessment",
        "骨密度与骨量",
        "学校体育和运动干预与骨矿含量、骨密度。",
        ("骨密度", "骨量", "bone mineral density", "bone mineral content"),
        (SCHOOL_PE, BONE),
    ),
    DirectionDefinition(
        "direction:health:comprehensive",
        "theme:health-assessment",
        "综合体质评价",
        "整体体质健康、身体素质和综合 fitness 指标。",
        ("体质健康", "身体素质", "physical fitness", "health-related fitness", "fitness assessment"),
        (EXERCISE, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:health:weight-status",
        "theme:health-assessment",
        "超重与肥胖",
        "超重、肥胖和体重状态相关结局。",
        ("超重", "肥胖", "overweight", "obesity", "excess body weight"),
        (EXERCISE, BMI, BODY_FAT),
    ),
    DirectionDefinition(
        "direction:intervention:aerobic",
        "theme:exercise-intervention",
        "有氧运动",
        "有氧运动和耐力训练干预。",
        ("有氧运动", "aerobic exercise", "endurance training"),
        (EXERCISE, CARDIO, VO2),
    ),
    DirectionDefinition(
        "direction:intervention:resistance",
        "theme:exercise-intervention",
        "力量训练",
        "抗阻、力量和爆发式训练。",
        ("力量训练", "抗阻训练", "resistance training", "strength training"),
        (EXERCISE, MUSCLE_STRENGTH, POWER),
    ),
    DirectionDefinition(
        "direction:intervention:hiit",
        "theme:exercise-intervention",
        "高强度间歇训练",
        "HIIT 和不同高强度间歇方案。",
        ("高强度间歇", "high-intensity interval", "hiit"),
        (INTENSITY, CARDIO, BODY_FAT),
    ),
    DirectionDefinition(
        "direction:intervention:plyometric",
        "theme:exercise-intervention",
        "增强式/跳跃训练",
        "Plyometric 和跳跃训练对体适能及运动表现的影响。",
        ("增强式", "跳跃训练", "plyometric", "jump training"),
        (EXERCISE, POWER, SPEED, MUSCLE_STRENGTH),
    ),
    DirectionDefinition(
        "direction:intervention:school-based",
        "theme:exercise-intervention",
        "学校体育课程",
        "体育课程、学校干预和课程化运动方案。",
        ("学校体育", "体育课程", "体育教学", "school-based", "physical education"),
        (SCHOOL_PE, PHYSICAL_FITNESS, PARTICIPATION),
    ),
    DirectionDefinition(
        "direction:intervention:active-breaks",
        "theme:exercise-intervention",
        "课堂活动与久坐中断",
        "课间活动、课堂身体活动和久坐中断。",
        ("课堂活动", "课间活动", "movement break", "classroom activity", "prolonged sitting"),
        (SCHOOL_PE, SEDENTARY, PARTICIPATION),
    ),
    DirectionDefinition(
        "direction:intervention:family",
        "theme:exercise-intervention",
        "家庭体育干预",
        "家庭参与、父母支持和家庭为基础的运动干预。",
        ("家庭体育", "亲子", "family-based", "parental", "family support"),
        (FAMILY, PARTICIPATION, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:intervention:digital",
        "theme:exercise-intervention",
        "数字运动与运动游戏",
        "mHealth、App、虚拟现实和运动游戏干预。",
        ("运动游戏", "虚拟现实", "mhealth", "mobile health", "virtual reality", "app-based"),
        (EXERCISE, PARTICIPATION, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:intervention:combined",
        "theme:exercise-intervention",
        "组合与同期训练",
        "组合训练、同期训练和多组分运动方案。",
        ("组合训练", "同期训练", "concurrent training", "combined training", "multi-component"),
        (EXERCISE, MUSCLE_STRENGTH, CARDIO),
    ),
    DirectionDefinition(
        "direction:prescription:intensity",
        "theme:exercise-prescription",
        "运动强度",
        "低、中、高强度和 MVPA 等运动剂量。",
        ("运动强度", "高强度", "中等强度", "exercise intensity", "vigorous", "mvpa"),
        (EXERCISE, INTENSITY, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:prescription:frequency",
        "theme:exercise-prescription",
        "运动频率",
        "每周次数和运动频率。",
        ("运动频率", "frequency", "sessions per week", "days per week"),
        (EXERCISE, FREQUENCY, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:prescription:duration",
        "theme:exercise-prescription",
        "干预周期",
        "干预持续周数、月数和长期方案。",
        ("干预周期", "duration", "weeks", "months", "program length"),
        (EXERCISE, DURATION, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:prescription:session-time",
        "theme:exercise-prescription",
        "单次运动时间",
        "每次运动分钟数和单次剂量。",
        ("单次运动时间", "session duration", "minutes per session", "min/session"),
        (EXERCISE, SESSION_TIME, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:prescription:mode",
        "theme:exercise-prescription",
        "运动方式与比较",
        "不同运动方式、训练类型和方案比较。",
    ("运动方式", "exercise mode", "training mode", "training type"),
        (EXERCISE, MODE, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:psych:mental-health",
        "theme:psych-behavior",
        "心理健康结局",
        "抑郁、焦虑、心理健康和幸福感。",
        ("心理健康", "mental health", "抑郁", "depression", "焦虑", "anxiety", "well-being"),
        (MENTAL_HEALTH, DEPRESSION, ANXIETY),
    ),
    DirectionDefinition(
        "direction:psych:motivation",
        "theme:psych-behavior",
        "运动动机",
        "运动动机及其与体育参与的关系。",
        ("运动动机", "motivation", "exercise motivation"),
        (MOTIVATION, PARTICIPATION, ADHERENCE),
    ),
    DirectionDefinition(
        "direction:psych:interest",
        "theme:psych-behavior",
        "运动兴趣与体验",
        "运动兴趣、enjoyment 和参与体验。",
        ("运动兴趣", "enjoyment", "interest", "experience"),
        (INTEREST, PARTICIPATION, ADHERENCE),
    ),
    DirectionDefinition(
        "direction:psych:participation",
        "theme:psych-behavior",
        "体育参与",
        "体育参与、身体活动参与和参与障碍。",
        ("体育参与", "运动参与", "sport participation", "physical activity participation", "barriers"),
        (PARTICIPATION, MOTIVATION, FAMILY),
    ),
    DirectionDefinition(
        "direction:psych:adherence",
        "theme:psych-behavior",
        "运动依从性",
        "运动干预完成率、依从性和持续参与。",
        ("运动依从性", "依从性", "adherence", "completion", "dropout"),
        (ADHERENCE, MOTIVATION, INTEREST),
    ),
    DirectionDefinition(
        "direction:psych:self-efficacy",
        "theme:psych-behavior",
        "自我效能",
        "自我效能与运动行为或心理健康。",
        ("自我效能", "self-efficacy", "self efficacy"),
        (SELF_EFFICACY, PARTICIPATION, MENTAL_HEALTH),
    ),
    DirectionDefinition(
        "direction:psych:body-image",
        "theme:psych-behavior",
        "身体意象",
        "身体意象、自尊和身体认知。",
        ("身体意象", "body image", "self-esteem", "self esteem"),
        (BODY_IMAGE, SELF_ESTEEM),
    ),
    DirectionDefinition(
        "direction:psych:cognition",
        "theme:psych-behavior",
        "认知与学业表现",
        "体育运动与认知、学业和教育结局。",
        ("认知", "学业", "academic achievement", "cognitive", "educational outcomes"),
        (ACADEMIC, MENTAL_HEALTH, SCHOOL_PE),
    ),
    DirectionDefinition(
        "direction:outcome:pre-post",
        "theme:outcome-evaluation",
        "干预前后变化",
        "干预前后或组间变化结果。",
        ("干预前后", "pre-post", "before and after", "组间比较", "between-group"),
        (EXERCISE, PHYSICAL_FITNESS, BODY_FAT),
    ),
    DirectionDefinition(
        "direction:outcome:effect-size",
        "theme:outcome-evaluation",
        "干预效果幅度",
        "效应量、改善幅度和统计显著性。",
        ("效应量", "effect size", "mean difference", "改善幅度", "confidence interval"),
        (PHYSICAL_FITNESS, MUSCLE_STRENGTH, CARDIO, BODY_FAT),
    ),
    DirectionDefinition(
        "direction:outcome:comparative",
        "theme:outcome-evaluation",
        "不同干预方式比较",
        "不同训练方案、课程和干预方式的比较。",
        ("比较", "comparative", "compared with", "network meta-analysis", "different exercise"),
        (MODE, INTENSITY, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:outcome:long-term",
        "theme:outcome-evaluation",
        "长期效果与持续性",
        "长期随访、效果维持和持续参与。",
        ("长期", "随访", "long-term", "follow-up", "maintenance", "sustainability"),
        (DURATION, PHYSICAL_FITNESS, ADHERENCE),
    ),
    DirectionDefinition(
        "direction:outcome:measurement",
        "theme:outcome-evaluation",
        "结局测量与评价指标",
        "评价指标、测量工具和结局定义。",
        ("测量指标", "评价指标", "outcome measure", "measurement", "questionnaire", "test"),
        (VO2, BMI, BODY_FAT, MUSCLE_STRENGTH),
    ),
    DirectionDefinition(
        "direction:outcome:safety",
        "theme:outcome-evaluation",
        "安全性与运动损伤",
        "运动干预的运动损伤和安全性结局。",
        ("运动损伤", "安全性", "injury incidence", "injury rate", "safety"),
        (INJURY, EXERCISE, INTENSITY),
    ),
    DirectionDefinition(
        "direction:factor:age",
        "theme:mechanism",
        "年龄与发育阶段",
        "年龄、青春期和成熟度对运动效果的影响。",
        ("年龄", "青春期", "成熟度", "age", "maturity", "puberty"),
        (AGE, BASELINE, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:factor:sex",
        "theme:mechanism",
        "性别差异",
        "性别和男女差异。",
        ("性别", "男孩", "女孩", "sex", "gender", "boys", "girls"),
        (SEX, PARTICIPATION, PHYSICAL_FITNESS),
    ),
    DirectionDefinition(
        "direction:factor:family",
        "theme:mechanism",
        "家庭因素",
        "父母、家庭环境和家庭支持对运动行为的影响。",
        ("家庭环境", "家庭支持", "父母", "family", "parental", "family support"),
        (FAMILY, PARTICIPATION, SEDENTARY),
    ),
    DirectionDefinition(
        "direction:factor:school",
        "theme:mechanism",
        "学校因素",
        "学校环境、课程和同伴支持。",
        ("学校环境", "学校支持", "school environment", "school support", "peer support"),
        (SCHOOL_ENV, SCHOOL_PE, PARTICIPATION),
    ),
    DirectionDefinition(
        "direction:factor:socioeconomic",
        "theme:mechanism",
        "社会经济因素",
        "社会经济地位和家庭资源差异。",
        ("社会经济地位", "socioeconomic", "ses", "family income"),
        (SOCIOECONOMIC, PARTICIPATION, BUILT_ENV),
    ),
    DirectionDefinition(
        "direction:factor:built-environment",
        "theme:mechanism",
        "建成环境",
        "社区、邻里和建成环境与身体活动。",
        ("建成环境", "邻里环境", "built environment", "neighborhood environment"),
        (BUILT_ENV, PARTICIPATION, SEDENTARY),
    ),
    DirectionDefinition(
        "direction:factor:social-support",
        "theme:mechanism",
        "社会支持",
        "同伴、家庭和社会支持。",
        ("社会支持", "同伴支持", "social support", "peer support"),
        (SOCIAL_SUPPORT, PARTICIPATION, MENTAL_HEALTH),
    ),
    DirectionDefinition(
        "direction:factor:sedentary",
        "theme:mechanism",
        "久坐行为",
        "久坐时间、屏幕行为和身体活动组合。",
        ("久坐行为", "久坐时间", "sedentary behavior", "sedentary behaviour", "screen time"),
        (SEDENTARY, PARTICIPATION, BMI),
    ),
    DirectionDefinition(
        "direction:factor:baseline",
        "theme:mechanism",
        "运动基础与基线水平",
        "基线体适能、训练状态和运动经历。",
        ("运动基础", "基线水平", "训练状态", "baseline", "training status", "maturity"),
        (BASELINE, PHYSICAL_FITNESS, MUSCLE_STRENGTH),
    ),
)


DIRECTION_BY_ID = {
    direction.direction_id: direction
    for direction in RESEARCH_DIRECTIONS
}

VARIABLE_BY_LABEL = {
    variable.label: variable
    for direction in RESEARCH_DIRECTIONS
    for variable in direction.variables
}


__all__ = [
    "DirectionDefinition",
    "VariableDefinition",
    "RESEARCH_DIRECTIONS",
    "DIRECTION_BY_ID",
    "VARIABLE_BY_LABEL",
]
