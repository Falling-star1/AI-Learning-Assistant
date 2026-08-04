DEFAULT_EXPERIMENT_SECTIONS = [
    "实验名称",
    "实验目的",
    "实验环境",
    "技术路线",
    "实验步骤",
    "核心代码说明",
    "实验结果",
    "问题与解决",
    "总结反思",
]


def build_experiment_report_prompt(topic: str, format_requirements: str = "") -> str:
    cleaned_format = format_requirements.strip()
    if cleaned_format:
        return f"""请根据用户提供的格式要求生成实验报告。

实验内容：{topic}

用户格式要求：
{cleaned_format}

请优先遵循用户提供的格式要求；如果某个必要信息缺失，可以在对应位置写“待补充”。"""

    default_sections = "\n".join(f"- {section}" for section in DEFAULT_EXPERIMENT_SECTIONS)
    return f"""请围绕以下实验内容生成一份实验报告草稿。

实验内容：{topic}

请按以下默认结构输出：
{default_sections}

要求：突出实验目标、技术路线、关键步骤、结果分析和总结反思。"""


def build_report_prompt(
    topic: str,
    report_type: str,
    format_requirements: str = "",
    has_context: bool = False,
) -> str:
    from modules.llm.prompt import build_learning_prompt

    if report_type == "实验报告":
        prompt = build_experiment_report_prompt(topic, format_requirements)
    else:
        prompt = build_learning_prompt(report_type, topic, has_context=has_context)
    return prompt