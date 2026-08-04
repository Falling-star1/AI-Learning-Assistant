SYSTEM_PROMPT = """
你是 AI 多模态学习助手。回答应准确、简洁，并优先依据用户提供的课程资料。如果资料不足，请明确说明，不要编造引用或结论。
""".strip()

RAG_PROMPT_TEMPLATE = """
请仅依据以下课程资料回答问题。回答末尾列出使用到的资料来源。

课程资料：
{context}

用户问题：
{question}
""".strip()

SUMMARY_TEMPLATE = """
请围绕主题“{topic}”生成课程总结。

输出结构：
1. 核心概念
2. 关键流程
3. 重要术语
4. 应用场景
5. 易错点
6. 练习题与参考答案\n   - 选择题 3 道\n   - 简答题 2 道\n   - 实践题 1 道

要求：语言适合复习和答辩说明；如果提供了课程资料，请优先依据资料总结。
""".strip()

OUTLINE_TEMPLATE = """
请围绕主题“{topic}”生成复习提纲。

输出结构：
1. 学习目标
2. 知识框架
3. 高频考点
4. 必会问题
5. 复习检查清单

要求：提纲要便于考试复习、课程答辩和项目讲解。
""".strip()

REPORT_TEMPLATE = """
请围绕主题“{topic}”生成报告大纲。

输出结构：
1. 报告标题
2. 章节结构
3. 每章写作要点
4. 可展示的项目亮点
5. 需要补充的材料

要求：大纲要适合作为课程大作业报告或答辩文稿的写作骨架。
""".strip()

PPT_OUTLINE_TEMPLATE = """
请围绕主题“{topic}”生成答辩 PPT 提纲。

输出结构：
1. PPT 标题与封面要点
2. 目录结构（按页给出标题）
3. 每页内容要点（标题 + 3-5 条要点）
4. 演讲节奏建议（每页大约讲解时长）
5. 可视化建议（适合用图表、截图或代码展示的页面）

要求：提纲要适合课程大作业答辩，页数控制在 8-12 页；如果提供了课程资料，请优先依据资料提炼要点。
""".strip()

YOLO_ANALYSIS_TEMPLATE = """
请根据 YOLO 目标检测结果，用自然语言解释图片内容。

用户需求：{question}
检测结果：
{detections}

要求：
1. 先说明检测到了哪些对象和数量。
2. 再推测可能的场景，但要说明这是基于检测结果的合理推断。
3. 如果检测类别明显不符合图片内容，提醒用户通用 YOLO 模型可能存在误检。
""".strip()

LEARNING_TEMPLATES = {
    "课程总结": SUMMARY_TEMPLATE,
    "复习提纲": OUTLINE_TEMPLATE,
    "报告大纲": REPORT_TEMPLATE,
    "PPT 提纲": PPT_OUTLINE_TEMPLATE,
    "学习报告": SUMMARY_TEMPLATE,
}


def build_learning_prompt(
    learning_type: str,
    topic: str,
    has_context: bool = False,
) -> str:
    template = LEARNING_TEMPLATES.get(learning_type, SUMMARY_TEMPLATE)
    context_rule = (
        "资料使用要求：请仅依据提供的课程资料总结；资料未覆盖的内容要明确标注为补充建议。"
        if has_context
        else "资料使用要求：当前没有检索到课程资料，请将结果标注为通用草稿。"
    )
    return f"{template.format(topic=topic)}\n\n{context_rule}"