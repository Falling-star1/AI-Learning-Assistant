from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from modules.llm.prompt import YOLO_ANALYSIS_TEMPLATE
from modules.llm.provider import LLMProvider

YOLO_DEMO_NOTE = "提示：V1 使用 COCO 通用预训练模型，适合识别真实场景中的常见物体；动漫、截图或课程页面可能出现误检，结果应作为演示参考。"


def summarize_detections(detections: Iterable[Mapping[str, Any]]) -> str:
    """Summarize model output before optional LLM interpretation."""
    counts = _count_detections(detections)
    if not counts:
        return f"未检测到目标。\n\n{YOLO_DEMO_NOTE}"

    summary = "，".join(
        f"{class_name} {count} 个"
        for class_name, count in sorted(counts.items())
    )
    return f"检测结果：{summary}。\n\n{YOLO_DEMO_NOTE}"


def analyze_with_llm(
    detections: Iterable[Mapping[str, Any]],
    question: str,
    provider: LLMProvider,
) -> str:
    detection_list = list(detections)
    if not detection_list:
        return summarize_detections(detection_list)

    prompt = YOLO_ANALYSIS_TEMPLATE.format(
        question=question,
        detections=_format_detections(detection_list),
    )
    return provider.generate(prompt).text


def _count_detections(detections: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(item.get("class_name", "unknown")) for item in detections)


def _format_detections(detections: Iterable[Mapping[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(detections, start=1):
        class_name = item.get("class_name", "unknown")
        confidence = item.get("confidence", "?")
        box = item.get("box", [])
        lines.append(f"{index}. class={class_name}, confidence={confidence}, box={box}")
    return "\n".join(lines)