from pathlib import Path
from typing import Any

from PIL import Image

from config import DETECTED_IMAGE_DIR, YOLO_CONFIDENCE_THRESHOLD, YOLO_MODEL_NAME
from modules.utils.file_utils import ensure_directory


def detect_image(
    image_path: str | Path,
    output_dir: str | Path = DETECTED_IMAGE_DIR,
    model_name: str = YOLO_MODEL_NAME,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Run YOLO detection and return data the UI/Agent can consume directly."""
    source_path = Path(image_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Image not found: {source_path}")

    target_dir = ensure_directory(Path(output_dir))
    model = _load_model(model_name, model_factory)
    results = model(str(source_path))
    first_result = results[0] if results else None

    detections = _extract_detections(first_result, confidence_threshold)
    annotated_path = target_dir / f"{source_path.stem}_detected{source_path.suffix}"
    _save_annotated_image(first_result, annotated_path, source_path)

    return {
        "image_path": str(source_path),
        "annotated_image_path": str(annotated_path),
        "detections": detections,
        "confidence_threshold": confidence_threshold,
    }


def _load_model(model_name: str, model_factory: Any | None) -> Any:
    if model_factory is not None:
        return model_factory(model_name)

    # YOLO weights can be slow to load/download, so import only when this mode is used.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，无法运行 YOLO 图片检测。") from exc
    return YOLO(model_name)


def _extract_detections(result: Any | None, confidence_threshold: float) -> list[dict[str, Any]]:
    if result is None or getattr(result, "boxes", None) is None:
        return []

    boxes = result.boxes
    coordinates = _as_list(boxes.xyxy)
    confidences = _as_list(boxes.conf)
    class_ids = [int(value) for value in _as_list(boxes.cls)]
    names = getattr(result, "names", {})

    # Keep detections structured so later LLM analysis and UI rendering do not parse text.
    detections: list[dict[str, Any]] = []
    for index, class_id in enumerate(class_ids):
        confidence = round(float(confidences[index]), 4)
        if confidence < confidence_threshold:
            continue
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": confidence,
                "box": [round(float(value), 2) for value in coordinates[index]],
            }
        )
    return detections


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _save_annotated_image(result: Any | None, annotated_path: Path, source_path: Path) -> None:
    if result is None or not hasattr(result, "plot"):
        annotated_path.write_bytes(source_path.read_bytes())
        return

    plotted = result.plot()
    Image.fromarray(plotted).save(annotated_path)
