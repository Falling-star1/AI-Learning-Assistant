import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeBoxes:
    xyxy = FakeTensor([[10.0, 20.0, 110.0, 120.0], [30.0, 40.0, 90.0, 100.0]])
    conf = FakeTensor([0.95, 0.81])
    cls = FakeTensor([0, 1])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person", 1: "book"}

    def plot(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


class FakeYOLOModel:
    def __call__(self, image_path):
        return [FakeResult()]


class YoloDetectionTests(unittest.TestCase):
    def test_detect_image_returns_structured_detections_and_annotated_image(self):
        from modules.yolo.detect import detect_image

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "scene.png"
            Image.new("RGB", (128, 128), color="white").save(image_path)

            result = detect_image(
                image_path,
                output_dir=tmp_path / "detected",
                model_factory=lambda _: FakeYOLOModel(),
            )

            self.assertEqual(result["image_path"], str(image_path))
            self.assertTrue(Path(result["annotated_image_path"]).exists())
            self.assertEqual(len(result["detections"]), 2)
            self.assertEqual(result["detections"][0]["class_name"], "person")
            self.assertEqual(result["detections"][0]["confidence"], 0.95)
            self.assertEqual(result["detections"][0]["box"], [10.0, 20.0, 110.0, 120.0])

    def test_detect_image_filters_low_confidence_detections(self):
        from modules.yolo.detect import detect_image

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "scene.png"
            Image.new("RGB", (128, 128), color="white").save(image_path)

            result = detect_image(
                image_path,
                output_dir=tmp_path / "detected",
                confidence_threshold=0.9,
                model_factory=lambda _: FakeYOLOModel(),
            )

            self.assertEqual(len(result["detections"]), 1)
            self.assertEqual(result["detections"][0]["class_name"], "person")

    def test_summarize_detections_counts_each_class(self):
        from modules.yolo.analyzer import summarize_detections

        summary = summarize_detections(
            [
                {"class_name": "person", "confidence": 0.9},
                {"class_name": "person", "confidence": 0.8},
                {"class_name": "book", "confidence": 0.7},
            ]
        )

        self.assertIn("person 2 个", summary)
        self.assertIn("book 1 个", summary)

    def test_route_user_request_handles_uploaded_image_detection(self):
        from modules.agent import router

        class UploadedImage:
            name = "classroom.png"

            def getbuffer(self):
                image = Image.new("RGB", (32, 32), color="white")
                buffer_path = Path(tempfile.gettempdir()) / "fake-upload-classroom.png"
                image.save(buffer_path)
                return buffer_path.read_bytes()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_detection = {
                "image_path": str(tmp_path / "images" / "classroom.png"),
                "annotated_image_path": str(tmp_path / "detected" / "classroom_detected.png"),
                "detections": [{"class_name": "person", "confidence": 0.93, "box": [1, 2, 3, 4]}],
            }

            with patch.object(router, "detect_image", return_value=fake_detection):
                result = router.route_user_request(
                    question="检测这张图片",
                    mode="图片目标检测",
                    uploaded_image=UploadedImage(),
                    image_dir=tmp_path / "images",
                    detected_dir=tmp_path / "detected",
                    db_path=tmp_path / "app.db",
                    session_id="session-image",
                )

            self.assertEqual(result["provider"], "yolo")
            self.assertIn("person 1 个", result["answer"])
            self.assertEqual(result["annotated_image_path"], fake_detection["annotated_image_path"])

    def test_route_user_request_passes_confidence_threshold_to_yolo(self):
        from modules.agent import router

        class UploadedImage:
            name = "classroom.png"

            def getbuffer(self):
                image = Image.new("RGB", (32, 32), color="white")
                buffer_path = Path(tempfile.gettempdir()) / "fake-threshold-classroom.png"
                image.save(buffer_path)
                return buffer_path.read_bytes()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_detection = {
                "image_path": str(tmp_path / "images" / "classroom.png"),
                "annotated_image_path": str(tmp_path / "detected" / "classroom_detected.png"),
                "detections": [],
            }

            with patch.object(router, "detect_image", return_value=fake_detection) as mock_detect:
                router.route_user_request(
                    question="检测这张图片",
                    mode="图片目标检测",
                    uploaded_image=UploadedImage(),
                    image_dir=tmp_path / "images",
                    detected_dir=tmp_path / "detected",
                    confidence_threshold=0.65,
                    db_path=tmp_path / "app.db",
                    session_id="session-image",
                )

            self.assertEqual(mock_detect.call_args.kwargs["confidence_threshold"], 0.65)


if __name__ == "__main__":
    unittest.main()
