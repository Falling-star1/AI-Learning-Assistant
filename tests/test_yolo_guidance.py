import unittest


class YoloSummaryGuidanceTests(unittest.TestCase):
    def test_summary_mentions_general_model_limitations(self):
        from modules.yolo.analyzer import summarize_detections

        summary = summarize_detections([
            {"class_name": "person", "confidence": 0.9},
        ])

        self.assertIn("V1 使用 COCO 通用预训练模型", summary)
        self.assertIn("动漫", summary)
        self.assertIn("误检", summary)


if __name__ == "__main__":
    unittest.main()
