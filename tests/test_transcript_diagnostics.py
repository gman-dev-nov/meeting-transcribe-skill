import importlib.util
import sys
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "diagnostic_dual", Path(__file__).resolve().parents[1] / "scripts" / "dual_transcribe.py"
)
dual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dual
SPEC.loader.exec_module(dual)


class TranscriptDiagnosticsTests(unittest.TestCase):
    def test_flags_repeated_output_without_erasing_or_selecting_text(self):
        data = {"segments": [{"start": i * 2.0, "end": (i + 1) * 2.0,
                               "text": "Повторяемая фраза."} for i in range(10)]}
        result = dual.transcript_diagnostics(data, 160)
        self.assertIn("suspected_repetition", result["warnings"])
        self.assertEqual(result["most_repeated_segment_count"], 10)
        self.assertEqual(len(data["segments"]), 10)
        self.assertNotIn("text", result)

    def test_two_legitimate_repetitions_do_not_trigger_warning(self):
        data = {"segments": [{"start": i, "end": i + 1, "text": text}
                              for i, text in enumerate(["Да", "Да", "Другое", "Пятое"])]}
        self.assertEqual(dual.transcript_diagnostics(data, 160)["warnings"], [])

    def test_zero_duration_is_reported_separately(self):
        data = {"segments": [{"start": i, "end": i, "text": str(i)} for i in range(3)]}
        result = dual.transcript_diagnostics(data, 160)
        self.assertEqual(result["warnings"], ["zero_duration_segments"])
        self.assertEqual(result["zero_duration_segments"], 3)


if __name__ == "__main__":
    unittest.main()
