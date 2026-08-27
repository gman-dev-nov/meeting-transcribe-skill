import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("transcribe_policy", SCRIPT)
transcribe = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = transcribe
SPEC.loader.exec_module(transcribe)


class ModelPolicyTests(unittest.TestCase):
    def test_cli_help_does_not_offer_model_size(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        self.assertNotIn("--preset", result.stdout)
        self.assertNotIn("--model", result.stdout)
        self.assertNotIn("large-v3-turbo", result.stdout)

    def test_cli_rejects_legacy_model_overrides(self):
        for option in (("--preset", "fast"), ("--model", "tiny"), ("--beam-size", "1")):
            with self.subTest(option=option):
                result = subprocess.run(
                    [sys.executable, str(SCRIPT), "missing.wav", *option],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("unrecognized arguments", result.stderr)

    def test_only_maximum_model_exists_in_policy(self):
        self.assertEqual(set(transcribe.PRESETS), {"quality"})
        self.assertEqual(transcribe.PRESETS["quality"]["model"], "large-v3")
        self.assertEqual(transcribe.PRESETS["quality"]["beam_size"], 5)

    def test_recommendation_is_large_v3_for_clean_and_risky_audio(self):
        clean = transcribe.recommend_preset(
            60.0,
            {"total_seconds": 0.0, "max_seconds": 0.0},
            {"sample_rate": 48000, "bitrate_kbps": 256},
            {"mean_db": -18.0},
            ["whisper-cpp"],
        )
        risky = transcribe.recommend_preset(
            4 * 3600.0,
            {"total_seconds": 1800.0, "max_seconds": 90.0},
            {"sample_rate": 8000, "bitrate_kbps": 32},
            {"mean_db": -35.0},
            ["whisper-cpp"],
        )

        self.assertEqual(clean["preset"], "quality")
        self.assertEqual(risky["preset"], "quality")
        self.assertIn("large-v3", clean["reason"])
        self.assertTrue(risky["no_condition_on_previous_text"])

    def test_estimate_report_only_exposes_large_v3(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "ggml-large-v3.bin"
            model.touch()
            report = transcribe.build_estimate_report(
                600.0,
                {
                    "whisper_cpp_bin": "/tmp/whisper-cli",
                    "whisper_cpp_models_dir": temporary,
                    "mlx_whisper": False,
                    "faster_whisper": False,
                },
                None,
            )

        self.assertEqual(len(report["options"]), 1)
        self.assertEqual(report["options"][0]["model"], "large-v3")
        self.assertEqual(report["options"][0]["preset"], "quality")

    def test_auto_backend_skips_whisper_cpp_without_large_v3(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = {
                "whisper_cpp_bin": "/tmp/whisper-cli",
                "whisper_cpp_models_dir": temporary,
                "mlx_whisper": True,
                "faster_whisper": False,
            }
            selected = transcribe.auto_default_backend(env, "large-v3")

        self.assertEqual(selected, "mlx-whisper")

    def test_quality_prefers_faster_whisper_over_mlx_greedy(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = {
                "whisper_cpp_bin": "/tmp/whisper-cli",
                "whisper_cpp_models_dir": temporary,
                "mlx_whisper": True,
                "faster_whisper": True,
            }
            selected = transcribe.auto_default_backend(env, "large-v3")

        self.assertEqual(selected, "faster-whisper")

    def test_no_backend_when_whisper_cpp_lacks_large_v3(self):
        with tempfile.TemporaryDirectory() as temporary:
            selected = transcribe.auto_default_backend(
                {
                    "whisper_cpp_bin": "/tmp/whisper-cli",
                    "whisper_cpp_models_dir": temporary,
                    "mlx_whisper": False,
                    "faster_whisper": False,
                },
                "large-v3",
            )

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
