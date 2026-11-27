"""Проверка адаптера длинных записей без torch, моделей и загрузок."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gigaam_longform.py"


class LongformReleaseTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.audio = Path(tmp.name) / "meeting.wav"
        self.audio.touch()
        self.sdk = ModuleType("gigaam")
        self.sdk.GigaAMVAD = object()
        self.model = mock.Mock()
        self.sdk.load_model = mock.Mock(return_value=self.model)
        spec = importlib.util.spec_from_file_location("longform", SCRIPT)
        self.longform = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            spec.loader.exec_module(self.longform)

    def run_main(self, result, *flags, native_silero=False):
        self.model.transcribe_longform.return_value = result
        with mock.patch.dict(sys.modules, {"gigaam": self.sdk, "silero_vad": None}), \
                mock.patch.object(self.longform, "native_silero_supported", return_value=native_silero), \
                mock.patch.object(self.longform, "install_silero_vad_shim") as shim, \
                mock.patch.object(sys, "argv", [str(SCRIPT), str(self.audio), *flags]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.longform.main(), 0)
        self.shim = shim
        return json.loads(self.audio.with_suffix(".transcript.json").read_text(encoding="utf-8"))

    def test_default_v4_selects_formatted_text_and_absolute_word_times(self):
        word = SimpleNamespace(text="Привет!", start=12.5, end=13.0)
        e2e = SimpleNamespace(start=12.4, end=13.2, text="Привет!", words=[word])
        raw = SimpleNamespace(start=12.4, end=13.2, text="привет", words=[])
        result = {"raw": SimpleNamespace(segments=[raw]), "e2e": SimpleNamespace(segments=[e2e])}
        data = self.run_main(result, "--batch-size", "3")
        self.sdk.load_model.assert_called_once_with("v4_rnnt", device="mps", fp16_encoder=False)
        self.model.transcribe_longform.assert_called_once_with(
            str(self.audio.resolve()), word_timestamps=True, fr_batch_size=3
        )
        self.shim.assert_not_called()
        self.assertEqual(data["metadata"]["vad"], "gigaam-vad")
        segment, = data["segments"]
        self.assertEqual(segment["text"], "Привет!")
        self.assertEqual(segment["words"], [
            {"word": "Привет!", "start": 12.5, "end": 13.0, "conf": None}
        ])
        for suffix in (".transcript.md", ".transcript.srt"):
            self.assertIn("Привет!", self.audio.with_suffix(suffix).read_text(encoding="utf-8"))

    def test_v4_with_no_speech_writes_an_empty_transcript(self):
        empty = SimpleNamespace(segments=[])
        self.assertEqual(self.run_main({"e2e": empty, "raw": empty})["segments"], [])
        self.assertEqual(self.audio.with_suffix(".transcript.srt").read_bytes(), b"")

    def test_v3_returns_single_result_on_new_sdk(self):
        seg = SimpleNamespace(text="Тест.", start=0.2, end=1.0, words=[])
        data = self.run_main(SimpleNamespace(segments=[seg]), "--model", "v3_e2e_rnnt")
        self.assertEqual(data["metadata"]["model"], "v3_e2e_rnnt")
        self.assertEqual(data["segments"][0]["text"], "Тест.")
        self.assertEqual(data["metadata"]["vad"], "gigaam-vad")
        self.shim.assert_not_called()

    def test_v3_keeps_both_legacy_silero_routes(self):
        del self.sdk.GigaAMVAD
        for native in (False, True):
            with self.subTest(native=native):
                data = self.run_main(SimpleNamespace(segments=[]), "--model", "v3_e2e_rnnt",
                                     native_silero=native)
                self.assertEqual(data["metadata"]["vad"], "silero-vad")
                kwargs = self.model.transcribe_longform.call_args.kwargs
                self.assertEqual(kwargs.get("vad_backend"), "silero" if native else None)
                self.assertEqual(self.shim.called, not native)

    def test_v4_on_legacy_sdk_reports_upgrade_before_loading_weights(self):
        del self.sdk.GigaAMVAD
        with mock.patch.dict(sys.modules, {"gigaam": self.sdk}), \
                mock.patch.object(sys, "argv", [str(SCRIPT), str(self.audio)]), \
                contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(self.longform.main(), 2)
        self.assertIn("references/setup.md", error.getvalue())
        self.sdk.load_model.assert_not_called()
