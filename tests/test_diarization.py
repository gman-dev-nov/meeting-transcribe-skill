import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("transcribe_diar", SCRIPT)
transcribe = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = transcribe
SPEC.loader.exec_module(transcribe)

DUAL = Path(__file__).resolve().parents[1] / "scripts" / "dual_transcribe.py"
DUAL_SPEC = importlib.util.spec_from_file_location("dual_diar", DUAL)
dual = importlib.util.module_from_spec(DUAL_SPEC)
assert DUAL_SPEC and DUAL_SPEC.loader
sys.modules[DUAL_SPEC.name] = dual
DUAL_SPEC.loader.exec_module(dual)

Segment = transcribe.Segment


class AssignSpeakersTests(unittest.TestCase):
    def test_labels_follow_maximum_overlap(self):
        segments = [Segment(0, 2, "раз"), Segment(2, 4, "два"), Segment(4, 6, "три")]

        speakers = transcribe.assign_speakers(segments, [(0.0, 3.5, "A"), (3.5, 6.0, "B")])

        self.assertEqual(speakers, 2)
        self.assertEqual([s.speaker for s in segments],
                         ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])

    def test_labels_are_renumbered_in_order_of_appearance(self):
        segments = [Segment(0, 2, "раз"), Segment(4, 6, "два")]

        transcribe.assign_speakers(segments, [(4.0, 6.0, "zzz"), (0.0, 2.0, "aaa")])

        self.assertEqual([s.speaker for s in segments], ["SPEAKER_00", "SPEAKER_01"])

    def test_segment_without_overlap_stays_unlabelled_by_default(self):
        segments = [Segment(0, 2, "речь"), Segment(10, 10.4, "короткий")]

        transcribe.assign_speakers(segments, [(0.0, 2.0, "A")])

        self.assertIsNone(segments[1].speaker)

    def test_new_assignment_clears_an_obsolete_speaker_label(self):
        segments = [Segment(10, 12, "нет речи", speaker="SPEAKER_99")]
        speakers = transcribe.assign_speakers(segments, [(0.0, 2.0, "A")])
        self.assertEqual(speakers, 0)
        self.assertIsNone(segments[0].speaker)

    def test_backfill_takes_the_label_of_the_nearest_neighbour(self):
        segments = [Segment(0, 2, "речь"), Segment(10, 10.4, "короткий")]

        transcribe.assign_speakers(segments, [(0.0, 2.0, "A")], backfill=True)

        self.assertEqual(segments[1].speaker, "SPEAKER_00")


class SpansFromSegmentsTests(unittest.TestCase):
    def test_consecutive_same_label_segments_merge(self):
        segments = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(4, 6, "c")]

        spans = transcribe.spans_from_labelled_segments(segments, [0, 1, 2], [7, 7, 3])

        self.assertEqual(spans, [(0, 4, "7"), (4, 6, "3")])

    def test_skipped_segments_do_not_break_ordering(self):
        segments = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(4, 6, "c")]

        spans = transcribe.spans_from_labelled_segments(segments, [0, 2], [1, 1])

        self.assertEqual(spans, [(0, 6, "1")])


class ResemblyzerTimelineTests(unittest.TestCase):
    def dependencies(self, waveform):
        """Лёгкие заглушки моделей с реальным эффектом удаления тишины."""
        encoder = mock.Mock()
        encoder.embed_utterance.side_effect = lambda chunk: [chunk[0]]

        def trim_silence(audio, source_sr=None):
            # Поддержка пути воспроизводит старое поведение: из полной записи
            # исчезали паузы, после чего ASR-срезы попадали в чужую речь.
            samples = waveform if isinstance(audio, (str, Path)) else audio
            return [sample for sample in samples if sample != 0]

        modules = {
            "librosa": mock.Mock(),
            "numpy": mock.Mock(),
            "resemblyzer": mock.Mock(),
            "sklearn": mock.Mock(),
            "sklearn.cluster": mock.Mock(),
            "sklearn.metrics": mock.Mock(),
        }
        modules["librosa"].load.return_value = (waveform, 16000)
        modules["resemblyzer"].VoiceEncoder.return_value = encoder
        modules["resemblyzer"].preprocess_wav.side_effect = trim_silence
        modules["numpy"].vstack.side_effect = lambda embeddings: embeddings
        modules["sklearn.cluster"].AgglomerativeClustering.return_value.fit_predict.side_effect = (
            lambda embeddings: [int(embedding[0]) - 1 for embedding in embeddings]
        )
        return modules, encoder

    def test_silence_removal_does_not_shift_asr_timestamp_slices(self):
        sr = 16000
        waveform = [0] * (2 * sr) + [1] * sr + [0] * (2 * sr) + [2] * sr
        segments = [Segment(1, 3, "первая реплика"), Segment(4, 6, "вторая реплика")]
        modules, encoder = self.dependencies(waveform)

        with mock.patch.dict(sys.modules, modules):
            spans = transcribe.resemblyzer_spans(Path("meeting.wav"), segments, 2)

        self.assertEqual(spans, [(1, 3, "0"), (4, 6, "1")])
        chunks = [call.args[0] for call in encoder.embed_utterance.call_args_list]
        self.assertEqual(chunks, [[1] * sr, [2] * sr])
        modules["librosa"].load.assert_called_once_with("meeting.wav", sr=16000, mono=True)

    def test_segment_with_no_speech_after_preprocessing_is_skipped(self):
        modules, encoder = self.dependencies([0] * 32000)
        with mock.patch.dict(sys.modules, modules):
            spans = transcribe.resemblyzer_spans(
                Path("silence.wav"), [Segment(0, 2, "ложное распознавание")], 1
            )
        self.assertEqual(spans, [])
        encoder.embed_utterance.assert_not_called()


class SharedDiarizationTests(unittest.TestCase):
    """Один расчёт спикеров — одинаковые метки в разных сегментациях."""

    def artifact(self, directory: Path, name: str, bounds, *, engine=None) -> Path:
        path = directory / name
        metadata = {"source_file": "meeting.wav", "model": "test",
                    "duration_seconds": 6.0}
        if engine:
            metadata["engine"] = engine
        path.write_text(
            json.dumps({
                "metadata": metadata,
                "segments": [
                    {"start": start, "end": end, "text": f"фрагмент {index}",
                     "speaker": None, "words": []}
                    for index, (start, end) in enumerate(bounds)
                ],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def test_both_transcripts_get_the_same_speaker_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse = self.artifact(root, "m.gigaam.transcript.json",
                                   [(0, 3), (3, 6)], engine="gigaam")
            fine = self.artifact(root, "m.whisper.transcript.json",
                                 [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 6)])
            spans = [(0.0, 3.0, "A"), (3.0, 6.0, "B")]

            with mock.patch.object(transcribe, "pyannote_spans", return_value=spans):
                speakers = transcribe.apply_shared_diarization(
                    root / "audio.wav", [coarse, fine], "pyannote", None, 6
                )

            self.assertEqual(speakers, 2)
            coarse_labels = [s["speaker"] for s in
                             json.loads(coarse.read_text(encoding="utf-8"))["segments"]]
            fine_labels = [s["speaker"] for s in
                           json.loads(fine.read_text(encoding="utf-8"))["segments"]]
            self.assertEqual(coarse_labels, ["SPEAKER_00", "SPEAKER_01"])
            self.assertEqual(fine_labels, ["SPEAKER_00", "SPEAKER_00",
                                           "SPEAKER_01", "SPEAKER_01"])

    def test_metadata_records_that_diarization_was_shared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = self.artifact(root, "m.gigaam.transcript.json", [(0, 6)],
                                engine="gigaam")
            with mock.patch.object(transcribe, "pyannote_spans",
                                   return_value=[(0.0, 6.0, "A")]):
                transcribe.apply_shared_diarization(
                    root / "audio.wav", [one], "pyannote", None, 6
                )

            metadata = json.loads(one.read_text(encoding="utf-8"))["metadata"]
            self.assertEqual(metadata["diarization_scope"], "shared")
            self.assertEqual(metadata["diarizer"], "pyannote")
            self.assertEqual(metadata["speakers_detected"], 1)

    def test_coarse_transcript_omitting_first_speaker_keeps_shared_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse = self.artifact(root, "m.gigaam.transcript.json", [(0, 6)],
                                   engine="gigaam")
            fine = self.artifact(root, "m.whisper.transcript.json", [(0, 1), (1, 6)])
            spans = [(0.0, 1.0, "A"), (1.0, 6.0, "B")]
            with mock.patch.object(transcribe, "pyannote_spans", return_value=spans):
                transcribe.apply_shared_diarization(
                    root / "audio.wav", [coarse, fine], "pyannote", None, 6
                )
            coarse_data = json.loads(coarse.read_text(encoding="utf-8"))
            fine_data = json.loads(fine.read_text(encoding="utf-8"))
        self.assertEqual([segment["speaker"] for segment in coarse_data["segments"]],
                         ["SPEAKER_01"])
        self.assertEqual([segment["speaker"] for segment in fine_data["segments"]],
                         ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(coarse_data["metadata"]["speakers_detected"], 1)
        self.assertEqual(fine_data["metadata"]["speakers_detected"], 2)

    def test_markdown_and_srt_are_rewritten_with_speakers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = self.artifact(root, "m.gigaam.transcript.json",
                                [(0, 3), (3, 6)], engine="gigaam")
            with mock.patch.object(transcribe, "pyannote_spans",
                                   return_value=[(0.0, 3.0, "A"), (3.0, 6.0, "B")]):
                transcribe.apply_shared_diarization(
                    root / "audio.wav", [one], "pyannote", None, 6
                )

            markdown = (root / "m.gigaam.transcript.md").read_text(encoding="utf-8")
            subtitles = (root / "m.gigaam.transcript.srt").read_text(encoding="utf-8")
            self.assertIn("**SPEAKER_00**", markdown)
            self.assertIn("**SPEAKER_01**", markdown)
            # заголовок GigaAM-артефакта не должен превратиться в whisper-овский
            self.assertIn("движок=gigaam", markdown)
            self.assertNotIn("preset=", markdown)
            self.assertIn("[SPEAKER_00]", subtitles)

    def test_resemblyzer_basis_is_the_finer_transcript(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse = self.artifact(root, "m.gigaam.transcript.json", [(0, 6)],
                                   engine="gigaam")
            fine = self.artifact(root, "m.whisper.transcript.json",
                                 [(0, 2), (2, 4), (4, 6)])
            seen = {}

            def fake_spans(audio, segments, num_speakers, max_speakers):
                seen["count"] = len(segments)
                return [(0.0, 6.0, "A")]

            with mock.patch.object(transcribe, "resemblyzer_spans", fake_spans):
                transcribe.apply_shared_diarization(
                    root / "audio.wav", [coarse, fine], "resemblyzer", None, 6
                )

            self.assertEqual(seen["count"], 3)


class RunnerWiringTests(unittest.TestCase):
    def namespace(self, **overrides) -> argparse.Namespace:
        base = dict(
            input=Path("/tmp/meeting.wav"),
            gigaam_python=sys.executable,
            whisper_python=sys.executable,
            gigaam_device="cpu",
            gigaam_model="v3_e2e_rnnt",
            gigaam_batch_size=4,
            whisper_backend=None,
            initial_prompt=None,
            no_condition_on_previous_text=False,
            diarize=True,
            diarizer="resemblyzer",
            num_speakers=None,
            max_speakers=6,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_asr_pass_never_diarizes_on_its_own(self):
        commands = dual.build_asr_commands(self.namespace(), Path("/tmp/stage"))

        self.assertNotIn("--diarize", commands["whisper"])
        self.assertNotIn("--diarizer", commands["whisper"])

    def test_diarize_command_covers_both_transcripts(self):
        transcripts = [Path("/tmp/stage/m.gigaam.transcript.json"),
                       Path("/tmp/stage/m.whisper.transcript.json")]

        command = dual.build_diarize_command(self.namespace(), transcripts)

        self.assertIn("--diarize-only", command)
        self.assertIn("--apply-to", command)
        for path in transcripts:
            self.assertIn(str(path), command)
        self.assertEqual(command[command.index("--diarizer") + 1], "resemblyzer")

    def test_exact_speaker_count_is_forwarded(self):
        command = dual.build_diarize_command(
            self.namespace(num_speakers=4), [Path("/tmp/a.json")]
        )

        self.assertEqual(command[command.index("--num-speakers") + 1], "4")


if __name__ == "__main__":
    unittest.main()
