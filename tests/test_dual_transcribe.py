import argparse
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dual_transcribe.py"
SPEC = importlib.util.spec_from_file_location("dual_transcribe", SCRIPT)
dual = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = dual
SPEC.loader.exec_module(dual)

SETUP_CHECK = Path(__file__).resolve().parents[1] / "scripts" / "setup_check.py"
SETUP_SPEC = importlib.util.spec_from_file_location("setup_check", SETUP_CHECK)
setup_check = importlib.util.module_from_spec(SETUP_SPEC)
assert SETUP_SPEC and SETUP_SPEC.loader
sys.modules[SETUP_SPEC.name] = setup_check
SETUP_SPEC.loader.exec_module(setup_check)


def transcript(words, *, source="meeting.wav", model="test"):
    text = " ".join(word[2] for word in words)
    start = words[0][0] if words else 0.0
    end = words[-1][1] if words else 0.0
    return {
        "metadata": {"source_file": source, "model": model},
        "segments": [
            {
                "start": start,
                "end": end,
                "text": text,
                "words": [
                    {"start": item_start, "end": item_end, "word": word}
                    for item_start, item_end, word in words
                ],
            }
        ]
        if words
        else [],
    }


class ComparisonTests(unittest.TestCase):
    def test_detects_negation_and_produces_source_interval(self):
        gigaam = transcript(
            [(10.0, 10.4, "Мы"), (10.5, 11.0, "будем"), (11.1, 12.0, "делать")]
        )
        whisper = transcript(
            [
                (10.0, 10.4, "Мы"),
                (10.4, 10.6, "не"),
                (10.6, 11.0, "будем"),
                (11.1, 12.0, "делать"),
            ]
        )

        result = dual.build_comparison(gigaam, whisper, Path("meeting.wav"), 30.0)

        self.assertEqual(result["summary"]["candidate_count"], 1)
        candidate = result["candidates"][0]
        self.assertIn("negation_or_polarity", candidate["machine_signals"])
        self.assertEqual(candidate["priority_hint"], "critical")
        self.assertLessEqual(candidate["interval"]["start_seconds"], 10.0)
        self.assertGreaterEqual(candidate["interval"]["end_seconds"], 11.0)
        self.assertEqual(candidate["interval"]["basis"], "source_audio")
        self.assertIn("Мы будем делать", candidate["versions"]["gigaam"])
        self.assertIn("Мы не будем делать", candidate["versions"]["whisper"])

    def test_ignores_case_and_punctuation(self):
        gigaam = transcript([(1.0, 1.5, "Привет,"), (1.5, 2.0, "мир!")])
        whisper = transcript([(1.0, 1.5, "привет"), (1.5, 2.0, "МИР")])

        result = dual.build_comparison(gigaam, whisper, Path("meeting.wav"), 5.0)

        self.assertEqual(result["candidates"], [])

    def test_split_word_is_not_marked_as_polarity_change(self):
        gigaam = transcript([(1.0, 2.0, "необязательно")])
        whisper = transcript([(1.0, 1.3, "не"), (1.3, 2.0, "обязательно")])

        result = dual.build_comparison(gigaam, whisper, Path("meeting.wav"), 5.0)

        self.assertEqual(len(result["candidates"]), 1)
        self.assertNotIn(
            "negation_or_polarity", result["candidates"][0]["machine_signals"]
        )

    def test_segment_fallback_is_explicit(self):
        gigaam = {
            "metadata": {"source_file": "meeting.wav"},
            "segments": [{"start": 1.0, "end": 3.0, "text": "один вариант"}],
        }
        whisper = transcript([(1.0, 2.0, "другой"), (2.0, 3.0, "вариант")])

        result = dual.build_comparison(gigaam, whisper, Path("meeting.wav"), 5.0)

        self.assertEqual(result["summary"]["word_timing_precision"], "segment")

    def test_coarse_segment_uses_honest_full_segment_interval(self):
        gigaam = {"metadata": {"source_file": "meeting.wav"}, "segments": []}
        whisper = {
            "metadata": {"source_file": "meeting.wav"},
            "segments": [{"start": 0.0, "end": 100.0, "text": "галлюцинация"}],
        }

        result = dual.build_comparison(gigaam, whisper, Path("meeting.wav"), 100.0)

        self.assertEqual(len(result["candidates"]), 1)
        interval = result["candidates"][0]["interval"]
        self.assertEqual(interval["start_seconds"], 0.0)
        self.assertEqual(interval["end_seconds"], 100.0)
        self.assertEqual(interval["precision"], "segment")


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.comparison = self.root / "meeting.comparison.json"
        self.comparison.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_audio": {
                        "ref": "/records/meeting.wav",
                        "display_name": "meeting.wav",
                        "duration_seconds": 30.0,
                    },
                    "transcripts": {"gigaam": {}, "whisper": {}},
                    "policy": {
                        "comparison": "llm_semantic_review",
                        "resolution": "human_only",
                        "audio_evidence": "original_recording_interval",
                    },
                    "candidates": [
                        {
                            "id": "D001",
                            "interval": {
                                "basis": "source_audio",
                                "start_seconds": 10.0,
                                "end_seconds": 15.0,
                                "from": "00:00:10",
                                "to": "00:00:15",
                                "precision": "word",
                            },
                            "versions": {
                                "gigaam": "Мы будем делать.",
                                "whisper": "Мы не будем делать.",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _write_review(self, candidate_id="D001"):
        review = self.root / "meeting.review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "comparison_file": str(self.comparison),
                    "decisions": [
                        {
                            "candidate_id": candidate_id,
                            "verdict": "discrepancy",
                            "severity": "critical",
                            "categories": ["negation_or_polarity", "decision"],
                            "review_reason": "Отрицание меняет решение.",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return review

    def test_render_uses_authoritative_candidate_data(self):
        json_path, md_path = dual.render_review(self._write_review())

        self.assertEqual(json_path.name, "meeting.disagreements.json")
        self.assertEqual(md_path.name, "meeting.disagreements.md")
        markdown = md_path.read_text(encoding="utf-8")
        self.assertIn("`00:00:10–00:00:15`", markdown)
        self.assertIn("Мы будем делать.", markdown)
        self.assertIn("Мы не будем делать.", markdown)
        self.assertNotIn("ffplay", markdown)
        self.assertNotIn("аудиофрагмент", markdown.lower())

    def test_render_rejects_unknown_candidate(self):
        with self.assertRaisesRegex(ValueError, "неизвестный candidate_id"):
            dual.render_review(self._write_review("D999"))

    def test_render_rejects_llm_invented_final_text(self):
        review = self._write_review()
        payload = json.loads(review.read_text(encoding="utf-8"))
        payload["decisions"][0]["final_text"] = "Выдуманная версия"
        review.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "неожиданные поля"):
            dual.render_review(review)

    def test_render_rejects_unreviewed_candidate(self):
        review = self.root / "meeting.review.json"
        dual.write_review_template(self.comparison, {"candidates": [{}]}, review)

        with self.assertRaisesRegex(ValueError, "не покрывает всех кандидатов"):
            dual.render_review(review)

    def test_same_meaning_decision_is_reviewed_but_not_shown(self):
        review = self.root / "meeting.review.json"
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "comparison_file": str(self.comparison),
                    "decisions": [
                        {
                            "candidate_id": "D001",
                            "verdict": "same_meaning",
                            "review_reason": "Различается только безопасная формулировка.",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        json_path, _ = dual.render_review(review)
        rendered = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(rendered["reviewed_candidates"], 1)
        self.assertEqual(rendered["discrepancies"], [])


class InterpreterTests(unittest.TestCase):
    """Пакеты Whisper живут в venv — интерпретатор нельзя брать наугад."""

    def test_env_override_wins(self):
        with mock.patch.dict(dual.os.environ, {"WHISPER_PYTHON": "/opt/py/bin/python"}):
            self.assertEqual(dual._default_whisper_python(), "/opt/py/bin/python")

    def test_skill_venv_is_preferred_over_current_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            venv_python = dual._venv_python("whisper")
            venv_python = home / venv_python.relative_to(Path.home())
            venv_python.parent.mkdir(parents=True)
            venv_python.write_text("", encoding="utf-8")

            with mock.patch.dict(dual.os.environ, {}, clear=False):
                dual.os.environ.pop("WHISPER_PYTHON", None)
                with mock.patch.object(dual.Path, "home", return_value=home):
                    resolved = dual._default_whisper_python()

            self.assertEqual(resolved, str(venv_python))

    def test_current_interpreter_is_the_last_resort(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(dual.os.environ, {}, clear=False):
                dual.os.environ.pop("WHISPER_PYTHON", None)
                with mock.patch.object(dual.Path, "home", return_value=Path(temporary)):
                    resolved = dual._default_whisper_python()

            self.assertEqual(resolved, sys.executable)

    def test_wizard_and_runner_resolve_the_same_interpreter(self):
        # setup_check намеренно дублирует правило: он обязан работать в одиночку.
        with mock.patch.dict(dual.os.environ, {"WHISPER_PYTHON": "/opt/py/bin/python"}):
            self.assertEqual(
                Path(dual._default_whisper_python()),
                setup_check.resolve_whisper_python(),
            )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(dual.os.environ, {}, clear=False):
                dual.os.environ.pop("WHISPER_PYTHON", None)
                with mock.patch.object(dual.Path, "home", return_value=Path(temporary)):
                    self.assertEqual(
                        Path(dual._default_whisper_python()),
                        setup_check.resolve_whisper_python(),
                    )


class RunnerTests(unittest.TestCase):
    def test_commands_use_separate_suffixes_and_word_timestamps(self):
        args = argparse.Namespace(
            input=Path("/tmp/file with spaces.wav"),
            gigaam_python=sys.executable,
            whisper_python=sys.executable,
            gigaam_device="cpu",
            gigaam_model="v3_e2e_rnnt",
            gigaam_batch_size=4,
            whisper_backend="whisper-cpp",
            initial_prompt=None,
            no_condition_on_previous_text=False,
            diarize=False,
            diarizer="auto",
            num_speakers=None,
            max_speakers=6,
        )
        commands = dual.build_asr_commands(args, Path("/tmp/stage with spaces"))

        self.assertIn(".gigaam", commands["gigaam"])
        self.assertIn(".whisper", commands["whisper"])
        self.assertIn("--word-timestamps", commands["whisper"])
        self.assertNotIn("--preset", commands["whisper"])
        self.assertNotIn("--model", commands["whisper"])
        self.assertNotIn("--beam-size", commands["whisper"])
        self.assertIn(str(Path("/tmp/file with spaces.wav").resolve()), commands["whisper"])

    def test_python_symlink_is_not_resolved_out_of_virtualenv(self):
        with tempfile.TemporaryDirectory() as temporary:
            python_link = Path(temporary) / "venv" / "bin" / "python"
            python_link.parent.mkdir(parents=True)
            python_link.symlink_to(sys.executable)
            args = argparse.Namespace(
                input=Path("/tmp/meeting.wav"),
                gigaam_python=str(python_link),
                whisper_python=sys.executable,
                gigaam_device="cpu",
                gigaam_model="v3_e2e_rnnt",
                gigaam_batch_size=1,
                whisper_backend=None,
                initial_prompt=None,
                no_condition_on_previous_text=False,
                diarize=False,
                diarizer="auto",
                num_speakers=None,
                max_speakers=6,
            )

            commands = dual.build_asr_commands(args, Path(temporary) / "stage")

        self.assertEqual(commands["gigaam"][0], str(python_link.absolute()))

    def test_gigaam_device_auto_uses_venv_torch_probe(self):
        completed = dual.subprocess.CompletedProcess(
            ["python", "-c", "probe"], 0, stdout="mps\n", stderr=""
        )
        with mock.patch.object(dual.subprocess, "run", return_value=completed) as run:
            device = dual._resolve_gigaam_device("/venv/bin/python", "auto")

        self.assertEqual(device, "mps")
        self.assertEqual(run.call_args.args[0][0], "/venv/bin/python")

    def test_processes_really_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            statuses = dual.run_parallel(
                {
                    "gigaam": [sys.executable, "-c", "import time; time.sleep(0.6)"],
                    "whisper": [sys.executable, "-c", "import time; time.sleep(0.6)"],
                },
                Path(temporary),
            )
            elapsed = time.monotonic() - started

        self.assertEqual(statuses, {"gigaam": 0, "whisper": 0})
        self.assertLess(elapsed, 1.05)

    def test_failure_stops_sibling_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            started = time.monotonic()
            statuses = dual.run_parallel(
                {
                    "gigaam": [sys.executable, "-c", "raise SystemExit(7)"],
                    "whisper": [sys.executable, "-c", "import time; time.sleep(3)"],
                },
                Path(temporary),
            )
            elapsed = time.monotonic() - started

        self.assertEqual(statuses["gigaam"], 7)
        self.assertNotEqual(statuses["whisper"], 0)
        self.assertLess(elapsed, 2.0)

    def test_validation_rejects_other_source_and_out_of_bounds_time(self):
        wrong_source = transcript([(1.0, 2.0, "текст")], source="other.wav")
        wrong_source["_artifact_path"] = "/tmp/wrong.json"
        with self.assertRaisesRegex(ValueError, "source_file"):
            dual.validate_transcript(wrong_source, "meeting.wav", 10.0)

        late = transcript([(1.0, 20.0, "текст")])
        late["_artifact_path"] = "/tmp/late.json"
        with self.assertRaisesRegex(ValueError, "за пределами"):
            dual.validate_transcript(late, "meeting.wav", 10.0)

        coarse = {
            "metadata": {"source_file": "meeting.wav"},
            "segments": [{"start": 1.0, "end": 2.0, "text": "важный текст"}],
        }
        with self.assertRaisesRegex(ValueError, "без word timestamps"):
            dual.validate_transcript(
                coarse,
                "meeting.wav",
                10.0,
                require_word_timestamps=True,
            )

    def test_compare_requires_real_source_and_finite_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "transcript.json"
            artifact.write_text(json.dumps({"segments": []}), encoding="utf-8")
            missing = root / "missing.wav"
            with self.assertRaisesRegex(ValueError, "не найдена"):
                dual.write_comparison(
                    artifact,
                    artifact,
                    missing,
                    10.0,
                    root / "comparison.json",
                )

            source = root / "meeting.wav"
            source.touch()
            with self.assertRaisesRegex(ValueError, "конечным"):
                dual.write_comparison(
                    artifact,
                    artifact,
                    source,
                    float("nan"),
                    root / "comparison.json",
                )

    def test_publish_is_all_or_nothing_when_artifact_missing(self):
        with (
            tempfile.TemporaryDirectory() as staging_raw,
            tempfile.TemporaryDirectory() as output_raw,
        ):
            staging = Path(staging_raw)
            output = Path(output_raw)
            stem = "meeting"
            required = [
                f"{stem}.gigaam.transcript.json",
                f"{stem}.gigaam.transcript.md",
                f"{stem}.gigaam.transcript.srt",
                f"{stem}.whisper.transcript.json",
                f"{stem}.whisper.transcript.md",
                # whisper SRT intentionally absent
            ]
            for name in required:
                content = json.dumps({"segments": []}) if name.endswith(".json") else "ok"
                (staging / name).write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "обязательный артефакт"):
                dual._atomic_publish(staging, output, stem)

            self.assertEqual(list(output.iterdir()), [])

    def test_publish_validates_source_before_moving_any_artifact(self):
        with (
            tempfile.TemporaryDirectory() as staging_raw,
            tempfile.TemporaryDirectory() as output_raw,
        ):
            staging = Path(staging_raw)
            output = Path(output_raw)
            data = transcript([(0.0, 1.0, "текст")], source="other.wav")
            for engine in ("gigaam", "whisper"):
                prefix = f"meeting.{engine}.transcript"
                (staging / f"{prefix}.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8"
                )
                (staging / f"{prefix}.md").write_text("text", encoding="utf-8")
                (staging / f"{prefix}.srt").write_text("text", encoding="utf-8")
            (staging / "meeting.comparison.json").write_text("{}", encoding="utf-8")
            (staging / "meeting.review-template.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source_file"):
                dual._atomic_publish(
                    staging,
                    output,
                    "meeting",
                    source_name="meeting.wav",
                    duration=10.0,
                )

            self.assertEqual(list(output.iterdir()), [])

    def test_publish_rolls_back_if_replacement_fails_midway(self):
        with (
            tempfile.TemporaryDirectory() as staging_raw,
            tempfile.TemporaryDirectory() as output_raw,
        ):
            staging = Path(staging_raw)
            output = Path(output_raw)
            names = [
                "meeting.gigaam.transcript.json",
                "meeting.gigaam.transcript.md",
                "meeting.gigaam.transcript.srt",
                "meeting.whisper.transcript.json",
                "meeting.whisper.transcript.md",
                "meeting.whisper.transcript.srt",
                "meeting.comparison.json",
                "meeting.review-template.json",
            ]
            valid_transcript = json.dumps(
                transcript([(0.0, 1.0, "текст")]), ensure_ascii=False
            )
            for name in names:
                if name.endswith("transcript.json"):
                    content = valid_transcript
                elif name.endswith(".json"):
                    content = "{}"
                else:
                    content = "new"
                (staging / name).write_text(content, encoding="utf-8")
                (output / name).write_text("old-" + name, encoding="utf-8")

            real_replace = dual.os.replace
            calls = 0
            failed = False

            def flaky_replace(source, destination):
                nonlocal calls, failed
                calls += 1
                if calls == 5 and not failed:
                    failed = True
                    raise OSError("injected publish failure")
                return real_replace(source, destination)

            with mock.patch.object(dual.os, "replace", side_effect=flaky_replace):
                with self.assertRaisesRegex(OSError, "injected"):
                    dual._atomic_publish(
                        staging,
                        output,
                        "meeting",
                        source_name="meeting.wav",
                        duration=10.0,
                    )

            for name in names:
                self.assertEqual(
                    (output / name).read_text(encoding="utf-8"), "old-" + name
                )


if __name__ == "__main__":
    unittest.main()
