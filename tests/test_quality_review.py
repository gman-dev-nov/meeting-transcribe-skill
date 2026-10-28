"""Quality workflow tests use tiny real comparison artifacts, never ASR models."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dual_transcribe as dual
import quality_review as quality


class QualityReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.counter = 0

    def write(self, path, data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def fixture(self, segments=None, whisper=None, prepare=True):
        self.counter += 1
        directory = self.root / str(self.counter)
        directory.mkdir()
        segments = segments or [(0, 5, "Срок в пятницу, нет, в четверг.")]
        whisper = segments if whisper is None else whisper
        raw = {}
        for engine, source in (("gigaam", segments), ("whisper", whisper)):
            path = directory / f"meeting.{engine}.transcript.json"
            self.write(path, {"metadata": {"source_file": "meeting.wav", "model": "test"},
                              "segments": [{"start": start, "end": end, "text": text}
                                           for start, end, text in source]})
            raw[engine] = path
        audio = directory / "meeting.wav"
        audio.write_bytes(b"unused fake audio; no inference")
        comparison = dual.build_comparison(
            dual.load_transcript(raw["gigaam"]), dual.load_transcript(raw["whisper"]),
            audio, max(end for _, end, _ in segments + whisper) + 1)
        comparison_path = directory / "meeting.comparison.json"
        semantic_path = directory / "meeting.review.json"
        self.write(comparison_path, comparison)
        self.write(semantic_path, {"schema_version": 1, "comparison_file": str(comparison_path),
                                  "decisions": [{"candidate_id": item["id"],
                                                 "verdict": "discrepancy", "severity": "critical",
                                                 "categories": ["other_semantic"],
                                                 "review_reason": "Содержательное различие."}
                                                for item in comparison["candidates"]]})
        result = {"directory": directory, "raw": raw, "comparison": comparison_path,
                  "semantic": semantic_path}
        if prepare:
            result["manifest"], result["review"] = quality.prepare(comparison_path, semantic_path)
            review = self.read(result["review"])
            review["reviewed_blocks"] = [b["id"] for b in self.read(result["manifest"])["blocks"]]
            self.write(result["review"], review)
        return result

    def text(self, fixture):
        manifest = self.read(fixture["manifest"])
        return manifest["transcripts"][manifest["base_engine"]]["text"]

    def anchor(self, fixture, text):
        start = self.text(fixture).index(text)
        return {"start": start, "end": start + len(text), "from": text}

    def finding(self, fixture, text=None):
        text = self.text(fixture) if text is None else text
        item = {"id": "Q001", "kind": "self_correction", "span": self.anchor(fixture, text),
                "severity": "critical", "categories": ["date_or_deadline"],
                "review_reason": "Говорящий поправил срок."}
        review = self.read(fixture["review"])
        review["findings"] = [item]
        self.write(fixture["review"], review)
        return item

    def edit(self, fixture, before, after, kind="punctuation", identity="E001"):
        item = {"id": identity, "kind": kind, "span": self.anchor(fixture, before),
                "to": after, "review_reason": "Редакторское оформление."}
        review = self.read(fixture["review"])
        review["edits"].append(item)
        self.write(fixture["review"], review)
        return item

    def human(self, fixture, before, after, identity="Q001"):
        path = fixture["directory"] / "meeting.human-resolutions.json"
        self.write(path, {"schema_version": 1, "manifest_file": str(fixture["manifest"]),
                          "manifest_sha256": quality.digest(fixture["manifest"]),
                          "resolutions": [{"candidate_id": identity, "confirmed_by": "user",
                                           "span": self.anchor(fixture, before),
                                           "confirmed_text": after}]})
        return path

    def test_prepare_covers_both_full_equal_transcripts_without_render_side_effects(self):
        fixture = self.fixture([(0, 5, "Первый блок."), (65, 70, "Второй блок.")])
        manifest = self.read(fixture["manifest"])
        self.assertEqual([], manifest["discrepancies"])
        self.assertEqual(["B001", "B002"], [b["id"] for b in manifest["blocks"]])
        for engine in ("gigaam", "whisper"):
            indices = [i for b in manifest["blocks"] for i in b["segment_indices"][engine]]
            self.assertEqual([0, 1], indices)
            self.assertEqual("Первый блок.\nВторой блок.", manifest["transcripts"][engine]["text"])
        self.assertFalse(list(fixture["directory"].glob("*.disagreements.*")))

    def test_equal_asr_self_correction_is_masked_and_overlapping_style_edit_skipped(self):
        fixture = self.fixture()
        original = self.text(fixture)
        self.finding(fixture)
        self.edit(fixture, original, original + " ", "spacing")
        paths = quality.render(fixture["review"])
        self.assertIn("[НЕ ПОДТВЕРЖДЕНО: Q001]", paths["clean_md"].read_text())
        self.assertNotIn(original, paths["clean_md"].read_text())
        changes = self.read(paths["changes_json"])
        self.assertEqual([], changes["applied"])
        self.assertEqual(["E001"], [p["id"] for p in changes["skipped_due_to_pending"]])

    def test_self_correction_across_segment_boundary_keeps_source_interval(self):
        fixture = self.fixture([(0, 5, "Срок в пятницу,"), (5, 9, "нет, в четверг.")])
        self.finding(fixture, "пятницу,\nнет, в четверг")
        paths = quality.render(fixture["review"])
        issue = self.read(paths["findings_json"])["findings"][0]
        self.assertEqual(0, issue["interval"]["start_seconds"])
        self.assertEqual(9, issue["interval"]["end_seconds"])
        self.assertIn("Q001", paths["clean_md"].read_text())

    def test_review_requires_exact_full_block_coverage_before_any_output(self):
        for coverage in (["B001"], ["B001", "B001", "B002"]):
            with self.subTest(coverage=coverage):
                fixture = self.fixture([(0, 5, "Один."), (65, 70, "Два.")])
                review = self.read(fixture["review"])
                review["reviewed_blocks"] = coverage
                self.write(fixture["review"], review)
                output = fixture["directory"] / "output"
                with self.assertRaises(ValueError):
                    quality.render(fixture["review"], output_dir=output)
                self.assertFalse(output.exists())

    def test_verified_no_issues_requires_no_extra_human_file_and_preserves_inputs(self):
        fixture = self.fixture([(0, 5, "Все согласовано.")])
        inputs = list(fixture["raw"].values()) + [fixture["comparison"], fixture["semantic"]]
        before = {p: p.read_bytes() for p in inputs}
        paths = quality.render(fixture["review"], verified=True)
        self.assertEqual("verified", self.read(paths["findings_json"])["status"])
        self.assertEqual([], self.read(paths["changes_json"])["pending_ids"])
        self.assertIn("Не источник дословных цитат", paths["clean_md"].read_text())
        self.assertEqual(before, {p: p.read_bytes() for p in inputs})

    def test_safe_punctuation_is_applied_with_exact_audit(self):
        fixture = self.fixture([(0, 5, "Привет коллеги.")])
        before = fixture["raw"]["gigaam"].read_bytes()
        self.edit(fixture, "Привет коллеги.", "Привет, коллеги.")
        paths = quality.render(fixture["review"], verified=True)
        self.assertIn("Привет, коллеги.", paths["clean_md"].read_text())
        patch = self.read(paths["changes_json"])["applied"][0]
        self.assertEqual("Привет коллеги.", patch["span"]["from"])
        self.assertEqual("Привет, коллеги.", patch["to"])
        self.assertEqual("editorial", patch["source"])
        self.assertEqual(before, fixture["raw"]["gigaam"].read_bytes())

    def test_editorial_fact_changes_are_rejected(self):
        for before, after, kind in [
            ("Мы не согласны.", "Мы согласны.", "punctuation"),
            ("Температура -5.", "Температура 5.", "punctuation"),
            ("Значение 1.5.", "Значение 15.", "punctuation"),
            ("Бюджет $100.", "Бюджет €100.", "punctuation"),
            ("Мощность 5 mW.", "Мощность 5 MW.", "case"),
            ("Мощность 5 мВт.", "Мощность 5 МВт.", "case"),
            ("Скорость 10 м/с.", "Скорость 10 м·с.", "punctuation"),
            ("Адрес github.com.", "Адрес github,com.", "punctuation"),
        ]:
            with self.subTest(before=before):
                fixture = self.fixture([(0, 5, before)])
                self.edit(fixture, before, after, kind)
                with self.assertRaises(ValueError):
                    quality.render(fixture["review"])
                self.assertFalse((fixture["directory"] / "meeting.clean.md").exists())

    def test_vocalization_removal_does_not_allow_negation_fillers_or_word_repetition(self):
        before = "ээ нет ну мы мы"
        for after, accepted in [("нет ну мы мы", True), ("ну мы мы", False),
                                ("нет мы мы", False), ("нет ну мы", False)]:
            with self.subTest(after=after):
                fixture = self.fixture([(0, 5, before)])
                self.edit(fixture, before, after, "vocalization")
                if accepted:
                    paths = quality.render(fixture["review"])
                    self.assertIn(after, paths["clean_md"].read_text())
                else:
                    with self.assertRaises(ValueError):
                        quality.render(fixture["review"])

    def test_exact_human_resolution_produces_verified_derived_text(self):
        fixture = self.fixture([(0, 5, "Срок в пятницу, нет, в четверг. Дальше.")])
        phrase = "в пятницу, нет, в четверг"
        self.finding(fixture, phrase)
        human = self.human(fixture, phrase, "в четверг")
        raw_before = {p: p.read_bytes() for p in fixture["raw"].values()}
        paths = quality.render(fixture["review"], human, verified=True)
        self.assertIn("Срок в четверг. Дальше.", paths["clean_md"].read_text())
        self.assertEqual([], self.read(paths["findings_json"])["pending_ids"])
        applied = self.read(paths["changes_json"])["applied"]
        self.assertEqual("user", applied[0]["source"])
        self.assertEqual(["Q001"], applied[0]["candidate_ids"])
        self.assertEqual(raw_before, {p: p.read_bytes() for p in fixture["raw"].values()})

    def test_pending_issue_blocks_verified_export_before_writes(self):
        fixture = self.fixture()
        self.finding(fixture)
        with self.assertRaises(ValueError):
            quality.render(fixture["review"], verified=True)
        self.assertFalse((fixture["directory"] / "meeting.clean.md").exists())

    def test_bad_span_and_overlapping_patches_fail_before_writes(self):
        fixture = self.fixture([(0, 5, "Привет коллеги.")])
        self.edit(fixture, "Привет коллеги.", "Привет, коллеги.")
        review = self.read(fixture["review"])
        review["edits"][0]["span"]["start"] += 1
        self.write(fixture["review"], review)
        with self.assertRaises(ValueError):
            quality.render(fixture["review"])
        fixture = self.fixture([(0, 5, "Привет коллеги.")])
        self.edit(fixture, "Привет коллеги.", "Привет, коллеги.")
        self.edit(fixture, "Привет", "привет", "case", "E002")
        with self.assertRaises(ValueError):
            quality.render(fixture["review"])
        self.assertFalse((fixture["directory"] / "meeting.clean.md").exists())

    def test_stale_raw_manifest_and_human_hash_are_rejected(self):
        for target in ("raw", "manifest", "human"):
            with self.subTest(target=target):
                fixture = self.fixture()
                human = None
                if target == "raw":
                    fixture["raw"]["gigaam"].write_text("{}", encoding="utf-8")
                elif target == "manifest":
                    fixture["manifest"].write_text(fixture["manifest"].read_text() + " ")
                else:
                    self.finding(fixture)
                    human = self.human(fixture, self.text(fixture), "В четверг.")
                    data = self.read(human)
                    data["manifest_sha256"] = "0" * 64
                    self.write(human, data)
                with self.assertRaises(ValueError):
                    quality.render(fixture["review"], human)

    def test_raw_changed_before_prepare_cannot_be_blessed_by_new_manifest(self):
        fixture = self.fixture(prepare=False)
        data = self.read(fixture["raw"]["gigaam"])
        data["segments"][0]["text"] = "Иная формулировка."
        self.write(fixture["raw"]["gigaam"], data)
        with self.assertRaises(ValueError):
            quality.prepare(fixture["comparison"], fixture["semantic"])
        self.assertFalse((fixture["directory"] / "meeting.quality-manifest.json").exists())

    def test_human_d_patch_cannot_target_adjacent_segment_at_time_boundary(self):
        fixture = self.fixture([(0, 1, "Срок 5"), (4, 5, "Соседний текст")],
                               [(0, 1, "Срок 6"), (4, 5, "Соседний текст")])
        manifest = self.read(fixture["manifest"])
        self.assertEqual(1, len(manifest["discrepancies"]))
        identity = manifest["discrepancies"][0]["id"]
        self.assertEqual(4, manifest["discrepancies"][0]["interval"]["end_seconds"])
        human = self.human(fixture, "Соседний текст", "Совсем иной текст", identity)
        with self.assertRaises(ValueError):
            quality.render(fixture["review"], human, verified=True)
        human = self.human(fixture, "5", "6", identity)
        paths = quality.render(fixture["review"], human, verified=True)
        self.assertIn("Срок 6\nСоседний текст", paths["clean_md"].read_text())

    def test_directory_at_late_output_path_leaves_no_partial_outputs(self):
        fixture = self.fixture([(0, 5, "Все согласовано.")])
        output = fixture["directory"] / "exports"
        output.mkdir()
        blocked = output / "meeting.clean.md"
        blocked.mkdir()
        with self.assertRaises(ValueError):
            quality.render(fixture["review"], output_dir=output)
        self.assertEqual([blocked], list(output.iterdir()))

    def test_late_replace_failure_restores_all_prior_versions(self):
        fixture = self.fixture([(0, 5, "Привет коллеги.")])
        output = fixture["directory"] / "exports"
        paths = quality.render(fixture["review"], output_dir=output)
        prior = {p: p.read_bytes() for p in paths.values()}
        self.edit(fixture, "Привет коллеги.", "Привет, коллеги.")
        original_replace = Path.replace
        failed = False

        def replace(source, destination):
            nonlocal failed
            if Path(destination) == paths["clean_md"] and not failed:
                failed = True
                raise OSError("injected late publication failure")
            return original_replace(source, destination)

        with mock.patch.object(Path, "replace", replace):
            with self.assertRaisesRegex(OSError, "injected"):
                quality.render(fixture["review"], output_dir=output)
        self.assertTrue(failed)
        self.assertEqual(prior, {p: p.read_bytes() for p in paths.values()})
        self.assertEqual(set(paths.values()), set(output.iterdir()))

    def test_rehashed_manifest_cannot_change_text_or_hide_discrepancies(self):
        for tamper in ("transcript", "discrepancies"):
            with self.subTest(tamper=tamper):
                fixture = self.fixture([(0, 5, "Срок 5")], [(0, 5, "Срок 6")])
                manifest = self.read(fixture["manifest"])
                if tamper == "transcript":
                    manifest["transcripts"]["gigaam"]["text"] = "Подменённый текст"
                else:
                    manifest["discrepancies"] = []
                self.write(fixture["manifest"], manifest)
                review = self.read(fixture["review"])
                review["manifest_sha256"] = quality.digest(fixture["manifest"])
                self.write(fixture["review"], review)
                with self.assertRaises(ValueError):
                    quality.render(fixture["review"], verified=True)


if __name__ == "__main__":
    unittest.main()
