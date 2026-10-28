"""Контракты CLI, чанков, публикации и native callbacks без установленного Type."""

import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "duo_transcribe.cjs"
NODE = shutil.which("node")
SAMPLE_RATE = 16_000  # адаптер нарезает аудио ровно с частотой 16 кГц
DEFAULT_APP = "/Applications/Type.app"
EXPORTS = ("parseArgs", "chunkRanges", "outputPaths", "writeOutputs")
CALL_TIMEOUT = 30  # зависший chunkRanges — тоже баг, но ждать дольше незачем

# JS-обвязка вызывает экспорт адаптера и печатает конверт с результатом,
# чтобы Python отличал «функция бросила исключение» от «вернула значение».
# Маркер отделяет конверт от возможного шума адаптера в stdout.
HARNESS = """\
const adapter = require(process.env.DUO_MODULE);
const fix = (item) => {
  if (item === "__NAN__") return NaN;
  if (item === "__INF__") return Infinity;
  if (item === "-__INF__") return -Infinity;
  if (Array.isArray(item)) return item.map(fix);
  if (item && typeof item === "object") {
    return Object.fromEntries(Object.entries(item).map(([k, v]) => [k, fix(v)]));
  }
  return item;
};
const args = JSON.parse(process.argv[3]).map(fix);
try {
  const fn = adapter[process.argv[2]];
  if (typeof fn !== "function") throw new TypeError("нет экспорта");
  const value = fn(...args);
  process.stdout.write("===DUO===" + JSON.stringify({
    ok: true,
    value: value === undefined ? null : value,
  }));
} catch (error) {
  process.stdout.write("===DUO===" + JSON.stringify({
    ok: false,
    name: (error && error.name) || "Error",
    message: String((error && error.message) || error),
  }));
}
"""


def _to_js(value):
    """NaN и Infinity не живут в JSON — передаём их сентинелами, обвязка чинит."""
    if isinstance(value, float):
        if math.isnan(value):
            return "__NAN__"
        if math.isinf(value):
            return "-__INF__" if value < 0 else "__INF__"
    if isinstance(value, list):
        return [_to_js(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_js(item) for key, item in value.items()}
    return value


def _leaf_values(value):
    """Все скаляры вложенной структуры — проверяем значения, не угадывая имена полей."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaf_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaf_values(item)
    else:
        yield value


class DuoAdapterTests(unittest.TestCase):
    """Общее: пропуск без node/адаптера, вызов экспортов и CLI через subprocess."""

    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise unittest.SkipTest("node не найден в PATH")
        if not SCRIPT.exists():
            raise AssertionError(f"нет адаптера: {SCRIPT}")
        cls.adapter_temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.adapter_temporary.cleanup)
        harness = Path(cls.adapter_temporary.name) / "harness.cjs"
        harness.write_text(HARNESS, encoding="utf-8")
        cls.harness = harness

    def call_export(self, name, *args):
        argv = [NODE, str(self.harness), name, json.dumps(_to_js(list(args)))]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT,
                env={**os.environ, "DUO_MODULE": str(SCRIPT)},
            )
        except subprocess.TimeoutExpired:
            self.fail(f"экспорт {name}{args!r} завис (таймаут {CALL_TIMEOUT} с)")
        try:
            envelope = json.loads(proc.stdout.split("===DUO===")[-1])
        except json.JSONDecodeError:
            self.fail(
                f"экспорт {name} ответил не-JSON: "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        self.assertIsInstance(envelope, dict)
        return envelope

    def call_ok(self, name, *args):
        envelope = self.call_export(name, *args)
        self.assertTrue(envelope.get("ok"), f"ожидали успех {name}{args!r}: {envelope}")
        return envelope["value"]

    def call_throws(self, name, *args):
        envelope = self.call_export(name, *args)
        self.assertFalse(envelope.get("ok"), f"ожидали отказ {name}{args!r}: {envelope}")

    def run_cli(self, *args):
        try:
            return subprocess.run(
                [NODE, str(SCRIPT), *args],
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            self.fail(f"CLI завис на аргументах {args!r}")


class ModuleContractTests(DuoAdapterTests):
    """Модуль можно require-ить без нативного рантайма Type."""

    def test_require_exports_four_functions(self):
        js = (
            "const adapter = require(process.env.DUO_MODULE);\n"
            f"for (const key of {json.dumps(list(EXPORTS))}) {{\n"
            "  if (typeof adapter[key] !== 'function') "
            "throw new Error('нет экспорта ' + key);\n"
            "}\n"
            "console.log('exports-ok');\n"
        )
        try:
            proc = subprocess.run(
                [NODE, "-e", js],
                capture_output=True,
                text=True,
                timeout=CALL_TIMEOUT,
                env={**os.environ, "DUO_MODULE": str(SCRIPT)},
            )
        except subprocess.TimeoutExpired:
            self.fail("require адаптера завис")
        self.assertEqual(
            proc.returncode, 0,
            f"require упал: stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )
        self.assertIn("exports-ok", proc.stdout)


class CliTests(DuoAdapterTests):
    """CLI без нативного рантайма: справка и отказ на плохих аргументах."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input = self.root / "запись с пробелами.m4a"
        self.input.write_bytes(b"")
        self.output_dir = self.root / "out dir"
        self.output_dir.mkdir()

    def test_help_documents_usage(self):
        proc = self.run_cli("--help")
        text = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, text)
        self.assertIn("--output-dir", text)
        self.assertIn("--chunk-seconds", text)

    def test_no_arguments_is_rejected(self):
        proc = self.run_cli()
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue((proc.stdout + proc.stderr).strip())

    def test_unknown_flag_is_rejected_with_its_name(self):
        proc = self.run_cli(
            str(self.input), "--output-dir", str(self.output_dir),
            "--no-such-flag", "x",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--no-such-flag", proc.stdout + proc.stderr)

    def test_relative_input_is_rejected(self):
        proc = self.run_cli("relative-input.wav", "--output-dir", str(self.output_dir))
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue((proc.stdout + proc.stderr).strip())

    def test_relative_output_dir_is_rejected(self):
        proc = self.run_cli(str(self.input), "--output-dir", "relative-out")
        self.assertNotEqual(proc.returncode, 0)
        self.assertTrue((proc.stdout + proc.stderr).strip())

    def test_flag_without_value_is_rejected(self):
        proc = self.run_cli(
            str(self.input), "--output-dir", str(self.output_dir), "--chunk-seconds",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--chunk-seconds", proc.stdout + proc.stderr)

    def test_bad_chunk_seconds_is_rejected(self):
        for bad in ("0", "0.5", "25.5", "-3", "NaN", "Infinity", "abc"):
            with self.subTest(bad=bad):
                proc = self.run_cli(
                    str(self.input), "--output-dir", str(self.output_dir),
                    "--chunk-seconds", bad,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertTrue((proc.stdout + proc.stderr).strip())


class ParseArgsTests(DuoAdapterTests):
    """parseArgs: контракты аргументов; имена полей результата не угадываем."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input = self.root / "meeting with spaces.wav"
        self.input.write_bytes(b"")
        self.input_path = str(self.input)
        self.output_dir = self.root / "results dir"
        self.output_dir.mkdir()

    def parsed(self, *argv):
        return self.call_ok("parseArgs", list(argv))

    def rejected(self, *argv):
        self.call_throws("parseArgs", list(argv))

    def _strings(self, value):
        return [item for item in _leaf_values(value) if isinstance(item, str)]

    def _numbers(self, value):
        return [
            item for item in _leaf_values(value)
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ]

    def test_minimal_arguments_use_defaults(self):
        value = self.parsed(self.input_path, "--output-dir", str(self.output_dir))
        strings = self._strings(value)
        self.assertIn(self.input_path, strings)
        self.assertIn(str(self.output_dir), strings)
        self.assertIn(DEFAULT_APP, strings)
        self.assertIn(20, self._numbers(value))  # --chunk-seconds по умолчанию

    def test_absolute_app_with_spaces_is_accepted(self):
        app = self.root / "My Type.app"
        app.mkdir()
        value = self.parsed(
            self.input_path, "--output-dir", str(self.output_dir), "--app", str(app),
        )
        self.assertIn(str(app), self._strings(value))

    def test_fractional_chunk_seconds_is_accepted(self):
        value = self.parsed(
            self.input_path, "--output-dir", str(self.output_dir),
            "--chunk-seconds", "12.5",
        )
        numbers = [float(item) for item in self._numbers(value)]
        self.assertIn(12.5, numbers)

    def test_chunk_seconds_bounds(self):
        for good in ("1", "12.5", "20", "25"):
            with self.subTest(good=good):
                self.parsed(
                    self.input_path, "--output-dir", str(self.output_dir),
                    "--chunk-seconds", good,
                )
        for bad in ("0", "0.99", "-1", "25.01", "26", "NaN", "Infinity",
                    "-Infinity", "abc", ""):
            with self.subTest(bad=bad):
                self.rejected(
                    self.input_path, "--output-dir", str(self.output_dir),
                    "--chunk-seconds", bad,
                )

    def test_relative_paths_are_rejected(self):
        self.rejected("relative-input.wav", "--output-dir", str(self.output_dir))
        self.rejected(self.input_path, "--output-dir", "relative-out")
        self.rejected(
            self.input_path, "--output-dir", str(self.output_dir), "--app", "Type.app",
        )

    def test_missing_input_is_rejected(self):
        self.rejected("--output-dir", str(self.output_dir))

    def test_flag_without_value_is_rejected(self):
        self.rejected(self.input_path, "--output-dir")
        self.rejected(self.input_path, "--output-dir", str(self.output_dir), "--app")
        self.rejected(
            self.input_path, "--output-dir", str(self.output_dir), "--chunk-seconds",
        )

    def test_duplicate_flags_are_rejected(self):
        other = self.root / "другой каталог"
        other.mkdir()
        self.rejected(
            self.input_path, "--output-dir", str(self.output_dir),
            "--output-dir", str(other),
        )
        self.rejected(
            self.input_path, "--output-dir", str(self.output_dir),
            "--chunk-seconds", "10", "--chunk-seconds", "20",
        )

    def test_unknown_flag_and_extra_positional_are_rejected(self):
        self.rejected(
            self.input_path, "--output-dir", str(self.output_dir), "--definitely-bogus",
        )
        twin = self.root / "twin.wav"
        twin.write_bytes(b"")
        self.rejected(self.input_path, str(twin), "--output-dir", str(self.output_dir))


class ChunkRangesTests(DuoAdapterTests):
    """chunkRanges: 16 кГц, без наложений и дыр, короткий последний кусок."""

    def ranges(self, total_samples, chunk_seconds):
        return self.call_ok("chunkRanges", total_samples, chunk_seconds)

    def assert_chunks(self, chunks, total_samples, chunk_seconds):
        full = chunk_seconds * SAMPLE_RATE
        self.assertEqual(full, int(full))  # в тестах только точные длины куска
        full = int(full)
        self.assertEqual(len(chunks), math.ceil(total_samples / full))
        previous_end = 0
        for index, chunk in enumerate(chunks):
            for key in ("startSample", "endSample", "start", "end"):
                self.assertIn(key, chunk)
            self.assertEqual(chunk["startSample"], previous_end)  # без наложений и дыр
            self.assertGreater(chunk["endSample"], chunk["startSample"])  # без пустых
            length = chunk["endSample"] - chunk["startSample"]
            if index < len(chunks) - 1:
                self.assertEqual(length, full)  # все, кроме последнего, — полные
            else:
                self.assertLessEqual(length, full)  # последний — тот же или короче
            self.assertAlmostEqual(
                chunk["start"], chunk["startSample"] / SAMPLE_RATE, places=6,
            )
            self.assertAlmostEqual(
                chunk["end"], chunk["endSample"] / SAMPLE_RATE, places=6,
            )
            previous_end = chunk["endSample"]
        self.assertEqual(chunks[0]["startSample"], 0)
        self.assertEqual(previous_end, total_samples)  # покрытие всего аудио

    def test_exact_boundary_has_no_empty_tail(self):
        chunks = self.ranges(SAMPLE_RATE * 20, 20)
        self.assert_chunks(chunks, SAMPLE_RATE * 20, 20)
        self.assertAlmostEqual(chunks[-1]["end"], 20.0, places=6)

    def test_exact_two_boundaries(self):
        self.assert_chunks(
            self.ranges(SAMPLE_RATE * 40, 20), SAMPLE_RATE * 40, 20,
        )

    def test_last_chunk_is_shorter(self):
        total = SAMPLE_RATE * 50
        chunks = self.ranges(total, 20)
        self.assert_chunks(chunks, total, 20)
        self.assertEqual(len(chunks), 3)
        last = chunks[-1]
        self.assertEqual(last["startSample"], SAMPLE_RATE * 40)
        self.assertEqual(last["endSample"], total)
        self.assertAlmostEqual(last["start"], 40.0, places=6)
        self.assertAlmostEqual(last["end"], 50.0, places=6)

    def test_one_sample_past_boundary(self):
        total = SAMPLE_RATE * 20 + 1
        chunks = self.ranges(total, 20)
        self.assert_chunks(chunks, total, 20)
        self.assertEqual(chunks[-1]["endSample"] - chunks[-1]["startSample"], 1)

    def test_fractional_chunk_seconds(self):
        total = SAMPLE_RATE * 6 + 7000
        chunks = self.ranges(total, 2.5)
        self.assert_chunks(chunks, total, 2.5)
        self.assertAlmostEqual(chunks[-1]["end"], total / SAMPLE_RATE, places=6)

    def test_zero_samples_produce_no_chunks(self):
        self.assertEqual(self.ranges(0, 20), [])

    def test_total_samples_must_be_safe_integer(self):
        for bad in (-SAMPLE_RATE, 16000.5, 2 ** 53, "8000", float("nan"),
                    float("inf")):
            with self.subTest(bad=bad):
                self.call_throws("chunkRanges", bad, 20)

    def test_missing_total_samples_is_rejected(self):
        self.call_throws("chunkRanges")

    def test_chunk_seconds_must_be_positive(self):
        for bad in (0, -5):
            with self.subTest(bad=bad):
                self.call_throws("chunkRanges", SAMPLE_RATE, bad)


class OutputPathsTests(DuoAdapterTests):
    """outputPaths: полный basename с расширением + суффикс .duo.transcript.*."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source dir"
        self.source.mkdir()
        self.output_dir = self.root / "out dir"
        self.output_dir.mkdir()

    def paths_for(self, name, output_dir):
        input_path = self.source / name
        input_path.write_bytes(b"")
        value = self.call_ok("outputPaths", str(input_path), str(output_dir))
        self.assertIn("json", value)
        self.assertIn("markdown", value)
        return value, input_path

    def test_names_keep_full_basename_with_extension(self):
        value, _ = self.paths_for("запись с пробелами.m4a", self.output_dir)
        json_path = Path(value["json"])
        markdown_path = Path(value["markdown"])
        self.assertEqual(json_path.name, "запись с пробелами.m4a.duo.transcript.json")
        self.assertEqual(markdown_path.name, "запись с пробелами.m4a.duo.transcript.md")
        self.assertEqual(json_path.parent, self.output_dir)
        self.assertEqual(markdown_path.parent, self.output_dir)
        self.assertNotEqual(json_path, markdown_path)

    def test_extensions_protect_colliding_stems(self):
        wav, _ = self.paths_for("a.wav", self.output_dir)
        md, _ = self.paths_for("a.md", self.output_dir)
        self.assertNotEqual(Path(wav["json"]), Path(md["json"]))
        self.assertNotEqual(Path(wav["markdown"]), Path(md["markdown"]))
        self.assertEqual(Path(wav["json"]).name, "a.wav.duo.transcript.json")
        self.assertEqual(Path(md["json"]).name, "a.md.duo.transcript.json")

    def test_output_next_to_input_is_allowed_for_distinct_names(self):
        value, input_path = self.paths_for("meeting.wav", self.source)
        json_path = Path(value["json"])
        markdown_path = Path(value["markdown"])
        self.assertEqual(json_path.parent, self.source)
        self.assertNotEqual(json_path, input_path)
        self.assertNotEqual(markdown_path, input_path)
        self.assertEqual(json_path.name, "meeting.wav.duo.transcript.json")

    def test_basename_without_extension_still_gets_suffix(self):
        value, _ = self.paths_for("recording", self.output_dir)
        self.assertEqual(Path(value["json"]).name, "recording.duo.transcript.json")
        self.assertEqual(
            Path(value["markdown"]).name, "recording.duo.transcript.md",
        )


class WriteOutputsTests(DuoAdapterTests):
    """writeOutputs: пара JSON/MD, без перезаписи и без частичного вывода.

    Отказ тестируем занятым путём (существующий файл, каталог на месте файла),
    а не monkeypatch fs.linkSync: так тест не завязан на способ публикации.
    Любая реализация — с ранней проверкой или с откатом по месту — обязана
    оставить каталог без новых файлов и не тронуть чужое содержимое.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input = self.root / "совещание отдела.wav"
        self.input.write_bytes(b"")
        self.document = {
            "metadata": {
                "source_file": self.input.name,
                "duration_seconds": 50.0,
                "model": "duo-test",
            },
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Привет, коллеги"},
                {"start": 2.5, "end": 5.75, "text": "второй сегмент расшифровки"},
            ],
        }

    def outputs(self, output_dir):
        return self.call_ok("outputPaths", str(self.input), str(output_dir))

    def test_writes_complete_pair_into_fresh_directory(self):
        output_dir = self.root / "brand new dir"  # каталога ещё нет
        paths = self.outputs(output_dir)
        self.call_ok("writeOutputs", paths, self.document)

        json_path = Path(paths["json"])
        markdown_path = Path(paths["markdown"])
        self.assertTrue(json_path.is_file())
        self.assertTrue(markdown_path.is_file())

        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertIn("metadata", parsed)
        self.assertIn("segments", parsed)
        self.assertEqual(len(parsed["segments"]), len(self.document["segments"]))
        for got, want in zip(parsed["segments"], self.document["segments"]):
            for key, value in want.items():
                self.assertEqual(got[key], value, f"сегмент искажён в поле {key}")
        for key, value in self.document["metadata"].items():
            self.assertEqual(parsed["metadata"][key], value)

        markdown = markdown_path.read_text(encoding="utf-8")
        for segment in self.document["segments"]:
            self.assertIn(segment["text"], markdown)  # весь текст — в markdown
        self.assertTrue(all("words" not in segment for segment in parsed["segments"]))

    def test_existing_json_is_never_overwritten(self):
        output_dir = self.root / "out"
        output_dir.mkdir()
        paths = self.outputs(output_dir)
        json_path = Path(paths["json"])
        json_path.write_text("OLD-JSON", encoding="utf-8")

        self.call_throws("writeOutputs", paths, self.document)

        self.assertEqual(json_path.read_text(encoding="utf-8"), "OLD-JSON")
        self.assertFalse(Path(paths["markdown"]).exists())

    def test_existing_markdown_is_never_overwritten(self):
        output_dir = self.root / "out"
        output_dir.mkdir()
        paths = self.outputs(output_dir)
        markdown_path = Path(paths["markdown"])
        markdown_path.write_text("OLD-MD", encoding="utf-8")

        self.call_throws("writeOutputs", paths, self.document)

        self.assertEqual(markdown_path.read_text(encoding="utf-8"), "OLD-MD")
        self.assertFalse(Path(paths["json"]).exists())

    def test_occupied_markdown_path_publishes_neither_file(self):
        output_dir = self.root / "out"
        output_dir.mkdir()
        paths = self.outputs(output_dir)
        markdown_path = Path(paths["markdown"])
        markdown_path.mkdir()  # каталог занимает место файла

        self.call_throws("writeOutputs", paths, self.document)

        self.assertTrue(markdown_path.is_dir())
        self.assertFalse(Path(paths["json"]).exists())

    def test_occupied_json_path_publishes_neither_file(self):
        output_dir = self.root / "out"
        output_dir.mkdir()
        paths = self.outputs(output_dir)
        json_path = Path(paths["json"])
        json_path.mkdir()  # каталог занимает место файла

        self.call_throws("writeOutputs", paths, self.document)

        self.assertTrue(json_path.is_dir())
        self.assertFalse(Path(paths["markdown"]).exists())


class NativeContractTests(DuoAdapterTests):
    def run_js(self, source, *args):
        proc = subprocess.run(
            [NODE, "-e", source, str(SCRIPT), *args],
            capture_output=True, text=True, timeout=CALL_TIMEOUT,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_both_callbacks_required_and_callback_order_is_irrelevant(self):
        self.run_js("""
const assert = require('node:assert/strict');
const {createResultCallbacks} = require(process.argv[1]);
(async () => {
  const callbacks = createResultCallbacks();
  assert.equal(await callbacks.request(() => {
    callbacks.onText('final', {complete: true});
    callbacks.onAsrText('raw');
  }, 'reverse callbacks', 100), 'raw');
  await assert.rejects(callbacks.request(() => {
    callbacks.onText('final', {complete: true});
  }, 'missing raw', 10), /timed out/);
  assert.equal(await callbacks.request(() => {
    callbacks.onAsrText(''); callbacks.onText('', {});
  }, 'silence', 100), '');
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_native_error_incomplete_and_invalid_text_are_rejected(self):
        self.run_js("""
const assert = require('node:assert/strict');
const {createResultCallbacks} = require(process.argv[1]);
(async () => {
  for (const answer of [{error: 'mock failure'}, {complete: false}]) {
    const callbacks = createResultCallbacks();
    await assert.rejects(callbacks.request(() => {
      callbacks.onAsrText('raw'); callbacks.onText('partial', answer);
    }, 'failed result', 100));
  }
  const callbacks = createResultCallbacks();
  await assert.rejects(callbacks.request(() => {
    callbacks.onAsrText(null); callbacks.onText('final', {});
  }, 'invalid result', 100), /Invalid onAsrText/);
})().catch(error => { console.error(error); process.exitCode = 1; });
""")

    def test_publication_failure_after_first_file_rolls_back_the_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_js("""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const adapter = require(process.argv[1]);
const directory = process.argv[2];
const outputs = adapter.outputPaths('/tmp/input.wav', directory);
const link = fs.linkSync;
fs.linkSync = (source, target) => {
  if (target === outputs.markdown) throw new Error('simulated publication failure');
  return link(source, target);
};
try {
  assert.throws(() => adapter.writeOutputs(outputs, {metadata: {}, segments: []}), /simulated/);
} finally { fs.linkSync = link; }
assert.deepEqual(fs.readdirSync(directory), []);
""", directory)

    def test_mocked_cli_closes_runtime_and_cleans_model_root_on_success_and_failure(self):
        with tempfile.TemporaryDirectory(prefix="duo fixture ") as directory:
            root = Path(directory)
            app = root / "Fake Type.app"
            resources = app / "Contents" / "Resources"
            models = resources / "gigatype-model"
            binaries = resources / "bin"
            for part in (models / "asr", models / "llm_passthrough", binaries):
                part.mkdir(parents=True, exist_ok=True)
            (app / "Contents" / "Info.plist").write_text("mock plist")
            (models / "meta.json").write_text(json.dumps({
                "release_id": "type.v1.019.20260917", "core": "core.tar",
                "parts": {"asr": "asr", "llm_passthrough": "llm_passthrough"},
            }))
            for part, version in (("asr", "asr.gigaam_duo.v1.3"),
                                  ("llm_passthrough", "llm.passthrough")):
                (models / part / "meta.json").write_text(json.dumps({
                    "versions": {version: "config.yaml"},
                }))
            (models / "core.tar").write_bytes(b"fake core")
            (binaries / "gigatype_runtime.node").write_bytes(b"mock addon")
            for name, body in (
                ("ffprobe", "import os\nprint(int(os.environ.get('DUO_FAKE_SAMPLES', '48000'))/16000)"),
                ("ffmpeg", "import sys,os\nopen(sys.argv[-1],'wb').write(b'\\0'*(4*int(os.environ.get('DUO_FAKE_SAMPLES', '48000'))))"),
            ):
                executable = binaries / name
                executable.write_text("#!/usr/bin/env python3\n" + body + "\n")
                executable.chmod(0o755)
            preload = root / "preload.cjs"
            preload.write_text("""
const fs = require('node:fs');
const cp = require('node:child_process');
const execFile = cp.execFileSync;
cp.execFileSync = (file, args, options) => file === '/usr/bin/plutil'
  ? '3.1.0\\n' : execFile(file, args, options);
require('node:module')._extensions['.node'] = module => {
  module.exports = {open: async options => {
    fs.writeFileSync(process.env.DUO_FAKE_AUDIT, JSON.stringify({
      root: options.modelRoot, features: options.features,
      meta: JSON.parse(fs.readFileSync(options.modelRoot + '/meta.json', 'utf8')),
    }));
    let count = 0, inputSamples = 0;
    return {
      features: () => ['llm_off'], startRecording: () => { inputSamples = 0; return true; },
      pushAudio: samples => {
        inputSamples += samples.length;
        if (inputSamples > 25 * 16000) throw new Error('native input exceeds 25s including padding');
      },
      stopRecording: () => {
        count++;
        const mode = process.env.DUO_FAKE_MODE;
        options.onText('final ' + count, mode === 'incomplete' ? {complete: false}
          : mode === 'error' ? {error: 'mock failure'} : {complete: true});
        options.onAsrText('raw ' + count);
      },
      close: async () => fs.writeFileSync(process.env.DUO_FAKE_CLOSED, 'yes'),
    };
  }};
};
""")
            audio = root / "input with spaces.wav"
            audio.write_bytes(b"mock input")
            for mode, segmentation, samples, chunk_seconds in (
                ("success", "fixed", 48000, "1.5"),
                ("success", "pause", 48000, "1.5"),
                ("success", "fixed", 416000, "25"),
                ("incomplete", "pause", 48000, "1.5"),
                ("error", "fixed", 48000, "1.5"),
            ):
                with self.subTest(mode=mode, segmentation=segmentation, samples=samples):
                    scenario = f"{mode}-{segmentation}-{samples}"
                    output = root / (scenario + " output")
                    audit, closed = root / (scenario + ".json"), root / (scenario + ".closed")
                    env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", ""),
                           "DUO_FAKE_MODE": mode, "DUO_FAKE_AUDIT": str(audit), "DUO_FAKE_CLOSED": str(closed),
                           "DUO_FAKE_SAMPLES": str(samples)}
                    proc = subprocess.run(
                        [NODE, "--require", str(preload), str(SCRIPT), str(audio),
                         "--app", str(app), "--output-dir", str(output), "--chunk-seconds", chunk_seconds,
                         "--segmentation", segmentation],
                        env=env, capture_output=True, text=True, timeout=CALL_TIMEOUT,
                    )
                    self.assertEqual(proc.returncode, 0 if mode == "success" else 1, proc.stderr)
                    self.assertTrue(closed.exists(), proc.stderr)
                    opened = json.loads(audit.read_text())
                    self.assertEqual(opened["features"], ["llm_off"])
                    self.assertFalse(Path(opened["root"]).exists())
                    for part in ("asr", "llm_passthrough"):
                        self.assertTrue(Path(opened["meta"]["parts"][part]).is_relative_to(resources.resolve()))
                    if mode != "success":
                        self.assertFalse(output.exists())
                        continue
                    document = json.loads(next(output.glob("*.json")).read_text())
                    expected = ([{"start": 0, "end": 3, "text": "raw 1"}]
                                if segmentation == "pause" else
                                [{"start": 0, "end": float(chunk_seconds), "text": "raw 1"},
                                 {"start": float(chunk_seconds), "end": samples / SAMPLE_RATE, "text": "raw 2"}])
                    self.assertEqual(document["segments"], expected)
                    self.assertEqual(document["metadata"]["segmentation"]["method"], segmentation)
                    self.assertFalse(document["metadata"]["llm_enabled"])
                    self.assertEqual(document["metadata"]["timing_precision"], "segment")
                    self.assertEqual(document["metadata"]["text_source"], "onAsrText")


if __name__ == "__main__":
    unittest.main()
