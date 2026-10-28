"""Контракты паузной сегментации scripts/duo_transcribe.cjs.

PAUSE_CONFIG, detectPauses (RMS-окна 20 мс, тишина <= -40 dBFS),
pauseChunkRanges (границы чанков по паузам) и --segmentation в parseArgs.
Только stdlib: unittest + node subprocess, PCM-фикстуры f32le 16 кГц
во временном каталоге.
"""

import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "duo_transcribe.cjs"
NODE = shutil.which("node")
SAMPLE_RATE = 16_000  # адаптер работает с PCM ровно 16 кГц
WINDOW = 320          # 20 мс при 16 кГц
CALL_TIMEOUT = 30     # зависший экспорт — тоже баг, но ждать дольше незачем
BATCH_TIMEOUT = 120   # пакет планировщика гоняет сотни случаев за один запуск node
TONE_AMPLITUDE = 0.1      # чередующиеся ±0.1 → RMS -20 dBFS (слышимо)
HUSH_AMPLITUDE = 0.001    # константа → RMS -60 dBFS (тишина)
MAX_CHUNK_SAMPLES = 25 * SAMPLE_RATE
REASONS = ("pause", "forced", "final")
_MISSING = object()  # Sentinel для пропущенного аргумента detectPauses

# JS-обвязка: вызывает экспорты адаптера и печатает конверт с результатом
# за маркером ===DUO===, чтобы Python отличал «бросила исключение» от
# «вернула значение». NaN и Infinity едут сентинелами — их нет в JSON.
HARNESS = """\
const fs = require('node:fs');
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
const write = (payload) => process.stdout.write("===DUO===" + JSON.stringify(payload));
try {
  const mode = process.argv[2];
  if (mode === "get") {
    const value = adapter[process.argv[3]];
    write({ ok: true, value: value === undefined ? null : value });
  } else if (mode === "call") {
    const fn = adapter[process.argv[3]];
    if (typeof fn !== "function") throw new TypeError("нет экспорта " + process.argv[3]);
    const args = process.argv[4] === undefined ? [] : JSON.parse(process.argv[4]).map(fix);
    const value = fn(...args);
    write({ ok: true, value: value === undefined ? null : value });
  } else if (mode === "plan-batch") {
    const fn = adapter.pauseChunkRanges;
    if (typeof fn !== "function") throw new TypeError("нет экспорта pauseChunkRanges");
    const results = fix(JSON.parse(process.argv[3])).map((one) => {
      try {
        return { ok: true, value: fn(one.total, one.seconds, one.pauses) };
      } catch (error) {
        return { ok: false, message: String((error && error.message) || error) };
      }
    });
    write({ ok: true, value: results });
  } else if (mode === "detect") {
    const fn = adapter.detectPauses;
    if (typeof fn !== "function") throw new TypeError("нет экспорта detectPauses");
    const fd = fs.openSync(process.argv[3], "r");
    try {
      const total = process.argv[4] === "__undef__"
        ? undefined : fix(JSON.parse(process.argv[4]));
      if (process.argv[5] === "twice") {
        const first = fn(fd, total);
        const second = fn(fd, total);
        write({ ok: true, value: {
          first, second,
          stable: JSON.stringify(first) === JSON.stringify(second),
        } });
      } else {
        write({ ok: true, value: fn(fd, total) });
      }
    } finally {
      try { fs.closeSync(fd); } catch (error) {}
    }
  } else {
    throw new Error("неизвестный режим обвязки: " + mode);
  }
} catch (error) {
  write({ ok: false, name: (error && error.name) || "Error",
    message: String((error && error.message) || error) });
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


def tone(samples, amplitude=TONE_AMPLITUDE):
    return ("alt", samples, amplitude)


def hush(samples, amplitude=HUSH_AMPLITUDE):
    return ("const", samples, amplitude)


class DuoPauseTests(unittest.TestCase):
    """Общее: пропуск без node, вызов экспортов и CLI через subprocess."""

    @classmethod
    def setUpClass(cls):
        if NODE is None:
            raise unittest.SkipTest("node не найден в PATH")
        if not SCRIPT.exists():
            raise AssertionError(f"нет адаптера: {SCRIPT}")
        cls.adapter_temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.adapter_temporary.cleanup)
        harness = Path(cls.adapter_temporary.name) / "pause_harness.cjs"
        harness.write_text(HARNESS, encoding="utf-8")
        cls.harness = harness

    def harness_envelope(self, *args, timeout=CALL_TIMEOUT):
        argv = [NODE, str(self.harness), *[str(item) for item in args]]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "DUO_MODULE": str(SCRIPT)},
            )
        except subprocess.TimeoutExpired:
            self.fail(f"обвязка зависла на аргументах {args!r} (таймаут {timeout} с)")
        try:
            envelope = json.loads(proc.stdout.split("===DUO===")[-1])
        except json.JSONDecodeError:
            self.fail(
                f"обвязка ответила не-JSON: stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        self.assertIsInstance(envelope, dict)
        return envelope

    def call_export(self, name, *args):
        return self.harness_envelope("call", name, json.dumps(_to_js(list(args))))

    def call_ok(self, name, *args):
        envelope = self.call_export(name, *args)
        self.assertTrue(envelope.get("ok"), f"ожидали успех {name}{args!r}: {envelope}")
        return envelope["value"]

    def call_throws(self, name, *args):
        envelope = self.call_export(name, *args)
        self.assertFalse(envelope.get("ok"), f"ожидали отказ {name}{args!r}: {envelope}")

    def get_value(self, name):
        envelope = self.harness_envelope("get", name)
        self.assertTrue(envelope.get("ok"), f"ожидали успех чтения {name}: {envelope}")
        return envelope["value"]

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


class PauseConfigTests(DuoPauseTests):
    """PAUSE_CONFIG: контрактные константы паузной сегментации."""

    def test_config_contains_contract_constants(self):
        value = self.get_value("PAUSE_CONFIG")
        self.assertIsInstance(value, dict)
        expected = {
            "sampleRate": 16000,
            "windowSamples": 320,
            "windowMs": 20,
            "thresholdDbfs": -40,
            "minPauseSamples": 4000,
            "minPauseMs": 250,
            "maxChunkSeconds": 25,
        }
        for key, constant in expected.items():
            self.assertEqual(value.get(key), constant, f"PAUSE_CONFIG.{key}")


class SegmentationArgumentTests(DuoPauseTests):
    """parseArgs: --segmentation pause|fixed, по умолчанию pause."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input = self.root / "meeting with spaces.wav"
        self.input.write_bytes(b"")
        self.output_dir = self.root / "results dir"
        self.output_dir.mkdir()

    def parsed(self, *argv):
        return self.call_ok("parseArgs", list(argv))

    def rejected(self, *argv):
        self.call_throws("parseArgs", list(argv))

    def _strings(self, value):
        return [item for item in _leaf_values(value) if isinstance(item, str)]

    def test_default_segmentation_is_pause(self):
        value = self.parsed(str(self.input), "--output-dir", str(self.output_dir))
        self.assertIn("pause", self._strings(value))

    def test_segmentation_pause_and_fixed_are_accepted(self):
        for good in ("pause", "fixed"):
            with self.subTest(good=good):
                value = self.parsed(
                    str(self.input), "--output-dir", str(self.output_dir),
                    "--segmentation", good,
                )
                self.assertIn(good, self._strings(value))

    def test_segmentation_other_values_are_rejected(self):
        for bad in ("mixed", "auto", "", "Pause", "FIXED", "0"):
            with self.subTest(bad=bad):
                self.rejected(
                    str(self.input), "--output-dir", str(self.output_dir),
                    "--segmentation", bad,
                )

    def test_duplicate_segmentation_is_rejected(self):
        self.rejected(
            str(self.input), "--output-dir", str(self.output_dir),
            "--segmentation", "pause", "--segmentation", "fixed",
        )

    def test_segmentation_without_value_is_rejected(self):
        self.rejected(
            str(self.input), "--output-dir", str(self.output_dir), "--segmentation",
        )
        self.rejected(
            str(self.input), "--output-dir", str(self.output_dir),
            "--segmentation", "--app", "/Applications/Type.app",
        )

    def test_cli_rejects_unknown_segmentation_value(self):
        proc = self.run_cli(
            str(self.input), "--output-dir", str(self.output_dir),
            "--segmentation", "mixed",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--segmentation", proc.stdout + proc.stderr)


class FixedApiSentinelTests(DuoPauseTests):
    """Старый фиксированный chunkRanges не меняется от появления пауз."""

    def test_chunkranges_fixed_boundaries_unchanged(self):
        chunks = self.call_ok("chunkRanges", SAMPLE_RATE * 40, 20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [(c["startSample"], c["endSample"], c["start"], c["end"]) for c in chunks],
            [(0, 320000, 0.0, 20.0), (320000, 640000, 20.0, 40.0)],
        )


class PauseTools(DuoPauseTests):
    """Инструменты без собственных тестов: планировщик, PCM-фикстуры, инварианты.

    Отдельный базовый класс нужен, чтобы классы с тестами не наследовали
    чужие тест-методы и discover не гонял их по два раза.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    # --- PCM-фикстуры и detectPauses ---

    def write_pcm(self, name, segments, corrupt=None):
        data = array.array("f")
        for kind, count, amplitude in segments:
            if kind == "alt":
                data.extend(
                    amplitude if index % 2 == 0 else -amplitude
                    for index in range(count)
                )
            else:
                data.extend(array.array("f", [amplitude]) * count)
        total = len(data)
        if corrupt is not None:
            index, value = corrupt
            data[index] = value
        if sys.byteorder == "big":
            data.byteswap()
        path = self.root / name
        path.write_bytes(data.tobytes())
        return path, total

    def detect(self, path, total, twice=False):
        total_arg = "__undef__" if total is _MISSING else json.dumps(_to_js(total))
        args = ("detect", path, total_arg) + (("twice",) if twice else ())
        envelope = self.harness_envelope(*args)
        self.assertTrue(envelope.get("ok"), f"ожидали успех detectPauses: {envelope}")
        return envelope["value"]

    def detect_rejected(self, path, total):
        total_arg = "__undef__" if total is _MISSING else json.dumps(_to_js(total))
        envelope = self.harness_envelope("detect", path, total_arg)
        self.assertFalse(envelope.get("ok"), f"ожидали отказ detectPauses: {envelope}")

    def pauses_of(self, value):
        self.assertIsInstance(value, list)
        return [(item["startSample"], item["endSample"]) for item in value]

    # --- pauseChunkRanges ---

    def plan(self, total_samples, chunk_seconds, pauses):
        value = self.call_ok("pauseChunkRanges", total_samples, chunk_seconds, pauses)
        self.assertIsInstance(value, list)
        return value

    def plan_batch(self, cases):
        envelope = self.harness_envelope(
            "plan-batch", json.dumps(_to_js(cases)), timeout=BATCH_TIMEOUT,
        )
        self.assertTrue(envelope.get("ok"), f"plan-batch отказался: {envelope}")
        results = envelope["value"]
        self.assertEqual(len(results), len(cases))
        return results

    def triples(self, chunks):
        return [
            (chunk["startSample"], chunk["endSample"], chunk["boundaryReason"])
            for chunk in chunks
        ]

    def assert_chunk_fields(self, chunk):
        for key in ("startSample", "endSample", "start", "end", "boundaryReason"):
            self.assertIn(key, chunk, f"в чанке нет поля {key}: {chunk}")
        for key in ("startSample", "endSample"):
            self.assertIsInstance(
                chunk[key], int, f"{key} не целое число: {chunk[key]!r}",
            )
            self.assertNotIsInstance(chunk[key], bool)
        self.assertGreater(chunk["endSample"], chunk["startSample"])  # без пустых
        self.assertAlmostEqual(
            chunk["start"], chunk["startSample"] / SAMPLE_RATE, places=6,
        )
        self.assertAlmostEqual(
            chunk["end"], chunk["endSample"] / SAMPLE_RATE, places=6,
        )
        self.assertIn(chunk["boundaryReason"], REASONS)

    def assert_partition(self, chunks, total_samples):
        previous_end = 0
        for chunk in chunks:
            self.assert_chunk_fields(chunk)
            self.assertEqual(chunk["startSample"], previous_end)  # без дыр и наложений
            self.assertLessEqual(  # планнер не имеет права превышать 25 с
                chunk["endSample"] - chunk["startSample"], MAX_CHUNK_SAMPLES,
            )
            previous_end = chunk["endSample"]
        self.assertEqual(previous_end, total_samples)  # всё аудио покрыто один раз

    def assert_planner_invariants(self, chunks, total_samples, chunk_seconds):
        """Свойства контракта планировщика без угадывания точных границ."""
        self.assert_partition(chunks, total_samples)
        if not chunks:
            return
        target = int(chunk_seconds * SAMPLE_RATE)  # в тестах длины точны до сэмпла
        minimum = max(SAMPLE_RATE, int(0.6 * chunk_seconds * SAMPLE_RATE))
        cap = min(25 * SAMPLE_RATE, int((chunk_seconds + 5) * SAMPLE_RATE))
        self.assertEqual(
            chunks[-1]["boundaryReason"], "final",
            f"последний чанк не final: {chunks[-1]}",
        )
        for index, chunk in enumerate(chunks):
            length = chunk["endSample"] - chunk["startSample"]
            last = index == len(chunks) - 1
            if last:
                self.assertLessEqual(length, cap)
            elif chunk["boundaryReason"] == "pause":
                # Допуск 2 сэмпла — на округление середины паузы.
                self.assertGreaterEqual(length, minimum - 2)
                self.assertLessEqual(length, cap + 2)
            elif chunk["boundaryReason"] == "forced":
                self.assertEqual(length, target)
            else:
                self.fail(f"не последний чанк с причиной {chunk['boundaryReason']!r}")

    def assert_cut_inside_spans(self, chunks, pauses):
        """Каждая pause-граница лежит строго внутри какой-то паузы."""
        spans = [(item["startSample"], item["endSample"]) for item in pauses]
        for chunk in chunks[:-1]:
            self.assertTrue(
                any(start < chunk["endSample"] < end for start, end in spans),
                f"срез {chunk['endSample']} не внутри пауз {spans}",
            )


class DetectPausesTests(PauseTools):
    """detectPauses: RMS-окна 20 мс, тишина <= -40 dBFS, спаны >= 250 мс.

    Все фикстуры выровнены по границам окон (кратны 320 сэмплам), чтобы
    границы детекции были предсказуемы; последний неполный файл — специально.
    """

    def test_single_260ms_pause_reported_with_exact_span(self):
        # 13 окон тишины = 260 мс — минимум из-за гранулярности окон.
        path, total = self.write_pcm("pause260.pcm", [
            tone(32000), hush(13 * WINDOW), tone(32000),
        ])
        self.assertEqual(self.pauses_of(self.detect(path, total)), [(32000, 36160)])

    def test_silence_of_240ms_is_below_minimum(self):
        # 12 окон = 240 мс < 250 мс — паузы нет.
        path, total = self.write_pcm("short240.pcm", [
            tone(32000), hush(12 * WINDOW), tone(32000),
        ])
        self.assertEqual(self.pauses_of(self.detect(path, total)), [])

    def test_threshold_separates_silence_from_audible(self):
        # Порог -40 dBFS ≈ амплитуда 0.01: -46 dBFS — тишина, -34 dBFS — нет.
        quiet, quiet_total = self.write_pcm("quiet.pcm", [
            tone(8000), hush(13 * WINDOW, 0.005), tone(8000),
        ])
        self.assertEqual(
            self.pauses_of(self.detect(quiet, quiet_total)), [(8000, 12160)],
        )
        audible, audible_total = self.write_pcm("audible.pcm", [
            ("alt", 40 * WINDOW, 0.02),
        ])
        self.assertEqual(self.pauses_of(self.detect(audible, audible_total)), [])

    def test_long_pause_spanning_buffer_boundaries(self):
        # Тишина пересекает вероятные границы внутренних буфов чтения:
        # одна длинная пауза не должна ни потеряться, ни расколоться.
        big, big_total = self.write_pcm("big.pcm", [
            tone(60800), hush(35200), tone(32000),  # тишина в байтах 243200..384000
        ])
        self.assertEqual(self.pauses_of(self.detect(big, big_total)), [(60800, 96000)])
        small, small_total = self.write_pcm("small.pcm", [
            tone(16000), hush(16000), tone(16000),  # тишина в байтах 64000..128000
        ])
        self.assertEqual(self.pauses_of(self.detect(small, small_total)), [(16000, 32000)])

    def test_multiple_pauses_are_sorted_and_disjoint(self):
        path, total = self.write_pcm("multi.pcm", [
            tone(16000), hush(4800), tone(16000), hush(8000), tone(16000),
        ])
        self.assertEqual(
            self.pauses_of(self.detect(path, total)),
            [(16000, 20800), (36800, 44800)],
        )

    def test_final_partial_window_extends_pause_to_eof(self):
        # Хвост 100 сэмплов — неполное окно; оно считается и входит в паузу.
        path, total = self.write_pcm("tail.pcm", [tone(16000), hush(4160 + 100)])
        self.assertEqual(total % WINDOW, 100)
        self.assertEqual(self.pauses_of(self.detect(path, total)), [(16000, 20260)])

    def test_leading_silence_all_silence_and_all_tone(self):
        leading, leading_total = self.write_pcm("leading.pcm", [
            hush(13 * WINDOW), tone(32000),
        ])
        self.assertEqual(self.pauses_of(self.detect(leading, leading_total)), [(0, 4160)])
        silent, silent_total = self.write_pcm("silent.pcm", [hush(16000)])
        self.assertEqual(self.pauses_of(self.detect(silent, silent_total)), [(0, 16000)])
        loud, loud_total = self.write_pcm("loud.pcm", [tone(32000)])
        self.assertEqual(self.pauses_of(self.detect(loud, loud_total)), [])

    def test_reads_are_bounded_by_total_samples(self):
        # В файле есть вторая пауза, но totalSamples кончается раньше —
        # детекция обязана читать не больше totalSamples сэмплов.
        path, file_total = self.write_pcm("bounded.pcm", [
            tone(32000), hush(4160), tone(32000), hush(4160), tone(32000),
        ])
        self.assertGreater(file_total, 68160)
        self.assertEqual(
            self.pauses_of(self.detect(path, 68160)), [(32000, 36160)],
        )

    def test_zero_samples_yield_no_pauses(self):
        path, total = self.write_pcm("empty.pcm", [])
        self.assertEqual(total, 0)
        self.assertEqual(self.pauses_of(self.detect(path, 0)), [])

    def test_invalid_total_samples_are_rejected(self):
        path, _ = self.write_pcm("valid.pcm", [tone(32000), hush(4160), tone(32000)])
        for bad in (-16000, 16000.5, float("nan"), float("inf"), 2 ** 53):
            with self.subTest(bad=bad):
                self.detect_rejected(path, bad)
        self.detect_rejected(path, _MISSING)

    def test_short_read_is_rejected(self):
        tiny, tiny_total = self.write_pcm("tiny.pcm", [hush(WINDOW)])
        self.assertEqual(tiny_total, WINDOW)
        self.detect_rejected(tiny, tiny_total * 2)
        near, near_total = self.write_pcm("near.pcm", [hush(4 * WINDOW)])
        self.detect_rejected(near, near_total + 1)

    def test_nonfinite_samples_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                path, total = self.write_pcm(
                    "broken.pcm", [tone(8000)], corrupt=(100, value),
                )
                self.detect_rejected(path, total)

    def test_repeat_detection_is_stable_on_same_fd(self):
        path, total = self.write_pcm("repeat.pcm", [
            tone(16000), hush(4800), tone(16000), hush(8000), tone(16000),
        ])
        value = self.detect(path, total, twice=True)
        self.assertTrue(value["stable"])
        self.assertEqual(value["first"], value["second"])
        # Повторное открытие файла даёт тот же результат.
        again = self.detect(path, total)
        self.assertEqual(value["first"], again)


class PauseChunkRangesTests(PauseTools):
    """pauseChunkRanges: паузы двигают границы, покрытие без потерь.

    Точные границы проверяем на паузах, целиком попадающих в допустимое
    окно [max(1с, 0.6t), min(25с, t+5с)] после текущего старта; для пауз,
    пересекающих край окна, контракт точную точку среза не фиксирует.
    """

    def test_pause_at_19s_moves_20s_boundary(self):
        # Пауза 18.9..19.1 с целиком в окне, середина 304000 = 19.0 с:
        # граница уходит с 20 с на 19 с, ни один сэмпл не теряется.
        chunks = self.plan(640000, 20, [
            {"startSample": 302400, "endSample": 305600},
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 304000, "pause"), (304000, 640000, "final")],
        )
        self.assert_partition(chunks, 640000)
        self.assertAlmostEqual(chunks[0]["end"], 19.0, places=6)

    def test_late_pause_at_23s_is_eligible_for_target_20(self):
        chunks = self.plan(960000, 20, [
            {"startSample": 366400, "endSample": 369600},  # середина 23.0 с
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 368000, "pause"), (368000, 688000, "forced"), (688000, 960000, "final")],
        )
        self.assert_partition(chunks, 960000)

    def test_pause_before_minimum_window_is_ignored(self):
        # Пауза 4.9..5.1 с целиком ниже нижней границы: срез форсится.
        chunks = self.plan(640000, 20, [
            {"startSample": 78400, "endSample": 81600},
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 320000, "forced"), (320000, 640000, "final")],
        )
        self.assert_partition(chunks, 640000)

    def test_window_bounds_are_inclusive(self):
        # Паузы целиком внутри окна: срез — середина паузы. Границы окна
        # проверяем с запасом в одну granularность паузы, чтобы не фиксировать
        # поведение срезов ровно на краю (контракт его не задаёт).
        cases = [
            # 12.0..12.3 с — у нижней границы 0.6*20 с.
            (640000, [{"startSample": 192000, "endSample": 196800}],
             [(0, 194400, "pause"), (194400, 514400, "forced"), (514400, 640000, "final")]),
            # 24.7..25.0 с — у верхней границы min(25, 20+5) с.
            (960000, [{"startSample": 395200, "endSample": 400000}],
             [(0, 397600, "pause"), (397600, 717600, "forced"), (717600, 960000, "final")]),
        ]
        for total, pauses, expected in cases:
            with self.subTest(total=total):
                chunks = self.plan(total, 20, pauses)
                self.assertEqual(self.triples(chunks), expected)
                self.assert_partition(chunks, total)

    def test_pauses_outside_window_are_ignored(self):
        cases = [
            # Кончится за один сэмпл до нижней границы 12 с.
            (640000, [{"startSample": 188000, "endSample": 191999}],
             [(0, 320000, "forced"), (320000, 640000, "final")]),
            # Начинается на один сэмпл после верхней границы 25 с.
            (960000, [{"startSample": 400001, "endSample": 404800}],
             [(0, 320000, "forced"), (320000, 640000, "forced"), (640000, 960000, "final")]),
        ]
        for total, pauses, expected in cases:
            with self.subTest(total=total):
                chunks = self.plan(total, 20, pauses)
                self.assertEqual(self.triples(chunks), expected)
                self.assert_partition(chunks, total)

    def test_midpoint_just_past_25s_falls_back_to_forced(self):
        chunks = self.plan(960000, 20, [
            {"startSample": 400000, "endSample": 401600},  # середина 25.05 с
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 320000, "forced"), (320000, 640000, "forced"), (640000, 960000, "final")],
        )
        self.assert_partition(chunks, 960000)

    def test_nearest_eligible_midpoint_wins(self):
        # Кандидаты 14 с и 18 с при цели 20 с — побеждает ближайший, 18 с.
        chunks = self.plan(960000, 20, [
            {"startSample": 222400, "endSample": 225600},  # середина 14.0 с
            {"startSample": 286400, "endSample": 289600},  # середина 18.0 с
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 288000, "pause"), (288000, 608000, "forced"), (608000, 960000, "final")],
        )
        self.assert_partition(chunks, 960000)

    def test_short_remaining_is_retained_as_single_final_chunk(self):
        # 24 с <= min(20+5, 25): хвост держится одним чанком с причиной final,
        # даже если внутри него есть подходящая пауза на 12 с.
        with_pause = self.plan(384000, 20, [
            {"startSample": 190400, "endSample": 193600},
        ])
        self.assertEqual(self.triples(with_pause), [(0, 384000, "final")])
        without_pause = self.plan(384000, 20, [])
        self.assertEqual(self.triples(without_pause), [(0, 384000, "final")])
        self.assert_partition(with_pause, 384000)

    def test_empty_file_returns_no_chunks(self):
        self.assertEqual(self.plan(0, 20, []), [])

    def test_single_sample_file_is_one_final_chunk(self):
        chunks = self.plan(1, 20, [])
        self.assertEqual(self.triples(chunks), [(0, 1, "final")])
        self.assert_partition(chunks, 1)

    def test_leading_silence_does_not_empty_first_chunk(self):
        # Короткая ведущая тишина (2.4 с < 12 с) не двигает границу.
        short = self.plan(640000, 20, [{"startSample": 0, "endSample": 4800}])
        self.assertEqual(
            self.triples(short),
            [(0, 320000, "forced"), (320000, 640000, "final")],
        )
        self.assert_partition(short, 640000)
        # Длинная ведущая тишина: граница уходит внутрь паузы, первый чанк
        # непуст, хвост добивается до конца файла, ничего не зацикливается.
        long_silence = self.plan(640000, 20, [{"startSample": 0, "endSample": 320000}])
        self.assertEqual(len(long_silence), 2)
        self.assertEqual(long_silence[0]["boundaryReason"], "pause")
        self.assert_cut_inside_spans(long_silence, [{"startSample": 0, "endSample": 320000}])
        self.assert_partition(long_silence, 640000)

    def test_all_silence_file_terminates_with_full_coverage(self):
        # Файл целиком из пауз: срезы обязаны попадать в тишину, покрытие
        # полное, пустых чанков и зависаний нет.
        pauses = [
            {"startSample": 0, "endSample": 320000},
            {"startSample": 320000, "endSample": 640000},
            {"startSample": 640000, "endSample": 960000},
        ]
        chunks = self.plan(960000, 20, pauses)
        self.assert_partition(chunks, 960000)
        for chunk in chunks[:-1]:
            self.assertEqual(chunk["boundaryReason"], "pause")
        self.assert_cut_inside_spans(chunks, pauses)

    def test_trailing_pause_can_bound_chunk(self):
        chunks = self.plan(480000, 20, [
            {"startSample": 336000, "endSample": 368000},  # середина 22.0 с
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 352000, "pause"), (352000, 480000, "final")],
        )
        self.assert_partition(chunks, 480000)

    def test_span_ending_exactly_at_eof_is_valid(self):
        chunks = self.plan(640000, 20, [
            {"startSample": 636800, "endSample": 640000},
        ])
        self.assertEqual(
            self.triples(chunks),
            [(0, 320000, "forced"), (320000, 640000, "final")],
        )
        self.assert_partition(chunks, 640000)

    def test_no_pauses_matches_fixed_boundaries_with_final_tail(self):
        chunks = self.plan(800000, 20, [])
        self.assertEqual(
            self.triples(chunks),
            [(0, 320000, "forced"), (320000, 640000, "forced"), (640000, 800000, "final")],
        )
        self.assert_partition(chunks, 800000)

    def test_invalid_inputs_are_rejected(self):
        bad = [
            ("отрицательный total", -16000, 20, []),
            ("дробный total", 16000.5, 20, []),
            ("NaN total", float("nan"), 20, []),
            ("inf total", float("inf"), 20, []),
            ("total вне safe integer", 2 ** 53, 20, []),
            ("total строкой", "8000", 20, []),
            ("нулевая длительность", 640000, 0, []),
            ("отрицательная длительность", 640000, -5, []),
            ("длительность выше 25", 640000, 25.5, []),
            ("NaN длительность", 640000, float("nan"), []),
            ("inf длительность", 640000, float("inf"), []),
            ("пауза с концом раньше начала", 640000, 20, [
                {"startSample": 320000, "endSample": 160000},
            ]),
            ("паузы не отсортированы", 640000, 20, [
                {"startSample": 320000, "endSample": 400000},
                {"startSample": 160000, "endSample": 200000},
            ]),
            ("паузы пересекаются", 640000, 20, [
                {"startSample": 160000, "endSample": 200000},
                {"startSample": 192000, "endSample": 240000},
            ]),
            ("отрицательное начало паузы", 640000, 20, [
                {"startSample": -100, "endSample": 4000},
            ]),
            ("пауза за концом файла", 640000, 20, [
                {"startSample": 639000, "endSample": 700000},
            ]),
            ("дробная граница паузы", 640000, 20, [
                {"startSample": 16000.5, "endSample": 48000},
            ]),
            ("пауза без поля endSample", 640000, 20, [{"startSample": 16000}]),
            ("чужие имена полей паузы", 640000, 20, [
                {"start": 16000, "end": 48000},
            ]),
        ]
        results = self.plan_batch(
            [{"total": total, "seconds": seconds, "pauses": pauses}
             for _, total, seconds, pauses in bad],
        )
        for (label, _, _, _), result in zip(bad, results):
            with self.subTest(case=label):
                self.assertFalse(result.get("ok"), f"ожидали отказ: {result}")

    def test_missing_arguments_are_rejected(self):
        self.call_throws("pauseChunkRanges")


def _layout_pauses(layout, total):
    """Наборы валидных пауз для property-проверок: без пересечений и выхода за файл."""
    spans = []
    if layout == "dense":  # короткая пауза каждые 3 с с 2-й секунды
        start = 2 * SAMPLE_RATE
        while start + 4800 <= total:
            spans.append({"startSample": start, "endSample": start + 4800})
            start += 3 * SAMPLE_RATE
    elif layout == "sparse":  # редкие паузы по 0.5 с каждые 11 с
        start = 11 * SAMPLE_RATE
        while start + 8000 <= total:
            spans.append({"startSample": start, "endSample": start + 8000})
            start += 11 * SAMPLE_RATE
    elif layout == "all-silence":  # файл целиком из пауз по 10 с
        start = 0
        while start < total:
            end = min(start + 10 * SAMPLE_RATE, total)
            if spans and 0 < total - end < 4800:
                spans[-1]["endSample"] = total  # клеим короткий хвост к предыдущей
                break
            spans.append({"startSample": start, "endSample": end})
            start = end
    elif layout == "edges":  # тишина в начале и в конце
        if total > 0:
            spans.append({"startSample": 0, "endSample": min(32000, total)})
        if total - 32000 > 32000:
            spans.append({"startSample": total - 32000, "endSample": total})
    return spans


class PausePlannerPropertyTests(PauseTools):
    """Инварианты планировщика на представительных длинах и раскладках пауз."""

    TOTALS = (0, 1, 8000, 16000, 118400, 320000, 320001, 384000, 640000, 859200, 1200000)
    DURATIONS = (1, 2.5, 10, 20, 25)
    LAYOUTS = ("none", "dense", "sparse", "all-silence", "edges")

    def test_invariants_across_lengths_durations_and_pauses(self):
        cases = [
            {"total": total, "seconds": seconds, "pauses": _layout_pauses(layout, total)}
            for total in self.TOTALS
            for seconds in self.DURATIONS
            for layout in self.LAYOUTS
        ]
        results = self.plan_batch(cases)
        for case, result in zip(cases, results):
            with self.subTest(**case):
                self.assertTrue(result.get("ok"), f"ожидали успех: {result}")
                self.assert_planner_invariants(
                    result["value"], case["total"], case["seconds"],
                )


class PausePipelineTests(PauseTools):
    """Конвейер detectPauses → pauseChunkRanges на настоящем PCM."""

    def test_detect_then_plan_moves_boundary_to_pause_midpoint(self):
        # setUp создаёт self.root для PCM-фикстуры. Тишина 302080..307520
        # (17 окон, выровнена по границам окон), середина 304800 = 19.05 с.
        path, total = self.write_pcm("pipeline.pcm", [
            tone(302080), hush(5440), tone(332480),
        ])
        self.assertEqual(total, 640000)
        pauses = self.detect(path, total)
        self.assertEqual(self.pauses_of(pauses), [(302080, 307520)])
        chunks = self.plan(total, 20, pauses)
        self.assertEqual(
            self.triples(chunks),
            [(0, 304800, "pause"), (304800, 640000, "final")],
        )
        self.assert_partition(chunks, total)


if __name__ == "__main__":
    unittest.main()
