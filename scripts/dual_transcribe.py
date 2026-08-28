#!/usr/bin/env python3
"""Параллельная транскрипция GigaAM + Whisper и подготовка LLM-сверки.

Команды:
  run      запустить оба ASR на всей записи и подготовить comparison JSON;
  compare  подготовить comparison JSON из двух готовых транскриптов;
  render   проверить выбор LLM и вывести расхождения для пользователя.

Скрипт не выбирает «правильную» версию. LLM классифицирует кандидатов по
контракту references/discrepancy-review.md, а человек проверяет указанные
интервалы в исходной записи.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
GIGAAM_SCRIPT = SCRIPT_DIR / "gigaam_longform.py"
WHISPER_SCRIPT = SCRIPT_DIR / "transcribe.py"
SCHEMA_VERSION = 1

NEGATIONS = {
    "без",
    "запрещено",
    "запрещен",
    "запрещена",
    "не",
    "невозможно",
    "нельзя",
    "нет",
    "ни",
    "никогда",
    "никак",
    "никакой",
    "ничего",
}
TIME_WORDS = {
    "вчера",
    "завтра",
    "неделя",
    "недели",
    "неделю",
    "месяц",
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
}
DECISION_STEMS = (
    "обяз",
    "долж",
    "договор",
    "надо",
    "нужно",
    "реш",
    "сдела",
    "срок",
)
SEVERITIES = {"critical", "substantive", "minor"}
CATEGORIES = {
    "commitment_or_action",
    "condition_or_scope",
    "date_or_deadline",
    "decision",
    "meaningful_omission",
    "name_or_entity",
    "negation_or_polarity",
    "number_or_unit",
    "other_semantic",
    "risk_or_compliance",
    "technical_term",
}


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class RawDifference:
    start: float
    end: float
    gigaam_changed: list[str]
    whisper_changed: list[str]
    signals: set[str]
    coarse_start: Optional[float] = None
    coarse_end: Optional[float] = None


def fmt_ts(seconds: float, *, ceil: bool = False) -> str:
    """Формат ЧЧ:ММ:СС; конец округлять вверх, чтобы интервал не обрезался."""
    value = math.ceil(seconds) if ceil else math.floor(seconds)
    value = max(0, int(value))
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).replace("_", "")


def _finite_number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}: ожидалось число, получено {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}: число должно быть конечным")
    return number


def load_transcript(path: Path) -> dict:
    path = path.expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"не удалось прочитать транскрипт {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"{path}: нет массива segments")

    previous_start = -1.0
    for index, segment in enumerate(data["segments"]):
        if not isinstance(segment, dict):
            raise ValueError(f"{path}: segments[{index}] должен быть объектом")
        start = _finite_number(segment.get("start"), f"segments[{index}].start")
        end = _finite_number(segment.get("end"), f"segments[{index}].end")
        if start < 0 or end < start:
            raise ValueError(f"{path}: неверный интервал segments[{index}]={start}..{end}")
        if start < previous_start:
            raise ValueError(f"{path}: сегменты не отсортированы по времени")
        previous_start = start
    data["_artifact_path"] = str(path)
    return data


def validate_transcript(
    data: dict,
    source_name: str,
    duration: float,
    *,
    require_word_timestamps: bool = False,
) -> None:
    metadata = data.get("metadata") or {}
    artifact_source = metadata.get("source_file")
    if artifact_source and artifact_source != source_name:
        raise ValueError(
            f"{data.get('_artifact_path')}: source_file={artifact_source!r}, "
            f"ожидался {source_name!r}"
        )
    tolerance = max(2.0, duration * 0.001)
    for index, segment in enumerate(data.get("segments", [])):
        end = _finite_number(segment.get("end"), f"segments[{index}].end")
        if duration > 0 and end > duration + tolerance:
            raise ValueError(
                f"{data.get('_artifact_path')}: segments[{index}].end={end} "
                f"за пределами записи {duration}"
            )
        words = segment.get("words") or []
        has_spoken_text = bool(normalize_token(str(segment.get("text", ""))))
        if require_word_timestamps and has_spoken_text and not words:
            raise ValueError(
                f"{data.get('_artifact_path')}: segments[{index}] без word timestamps"
            )
        previous_word_start = -1.0
        for word_index, word in enumerate(words):
            if not isinstance(word, dict):
                raise ValueError(
                    f"segments[{index}].words[{word_index}] должен быть объектом"
                )
            start = _finite_number(
                word.get("start"), f"segments[{index}].words[{word_index}].start"
            )
            word_end = _finite_number(
                word.get("end"), f"segments[{index}].words[{word_index}].end"
            )
            if start < 0 or word_end < start or start < previous_word_start:
                raise ValueError(
                    f"segments[{index}].words[{word_index}]: неверный интервал"
                )
            if duration > 0 and word_end > duration + tolerance:
                raise ValueError(
                    f"segments[{index}].words[{word_index}].end={word_end} "
                    f"за пределами записи {duration}"
                )
            previous_word_start = start


def transcript_words(data: dict) -> tuple[list[Word], str]:
    """Вернуть слова и точность таймингов; старые JSON поддержать через сегменты."""
    result: list[Word] = []
    used_fallback = False
    for segment in data.get("segments", []):
        seg_start = _finite_number(segment.get("start", 0), "segment.start")
        seg_end = _finite_number(segment.get("end", seg_start), "segment.end")
        raw_words = segment.get("words") or []
        timed_words: list[Word] = []
        for item in raw_words:
            if not isinstance(item, dict) or not normalize_token(str(item.get("word", ""))):
                continue
            start = _finite_number(item.get("start", seg_start), "word.start")
            end = _finite_number(item.get("end", start), "word.end")
            if start < 0 or end < start:
                continue
            timed_words.append(Word(start, end, str(item.get("word", "")).strip()))
        if timed_words:
            result.extend(timed_words)
            continue

        tokens = re.findall(r"[\w-]+", str(segment.get("text", "")), re.UNICODE)
        if not tokens:
            continue
        used_fallback = True
        span = max(0.001, seg_end - seg_start)
        step = span / len(tokens)
        result.extend(
            Word(
                seg_start + index * step,
                seg_start + (index + 1) * step,
                token,
                seg_start,
                seg_end,
            )
            for index, token in enumerate(tokens)
        )
    result.sort(key=lambda word: (word.start, word.end))
    return result, "segment" if used_fallback else "word"


def _join_words(words: Iterable[Word | str]) -> str:
    values = [item.text if isinstance(item, Word) else item for item in words]
    text = " ".join(value.strip() for value in values if value.strip())
    text = re.sub(r"\s+([,.;:!?%\)\]»])", r"\1", text)
    text = re.sub(r"([\(\[«])\s+", r"\1", text)
    return text.strip()


def _slice_words(words: list[Word], start: float, end: float) -> list[Word]:
    return [word for word in words if start <= word.midpoint <= end]


def _changed_signals(gigaam: list[Word], whisper: list[Word]) -> set[str]:
    g_norm = [normalize_token(word.text) for word in gigaam]
    w_norm = [normalize_token(word.text) for word in whisper]
    signals: set[str] = set()

    g_neg = [word for word in g_norm if word in NEGATIONS]
    w_neg = [word for word in w_norm if word in NEGATIONS]
    # «необязательно» ↔ «не обязательно» — split/join, а не смена полярности.
    joined_g = "".join(g_norm)
    joined_w = "".join(w_norm)
    if g_neg != w_neg and joined_g != joined_w:
        signals.add("negation_or_polarity")

    number = re.compile(r"^\d+(?:[.,]\d+)?$")
    g_numbers = [word for word in g_norm if number.match(word)]
    w_numbers = [word for word in w_norm if number.match(word)]
    if g_numbers != w_numbers:
        signals.add("number_or_unit")

    changed = set(g_norm + w_norm)
    if changed & TIME_WORDS:
        signals.add("date_or_deadline")
    if any(any(word.startswith(stem) for stem in DECISION_STEMS) for word in changed):
        signals.add("decision_or_commitment")
    if not gigaam or not whisper:
        signals.add("meaningful_omission")
    if any(re.search(r"[A-Za-zА-ЯЁ].*[A-ZА-ЯЁ]|[A-Za-z]", word.text) for word in gigaam + whisper):
        signals.add("name_or_term")
    return signals


def _raw_differences(
    gigaam_words: list[Word],
    whisper_words: list[Word],
    duration: float,
    *,
    window_seconds: float = 20.0,
    overlap_seconds: float = 3.0,
) -> list[RawDifference]:
    """Локальный word diff: overlap даёт контекст, core-владение убирает дубли."""
    if duration <= 0:
        duration = max(
            gigaam_words[-1].end if gigaam_words else 0,
            whisper_words[-1].end if whisper_words else 0,
        )
    raw: list[RawDifference] = []
    core_start = 0.0
    while core_start < duration or (core_start == 0 and duration == 0):
        core_end = min(duration, core_start + window_seconds)
        analysis_start = max(0.0, core_start - overlap_seconds)
        analysis_end = min(duration, core_end + overlap_seconds)
        g_window = _slice_words(gigaam_words, analysis_start, analysis_end)
        w_window = _slice_words(whisper_words, analysis_start, analysis_end)
        g_norm = [normalize_token(word.text) for word in g_window]
        w_norm = [normalize_token(word.text) for word in w_window]
        matcher = SequenceMatcher(None, g_norm, w_norm, autojunk=False)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            g_changed = g_window[i1:i2]
            w_changed = w_window[j1:j2]
            timed = g_changed + w_changed
            if not timed:
                continue
            start = max(analysis_start, min(word.start for word in timed))
            end = min(analysis_end, max(word.end for word in timed))
            midpoint = (start + end) / 2.0
            owns = core_start <= midpoint < core_end
            if core_end == duration:
                owns = core_start <= midpoint <= core_end
            if not owns:
                continue
            coarse_words = [
                word for word in timed if word.segment_start is not None
            ]
            raw.append(
                RawDifference(
                    start=start,
                    end=max(end, start + 0.2),
                    gigaam_changed=[word.text for word in g_changed],
                    whisper_changed=[word.text for word in w_changed],
                    signals=_changed_signals(g_changed, w_changed),
                    coarse_start=(
                        min(word.segment_start for word in coarse_words)
                        if coarse_words
                        else None
                    ),
                    coarse_end=(
                        max(word.segment_end for word in coarse_words)
                        if coarse_words
                        else None
                    ),
                )
            )
        if core_end >= duration:
            break
        core_start = core_end
    return raw


def _merge_nearby(
    raw: list[RawDifference], max_gap: float = 4.0, max_span: float = 30.0
) -> list[RawDifference]:
    merged: list[RawDifference] = []
    for item in sorted(raw, key=lambda value: (value.start, value.end)):
        can_merge = (
            merged
            and item.start <= merged[-1].end + max_gap
            and max(merged[-1].end, item.end) - merged[-1].start <= max_span
        )
        if can_merge:
            previous = merged[-1]
            previous.end = max(previous.end, item.end)
            previous.gigaam_changed.extend(item.gigaam_changed)
            previous.whisper_changed.extend(item.whisper_changed)
            previous.signals.update(item.signals)
            coarse_starts = [
                value
                for value in (previous.coarse_start, item.coarse_start)
                if value is not None
            ]
            coarse_ends = [
                value
                for value in (previous.coarse_end, item.coarse_end)
                if value is not None
            ]
            previous.coarse_start = min(coarse_starts) if coarse_starts else None
            previous.coarse_end = max(coarse_ends) if coarse_ends else None
        else:
            merged.append(item)
    return merged


def build_comparison(
    gigaam_data: dict,
    whisper_data: dict,
    source_audio: Path,
    duration: float,
    *,
    context_padding: float = 3.0,
) -> dict:
    gigaam_words, gigaam_precision = transcript_words(gigaam_data)
    whisper_words, whisper_precision = transcript_words(whisper_data)
    precision = "word" if gigaam_precision == whisper_precision == "word" else "segment"
    raw = _merge_nearby(_raw_differences(gigaam_words, whisper_words, duration))
    candidates = []
    for index, item in enumerate(raw, start=1):
        interval_start = item.coarse_start if item.coarse_start is not None else item.start
        interval_end = item.coarse_end if item.coarse_end is not None else item.end
        start = max(0.0, interval_start - context_padding)
        end = min(duration, interval_end + context_padding) if duration > 0 else (
            interval_end + context_padding
        )
        end = max(end, start + 1.0)
        gigaam_text = _join_words(_slice_words(gigaam_words, start, end))
        whisper_text = _join_words(_slice_words(whisper_words, start, end))
        if not gigaam_text:
            gigaam_text = None
        if not whisper_text:
            whisper_text = None
        critical_signals = {
            "date_or_deadline",
            "decision_or_commitment",
            "negation_or_polarity",
            "number_or_unit",
        }
        hint = "critical" if item.signals & critical_signals else "review"
        candidates.append(
            {
                "id": f"D{index:03d}",
                "interval": {
                    "basis": "source_audio",
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "from": fmt_ts(start),
                    "to": fmt_ts(end, ceil=True),
                    "precision": precision,
                },
                "versions": {"gigaam": gigaam_text, "whisper": whisper_text},
                "changed_words": {
                    "gigaam": _join_words(item.gigaam_changed) or None,
                    "whisper": _join_words(item.whisper_changed) or None,
                },
                "machine_signals": sorted(item.signals),
                "priority_hint": hint,
            }
        )

    g_meta = gigaam_data.get("metadata") or {}
    w_meta = whisper_data.get("metadata") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_audio": {
            "ref": str(source_audio.expanduser().resolve()),
            "display_name": source_audio.name,
            "duration_seconds": round(duration, 3),
        },
        "transcripts": {
            "gigaam": {
                "artifact_ref": gigaam_data.get("_artifact_path"),
                "model": g_meta.get("model", "gigaam"),
            },
            "whisper": {
                "artifact_ref": whisper_data.get("_artifact_path"),
                "model": w_meta.get("model", "whisper"),
                "backend": w_meta.get("backend"),
            },
        },
        "policy": {
            "comparison": "llm_semantic_review",
            "resolution": "human_only",
            "audio_evidence": "original_recording_interval",
        },
        "summary": {
            "candidate_count": len(candidates),
            "word_timing_precision": precision,
        },
        "candidates": candidates,
    }


def _probe_duration(source: Path) -> float:
    if not shutil.which("ffprobe"):
        raise ValueError("ffprobe не найден в PATH")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(source),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"ffprobe не определил длительность: {proc.stderr.strip()}") from exc
    if proc.returncode != 0 or not math.isfinite(duration) or duration < 0:
        raise ValueError(f"ffprobe не определил длительность: {proc.stderr.strip()}")
    return duration


def _resolve_executable(value: str, label: str) -> str:
    expanded = Path(value).expanduser()
    if expanded.is_file():
        # Не resolve(): venv/bin/python обычно является symlink, а запуск его
        # target напрямую теряет pyvenv.cfg и пакеты виртуального окружения.
        return os.path.abspath(os.fspath(expanded))
    found = shutil.which(value)
    if found:
        return found
    raise ValueError(f"{label} не найден: {value}")


def _venv_python(name: str) -> Path:
    if os.name == "nt":
        return Path.home() / ".venvs" / name / "Scripts" / "python.exe"
    return Path.home() / ".venvs" / name / "bin" / "python"


def _default_gigaam_python() -> str:
    configured = os.environ.get("GIGAAM_PYTHON")
    if configured:
        return configured
    return str(_venv_python("asr"))


def _default_whisper_python() -> str:
    """Интерпретатор для transcribe.py.

    whisper.cpp — отдельный бинарник и работает из любого Python, а вот
    faster-whisper, resemblyzer и pyannote живут в venv (по setup.md это
    `~/.venvs/whisper`). Системный python3 в таком окружении честно
    отвечает «бэкендов нет», хотя всё установлено — поэтому venv скилла,
    если он есть, приоритетнее текущего интерпретатора.
    """
    configured = os.environ.get("WHISPER_PYTHON")
    if configured:
        return configured
    candidate = _venv_python("whisper")
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def _resolve_gigaam_device(python: str, requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        probe = subprocess.run(
            [
                python,
                "-c",
                "import torch; print('cuda' if torch.cuda.is_available() else "
                "('mps' if getattr(torch.backends, 'mps', None) "
                "and torch.backends.mps.is_available() else 'cpu'))",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("определение устройства GigaAM превысило 30 секунд") from exc
    device = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    if probe.returncode != 0 or device not in {"mps", "cuda", "cpu"}:
        detail = probe.stderr.strip() or probe.stdout.strip() or "нет ответа"
        raise ValueError(f"не удалось определить устройство GigaAM: {detail}")
    return device


def build_asr_commands(args: argparse.Namespace, staging_dir: Path) -> dict[str, list[str]]:
    source = args.input.expanduser().resolve()
    gigaam_python = _resolve_executable(args.gigaam_python, "Python для GigaAM")
    whisper_python = _resolve_executable(args.whisper_python, "Python для Whisper")
    gigaam_device = _resolve_gigaam_device(gigaam_python, args.gigaam_device)
    gigaam = [
        gigaam_python,
        str(GIGAAM_SCRIPT),
        str(source),
        "--device",
        gigaam_device,
        "--model",
        args.gigaam_model,
        "--batch-size",
        str(args.gigaam_batch_size),
        "--output-dir",
        str(staging_dir),
        "--suffix",
        ".gigaam",
    ]
    whisper = [
        whisper_python,
        str(WHISPER_SCRIPT),
        str(source),
        "--language",
        "ru",
        "--word-timestamps",
        "--yes",
        "--output-dir",
        str(staging_dir),
        "--suffix",
        ".whisper",
    ]
    if args.whisper_backend:
        whisper.extend(["--backend", args.whisper_backend])
    if args.initial_prompt:
        whisper.extend(["--initial-prompt", args.initial_prompt])
    if args.no_condition_on_previous_text:
        whisper.append("--no-condition-on-previous-text")
    # Диаризация не встраивается в ASR-проход: спикеры считаются один раз
    # после обоих проходов и раскладываются по обоим транскриптам, иначе
    # метки в двух версиях были бы несопоставимы между собой.
    return {"gigaam": gigaam, "whisper": whisper}


def build_diarize_command(
    args: argparse.Namespace, transcripts: list[Path]
) -> list[str]:
    command = [
        _resolve_executable(args.whisper_python, "Python для Whisper"),
        str(WHISPER_SCRIPT),
        str(args.input.expanduser().resolve()),
        "--diarize-only",
        "--diarizer",
        args.diarizer,
        "--max-speakers",
        str(args.max_speakers),
        "--apply-to",
        *[str(path) for path in transcripts],
    ]
    if args.num_speakers is not None:
        command.extend(["--num-speakers", str(args.num_speakers)])
    return command


def _pump_output(name: str, process: subprocess.Popen, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{name}] {line}", end="", flush=True)
        finally:
            process.stdout.close()


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            elif os.name == "nt" and shutil.which("taskkill"):
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.kill()
        except ProcessLookupError:
            return
        process.wait(timeout=5)


ENGINE_ORDER = ("gigaam", "whisper")


def run_sequential(commands: dict[str, list[str]], log_dir: Path) -> dict[str, int]:
    """Прогнать движки по очереди, GigaAM первым — он в разы быстрее.

    Это дефолт. На машине с общей памятью параллельный запуск проигрывает
    катастрофически: замер на 16 ГБ дал 1:53:48 у Whisper против 8:57 у того
    же прохода в одиночку (выход в своп, из которого процесс не возвращается
    даже после того, как GigaAM освободил ресурсы). Выигрыш параллельности
    здесь ограничен временем более быстрого движка — около минуты на
    24-минутной записи, — поэтому размен невыгоден.

    GigaAM идёт первым, чтобы отказ окружения всплывал через пару минут, а не
    после двадцатиминутного прохода Whisper.
    """
    statuses: dict[str, int] = {}
    for name in ENGINE_ORDER:
        command = commands.get(name)
        if command is None:
            continue
        print(f"[{name}] запуск: {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        try:
            _pump_output(name, process, log_dir / f"{name}.log")
            statuses[name] = int(process.wait())
        except BaseException:
            _terminate_process(process)
            raise
        if statuses[name] != 0:
            # Второй проход не запускаем: пару всё равно нельзя опубликовать.
            print(
                f"[{name}] завершился с кодом {statuses[name]}; "
                "второй проход не запускаю",
                file=sys.stderr,
                flush=True,
            )
            break
    return statuses


def run_parallel(commands: dict[str, list[str]], log_dir: Path) -> dict[str, int]:
    processes: dict[str, subprocess.Popen] = {}
    threads: list[threading.Thread] = []
    try:
        for name, command in commands.items():
            print(f"[{name}] запуск: {' '.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name == "posix"),
            )
            processes[name] = process
            thread = threading.Thread(
                target=_pump_output,
                args=(name, process, log_dir / f"{name}.log"),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        while any(process.poll() is None for process in processes.values()):
            failed = [
                name
                for name, process in processes.items()
                if process.poll() not in (None, 0)
            ]
            if failed:
                for process in processes.values():
                    _terminate_process(process)
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        for process in processes.values():
            _terminate_process(process)
        raise
    except BaseException:
        for process in processes.values():
            _terminate_process(process)
        raise
    finally:
        for thread in threads:
            thread.join(timeout=5)
    return {name: int(process.wait()) for name, process in processes.items()}


def _atomic_publish(
    staging_dir: Path,
    output_dir: Path,
    stem: str,
    *,
    source_name: Optional[str] = None,
    duration: Optional[float] = None,
    require_word_timestamps: bool = False,
) -> list[Path]:
    names = [
        f"{stem}.gigaam.transcript.json",
        f"{stem}.gigaam.transcript.md",
        f"{stem}.gigaam.transcript.srt",
        f"{stem}.whisper.transcript.json",
        f"{stem}.whisper.transcript.md",
        f"{stem}.whisper.transcript.srt",
        f"{stem}.comparison.json",
        f"{stem}.review-template.json",
    ]
    for name in names:
        source = staging_dir / name
        if not source.is_file():
            raise ValueError(f"ASR не создал обязательный артефакт: {source}")
    # Всё проверить до первой замены: старые/частичные артефакты не подмешивать.
    for name in (names[0], names[3]):
        transcript = load_transcript(staging_dir / name)
        if source_name is not None and duration is not None:
            validate_transcript(
                transcript,
                source_name,
                duration,
                require_word_timestamps=require_word_timestamps,
            )
    for name in names[6:]:
        json.loads((staging_dir / name).read_text(encoding="utf-8"))

    published: list[Path] = []
    backups: dict[Path, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = staging_dir / ".publish-backup"
    backup_dir.mkdir()
    try:
        for name in names:
            source = staging_dir / name
            destination = output_dir / name
            if destination.exists():
                backup = backup_dir / name
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            if destination.exists():
                os.replace(destination, staging_dir / destination.name)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    return published


def write_comparison(
    gigaam_json: Path,
    whisper_json: Path,
    source: Path,
    duration: float,
    output: Path,
) -> dict:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"исходная запись не найдена: {source}")
    duration = _finite_number(duration, "duration")
    if duration < 0:
        raise ValueError("duration не может быть отрицательной")
    gigaam_data = load_transcript(gigaam_json)
    whisper_data = load_transcript(whisper_json)
    validate_transcript(gigaam_data, source.name, duration)
    validate_transcript(whisper_data, source.name, duration)
    comparison = build_comparison(gigaam_data, whisper_data, source, duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return comparison


def write_review_template(comparison_path: Path, comparison: dict, output: Path) -> None:
    template = {
        "schema_version": SCHEMA_VERSION,
        "comparison_file": str(comparison_path.resolve()),
        "decisions": [],
    }
    output.write_text(
        json.dumps(template, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _review_output_paths(review_path: Path, output: Optional[Path]) -> tuple[Path, Path]:
    if output:
        base = output.expanduser().resolve()
        if base.suffix in {".json", ".md"}:
            base = base.with_suffix("")
    else:
        name = review_path.name
        for suffix in (".review.json", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        base = review_path.parent / f"{name}.disagreements"
    return Path(str(base) + ".json"), Path(str(base) + ".md")


def render_review(review_path: Path, output: Optional[Path] = None) -> tuple[Path, Path]:
    review_path = review_path.expanduser().resolve()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("review JSON должен быть объектом")
    review_fields = {"schema_version", "comparison_file", "decisions"}
    unexpected_review_fields = set(review) - review_fields
    if unexpected_review_fields:
        raise ValueError(
            "неожиданные поля review JSON: "
            + ", ".join(sorted(unexpected_review_fields))
        )
    if review.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"неподдерживаемая schema_version: {review.get('schema_version')!r}")
    comparison_ref = Path(str(review.get("comparison_file", ""))).expanduser()
    if not comparison_ref.is_absolute():
        comparison_ref = review_path.parent / comparison_ref
    comparison = json.loads(comparison_ref.read_text(encoding="utf-8"))
    if not isinstance(comparison, dict) or comparison.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("comparison JSON имеет неподдерживаемую схему")
    candidates = {item["id"]: item for item in comparison.get("candidates", [])}
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("decisions должен быть массивом")

    selected = []
    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"decisions[{index}] должен быть объектом")
        verdict = decision.get("verdict")
        if verdict == "discrepancy":
            decision_fields = {
                "candidate_id",
                "verdict",
                "severity",
                "categories",
                "review_reason",
            }
        elif verdict == "same_meaning":
            decision_fields = {"candidate_id", "verdict", "review_reason"}
        else:
            raise ValueError(f"decisions[{index}]: неизвестный verdict {verdict!r}")
        unexpected_fields = set(decision) - decision_fields
        if unexpected_fields:
            raise ValueError(
                f"decisions[{index}]: неожиданные поля: "
                + ", ".join(sorted(unexpected_fields))
            )
        candidate_id = decision.get("candidate_id")
        if candidate_id not in candidates:
            raise ValueError(f"decisions[{index}]: неизвестный candidate_id {candidate_id!r}")
        if candidate_id in seen:
            raise ValueError(f"decisions[{index}]: повтор candidate_id {candidate_id!r}")
        seen.add(candidate_id)
        reason_value = decision.get("review_reason")
        if not isinstance(reason_value, str) or not reason_value.strip():
            raise ValueError(f"decisions[{index}]: пустой review_reason")
        reason = reason_value.strip()
        if verdict == "same_meaning":
            continue

        severity = decision.get("severity")
        if severity not in SEVERITIES:
            raise ValueError(f"decisions[{index}]: неизвестная severity {severity!r}")
        categories = decision.get("categories")
        if (
            not isinstance(categories, list)
            or not categories
            or not all(isinstance(category, str) for category in categories)
            or len(categories) != len(set(categories))
            or not set(categories) <= CATEGORIES
        ):
            raise ValueError(f"decisions[{index}]: неверные categories")
        selected.append(
            {
                "id": candidate_id,
                "status": "pending_human_review",
                "priority": severity,
                "categories": categories,
                "interval": candidates[candidate_id]["interval"],
                "versions": candidates[candidate_id]["versions"],
                "review_reason": reason,
            }
        )

    missing = set(candidates) - seen
    if missing:
        raise ValueError(
            "LLM-review не покрывает всех кандидатов; отсутствуют: "
            + ", ".join(sorted(missing))
        )

    rank = {"critical": 0, "substantive": 1, "minor": 2}
    selected.sort(key=lambda item: (rank[item["priority"]], item["interval"]["start_seconds"]))
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_audio": comparison["source_audio"],
        "transcripts": comparison["transcripts"],
        "policy": comparison["policy"],
        "reviewed_candidates": len(candidates),
        "discrepancies": selected,
    }
    json_path, md_path = _review_output_paths(review_path, output)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    counts = {
        severity: sum(item["priority"] == severity for item in selected)
        for severity in SEVERITIES
    }
    lines = [
        "# Расхождения транскрипции",
        "",
        f"- Исходная запись: `{comparison['source_audio']['ref']}`",
        f"- Проверено кандидатов: {len(candidates)}",
        f"- Требуют проверки человеком: {len(selected)} "
        f"(критических: {counts['critical']}, существенных: {counts['substantive']}, "
        f"незначительных: {counts['minor']})",
        "",
        "Порядок версий не означает, что одна модель предпочтительнее. "
        "Правильный вариант определяется только после прослушивания исходной записи.",
    ]
    if not selected:
        lines.extend(["", "Смысловых расхождений для проверки не найдено."])
    labels = {"critical": "критическое", "substantive": "существенное", "minor": "незначительное"}
    for item in selected:
        interval = item["interval"]
        precision_note = (
            " (приблизительно: доступны только границы ASR-сегмента)"
            if interval.get("precision") == "segment"
            else ""
        )
        gigaam = item["versions"].get("gigaam") or "— фрагмент отсутствует"
        whisper = item["versions"].get("whisper") or "— фрагмент отсутствует"
        lines.extend(
            [
                "",
                f"## {item['id']} — {labels[item['priority']]}: {', '.join(item['categories'])}",
                "",
                f"- Интервал в исходной записи: "
                f"`{interval['from']}–{interval['to']}`{precision_note}",
                "",
                "**GigaAM**",
                "",
                f"> {gigaam}",
                "",
                "**Whisper**",
                "",
                f"> {whisper}",
                "",
                f"**Почему нужно проверить:** {item['review_reason']}",
                "",
                "Прослушайте указанный интервал в исходной записи и сообщите "
                f"правильную формулировку, например: `{item['id']}: «…»`.",
            ]
        )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return json_path, md_path


def cmd_run(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: файл не найден: {source}", file=sys.stderr)
        return 1
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ERROR: нужны ffmpeg и ffprobe в PATH", file=sys.stderr)
        return 1
    output_dir = (args.output_dir or source.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    try:
        duration = _probe_duration(source)
        with tempfile.TemporaryDirectory(prefix=".meeting-dual-", dir=output_dir) as temporary:
            staging = Path(temporary)
            commands = build_asr_commands(args, staging)
            if args.parallel:
                print(
                    "Запускаю два полных ASR-прохода параллельно. "
                    "На машине с общей памятью это может выйти многократно "
                    "медленнее последовательного запуска.",
                    flush=True,
                )
                statuses = run_parallel(commands, staging)
            else:
                print(
                    "Запускаю два полных ASR-прохода по очереди: GigaAM, затем Whisper.",
                    flush=True,
                )
                statuses = run_sequential(commands, staging)
            failed = {name: code for name, code in statuses.items() if code != 0}
            if failed:
                print(f"ERROR: ASR завершился с ошибкой: {failed}", file=sys.stderr)
                for name in ("gigaam", "whisper"):
                    log_source = staging / f"{name}.log"
                    if log_source.is_file():
                        log_destination = output_dir / f"{source.stem}.{name}.failed.log"
                        os.replace(log_source, log_destination)
                        print(f"  лог: {log_destination}", file=sys.stderr)
                return 2
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                print("ERROR: исходная запись изменилась во время транскрипции", file=sys.stderr)
                return 2

            gigaam_json = staging / f"{source.stem}.gigaam.transcript.json"
            whisper_json = staging / f"{source.stem}.whisper.transcript.json"
            if args.diarize:
                command = build_diarize_command(args, [gigaam_json, whisper_json])
                print("\nСчитаю спикеров один раз для обоих транскриптов.", flush=True)
                print(f"[diarize] запуск: {' '.join(command)}", flush=True)
                code = subprocess.call(command)
                if code != 0:
                    raise ValueError(f"диаризация завершилась с кодом {code}")
            staged_comparison = staging / f"{source.stem}.comparison.json"
            comparison = write_comparison(
                gigaam_json,
                whisper_json,
                source,
                duration,
                staged_comparison,
            )
            comparison_path = output_dir / staged_comparison.name
            comparison["transcripts"]["gigaam"]["artifact_ref"] = str(
                output_dir / gigaam_json.name
            )
            comparison["transcripts"]["whisper"]["artifact_ref"] = str(
                output_dir / whisper_json.name
            )
            staged_comparison.write_text(
                json.dumps(
                    comparison,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            staged_template = staging / f"{source.stem}.review-template.json"
            write_review_template(comparison_path, comparison, staged_template)
            published = _atomic_publish(
                staging,
                output_dir,
                source.stem,
                source_name=source.name,
                duration=duration,
                require_word_timestamps=True,
            )
            for name in ("gigaam", "whisper"):
                log_source = staging / f"{name}.log"
                if log_source.is_file():
                    os.replace(log_source, output_dir / f"{source.stem}.{name}.log")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nГотовы раздельные транскрипты и вход для LLM-сверки:")
    for path in published:
        print(f"  {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    source = args.source.expanduser().resolve()
    try:
        if not source.is_file():
            raise ValueError(f"исходная запись не найдена: {source}")
        duration = args.duration if args.duration is not None else _probe_duration(source)
        output = (
            args.output.expanduser().resolve()
            if args.output
            else source.parent / f"{source.stem}.comparison.json"
        )
        comparison = write_comparison(args.gigaam, args.whisper, source, duration, output)
        if output.name.endswith(".comparison.json"):
            template_name = output.name[: -len(".comparison.json")] + ".review-template.json"
        else:
            template_name = output.stem + ".review-template.json"
        template = output.with_name(template_name)
        write_review_template(output, comparison, template)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"кандидатов: {comparison['summary']['candidate_count']}")
    print(f"  {output}")
    print(f"  {template}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    try:
        json_path, md_path = render_review(args.review, args.output)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"  {json_path}")
    print(f"  {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="запустить GigaAM и Whisper параллельно")
    run.add_argument("input", type=Path)
    run.add_argument("--output-dir", type=Path, default=None)
    run.add_argument("--gigaam-python", default=_default_gigaam_python())
    run.add_argument("--whisper-python", default=_default_whisper_python())
    run.add_argument(
        "--gigaam-device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
    )
    run.add_argument("--gigaam-model", default="v3_e2e_rnnt")
    run.add_argument("--gigaam-batch-size", type=int, default=8)
    run.add_argument(
        "--whisper-backend",
        choices=["whisper-cpp", "mlx-whisper", "faster-whisper"],
        default=None,
    )
    run.add_argument(
        "--parallel",
        action="store_true",
        help="Запускать оба движка одновременно. По умолчанию они идут по "
             "очереди: на машине с общей памятью параллельный запуск даёт "
             "многократное замедление при выигрыше в минуту.",
    )
    run.add_argument("--initial-prompt", default=None)
    run.add_argument("--no-condition-on-previous-text", action="store_true")
    run.add_argument("--diarize", action="store_true")
    run.add_argument(
        "--diarizer",
        choices=["auto", "pyannote", "resemblyzer", "none"],
        default="auto",
    )
    run.add_argument("--num-speakers", type=int, default=None)
    run.add_argument("--max-speakers", type=int, default=6)
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="сопоставить два готовых JSON-транскрипта")
    compare.add_argument("--gigaam", type=Path, required=True)
    compare.add_argument("--whisper", type=Path, required=True)
    compare.add_argument("--source", type=Path, required=True)
    compare.add_argument("--duration", type=float, default=None)
    compare.add_argument("--output", type=Path, default=None)
    compare.set_defaults(func=cmd_compare)

    render = sub.add_parser("render", help="проверить LLM-review JSON и вывести Markdown")
    render.add_argument("review", type=Path)
    render.add_argument("--output", type=Path, default=None)
    render.set_defaults(func=cmd_render)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
