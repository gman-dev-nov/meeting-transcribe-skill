#!/usr/bin/env python3
"""
Локальная транскрипция аудио/видео через Whisper.

Пресеты скорости/качества (для часа аудио на M4):

  quality   — large-v3,        beam_size=5  (макс качество)
  balanced  — large-v3-turbo,  beam_size=5  (рекомендуется по умолчанию)
  fast      — large-v3-turbo,  beam_size=1  (черновики, минимум)

Бэкенды whisper:
  faster-whisper  — кросс-платформенный (CPU). На M4 даёт ~RTF 0.4–1.0.
  mlx-whisper     — только Apple Silicon, ускорение через MLX. RTF 0.05–0.20.

Диаризация (опциональная, --diarize):
  pyannote     — точнее, требует бесплатный HF_TOKEN и accept условий модели
  resemblyzer  — полностью локально, чуть менее точно
  auto         — pyannote если HF_TOKEN есть, иначе resemblyzer

Использование:
    python transcribe.py video.mp4 --preset balanced --language ru --diarize
    python transcribe.py audio.m4a --estimate-only            # только оценка
    python transcribe.py audio.m4a --preset fast --yes        # без интерактива

Артефакты сохраняются рядом с исходным файлом:
    <name>.transcript.json
    <name>.transcript.md
    <name>.transcript.srt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".aif", ".aiff", ".aifc"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


# -------------- ПРЕСЕТЫ И ОЦЕНКА ВРЕМЕНИ --------------

PRESETS = {
    "quality": {
        "model": "large-v3",
        "beam_size": 5,
        "description": "Максимальное качество (large-v3, beam=5)",
    },
    "balanced": {
        "model": "large-v3-turbo",
        "beam_size": 5,
        "description": "Сбалансированный (large-v3-turbo, beam=5)",
    },
    "fast": {
        "model": "large-v3-turbo",
        "beam_size": 1,
        "description": "Быстрый, для черновиков (large-v3-turbo, beam=1)",
    },
}

# Real-time factor: сколько секунд CPU/GPU времени тратится на 1 секунду аудио.
# Это приблизительные цифры для Mac Mini M4 16GB.
# Реальная скорость зависит от загрузки системы, длины пауз, языка.
RTF_TABLE = {
    "faster-whisper": {
        "quality": 0.9,
        "balanced": 0.40,
        "fast": 0.20,
    },
    "mlx-whisper": {
        "quality": 0.20,
        "balanced": 0.10,  # mlx auto-fallback to greedy → effectively as fast
        "fast": 0.05,
    },
    "whisper-cpp": {
        # Замерено на M4 на 2ч38м файле. Metal-ускоренный, beam search есть.
        "quality": 0.16,
        "balanced": 0.07,
        "fast": 0.05,
    },
}

# Фиксированный оверхед на загрузку моделей (одноразовый, в секундах)
LOAD_OVERHEAD = {
    "faster-whisper": 8,
    "mlx-whisper": 6,
    "whisper-cpp": 5,
}

# Пути для поиска бинарника и моделей whisper.cpp
WCPP_BIN_CANDIDATES = [
    os.environ.get("WHISPER_CPP_BIN", ""),
    f"{os.environ.get('WHISPER_CPP_HOME', '')}/build/bin/whisper-cli",
    f"{Path.home()}/whisper.cpp/build/bin/whisper-cli",
    f"{Path.home()}/.local/share/whisper.cpp/build/bin/whisper-cli",
    "/usr/local/bin/whisper-cli",
    "/opt/homebrew/bin/whisper-cli",
]
WCPP_MODELS_CANDIDATES = [
    os.environ.get("WHISPER_CPP_MODELS", ""),
    f"{os.environ.get('WHISPER_CPP_HOME', '')}/models",
    f"{Path.home()}/whisper.cpp/models",
    f"{Path.home()}/.local/share/whisper.cpp/models",
]

# Дополнительный RTF для диаризации
DIARIZE_RTF = {
    "pyannote": 0.12,
    "resemblyzer": 0.05,
}
DIARIZE_LOAD = {
    "pyannote": 25,    # pyannote.audio долго грузится
    "resemblyzer": 8,
}


def estimate_seconds(
    duration_sec: float,
    backend: str,
    preset: str,
    diarizer: Optional[str],
) -> float:
    """Оценка времени обработки в секундах."""
    rtf = RTF_TABLE[backend][preset]
    base = LOAD_OVERHEAD[backend] + duration_sec * rtf
    if diarizer:
        base += DIARIZE_LOAD[diarizer] + duration_sec * DIARIZE_RTF[diarizer]
    return base


def fmt_estimate(seconds: float) -> str:
    """Человекочитаемая длительность с округлением вверх."""
    if seconds < 60:
        return f"~{int(seconds)} сек"
    minutes = seconds / 60
    if minutes < 10:
        return f"~{minutes:.1f} мин"
    return f"~{int(round(minutes))} мин"


# -------------- УТИЛИТЫ --------------

def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_ts_srt(seconds: float) -> str:
    # Через целые миллисекунды: округление 59.9996 → 60.0 не даст «00:00:60,000»
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print(
            "ERROR: ffmpeg не найден в PATH.\n"
            "  brew install ffmpeg     # macOS\n"
            "  sudo apt install ffmpeg # Linux\n"
            "  без Homebrew: статические ffmpeg+ffprobe → ~/.local/bin"
            " (references/setup.md → «ffmpeg без Homebrew»)",
            file=sys.stderr,
        )
        sys.exit(1)


def probe_duration(path: Path) -> float:
    """Длительность медиафайла в секундах через ffprobe."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def probe_audio_properties(path: Path) -> dict:
    """Битрейт / sample_rate / channels первого аудиопотока через ffprobe."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=bit_rate,sample_rate,channels,codec_name",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(proc.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        return {
            "codec": stream.get("codec_name"),
            "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "channels": stream.get("channels"),
            "bitrate_kbps": int(stream["bit_rate"]) // 1000 if stream.get("bit_rate") else None,
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {"codec": None, "sample_rate": None, "channels": None, "bitrate_kbps": None}


def probe_silence_stats(path: Path, threshold_db: int = -35, min_silence_sec: float = 5.0) -> dict:
    """Считает длинные тишины (>= min_silence_sec) через ffmpeg silencedetect.
    Длинные тишины — главный сигнал loop-риска для turbo-пресетов."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_sec}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    durations: list[float] = []
    for line in proc.stderr.splitlines():
        if "silence_duration:" in line:
            try:
                durations.append(float(line.rsplit("silence_duration:", 1)[1].strip()))
            except ValueError:
                pass
    return {
        "threshold_db": threshold_db,
        "min_silence_sec": min_silence_sec,
        "count": len(durations),
        "total_seconds": round(sum(durations), 1),
        "max_seconds": round(max(durations), 1) if durations else 0.0,
    }


def probe_volume_stats(path: Path) -> dict:
    """Mean/max громкость в dB через ffmpeg volumedetect."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    mean_db = max_db = None
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "max_volume:" in line:
            try:
                max_db = float(line.split("max_volume:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return {"mean_db": mean_db, "max_db": max_db}


def recommend_preset(duration_sec: float, silence: dict, audio: dict, volume: dict,
                     available_backends: list[str]) -> dict:
    """Эвристика выбора пресета. Возвращает {preset, backend, warnings, reason}."""
    warnings: list[str] = []
    duration_min = duration_sec / 60.0

    if duration_min >= 150:
        warnings.append("very_long")
    elif duration_min >= 90:
        warnings.append("long")

    silence_ratio = silence["total_seconds"] / duration_sec if duration_sec > 0 else 0
    if silence["max_seconds"] >= 30:
        warnings.append("long_silence_block")
    if silence_ratio > 0.10:
        warnings.append("high_silence_ratio")

    if audio.get("sample_rate") and audio["sample_rate"] <= 8000:
        warnings.append("low_sample_rate")
    if audio.get("bitrate_kbps") and audio["bitrate_kbps"] < 64:
        warnings.append("low_bitrate")

    if volume.get("mean_db") is not None and volume["mean_db"] < -30:
        warnings.append("quiet_audio")

    # Решение по пресету: чем больше факторов риска — тем выше пресет.
    # turbo (fast/balanced) на длинных тишинах склонен зацикливаться;
    # large-v3 (quality) с beam=5 устойчивее.
    risk_long = "very_long" in warnings or "long_silence_block" in warnings
    risk_audio = ("low_sample_rate" in warnings or "low_bitrate" in warnings
                  or "quiet_audio" in warnings)

    if risk_long:
        preset = "quality"
    elif "long" in warnings or risk_audio or "high_silence_ratio" in warnings:
        preset = "balanced"
    else:
        preset = "fast"

    # Бэкенд: предпочитаем whisper-cpp (Metal + честный beam search), иначе авто-приоритет
    backend_order = ["whisper-cpp", "mlx-whisper", "faster-whisper"]
    backend = next((b for b in backend_order if b in available_backends), None)

    # mlx-whisper не умеет beam search — для balanced/quality лучше whisper-cpp или faster-whisper
    if preset in {"balanced", "quality"} and backend == "mlx-whisper":
        switched = False
        for alt in ("whisper-cpp", "faster-whisper"):
            if alt in available_backends:
                backend = alt
                switched = True
                break
        if not switched:
            warnings.append("mlx_only_no_beam")

    reason_parts: list[str] = []
    if "very_long" in warnings:
        reason_parts.append(f"запись {int(duration_min)} мин — длинная")
    elif "long" in warnings:
        reason_parts.append(f"запись {int(duration_min)} мин")
    if "long_silence_block" in warnings:
        reason_parts.append(f"есть тишина ≥{int(silence['max_seconds'])}с (риск loop'а на turbo)")
    if "high_silence_ratio" in warnings:
        reason_parts.append(f"тишины {int(silence_ratio*100)}%")
    if "low_sample_rate" in warnings:
        reason_parts.append(f"низкая частота {audio['sample_rate']}Hz (телефон?)")
    if "low_bitrate" in warnings:
        reason_parts.append(f"низкий битрейт {audio['bitrate_kbps']}kbps")
    if "quiet_audio" in warnings:
        reason_parts.append(f"тихая запись (mean {volume['mean_db']}dB)")

    # Длинные тишины ломают context-окно whisper и провоцируют hallucination-loops
    # (классика — повторяющееся «Так./Видно?» или «Продолжение следует…» час подряд).
    # Авто-выставляем --no-condition-on-previous-text для таких записей.
    no_condition = "long_silence_block" in warnings or "high_silence_ratio" in warnings

    if reason_parts:
        reason = f"Рекомендую {preset}: " + ", ".join(reason_parts) + "."
    else:
        reason = f"Рекомендую {preset}: запись короткая и чистая, turbo справится."

    return {
        "preset": preset,
        "backend": backend,
        "warnings": warnings,
        "reason": reason,
        "no_condition_on_previous_text": no_condition,
    }


def build_analysis(path: Path, duration: float, env: dict) -> dict:
    """Полный preflight: длительность + audio props + silence + volume + рекомендация."""
    audio = probe_audio_properties(path)
    silence = probe_silence_stats(path)
    volume = probe_volume_stats(path)
    available = []
    if env.get("whisper_cpp_bin"):
        available.append("whisper-cpp")
    if env["mlx_whisper"]:
        available.append("mlx-whisper")
    if env["faster_whisper"]:
        available.append("faster-whisper")
    rec = recommend_preset(duration, silence, audio, volume, available)
    return {
        "duration_seconds": round(duration, 3),
        "duration_hms": fmt_ts(duration),
        "audio": audio,
        "silence": silence,
        "volume": volume,
        "recommendation": rec,
    }


def extract_audio(input_path: Path, dest_wav: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dest_wav),
    ]
    print(f"[ffmpeg] извлекаю аудио → {dest_wav.name}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ffmpeg failed:\n" + proc.stderr, file=sys.stderr)
        sys.exit(2)


# -------------- АВТОДЕТЕКТ ОКРУЖЕНИЯ --------------

def find_whisper_cpp() -> tuple[Optional[str], Optional[str]]:
    """Returns (binary_path, models_dir) or (None, None) if not found."""
    bin_path = None
    for cand in WCPP_BIN_CANDIDATES:
        if cand and Path(cand).is_file():
            bin_path = cand
            break
    if not bin_path:
        bin_path = shutil.which("whisper-cli")
    if not bin_path:
        return None, None
    models_dir = None
    for cand in WCPP_MODELS_CANDIDATES:
        if cand and Path(cand).is_dir():
            models_dir = cand
            break
    if not models_dir:
        # Try to infer relative to binary: <home>/build/bin/whisper-cli → <home>/models
        try:
            inferred = Path(bin_path).resolve().parents[2] / "models"
            if inferred.is_dir():
                models_dir = str(inferred)
        except IndexError:
            pass
    return bin_path, models_dir


WCPP_MODEL_FILES = {
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
    "large-v2": "ggml-large-v2.bin",
    "medium": "ggml-medium.bin",
    "small": "ggml-small.bin",
    "base": "ggml-base.bin",
    "tiny": "ggml-tiny.bin",
}


def whisper_cpp_model_path(models_dir: Optional[str], model_name: str) -> Optional[str]:
    if not models_dir:
        return None
    fname = WCPP_MODEL_FILES.get(model_name, f"ggml-{model_name}.bin")
    p = Path(models_dir) / fname
    return str(p) if p.is_file() else None


def detect_environment() -> dict:
    """Возвращает информацию о доступных бэкендах и диаризаторах."""
    wcpp_bin, wcpp_models = find_whisper_cpp()
    env = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "faster_whisper": False,
        "mlx_whisper": False,
        "whisper_cpp_bin": wcpp_bin,
        "whisper_cpp_models_dir": wcpp_models,
        "pyannote": False,
        "resemblyzer": False,
        "hf_token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    }
    try:
        import faster_whisper  # noqa
        env["faster_whisper"] = True
    except ImportError:
        pass
    try:
        import mlx_whisper  # noqa
        env["mlx_whisper"] = True
    except ImportError:
        pass
    try:
        import pyannote.audio  # noqa
        env["pyannote"] = True
    except ImportError:
        pass
    try:
        import resemblyzer  # noqa
        import sklearn  # noqa
        env["resemblyzer"] = True
    except ImportError:
        pass
    return env


def resolve_diarizer(requested: str, env: dict) -> Optional[str]:
    """
    Преобразует --diarizer (auto/pyannote/resemblyzer/none) в фактический бэкенд
    с учётом установленных пакетов и токена. Возвращает 'pyannote'/'resemblyzer'/None.
    """
    if requested == "none":
        return None
    if requested == "pyannote":
        if not env["pyannote"]:
            print("ERROR: pyannote.audio не установлен. Поставь: pip install pyannote.audio", file=sys.stderr)
            sys.exit(4)
        if not env["hf_token"]:
            print(
                "ERROR: для pyannote нужен HF_TOKEN.\n"
                "  1. Зарегистрируйся на huggingface.co\n"
                "  2. Прими условия: https://hf.co/pyannote/speaker-diarization-3.1\n"
                "  3. Создай токен: https://hf.co/settings/tokens\n"
                "  4. export HF_TOKEN=hf_xxx",
                file=sys.stderr,
            )
            sys.exit(5)
        return "pyannote"
    if requested == "resemblyzer":
        if not env["resemblyzer"]:
            print("ERROR: resemblyzer не установлен. Поставь: pip install resemblyzer scikit-learn", file=sys.stderr)
            sys.exit(4)
        return "resemblyzer"
    # auto: pyannote приоритетнее, если всё на месте
    if env["pyannote"] and env["hf_token"]:
        return "pyannote"
    if env["resemblyzer"]:
        return "resemblyzer"
    print(
        "ERROR: ни один диаризатор не настроен.\n"
        "  Поставь хотя бы один:\n"
        "    pip install resemblyzer scikit-learn         # без регистраций\n"
        "    pip install pyannote.audio + HF_TOKEN        # точнее\n"
        "  Или запусти без --diarize.",
        file=sys.stderr,
    )
    sys.exit(4)


# -------------- МОДЕЛЬ ДАННЫХ --------------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
            "speaker": self.speaker,
            "words": self.words,
        }


# -------------- ТРАНСКРИПЦИЯ --------------

def transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    language: str,
    word_timestamps: bool,
    compute_type: str,
    beam_size: int,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
) -> list[Segment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: pip install faster-whisper", file=sys.stderr)
        sys.exit(3)

    print(f"[faster-whisper] модель '{model_name}', beam_size={beam_size}, compute_type={compute_type}", flush=True)
    if not condition_on_previous_text:
        print("[faster-whisper] condition_on_previous_text=False (борьба с залипаниями)", flush=True)
    if initial_prompt:
        print(f"[faster-whisper] initial_prompt: {initial_prompt[:60]}…" if len(initial_prompt) > 60 else f"[faster-whisper] initial_prompt: {initial_prompt}", flush=True)
    model = WhisperModel(model_name, device="auto", compute_type=compute_type)

    print("[faster-whisper] транскрибирую…", flush=True)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language if language != "auto" else None,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    print(f"[faster-whisper] язык: {info.language} (p={info.language_probability:.2f}), "
          f"длительность: {fmt_ts(info.duration)}", flush=True)

    out: list[Segment] = []
    last_progress = -30.0
    for s in segments_iter:
        words = []
        if word_timestamps and s.words:
            words = [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word} for w in s.words]
        out.append(Segment(start=s.start, end=s.end, text=s.text, words=words))
        if s.end - last_progress >= 30:
            print(f"  …{fmt_ts(s.end)}", flush=True)
            last_progress = s.end
    return out


def transcribe_whisper_cpp(
    audio_path: Path,
    model_name: str,
    language: str,
    word_timestamps: bool,
    beam_size: int,
    bin_path: str,
    models_dir: str,
    initial_prompt: Optional[str] = None,
    condition_on_previous_text: bool = True,
) -> list[Segment]:
    """Run whisper.cpp via the whisper-cli binary; parse its JSON output."""
    model_path = whisper_cpp_model_path(models_dir, model_name)
    if not model_path:
        print(
            f"ERROR: ggml-{model_name}.bin не найден в {models_dir}.\n"
            f"  Скачай: cd $(dirname $(dirname {bin_path}))/.. && "
            f"bash models/download-ggml-model.sh {model_name}",
            file=sys.stderr,
        )
        sys.exit(3)

    print(f"[whisper.cpp] {Path(bin_path).name} model={model_name} beam={beam_size}", flush=True)
    if initial_prompt:
        prefix = initial_prompt[:60] + "…" if len(initial_prompt) > 60 else initial_prompt
        print(f"[whisper.cpp] initial_prompt: {prefix}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        out_prefix = Path(tmp) / "out"
        cmd = [
            bin_path,
            "-m", model_path,
            "-f", str(audio_path),
            "-l", language if language != "auto" else "auto",
            "-bs", str(max(1, beam_size)),
            "-of", str(out_prefix),
            "-ojf",  # full JSON output
            "-osrt",  # SRT (we ignore but harmless)
            "--print-progress",
        ]
        if initial_prompt:
            cmd.extend(["--prompt", initial_prompt])
        if word_timestamps:
            cmd.append("-otoken")  # token-level (closest to word-level)
        if not condition_on_previous_text:
            # --max-context 0: запрещает прокидывать предыдущий текст как promt в новое окно.
            # Лечит залипания вида «Так./Видно?» после длинных тишин — критично для
            # длинных записей. Флаг `--no-context` в этой версии whisper-cli не существует.
            cmd.extend(["--max-context", "0"])
            print("[whisper.cpp] --max-context 0 (борьба с залипаниями после тишин)", flush=True)
        # Run, streaming stderr so user sees progress.
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            print(
                f"whisper-cli failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}",
                file=sys.stderr,
            )
            sys.exit(2)

        json_path = Path(str(out_prefix) + ".json")
        if not json_path.is_file():
            print(
                f"ERROR: whisper.cpp did not produce {json_path}\n"
                f"--- whisper-cli stderr (last 2000 chars) ---\n{proc.stderr[-2000:]}\n"
                f"--- whisper-cli stdout (last 2000 chars) ---\n{proc.stdout[-2000:]}",
                file=sys.stderr,
            )
            sys.exit(2)
        data = json.loads(json_path.read_text(encoding="utf-8"))

    out: list[Segment] = []
    for s in data.get("transcription", []):
        offs = s.get("offsets") or {}
        start_ms = offs.get("from", 0)
        end_ms = offs.get("to", 0)
        text = s.get("text", "").strip()
        if not text:
            continue
        out.append(Segment(start=start_ms / 1000.0, end=end_ms / 1000.0, text=text))
    return out


def transcribe_mlx_whisper(
    audio_path: Path,
    model_name: str,
    language: str,
    word_timestamps: bool,
    beam_size: int,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
) -> list[Segment]:
    try:
        import mlx_whisper
    except ImportError:
        print("ERROR: pip install mlx-whisper", file=sys.stderr)
        sys.exit(3)

    mlx_repo_map = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base",
        "small": "mlx-community/whisper-small",
        "medium": "mlx-community/whisper-medium",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    repo = mlx_repo_map.get(model_name, model_name)

    # mlx-whisper does not implement beam search (NotImplementedError on
    # decoding.py:437 as of mlx-whisper 0.4.x). Fall back to greedy with a
    # visible warning so users know what they're getting; the alternative —
    # crashing mid-run — is worse on long files.
    effective_beam_size = beam_size
    if beam_size and beam_size > 1:
        print(
            f"[mlx-whisper] WARNING: beam_size={beam_size} requested, but mlx-whisper "
            f"does not implement beam search. Falling back to greedy (beam=1). "
            f"For real beam search use --backend faster-whisper.",
            file=sys.stderr, flush=True,
        )
        effective_beam_size = 1

    print(f"[mlx-whisper] {repo}, beam_size={effective_beam_size}", flush=True)
    if not condition_on_previous_text:
        print("[mlx-whisper] condition_on_previous_text=False (борьба с залипаниями)", flush=True)
    if initial_prompt:
        prefix = initial_prompt[:60] + "…" if len(initial_prompt) > 60 else initial_prompt
        print(f"[mlx-whisper] initial_prompt: {prefix}", flush=True)

    decode_kwargs = {}

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo,
        language=None if language == "auto" else language,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        verbose=False,
        **decode_kwargs,
    )

    out: list[Segment] = []
    for s in result.get("segments", []):
        words = []
        if word_timestamps and s.get("words"):
            words = [{"start": round(w["start"], 3), "end": round(w["end"], 3), "word": w["word"]} for w in s["words"]]
        out.append(Segment(start=float(s["start"]), end=float(s["end"]), text=str(s["text"]), words=words))
    return out


# -------------- ДИАРИЗАЦИЯ: pyannote --------------

def diarize_pyannote(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: Optional[int],
) -> int:
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("ERROR: pip install pyannote.audio", file=sys.stderr)
        sys.exit(4)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN не выставлен (см. references/setup.md → pyannote)", file=sys.stderr)
        sys.exit(5)

    print("[pyannote] загружаю pipeline speaker-diarization-3.1…", flush=True)
    # pyannote.audio renamed `use_auth_token` → `token` in 4.x. Try new-style
    # first, fall back to legacy if running on 3.x.
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    # Move pipeline to GPU if available. pyannote 4.x defaults to CPU,
    # which on Apple Silicon is ~3× slower than MPS for the speaker
    # embedding pass. CUDA path covers Linux/Windows GPU users.
    try:
        import torch
        if torch.backends.mps.is_available():
            pipe.to(torch.device("mps"))
            print("[pyannote] using device: mps (Apple GPU)", flush=True)
        elif torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
            print("[pyannote] using device: cuda", flush=True)
    except Exception as e:
        print(f"[pyannote] device move skipped: {e}", flush=True)

    print("[pyannote] выполняю диаризацию (это самый медленный шаг)…", flush=True)
    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    # Pyannote 4.x uses torchcodec to load audio, which dlopens shared FFmpeg
    # libs (libavutil.X.dylib etc). Static FFmpeg builds (e.g. evermeet.cx)
    # don't ship those, so torchcodec fails with OSError. Detect that and
    # fall back to loading the WAV ourselves and passing the in-memory dict
    # form pyannote also accepts.
    try:
        diarization = pipe(str(audio_path), **kwargs)
    except (OSError, RuntimeError) as e:
        if "torchcodec" not in str(e) and "libav" not in str(e):
            raise
        import wave, torch, numpy as np
        print(
            "[pyannote] torchcodec не загрузил FFmpeg-библиотеки — "
            "загружаю WAV вручную и передаю как dict",
            flush=True,
        )
        with wave.open(str(audio_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sample_width != 2:
            raise SystemExit(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit")
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, n_channels).T if n_channels > 1 else samples.reshape(1, -1)
        diarization = pipe(
            {"waveform": torch.from_numpy(samples), "sample_rate": sample_rate},
            **kwargs,
        )

    # pyannote 4.x returns DiarizeOutput; .speaker_diarization is the
    # Annotation. pyannote 3.x returns the Annotation directly.
    diarization = getattr(diarization, "speaker_diarization", diarization)
    spans = []
    for turn, _, label in diarization.itertracks(yield_label=True):
        spans.append((turn.start, turn.end, label))
    spans.sort(key=lambda x: x[0])
    speakers = sorted({s[2] for s in spans})
    print(f"[pyannote] найдено спикеров: {len(speakers)}", flush=True)

    # Привязываем спикера к каждому сегменту по максимальному перекрытию
    label_remap = {}
    next_id = 0
    for seg in segments:
        best_label = None
        best_overlap = 0.0
        for s_start, s_end, label in spans:
            overlap = max(0.0, min(seg.end, s_end) - max(seg.start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_label is not None:
            if best_label not in label_remap:
                label_remap[best_label] = f"SPEAKER_{next_id:02d}"
                next_id += 1
            seg.speaker = label_remap[best_label]

    return len(label_remap)


# -------------- ДИАРИЗАЦИЯ: resemblyzer --------------

def diarize_resemblyzer(
    audio_path: Path,
    segments: list[Segment],
    num_speakers: Optional[int],
    max_speakers: int = 6,
) -> int:
    try:
        import numpy as np
        from resemblyzer import VoiceEncoder, preprocess_wav
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError as e:
        print(f"ERROR: pip install resemblyzer scikit-learn (нет: {e.name})", file=sys.stderr)
        sys.exit(4)

    print("[resemblyzer] загружаю voice encoder…", flush=True)
    encoder = VoiceEncoder(verbose=False)
    wav = preprocess_wav(str(audio_path))
    sr = 16000

    print(f"[resemblyzer] эмбеддинги для {len(segments)} сегментов…", flush=True)
    embeds, indices, skipped = [], [], 0
    for i, seg in enumerate(segments):
        if seg.end - seg.start < 0.6:
            skipped += 1
            continue
        a, b = int(seg.start * sr), int(seg.end * sr)
        chunk = wav[a:b]
        if len(chunk) < int(0.4 * sr):
            skipped += 1
            continue
        try:
            embeds.append(encoder.embed_utterance(chunk))
            indices.append(i)
        except Exception as e:
            skipped += 1
            print(f"  [warn] сегмент {i}: {e}", file=sys.stderr)

    if skipped:
        print(f"[resemblyzer] пропущено коротких: {skipped}", flush=True)
    if not embeds:
        print("[resemblyzer] нет валидных эмбеддингов — пропускаю диаризацию", file=sys.stderr)
        return 0

    embeds_np = np.vstack(embeds)

    if num_speakers is not None:
        k = max(1, num_speakers)
        if k == 1:
            labels = np.zeros(len(embeds_np), dtype=int)
        else:
            labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embeds_np)
        chosen_k = k
        print(f"[resemblyzer] n_speakers={k} (задано)", flush=True)
    else:
        print(f"[resemblyzer] подбираю k в 2..{max_speakers} по silhouette…", flush=True)
        best_k, best_score, best_labels = 1, -1.0, np.zeros(len(embeds_np), dtype=int)
        upper = min(max_speakers, max(2, len(embeds_np) - 1))
        for k in range(2, upper + 1):
            try:
                lbl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embeds_np)
                if len(set(lbl)) < 2:
                    continue
                score = silhouette_score(embeds_np, lbl, metric="cosine")
                print(f"  k={k}: silhouette={score:.3f}", flush=True)
                if score > best_score:
                    best_score, best_k, best_labels = score, k, lbl
            except Exception as e:
                print(f"  k={k}: ошибка ({e})", flush=True)
        if best_score < 0.10:
            print(f"[resemblyzer] silhouette={best_score:.3f} низкий — считаю 1 спикера", flush=True)
            labels, chosen_k = np.zeros(len(embeds_np), dtype=int), 1
        else:
            labels, chosen_k = best_labels, best_k
            print(f"[resemblyzer] выбрано {chosen_k} спикер(ов), silhouette={best_score:.3f}", flush=True)

    label_to_speaker = {}
    next_id = 0
    for idx, lbl in zip(indices, labels):
        if lbl not in label_to_speaker:
            label_to_speaker[lbl] = f"SPEAKER_{next_id:02d}"
            next_id += 1
        segments[idx].speaker = label_to_speaker[lbl]

    # пробросить пропущенным короткие сегменты — спикер от соседа
    for i, seg in enumerate(segments):
        if seg.speaker is None:
            for j in range(i - 1, -1, -1):
                if segments[j].speaker:
                    seg.speaker = segments[j].speaker
                    break
            else:
                for j in range(i + 1, len(segments)):
                    if segments[j].speaker:
                        seg.speaker = segments[j].speaker
                        break

    return chosen_k


# -------------- АРТЕФАКТЫ --------------

def write_json(segments: list[Segment], duration: float, meta: dict, out_path: Path) -> None:
    payload = {
        "metadata": {
            "duration_seconds": round(duration, 3),
            "duration_hms": fmt_ts(duration),
            **meta,
        },
        "segments": [s.to_dict() for s in segments],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_path.name}", flush=True)


def write_markdown(segments: list[Segment], meta: dict, out_path: Path) -> None:
    lines = [
        f"# Транскрипт: {meta.get('source_file', '')}",
        "",
        f"- Длительность: {meta.get('duration_hms', '?')}",
        f"- Модель: {meta.get('model', '?')} (preset={meta.get('preset', '?')}, backend={meta.get('backend', '?')})",
        f"- Язык: {meta.get('language', '?')}",
    ]
    if meta.get("speakers_detected"):
        lines.append(f"- Спикеров найдено: {meta['speakers_detected']} (диаризатор: {meta.get('diarizer')})")
    lines.extend(["", "---", ""])

    last_speaker = object()
    for seg in segments:
        ts = fmt_ts(seg.start)
        text = seg.text.strip()
        if seg.speaker:
            if seg.speaker != last_speaker:
                lines.append("")
                lines.append(f"**{seg.speaker}**")
                last_speaker = seg.speaker
            lines.append(f"[{ts}] {text}")
        else:
            lines.append(f"[{ts}] {text}")

    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"[write] {out_path.name}", flush=True)


def write_srt(segments: list[Segment], out_path: Path) -> None:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_ts_srt(seg.start)} --> {fmt_ts_srt(seg.end)}")
        prefix = f"[{seg.speaker}] " if seg.speaker else ""
        lines.append(prefix + seg.text.strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write] {out_path.name}", flush=True)


# -------------- ОЦЕНКА И ВЫБОР --------------

def build_estimate_report(duration: float, env: dict, diarizer: Optional[str]) -> dict:
    """JSON с длительностью и оценкой по всем пресетам и доступным бэкендам."""
    # Order matters — first backend in this list is the auto-default if user
    # doesn't pick. whisper.cpp is fastest on Apple Silicon (Metal) AND
    # supports beam search, so it leads.
    available_backends = []
    if env.get("whisper_cpp_bin"):
        available_backends.append("whisper-cpp")
    if env["mlx_whisper"]:
        available_backends.append("mlx-whisper")
    if env["faster_whisper"]:
        available_backends.append("faster-whisper")

    options = []
    for backend in available_backends:
        for preset_name, cfg in PRESETS.items():
            # Skip presets that require a model file we don't have (whisper.cpp).
            if backend == "whisper-cpp":
                if not whisper_cpp_model_path(env.get("whisper_cpp_models_dir"), cfg["model"]):
                    continue
            est_sec = estimate_seconds(duration, backend, preset_name, diarizer)
            options.append({
                "backend": backend,
                "preset": preset_name,
                "model": cfg["model"],
                "beam_size": cfg["beam_size"],
                "estimate_seconds": round(est_sec, 1),
                "estimate_human": fmt_estimate(est_sec),
                "description": cfg["description"],
            })
    return {
        "duration_seconds": round(duration, 3),
        "duration_hms": fmt_ts(duration),
        "diarizer": diarizer,
        "available_backends": available_backends,
        "options": options,
    }


def print_estimate_table(report: dict) -> None:
    """Красивая табличка для человека или Claude."""
    print()
    print(f"📂 Длительность: {report['duration_hms']}")
    if report["diarizer"]:
        print(f"🎭 Диаризация:   {report['diarizer']}")
    else:
        print("🎭 Диаризация:   нет")
    print()
    print(f"{'#':<3} {'бэкенд':<16} {'пресет':<10} {'модель':<18} {'beam':<5} {'оценка':<10}")
    print("─" * 70)
    for i, opt in enumerate(report["options"], 1):
        print(f"{i:<3} {opt['backend']:<16} {opt['preset']:<10} "
              f"{opt['model']:<18} {opt['beam_size']:<5} {opt['estimate_human']:<10}")
    print()


def auto_default_backend(env: dict) -> str:
    """The implicit default when --backend / --preset not given. whisper.cpp
    on Apple Silicon is fastest AND supports beam search, so prefer it."""
    if env.get("whisper_cpp_bin"):
        return "whisper-cpp"
    if env["mlx_whisper"]:
        return "mlx-whisper"
    return "faster-whisper"


def interactive_select(report: dict) -> tuple[str, str]:
    """Спрашивает пользователя выбор. Возвращает (backend, preset)."""
    print_estimate_table(report)
    while True:
        try:
            answer = input(f"Выбери вариант [1-{len(report['options'])}] (Enter = fast на самом быстром бэкенде): ").strip()
        except EOFError:
            answer = ""
        if not answer:
            # Default: fast preset on the fastest available backend (whisper.cpp > mlx > faster).
            order = ["whisper-cpp", "mlx-whisper", "faster-whisper"]
            preferred = next((b for b in order if b in report["available_backends"]),
                             report["available_backends"][0])
            return preferred, "fast"
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(report["options"]):
                opt = report["options"][idx]
                return opt["backend"], opt["preset"]
        print(f"Не понял. Введи число от 1 до {len(report['options'])} или Enter.")


# -------------- MAIN --------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Локальная транскрипция аудио/видео через Whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Путь к видео или аудио.")
    parser.add_argument(
        "--preset", choices=["quality", "balanced", "fast"], default=None,
        help="Пресет качества/скорости. Если не задан — спрашивает (TTY) или balanced.",
    )
    parser.add_argument(
        "--backend", choices=["whisper-cpp", "mlx-whisper", "faster-whisper"], default=None,
        help="Бэкенд транскрипции. Если не задан — выбирается автоматически "
             "(приоритет: whisper-cpp > mlx-whisper > faster-whisper).",
    )
    parser.add_argument("--model", default=None, help="Переопределить модель из пресета.")
    parser.add_argument("--beam-size", type=int, default=None, help="Переопределить beam_size из пресета.")
    parser.add_argument("--language", default="ru", help="Код языка ISO-639-1 или 'auto' (по умолчанию: ru).")
    parser.add_argument(
        "--compute-type", default="auto",
        help="Для faster-whisper: int8/int8_float16/float16/float32/auto.",
    )
    parser.add_argument("--diarize", action="store_true", help="Включить диаризацию.")
    parser.add_argument(
        "--diarizer", choices=["auto", "pyannote", "resemblyzer", "none"], default="auto",
        help="Какой диаризатор использовать. auto = pyannote если есть HF_TOKEN, иначе resemblyzer.",
    )
    parser.add_argument("--num-speakers", type=int, default=None, help="Точное число спикеров (если знаешь).")
    parser.add_argument("--max-speakers", type=int, default=6, help="Верхняя граница автоопределения (resemblyzer).")
    parser.add_argument("--word-timestamps", action="store_true", help="Тайм-коды на уровне слов.")
    parser.add_argument(
        "--no-condition-on-previous-text", action="store_true",
        help="Не передавать предыдущий текст модели — лечит залипания на длинных тихих участках.",
    )
    parser.add_argument(
        "--initial-prompt", default=None,
        help="Подсказка модели в начале (имена участников, термины проекта). "
             "Сильно улучшает распознавание имён и жаргона.",
    )
    parser.add_argument("--keep-audio", action="store_true", help="Не удалять промежуточный WAV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Куда складывать артефакты.")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Только напечатать оценку времени (JSON) и выйти.")
    parser.add_argument("--analyze", action="store_true",
                        help="Preflight-анализ: длительность + аудио-свойства + статистика тишин + "
                             "рекомендация пресета. Выводит JSON и выходит.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Не спрашивать ничего интерактивно (берёт --preset или balanced).")
    parser.add_argument("--json", action="store_true", help="С --estimate-only выводить чистый JSON.")

    args = parser.parse_args()

    input_path: Path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"ERROR: файл не найден: {input_path}", file=sys.stderr)
        return 1

    ext = input_path.suffix.lower()
    if ext not in AUDIO_EXTS | VIDEO_EXTS:
        print(f"WARNING: расширение {ext} не в списке известных. Пробую всё равно…", file=sys.stderr)

    check_ffmpeg()
    env = detect_environment()

    if not (env["faster_whisper"] or env["mlx_whisper"] or env.get("whisper_cpp_bin")):
        print(
            "ERROR: ни один бэкенд whisper не установлен.\n"
            "  whisper.cpp (Metal на Apple Silicon, рекомендуется): см. references/setup.md\n"
            "  pip install mlx-whisper              # альтернатива на Apple Silicon\n"
            "  pip install faster-whisper           # кросс-платформенный CPU",
            file=sys.stderr,
        )
        return 1

    # Длительность файла — для оценки и метаданных
    duration = probe_duration(input_path)
    if duration <= 0:
        print("WARNING: не удалось определить длительность файла", file=sys.stderr)

    # Какой диаризатор будет фактически использоваться (для оценки)
    effective_diarizer: Optional[str] = None
    if args.diarize:
        effective_diarizer = resolve_diarizer(args.diarizer, env)

    # Режим preflight-анализ (длительность + аудио + тишины + рекомендация)
    if args.analyze:
        analysis = build_analysis(input_path, duration, env)
        report = build_estimate_report(duration, env, effective_diarizer)
        combined = {**report, "analysis": analysis}
        print(json.dumps(combined, ensure_ascii=False, indent=2))
        return 0

    # Режим только-оценка
    if args.estimate_only:
        report = build_estimate_report(duration, env, effective_diarizer)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_estimate_table(report)
        return 0

    # Определяем backend и preset
    backend = args.backend
    preset_name = args.preset

    # Интерактивный выбор: только если есть TTY и пользователь ничего не передал и не --yes
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    if not args.yes and preset_name is None and backend is None and is_tty:
        report = build_estimate_report(duration, env, effective_diarizer)
        backend, preset_name = interactive_select(report)
    else:
        if preset_name is None:
            preset_name = "fast"  # bench: fast почти не отличается от balanced/quality на финальном отчёте
        if backend is None:
            backend = auto_default_backend(env)

    if backend == "whisper-cpp" and not env.get("whisper_cpp_bin"):
        print("ERROR: --backend whisper-cpp, но whisper-cli не найден. См. references/setup.md", file=sys.stderr)
        return 1
    if backend == "mlx-whisper" and not env["mlx_whisper"]:
        print("ERROR: --backend mlx-whisper, но пакет не установлен. pip install mlx-whisper", file=sys.stderr)
        return 1
    if backend == "faster-whisper" and not env["faster_whisper"]:
        print("ERROR: --backend faster-whisper, но пакет не установлен. pip install faster-whisper", file=sys.stderr)
        return 1

    preset = PRESETS[preset_name]
    model = args.model or preset["model"]
    beam_size = args.beam_size if args.beam_size is not None else preset["beam_size"]

    # Финальная сводка перед запуском
    final_estimate = estimate_seconds(duration, backend, preset_name, effective_diarizer)
    print()
    print(f"📂 {input_path.name}  ({fmt_ts(duration)})")
    print(f"⚙️  preset={preset_name}  backend={backend}  model={model}  beam={beam_size}")
    if effective_diarizer:
        print(f"🎭 diarize={effective_diarizer}")
    print(f"⏱  ожидаемое время: {fmt_estimate(final_estimate)}")
    print()

    out_dir = (args.output_dir or input_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    json_path = out_dir / f"{stem}.transcript.json"
    md_path = out_dir / f"{stem}.transcript.md"
    srt_path = out_dir / f"{stem}.transcript.srt"

    started = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / f"{stem}.wav"
        extract_audio(input_path, wav_path)
        if duration <= 0:
            duration = probe_duration(wav_path)

        word_ts = args.word_timestamps or (effective_diarizer == "pyannote")
        condition = not args.no_condition_on_previous_text
        if backend == "whisper-cpp":
            segments = transcribe_whisper_cpp(
                wav_path, model, args.language, word_ts, beam_size,
                bin_path=env["whisper_cpp_bin"],
                models_dir=env["whisper_cpp_models_dir"],
                initial_prompt=args.initial_prompt,
                condition_on_previous_text=condition,
            )
        elif backend == "faster-whisper":
            segments = transcribe_faster_whisper(
                wav_path, model, args.language, word_ts, args.compute_type, beam_size,
                condition_on_previous_text=condition,
                initial_prompt=args.initial_prompt,
            )
        else:
            segments = transcribe_mlx_whisper(
                wav_path, model, args.language, word_ts, beam_size,
                condition_on_previous_text=condition,
                initial_prompt=args.initial_prompt,
            )

        speakers_detected: Optional[int] = None
        if effective_diarizer == "pyannote":
            speakers_detected = diarize_pyannote(wav_path, segments, args.num_speakers)
        elif effective_diarizer == "resemblyzer":
            speakers_detected = diarize_resemblyzer(wav_path, segments, args.num_speakers, args.max_speakers)

        if args.keep_audio:
            kept = out_dir / f"{stem}.wav"
            shutil.copy2(wav_path, kept)
            print(f"[write] {kept.name}", flush=True)

    elapsed = time.time() - started
    rtf_actual = elapsed / duration if duration > 0 else 0

    meta = {
        "source_file": input_path.name,
        "preset": preset_name,
        "backend": backend,
        "model": model,
        "beam_size": beam_size,
        "language": args.language,
        "diarizer": effective_diarizer,
        "speakers_detected": speakers_detected,
        "duration_hms": fmt_ts(duration),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hms": fmt_ts(elapsed),
        "rtf_actual": round(rtf_actual, 3),
    }
    write_json(segments, duration, meta, json_path)
    write_markdown(segments, meta, md_path)
    write_srt(segments, srt_path)

    print(f"\n✅ Готово за {fmt_ts(elapsed)} (оценка была {fmt_estimate(final_estimate)}, фактический RTF={rtf_actual:.2f}).")
    print(f"   {md_path}")
    print(f"   {json_path}")
    print(f"   {srt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
