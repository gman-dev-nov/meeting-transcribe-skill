#!/usr/bin/env python3
"""
Longform-транскрипция через GigaAM без HF-токена.

Штатный `transcribe_longform` в старых версиях gigaam тянет
pyannote/segmentation-3.0 (gated → нужен HF_TOKEN). Здесь тот же алгоритм
нарезки, но VAD — локальный Silero (веса в пакете silero-vad).

Свежие версии gigaam поддерживают `vad_backend="silero"` штатно — тогда
скрипт просто пробрасывает параметр. Для старых версий модуль
gigaam.vad_utils подменяется шимом на Silero: pyannote не используется и
не скачивается (а если pyannote не установлен — и не импортируется вовсе).

Артефакты пишутся в формате, совместимом со скиллом meeting-transcribe:
    <stem>.transcript.json / .transcript.md / .transcript.srt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch


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


def native_silero_supported() -> bool:
    """True, если установленный gigaam умеет vad_backend="silero" штатно.

    В новых версиях silero-бэкенд живёт отдельным модулем
    gigaam.silero_vad_utils; find_spec не импортирует ни pyannote, ни silero.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("gigaam.silero_vad_utils") is not None
    except ModuleNotFoundError:
        return False


def install_silero_vad_shim() -> None:
    """Регистрирует поддельный gigaam.vad_utils с Silero вместо pyannote."""
    import gigaam
    from gigaam.preprocess import load_audio

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError:
        print(
            "ERROR: не установлен пакет silero-vad (нужен для нарезки без HF-токена).\n"
            "  pip install silero-vad   # в тот же venv, где стоит gigaam\n"
            "  подробнее: references/setup.md → «GigaAM v3»",
            file=sys.stderr,
        )
        sys.exit(2)

    def segment_audio_file(
        wav_file,
        sr,
        max_duration: float = 22.0,
        min_duration: float = 15.0,
        strict_limit_duration: float = 30.0,
        new_chunk_threshold: float = 0.2,
        device=torch.device("cpu"),
    ):
        audio = load_audio(wav_file)
        print(f"[vad] аудио загружено: {audio.shape[0] / sr:.1f} сек", flush=True)

        vad = load_silero_vad()
        spans = get_speech_timestamps(
            audio, vad, sampling_rate=int(sr), return_seconds=True
        )
        print(f"[vad] speech-сегментов: {len(spans)}", flush=True)

        segments: list = []
        boundaries: list = []
        curr_duration = 0.0
        curr_start = 0.0
        curr_end = 0.0

        def _update(curr_start: float, curr_end: float, curr_duration: float) -> None:
            # Куски длиннее strict_limit_duration режем равномерно — энкодер
            # GigaAM обучен на ≤25 сек, на длинном контексте деградирует.
            if curr_duration > strict_limit_duration:
                n = int(curr_duration / strict_limit_duration) + 1
                step = curr_duration / n
                curr_end = curr_start + step
                for _ in range(n - 1):
                    segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
                    boundaries.append((curr_start, curr_end))
                    curr_start = curr_end
                    curr_end += step
            segments.append(audio[int(curr_start * sr) : int(curr_end * sr)])
            boundaries.append((curr_start, curr_end))

        total = audio.shape[0] / sr
        for span in spans:
            start = max(0.0, float(span["start"]))
            end = min(total, float(span["end"]))
            if curr_duration == 0.0:
                curr_start = start
            elif curr_duration > new_chunk_threshold and (
                curr_duration + (end - curr_end) > max_duration
                or curr_duration > min_duration
            ):
                _update(curr_start, curr_end, curr_duration)
                curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration > new_chunk_threshold:
            _update(curr_start, curr_end, curr_duration)

        print(f"[vad] чанков для ASR: {len(segments)}", flush=True)
        return segments, boundaries

    mod = types.ModuleType("gigaam.vad_utils")
    mod.segment_audio_file = segment_audio_file  # type: ignore[attr-defined]
    sys.modules["gigaam.vad_utils"] = mod
    gigaam.vad_utils = mod  # type: ignore[attr-defined]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--model", default="v3_e2e_rnnt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--fp16-encoder", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    src = args.input.expanduser().resolve()
    out_dir = (args.output_dir or src.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem + args.suffix

    use_native = native_silero_supported()
    if use_native:
        print("[vad] gigaam поддерживает silero штатно (vad_backend)", flush=True)
    else:
        install_silero_vad_shim()
    import gigaam

    print(f"[gigaam] загружаю модель {args.model} на {args.device}…", flush=True)
    model = gigaam.load_model(
        args.model, device=args.device, fp16_encoder=args.fp16_encoder
    )

    lf_kwargs = {"word_timestamps": True, "fr_batch_size": args.batch_size}
    if use_native:
        lf_kwargs["vad_backend"] = "silero"

    started = time.time()
    result = model.transcribe_longform(str(src), **lf_kwargs)
    elapsed = time.time() - started

    segments = [
        {
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
            "speaker": None,
            "words": [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.text}
                for w in (s.words or [])
            ],
        }
        for s in result.segments
        if s.text.strip()
    ]
    duration = segments[-1]["end"] if segments else 0.0

    meta = {
        "source_file": src.name,
        "engine": "gigaam",
        "model": args.model,
        "device": args.device,
        "fp16_encoder": args.fp16_encoder,
        "vad": "silero-vad",
        "language": "ru",
        "duration_hms": fmt_ts(duration),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_hms": fmt_ts(elapsed),
        "rtf_actual": round(elapsed / duration, 3) if duration else None,
        "segments_count": len(segments),
    }

    json_path = out_dir / f"{stem}.transcript.json"
    md_path = out_dir / f"{stem}.transcript.md"
    srt_path = out_dir / f"{stem}.transcript.srt"

    json_path.write_text(
        json.dumps(
            {"metadata": {"duration_seconds": round(duration, 3), **meta},
             "segments": segments},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Транскрипт: {src.name}",
        "",
        f"- Длительность: {meta['duration_hms']}",
        f"- Модель: {args.model} (движок=gigaam, VAD=silero, device={args.device})",
        "- Язык: ru",
        "",
        "---",
        "",
    ]
    lines += [f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments]
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    srt = []
    for i, s in enumerate(segments, 1):
        srt += [str(i), f"{fmt_ts_srt(s['start'])} --> {fmt_ts_srt(s['end'])}", s["text"], ""]
    srt_path.write_text("\n".join(srt), encoding="utf-8")

    print(
        f"\n✅ GigaAM готово за {fmt_ts(elapsed)} "
        f"(RTF={meta['rtf_actual']}), сегментов: {len(segments)}"
    )
    for p in (md_path, json_path, srt_path):
        print(f"   {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
