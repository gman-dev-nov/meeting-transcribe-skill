#!/usr/bin/env python3
"""
Setup wizard для meeting-transcribe.

Проверяет окружение для двойного GigaAM + Whisper workflow, помогает выбрать
диаризатор, рассказывает про особенности длинного/шумного аудио.

Запуск:
    python3 scripts/setup_check.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def venv_python(name: str) -> Path:
    if os.name == "nt":
        return Path.home() / ".venvs" / name / "Scripts" / "python.exe"
    return Path.home() / ".venvs" / name / "bin" / "python"


def resolve_whisper_python() -> Path:
    """Тот же выбор интерпретатора, что делает dual_transcribe.py.

    Копия намеренная: wizard обязан работать, даже если рядом нет
    dual_transcribe.py. Совпадение правил закреплено тестом.
    """
    configured = os.environ.get("WHISPER_PYTHON")
    if configured:
        return Path(configured).expanduser()
    candidate = venv_python("whisper")
    if candidate.is_file():
        return candidate
    return Path(sys.executable)


def probe_imports(python: Path, modules: dict) -> dict:
    """Проверить импорты в чужом интерпретаторе, а не в своём.

    Wizard могут запустить системным python3, пока пакеты лежат в venv —
    тогда проверка «у себя» врёт про отсутствие бэкенда.
    """
    code = (
        "import importlib.util, json, sys\n"
        "names = json.loads(sys.argv[1])\n"
        "print(json.dumps({k: all(importlib.util.find_spec(m) is not None for m in v)\n"
        "                  for k, v in names.items()}))"
    )
    empty = {key: False for key in modules}
    try:
        probe = subprocess.run(
            [str(python), "-c", code, json.dumps(modules)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return empty
    if probe.returncode != 0 or not probe.stdout.strip():
        return empty
    try:
        result = json.loads(probe.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return empty
    return {key: bool(result.get(key)) for key in modules}


def header(text: str) -> None:
    print()
    print("=" * 64)
    print(text)
    print("=" * 64)


def check(label: str, ok: bool, hint: str = "") -> None:
    mark = "✅" if ok else "❌"
    print(f"  {mark} {label}")
    if not ok and hint:
        print(f"      → {hint}")


def warn(label: str, hint: str = "") -> None:
    print(f"  ⚠️  {label}")
    if hint:
        print(f"      → {hint}")


def main() -> int:
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")
    )

    header("Meeting Transcribe — проверка окружения")

    # ---- ffmpeg ----
    print("\n1) ffmpeg")
    has_ffmpeg = shutil.which("ffmpeg") is not None
    has_ffprobe = shutil.which("ffprobe") is not None
    check("ffmpeg в PATH", has_ffmpeg,
          "macOS: brew install ffmpeg | Linux: sudo apt install ffmpeg | "
          "без brew: статические ffmpeg+ffprobe → ~/.local/bin "
          "(references/setup.md → «ffmpeg без Homebrew»)")
    check("ffprobe в PATH", has_ffprobe, "установи тот же пакет ffmpeg")

    # ---- whisper backends ----
    print("\n2) Whisper-бэкенды")

    # whisper.cpp — рекомендуемый дефолт на Apple Silicon: самый быстрый + честный beam search
    wcpp_paths = [
        os.environ.get("WHISPER_CPP_BIN", ""),
        f"{os.environ.get('WHISPER_CPP_HOME', '')}/build/bin/whisper-cli",
        f"{Path.home()}/whisper.cpp/build/bin/whisper-cli",
        f"{Path.home()}/.local/share/whisper.cpp/build/bin/whisper-cli",
        "/usr/local/bin/whisper-cli",
        "/opt/homebrew/bin/whisper-cli",
    ]
    has_wcpp_bin = any(p and Path(p).is_file() for p in wcpp_paths) or shutil.which("whisper-cli") is not None

    def _has_wcpp_model(name: str) -> bool:
        candidates = [
            f"{Path.home()}/whisper.cpp/models/ggml-{name}.bin",
            f"{os.environ.get('WHISPER_CPP_HOME', '')}/models/ggml-{name}.bin",
            f"{Path.home()}/.local/share/whisper.cpp/models/ggml-{name}.bin",
        ]
        return any(p and Path(p).is_file() for p in candidates)

    wcpp_has_v3 = has_wcpp_bin and _has_wcpp_model("large-v3")
    wcpp_usable = has_wcpp_bin and wcpp_has_v3

    check("whisper.cpp (рекомендуемый дефолт на Apple Silicon)", has_wcpp_bin,
          "pip install cmake && git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp && "
          "cd ~/whisper.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && "
          "cmake --build build --config Release -j 8")
    if has_wcpp_bin:
        if wcpp_has_v3:
            check("  ggml-large-v3.bin (максимальная модель, ~3 ГБ)", True)
        else:
            warn("  ggml-large-v3.bin не скачан — без него скилл не запускает Whisper",
                 "скачать максимальную модель: "
                 "cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3")

    whisper_python = resolve_whisper_python()
    whisper_probe = probe_imports(
        whisper_python,
        {
            "mlx": ["mlx_whisper"],
            "faster": ["faster_whisper"],
            "resemblyzer": ["resemblyzer", "sklearn"],
            "pyannote": ["pyannote.audio"],
        },
    )
    has_mlx = whisper_probe["mlx"]
    has_fw = whisper_probe["faster"]
    print(f"  ℹ️  Python для Whisper: {whisper_python}")
    print("      (переопределяется переменной WHISPER_PYTHON)")

    if is_apple_silicon:
        check("mlx-whisper (альтернатива; нет beam search)", has_mlx,
              "pip install mlx-whisper")
        check("faster-whisper (опционально)", has_fw,
              "нужен только если whisper.cpp недоступен или нужен реальный beam search без wcpp")
    else:
        check("faster-whisper (нужен на не-Apple-Silicon)", has_fw,
              "pip install faster-whisper")
        warn("mlx-whisper недоступен на этой платформе",
             "пакет работает только на Apple Silicon (M-чипы), пропусти этот пункт")

    if not (wcpp_usable or has_mlx or has_fw):
        print("\n  ⚠️  Без бэкенда транскрипция не запустится.")

    # ---- GigaAM in its dedicated venv ----
    print("\n3) GigaAM (первая половина двойной сверки русской речи)")
    default_gigaam_python = (
        Path.home() / ".venvs/asr/Scripts/python.exe"
        if os.name == "nt"
        else Path.home() / ".venvs/asr/bin/python"
    )
    gigaam_python = Path(
        os.environ.get("GIGAAM_PYTHON", str(default_gigaam_python))
    ).expanduser()
    has_gigaam_python = gigaam_python.is_file()
    check(
        f"Python для GigaAM: {gigaam_python}",
        has_gigaam_python,
        "создай ~/.venvs/asr и установи GigaAM по references/setup.md",
    )
    has_gigaam = False
    if has_gigaam_python:
        try:
            probe = subprocess.run(
                [
                    str(gigaam_python),
                    "-c",
                    "import gigaam, torch, silero_vad; print('ok')",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            has_gigaam = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            has_gigaam = False
        check(
            "gigaam + torch + silero-vad доступны",
            has_gigaam,
            "~/.venvs/asr/bin/pip install "
            "'gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git' "
            "silero-vad",
        )

    # ---- diarizers ----
    print("\n4) Диаризация (опционально, для разметки спикеров)")
    has_resemblyzer = whisper_probe["resemblyzer"]
    has_pyannote = whisper_probe["pyannote"]

    has_token = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    check("resemblyzer + scikit-learn (полностью локально)", has_resemblyzer,
          "pip install resemblyzer scikit-learn")
    check("pyannote.audio (точнее, нужны faster-whisper + HF_TOKEN)", has_pyannote,
          "pip install pyannote.audio  (см. ниже про HF_TOKEN)")
    check("HF_TOKEN выставлен (только для pyannote)", has_token,
          "export HF_TOKEN=hf_xxx  (см. ниже)")

    if has_pyannote and not has_fw:
        warn("pyannote установлен, но faster-whisper — нет.",
             "Pyannote лучше работает с word-level timestamps от faster-whisper. "
             "Поставь: pip install faster-whisper")

    # ---- summary ----
    header("Что у тебя получится по умолчанию")

    # Приоритет в transcribe.py: whisper-cpp → faster-whisper → mlx-whisper.
    whisper_quality_usable = wcpp_has_v3 or has_mlx or has_fw
    if has_gigaam and whisper_quality_usable:
        print("  • Русская речь: параллельные GigaAM + Whisper с LLM-сверкой.")
    elif whisper_quality_usable:
        print("  • Русская речь: ⚠️ только Whisper; двойная сверка недоступна без GigaAM.")
    elif has_gigaam and has_wcpp_bin:
        print("  • Русская речь: ⚠️ для двойной сверки не хватает Whisper large-v3.")
    else:
        print("  • Русская речь: ❌ нет полного набора GigaAM + Whisper.")

    if wcpp_usable:
        print("  • Транскрипция: whisper.cpp + максимальная large-v3.")
    elif has_mlx:
        print("  • Транскрипция: mlx-whisper (быстро, ~RTF 0.10 на M-серии).")
    elif has_fw:
        print("  • Транскрипция: faster-whisper (CPU, ~RTF 0.40 на M-серии).")
    else:
        print("  • Транскрипция: ❌ ничего. Поставь хотя бы один бэкенд.")

    if has_pyannote and has_token:
        print("  • Диаризация:   pyannote (приоритетно).")
    elif has_resemblyzer:
        print("  • Диаризация:   resemblyzer (локально, без токенов).")
    else:
        print("  • Диаризация:   не настроена. Запуски без --diarize всё равно работают.")

    # ---- HF_TOKEN setup ----
    if not has_token or not has_pyannote:
        header("Если хочешь поставить pyannote (точнее на сложных записях)")
        print("""
HF_TOKEN полностью бесплатный — это просто способ скачивать публичные модели.
Никаких списаний за использование, всё работает локально после загрузки.

1. Зарегистрируйся на huggingface.co (если ещё нет аккаунта).

2. Прими условия использования двух моделей (просто галочка, без оплаты):
   • https://hf.co/pyannote/speaker-diarization-3.1
   • https://hf.co/pyannote/segmentation-3.0

3. Создай read-токен:
   https://hf.co/settings/tokens → New token → role: read → Generate

4. Положи в окружение (zsh):
   echo 'export HF_TOKEN=hf_xxx_твой_токен' >> ~/.zshrc
   source ~/.zshrc

5. Поставь пакеты:
   pip install pyannote.audio faster-whisper

6. Перезапусти этот wizard, чтобы проверить.
""")

    # ---- Особенности и ограничения ----
    header("Особенности Whisper, которые стоит знать")
    print("""
🔁  ЗАЛИПАНИЯ И ГАЛЛЮЦИНАЦИИ

    Whisper иногда «зацикливается» — повторяет одну и ту же фразу,
    или генерирует мусор вроде «Subtitles by …», «Спасибо за просмотр»,
    «Продолжение следует». Чаще всего это происходит на:
    • длинных тишинах
    • музыкальных вставках
    • очень тихом / зашумлённом аудио
    • записях длиннее ~2 часов

    Что помогает:
    1) Использовать VAD (фильтр пауз). У faster-whisper он встроен и
       включён в этом скилле по умолчанию. У mlx-whisper VAD слабее —
       если на длинных записях видишь повторы, попробуй пересобрать
       аудио через ffmpeg + silenceremove (см. ниже).
    2) Передать --no-condition-on-previous-text. Модель перестаёт
       опираться на предыдущий контекст — повторы прекращаются, но
       немного страдает связность пунктуации.
    3) Резать длинное аудио на куски по 30–60 минут. Это лучшая
       страховка от залипаний и от out-of-memory одновременно:

         ffmpeg -i big.mp4 -c copy -f segment \\
                -segment_time 1800 part_%03d.mp4

🔊  ШУМ И ФОН

    Whisper неплохо борется с типичным фоном (вентилятор, эхо комнаты),
    но плохо переносит:
    • громкую музыку в записи
    • двух людей говорящих одновременно (потеряет одного)
    • TV/радио на заднем фоне (попытается транскрибировать всё)

    Что помогает:
    1) Максимальная large-v3 уже закреплена политикой скилла.
    2) Пред-обработка через ffmpeg:

         # Шумодав + усиление речи + нормализация громкости
         ffmpeg -i in.m4a -af \\
           "highpass=f=100, lowpass=f=8000, afftdn=nf=-25, dynaudnorm" \\
           clean.wav

    3) Если в созвоне явно больше одного потока (Zoom/Meet/Teams) —
       лучше брать раздельные дорожки участников, если платформа их
       выдаёт. Диаризация по моно-миксу всегда хуже.

🕒  ДЛИННОЕ АУДИО (3+ ЧАСА)

    Известные проблемы:
    • Whisper-модель и pyannote держат всё в памяти. На 16 ГБ RAM
      запись на 3+ часа может упереться в OOM, особенно с pyannote.
    • Чем длиннее запись, тем выше риск залипаний.
    • Транскрипт может не влезть в один контекст для саммари —
      Claude будет читать его частями.

    Что делать:
    1) Резать ffmpeg-ом на 30–60-минутные куски (см. выше) и
       обрабатывать по очереди. Скилл сам соберёт несколько отчётов
       или сведёт их в один по запросу.
    2) Не уменьшать модель: при нехватке памяти обрабатывать части по очереди.
    3) Не запускать одновременно с другими ресурсоёмкими процессами.

🗣️  ИМЕНА, ТЕРМИНЫ, ЖАРГОН

    Whisper транслитерирует имена и проектные термины как попало:
    «Тима» может стать «тема», «k8s» — «кубернетес», «бэкенд» —
    «backend» в латинице и наоборот.

    Что помогает:
    1) initial_prompt — текстовая подсказка модели в начале. Передай
       глоссарий через --initial-prompt:

         python3 scripts/transcribe.py созвон.mp4 \\
           --initial-prompt "Участники: Ирина, Тима, Глеб. \\
                             Проект: Терапия чернилами, k8s, OpenCode."

       Модель начнёт распознавать эти имена/термины правильно.
    2) Постобработка: после транскрипции Claude может выровнять
       написание по контексту — попроси его явно.

📉  ТИХАЯ ИЛИ НЕРОВНАЯ ГРОМКОСТЬ

    Если запись где-то очень тихая, а где-то громкая (типичный созвон
    с одним «громким» и двумя «тихими» участниками) — Whisper может
    проглатывать тихие реплики.

    Лечится нормализацией громкости через ffmpeg перед транскрипцией:

      ffmpeg -i in.m4a -af "loudnorm=I=-16:TP=-1.5:LRA=11" out.wav

🎯  МОДЕЛЬ WHISPER

    Скилл всегда использует максимальную large-v3 и не предлагает выбор размера.
""")

    return 0 if (
        has_ffmpeg
        and has_ffprobe
        and has_gigaam
        and whisper_quality_usable
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
