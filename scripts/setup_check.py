#!/usr/bin/env python3
"""
Setup wizard для meeting-transcribe.

Проверяет окружение, помогает выбрать диаризатор, рассказывает про
особенности Whisper на длинном/шумном/многоголосом аудио.

Запуск:
    python scripts/setup_check.py
"""

from __future__ import annotations

import os
import platform
import shutil
import sys


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
    check("ffmpeg в PATH", has_ffmpeg,
          "macOS: brew install ffmpeg | Linux: sudo apt install ffmpeg | "
          "без brew: статические ffmpeg+ffprobe → ~/.local/bin "
          "(references/setup.md → «ffmpeg без Homebrew»)")

    # ---- whisper backends ----
    print("\n2) Whisper-бэкенды")

    # whisper.cpp — рекомендуемый дефолт на Apple Silicon: самый быстрый + честный beam search
    from pathlib import Path
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

    wcpp_has_turbo = has_wcpp_bin and _has_wcpp_model("large-v3-turbo")
    wcpp_has_v3 = has_wcpp_bin and _has_wcpp_model("large-v3")
    wcpp_usable = has_wcpp_bin and (wcpp_has_turbo or wcpp_has_v3)

    check("whisper.cpp (рекомендуемый дефолт на Apple Silicon)", has_wcpp_bin,
          "pip install cmake && git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp && "
          "cd ~/whisper.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release && "
          "cmake --build build --config Release -j 8")
    if has_wcpp_bin:
        # Обе модели опциональны по отдельности, но минимум одна нужна.
        # Дефолтный выбор — turbo (быстро, ~1.5 ГБ, пресеты fast/balanced).
        # large-v3 нужен только для пресета quality (~3 ГБ, точнее на сложном аудио).
        check("  ggml-large-v3-turbo.bin (fast/balanced, ~1.5 ГБ)", wcpp_has_turbo,
              "cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3-turbo")
        if wcpp_has_v3:
            check("  ggml-large-v3.bin (quality, ~3 ГБ)", True)
        else:
            warn("  ggml-large-v3.bin не скачан — нужен только для пресета quality (~3 ГБ)",
                 "докачать когда понадобится: "
                 "cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3")
        if not wcpp_has_turbo and not wcpp_has_v3:
            print("      → ни одной модели не скачано, бэкенд не запустится. "
                  "Минимум одну (см. выше).")

    try:
        import mlx_whisper  # noqa
        has_mlx = True
    except ImportError:
        has_mlx = False
    try:
        import faster_whisper  # noqa
        has_fw = True
    except ImportError:
        has_fw = False

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

    # ---- diarizers ----
    print("\n3) Диаризация (опционально, для разметки спикеров)")
    try:
        import resemblyzer  # noqa
        import sklearn  # noqa
        has_resemblyzer = True
    except ImportError:
        has_resemblyzer = False
    try:
        import pyannote.audio  # noqa
        has_pyannote = True
    except ImportError:
        has_pyannote = False

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

    # Приоритет в transcribe.py: whisper-cpp → mlx-whisper → faster-whisper.
    if wcpp_usable:
        if wcpp_has_turbo and wcpp_has_v3:
            print("  • Транскрипция: whisper.cpp + turbo и large-v3 (все пресеты доступны).")
        elif wcpp_has_turbo:
            print("  • Транскрипция: whisper.cpp + large-v3-turbo (пресеты fast/balanced).")
            print("                  Для quality нужен large-v3 — докачать:")
            print("                    cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3")
        else:
            print("  • Транскрипция: whisper.cpp + large-v3 (только пресет quality).")
            print("                  Fast/balanced не запустятся без large-v3-turbo — transcribe.py")
            print("                  жёстко требует ggml-large-v3-turbo.bin для этих пресетов. Докачать:")
            print("                    cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3-turbo")
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
    1) Пресет quality (large-v3) заметно устойчивее на грязном аудио.
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
    2) Использовать пресет fast или balanced — сократит время прогона
       и снизит вероятность залипаний.
    3) Не запускать одновременно с другими ресурсоёмкими процессами.

🗣️  ИМЕНА, ТЕРМИНЫ, ЖАРГОН

    Whisper транслитерирует имена и проектные термины как попало:
    «Тима» может стать «тема», «k8s» — «кубернетес», «бэкенд» —
    «backend» в латинице и наоборот.

    Что помогает:
    1) initial_prompt — текстовая подсказка модели в начале. Передай
       глоссарий через --initial-prompt:

         python scripts/transcribe.py созвон.mp4 \\
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

🎯  КОГДА КАКОЙ ПРЕСЕТ ВЫБРАТЬ

    quality   — плохое аудио, юр. важные записи, нужны точные имена
    balanced  — нормальная Zoom-запись, рабочий созвон, дефолт
    fast      — длинная запись для черновика, пересмотр перед удалением
""")

    return 0 if (has_ffmpeg and (wcpp_usable or has_fw or has_mlx)) else 1


if __name__ == "__main__":
    sys.exit(main())
