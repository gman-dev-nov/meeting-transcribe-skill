# meeting-transcribe

Локальный скилл для Claude Code: расшифровка созвонов и встреч + автоматический отчёт с TL;DR, решениями, action items и цитатами. Работает полностью офлайн на Mac (Apple Silicon), Linux и Windows. Заточен под русский язык.

## Как пользоваться из Claude

Когда скилл установлен — просто упомяни путь к записи в Claude Code:

> разбери созвон `~/Downloads/meeting.mp4`

Дальше Claude всё сделает сам:

1. посчитает длительность и оценку времени, запустит транскрипцию (пресет `fast` по умолчанию),
2. подхватит установленный Whisper-бэкенд (whisper.cpp / mlx / faster-whisper),
3. сохранит транскрипт рядом с файлом (`<имя>.transcript.md` + `.json` + `.srt`),
4. напишет финальный отчёт `<имя>.report.md` на русском по шаблону: TL;DR, темы, **решения**, **action items** с дедлайнами, цитаты с тайм-кодами, открытые вопросы, follow-ups,
5. перескажет 2–3 главных тезиса в чате.

После этого можно задавать вопросы по записи: «что Илья говорил про дедлайн?», «какой выбрали стек?», «дай дословную цитату про найм».

**Когда стоит уточнить запрос явно:**

| Хочу | Скажи Claude |
|---|---|
| качество выше дефолтного (важная запись, плохое аудио) | «запусти с пресетом `quality`» |
| метки спикеров (кто что сказал) | «раздели по спикерам» / «нужна диаризация» |
| язык не русский | «это на английском» / «`--language en`» |
| знаю точное число участников (улучшит диаризацию) | «нас было 4 человека» |

## Ленивая установка через Claude

Если не хочется ставить руками — открой Claude Code в любой папке и скопипасти этот промпт. Claude сам всё проверит, поставит и спросит только то, что важно.

```text
Поставь скилл meeting-transcribe (https://github.com/gman-dev-nov/meeting-transcribe-skill).

1. Если папки ~/.claude/skills/meeting-transcribe ещё нет — клонируй туда репу.
2. Проверь ffmpeg (which ffmpeg). Нет — поставь (Mac: brew install ffmpeg,
   Linux: подскажи команду под дистрибутив).
3. Создай venv ~/.venvs/whisper и активируй. Используй ~/.venvs/whisper/bin/pip
   для всех установок (НЕ системный pip).
4. Определи платформу (uname -sm) и **поставь рекомендуемый скиллом
   бэкенд по умолчанию, без выбора**:

   - **Apple Silicon → whisper.cpp с Metal (fp16).** Топ по скорости
     (RTF ~0.05 fast / ~0.16 quality на M4 — часовая запись за 4–8 мин),
     поддерживает реальный beam search. Скилл сам ставит его в первый
     приоритет автодетекта.
   - **Linux / Windows / Intel-Mac → faster-whisper** (`pip install
     faster-whisper`). Единственный кросс-платформенный вариант.

   **Важный нюанс про качество:** все три бэкенда (whisper.cpp / mlx-whisper /
   faster-whisper) гоняют ОДНИ И ТЕ ЖЕ чекпоинты OpenAI Whisper, поэтому
   качество транскрипта различается в пределах 1–2% WER — на слух не услышишь.
   Выбор по сути идёт по скорости и удобству установки, не по «качеству».

   Сборка whisper.cpp:
       ~/.venvs/whisper/bin/pip install -r ~/.claude/skills/meeting-transcribe/scripts/requirements.txt
       brew install cmake  # если ещё нет
       git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
       cd ~/whisper.cpp
       cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
       cmake --build build --config Release -j 8
       bash models/download-ggml-model.sh large-v3-turbo   # ~1.5 ГБ для fast/balanced
       bash models/download-ggml-model.sh large-v3         # +3 ГБ для quality (опционально)

   Скилл найдёт whisper.cpp автоматически по `~/whisper.cpp/`. Полный гайд
   с альтернативными путями — в `references/setup.md`.

   Альтернативы — упоминаешь ТОЛЬКО если я попрошу:
   - **mlx-whisper** (`pip install mlx-whisper`) — если я скажу «не хочу
     cmake / лень собирать / просто попробовать». На M-чипе час аудио
     ≈ 6–12 мин, ставится за минуту. Минус: не умеет beam search, пресеты
     balanced/quality молча падают на greedy.
   - **faster-whisper на Apple Silicon** — единственный из трёх со встроенным
     Silero VAD, поэтому лучше держит длинные записи с тишинами / музыкой
     на холде / долгими паузами (меньше галлюцинаций). Минус: на M-чипе
     CPU-only, час аудио ≈ 15–30 мин (в 2–4 раза медленнее остальных).
     Имеет смысл если: (а) запись с большим количеством тишин и whisper.cpp
     зацикливается даже с `--max-context 0`, или (б) планируется
     pyannote-диаризация (она лучше работает с word-level тайм-кодами
     faster-whisper).

   Не предлагай эти альтернативы по своей инициативе — по дефолту ставь
   whisper.cpp.

5. Спроси, нужна ли диаризация (метки спикеров — кто что сказал). Цена:

   - **Скорость:** добавляет отдельный проход после транскрипции. На M-чипе
     resemblyzer добавляет ~0.05 RTF (часовое аудио → +3 мин), pyannote —
     ~0.12 RTF (часовое аудио → +7 мин). На 2-часовом созвоне это +6 / +14 мин.
   - **Память:** pyannote держит ~2–3 ГБ дополнительно. На 8 ГБ Mac вместе с
     Whisper large будет тесно — учитывай это, если параллельно открыт браузер
     с десятком вкладок.
   - **Точность:** на 2–5 спикерах с нормальным микрофоном работает уверенно.
     На 6+ спикерах или громких звонках начинает путать.
   - **Когда стоит включать:** если запись длиннее 2ч и спикеров 6+, или нужен
     Q&A постфактум («что Илья сказал про дедлайн?»), или нужна юр. точность
     цитат. Иначе Claude атрибутирует реплики по контексту сам.

   Если да — `~/.venvs/whisper/bin/pip install pyannote.audio` (точнее) или
   `~/.venvs/whisper/bin/pip install resemblyzer scikit-learn` (без токенов).
   Для pyannote расскажи как получить бесплатный HF_TOKEN
   (references/setup.md).

6. Запусти `~/.venvs/whisper/bin/python ~/.claude/skills/meeting-transcribe/scripts/setup_check.py`
   и покажи результат. После этого скилл активен в следующей же сессии Claude
   Code (загружается из `~/.claude/skills/meeting-transcribe/SKILL.md`).
```

После этого скилл готов: брось любой `.mp4`/`.m4a` Claude Code'у со словами «транскрибируй» — он сам всё сделает.

Если хочется ставить руками или понять, что куда кладётся — см. раздел [Установка](#установка) ниже.

## Что умеет

- **Транскрипция** видео/аудио (.mp4, .mov, .mkv, .webm, .m4a, .mp3, .wav, .flac, .ogg) через локальный Whisper.
- **Preflight-анализ перед каждой транскрипцией** — измеряет длительность, доли тишин, аудио-свойства (sample rate, битрейт, громкость) и **рекомендует пресет с обоснованием**. Длинные записи (>2.5ч) и записи с тишинами ≥30 сек уходят на `quality` (large-v3, beam=5) — не на `fast`/`balanced` с `large-v3-turbo`, который зацикливается на длинных тишинах.
- **Три пресета скорости/качества** — `fast` / `balanced` / `quality`. По умолчанию ассистент запускает то, что вернул preflight; можно перезадать словами («запусти с quality», «нужен черновик», «дай выбрать»).
- **Авто-выбор уже установленного бэкенда** в порядке `whisper.cpp` → `mlx-whisper` → `faster-whisper`.
- **Если ни одного бэкенда нет — ассистент спросит, какой поставить, и поставит сам** (одной `pip install`-командой для `mlx-whisper` или `faster-whisper`). Для `whisper.cpp` нужна cmake-сборка — ассистент проведёт по гайду из `references/setup.md`.
- **Диаризация спикеров** (опционально) — кто что сказал. Поддерживается `pyannote` (точнее, нужен бесплатный HF-токен) и `resemblyzer` (без токенов).
- **Структурированный отчёт** на русском по фиксированному шаблону: метаданные → TL;DR → темы → решения → action items → цитаты с тайм-кодами → открытые вопросы → follow-ups.
- **Q&A по записи** после расшифровки — можно задавать вопросы вроде «что Илья говорил про дедлайн?».

### Что включено по умолчанию

| Шаг | Дефолт | Можно поменять |
|---|---|---|
| Бэкенд | whisper.cpp (если стоит), иначе mlx-whisper, иначе faster-whisper | `--backend whisper-cpp\|mlx-whisper\|faster-whisper` |
| Пресет | По результатам preflight: чистая запись <90 мин → `fast`; длиннее или с длинными тишинами / низким битрейтом → `balanced`/`quality` | `--preset fast\|balanced\|quality` или просто скажи «запусти с quality» / «дай выбрать» |
| Язык | `ru` (русский). Whisper мультиязычный — поддерживает 90+ языков, просто на русские созвоны выгоднее задавать явно, без автодетекта. | `--language auto` (автодетект), `--language en` и любой код ISO-639-1 |
| Диаризация | **выключена** | `--diarize` |
| Диаризатор | `auto` (pyannote если есть HF_TOKEN, иначе resemblyzer) | `--diarizer pyannote\|resemblyzer` |

## Установка

### Шаг 1. Базовые зависимости (нужны для обоих вариантов)

```bash
# ffmpeg
brew install ffmpeg

# Python venv
python3 -m venv ~/.venvs/whisper
source ~/.venvs/whisper/bin/activate
pip install --upgrade pip
```

### Шаг 2. Бэкенд транскрипции — выбери один

**Вариант A — whisper.cpp (рекомендуется на Apple Silicon, самый быстрый):**

```bash
pip install cmake
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
bash models/download-ggml-model.sh large-v3-turbo   # ~1.5 ГБ для fast/balanced
bash models/download-ggml-model.sh large-v3         # ~3 ГБ опционально для quality
```

**Вариант B — mlx-whisper (проще, без сборки, только Apple Silicon):**

```bash
pip install mlx-whisper
```

**Вариант C — faster-whisper (Linux/Windows или если нужен pyannote):**

```bash
pip install faster-whisper
```

### Шаг 3. Диаризация — опционально

```bash
# Локальная, без токенов (на 2–5 спикерах достаточно)
pip install resemblyzer scikit-learn

# Или точнее, но нужен бесплатный HF_TOKEN
pip install pyannote.audio
```

Получение `HF_TOKEN`: см. `references/setup.md` → раздел «HF_TOKEN для pyannote».

### Шаг 4. Клонировать сам скилл

```bash
mkdir -p ~/.claude/skills
git clone https://github.com/gman-dev-nov/meeting-transcribe-skill.git ~/.claude/skills/meeting-transcribe
```

### Шаг 5. Проверить, что всё на месте

```bash
~/.venvs/whisper/bin/python ~/.claude/skills/meeting-transcribe/scripts/setup_check.py
```

Wizard покажет, что установлено и чего не хватает.

## Использование в Claude Code

После установки скилл работает сразу — Claude Code автоматически подхватывает скиллы из `~/.claude/skills/`. Базовый сценарий и список явных уточнений — в разделе [«Как пользоваться из Claude»](#как-пользоваться-из-claude) выше.

Под капотом ассистент:
1. сообщит длительность и оценку времени, сразу запустит `fast`,
2. дождётся транскрипции,
3. прочитает `.transcript.md` и напишет рядом отчёт `<имя>.report.md`.

Если хочешь другой пресет (`balanced` / `quality`) — скажи об этом до запуска или в момент запроса. Если нужна диаризация — скажи явно: «раздели по спикерам».

## Запуск из командной строки (без Claude)

Скрипт можно дёргать и руками:

```bash
# Только оценка времени
python ~/.claude/skills/meeting-transcribe/scripts/transcribe.py ~/meeting.mp4 --estimate-only

# Транскрипция с пресетом fast (без диаризации)
python ~/.claude/skills/meeting-transcribe/scripts/transcribe.py ~/meeting.mp4 \
  --preset fast --language ru --yes

# С диаризацией
python ~/.claude/skills/meeting-transcribe/scripts/transcribe.py ~/meeting.mp4 \
  --preset fast --language ru --diarize --num-speakers 4 --yes
```

Результаты лягут рядом с исходником: `meeting.transcript.json`, `meeting.transcript.md`, `meeting.transcript.srt`.

## Структура проекта

```text
meeting-transcribe/
├── SKILL.md                   # инструкции для модели (Claude читает их сам)
├── scripts/
│   ├── transcribe.py          # основной скрипт транскрипции
│   ├── setup_check.py         # wizard проверки окружения
│   └── requirements.txt
├── assets/
│   └── report_template.md     # шаблон финального отчёта
└── references/
    ├── setup.md               # подробная установка, HF_TOKEN
    ├── backends.md            # сравнение бэкендов и моделей, RTF на M4
    └── troubleshooting.md     # типичные проблемы
```

## Если что-то не работает

1. Запусти `python scripts/setup_check.py` — покажет, чего не хватает.
2. См. `references/troubleshooting.md` — типичные проблемы.

## Лицензия и зависимости

Скрипт сам ничего не отправляет в облако. Whisper-модели качаются с HuggingFace при первом запуске и кешируются в `~/.cache/huggingface/`. После загрузки всё работает офлайн.
