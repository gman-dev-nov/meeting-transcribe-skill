# meeting-transcribe

Скилл для Claude Code: локальная расшифровка созвонов и встреч + автоматический отчёт с TL;DR, решениями, action items и цитатами. ASR работает локально на Mac (Apple Silicon), Linux и Windows. Для русской речи два независимых движка сверяются через активный LLM, а спорные места подтверждает человек.

## Как пользоваться из Claude

Когда скилл установлен — просто упомяни путь к записи в Claude Code:

> разбери созвон `~/Downloads/meeting.mp4`

Дальше Claude всё сделает сам:

1. посчитает длительность и проверит окружение,
2. для русской речи одновременно запустит полные GigaAM и Whisper `large-v3` (`quality`, beam=5),
3. сохранит раздельные транскрипты `.gigaam.*` и `.whisper.*`, не перезаписывая их,
4. сопоставит их по времени и покажет смысловые расхождения: точный интервал исходной записи и обе версии,
5. попросит прослушать спорные интервалы; правильную версию не выберет автоматически,
6. после подтверждения напишет `<имя>.report.md` на русском по шаблону.

После этого можно задавать вопросы по записи: «что Илья говорил про дедлайн?», «какой выбрали стек?», «дай дословную цитату про найм».

**Когда стоит уточнить запрос явно:**

| Хочу | Скажи Claude |
|---|---|
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
3. Создай venv ~/.venvs/whisper. Если в PATH есть uv — `uv venv ~/.venvs/whisper`
   и ставь пакеты через `uv pip install --python ~/.venvs/whisper/bin/python …`;
   иначе `python3 -m venv` и `~/.venvs/whisper/bin/python -m pip install …`.
   Учти: в venv, созданном uv, нет `pip`, поэтому команды вида
   `~/.venvs/whisper/bin/pip install` там не работают. Системный pip не
   использовать. Пропиши в ~/.zshrc
   export WHISPER_PYTHON=$HOME/.venvs/whisper/bin/python — иначе скрипты,
   запущенные системным python3, не увидят пакеты этого venv.
4. Определи платформу (uname -sm) и **поставь рекомендуемый скиллом
   бэкенд по умолчанию, без выбора**:

   - **Apple Silicon → whisper.cpp с Metal (fp16).** Топ по скорости
     (RTF ~0.16 на M4 в штатном quality-режиме),
     поддерживает реальный beam search. Скилл сам ставит его в первый
     приоритет автодетекта.
   - **Linux / Windows / Intel-Mac → faster-whisper** (`pip install
     faster-whisper`). Единственный кросс-платформенный вариант.

   Бэкенд определяется платформой, но режим распознавания не
   выбирается: всегда устанавливай и запускай полную `large-v3` с
   `quality`, beam=5. Не спрашивай пользователя о размере или модели Whisper.

   Сборка whisper.cpp:
       # пакеты скилла — тем же способом, что выбран в шаге 3
       brew install cmake  # если ещё нет
       git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
       cd ~/whisper.cpp
       cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
       cmake --build build --config Release -j 8

   Сразу скачай обязательную `large-v3` (~3 ГБ):
       cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3

   `setup_check.py` должен подтвердить наличие `large-v3`; без неё штатный
   двойной workflow не запускается.

   Скилл найдёт whisper.cpp автоматически по `~/whisper.cpp/`. Полный гайд
   с альтернативными путями — в `references/setup.md`.

   Не предлагай другую Whisper-модель или пониженный пресет даже ради
   скорости или экономии места.

5. Поставь GigaAM в отдельный venv ~/.venvs/asr (uv или python -m venv, как в
   шаге 3) и в него:
         "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git"
         silero-vad

6. Спроси, нужна ли диаризация (метки спикеров — кто что сказал). Цена:

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

   Если да — поставь в ~/.venvs/whisper либо `pyannote.audio` (точнее), либо
   `resemblyzer scikit-learn` (без токенов).
   Для pyannote расскажи как получить бесплатный HF_TOKEN
   (references/setup.md).

7. Запусти `python3 ~/.claude/skills/meeting-transcribe/scripts/setup_check.py`
   и покажи результат. После этого скилл активен в следующей же сессии Claude
   Code (загружается из `~/.claude/skills/meeting-transcribe/SKILL.md`).
```

После этого скилл готов: брось любой `.mp4`/`.m4a` Claude Code'у со словами «транскрибируй» — он сам всё сделает.

Если хочется ставить руками или понять, что куда кладётся — см. раздел [Установка](#установка) ниже.

## Что умеет

- **Двойная транскрипция русской речи** через локальные GigaAM + Whisper на всей записи, запущенные параллельно.
- **LLM-сверка без автоисправления** — критические отрицания, числа, сроки, имена, решения и обязательства показываются человеку с интервалом исходной записи и обеими версиями.
- **Preflight-анализ перед каждой транскрипцией** — измеряет длительность, тишины, sample rate, битрейт и громкость для оценки времени и предупреждений. Он не выбирает модель или пресет.
- **Один штатный Whisper-режим** — полная `large-v3`, `quality`, beam=5 для каждой записи; скилл не предлагает более маленькую модель или быстрый пресет.
- **Бэкенд по платформе** — `whisper.cpp` на Apple Silicon, `faster-whisper` на Linux, Windows и Intel Mac. Если его нет, ассистент сразу устанавливает штатный вариант, не спрашивая о модели Whisper.
- **Диаризация спикеров** (опционально) — кто что сказал. Поддерживается `pyannote` (точнее, нужен бесплатный HF-токен) и `resemblyzer` (без токенов).
- **Структурированный отчёт** на русском по фиксированному шаблону: метаданные → TL;DR → темы → решения → action items → цитаты с тайм-кодами → открытые вопросы → follow-ups.
- **Q&A по записи** после расшифровки — можно задавать вопросы вроде «что Илья говорил про дедлайн?».

### Штатное поведение

| Компонент | Поведение |
|---|---|
| Русская речь | GigaAM + Whisper параллельно на всей записи, затем LLM-сверка и подтверждение человеком |
| Whisper-режим | Только `large-v3`, `quality`, beam=5; выбор модели пользователю не показывается |
| Whisper-бэкенд | `whisper.cpp` на Apple Silicon; `faster-whisper` на остальных платформах |
| Сверка | Точный интервал исходной записи и обе версии; без аудиоклипа и без предположения о встроенном плеере |
| Другой язык | Whisper `large-v3` (`quality`, beam=5), без GigaAM |
| Диаризация | Выключена; включается только по запросу |

## Установка

### Шаг 1. Клонировать сам скилл

```bash
mkdir -p ~/.claude/skills
git clone https://github.com/gman-dev-nov/meeting-transcribe-skill.git ~/.claude/skills/meeting-transcribe
```

### Шаг 2. Базовые зависимости

```bash
brew install ffmpeg
```

Дальше нужен venv скилла. Если у тебя есть [uv](https://docs.astral.sh/uv/) —
бери его, он быстрее и сам поставит нужный Python:

```bash
uv venv ~/.venvs/whisper
```

Без uv — штатным модулем:

```bash
python3 -m venv ~/.venvs/whisper
~/.venvs/whisper/bin/python -m pip install --upgrade pip
```

> **venv от uv не содержит `pip`.** Это не поломка, а его штатное поведение:
> ставить в такой venv нужно `uv pip install --python ~/.venvs/whisper/bin/python …`.
> Команды ниже даны в обоих вариантах.

Чтобы скрипты нашли пакеты этого venv при запуске из-под другого
интерпретатора, один раз пропиши:

```bash
echo 'export WHISPER_PYTHON=$HOME/.venvs/whisper/bin/python' >> ~/.zshrc
```

### Шаг 3. Установить Whisper-бэкенд по платформе

**Apple Silicon — whisper.cpp с Metal:**

```bash
pip install cmake
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
bash models/download-ggml-model.sh large-v3         # ~3 ГБ, обязательная модель
```

**Linux, Windows и Intel Mac — faster-whisper:**

```bash
pip install faster-whisper
```

`faster-whisper` скачает `large-v3` при первом запуске. В обоих случаях скилл
использует `quality`, beam=5 и не предлагает выбор модели.

### Шаг 4. Установить GigaAM

С uv:

```bash
uv venv ~/.venvs/asr
uv pip install --python ~/.venvs/asr/bin/python \
  "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git" \
  silero-vad
```

Без uv:

```bash
python3 -m venv ~/.venvs/asr
~/.venvs/asr/bin/python -m pip install --upgrade pip
~/.venvs/asr/bin/python -m pip install \
  "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git" \
  silero-vad
```

GigaAM живёт в отдельном venv намеренно: его torch не должен конфликтовать с
Whisper-бэкендом. Нестандартный путь передаётся переменной `GIGAAM_PYTHON`.

### Шаг 5. Диаризация — опционально

```bash
# Локальная, без токенов (на 2–5 спикерах достаточно)
uv pip install --python ~/.venvs/whisper/bin/python resemblyzer scikit-learn

# Или точнее, но нужен бесплатный HF_TOKEN
uv pip install --python ~/.venvs/whisper/bin/python pyannote.audio
```

Без uv — тем же venv: `~/.venvs/whisper/bin/python -m pip install …`.

Получение `HF_TOKEN`: см. `references/setup.md` → раздел «HF_TOKEN для pyannote».

### Шаг 6. Проверить, что всё на месте

```bash
python3 ~/.claude/skills/meeting-transcribe/scripts/setup_check.py
```

Wizard покажет, что установлено и чего не хватает, и явно напечатает, какими
интерпретаторами будут запущены Whisper и GigaAM. Ненулевой код возврата
означает, что штатный двойной workflow ещё не запустится.

## Использование в Claude Code

После установки скилл работает сразу — Claude Code автоматически подхватывает скиллы из `~/.claude/skills/`. Базовый сценарий и список явных уточнений — в разделе [«Как пользоваться из Claude»](#как-пользоваться-из-claude) выше.

Под капотом ассистент:
1. сообщит длительность и оценку времени, не предлагая модель или пресет,
2. запустит GigaAM и Whisper `large-v3` (`quality`, beam=5) параллельно на всей записи,
3. покажет расхождения с интервалами исходной записи и обеими версиями,
4. попросит самостоятельно прослушать эти интервалы в исходном файле и не будет создавать аудиоклипы или предполагать встроенный плеер,
5. после ответа человека напишет `<имя>.report.md`.

Если нужна диаризация, скажи явно: «раздели по спикерам».

## Запуск из командной строки (без Claude)

Штатный двойной workflow можно запустить и вручную:

```bash
~/.venvs/whisper/bin/python3 \
  ~/.claude/skills/meeting-transcribe/scripts/dual_transcribe.py run \
  ~/meeting.mp4 \
  --whisper-backend whisper-cpp
```

На Linux, Windows и Intel Mac подставь `--whisper-backend faster-whisper`.
Режим `large-v3` / `quality` / beam=5 зафиксирован в `dual_transcribe.py` и
не передаётся как пользовательская опция.

Раздельные результаты лягут рядом с исходником: `meeting.gigaam.transcript.*`,
`meeting.whisper.transcript.*`, `meeting.comparison.json`,
`meeting.review-template.json` и логи обоих проходов
(`meeting.gigaam.log`, `meeting.whisper.log`; при падении — `*.failed.log`).

`scripts/transcribe.py` больше не принимает `--preset`, `--model` и
`--beam-size`: политика модели зафиксирована в коде, а не в аргументах.

Если Whisper-бэкенд стоит в venv (`faster-whisper`, `resemblyzer`,
`pyannote`), запускай скрипты интерпретатором этого venv либо укажи его один
раз: `export WHISPER_PYTHON=~/.venvs/whisper/bin/python`. `dual_transcribe.py`
и `setup_check.py` берут интерпретатор оттуда — иначе системный `python3`
честно ответит «бэкендов нет», хотя всё установлено. whisper.cpp — отдельный
бинарник, ему интерпретатор безразличен.

## Структура проекта

```text
meeting-transcribe/
├── SKILL.md                   # инструкции для модели (Claude читает их сам)
├── scripts/
│   ├── transcribe.py          # основной скрипт транскрипции
│   ├── gigaam_longform.py     # GigaAM + Silero VAD
│   ├── dual_transcribe.py     # параллельный запуск и сверка
│   ├── hybrid.py              # словарь и диагностика терминов
│   ├── lexicons/terms.json    # канонические написания терминов
│   ├── setup_check.py         # wizard проверки окружения
│   └── requirements.txt
├── assets/
│   └── report_template.md     # шаблон финального отчёта
├── references/
│   ├── setup.md               # подробная установка, HF_TOKEN, WHISPER_PYTHON
│   ├── backends.md            # сравнение бэкендов и моделей, RTF на M4
│   ├── discrepancy-review.md  # контракт LLM и проверки человеком
│   └── troubleshooting.md     # типичные проблемы
└── tests/                     # python3 -m unittest discover -s tests
```

## Если что-то не работает

1. Запусти `python3 scripts/setup_check.py` — покажет, чего не хватает.
2. См. `references/troubleshooting.md` — типичные проблемы.

## Лицензия и зависимости

ASR-скрипты сами ничего не отправляют в облако. Модели скачиваются при установке/первом запуске и затем работают локально. Тексты транскриптов обрабатываются тем Claude/ChatGPT/Codex-интерфейсом, в котором запущен скилл; его политика данных применяется и к LLM-сверке.
