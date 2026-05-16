---
name: meeting-transcribe
description: Use this skill whenever the user has a video/audio recording (.mp4, .mov, .mkv, .webm, .m4a, .mp3, .wav, .flac, .ogg) of a meeting, call, interview, or lecture and wants a transcript, summary, action items, decisions, quotes, risks, or insights — especially Russian-language calls (созвоны, встречи, интервью). Trigger phrases include 'транскрибируй', 'расшифруй созвон', 'саммари со встречи', 'выпиши решения и action items', 'разбери митинг', 'transcribe this meeting', 'summarize this call'. Runs local Whisper (whisper.cpp / mlx-whisper / faster-whisper) with selectable speed/quality presets, then produces a structured Russian-language report with TL;DR, topics, decisions, action items, quotes, risks, and follow-ups. Optional speaker diarization via pyannote (needs free HF_TOKEN) or resemblyzer (fully local). Trigger even if user only mentions a file path — they almost always want analysis. Do NOT trigger for short voice notes.
---

# Meeting Transcribe

Локальная транскрипция и анализ записей созвонов на Mac (Apple Silicon). Оптимизирован под русскоязычные встречи 30 минут – несколько часов.

## Что делает

1. Извлекает аудио через `ffmpeg`.
2. Запускает локальный Whisper с одним из пресетов:
   - **fast** — `large-v3-turbo`, beam=1 — **дефолт**, годится для большинства рабочих созвонов.
   - **balanced** — `large-v3-turbo`, beam=5 — маргинально лучше, в ~4× медленнее.
   - **quality** — `large-v3`, beam=5 — для юр. важных записей, плохого аудио, точных цитат.
3. **Опционально** диаризирует спикеров (см. ниже — по умолчанию выключено).
4. Claude сам пишет отчёт по шаблону `assets/report_template.md` на основе транскрипта.

> **Что важно понимать про модели:** `fast`/`balanced` используют `large-v3-turbo` (дистиллят `large-v3`), `quality` — полную `large-v3`. Это **разные чекпоинты**, и на сложном аудио (тихая/эмоциональная речь, длинные паузы, перекрытия) `large-v3` заметно точнее. На чистом рабочем созвоне разница в финальном отчёте обычно невелика — Claude восстанавливает структуру (решения, action items, имена) даже из шумноватого транскрипта. Бери `fast` по умолчанию, `quality` — когда нужны точные дословные цитаты ИЛИ аудио объективно сложное.
>
> Это **не** то же самое, что «whisper.cpp vs mlx vs faster-whisper» — у них разница 1–2% WER на одном чекпоинте, выбор бэкенда идёт по скорости и удобству установки, не по качеству.

Бэкенд выбирается автоматически в порядке: `whisper-cpp` → `mlx-whisper` → `faster-whisper`. Детали и сравнение — в `references/backends.md`. Установка — в `references/setup.md`.

## Когда включать диаризацию

**По умолчанию НЕ включай `--diarize`.** Запускай только если:
- Пользователь явно попросил («раздели по спикерам», «кто что сказал», «нужны метки»).
- Запись длиннее 2 часов И спикеров 6+.
- Нужен Q&A-режим по записи («что Илья говорил про X?»).
- Нужна юридическая точность атрибуции цитат.

На типовом созвоне 4–6 спикеров Claude атрибутирует реплики по контексту самостоятельно. Диаризация добавляет ~10–20% точности и улучшает читаемость, но стоит времени.

Выбор диаризатора: pyannote (точнее, нужен `HF_TOKEN`) или resemblyzer (fallback, без токенов). Скрипт сам выберет `pyannote`, если токен есть. Подробности — в `references/setup.md` и `references/backends.md`.

## Workflow

### Шаг 0. Проверить окружение (только при первом запуске)

```bash
python scripts/setup_check.py
```

После первой успешной транскрипции этот шаг можно пропускать.

### Шаг 1. Принять путь и проверить файл

`ls -la <path>`. Если файла нет — переспроси путь.

### Шаг 2. Preflight-анализ (длительность + риск-факторы → рекомендованный пресет)

Перед каждой транскрипцией:

```bash
python scripts/transcribe.py "<путь>" --analyze
```

(Добавь `--diarize` если пользователь просил диаризацию.)

Скрипт выведет JSON с двумя блоками:

- `options[]` — оценки времени по всем (бэкенд, пресет) парам.
- `analysis` — длительность, audio-properties (sample_rate, bitrate, codec), статистика тишин (count/total/max), volume (mean_db/max_db) и **`recommendation`** — `{preset, backend, warnings, reason}`.

Поле `recommendation.reason` — это **готовая фраза для пользователя**, объясняющая выбор. Используй её в своём сообщении.

#### Шаг 2a. Если `available_backends` пустой — поставить бэкенд

Если в JSON пришло `"available_backends": []` (или `"options": []`) — у пользователя **не установлен ни один Whisper-бэкенд**. Скрипт сам ничего не ставит. **Не пытайся запускать транскрипцию.** Вместо этого:

1. Определи платформу: `uname -sm`.
2. Спроси пользователя, какой бэкенд поставить. Покажи варианты, рекомендация зависит от платформы:

   **На Apple Silicon (Darwin arm64):**
   - **mlx-whisper** — `pip install mlx-whisper`. Одной командой, ставится за минуту. **Рекомендованный быстрый старт.**
   - **whisper.cpp** — самый быстрый, но требует cmake-сборки из исходников (~5 минут). Если у пользователя уже стоит cmake — можно предложить.
   - **faster-whisper** — `pip install faster-whisper`. Работает, но на M-чипах медленнее остальных. Нужен только если планируется pyannote-диаризация.

   **На Linux/Windows / Intel Mac:**
   - **faster-whisper** — `pip install faster-whisper`. Единственный вариант (mlx и whisper.cpp требуют Apple Silicon / отдельной сборки).

3. После подтверждения — поставь выбранный бэкенд (используя тот же python, в котором будет запускаться скрипт; обычно `pip install <pkg>` или `~/.venvs/whisper/bin/pip install <pkg>`).
4. Для **whisper.cpp** дай ссылку на `references/setup.md` → раздел установки (там cmake-сборка + скачивание моделей) и не пытайся ставить сам.
5. После установки **повтори шаг 2** (`--analyze`) и продолжай как обычно.

#### Шаг 2b. Использование рекомендации

**По умолчанию запускай тот пресет и бэкенд, что вернул `recommendation`.** Эвристика учитывает:

- **Длительность** — `≥150 мин → quality`, `≥90 мин → balanced`, `<90 мин → fast`.
- **Длинные тишины (≥30 сек)** — главный сигнал loop-риска для `large-v3-turbo`. Бьют до `quality`.
- **Высокая доля тишины (>10%)**, **низкий sample_rate (≤8 kHz)**, **низкий битрейт (<64 kbps)**, **тихая запись (mean < −30 dB)** — поднимают до `balanced` если запись ещё не длинная.
- **Бэкенд** — приоритет `whisper-cpp` (Metal + честный beam search), потом `mlx-whisper`. Для `balanced`/`quality` `mlx-whisper` пропускается (он не умеет beam search).

Покажи пользователю 1–2 строки с длительностью, рекомендацией и причиной. Сразу запускай — без вопроса. Пример для длинной записи с тишинами:

> Запись 2ч 38мин, найдены тишины ≥30 сек (риск зацикливания на turbo). Рекомендую `quality` (large-v3, beam=5) на whisper-cpp — ~26 мин. Запускаю.

Для короткой чистой записи:

> Запись 42 мин, чистое аудио. Запускаю `fast` (~3 мин).

**Когда переопределить рекомендацию:**
- Пользователь сам сказал «нужно качество», «важная запись», «нужны точные цитаты» → `quality` независимо от рекомендации.
- Пользователь сказал «быстро», «черновик», «не важно качество» → `fast`.
- Пользователь явно просит варианты («дай выбрать», «покажи все опции») → покажи таблицу `options[]` и спроси.

### Шаг 3. Запустить транскрипцию

```bash
python scripts/transcribe.py "<путь>" \
  --preset <recommendation.preset> \
  --backend <recommendation.backend> \
  --language ru \
  --yes
```

Подставляй `--preset` и `--backend` прямо из `analysis.recommendation`. `--backend` обязательно прокидывай явно — иначе скрипт может взять `mlx-whisper` (нет beam search) и `balanced`/`quality` молча откатится в greedy.

**Если `recommendation.no_condition_on_previous_text == true`** — добавь флаг `--no-condition-on-previous-text`. Это критично для длинных записей с тишинами: иначе whisper после длинной паузы залипнет на одной фразе («Так./Видно?», «Продолжение следует…») и hallucination'ит её все оставшееся время через condition-окно.

Дополнительные параметры:
- `--diarize` — только по запросу (см. выше).
- `--diarizer pyannote|resemblyzer|auto` — обычно `auto` (дефолт).
- `--num-speakers N` — если пользователь знает точное число, повышает качество диаризации.

Скрипт сохранит рядом с исходником: `<имя>.transcript.json`, `<имя>.transcript.md`, `<имя>.transcript.srt`. Если падает — `references/troubleshooting.md`.

### Шаг 4. Прочитать транскрипт

Прочитай `.transcript.md`. Часовой созвон обычно ~10–20k токенов. Длинные записи (3+ часа) — читай частями.

### Шаг 5. Написать отчёт по шаблону

Открой `assets/report_template.md` и заполни **все** секции на русском, в том же порядке: метаданные → TL;DR → темы → решения → action items → цитаты → открытые вопросы и риски → follow-ups.

### Шаг 6. Сохранить отчёт

Сохрани как `<имя>.report.md` рядом с записью. Сообщи путь и кратко перескажи 2–3 главных тезиса.

## Принципы отчёта

- Пиши **на русском** независимо от языка интерфейса.
- Опирайся на факты из транскрипта, не выдумывай.
- Тайм-коды в формате `[ЧЧ:ММ:СС]` к цитатам и важным моментам.
- Спикеры: с диаризацией используй метки `SPEAKER_00`/…; если из контекста ясны имена — мягко переименуй или спроси пользователя. Без диаризации — автора цитаты не указывай.
- Action items конкретные: не «обсудить роадмап», а «Ивану — собрать список рискованных тикетов к пятнице».
- Цитаты дословные (можно убрать «эээ» и обрезать тишины).
- Не заполняй секцию ради заполнения. Решений не было — так и пиши.

## Файлы скилла

- `scripts/transcribe.py` — основной скрипт.
- `scripts/setup_check.py` — wizard окружения.
- `references/setup.md` — установка ffmpeg, Python, whisper.cpp, HF_TOKEN.
- `references/backends.md` — детали бэкендов, моделей, пресетов, диаризаторов.
- `references/troubleshooting.md` — типичные проблемы.
- `assets/report_template.md` — обязательный шаблон отчёта.
