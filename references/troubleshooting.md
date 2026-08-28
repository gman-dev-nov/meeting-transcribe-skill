# Troubleshooting

## ffmpeg: command not found

Установи: `brew install ffmpeg` (macOS) или `sudo apt install ffmpeg` (Linux).

## ImportError: No module named 'faster_whisper' / 'resemblyzer' / 'pyannote'

Активируй то же venv, в котором ставил пакеты:

```bash
source ~/.venvs/whisper/bin/activate
pip install faster-whisper resemblyzer scikit-learn
# опционально для pyannote-диаризации:
pip install pyannote.audio
```

Если ставил с `--user` или системно, проверь `which python3` и убедись, что скрипт запускается тем же интерпретатором.

Лучший способ диагностики — `python3 scripts/setup_check.py`. Wizard покажет, чего не хватает.

## `~/.venvs/<name>/bin/pip: no such file` или `No module named pip`

venv создан через `uv` — в таких окружениях `pip` не устанавливается. Это не
поломка. Ставить в него нужно так:

```bash
uv pip install --python ~/.venvs/asr/bin/python <пакеты>
```

Если uv под рукой нет, а pip внутри нужен: `~/.venvs/asr/bin/python -m ensurepip
--upgrade`. Проверить происхождение venv: `cat ~/.venvs/asr/pyvenv.cfg` —
строка `uv = …` означает, что окружение создано uv.

## «Ни один бэкенд не готов», хотя faster-whisper установлен

Пакет стоит в venv, а скрипт запущен другим интерпретатором — импорта не
происходит. whisper.cpp это не задевает (он бинарник), а faster-whisper,
resemblyzer и pyannote задевает.

```bash
python3 scripts/setup_check.py     # строка «Python для Whisper: …» покажет, кем запускается transcribe.py
export WHISPER_PYTHON=$HOME/.venvs/whisper/bin/python
```

`dual_transcribe.py` и `setup_check.py` берут интерпретатор из `WHISPER_PYTHON`;
если переменная не выставлена — из `~/.venvs/whisper`, если его нет — текущий.
Разово можно передать флагом: `dual_transcribe.py run … --whisper-python …`.
GigaAM живёт в своём venv и настраивается переменной `GIGAAM_PYTHON`.

## Один из двух ASR-процессов упал

`dual_transcribe.py` останавливает второй процесс и не публикует частично
готовую пару: новый результат нельзя смешивать со старым транскриптом. Смотри
`<имя>.gigaam.failed.log` и `<имя>.whisper.failed.log` рядом с записью.

Типовые проверки:

```bash
~/.venvs/asr/bin/python3 -c "import gigaam, torch, silero_vad; print('GigaAM OK')"
python3 scripts/transcribe.py "<путь>" --analyze
```

При нехватке общей GPU-памяти закрыть ресурсоёмкие приложения; если запуск
шёл с `--parallel` — убрать флаг, движки пойдут по очереди. Крайний вариант —
вынести GigaAM на CPU: `--gigaam-device cpu`. Скрипт намеренно не переключает
модель и не переиспользует старый артефакт молча.

## В чате нет аудиоплеера для расхождения

Это ожидаемо. Скилл не создаёт аудиоклипы и не рассчитывает на встроенное
воспроизведение. Для каждого расхождения он показывает точный интервал
исходной записи и обе версии. Открой исходный файл в любом локальном плеере,
прослушай указанный интервал и сообщи правильную формулировку.

## pyannote: 401 Unauthorized / Could not download

1. Убедись, что принял условия на страницах моделей:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. Проверь, что токен виден: `echo $HF_TOKEN`. Если пусто — токен не выставлен или новый shell не подхватил `~/.zshrc`.
3. Создай новый read-токен, если потерял старый: https://hf.co/settings/tokens

Если pyannote всё ещё не работает — добавь `--diarizer resemblyzer`, и скрипт переключится на локальный вариант без токена.

## resemblyzer: `ModuleNotFoundError: No module named 'pkg_resources'`

Падает не resemblyzer, а его зависимость `webrtcvad`: она делает
`import pkg_resources` на верхнем уровне, а setuptools 81+ этот модуль больше
не поставляет. Лечится пином:

```bash
uv pip install --python ~/.venvs/whisper/bin/python "setuptools<81"
```

Останется предупреждение `pkg_resources is deprecated` — это нормально.

## Resemblyzer не может скачать веса

При первом запуске resemblyzer скачивает голосовой энкодер с GitHub releases (~17 МБ). Если корпоративный прокси блокирует — можно скачать вручную: репозиторий [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) → веса в папку пакета.

Проверка: `~/.venvs/whisper/bin/python3 -c "from resemblyzer import VoiceEncoder; VoiceEncoder()"` — если успешно, всё закешировалось.

## Out of memory на длинных записях

Whisper `large-v3` ест больше всего памяти. Не переключай модель или
пресет: штатный проход всегда остаётся `quality`, beam=5. Вместо этого:

- закрой ресурсоёмкие приложения;
- для GigaAM явно задай CPU-оверрайд `--gigaam-device cpu`;
- если этого недостаточно, разрежь запись на куски и обработай каждую
  часть обоими движками:
  ```bash
  ffmpeg -i big.mp4 -c copy -f segment -segment_time 1800 part_%03d.mp4
  ```

Pyannote тоже требует памяти на длинных записях. Если падает на нём — переключись на resemblyzer (`--diarizer resemblyzer`).

## Транскрипт пустой / очень короткий

- Проверь, что в файле есть звук: `ffmpeg -i video.mp4` покажет аудиодорожку.
- Если записан только один канал, при стерео-сведении он может пропасть. Скрипт уже принудительно сводит в моно (`-ac 1`), так что обычно всё ок.
- Попробуй `--language auto` — модель сама определит язык. Бывает, что в начале записи русская речь идёт после долгого молчания, и Whisper думает, что язык английский.

## Спикеры в диаризации перепутаны

Скрипт назначает метки `SPEAKER_00`, `SPEAKER_01` в порядке появления. Это нормально, что для одного и того же человека на разных записях метки могут отличаться. В отчёте либо переименуй вручную, либо попроси пользователя соотнести метки с именами в начале работы.

Если есть ощущение, что число спикеров определилось неправильно — передай явно:
```bash
--num-speakers 3
```

Resemblyzer чувствителен к коротким репликам (<1 сек) — они могут попадать «не к тому» спикеру. Это компромисс полностью локального решения; если важна максимальная точность — рассмотри pyannote (но он требует HF-токен).

Если же в один кластер попали разные люди целыми блоками, дело не в диаризаторе,
а в записи: при одном микрофоне и спикерфоне удалённые участники акустически
неразличимы. Смена диаризатора и `--num-speakers` тут не помогают — см.
`references/backends.md` → «Ограничение: диаризация разделяет тракты, а не людей».

## Очень медленно на M4

- Убедись, что используется штатный `whisper.cpp` с Metal, а не CPU-бэкенд.
- Выключи параллельные ресурсоёмкие приложения; при дефиците общей памяти перенеси GigaAM на CPU.
- Учти, что полная `large-v3` с beam=5 намеренно медленнее; понижение модели не является штатным решением.
- Resemblyzer работает на CPU, но он быстрый: на часовом созвоне диаризация занимает ~1–2 минуты сверх транскрипции.

## Транскрипт без знаков препинания

Это особенность Whisper при определённых параметрах. Попробуй:
- Убедиться, что `vad_filter=True` (по умолчанию в скрипте включён).
- Поставить `--language ru` явно (иногда auto-detect сбивает).

## Модель скачивается каждый раз

Кэш `large-v3` для HuggingFace-бэкенда по умолчанию находится в
`~/.cache/huggingface/`. Если он очищается, проверь настройки автоочистки. Для
CTranslate2 (faster-whisper) кэш лежит в `~/.cache/huggingface/hub/`.

## Скрипт падает на длинных файлах с MemoryError на этапе ffmpeg

ffmpeg сам по себе памяти ест мало. Если падает Python — это уже whisper. См. секцию про OOM выше.
