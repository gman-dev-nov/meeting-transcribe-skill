# Troubleshooting

## ffmpeg: command not found

Установи: `brew install ffmpeg` (macOS) или `sudo apt install ffmpeg` (Linux).

## ImportError: No module named 'faster_whisper' / 'mlx_whisper' / 'resemblyzer' / 'pyannote'

Активируй то же venv, в котором ставил пакеты:

```bash
source ~/.venvs/whisper/bin/activate
pip install faster-whisper resemblyzer scikit-learn
# опционально:
pip install mlx-whisper pyannote.audio
```

Если ставил с `--user` или системно — проверь `which python` и убедись, что скрипт запускается тем же интерпретатором.

Лучший способ диагностики — `python scripts/setup_check.py`. Wizard покажет, чего не хватает.

## pyannote: 401 Unauthorized / Could not download

1. Убедись, что принял условия на страницах моделей:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. Проверь, что токен виден: `echo $HF_TOKEN`. Если пусто — токен не выставлен или новый shell не подхватил `~/.zshrc`.
3. Создай новый read-токен, если потерял старый: https://hf.co/settings/tokens

Если pyannote всё ещё не работает — добавь `--diarizer resemblyzer`, и скрипт переключится на локальный вариант без токена.

## Resemblyzer не может скачать веса

При первом запуске resemblyzer скачивает голосовой энкодер с GitHub releases (~17 МБ). Если корпоративный прокси блокирует — можно скачать вручную: репозиторий [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) → веса в папку пакета.

Проверка: `python -c "from resemblyzer import VoiceEncoder; VoiceEncoder()"` — если успешно, всё закешировалось.

## Out of memory на длинных записях

Whisper-модель ест больше всего памяти. Варианты:
- Используй `--compute-type int8` (это и так дефолт через `auto`).
- Возьми пресет `fast` или `balanced` (turbo вместо large-v3).
- Разрежь запись на куски ffmpeg-ом и обработай по частям:
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

## Очень медленно на M4

- Используй `--backend mlx-whisper` — в 2–3 раза быстрее на Apple Silicon (но без диаризации).
- Если нужна диаризация — установи `--compute-type int8`, выключи параллельные ресурсоёмкие приложения.
- Для черновых прогонов возьми `--model large-v3-turbo`.
- Resemblyzer работает на CPU, но он быстрый: на часовом созвоне диаризация занимает ~1–2 минуты сверх транскрипции.

## Транскрипт без знаков препинания

Это особенность Whisper при определённых параметрах. Попробуй:
- Убедиться, что `vad_filter=True` (по умолчанию в скрипте включён).
- Поставить `--language ru` явно (иногда auto-detect сбивает).

## Модель скачивается каждый раз

Кэш HuggingFace по умолчанию в `~/.cache/huggingface/`. Если он очищается — проверь, что в системе нет агрессивной чистки `tmpwatch` или подобного. Для CTranslate2 (faster-whisper) кэш в `~/.cache/huggingface/hub/`.

## Скрипт падает на длинных файлах с MemoryError на этапе ffmpeg

ffmpeg сам по себе памяти ест мало. Если падает Python — это уже whisper. См. секцию про OOM выше.
