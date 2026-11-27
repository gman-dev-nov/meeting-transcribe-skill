# Установка и настройка

После одной начальной настройки всё работает локально. Никаких платных сервисов.

## Рекомендуемая установка (Apple Silicon)

**Дефолтный бэкенд скилла — `whisper.cpp` с Metal-ускорением.** Самый быстрый, поддерживает beam search, авто-обнаруживается скиллом если стоит в `~/whisper.cpp/`.

```bash
# 1. ffmpeg
brew install ffmpeg              # без Homebrew — см. «ffmpeg без Homebrew» ниже

# 2. whisper.cpp + Metal
pip install cmake                # если нет
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8

# 3. Обязательная Whisper-модель
bash models/download-ggml-model.sh large-v3         # ~3 ГБ; quality, beam=5

# 4. Локальная диаризация (опционально — нужна только если будешь использовать --diarize)
uv venv ~/.venvs/whisper                 # или: python3 -m venv ~/.venvs/whisper
uv pip install --python ~/.venvs/whisper/bin/python resemblyzer scikit-learn "setuptools<81"
```

`setuptools<81` здесь не прихоть: `webrtcvad` внутри resemblyzer импортирует
`pkg_resources`, которого в setuptools 81+ уже нет.

Скилл найдёт whisper.cpp автоматически по путям: `$WHISPER_CPP_HOME`, `~/whisper.cpp/`, `~/.local/share/whisper.cpp/`, или через `whisper-cli` в `$PATH`. Если нестандартное место — выстави `export WHISPER_CPP_HOME=/path/to/whisper.cpp`.

Проверка: `python3 scripts/setup_check.py` — wizard покажет, всё ли на месте,
и явно напечатает, какими интерпретаторами будут запущены Whisper и GigaAM.

### venv: чем ставить и где держать

Ставить можно чем угодно — важен только путь к интерпретатору. Практическая
разница одна:

| Способ | Как ставить пакеты | Есть ли `pip` внутри |
|---|---|---|
| `uv venv ~/.venvs/<name>` | `uv pip install --python ~/.venvs/<name>/bin/python …` | **нет** |
| `python3 -m venv ~/.venvs/<name>` | `~/.venvs/<name>/bin/python -m pip install …` | да |

uv быстрее и сам подтягивает нужную версию Python, поэтому в инструкциях он
первым. Но **в его venv нет `pip`**, и команды вида `~/.venvs/asr-v4/bin/pip
install` там молча не существуют — это самая частая причина «инструкция не
работает».

Каталог `~/.venvs/<name>` выбран потому, что venv-ов у скилла два и они
именованные; `.venv` в единственном числе принято класть рядом с проектом, а не
в домашнюю папку. Путь ни на что не завязан: обе переменные ниже перекрывают
дефолт, так что держать окружения можно где угодно.

### Каким Python запускаются скрипты

| Переменная | Что задаёт | Если не выставлена |
|---|---|---|
| `WHISPER_PYTHON` | интерпретатор для `transcribe.py` (faster-whisper, resemblyzer, pyannote) | `~/.venvs/whisper`, иначе текущий `python3` |
| `GIGAAM_PYTHON` | интерпретатор для `gigaam_longform.py` | `~/.venvs/asr-v4`, затем существующий `~/.venvs/asr` |

whisper.cpp — отдельный бинарник, ему интерпретатор безразличен; всё остальное
ставится в venv, и системный `python3` этих пакетов не увидит. Если пакеты
лежат в `~/.venvs/whisper`, полезно один раз прописать:

```bash
echo 'export WHISPER_PYTHON=$HOME/.venvs/whisper/bin/python' >> ~/.zshrc
```

### ffmpeg без Homebrew

Если brew нет и ставить его не хочется — статические бинарники в `~/.local/bin` (проверено на чистой машине без brew/cmake):

```bash
mkdir -p ~/.local/bin
# Статические сборки ffmpeg И ffprobe (нужны оба):
#   arm64: https://www.osxexperts.net
#   x86_64 (или Rosetta): https://evermeet.cx/ffmpeg/
# Распакуй оба бинарника в ~/.local/bin, затем:
chmod +x ~/.local/bin/ffmpeg ~/.local/bin/ffprobe
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

`cmake` для сборки whisper.cpp тоже ставится без brew: `pip install cmake` (уже в инструкции выше).

## GigaAM v4 для русских записей

Для `v4_rnnt` установи официальный пакет GigaAM 1.0.0 в отдельное окружение. Команды ниже используют проверенную ревизию и Python 3.12. Пакет поддерживает Python начиная с 3.10.

```bash
uv venv --python 3.12 ~/.venvs/asr-v4
uv pip install --python ~/.venvs/asr-v4/bin/python \
  "gigaam[torch] @ git+https://github.com/salute-developers/GigaAM.git@e9bbf2e773eedd217c6f0d3b89b9728c41332b7d"
```

При установке через `venv` создай окружение командой `python3.12 -m venv ~/.venvs/asr-v4` и передай ту же зависимость в `~/.venvs/asr-v4/bin/python -m pip install`. На Windows интерпретатор окружения находится в `Scripts/python.exe`.

При первом запуске пакет скачает ASR-модель (~464 МиБ), GigaAM VAD (~5,2 МиБ) и два токенизатора в `~/.cache/gigaam`. Дальше распознавание и нарезка работают локально. Для декодирования нужен ffmpeg. HF-токен не требуется.

Для сравнения с v3 добавь к двойному проходу `--gigaam-model v3_e2e_rnnt`. В пакете 1.0 обе модели используют GigaAM VAD. Чтобы воспроизвести прежнюю нарезку Silero, выбери также старое окружение через `--gigaam-python ~/.venvs/asr/bin/python`. У отдельного скрипта `gigaam_longform.py` флаг модели называется `--model`.

Проверка полного русского workflow (движки идут по очереди, артефакты
получают разные суффиксы):

```bash
python3 scripts/dual_transcribe.py run "запись.m4a" \
  --whisper-backend whisper-cpp
```

По умолчанию первым запускается GigaAM, затем Whisper. Последовательный запуск
снижает пиковое потребление памяти. Для одновременного запуска есть флаг `--parallel`.

При OOM скрипт не переключает модель молча — он останавливает процессы и
сохраняет логи ошибки.

### Фиксированный Whisper-режим

Штатный workflow всегда использует полную `large-v3` с `quality` и beam=5.
Скилл не спрашивает, какую модель или размер скачать, и не понижает
качество ради скорости. `mlx-whisper` сохранён в низкоуровневом CLI
только для совместимости: он не поддерживает beam=5 и не является вариантом
штатной установки.

## faster-whisper на Linux, Windows и Intel Mac

`faster-whisper` — штатный кросс-платформенный CPU-движок (CTranslate2) вне
Apple Silicon. Он поддерживает `large-v3` и реальный beam=5. На Apple Silicon
штатно использовать whisper.cpp; faster-whisper там может понадобиться
только для pyannote-диаризации или диагностики проблем бэкенда.

```bash
pip install faster-whisper
```

## Когда нужен pyannote (вместо resemblyzer)

`resemblyzer` — простая локальная диаризация без регистраций. Точность достаточная для большинства созвонов 2–5 человек.

`pyannote.audio` — заметно точнее на сложных записях:
- много пересекающейся речи
- очень короткие реплики (<1 сек)
- более 5 спикеров

Стоимость: один раз получить бесплатный HF_TOKEN (см. ниже). После настройки скилл сам выбирает pyannote, если токен и пакет доступны.

```bash
pip install pyannote.audio faster-whisper   # pyannote лучше работает с faster-whisper
```

## HF_TOKEN для pyannote (бесплатно)

HF_TOKEN — просто способ скачивать публичные модели с HuggingFace. Никаких списаний, всё работает локально после загрузки.

1. Зарегистрироваться на [huggingface.co](https://huggingface.co/).
2. Принять условия использования двух моделей (просто галочка):
   - [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
3. Создать read-токен: [Settings → Access Tokens](https://hf.co/settings/tokens) → New token → role: read.
4. Положить токен в окружение:
   ```bash
   echo 'export HF_TOKEN=hf_xxx_твой_токен' >> ~/.zshrc
   source ~/.zshrc
   ```

## Первый запуск

При первом запуске скачаются:
- Модель Whisper `large-v3` (≈ 3 ГБ), если бэкенд загружает её лениво.
- Resemblyzer — голосовой энкодер (~17 МБ), если установлен.
- Pyannote — модели сегментации и эмбеддинга (~1 ГБ), если установлен.

Всё кешируется в `~/.cache/huggingface/`. Дальнейшие запуски моментально стартуют.

## Штатная установка по платформе

| Платформа | Whisper-бэкенд |
|---|---|
| Apple Silicon | whisper.cpp с Metal |
| Linux, Windows, Intel Mac | faster-whisper |

В обоих случаях режим один: `large-v3`, `quality`, beam=5. Для русской речи
обязательно запускается ещё и GigaAM, а LLM показывает человеку
обе версии и интервал исходной записи. Аудиоклипы не создаются, наличие
встроенного плеера не предполагается.

Подробности про бэкенды и фиксированный quality-режим — в `backends.md`.
Диагностика окружения: `python3 scripts/setup_check.py`.
