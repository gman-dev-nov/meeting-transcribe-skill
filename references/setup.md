# Установка и настройка

После одной начальной настройки всё работает локально. Никаких платных сервисов.

## Рекомендуемая установка (Apple Silicon)

**Дефолтный бэкенд скилла — `whisper.cpp` с Metal-ускорением.** Самый быстрый, поддерживает beam search, авто-обнаруживается скиллом если стоит в `~/whisper.cpp/`.

```bash
# 1. ffmpeg
brew install ffmpeg              # или скачай со сборкой evermeet.cx → положи в /usr/local/bin/

# 2. whisper.cpp + Metal
pip install cmake                # если нет
git clone https://github.com/ggml-org/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8

# 3. Модели в ggml-формате
bash models/download-ggml-model.sh large-v3-turbo   # ~1.5 ГБ — для fast/balanced
bash models/download-ggml-model.sh large-v3         # ~3 ГБ — опционально для quality

# 4. Локальная диаризация (опционально — нужна только если будешь использовать --diarize)
python3 -m venv ~/.venvs/whisper
source ~/.venvs/whisper/bin/activate
pip install --upgrade pip
pip install resemblyzer scikit-learn
```

Скилл найдёт whisper.cpp автоматически по путям: `$WHISPER_CPP_HOME`, `~/whisper.cpp/`, `~/.local/share/whisper.cpp/`, или через `whisper-cli` в `$PATH`. Если нестандартное место — выстави `export WHISPER_CPP_HOME=/path/to/whisper.cpp`.

Проверка: `python scripts/setup_check.py` — wizard покажет, всё ли на месте.

## Минимальная альтернатива (без сборки whisper.cpp)

Если не хочется ставить cmake и собирать из исходников — можно остановиться на mlx-whisper:

```bash
brew install ffmpeg
python3 -m venv ~/.venvs/whisper && source ~/.venvs/whisper/bin/activate
pip install --upgrade pip
pip install mlx-whisper resemblyzer scikit-learn
```

Скилл будет работать, но потеряет:
- ~30% скорости (mlx RTF 0.06 vs wcpp 0.05 на fast; mlx 0.20 vs wcpp 0.16 на quality)
- **поддержку beam search** (mlx-whisper её не имеет — `balanced`/`quality` пресеты автоматически откатываются на greedy с warning)

## Когда нужен faster-whisper

`faster-whisper` — кросс-платформенный CPU-движок (CTranslate2). На Apple Silicon в большинстве случаев медленнее mlx-whisper и сам по себе не нужен. Ставь его только если:

- Ты не на Apple Silicon (Linux/Windows) — тогда это единственный вариант.
- Хочешь использовать **pyannote-диаризацию**. Pyannote опирается на word-level тайм-коды; faster-whisper выдаёт их аккуратнее, чем mlx.
- Сталкиваешься с **залипаниями на длинных записях (3+ часа)**. У faster-whisper встроенный Silero VAD, который пропускает тишины и снижает галлюцинации.

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
- Модель Whisper (large-v3 ≈ 3 ГБ, large-v3-turbo ≈ 1.5 ГБ).
- Resemblyzer — голосовой энкодер (~17 МБ), если установлен.
- Pyannote — модели сегментации и эмбеддинга (~1 ГБ), если установлен.

Всё кешируется в `~/.cache/huggingface/`. Дальнейшие запуски моментально стартуют.

## Что выбрать для разных сценариев

| Сценарий | Установка |
|---|---|
| Mac, обычные Zoom-созвоны 2–5 человек | mlx-whisper + resemblyzer (минимум выше) |
| Mac, важные/сложные записи, нужна точная диаризация | + pyannote.audio + faster-whisper + HF_TOKEN |
| Mac, много 3+ часовых записей | + faster-whisper (его VAD устойчивее на длинном) |
| Linux/Windows | faster-whisper (mlx не работает) + resemblyzer/pyannote |
| Только лекции / интервью с одним говорящим | mlx-whisper, без `--diarize` |

## Альтернативные ASR (опционально, по бенчмарку быстрее на M4)

Эти инструменты в нашем тесте дали более быструю транскрипцию при сопоставимом качестве финального meeting-отчёта. **Не интегрированы** в `transcribe.py` — используются как самостоятельные CLI; после получения `.txt`/`.srt` Claude читает результат и пишет отчёт по `report_template.md` как обычно.

```bash
# whisper.cpp с Metal (RTF ~0.06 для turbo, ~0.18 для large-v3 на M4)
pip install cmake
git clone https://github.com/ggml-org/whisper.cpp.git ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
bash models/download-ggml-model.sh large-v3-turbo
# Использование:
~/whisper.cpp/build/bin/whisper-cli -m ~/whisper.cpp/models/ggml-large-v3-turbo.bin \
    -f audio.wav -l ru -bs 1 -of out -otxt -osrt -ojf
```

```bash
# gigaam-mlx (русская специализированная модель Сбера, RTF ~0.08 для rnnt)
pip install git+https://github.com/aystream/gigaam-mlx.git
gigaam-mlx audio.mp4 --model-type rnnt --format both --output-dir out/
# rnnt = быстрее и чище в этом MLX-форке; ctc — базовый вариант
```

```bash
# T-one (Conformer 71M от Т-Банка, самый быстрый, но БЕЗ пунктуации/заглавных)
pip install git+https://github.com/voicekit-team/T-one.git miniaudio
python -c "
from tone import StreamingCTCPipeline, read_audio
pipe = StreamingCTCPipeline.from_hugging_face()
phrases = pipe.forward_offline(read_audio('audio.wav'))
print(' '.join(p.text for p in phrases))
"
```

Подробности про скорость, качество и пресеты — в `backends.md`. Подробности про залипания, шумы и предобработку — wizard `python scripts/setup_check.py`.
