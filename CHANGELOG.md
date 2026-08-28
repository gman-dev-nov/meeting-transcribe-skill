# Changelog

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
версии — [SemVer](https://semver.org/lang/ru/).

Записи этого файла — исходник текста релиза на GitHub: скилл показывает
пользователю именно буллеты из тела релиза, поэтому они пишутся для человека,
а не как пересказ коммитов.

## [Unreleased]

### Добавлено

- Проверка обновлений после готового отчёта: `scripts/check_update.py` ходит в
  публичный API релизов GitHub не чаще раза в сутки, показывает список
  изменений и предлагает обновиться только с явного согласия.
- Файл `VERSION` и этот changelog: до них установленную копию не с чем было
  сравнивать.

## [0.1.0] — 2026-08-28

Первая версия с номером. Фиксирует состояние скилла на момент появления
версионирования.

### Добавлено

- Двойной проход GigaAM v3 + Whisper `large-v3` для русских записей с
  выравниванием по таймингам и разбором расхождений через LLM.
- Диаризация спикеров через pyannote или resemblyzer (по запросу).
- Wizard окружения `scripts/setup_check.py` и отчёт по шаблону
  `assets/report_template.md`.

[Unreleased]: https://github.com/gman-dev-nov/meeting-transcribe-skill/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gman-dev-nov/meeting-transcribe-skill/releases/tag/v0.1.0
