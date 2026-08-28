#!/usr/bin/env python3
"""
Проверка обновлений скилла meeting-transcribe.

Вызывается из SKILL.md **после** сохранения отчёта и никогда до транскрипции.
Ходит в публичный API релизов GitHub не чаще раза в сутки, ничего о
пользователе не передаёт и при любой сетевой проблеме молча выходит с нулём:
проверка обновлений не имеет права испортить готовый отчёт.

Запуск:
    python3 scripts/check_update.py            # тихая проверка (штатный вызов)
    python3 scripts/check_update.py --force    # игнорировать суточный кеш
    python3 scripts/check_update.py --status    # JSON-диагностика, без сети
    python3 scripts/check_update.py --update   # обновиться, только по явному согласию

Выключается полностью: MEETING_TRANSCRIBE_NO_UPDATE_CHECK=1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPO = "gman-dev-nov/meeting-transcribe-skill"
RELEASES_API_URL = "https://api.github.com/repos/{}/releases/latest".format(REPO)
RELEASES_PAGE_URL = "https://github.com/{}/releases".format(REPO)

SKILL_DIR = Path(__file__).resolve().parent.parent

DISABLE_ENV = "MEETING_TRANSCRIBE_NO_UPDATE_CHECK"
CHECK_INTERVAL_SECONDS = 24 * 60 * 60
NETWORK_TIMEOUT_SECONDS = 5
MAX_NOTES = 5
MAX_NOTE_CHARS = 120

# Единственный исходящий запрос скилла. User-Agent обязателен для api.github.com,
# поэтому он статичен и не содержит ничего о машине пользователя.
USER_AGENT = "meeting-transcribe-skill update check"

# Реестр маркетплейсов Claude Code: каждый элемент несёт installLocation —
# каталог, куда платформа положила свою копию. Это единственный источник о
# плагинной раскладке, который не приходится угадывать (в публичной
# документации её нет), поэтому он проверяется первым — но best-effort:
# файла может не быть или формат может измениться.
KNOWN_MARKETPLACES = Path(".claude") / "plugins" / "known_marketplaces.json"


# --------------------------------------------------------------------------- #
# Настройки и кеш
# --------------------------------------------------------------------------- #


def update_check_disabled(env=None):
    """Выключена ли проверка переменной окружения."""
    env = os.environ if env is None else env
    value = str(env.get(DISABLE_ENV, "")).strip().lower()
    return value not in ("", "0", "false", "no", "off")


def cache_path(env=None):
    env = os.environ if env is None else env
    base = env.get("XDG_CACHE_HOME") or ""
    root = Path(base) if base else Path(env.get("HOME", "~")).expanduser() / ".cache"
    return root / "meeting-transcribe" / "update-check.json"


def read_cache(path):
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_cache(path, data):
    """Записать кеш. Любая ошибка записи не должна ничего ломать.

    Если каталог кеша недоступен на запись, отметка времени не сохранится и
    проверка пойдёт в сеть на каждом отчёте. Это осознанный компромисс:
    отчётов на машине единицы в день, а альтернатива — писать куда-то ещё,
    не спросив пользователя.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
    except Exception:
        pass


def cache_is_fresh(cache, now, interval=CHECK_INTERVAL_SECONDS):
    try:
        last_check = float(cache.get("last_check", 0))
    except (TypeError, ValueError):
        return False
    # Сдвинутые назад часы не должны блокировать проверку навсегда.
    return 0 <= (now - last_check) < interval


# --------------------------------------------------------------------------- #
# Версии
# --------------------------------------------------------------------------- #


def local_version(skill_dir=SKILL_DIR):
    """Версия установленной копии из файла VERSION."""
    try:
        text = (Path(skill_dir) / "VERSION").read_text(encoding="utf-8")
    except Exception:
        return None
    text = text.strip()
    return text or None


def parse_version(text):
    """'v1.2.3' -> (1, 2, 3). Нечисловой хвост игнорируется."""
    if not text:
        return None
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(text))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def is_newer(remote, local):
    """Строго ли remote новее local. Неразбираемые версии — не повод шуметь."""
    remote_parts = parse_version(remote)
    local_parts = parse_version(local)
    if remote_parts is None or local_parts is None:
        return False
    length = max(len(remote_parts), len(local_parts))
    remote_parts += (0,) * (length - len(remote_parts))
    local_parts += (0,) * (length - len(local_parts))
    return remote_parts > local_parts


# --------------------------------------------------------------------------- #
# Сеть
# --------------------------------------------------------------------------- #


def fetch_latest_release(url=RELEASES_API_URL, timeout=NETWORK_TIMEOUT_SECONDS):
    """GET последнего релиза. Бросает исключение — вызывающий его гасит.

    Запрос анонимный: ни токена, ни cookie, ни идентификаторов установки.
    `releases/latest` сам отсекает draft и pre-release, поэтому отдельной
    фильтрации предрелизов тут нет.
    """
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    response = urlopen(request, timeout=timeout)  # noqa: S310 — константный https URL
    try:
        payload = response.read()
    finally:
        response.close()
    return json.loads(payload.decode("utf-8"))


def _clean_note(raw):
    """Буллет -> строка для человека."""
    note = re.sub(r"^[-*+]\s+", "", raw.strip())
    note = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", note)  # markdown-ссылки
    note = note.replace("`", "").replace("**", "")
    note = " ".join(note.split())  # переносы строк схлопываем в пробелы
    if len(note) > MAX_NOTE_CHARS:
        note = note[: MAX_NOTE_CHARS - 1].rstrip() + "…"
    return note


def extract_notes(body):
    """Список улучшений из тела релиза: только буллеты, коротко.

    Буллет в CHANGELOG почти всегда перенесён на несколько строк, и его
    продолжение — это отступ, а не новый пункт. Пока продолжения отбрасывались,
    пользователь читал обрывки вроде «Спорные места подтверждает человек: скилл
    показывает точный интервал» — фраза кончалась на середине.
    """
    notes = []
    current = None

    def flush(current):
        if current is None:
            return
        note = _clean_note(current)
        if note:
            notes.append(note)

    for line in str(body or "").splitlines():
        stripped = line.strip()
        if re.match(r"^[-*+]\s+\S", stripped):
            flush(current)
            if len(notes) >= MAX_NOTES:
                return notes
            current = stripped
        elif current is not None and stripped and line[:1].isspace():
            current = "{} {}".format(current, stripped)
        else:
            flush(current)
            current = None
            if len(notes) >= MAX_NOTES:
                return notes

    flush(current)
    return notes[:MAX_NOTES]


def release_summary(payload):
    """Сырой ответ API -> то немногое, что нужно и кладётся в кеш."""
    if not isinstance(payload, dict):
        return None
    tag = payload.get("tag_name") or payload.get("name")
    if not tag:
        return None
    return {
        "version": str(tag).lstrip("v"),
        "tag": str(tag),
        "name": str(payload.get("name") or tag),
        "url": str(payload.get("html_url") or RELEASES_PAGE_URL),
        "notes": extract_notes(payload.get("body")),
    }


# --------------------------------------------------------------------------- #
# Тип установки
# --------------------------------------------------------------------------- #


def _is_inside(path, parent):
    """Лежит ли path внутри parent (или совпадает с ним)."""
    try:
        parent = Path(parent)
    except Exception:
        return False
    return path == parent or parent in path.parents


def _plausible_plugin_location(location):
    """Похоже ли значение из реестра на каталог хранилища плагинов.

    `Path.parents` включает корень, поэтому патологический installLocation
    (`/`, домашний каталог) объявил бы плагином любой клон на машине и тихо,
    необратимо отключил бы `--update` — правкой чужого файла, которую
    пользователь не увидит. Реестр читается как подсказка, а не как истина:
    доверяем значению, только если оно само выглядит как каталог плагинов.
    """
    try:
        parts = Path(location).parts
    except Exception:
        return False
    return "plugins" in parts and len(parts) > 2


def _marketplace_locations(env):
    """installLocation всех известных маркетплейсов. Best-effort."""
    try:
        registry = Path(env.get("HOME", "~")).expanduser() / KNOWN_MARKETPLACES
        with open(str(registry), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        locations = []
        for entry in (data or {}).values():
            location = (entry or {}).get("installLocation")
            if location and _plausible_plugin_location(location):
                locations.append(str(location))
        return locations
    except Exception:
        return []


def detect_install(skill_dir=SKILL_DIR, env=None):
    """('git' | 'plugin' | 'plain', человекочитаемая деталь).

    Плагинную установку обновляет сама платформа, поэтому её нужно отличать
    от ручной копии: `git pull` в кеше плагина не наш способ обновления.
    Ручная копия (проверено на реальной машине: ~/.claude/skills/... без .git)
    не обновляется автоматически вообще.

    Раскладки плагинов нет в публичной документации, поэтому сигналов три и
    любой из них достаточен: реестр маркетплейсов, CLAUDE_PLUGIN_ROOT и
    само расположение внутри ~/.claude/plugins. Путь проверяется и сырым, и
    разрешённым: `resolve()` идёт по симлинкам, и если платформа когда-нибудь
    начнёт симлинкать хранилище, проверка только по разрешённому пути молча
    перестала бы видеть плагин. Ошибка в эту сторону — `git pull` в каталоге,
    которым распоряжается платформа, поэтому сомнение трактуется в пользу
    "plugin".
    """
    env = os.environ if env is None else env
    raw = Path(skill_dir)
    try:
        candidates = [raw, raw.resolve()]
    except Exception:
        candidates = [raw]

    for location in _marketplace_locations(env):
        for candidate in candidates:
            if _is_inside(candidate, location):
                return "plugin", "каталог маркетплейса {}".format(location)

    plugin_root = env.get("CLAUDE_PLUGIN_ROOT")
    if plugin_root:
        for candidate in candidates:
            if _is_inside(candidate, plugin_root) or _is_inside(candidate, Path(plugin_root).resolve()):
                return "plugin", "CLAUDE_PLUGIN_ROOT={}".format(plugin_root)

    for candidate in candidates:
        parts = candidate.parts
        for index in range(len(parts) - 1):
            if parts[index] == ".claude" and parts[index + 1] == "plugins":
                return "plugin", "каталог внутри ~/.claude/plugins"

    resolved = candidates[-1]
    if (resolved / ".git").exists():
        return "git", str(resolved)

    return "plain", str(resolved)


def git(args, skill_dir=SKILL_DIR):
    """Обёртка над git в каталоге скилла: (код возврата, stdout+stderr).

    stdin закрыт и терминальные промпты запрещены: иначе `git pull` на
    незнакомом SSH-ключе или закрытом репозитории повис бы, ожидая ввода в
    том stdin, который ему достался от вызывающей стороны. Нам нужен быстрый
    отказ с внятным текстом, а не зависшая сессия.
    """
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        result = subprocess.run(
            ["git", "-C", str(skill_dir)] + list(args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, result.stdout.strip()


# --------------------------------------------------------------------------- #
# Проверка
# --------------------------------------------------------------------------- #


def render_notice(summary, installed, install_kind, skill_dir=SKILL_DIR):
    """Текст, который увидит пользователь после отчёта."""
    lines = [
        "",
        "---",
        "Доступна новая версия скилла meeting-transcribe: {} (установлена {}).".format(
            summary.get("version"), installed or "неизвестная"
        ),
    ]
    notes = summary.get("notes") or []
    if notes:
        lines.append("Что изменилось:")
        lines.extend("- {}".format(note) for note in notes)
    lines.append(summary.get("url") or RELEASES_PAGE_URL)

    if install_kind == "plugin":
        lines.append(
            "Скилл установлен как плагин — обновляй его штатным механизмом Claude Code "
            "(команда /plugin), а не из скилла."
        )
    elif install_kind == "git":
        lines.append(
            "Обновить: `python3 {}/scripts/check_update.py --update` — "
            "только с твоего явного согласия.".format(skill_dir)
        )
    else:
        lines.append(
            "Каталог скилла не под git — обновление вручную: скачай новую версию "
            "со страницы релизов и замени каталог {}.".format(skill_dir)
        )
    lines.append(
        "Изменения вступят в силу со следующей сессии: текущая уже держит "
        "SKILL.md в контексте."
    )
    lines.append("Отключить проверку обновлений: {}=1".format(DISABLE_ENV))
    return "\n".join(lines)


def check(force=False, env=None, skill_dir=SKILL_DIR, fetch=fetch_latest_release, now=None):
    """Основной поток проверки. Возвращает текст уведомления или None."""
    env = os.environ if env is None else env
    if update_check_disabled(env):
        return None

    now = time.time() if now is None else now
    path = cache_path(env)
    cache = read_cache(path)

    if force or not cache_is_fresh(cache, now):
        cache = dict(cache)
        cache["last_check"] = now
        try:
            summary = release_summary(fetch())
        except Exception:
            # Офлайн, таймаут, rate limit (60 запросов в час без токена), 404 на
            # репозитории без релизов, битый JSON, оборванное на середине тело
            # ответа — молча. Ловим именно `Exception`, а не перечисление типов:
            # http.client.IncompleteRead не наследует OSError, и любой такой
            # промах перечисления стоил бы пользователю сообщения об ошибке
            # поверх готового отчёта. Отметку времени всё равно сохраняем,
            # чтобы не долбить сеть на каждом отчёте.
            summary = None
        if summary:
            cache["latest"] = summary
        write_cache(path, cache)

    summary = cache.get("latest")
    if not isinstance(summary, dict):
        return None

    installed = local_version(skill_dir)
    if installed is None:
        # Копия без VERSION поставлена до того, как версионирование появилось,
        # то есть заведомо старее любого релиза. Молчать тут значило бы, что
        # ровно те пользователи, ради которых задумана проверка, не узнают о
        # новых версиях никогда: файла VERSION в их копии нет и не появится,
        # пока они не обновятся.
        if parse_version(summary.get("version")) is None:
            return None
    elif not is_newer(summary.get("version"), installed):
        return None

    install_kind, _ = detect_install(skill_dir, env)
    return render_notice(summary, installed, install_kind, skill_dir)


def status(env=None, skill_dir=SKILL_DIR, now=None):
    """Диагностика без сети."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    path = cache_path(env)
    cache = read_cache(path)
    install_kind, detail = detect_install(skill_dir, env)
    return {
        "enabled": not update_check_disabled(env),
        "installed_version": local_version(skill_dir),
        "install_kind": install_kind,
        "install_detail": detail,
        "cache_file": str(path),
        "cache_is_fresh": cache_is_fresh(cache, now),
        "last_check": cache.get("last_check"),
        "known_latest": cache.get("latest"),
        "check_interval_seconds": CHECK_INTERVAL_SECONDS,
        "endpoint": RELEASES_API_URL,
    }


# --------------------------------------------------------------------------- #
# Обновление
# --------------------------------------------------------------------------- #


def run_update(skill_dir=SKILL_DIR, env=None):
    """Обновить установленную копию. Вызывать только по явному согласию.

    Возвращает (код возврата, текст). Ничего не делает молча и никогда не
    трогает локальные правки: агент правит собственный код, поэтому любая
    неоднозначность — отказ, а не «разрулю сам».
    """
    env = os.environ if env is None else env
    install_kind, detail = detect_install(skill_dir, env)

    if install_kind == "plugin":
        return 1, (
            "Скилл установлен как плагин ({}). Обновляй его штатным механизмом "
            "Claude Code (/plugin), из скилла обновление не выполняется.".format(detail)
        )

    if install_kind != "git":
        return 1, (
            "Каталог {} не под git — обновиться автоматически нельзя.\n"
            "Варианты: скачать новую версию со страницы {} и заменить каталог, "
            "либо переустановить скилл клоном репозитория.".format(detail, RELEASES_PAGE_URL)
        )

    code, dirty = git(["status", "--porcelain"], skill_dir)
    if code != 0:
        return 1, "Не удалось выполнить git status в {}:\n{}".format(skill_dir, dirty)
    if dirty:
        return 1, (
            "В каталоге скилла есть локальные изменения — обновление отменено, "
            "чтобы их не потерять:\n{}\n"
            "Сохрани или откати их (git stash / git commit) и повтори.".format(dirty)
        )

    code, branch = git(["rev-parse", "--abbrev-ref", "HEAD"], skill_dir)
    if code != 0 or branch == "HEAD":
        return 1, (
            "Скилл не на ветке (detached HEAD) — обновление отменено. "
            "Переключись на ветку вручную и повтори."
        )

    # --recurse-submodules: на репозитории без сабмодулей это no-op, но если
    # релиз когда-нибудь сдвинет указатель сабмодуля, без флага суперпроект
    # уехал бы вперёд, а сабмодуль остался на старом коммите — рассогласование,
    # которое предварительная проверка git status уже не поймала бы.
    code, output = git(["pull", "--ff-only", "--recurse-submodules"], skill_dir)
    if code != 0:
        return 1, (
            "git pull --ff-only не прошёл — ничего не изменено:\n{}\n"
            "Обычные причины: у ветки не настроен upstream, ветка разошлась с "
            "удалённой или нет доступа к репозиторию. Разберись вручную.".format(output)
        )

    version = local_version(skill_dir) or "неизвестная"
    return 0, (
        "Скилл обновлён (ветка {}), установленная версия {}.\n{}\n"
        "Изменения вступят в силу со следующей сессии: текущая уже держит "
        "SKILL.md в контексте.".format(branch, version, output)
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Проверка обновлений скилла meeting-transcribe (после отчёта, не до него)."
    )
    parser.add_argument("--force", action="store_true", help="игнорировать суточный кеш")
    parser.add_argument("--status", action="store_true", help="JSON-диагностика без обращения к сети")
    parser.add_argument(
        "--update",
        action="store_true",
        help="обновить установленную копию (только по явному согласию пользователя)",
    )
    args = parser.parse_args(argv)

    if args.status:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return 0

    if args.update:
        code, message = run_update()
        print(message)
        return code

    try:
        notice = check(force=args.force)
    except Exception:
        # Последний рубеж: что бы ни случилось, отчёт уже написан.
        return 0
    if notice:
        print(notice)
    return 0


if __name__ == "__main__":
    sys.exit(main())
