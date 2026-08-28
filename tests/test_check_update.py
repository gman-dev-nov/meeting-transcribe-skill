import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_update.py"
SPEC = importlib.util.spec_from_file_location("check_update", SCRIPT)
check_update = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = check_update
SPEC.loader.exec_module(check_update)


RELEASE = {
    "tag_name": "v9.9.9",
    "name": "9.9.9 — сверка",
    "html_url": "https://github.com/gman-dev-nov/meeting-transcribe-skill/releases/tag/v9.9.9",
    "body": "### Добавлено\n\n- Первое улучшение\n- Второе `улучшение`\n\nхвост\n",
}


class TempInstall:
    """Каталог скилла с VERSION и изолированным HOME под кеш."""

    def __init__(self, version="0.1.0"):
        self.version = version

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.skill_dir = root / "skill"
        self.skill_dir.mkdir()
        if self.version is not None:
            (self.skill_dir / "VERSION").write_text(self.version + "\n", encoding="utf-8")
        self.home = root / "home"
        self.home.mkdir()
        self.env = {"HOME": str(self.home)}
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()


def boom(*args, **kwargs):
    raise AssertionError("сеть не должна использоваться в этом сценарии")


class DisableTests(unittest.TestCase):
    def test_env_var_disables_check(self):
        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value):
                self.assertTrue(check_update.update_check_disabled({check_update.DISABLE_ENV: value}))

    def test_absent_or_zero_keeps_check_enabled(self):
        for env in ({}, {check_update.DISABLE_ENV: ""}, {check_update.DISABLE_ENV: "0"}):
            with self.subTest(env=env):
                self.assertFalse(check_update.update_check_disabled(env))

    def test_disabled_check_makes_no_request_and_says_nothing(self):
        with TempInstall() as install:
            env = dict(install.env, **{check_update.DISABLE_ENV: "1"})
            notice = check_update.check(env=env, skill_dir=install.skill_dir, fetch=boom)
        self.assertIsNone(notice)


class VersionTests(unittest.TestCase):
    def test_newer_version_detected(self):
        self.assertTrue(check_update.is_newer("0.2.0", "0.1.9"))
        self.assertTrue(check_update.is_newer("v1.0", "0.9.9"))

    def test_same_or_older_is_not_newer(self):
        self.assertFalse(check_update.is_newer("0.1.0", "0.1.0"))
        self.assertFalse(check_update.is_newer("0.1.0", "0.1.0.1"))
        self.assertFalse(check_update.is_newer("0.1", "0.1.0"))

    def test_unparsable_versions_stay_silent(self):
        self.assertFalse(check_update.is_newer(None, "0.1.0"))
        self.assertFalse(check_update.is_newer("0.2.0", None))
        self.assertFalse(check_update.is_newer("latest", "0.1.0"))

    def test_repo_ships_a_version_file(self):
        self.assertIsNotNone(check_update.local_version())


class NotesTests(unittest.TestCase):
    def test_only_bullets_are_taken_and_markdown_stripped(self):
        notes = check_update.extract_notes(RELEASE["body"])
        self.assertEqual(notes, ["Первое улучшение", "Второе улучшение"])

    def test_multiline_bullet_is_joined(self):
        # CHANGELOG переносит длинные пункты, и продолжение — это отступ.
        # Пока оно отбрасывалось, пользователь читал обрывок фразы.
        body = (
            "### Добавлено\n\n"
            "- Двойной проход GigaAM и Whisper:\n"
            "  два независимых прохода и разбор расхождений.\n"
            "- Короткий пункт.\n"
        )
        notes = check_update.extract_notes(body)
        self.assertEqual(
            notes,
            [
                "Двойной проход GigaAM и Whisper: два независимых прохода и разбор расхождений.",
                "Короткий пункт.",
            ],
        )

    def test_heading_after_bullet_does_not_glue_to_it(self):
        body = "- Пункт один.\n\n### Исправлено\n\n- Пункт два.\n"
        self.assertEqual(check_update.extract_notes(body), ["Пункт один.", "Пункт два."])

    def test_joined_bullet_is_truncated(self):
        body = "- {}\n  {}\n".format("я" * 80, "ю" * 80)
        note = check_update.extract_notes(body)[0]
        self.assertLessEqual(len(note), check_update.MAX_NOTE_CHARS)
        self.assertTrue(note.endswith("…"))

    def test_real_changelog_bullets_survive_the_round_trip(self):
        # Тело релиза делается из CHANGELOG этого репозитория, так что
        # проверяем парсер ровно на нём, а не на выдуманной разметке.
        changelog = SCRIPT.resolve().parents[1] / "CHANGELOG.md"
        body = changelog.read_text(encoding="utf-8").split("## [1.0.0]", 1)[1]
        for note in check_update.extract_notes(body):
            self.assertNotIn("\n", note)
            self.assertTrue(
                note.rstrip().endswith((".", "…", ":")) or len(note) >= check_update.MAX_NOTE_CHARS - 1,
                "пункт оборван на середине: {!r}".format(note),
            )

    def test_notes_are_capped(self):
        body = "\n".join("- пункт {}".format(index) for index in range(20))
        self.assertEqual(len(check_update.extract_notes(body)), check_update.MAX_NOTES)

    def test_long_note_is_truncated(self):
        note = check_update.extract_notes("- " + "я" * 500)[0]
        self.assertLessEqual(len(note), check_update.MAX_NOTE_CHARS)

    def test_empty_body_gives_no_notes(self):
        self.assertEqual(check_update.extract_notes(None), [])


class CheckFlowTests(unittest.TestCase):
    def test_notice_names_the_checked_copy_not_the_module_directory(self):
        # Сообщение печатало каталог самого скрипта, из-за чего в проверке на
        # чужой копии пользователю называли бы не тот путь.
        with TempInstall() as install:
            (install.skill_dir / ".git").mkdir()
            notice = check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE
            )
        self.assertIn(str(install.skill_dir), notice)
        self.assertNotIn(str(check_update.SKILL_DIR), notice)

    def test_new_release_produces_notice_and_caches_it(self):
        with TempInstall() as install:
            notice = check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE
            )
            self.assertIsNotNone(notice)
            self.assertIn("9.9.9", notice)
            self.assertIn("Первое улучшение", notice)
            self.assertIn("следующей сессии", notice)
            self.assertIn(check_update.DISABLE_ENV, notice)

            cache = check_update.read_cache(check_update.cache_path(install.env))
            self.assertEqual(cache["latest"]["version"], "9.9.9")

    def test_same_version_says_nothing(self):
        with TempInstall(version="9.9.9") as install:
            notice = check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE
            )
        self.assertIsNone(notice)

    def test_fresh_cache_does_not_touch_network(self):
        with TempInstall() as install:
            check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE)
            notice = check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=boom)
        self.assertIn("9.9.9", notice)

    def test_force_bypasses_cache(self):
        calls = []

        def fetch():
            calls.append(1)
            return RELEASE

        with TempInstall() as install:
            check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=fetch)
            check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=fetch, force=True)
        self.assertEqual(len(calls), 2)

    def test_stale_cache_triggers_new_request(self):
        with TempInstall() as install:
            now = time.time()
            check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE, now=now
            )
            later = now + check_update.CHECK_INTERVAL_SECONDS + 1
            calls = []

            def fetch():
                calls.append(1)
                return RELEASE

            check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=fetch, now=later
            )
        self.assertEqual(len(calls), 1)

    def test_network_failures_are_silent(self):
        failures = [
            URLError("offline"),
            HTTPError(check_update.RELEASES_API_URL, 403, "rate limit exceeded", {}, None),
            HTTPError(check_update.RELEASES_API_URL, 404, "no releases yet", {}, None),
            OSError("timed out"),
            ValueError("битый JSON"),
            RuntimeError("что-то совсем неожиданное"),
        ]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with TempInstall() as install:
                    def fetch():
                        raise failure

                    notice = check_update.check(
                        env=install.env, skill_dir=install.skill_dir, fetch=fetch
                    )
                self.assertIsNone(notice)

    def test_failed_request_still_marks_the_daily_window(self):
        with TempInstall() as install:
            def fetch():
                raise URLError("offline")

            check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=fetch)
            check_update.check(env=install.env, skill_dir=install.skill_dir, fetch=boom)

    def test_unreadable_cache_dir_does_not_break_check(self):
        with TempInstall() as install:
            # HOME указывает на файл: кеш записать некуда, отчёт это ломать не должно.
            broken = Path(install.home) / "not-a-dir"
            broken.write_text("", encoding="utf-8")
            env = {"HOME": str(broken)}
            notice = check_update.check(
                env=env, skill_dir=install.skill_dir, fetch=lambda: RELEASE
            )
        self.assertIn("9.9.9", notice)

    def test_copy_without_version_file_is_treated_as_outdated(self):
        # Копии, поставленные до появления версионирования, — это ровно те
        # пользователи, ради которых проверка задумана. Молчание для них
        # означало бы, что о релизах они не узнают никогда.
        with TempInstall(version=None) as install:
            notice = check_update.check(
                env=install.env, skill_dir=install.skill_dir, fetch=lambda: RELEASE
            )
        self.assertIsNotNone(notice)
        self.assertIn("9.9.9", notice)
        self.assertIn("неизвестная", notice)

    def test_unparsable_remote_version_stays_silent_without_local_version(self):
        with TempInstall(version=None) as install:
            notice = check_update.check(
                env=install.env,
                skill_dir=install.skill_dir,
                fetch=lambda: dict(RELEASE, tag_name="latest"),
            )
        self.assertIsNone(notice)


class InstallKindTests(unittest.TestCase):
    def test_plain_directory(self):
        with TempInstall() as install:
            kind, _ = check_update.detect_install(install.skill_dir, {})
        self.assertEqual(kind, "plain")

    def test_git_clone(self):
        with TempInstall() as install:
            (install.skill_dir / ".git").mkdir()
            kind, _ = check_update.detect_install(install.skill_dir, {})
        self.assertEqual(kind, "git")

    def test_plugin_root_env(self):
        with TempInstall() as install:
            env = {"CLAUDE_PLUGIN_ROOT": str(install.skill_dir)}
            kind, _ = check_update.detect_install(install.skill_dir, env)
        self.assertEqual(kind, "plugin")

    def test_plugin_cache_path(self):
        with TempInstall() as install:
            plugin_dir = install.home / ".claude" / "plugins" / "marketplaces" / "mt"
            plugin_dir.mkdir(parents=True)
            kind, _ = check_update.detect_install(plugin_dir, {})
        self.assertEqual(kind, "plugin")

    def test_symlinked_plugin_storage_is_still_a_plugin(self):
        # Если платформа начнёт симлинкать хранилище, проверка только по
        # разрешённому пути перестала бы видеть плагин и разрешила бы git pull
        # в каталоге, которым распоряжается платформа.
        with TempInstall() as install:
            real = install.home / "storage" / "meeting-transcribe"
            real.mkdir(parents=True)
            link_parent = install.home / ".claude" / "plugins" / "marketplaces"
            link_parent.mkdir(parents=True)
            link = link_parent / "meeting-transcribe"
            link.symlink_to(real, target_is_directory=True)
            kind, _ = check_update.detect_install(link, {})
        self.assertEqual(kind, "plugin")

    def test_marketplace_registry_wins_over_missing_path_convention(self):
        # Раскладки плагинов нет в публичной документации, поэтому реестр
        # маркетплейсов — сигнал, не зависящий от угаданного пути.
        with TempInstall() as install:
            # Не ~/.claude/plugins: путевое соглашение тут не сработало бы,
            # реестр — сработает. При этом сам installLocation остаётся
            # правдоподобным каталогом хранилища плагинов.
            storage = install.home / "elsewhere" / "plugins" / "market"
            skill = storage / "meeting-transcribe"
            skill.mkdir(parents=True)
            registry = install.home / ".claude" / "plugins" / "known_marketplaces.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text(
                json.dumps({"m": {"installLocation": str(storage)}}), encoding="utf-8"
            )
            kind, detail = check_update.detect_install(skill, install.env)
        self.assertEqual(kind, "plugin")
        self.assertIn(str(storage), detail)

    def test_overly_broad_marketplace_location_is_ignored(self):
        # Патологический installLocation (корень, домашний каталог) объявил бы
        # плагином любой клон на машине и необратимо отключил бы --update.
        for location in ("/", "~", "{home}", "{home}/projects"):
            with self.subTest(location=location):
                with TempInstall() as install:
                    (install.skill_dir / ".git").mkdir()
                    registry = install.home / ".claude" / "plugins" / "known_marketplaces.json"
                    registry.parent.mkdir(parents=True, exist_ok=True)
                    registry.write_text(
                        json.dumps(
                            {"m": {"installLocation": location.format(home=install.home)}}
                        ),
                        encoding="utf-8",
                    )
                    kind, _ = check_update.detect_install(install.skill_dir, install.env)
                self.assertEqual(kind, "git")

    def test_broken_marketplace_registry_is_ignored(self):
        with TempInstall() as install:
            registry = install.home / ".claude" / "plugins" / "known_marketplaces.json"
            registry.parent.mkdir(parents=True, exist_ok=True)
            registry.write_text("{не json", encoding="utf-8")
            kind, _ = check_update.detect_install(install.skill_dir, install.env)
        self.assertEqual(kind, "plain")

    def test_ordinary_clone_outside_plugin_storage_is_git(self):
        with TempInstall() as install:
            (install.skill_dir / ".git").mkdir()
            kind, _ = check_update.detect_install(install.skill_dir, install.env)
        self.assertEqual(kind, "git")

    def test_git_clone_inside_plugin_storage_is_still_a_plugin(self):
        with TempInstall() as install:
            plugin_dir = install.home / ".claude" / "plugins" / "marketplaces" / "mt"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / ".git").mkdir()
            kind, _ = check_update.detect_install(plugin_dir, {})
        self.assertEqual(kind, "plugin")

    def test_plugin_install_is_not_offered_a_git_pull(self):
        with TempInstall() as install:
            plugin_dir = install.home / ".claude" / "plugins" / "marketplaces" / "mt"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "VERSION").write_text("0.1.0\n", encoding="utf-8")
            notice = check_update.check(
                env=install.env, skill_dir=plugin_dir, fetch=lambda: RELEASE
            )
        self.assertIn("/plugin", notice)
        self.assertNotIn("--update", notice)


class UpdateTests(unittest.TestCase):
    def test_refuses_outside_git(self):
        with TempInstall() as install:
            code, message = check_update.run_update(install.skill_dir, {})
        self.assertEqual(code, 1)
        self.assertIn("не под git", message)

    def test_refuses_plugin_install(self):
        with TempInstall() as install:
            code, message = check_update.run_update(
                install.skill_dir, {"CLAUDE_PLUGIN_ROOT": str(install.skill_dir)}
            )
        self.assertEqual(code, 1)
        self.assertIn("/plugin", message)

    def test_refuses_dirty_worktree(self):
        with TempInstall() as install:
            self._git_init(install.skill_dir)
            (install.skill_dir / "SKILL.md").write_text("локальная правка\n", encoding="utf-8")
            code, message = check_update.run_update(install.skill_dir, {})
        self.assertEqual(code, 1)
        self.assertIn("локальные изменения", message)
        self.assertIn("SKILL.md", message)

    def test_refuses_detached_head(self):
        with TempInstall() as install:
            self._git_init(install.skill_dir)
            subprocess.run(
                ["git", "-C", str(install.skill_dir), "checkout", "--detach"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            code, message = check_update.run_update(install.skill_dir, {})
        self.assertEqual(code, 1)
        self.assertIn("detached HEAD", message)

    def test_successful_fast_forward_reports_new_version(self):
        # Единственный сценарий, где действительно выполняется git pull:
        # проверяем его на настоящем клоне, вместе с флагами обёртки.
        with TempInstall() as install:
            upstream = Path(install.home) / "upstream"
            upstream.mkdir()
            (upstream / "VERSION").write_text("0.1.0\n", encoding="utf-8")
            self._git_init(upstream)

            clone = Path(install.home) / "clone"
            self._git(["clone", "-q", str(upstream), str(clone)])

            (upstream / "VERSION").write_text("0.2.0\n", encoding="utf-8")
            self._git(["-C", str(upstream), "add", "-A"])
            self._git(["-C", str(upstream), "commit", "-qm", "release"])

            code, message = check_update.run_update(clone, {})

        self.assertEqual(code, 0, message)
        self.assertIn("0.2.0", message)
        self.assertIn("следующей сессии", message)

    @staticmethod
    def _git(args):
        env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
        subprocess.run(
            ["git"] + list(args),
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _git_init(path):
        env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
        commands = [
            ["init", "-q"],
            ["config", "user.email", "t@example.com"],
            ["config", "user.name", "T"],
            ["add", "-A"],
            ["commit", "-qm", "init"],
        ]
        for command in commands:
            subprocess.run(
                ["git", "-C", str(path)] + command,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


class GitWrapperTests(unittest.TestCase):
    def test_git_runs_without_inherited_stdin_and_without_prompts(self):
        # Промпт за паролем или за незнакомым SSH-ключом не должен вешать
        # сессию: stdin закрыт, GIT_TERMINAL_PROMPT=0.
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(kwargs)
            captured["command"] = command

            class Result:
                returncode = 0
                stdout = ""

            return Result()

        with mock.patch.object(check_update.subprocess, "run", fake_run):
            check_update.git(["status", "--porcelain"], "/nowhere")

        self.assertEqual(captured["stdin"], check_update.subprocess.DEVNULL)
        self.assertEqual(captured["env"]["GIT_TERMINAL_PROMPT"], "0")

    def test_pull_recurses_submodules(self):
        # Без флага сдвиг указателя сабмодуля в релизе оставил бы сабмодуль на
        # старом коммите, и предварительный git status этого уже не поймал бы.
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"pull", "--ff-only", "--recurse-submodules"', source)


class CliTests(unittest.TestCase):
    def test_status_is_valid_json_and_makes_no_request(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertIn("install_kind", payload)
        self.assertEqual(payload["endpoint"], check_update.RELEASES_API_URL)

    def test_disabled_run_is_quiet_and_exits_zero(self):
        env = dict(os.environ, **{check_update.DISABLE_ENV: "1"})
        result = subprocess.run(
            [sys.executable, str(SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(result.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
