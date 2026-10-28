import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dual_transcribe.py"
SPEC = importlib.util.spec_from_file_location("dual_transcribe", SCRIPT)
dual = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = dual
SPEC.loader.exec_module(dual)


def run_args(**overrides) -> argparse.Namespace:
    values = dict(
        input=Path("/tmp/meeting.wav"),
        gigaam_python=sys.executable,
        whisper_python=sys.executable,
        gigaam_device="cpu",
        gigaam_model="v3_e2e_rnnt",
        gigaam_batch_size=4,
        whisper_backend=None,
        whisper_language="ru",
        initial_prompt=None,
        user_lexicon=None,
        no_lexicon_prompt=False,
        no_condition_on_previous_text=False,
        diarize=False,
        diarizer="auto",
        num_speakers=None,
        max_speakers=6,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def legacy_args(**overrides) -> argparse.Namespace:
    """Namespace без новых атрибутов — как у старых вызывающих сторон."""
    values = dict(
        input=Path("/tmp/meeting.wav"),
        gigaam_python=sys.executable,
        whisper_python=sys.executable,
        gigaam_device="cpu",
        gigaam_model="v3_e2e_rnnt",
        gigaam_batch_size=4,
        whisper_backend=None,
        initial_prompt=None,
        no_condition_on_previous_text=False,
        diarize=False,
        diarizer="auto",
        num_speakers=None,
        max_speakers=6,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def term(canonical: str, policy: str = "auto") -> dict:
    return {
        "canonical": canonical,
        "policy": policy,
        "ru_stems": [],
        "ru_exact": [],
        "latin_variants": [],
    }


def flag_value(argv: list, flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


class DualLanguageTests(unittest.TestCase):
    """Язык Whisper и автоподсказка из личного словаря в двойном проходе."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Дефолтный путь указывает на несуществующий файл в изолированном
        # каталоге: реальный ~/.config в тестах не читается никогда.
        patcher = mock.patch.object(
            dual, "DEFAULT_USER_LEXICON", self.root / "missing-lexicon.json"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **overrides) -> dict:
        return dual.build_asr_commands(run_args(**overrides), self.root / "stage")

    def write_lexicon(self, terms, name="lexicon.json") -> Path:
        path = self.root / name
        path.write_text(
            json.dumps({"version": 1, "terms": terms}, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    # ---------- язык: фактические argv движков ----------

    def test_default_language_ru_and_no_prompt_without_lexicon(self):
        commands = self.build()

        self.assertEqual(flag_value(commands["whisper"], "--language"), "ru")
        self.assertNotIn("--initial-prompt", commands["whisper"])
        self.assertNotIn("--language", commands["gigaam"])
        self.assertNotIn("--initial-prompt", commands["gigaam"])

    def test_language_choices_reach_only_whisper_argv(self):
        for language in ("ru", "auto", "en"):
            with self.subTest(language=language):
                commands = self.build(
                    whisper_language=language, no_lexicon_prompt=True
                )

                self.assertEqual(
                    flag_value(commands["whisper"], "--language"), language
                )
                self.assertNotIn("--language", commands["gigaam"])

    def test_parser_defaults(self):
        args = dual.build_parser().parse_args(["run", "meeting.wav"])

        self.assertEqual(args.whisper_language, "ru")
        self.assertIsNone(args.user_lexicon)
        self.assertFalse(args.no_lexicon_prompt)
        self.assertIsNone(args.initial_prompt)

    def test_invalid_language_is_rejected_by_parser(self):
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stderr(stderr):
                dual.build_parser().parse_args(
                    ["run", "meeting.wav", "--whisper-language", "fr"]
                )

        self.assertEqual(raised.exception.code, 2)

    def test_invalid_language_value_is_rejected_before_subprocesses(self):
        # Программный обход argparse тоже не должен доходить до запуска ASR.
        def no_subprocess(*call_args, **kwargs):
            raise AssertionError("subprocess запущен до проверки аргументов")

        with mock.patch.object(dual.subprocess, "run", side_effect=no_subprocess):
            with self.assertRaisesRegex(ValueError, "недопустимый язык"):
                self.build(whisper_language="fr")

    def test_legacy_namespace_without_new_attributes_keeps_working(self):
        commands = dual.build_asr_commands(legacy_args(), self.root / "stage")

        self.assertEqual(flag_value(commands["whisper"], "--language"), "ru")
        self.assertNotIn("--initial-prompt", commands["whisper"])

    # ---------- автоподсказка из личного словаря ----------

    def test_auto_canonicals_only_with_case_insensitive_dedup(self):
        path = self.write_lexicon(
            [
                term("Capybara"),
                term("RAG", policy="suggest"),
                term("capybara"),
                term("Guardrail"),
            ]
        )

        commands = self.build(user_lexicon=path)

        prompt = flag_value(commands["whisper"], "--initial-prompt")
        self.assertEqual(prompt, "Термины: Capybara, Guardrail.")
        self.assertNotIn("RAG", prompt)

    def test_suggest_only_lexicon_adds_no_prompt(self):
        path = self.write_lexicon([term("RAG", policy="suggest")])

        commands = self.build(user_lexicon=path)

        self.assertNotIn("--initial-prompt", commands["whisper"])

    def test_prompt_is_limited_to_thirty_unique_terms(self):
        terms = [term(f"Term{index:02d}") for index in range(1, 41)]
        terms.append(term("term01"))
        path = self.write_lexicon(terms)

        commands = self.build(user_lexicon=path)

        prompt = flag_value(commands["whisper"], "--initial-prompt")
        listed = prompt[len("Термины: "):-1].split(", ")
        self.assertEqual(len(listed), 30)
        self.assertEqual(listed[0], "Term01")
        self.assertEqual(listed[-1], "Term30")
        self.assertNotIn("Term31", prompt)

    def test_explicit_prompt_is_extended_not_replaced(self):
        path = self.write_lexicon([term("Capybara")])

        commands = self.build(
            user_lexicon=path, initial_prompt="Участники: Григорий и Анна."
        )

        prompt = flag_value(commands["whisper"], "--initial-prompt")
        self.assertEqual(prompt, "Участники: Григорий и Анна. Термины: Capybara.")
        self.assertTrue(prompt.startswith("Участники: Григорий и Анна."))

    def test_no_lexicon_prompt_opts_out(self):
        path = self.write_lexicon([term("Capybara")])

        opt_out = self.build(user_lexicon=path, no_lexicon_prompt=True)
        self.assertNotIn("--initial-prompt", opt_out["whisper"])

        kept = self.build(
            user_lexicon=path,
            no_lexicon_prompt=True,
            initial_prompt="Имена: Анна.",
        )
        self.assertEqual(
            flag_value(kept["whisper"], "--initial-prompt"), "Имена: Анна."
        )

    # ---------- ошибки словаря: до любых subprocess'ов ----------

    def test_missing_explicit_lexicon_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "личный словарь не найден"):
            self.build(user_lexicon=self.root / "no-such.json")

    def test_malformed_explicit_lexicon_is_rejected_before_inference(self):
        broken = self.root / "broken.json"
        broken.write_text("{не json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "не читается как JSON"):
            self.build(user_lexicon=broken)

        structure = self.root / "structure.json"
        structure.write_text(json.dumps({"terms": "nope"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "поле terms должно быть списком"):
            self.build(user_lexicon=structure)

        no_canonical = self.write_lexicon(
            [{"policy": "auto", "ru_stems": []}], name="no-canonical.json"
        )
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.build(user_lexicon=no_canonical)

    def test_malformed_default_lexicon_is_rejected_before_inference(self):
        broken = self.root / "default-broken.json"
        broken.write_text("[1, 2,", encoding="utf-8")
        with mock.patch.object(dual, "DEFAULT_USER_LEXICON", broken):
            with self.assertRaisesRegex(ValueError, "не читается как JSON"):
                self.build()

    def test_lexicon_error_precedes_any_subprocess(self):
        broken = self.root / "broken.json"
        broken.write_text("{oops", encoding="utf-8")
        args = run_args(user_lexicon=broken, gigaam_device="auto")

        def no_subprocess(*call_args, **kwargs):
            raise AssertionError("subprocess запущен до проверки личного словаря")

        with mock.patch.object(dual.subprocess, "run", side_effect=no_subprocess):
            with self.assertRaisesRegex(ValueError, "не читается как JSON"):
                dual.build_asr_commands(args, self.root / "stage")


if __name__ == "__main__":
    unittest.main()
