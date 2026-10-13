import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hybrid.py"
LEXICON = Path(__file__).resolve().parents[1] / "scripts" / "lexicons" / "terms.json"
SPEC = importlib.util.spec_from_file_location("hybrid", SCRIPT)
hybrid = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = hybrid
SPEC.loader.exec_module(hybrid)


def lexicon(*terms) -> "hybrid.Lexicon":
    return hybrid.Lexicon({"version": 99, "terms": list(terms)})


PROMPT = {
    "canonical": "prompt",
    "policy": "auto",
    "ru_stems": ["промт", "пронт"],
    "ru_exact": [],
    "latin_variants": ["Promt"],
}
API = {
    "canonical": "API",
    "policy": "suggest",
    "ru_stems": [],
    "ru_exact": ["аппы"],
    "latin_variants": [],
}


class NormalizeTests(unittest.TestCase):
    def normalize(self, text, *terms):
        sink = []
        return hybrid.normalize_text(text, lexicon(*terms), sink), sink

    def test_declensions_are_fixed_through_stems(self):
        for form in ("промт", "промта", "промтом", "промтами", "пронте"):
            with self.subTest(form=form):
                result, sink = self.normalize(form, PROMPT)

                self.assertEqual(result, "prompt")
                self.assertEqual(sink[0]["reason"], "stem")

    def test_too_long_ending_is_not_a_declension(self):
        result, sink = self.normalize("промтуализация", PROMPT)

        self.assertEqual(result, "промтуализация")
        self.assertEqual(sink, [])

    def test_whitespace_and_punctuation_survive(self):
        result, _ = self.normalize("Это  «промт», да.", PROMPT)

        self.assertEqual(result, "Это  «prompt», да.")

    def test_leading_capital_is_inherited_from_cyrillic_form(self):
        result, _ = self.normalize("Промт важен", PROMPT)

        self.assertTrue(result.startswith("Prompt"))

    def test_compound_word_keeps_its_second_half(self):
        result, sink = self.normalize("промт-инъекция", PROMPT)

        self.assertEqual(result, "prompt-инъекция")
        self.assertTrue(sink[0]["reason"].startswith("compound/"))

    def test_suggest_policy_is_never_replaced_automatically(self):
        result, sink = self.normalize("аппы сломались", API)

        self.assertEqual(result, "аппы сломались")
        self.assertEqual(sink, [{"kind": "suggest", "from": "аппы", "canonical": "API"}])

    def test_unknown_words_are_left_byte_for_byte(self):
        text = "совершенно посторонний текст со словом guardrail"
        result, sink = self.normalize(text, PROMPT)

        self.assertEqual(result, text)
        self.assertEqual(sink, [])


class MatchTests(unittest.TestCase):
    def test_short_latin_token_is_never_fuzzy_matched(self):
        lex = lexicon({"canonical": "Guardrail", "policy": "auto", "ru_stems": [],
                       "ru_exact": [], "latin_variants": []})

        self.assertEqual(lex.match("Guar")[0], None)

    def test_latin_wordform_is_not_treated_as_distortion(self):
        lex = lexicon({"canonical": "tool", "policy": "auto", "ru_stems": [],
                       "ru_exact": [], "latin_variants": []})

        self.assertEqual(lex.match("tooling")[0], None)

    def test_longer_stem_wins_over_shorter_one(self):
        lex = lexicon(
            {"canonical": "guard", "policy": "auto", "ru_stems": ["гард"],
             "ru_exact": [], "latin_variants": []},
            {"canonical": "guardrail", "policy": "auto", "ru_stems": ["гардрейл"],
             "ru_exact": [], "latin_variants": []},
        )

        self.assertEqual(lex.match("гардрейлы")[0], "guardrail")


class MergeTermsTests(unittest.TestCase):
    """Наложение личного словаря на встроенный (merge_terms)."""

    def merged(self, built_in_terms, user_terms) -> "hybrid.Lexicon":
        return hybrid.Lexicon(
            {"version": 1, "terms": hybrid.merge_terms(built_in_terms, user_terms)})

    def test_user_auto_unblocks_builtin_suggest_form(self):
        # Встроенное «аппы → API» помечено suggest; пользовательская запись
        # с той же формой и policy auto должна разрешать автозамену.
        user = {"canonical": "API", "policy": "auto", "ru_stems": [],
                "ru_exact": ["аппы"], "latin_variants": []}
        lex = self.merged([API], [user])

        self.assertEqual(lex.match("аппы"), ("API", "exact"))

    def test_user_form_is_rebound_to_another_canonical(self):
        # «Githab» во встроенном словаре чинится в GitHub; пользователь
        # решил, что у него это GitLab. Остальные формы канона не страдают.
        built_in = [
            PROMPT,
            {"canonical": "GitHub", "policy": "auto", "ru_stems": [],
             "ru_exact": [], "latin_variants": ["Githab", "Geathab"]},
        ]
        user = {"canonical": "GitLab", "policy": "auto", "ru_stems": [],
                "ru_exact": [], "latin_variants": ["Githab"]}
        lex = self.merged(built_in, [user])

        self.assertEqual(lex.match("Githab")[0], "GitLab")
        self.assertEqual(lex.match("Geathab")[0], "GitHub")
        self.assertEqual(lex.match("промта")[0], "prompt")

    def test_user_stems_cover_declensions(self):
        user = {"canonical": "Capybara", "policy": "auto",
                "ru_stems": ["капибар"], "ru_exact": [], "latin_variants": []}
        lex = self.merged([PROMPT], [user])

        self.assertEqual(lex.match("капибару")[0], "Capybara")
        self.assertEqual(lex.match("капибарами")[0], "Capybara")

    def test_suggest_forms_of_same_canonical_survive_user_auto(self):
        # policy принадлежит форме: пользовательская auto-форма не отменяет
        # остальные suggest-формы того же канона.
        user = {"canonical": "RAG", "policy": "auto", "ru_stems": [],
                "ru_exact": ["рэг"], "latin_variants": []}
        built_in = {"canonical": "RAG", "policy": "suggest", "ru_stems": [],
                    "ru_exact": ["рага", "рагу"], "latin_variants": []}
        lex = self.merged([built_in], [user])

        self.assertEqual(lex.match("рэг")[0], "RAG")
        self.assertEqual(lex.match("рага"), (None, "suggest:RAG"))


class ToStemTests(unittest.TestCase):
    """lexicon add режет флексию, чтобы основа покрыла склонение."""

    def test_feminine_form_gives_consonant_stem(self):
        self.assertEqual(hybrid._to_stem("капибара"), "капибар")
        self.assertEqual(hybrid._to_stem("джиру"), "джир")

    def test_hard_consonant_stem_is_unchanged(self):
        self.assertEqual(hybrid._to_stem("промт"), "промт")

    def test_four_letter_form_still_gives_stem(self):
        self.assertEqual(hybrid._to_stem("рага"), "раг")

    def test_three_letter_form_is_not_mutilated(self):
        self.assertEqual(hybrid._to_stem("пул"), "пул")


class FuzzyGuardTests(unittest.TestCase):
    """Нечёткая замена латиницы держится на системном словаре английского."""

    def setUp(self):
        self.lex = lexicon({"canonical": "Guardrail", "policy": "auto", "ru_stems": [],
                            "ru_exact": [], "latin_variants": []})

    def test_fires_when_the_dictionary_is_available(self):
        with mock.patch.object(hybrid, "ENGLISH_WORDS", frozenset({"guard"})):
            canonical, reason = self.lex.match("Guardrale")

        self.assertEqual(canonical, "Guardrail")
        self.assertTrue(reason.startswith("fuzzy:"))

    def test_is_disabled_without_the_dictionary(self):
        # Linux/Windows: /usr/share/dict/words нет, и «починить» настоящее
        # английское слово в термин было бы тихой порчей текста.
        with mock.patch.object(hybrid, "ENGLISH_WORDS", frozenset()):
            self.assertEqual(self.lex.match("Guardrale"), (None, ""))

    def test_real_english_word_is_never_rewritten(self):
        with mock.patch.object(hybrid, "ENGLISH_WORDS", frozenset({"guardian"})):
            self.assertEqual(self.lex.match("guardian"), (None, ""))


class ShippedLexiconTests(unittest.TestCase):
    def test_every_term_has_a_canonical_and_known_policy(self):
        data = json.loads(LEXICON.read_text(encoding="utf-8"))

        self.assertIsInstance(data.get("terms"), list)
        for term in data["terms"]:
            with self.subTest(term=term.get("canonical")):
                self.assertTrue(term.get("canonical"))
                self.assertIn(term.get("policy", "auto"), {"auto", "suggest"})
                for key in ("ru_stems", "ru_exact", "latin_variants", "latin_suggest"):
                    self.assertIsInstance(term.get(key, []), list)

    def test_loads_without_error(self):
        lex = hybrid.Lexicon(json.loads(LEXICON.read_text(encoding="utf-8")))

        self.assertEqual(lex.version, 1)


class CliTests(unittest.TestCase):
    def transcript(self, directory: Path) -> Path:
        path = directory / "meeting.gigaam.transcript.json"
        path.write_text(
            json.dumps(
                {
                    "metadata": {"source_file": "meeting.wav"},
                    "segments": [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "text": "промт готов",
                            "words": [
                                {"start": 0.0, "end": 1.0, "word": "промт", "conf": 0.4},
                                {"start": 1.0, "end": 2.0, "word": "готов", "conf": 0.9},
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def run_cli(self, *args) -> subprocess.CompletedProcess:
        # HOME изолируется: автозагрузка личного словаря из реального
        # ~/.config не должна влиять на детерминизм тестов.
        with tempfile.TemporaryDirectory() as isolated_home:
            env = {**os.environ, "HOME": isolated_home}
            return subprocess.run(
                [sys.executable, str(SCRIPT), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.transcript(Path(temporary))
            before = source.read_text(encoding="utf-8")

            result = self.run_cli("normalize", str(source), "--dry-run")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_text(encoding="utf-8"), before)
            self.assertEqual(list(Path(temporary).iterdir()), [source])

    def test_normalize_writes_a_sibling_and_keeps_the_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.transcript(Path(temporary))
            before = source.read_text(encoding="utf-8")

            result = self.run_cli("normalize", str(source))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_text(encoding="utf-8"), before)
            output = Path(temporary) / "meeting.gigaam.transcript.normalized.json"
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["segments"][0]["text"], "prompt готов")
            self.assertEqual(data["segments"][0]["words"][0]["word"], "prompt")
            self.assertEqual(data["metadata"]["normalized"]["lexicon_version"], 1)

    def test_plan_ranks_candidates_without_editing(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.transcript(Path(temporary))
            before = source.read_text(encoding="utf-8")

            result = self.run_cli("plan", str(source), "--budget", "5")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("кандидатов:", result.stdout)
            self.assertEqual(source.read_text(encoding="utf-8"), before)


class LexiconCliTests(unittest.TestCase):
    """Команды `lexicon add/list/remove/prompt` и подхват словаря в normalize."""

    def run_cli(self, home: Path, *args) -> subprocess.CompletedProcess:
        env = {**os.environ, "HOME": str(home)}
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

    def lexicon_path(self, home: Path) -> Path:
        return home / ".config" / "meeting-transcribe" / "lexicon.json"

    def test_add_cyrillic_defaults_to_stems_and_merges_forms(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)

            first = self.run_cli(home, "lexicon", "add", "капибара", "Capybara")
            second = self.run_cli(home, "lexicon", "add", "капибарра", "Capybara")

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            data = json.loads(self.lexicon_path(home).read_text(encoding="utf-8"))
            self.assertEqual(len(data["terms"]), 1)
            self.assertEqual(data["terms"][0]["policy"], "auto")
            # Флексия отрезана: «капибара»/«капибару» — одна основа «капибар».
            self.assertEqual(data["terms"][0]["ru_stems"],
                             ["капибар", "капибарр"])

    def test_add_latin_goes_to_latin_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)

            result = self.run_cli(home, "lexicon", "add", "Geathab", "GitHub")

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(self.lexicon_path(home).read_text(encoding="utf-8"))
            self.assertEqual(data["terms"][0]["latin_variants"], ["Geathab"])
            self.assertEqual(data["terms"][0]["ru_stems"], [])

    def test_add_suggest_marks_instead_of_replacing(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)

            result = self.run_cli(home, "lexicon", "add", "рага", "RAG", "--suggest")

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(self.lexicon_path(home).read_text(encoding="utf-8"))
            self.assertEqual(data["terms"][0]["policy"], "suggest")
            self.assertEqual(data["terms"][0]["ru_exact"], ["рага"])

    def test_add_refuses_multiword_heard(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)

            result = self.run_cli(home, "lexicon", "add", "пул реквест", "PR")

            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.lexicon_path(home).exists())

    def test_prompt_lists_user_canonicals_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            self.run_cli(home, "lexicon", "add", "капибара", "Capybara")
            self.run_cli(home, "lexicon", "add", "рага", "RAG", "--suggest")

            result = self.run_cli(home, "lexicon", "prompt")

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Capybara", result.stdout)
            self.assertNotIn("RAG", result.stdout)

    def test_normalize_applies_user_lexicon_from_default_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            self.run_cli(home, "lexicon", "add", "капибара", "Capybara")
            source = Path(temporary) / "meeting.gigaam.transcript.json"
            source.write_text(
                json.dumps(
                    {"segments": [{"start": 0.0, "end": 2.0,
                                   "text": "капибару завели",
                                   "words": []}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = self.run_cli(home, "normalize", str(source))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("личный словарь", result.stdout)
            output = Path(temporary) / "meeting.gigaam.transcript.normalized.json"
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["segments"][0]["text"], "Capybara завели")
            self.assertEqual(
                data["metadata"]["normalized"]["user_lexicon"]["terms"], 1)

    def test_remove_drops_single_form_then_whole_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            self.run_cli(home, "lexicon", "add", "капибара", "Capybara")
            self.run_cli(home, "lexicon", "add", "капибарра", "Capybara")

            first = self.run_cli(home, "lexicon", "remove", "Capybara",
                                 "--heard", "капибарра")
            data = json.loads(self.lexicon_path(home).read_text(encoding="utf-8"))
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(data["terms"][0]["ru_stems"], ["капибар"])

            second = self.run_cli(home, "lexicon", "remove", "Capybara")
            data = json.loads(self.lexicon_path(home).read_text(encoding="utf-8"))
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(data["terms"], [])


if __name__ == "__main__":
    unittest.main()
