import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "transcribe.py"
SPEC = importlib.util.spec_from_file_location("transcribe_words", SCRIPT)
transcribe = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = transcribe
SPEC.loader.exec_module(transcribe)

DUAL = Path(__file__).resolve().parents[1] / "scripts" / "dual_transcribe.py"
DUAL_SPEC = importlib.util.spec_from_file_location("dual_words", DUAL)
dual = importlib.util.module_from_spec(DUAL_SPEC)
assert DUAL_SPEC and DUAL_SPEC.loader
sys.modules[DUAL_SPEC.name] = dual
DUAL_SPEC.loader.exec_module(dual)


def token(text, start_ms, end_ms, p=0.9):
    return {"text": text, "offsets": {"from": start_ms, "to": end_ms}, "p": p}


class TokensToWordsTests(unittest.TestCase):
    def test_groups_bpe_pieces_into_words(self):
        words = transcribe._tokens_to_words([
            token(" При", 1000, 1200),
            token("вет", 1200, 1400),
            token(" мир", 1400, 1800),
        ])

        self.assertEqual([w["word"] for w in words], ["Привет", "мир"])
        self.assertEqual(words[0]["start"], 1.0)
        self.assertEqual(words[0]["end"], 1.4)

    def test_service_markers_are_dropped(self):
        words = transcribe._tokens_to_words([
            token("[_BEG_]", 0, 0),
            token(" Да", 500, 700),
            token("[_TT_120]", 700, 700),
        ])

        self.assertEqual([w["word"] for w in words], ["Да"])

    def test_confidence_is_the_mean_over_tokens(self):
        words = transcribe._tokens_to_words([
            token(" При", 1000, 1200, p=0.6),
            token("вет", 1200, 1400, p=0.8),
        ])

        self.assertAlmostEqual(words[0]["conf"], 0.7, places=4)

    def test_backwards_token_offsets_never_produce_end_before_start(self):
        # Реальный вывод whisper.cpp без --dtw: `from` застревает на начале
        # сегмента, `to` оказывается раньше него. На 24-минутном созвоне так
        # получилось 172 слова с end < start, и вся сверка падала на валидации.
        words = transcribe._tokens_to_words([
            token(" Да,", 121800, 121320),
            token(" я", 121800, 121450),
            token(" думаю", 121450, 122100),
        ])

        for word in words:
            with self.subTest(word=word["word"]):
                self.assertGreaterEqual(word["end"], word["start"])
        starts = [word["start"] for word in words]
        self.assertEqual(starts, sorted(starts))

    def test_repaired_words_pass_the_comparison_validator(self):
        words = transcribe._tokens_to_words([
            token(" Да,", 121800, 121320),
            token(" я", 121800, 121450),
            token(" думаю", 121450, 122100),
        ])
        artifact = {
            "metadata": {"source_file": "meeting.wav"},
            "segments": [{
                "start": 121.32,
                "end": 122.1,
                "text": "Да, я думаю",
                "words": words,
            }],
        }

        dual.validate_transcript(
            artifact, "meeting.wav", 600.0, require_word_timestamps=True
        )


if __name__ == "__main__":
    unittest.main()
