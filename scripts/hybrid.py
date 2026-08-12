#!/usr/bin/env python3
"""
Гибридная сверка транскрипта: приведение терминов к каноническому написанию.

Подкоманда `normalize` чинит то, что GigaAM транслитерирует на слух
(«гардрейл» → guardrail, «пронт» → prompt) и то, что оба движка коверкают
внутри латиницы («Githab» → GitHub, «Impoot» → input).

Почему по основам, а не по словоформам: на корпусе из 6 лекций (37669 слов)
ручная сверка привела к канону «гардрейл», «пул-реквест» и «ЛМ» на 100%, но
из 87 вхождений «промт/пронт» в финальный текст просочились 56 — и это были
именно склонённые формы (промта, промтом, промте, промтов, промтами). Русский
язык флективный: словарь, ключом которого служит словоформа, систематически
теряет парадигму.

Инварианты:
  * заменяются только токены, попавшие в словарь; всё остальное побайтово цело;
  * пробелы и пунктуация исходного текста сохраняются;
  * термины с policy="suggest" НИКОГДА не заменяются автоматически — они
    омонимичны реальным русским словам («рага», «митра», «аппы») и только
    помечаются;
  * каждая правка попадает в отчёт с тайм-кодом.

Использование:
    python hybrid.py normalize <transcript.json> [--lexicon PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

DEFAULT_LEXICON = Path(__file__).parent / "lexicons" / "terms.json"

# Максимальная длина русского окончания, дописываемого к основе.
MAX_ENDING = 4
# Порог похожести для латинских искажений, не перечисленных в словаре явно.
FUZZY_THRESHOLD = 0.82
# Короткие латинские токены слишком легко «починить» не туда.
MIN_FUZZY_LEN = 5
# Системный словарь английского: если слово в нём есть, нечёткую замену не
# применяем. Искажения, которые сами являются английскими словами («Reword»,
# «Impot», «graff»), вынесены в лексиконе в latin_suggest и тоже не заменяются
# автоматически — на другой записи то же слово может стоять по назначению.
_WORDS_FILE = Path("/usr/share/dict/words")


def _load_english_words() -> frozenset[str]:
    try:
        return frozenset(
            line.strip().lower()
            for line in _WORDS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        )
    except OSError:
        return frozenset()


ENGLISH_WORDS = _load_english_words()

CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
# Окончание — это цепочка кириллицы целиком, а не один символ: «промтом»,
# «промтов», «промтами» должны чиниться так же, как «промта».
CYRILLIC_RUN = re.compile(r"[а-яё]+")
LATIN = re.compile(r"[A-Za-z]")
# Пунктуация, которую снимаем с краёв токена и возвращаем на место после замены.
EDGE = "«»\"'()[]{}.,;:!?…-—–"


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def norm_ru(text: str) -> str:
    """Регистр и ё→е — чтобы «Гардрейлы» и «гардрейлы» шли одной основой."""
    return text.lower().replace("ё", "е")


def split_edges(token: str) -> tuple[str, str, str]:
    """Разложить токен на (левая пунктуация, ядро, правая пунктуация)."""
    left = 0
    while left < len(token) and token[left] in EDGE:
        left += 1
    right = len(token)
    while right > left and token[right - 1] in EDGE:
        right -= 1
    return token[:left], token[left:right], token[right:]


class Lexicon:
    def __init__(self, data: dict):
        self.version = data.get("version")
        self.terms = data.get("terms", [])
        self._stems: list[tuple[str, str]] = []      # (основа, канон)
        self._exact: dict[str, str] = {}             # словоформа → канон
        self._latin: dict[str, str] = {}             # искажение → канон
        self._suggest_exact: dict[str, str] = {}
        self._fuzzy_targets: list[str] = []

        for term in self.terms:
            canonical = term["canonical"]
            auto = term.get("policy", "auto") == "auto"
            if auto:
                for stem in term.get("ru_stems", []):
                    self._stems.append((norm_ru(stem), canonical))
            for form in term.get("ru_exact", []):
                target = self._exact if auto else self._suggest_exact
                target[norm_ru(form)] = canonical
            for variant in term.get("latin_variants", []):
                if auto:
                    self._latin[variant.lower()] = canonical
            for variant in term.get("latin_suggest", []):
                self._suggest_exact[variant.lower()] = canonical
            if auto and LATIN.search(canonical):
                self._fuzzy_targets.append(canonical)

        # Длинные основы проверяем первыми: «гвардрейл» должен выиграть у «гард».
        self._stems.sort(key=lambda pair: -len(pair[0]))

    def match(self, core: str) -> tuple[Optional[str], str]:
        """Вернуть (канон или None, причина)."""
        if not core:
            return None, ""
        low = norm_ru(core)

        if low in self._suggest_exact:
            return None, f"suggest:{self._suggest_exact[low]}"

        if low in self._exact:
            return self._exact[low], "exact"

        if CYRILLIC.search(core):
            for stem, canonical in self._stems:
                if low.startswith(stem):
                    ending = low[len(stem):]
                    if len(ending) <= MAX_ENDING and (
                        not ending or CYRILLIC_RUN.fullmatch(ending)
                    ):
                        return canonical, "stem"
            return None, ""

        if LATIN.search(core):
            if low in self._latin:
                return self._latin[low], "latin"
            if len(low) < MIN_FUZZY_LEN or low in ENGLISH_WORDS:
                return None, ""
            best, score = None, 0.0
            for target in self._fuzzy_targets:
                tl = target.lower()
                if tl == low:
                    return None, ""          # уже канон
                if tl[0] != low[0]:
                    continue                  # первая буква обязана совпасть
                # Отличие только хвостом — это словоформа, а не искажение:
                # tools/tool, models/model, learn/learning. Такое не трогаем.
                if low.startswith(tl) or tl.startswith(low):
                    continue
                ratio = SequenceMatcher(None, low, tl).ratio()
                if ratio > score:
                    best, score = target, ratio
            if best and score >= FUZZY_THRESHOLD:
                return best, f"fuzzy:{score:.2f}"
        return None, ""


def restore_case(canonical: str, original_core: str, reason: str) -> str:
    """Канон пишется так, как записан в словаре.

    Заглавную наследуем только от кириллических совпадений: там заглавная —
    осмысленный признак начала фразы («Гардрейл» → «Guardrail»). У латинского
    мусора регистр сам по себе шум («Impot», «Modul» посреди предложения), и
    наследовать его значит тиражировать ошибку.
    """
    if reason in {"stem", "exact"} and original_core[:1].isupper() and canonical[:1].islower():
        return canonical[:1].upper() + canonical[1:]
    return canonical


def normalize_text(text: str, lex: Lexicon, sink: list) -> str:
    """Заменить термины, сохранив пробелы и пунктуацию исходника."""
    parts = re.split(r"(\s+)", text)
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue
        left, core, right = split_edges(part)
        canonical, reason = lex.match(core)

        # Составное слово чиним по частям: «промт-инъекция» → «prompt-инъекция».
        # Целиком заменять нельзя — потеряется вторая половина термина.
        if not canonical and not reason and "-" in core:
            chunks = core.split("-")
            rebuilt, touched, why = [], False, ""
            for chunk in chunks:
                sub, sub_reason = lex.match(chunk)
                if sub and not sub_reason.startswith("suggest:") \
                        and norm_ru(sub) != norm_ru(chunk):
                    rebuilt.append(restore_case(sub, chunk, sub_reason))
                    touched, why = True, sub_reason
                else:
                    rebuilt.append(chunk)
            if touched:
                replacement = "-".join(rebuilt)
                parts[i] = left + replacement + right
                sink.append({"kind": "replace", "from": core,
                             "to": replacement, "reason": f"compound/{why}"})
            continue

        if reason.startswith("suggest:"):
            sink.append({"kind": "suggest", "from": core,
                         "canonical": reason.split(":", 1)[1]})
            continue
        if canonical and norm_ru(canonical) != norm_ru(core):
            replacement = restore_case(canonical, core, reason)
            parts[i] = left + replacement + right
            sink.append({"kind": "replace", "from": core,
                         "to": replacement, "reason": reason})
    return "".join(parts)


def cmd_normalize(args: argparse.Namespace) -> int:
    src = args.transcript.expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8"))
    lex = Lexicon(json.loads(args.lexicon.expanduser().read_text(encoding="utf-8")))

    replacements: list[dict] = []
    suggestions: list[dict] = []

    for seg in data.get("segments", []):
        sink: list[dict] = []
        seg_text = normalize_text(seg.get("text", ""), lex, sink)
        for word in seg.get("words") or []:
            wsink: list[dict] = []
            word["word"] = normalize_text(word.get("word", ""), lex, wsink)
        if not args.dry_run:
            seg["text"] = seg_text
        for event in sink:
            event["at"] = fmt_ts(seg.get("start", 0.0))
            (suggestions if event["kind"] == "suggest" else replacements).append(event)

    counts: dict[str, int] = {}
    for event in replacements:
        key = f'{event["from"]} → {event["to"]}'
        counts[key] = counts.get(key, 0) + 1

    total_words = sum(len(s.get("text", "").split()) for s in data.get("segments", []))
    print(f"слов в транскрипте: {total_words}")
    print(f"замен: {len(replacements)} ({len(counts)} уникальных пар)")
    print(f"помечено для проверки: {len(suggestions)}")
    if counts:
        print("\n--- частые замены ---")
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:25]:
            print(f"  {n:4d}  {key}")
    if suggestions:
        print("\n--- требуют решения (омонимичны русским словам) ---")
        seen: set[str] = set()
        for event in suggestions:
            key = f'{event["from"]} → {event["canonical"]}?'
            if key not in seen:
                seen.add(key)
                print(f"  [{event['at']}] {key}")

    if args.dry_run:
        print("\n(dry-run: файлы не изменялись)")
        return 0

    data.setdefault("metadata", {})["normalized"] = {
        "lexicon_version": lex.version,
        "replacements": len(replacements),
        "suggestions": len(suggestions),
    }
    out = src.with_suffix("")
    out = out.parent / (out.name + ".normalized.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    norm = sub.add_parser("normalize", help="привести термины к каноническому написанию")
    norm.add_argument("transcript", type=Path)
    norm.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    norm.add_argument("--dry-run", action="store_true",
                      help="показать, что изменится, не трогая файлы")
    norm.set_defaults(func=cmd_normalize)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
