#!/usr/bin/env python3
"""
Гибридная сверка транскрипта: приведение терминов к каноническому написанию.

Подкоманда `normalize` чинит то, что GigaAM транслитерирует на слух
(«гардрейл» → guardrail, «пронт» → prompt) и то, что оба движка коверкают
внутри латиницы («Githab» → GitHub, «Impoot» → input).

Почему по основам, а не по словоформам: на корпусе из 6 лекций (37669 слов)
прошлая сверка с Whisper (metadata.term_normalization = "whisper-large-v3
cross-check") привела к канону «гардрейл», «пул-реквест» и «ЛМ» на 100%, но из
87 вхождений «промт/пронт» в финальный текст просочились 56 — и это были именно
склонённые формы (промта, промтом, промте, промтов, промтами). Русский язык
флективный: словарь, ключом которого служит словоформа, систематически теряет
парадигму.

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
            # Без системного словаря английского нечёткую замену не делаем:
            # именно он не даёт «починить» настоящее английское слово в
            # похожий термин. На Linux/Windows файла обычно нет — там
            # остаются только явные latin_variants из словаря.
            if not ENGLISH_WORDS or len(low) < MIN_FUZZY_LEN or low in ENGLISH_WORDS:
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


# -------------- ДЕТЕКТОР ПОДОЗРИТЕЛЬНЫХ МЕСТ --------------

# Латинские токены, которые GigaAM пишет верно: помечать их — шум.
LATIN_OK = {
    "llm", "gpt", "mcp", "http", "https", "api", "ai", "rag", "ci", "cd", "ui", "ux",
    "sql", "json", "yaml", "xml", "html", "css", "js", "ts", "url", "pdf", "csv",
    "gpu", "cpu", "ram", "sdk", "ide", "os", "vpn", "dns", "ssh", "tls", "jwt",
    "gigachat", "github", "openai", "docker", "kubernetes", "python", "java", "go",
    "react", "typescript", "linux", "windows", "macos", "ios", "android", "aws",
}
# Филлеры и разговорные обрывки — не ошибки распознавания.
FILLERS = {
    "ну", "вот", "это", "как", "бы", "типа", "щас", "че", "чё", "ага", "угу", "мм",
    "хмм", "нда", "э", "эм", "а", "и", "но", "да", "нет", "так", "там", "то",
}

HAS_DIGIT = re.compile(r"\d")


def detect_signals(word: str, conf: Optional[float], low_cut: Optional[float],
                   very_low_cut: Optional[float], lex: Lexicon) -> tuple[int, list[str]]:
    """Вернуть (вес, список сигналов) для одного слова."""
    _, core, _ = split_edges(word)
    if not core:
        return 0, []
    low = norm_ru(core)
    if low in FILLERS or len(core) < 4:
        return 0, []
    # То, что уже чинится словарём, кандидатом не является.
    canonical, reason = lex.match(core)
    if canonical and not reason.startswith("suggest:"):
        return 0, []

    score, signals = 0, []
    mixed = re.search(r"[а-яёА-ЯЁ]", core) and re.search(r"[A-Za-z]", core)
    if mixed:
        # «MCP-сервер», «SSH-серверу», «LLM-модель» — законные сложные слова:
        # известный акроним латиницей плюс русское слово через дефис. Помечать
        # их значит топить отчёт в шуме. Подозрительно другое: либо акроним
        # неизвестен («NSF-контент», «I-продукта»), либо алфавиты смешаны
        # ВНУТРИ одной части слова («имaдж» с латинской a) — это гомоглиф,
        # и он всегда сбой.
        chunks = [c for c in re.split(r"[-–—]", core) if c]
        homoglyph = any(
            re.search(r"[а-яёА-ЯЁ]", c) and re.search(r"[A-Za-z]", c) for c in chunks
        )
        unknown_acronym = any(
            re.fullmatch(r"[A-Za-z]+", c) and norm_ru(c) not in LATIN_OK for c in chunks
        )
        if homoglyph:
            score += 4
            signals.append("M")      # алфавиты смешаны внутри одной части слова
        elif unknown_acronym:
            score += 2
            signals.append("M-")     # акроним не из белого списка
    elif re.fullmatch(r"[A-Za-z][A-Za-z.-]*", core):
        if low not in LATIN_OK:
            score += 2
            signals.append("L")      # латиница вне белого списка
            if low not in ENGLISH_WORDS:
                score += 1
                signals.append("L+")  # и слова такого в английском нет
    if reason.startswith("suggest:"):
        score += 3
        signals.append("S")          # омоним из словаря, решение за человеком
    if conf is not None and very_low_cut is not None and conf <= very_low_cut:
        score += 2
        signals.append("C")          # уверенность в нижних процентах
    elif conf is not None and low_cut is not None and conf <= low_cut:
        score += 1
        signals.append("C-")
    if HAS_DIGIT.search(core) and conf is not None and low_cut is not None and conf <= low_cut:
        score += 1
        signals.append("N")          # число под сомнением
    return score, signals


def cmd_plan(args: argparse.Namespace) -> int:
    src = args.transcript.expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8"))
    lex = Lexicon(json.loads(args.lexicon.expanduser().read_text(encoding="utf-8")))

    words: list[dict] = []
    for seg in data.get("segments", []):
        for word in seg.get("words") or []:
            words.append({"word": word.get("word", ""), "start": word.get("start", seg.get("start", 0.0)),
                          "conf": word.get("conf")})
        if not (seg.get("words") or []):
            for token in seg.get("text", "").split():
                words.append({"word": token, "start": seg.get("start", 0.0), "conf": seg.get("conf")})

    confs = sorted(w["conf"] for w in words if w["conf"] is not None)
    low_cut = confs[int(len(confs) * 0.15)] if confs else None
    very_low_cut = confs[int(len(confs) * 0.05)] if confs else None
    if not confs:
        print("! в транскрипте нет поля conf — канал уверенности выключен", file=sys.stderr)

    # Кластеры вариантов: разные написания с одинаковым согласным скелетом.
    skeleton = lambda s: re.sub(r"[аеиоуыэюяaeiouy\W_]", "", norm_ru(s))
    by_skel: dict[str, set[str]] = {}
    for w in words:
        _, core, _ = split_edges(w["word"])
        if len(core) >= 5:
            by_skel.setdefault(skeleton(core), set()).add(norm_ru(core))

    candidates = []
    for w in words:
        score, signals = detect_signals(w["word"], w["conf"], low_cut, very_low_cut, lex)
        if not score:
            continue
        _, core, _ = split_edges(w["word"])
        variants = by_skel.get(skeleton(core), set())
        if len(variants) > 1:
            score += 3
            signals.append("V")      # тот же термин записан по-разному
        candidates.append({"at": fmt_ts(w["start"]), "start": w["start"], "word": core,
                           "conf": w["conf"], "score": score, "signals": signals,
                           "variants": sorted(variants) if len(variants) > 1 else []})

    # Один термин — одна строка. Иначе «MCP-сервер», встретившийся 10 раз,
    # занимает весь бюджет и вытесняет остальные находки.
    grouped: dict[str, dict] = {}
    for c in candidates:
        key = norm_ru(c["word"])
        if key in grouped:
            grouped[key]["count"] += 1
        else:
            grouped[key] = {**c, "count": 1}
    candidates = sorted(grouped.values(), key=lambda c: (-c["score"], -c["count"], c["start"]))
    top = candidates[: args.budget]

    print(f"слов: {len(words)}   кандидатов: {len(candidates)}   в бюджете: {len(top)}")
    if confs:
        print(f"порог уверенности: p05={very_low_cut:.4f}  p15={low_cut:.4f}")
    print(f"\n{'время':>9s} {'вес':>4s} {'×':>3s} {'conf':>7s}  сигналы     слово")
    for c in top:
        conf = f"{c['conf']:.4f}" if c["conf"] is not None else "  —  "
        marks = ",".join(c["signals"])
        extra = f"   ← варианты: {', '.join(c['variants'])}" if c["variants"] else ""
        print(f"{c['at']:>9s} {c['score']:>4d} {c['count']:>3d} {conf:>7s}  "
              f"{marks:<11s} {c['word']}{extra}")

    if args.json_out:
        args.json_out.write_text(json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.json_out}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    src = args.transcript.expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8"))
    lex = Lexicon(json.loads(args.lexicon.expanduser().read_text(encoding="utf-8")))

    replacements: list[dict] = []
    suggestions: list[dict] = []

    for seg in data.get("segments", []):
        sink: list[dict] = []
        seg_text = normalize_text(seg.get("text", ""), lex, sink)
        # Слова чиним тем же словарём, но события отчёта берём только из
        # текста сегмента: слова — те же токены, и второй сток удвоил бы счёт.
        word_events: list[dict] = []
        words = [
            normalize_text(word.get("word", ""), lex, word_events)
            for word in seg.get("words") or []
        ]
        if not args.dry_run:
            seg["text"] = seg_text
            for word, replacement in zip(seg.get("words") or [], words):
                word["word"] = replacement
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

    plan = sub.add_parser("plan", help="найти места, которые словарь не чинит")
    plan.add_argument("transcript", type=Path)
    plan.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    plan.add_argument("--budget", type=int, default=40,
                      help="сколько кандидатов оставить (по убыванию веса)")
    plan.add_argument("--json-out", type=Path, default=None)
    plan.set_defaults(func=cmd_plan)

    args = ap.parse_args()
    if not ENGLISH_WORDS:
        print(
            f"! {_WORDS_FILE} недоступен — нечёткая сверка латиницы выключена; "
            "чинятся только явные варианты из словаря",
            file=sys.stderr,
        )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
