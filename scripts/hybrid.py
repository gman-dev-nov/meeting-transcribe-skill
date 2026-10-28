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
    python hybrid.py plan <transcript.json> [--budget N]
    python hybrid.py lexicon add "как слышится" "канон" [--exact|--suggest]
    python hybrid.py lexicon list | remove <канон> | prompt
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
# Личный словарь живёт вне каталога скилла: обновление через check_update.py —
# это git-pull, который отказывается работать при локальных правках, поэтому
# файл внутри скилла либо потерялся бы при обновлении, либо блокировал бы его.
DEFAULT_USER_LEXICON = Path("~/.config/meeting-transcribe/lexicon.json").expanduser()

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


# -------------- ЛИЧНЫЙ СЛОВАРЬ ПОВЕРХ ВСТРОЕННОГО --------------

_VARIANT_FIELDS = ("ru_stems", "ru_exact", "latin_variants", "latin_suggest")


def _variant_key(field: str, variant: str) -> str:
    """Ключ сравнения словарных форм: кириллицу — по norm_ru, латиницу — по lower."""
    return norm_ru(variant) if field.startswith("ru_") else variant.lower()


# Флексии, которые отрезаются от словарной формы при записи основы.
_STEM_FLEXION = set("аяоеыиэуюьй")


def _to_stem(form: str) -> str:
    """Отрезать флексию от словарной формы, чтобы основа покрыла склонение.

    Женские и мягкие формы меняют гласную внутри слова, а не дописывают
    окончание: «капибара → капибару», «джира → джиру». Поэтому основой
    служит согласный каркас: капибар, джир — «капибару» и «джире» тогда
    матчатся как основа+окончание. Мужские согласные основы («промт»)
    не меняются.
    """
    last = form[-1:].lower().replace("ё", "е")
    if len(form) > 3 and last in _STEM_FLEXION:
        return form[:-1]
    return form


def merge_terms(built_in_terms: list[dict], user_terms: list[dict]) -> list[dict]:
    """Наложить личный словарь на встроенный.

    Правила наложения:
      * словарная форма из личного словаря перекрывает встроенную — в том
        числе переводит её из «suggest» в автоматическую замену;
      * написание канона берётся из личной записи;
      * policy принадлежит форме, а не канону: если пользователь добавил
        «аппы → API» как auto, встроенное «аппы → API (suggest)» перестаёт
        действовать, остальные suggest-формы того же канона остаются.

    Выход может содержать два терма на один канон (auto и suggest) — Lexicon
    обрабатывает их независимо.
    """
    user_keys = {
        _variant_key(field, variant)
        for term in user_terms
        for field in _VARIANT_FIELDS
        for variant in term.get(field) or []
    }
    stripped = []
    for term in built_in_terms:
        term = dict(term)
        for field in _VARIANT_FIELDS:
            term[field] = [v for v in term.get(field) or []
                           if _variant_key(field, v) not in user_keys]
        stripped.append(term)

    buckets: dict[str, dict] = {}
    for term in stripped + user_terms:
        bucket = buckets.setdefault(term["canonical"].lower(), {
            "canonical": term["canonical"],
            "auto": {"ru_stems": [], "ru_exact": [], "latin_variants": []},
            "suggest": {"ru_exact": [], "latin_suggest": []},
        })
        bucket["canonical"] = term["canonical"]      # личная запись идёт последней
        if term.get("policy", "auto") == "auto":
            for field in ("ru_stems", "ru_exact", "latin_variants"):
                bucket["auto"][field].extend(term.get(field) or [])
        else:
            bucket["suggest"]["ru_exact"].extend(term.get("ru_exact") or [])
            bucket["suggest"]["latin_suggest"].extend(term.get("latin_variants") or [])
        # latin_suggest — всегда suggest, независимо от policy терма.
        bucket["suggest"]["latin_suggest"].extend(term.get("latin_suggest") or [])

    def dedupe(values: list[str], field: str) -> list[str]:
        seen, out = set(), []
        for value in values:
            key = _variant_key(field, value)
            if key not in seen:
                seen.add(key)
                out.append(value)
        return out

    result = []
    for bucket in buckets.values():
        auto = {f: dedupe(vs, f) for f, vs in bucket["auto"].items()}
        suggest = {f: dedupe(vs, f) for f, vs in bucket["suggest"].items()}
        if any(auto.values()):
            result.append({"canonical": bucket["canonical"], "policy": "auto", **auto})
        if any(suggest.values()):
            result.append({"canonical": bucket["canonical"], "policy": "suggest",
                           "ru_stems": [], "ru_exact": suggest["ru_exact"],
                           "latin_variants": [], "latin_suggest": suggest["latin_suggest"]})
    return result


def _resolve_lexicons(args: argparse.Namespace) -> tuple[Lexicon, dict]:
    """Собрать лексикон: встроенный словарь + личный поверх него.

    Личный словарь ищется по --user-lexicon, иначе — в стандартном месте
    (~/.config/meeting-transcribe/lexicon.json), если файл существует.
    --no-user-lexicon отключает автозагрузку.
    """
    built_in = json.loads(args.lexicon.expanduser().read_text(encoding="utf-8"))
    user_path: Optional[Path] = None
    if getattr(args, "user_lexicon", None) is not None:
        user_path = args.user_lexicon.expanduser()
        if not user_path.exists():
            raise SystemExit(f"! {user_path} не существует")
    elif not getattr(args, "no_user_lexicon", False) and DEFAULT_USER_LEXICON.exists():
        user_path = DEFAULT_USER_LEXICON
    user = json.loads(user_path.read_text(encoding="utf-8")) if user_path else {"terms": []}
    terms = merge_terms(built_in.get("terms", []), user.get("terms", []))
    info = {"built_in_version": built_in.get("version"),
            "user_path": user_path, "user_terms": len(user.get("terms", []))}
    return Lexicon({"version": built_in.get("version"), "terms": terms}), info


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
    lex, info = _resolve_lexicons(args)
    if info["user_path"]:
        print(f"личный словарь: {info['user_path']} ({info['user_terms']} терминов)")

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
    lex, info = _resolve_lexicons(args)
    if info["user_path"]:
        print(f"личный словарь: {info['user_path']} ({info['user_terms']} терминов)")

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

    normalized_meta = data.setdefault("metadata", {})["normalized"] = {
        "lexicon_version": lex.version,
        "replacements": len(replacements),
        "suggestions": len(suggestions),
    }
    if info["user_path"]:
        normalized_meta["user_lexicon"] = {"path": str(info["user_path"]),
                                           "terms": info["user_terms"]}
    out = src.with_suffix("")
    out = out.parent / (out.name + ".normalized.json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


# -------------- ЛИЧНЫЙ СЛОВАРЬ: ФАЙЛ И КОМАНДЫ --------------

def _empty_user_lexicon() -> dict:
    return {
        "version": 1,
        "note": "Личный словарь пользователя. Записи добавляются командой "
                "`hybrid.py lexicon add` и действуют во всех следующих "
                "транскрипциях; формы из этого файла перекрывают встроенный "
                "словарь скилла.",
        "terms": [],
    }


def _resolve_user_lexicon_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "user_lexicon", None)
    return path.expanduser() if path is not None else DEFAULT_USER_LEXICON


def _load_user_lexicon(path: Path) -> dict:
    if not path.exists():
        return _empty_user_lexicon()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"! {path} не читается как JSON: {exc}")
    if not isinstance(data.get("terms"), list):
        raise SystemExit(f"! {path}: поле terms должно быть списком")
    return data


def _save_user_lexicon(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")


_HOW_APPLIED = {
    "ru_stems": "по основе, вместе со склонениями (промта, промтом, капибару)",
    "ru_exact": "только точная словоформа",
    "latin_variants": "точная латинская форма",
    "latin_suggest": "только помечается, автоматически не заменяется",
}


def cmd_lexicon_add(args: argparse.Namespace) -> int:
    heard, canonical = args.heard.strip(), args.canonical.strip()
    if not heard or not canonical:
        print("! «как слышится» и «канон» не могут быть пустыми", file=sys.stderr)
        return 2
    if any(ch.isspace() for ch in heard):
        print("! «как слышится» должно быть одним словом: движок заменяет токены, "
              "а не фразы. Для «пул реквест» добавь каждую часть отдельной записью "
              "или пиши через дефис.", file=sys.stderr)
        return 2
    if norm_ru(heard) == norm_ru(canonical) or heard.lower() == canonical.lower():
        print(f"«{heard}» уже совпадает с каноном «{canonical}» — замена не нужна")
        return 0

    is_cyrillic = bool(CYRILLIC.search(heard))
    if args.suggest:
        field = "ru_exact" if is_cyrillic else "latin_suggest"
    elif is_cyrillic:
        field = "ru_exact" if args.exact else "ru_stems"
    else:
        field = "latin_variants"   # латиница и так сверяется точной формой
    policy = "suggest" if args.suggest else "auto"

    path = _resolve_user_lexicon_path(args)
    data = _load_user_lexicon(path)
    value = _to_stem(heard) if field == "ru_stems" else heard
    key = _variant_key(field, value)

    # Одна словарная форма принадлежит одному канону: из других записей её убираем.
    for term in data["terms"]:
        for other_field in _VARIANT_FIELDS:
            term[other_field] = [v for v in term.get(other_field) or []
                                 if _variant_key(other_field, v) != key]

    target = next(
        (t for t in data["terms"]
         if t["canonical"].lower() == canonical.lower()
         and t.get("policy", "auto") == policy),
        None)
    if target is None:
        target = {"canonical": canonical, "policy": policy, "ru_stems": [],
                  "ru_exact": [], "latin_variants": [], "latin_suggest": []}
        data["terms"].append(target)

    if key in {_variant_key(field, v) for v in target.get(field) or []}:
        print(f"уже записано: {heard} → {canonical}")
    else:
        target.setdefault(field, []).append(value)
        if policy == "suggest":
            how = "омоним: только помечается, автоматически не заменяется"
        else:
            how = _HOW_APPLIED[field]
            if field == "ru_stems" and value != heard:
                how += f"; основа: «{value}»"
        print(f"записано: {heard} → {canonical}  ({how})")
    _save_user_lexicon(path, data)
    print(f"→ {path}  (записей: {len(data['terms'])})")
    print("normalize и plan подхватят это автоматически; `lexicon prompt` "
          "соберёт строку терминов для --initial-prompt")
    return 0


def cmd_lexicon_remove(args: argparse.Namespace) -> int:
    path = _resolve_user_lexicon_path(args)
    if not path.exists():
        print(f"! {path} не существует — личный словарь пуст", file=sys.stderr)
        return 1
    data = _load_user_lexicon(path)
    matching = [t for t in data["terms"]
                if t["canonical"].lower() == args.canonical.lower()]
    if not matching:
        print(f"! «{args.canonical}» нет в личном словаре", file=sys.stderr)
        return 1
    if args.heard:
        # add хранит не введённую форму, а отрезанную основу — сверяемся с обеими.
        keys = {_variant_key("ru_exact", args.heard),
                _variant_key("ru_stems", _to_stem(args.heard)),
                args.heard.lower()}
        removed = 0
        for term in matching:
            for field in _VARIANT_FIELDS:
                before = len(term.get(field) or [])
                term[field] = [v for v in term.get(field) or []
                               if _variant_key(field, v) not in keys]
                removed += before - len(term[field])
        if not removed:
            print(f"! формы «{args.heard}» у {args.canonical} в словаре нет", file=sys.stderr)
            return 1
        drop = {id(t) for t in matching if not any(t.get(f) for f in _VARIANT_FIELDS)}
        data["terms"] = [t for t in data["terms"] if id(t) not in drop]
        print(f"удалена форма: {args.heard} (канон {args.canonical})")
    else:
        drop = {id(t) for t in matching}
        data["terms"] = [t for t in data["terms"] if id(t) not in drop]
        print(f"удалена запись: {args.canonical}")
    _save_user_lexicon(path, data)
    print(f"→ {path}")
    return 0


def cmd_lexicon_list(args: argparse.Namespace) -> int:
    built_in = json.loads(args.lexicon.expanduser().read_text(encoding="utf-8"))
    print(f"встроенный словарь: {args.lexicon} "
          f"({len(built_in.get('terms', []))} терминов)")
    path = _resolve_user_lexicon_path(args)
    data = _load_user_lexicon(path)
    if not data["terms"]:
        print(f"\nличный словарь пуст: {path}")
        print('добавить: python3 scripts/hybrid.py lexicon add "как слышится" "канон"')
        return 0
    print(f"личный словарь: {path} ({len(data['terms'])} записей)\n")
    for term in data["terms"]:
        forms = ", ".join(
            f"{v} ({field})" for field in _VARIANT_FIELDS
            for v in term.get(field) or [])
        mark = "  [suggest]" if term.get("policy", "auto") != "auto" else ""
        print(f"  {term['canonical']}{mark} ← {forms}")
    return 0


# Ограничения строки терминов: длинный initial-prompt расшатывает
# декодирование Whisper, поэтому она ограничена и по количеству терминов,
# и по общей длине.
LEXICON_PROMPT_LIMIT = 30
LEXICON_PROMPT_MAX_CHARS = 1000


def lexicon_prompt_text(
    canonicals: list[str],
    *,
    limit: int = LEXICON_PROMPT_LIMIT,
    max_chars: int = LEXICON_PROMPT_MAX_CHARS,
) -> str:
    """Собрать ограниченную строку «Термины: …» для --initial-prompt.

    Чистый helper, общий для `lexicon prompt` и dual_transcribe: не более
    `limit` уникальных (без учёта регистра) терминов и не более `max_chars`
    символов на всю строку. Термин, не влезающий целиком, пропускается —
    обрезанное слово в промпте хуже его отсутствия; за ним могут уместиться
    более короткие. Пустой список (или ничего не влезло) даёт "".
    """
    prefix, suffix = "Термины: ", "."
    seen: set[str] = set()
    selected: list[str] = []
    length = len(prefix) + len(suffix)
    for value in canonicals:
        if len(selected) >= limit:
            break
        term = " ".join(str(value).split())
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        piece = term if not selected else ", " + term
        if length + len(piece) > max_chars:
            continue
        selected.append(term)
        length += len(piece)
    if not selected:
        return ""
    return prefix + ", ".join(selected) + suffix


def cmd_lexicon_prompt(args: argparse.Namespace) -> int:
    path = _resolve_user_lexicon_path(args)
    data = _load_user_lexicon(path)
    canonicals: list[str] = [
        term["canonical"] for term in data["terms"]
        if term.get("policy", "auto") == "auto"
    ]
    if args.all:
        built_in = json.loads(args.lexicon.expanduser().read_text(encoding="utf-8"))
        canonicals.extend(
            term["canonical"]
            for term in sorted(built_in.get("terms", []),
                               key=lambda t: t["canonical"].lower())
        )
    if not canonicals:
        print(f"личный словарь пуст: {path}", file=sys.stderr)
        return 1
    prompt = lexicon_prompt_text(canonicals, limit=args.limit)
    if not prompt:
        print(f"! термины из {path} не уместились в лимит промпта", file=sys.stderr)
        return 1
    print(prompt)
    sys.stdout.flush()
    print("\nпередай строку в dual_transcribe.py run … --initial-prompt "
          "(вместе с именами участников); держись лимита ~30 терминов — "
          "длинный промпт расшатывает декодирование Whisper", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    norm = sub.add_parser("normalize", help="привести термины к каноническому написанию")
    norm.add_argument("transcript", type=Path)
    norm.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    norm.add_argument("--user-lexicon", type=Path, default=None,
                      help="путь к личному словарю (по умолчанию "
                           "~/.config/meeting-transcribe/lexicon.json, если существует)")
    norm.add_argument("--no-user-lexicon", action="store_true",
                      help="игнорировать личный словарь")
    norm.add_argument("--dry-run", action="store_true",
                      help="показать, что изменится, не трогая файлы")
    norm.set_defaults(func=cmd_normalize)

    plan = sub.add_parser("plan", help="найти места, которые словарь не чинит")
    plan.add_argument("transcript", type=Path)
    plan.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    plan.add_argument("--user-lexicon", type=Path, default=None)
    plan.add_argument("--no-user-lexicon", action="store_true")
    plan.add_argument("--budget", type=int, default=40,
                      help="сколько кандидатов оставить (по убыванию веса)")
    plan.add_argument("--json-out", type=Path, default=None)
    plan.set_defaults(func=cmd_plan)

    lex = sub.add_parser("lexicon",
                         help="личный словарь постоянных замен «как слышится → канон»")
    lex_sub = lex.add_subparsers(dest="lex_cmd", required=True)

    lex_add = lex_sub.add_parser("add", help="записать замену")
    lex_add.add_argument("heard", help="как модель слышит слово (одним словом)")
    lex_add.add_argument("canonical", help="как слово должно быть написано")
    lex_add.add_argument("--exact", action="store_true",
                         help="кириллица: только точная форма, без склонений")
    lex_add.add_argument("--suggest", action="store_true",
                         help="омоним: только помечать, не заменять автоматически")
    lex_add.add_argument("--user-lexicon", type=Path, default=None)
    lex_add.set_defaults(func=cmd_lexicon_add)

    lex_rm = lex_sub.add_parser("remove", help="удалить запись или одну форму")
    lex_rm.add_argument("canonical")
    lex_rm.add_argument("--heard", default=None,
                        help="удалить только эту форму, а не запись целиком")
    lex_rm.add_argument("--user-lexicon", type=Path, default=None)
    lex_rm.set_defaults(func=cmd_lexicon_remove)

    lex_ls = lex_sub.add_parser("list", help="показать содержимое словарей")
    lex_ls.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    lex_ls.add_argument("--user-lexicon", type=Path, default=None)
    lex_ls.set_defaults(func=cmd_lexicon_list)

    lex_prompt = lex_sub.add_parser(
        "prompt", help="собрать строку терминов для --initial-prompt")
    lex_prompt.add_argument("--all", action="store_true",
                            help="включить и термины встроенного словаря")
    lex_prompt.add_argument("--limit", type=int, default=30)
    lex_prompt.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    lex_prompt.add_argument("--user-lexicon", type=Path, default=None)
    lex_prompt.set_defaults(func=cmd_lexicon_prompt)

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
