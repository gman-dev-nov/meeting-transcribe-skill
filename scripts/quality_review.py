#!/usr/bin/env python3
"""Проверка самоисправлений и производная редакторская версия после двух ASR.

Смысловой review делает активная модель агента. Этот CLI проверяет покрытие,
привязки и подтверждения; сам не исправляет факты и не вызывает LLM/API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import dual_transcribe as dual

SCHEMA_VERSION = 1
BLOCK_SECONDS = 60.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: неподдерживаемая схема JSON")
    return data


def resolve_ref(value: str, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Пустая ссылка на артефакт")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def fields(data: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(data, dict) or set(data) - allowed or required - set(data):
        raise ValueError(f"{label}: неверные поля")


def publish(contents: dict, protected: set) -> None:
    """Все проверки закончены до записи; существующие входы никогда не заменяются."""
    paths = [Path(path).resolve() for path in contents]
    if len(set(paths)) != len(paths) or set(paths) & protected:
        raise ValueError("Выходной путь совпадает с исходным артефактом")
    staged, backups, published = [], {}, []
    try:
        for path, content in zip(paths, contents.values()):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not path.is_file():
                raise ValueError(f"Выходной путь занят не файлом: {path}")
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", delete=False) as stream:
                stream.write(content)
                staged.append((Path(stream.name), path))
            if path.exists():
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.backup.",
                                                 delete=False) as backup:
                    backup_path = Path(backup.name)
                backups[path] = backup_path
                shutil.copy2(path, backup_path)
        for temporary, destination in staged:
            temporary.replace(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            backup = backups.pop(destination, None)
            if backup:
                backup.replace(destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def canonical_transcript(path: Path, source_name: str, duration: float) -> dict:
    data = dual.load_transcript(path)
    dual.validate_transcript(data, source_name, duration)
    segments = []
    texts = []
    offset = 0
    for index, segment in enumerate(data.get("segments", [])):
        text = segment.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Текст сегмента должен быть строкой")
        if index:
            offset += 1
        segments.append({"index": index, "start": segment["start"], "end": segment["end"],
                         "char_start": offset, "char_end": offset + len(text)})
        texts.append(text)
        offset += len(text)
    return {"text": "\n".join(texts), "segments": segments}


def validate_comparison_sources(comparison: dict, comparison_path: Path) -> None:
    """Comparison обязан быть связан с raw ещё при исходной сверке, не позже."""
    data = {}
    for engine in ("gigaam", "whisper"):
        info = comparison["transcripts"][engine]
        raw_path = resolve_ref(info["artifact_ref"], comparison_path.parent)
        data[engine] = dual.load_transcript(raw_path)
        expected = info.get("segments_sha256")
        if not expected:
            raise ValueError("Comparison без fingerprints raw: пересоздайте его командой dual_transcribe.py compare и повторите semantic review")
        if expected != dual.transcript_segments_sha256(data[engine]):
            raise ValueError("Raw изменился после comparison; нужна новая сверка")
    fresh = dual.build_comparison(data["gigaam"], data["whisper"],
                                  Path(comparison["source_audio"]["ref"]),
                                  comparison["source_audio"]["duration_seconds"])
    if comparison["candidates"] != fresh["candidates"]:
        raise ValueError("Comparison не соответствует исходникам; нужна новая сверка")


def prepare(comparison_path: Path, discrepancy_review_path: Path,
            output_dir: Optional[Path] = None, base_engine: str = "gigaam") -> tuple[Path, Path]:
    comparison_path = comparison_path.expanduser().resolve()
    discrepancy_review_path = discrepancy_review_path.expanduser().resolve()
    comparison = read_json(comparison_path)
    validate_comparison_sources(comparison, comparison_path)
    semantic_review = read_json(discrepancy_review_path)
    if resolve_ref(semantic_review.get("comparison_file"), discrepancy_review_path.parent) != comparison_path:
        raise ValueError("Review относится к другому comparison")
    if base_engine not in ("gigaam", "whisper"):
        raise ValueError("Основа должна быть gigaam или whisper")
    # Повторно используем строгий валидатор semantic review, не публикуя его артефакты.
    with tempfile.TemporaryDirectory(prefix="meeting-quality-review-") as temporary:
        selected_path, _ = dual.render_review(discrepancy_review_path, Path(temporary) / "review")
        selected = read_json(selected_path)
    audio = comparison["source_audio"]
    duration = audio["duration_seconds"]
    refs = {"comparison": reference(comparison_path),
            "discrepancy_review": reference(discrepancy_review_path)}
    transcripts = {}
    for engine in ("gigaam", "whisper"):
        path = resolve_ref(comparison["transcripts"][engine]["artifact_ref"], comparison_path.parent)
        transcripts[engine] = canonical_transcript(path, audio["display_name"], duration)
        refs[engine] = reference(path)
    last = max([float(duration)] + [float(segment["end"])
               for item in transcripts.values() for segment in item["segments"]])
    count = max(1, int(math.ceil(last / BLOCK_SECONDS)))
    blocks = [{"id": f"B{index + 1:03d}", "start_seconds": index * BLOCK_SECONDS,
               "end_seconds": min(last, (index + 1) * BLOCK_SECONDS),
               "segment_indices": {engine: [] for engine in transcripts}}
              for index in range(count)]
    for engine, item in transcripts.items():
        for segment in item["segments"]:
            index = min(count - 1, int(segment["start"] // BLOCK_SECONDS))
            blocks[index]["segment_indices"][engine].append(segment["index"])
    base_text = transcripts[base_engine]["text"]
    hints = []
    for match in re.finditer(r"(?iu)\b(?:ой|точнее|вернее|поправлюсь|передумал[аи]?|"
                             r"нет[,.]?\s+(?:лучше|в|на|не|давайте))\b", base_text):
        hints.append({"start": match.start(), "end": match.end(), "cue": match.group(),
                      "meaning": "Подсказка для чтения; не доказательство ошибки или самоисправления"})
    stem = comparison_path.name.removesuffix(".comparison.json")
    if stem == comparison_path.name:
        stem = comparison_path.stem
    destination = (output_dir or comparison_path.parent).expanduser().resolve()
    manifest_path = destination / f"{stem}.quality-manifest.json"
    review_path = destination / f"{stem}.quality-review.json"
    manifest = {"schema_version": 1, "stem": stem, "refs": refs, "base_engine": base_engine,
                "source_audio": audio, "transcripts": transcripts, "blocks": blocks,
                "discrepancies": selected["discrepancies"], "hints": hints}
    manifest_content = dump(manifest)
    template = {"schema_version": 1, "manifest_file": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_content.encode()).hexdigest(),
                "reviewed_blocks": [], "findings": [], "edits": []}
    protected = {Path(item["path"]) for item in refs.values()}
    # Подготовленный review содержит работу агента; повторный prepare её не стирает.
    if manifest_path.exists() or review_path.exists():
        raise ValueError("Quality manifest/review уже существует; выберите новый output-dir")
    publish({manifest_path: manifest_content, review_path: dump(template)}, protected)
    return manifest_path, review_path


def span(data: dict, text: str) -> dict:
    fields(data, {"start", "end", "from"}, {"start", "end", "from"}, "span")
    start, end = data["start"], data["end"]
    if (type(start) is not int or type(end) is not int or
            not 0 <= start <= end <= len(text) or data["from"] != text[start:end]):
        raise ValueError("span: неверные offsets или исходный from")
    return dict(data)


def source_interval(patch: dict, segments: list) -> dict:
    overlaps = [s for s in segments if s["char_end"] > patch["start"] and s["char_start"] < patch["end"]]
    if not overlaps:
        overlaps = [s for s in segments if s["char_start"] <= patch["start"] <= s["char_end"]]
    if not overlaps and segments:
        overlaps = [min(segments, key=lambda s: abs(s["char_start"] - patch["start"]))]
    return {"basis": "source_audio", "start_seconds": min((s["start"] for s in overlaps), default=0),
            "end_seconds": max((s["end"] for s in overlaps), default=0), "precision": "segment"}


def interval_span(interval: dict, transcript: dict) -> dict:
    segments = transcript["segments"]
    matches = [s for s in segments if s["end"] > interval["start_seconds"] and
               s["start"] < interval["end_seconds"]]
    if matches:
        start, end = matches[0]["char_start"], matches[-1]["char_end"]
    else:
        following = [s for s in segments if s["start"] >= interval["end_seconds"]]
        start = end = following[0]["char_start"] if following else len(transcript["text"])
    return {"start": start, "end": end, "from": transcript["text"][start:end]}


def lexical(text: str) -> list:
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def numeric_mentions(text: str) -> list:
    # Пунктуация может менять знак, диапазон и десятичную дробь, не меняя слов.
    return re.findall(r"(?:[<>≤≥=+\-−×*/]\s*)?\d+(?:[.,:/+\-−–—]\d+)*"
                      r"(?:\s*(?:[%₽$€]|[^\W\d_]+))?", text)


def protected_symbols(text: str) -> list:
    return [ch for ch in text if unicodedata.category(ch) in ("Sc", "Sm") or ch in "%‰№§°−±/*·⁄∕^"]


def compound_notations(text: str) -> list:
    # Разделитель внутри домена, единицы или обозначения — часть сущности.
    return re.findall(r"\w+(?:[./:@#%+*·⁄∕^−-]\w+)+", text, re.UNICODE)


def intersects(first: dict, second: dict) -> bool:
    if first["start"] == first["end"] or second["start"] == second["end"]:
        return max(first["start"], second["start"]) <= min(first["end"], second["end"])
    return max(first["start"], second["start"]) < min(first["end"], second["end"])


def validate_patches(patches: list) -> None:
    ordered = sorted(patches, key=lambda p: (p["span"]["start"], p["span"]["end"]))
    for previous, current in zip(ordered, ordered[1:]):
        if intersects(previous["span"], current["span"]):
            raise ValueError("Правки пересекаются; нужна одна точная согласованная правка")


def nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: требуется непустая строка")
    return value.strip()


def load_manifest(review_path: Path) -> tuple[dict, dict, Path]:
    review = read_json(review_path)
    allowed = {"schema_version", "manifest_file", "manifest_sha256", "reviewed_blocks", "findings", "edits"}
    fields(review, allowed, allowed, "quality review")
    manifest_path = resolve_ref(review["manifest_file"], review_path.parent)
    if digest(manifest_path) != review["manifest_sha256"]:
        raise ValueError("Manifest изменился после подготовки review")
    manifest = read_json(manifest_path)
    if set(manifest["refs"]) != {"comparison", "discrepancy_review", "gigaam", "whisper"}:
        raise ValueError("Manifest: неверные ссылки на исходные артефакты")
    if manifest["base_engine"] not in ("gigaam", "whisper"):
        raise ValueError("Manifest: неизвестная основа")
    if Path(manifest["stem"]).name != manifest["stem"] or manifest["stem"] in ("", ".", ".."):
        raise ValueError("Manifest: неверное имя результатов")
    for ref in manifest["refs"].values():
        if digest(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("Исходный артефакт изменился; подготовьте новый review")
    comparison_path = Path(manifest["refs"]["comparison"]["path"])
    comparison = read_json(comparison_path)
    validate_comparison_sources(comparison, comparison_path)
    if manifest["source_audio"] != comparison["source_audio"]:
        raise ValueError("Manifest: изменены метаданные источника")
    for engine in ("gigaam", "whisper"):
        raw_path = Path(manifest["refs"][engine]["path"])
        expected_path = resolve_ref(comparison["transcripts"][engine]["artifact_ref"], comparison_path.parent)
        expected_transcript = canonical_transcript(raw_path, comparison["source_audio"]["display_name"],
                                                   comparison["source_audio"]["duration_seconds"])
        if raw_path != expected_path or manifest["transcripts"][engine] != expected_transcript:
            raise ValueError("Manifest: текст/таймкоды не соответствуют raw")
    semantic_path = Path(manifest["refs"]["discrepancy_review"]["path"])
    semantic = read_json(semantic_path)
    if resolve_ref(semantic["comparison_file"], semantic_path.parent) != comparison_path:
        raise ValueError("Manifest: semantic review относится к другому comparison")
    with tempfile.TemporaryDirectory(prefix="meeting-quality-validate-") as temporary:
        selected_path, _ = dual.render_review(semantic_path, Path(temporary) / "review")
        if manifest["discrepancies"] != read_json(selected_path)["discrepancies"]:
            raise ValueError("Manifest: изменены результаты сверки ASR")
    # Индексы блоков должны покрывать обе полные расшифровки, без пропусков/дублей.
    for engine in ("gigaam", "whisper"):
        indices = [i for block in manifest["blocks"] for i in block["segment_indices"][engine]]
        if sorted(indices) != list(range(len(manifest["transcripts"][engine]["segments"]))):
            raise ValueError("Manifest: блоки не покрывают все сегменты")
    expected = {block["id"] for block in manifest["blocks"]}
    covered = review["reviewed_blocks"]
    if (not isinstance(covered, list) or not all(isinstance(x, str) for x in covered) or
            len(covered) != len(set(covered)) or set(covered) != expected):
        raise ValueError("Review должен покрывать все блоки ровно один раз")
    return manifest, review, manifest_path


def render(review_path: Path, human_resolutions_path: Optional[Path] = None,
           verified: bool = False, output_dir: Optional[Path] = None) -> dict[str, Path]:
    review_path = review_path.expanduser().resolve()
    manifest, review, manifest_path = load_manifest(review_path)
    base = manifest["transcripts"][manifest["base_engine"]]
    text, segments = base["text"], base["segments"]
    issues = {}
    for item in manifest["discrepancies"]:
        issues[item["id"]] = {**item, "span": interval_span(item["interval"], base), "kind": "asr_discrepancy"}
    if not isinstance(review["findings"], list) or not isinstance(review["edits"], list):
        raise ValueError("findings и edits должны быть массивами")
    for finding in review["findings"]:
        keys = {"id", "kind", "span", "severity", "categories", "review_reason"}
        fields(finding, keys, keys, "finding")
        identity = finding["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"Q\d{3,}", identity) or identity in issues:
            raise ValueError("Неверный или повторный Q ID")
        if finding["kind"] not in ("self_correction", "ambiguity", "other"):
            raise ValueError("Неизвестный kind finding")
        categories = finding["categories"]
        if (finding["severity"] not in dual.SEVERITIES or not isinstance(categories, list) or
                not categories or not all(isinstance(c, str) for c in categories) or
                len(set(categories)) != len(categories) or not set(categories) <= dual.CATEGORIES):
            raise ValueError("Неверная severity/categories finding")
        anchor = span(finding["span"], text)
        if anchor["start"] == anchor["end"]:
            raise ValueError("Finding должен указывать исходный текст")
        issues[identity] = {**finding, "span": anchor, "interval": source_interval(anchor, segments),
                            "review_reason": nonempty(finding["review_reason"], "review_reason")}
    patches = []
    edit_ids = set()
    for edit in review["edits"]:
        keys = {"id", "kind", "span", "to", "review_reason"}
        fields(edit, keys, keys, "edit")
        identity = edit["id"]
        if not isinstance(identity, str) or not re.fullmatch(r"E\d{3,}", identity) or identity in edit_ids:
            raise ValueError("Неверный или повторный E ID")
        edit_ids.add(identity)
        anchor = span(edit["span"], text)
        after = edit["to"]
        if not isinstance(after, str):
            raise ValueError("edit.to должен быть строкой")
        before_words, after_words = lexical(anchor["from"]), lexical(after)
        kind = edit["kind"]
        if kind == "vocalization":
            remove = lambda words: [w for w in words if not re.fullmatch(r"э{2,}|а{2,}|м{3,}", w)]
            valid = remove(before_words) == after_words
        else:
            valid = kind in ("punctuation", "spacing", "case") and before_words == after_words
        valid = valid and numeric_mentions(anchor["from"]) == numeric_mentions(after)
        valid = valid and protected_symbols(anchor["from"]) == protected_symbols(after)
        valid = valid and compound_notations(anchor["from"]) == compound_notations(after)
        # Регистр латинских обозначений может менять единицы/имена (mW/MW).
        valid = valid and re.findall(r"[A-Za-z]+", anchor["from"]) == re.findall(r"[A-Za-z]+", after)
        if not valid:
            raise ValueError("Редакторская правка меняет слова/факты; создайте finding и получите подтверждение")
        patches.append({**edit, "span": anchor, "review_reason": nonempty(edit["review_reason"], "review_reason"),
                        "source": "editorial", "interval": source_interval(anchor, segments)})
    resolved = set()
    if human_resolutions_path is not None:
        human_resolutions_path = human_resolutions_path.expanduser().resolve()
        human = read_json(human_resolutions_path)
        keys = {"schema_version", "manifest_file", "manifest_sha256", "resolutions"}
        fields(human, keys, keys, "human resolutions")
        if (resolve_ref(human["manifest_file"], human_resolutions_path.parent) != manifest_path or
                human["manifest_sha256"] != digest(manifest_path)):
            raise ValueError("Подтверждения относятся к другому manifest")
        if not isinstance(human["resolutions"], list):
            raise ValueError("resolutions должен быть массивом")
        for index, resolution in enumerate(human["resolutions"]):
            keys = {"candidate_id", "candidate_ids", "confirmed_by", "span", "confirmed_text"}
            fields(resolution, keys, {"confirmed_by", "span", "confirmed_text"}, "resolution")
            if ("candidate_id" in resolution) == ("candidate_ids" in resolution):
                raise ValueError("Укажите candidate_id либо candidate_ids")
            identities = resolution.get("candidate_ids", [resolution.get("candidate_id")])
            if (not isinstance(identities, list) or not identities or
                    not all(isinstance(i, str) for i in identities) or len(set(identities)) != len(identities) or
                    set(identities) - set(issues) or set(identities) & resolved):
                raise ValueError("Неизвестный или повторно подтверждённый candidate ID")
            if resolution["confirmed_by"] != "user" or not isinstance(resolution["confirmed_text"], str):
                raise ValueError("Требуется дословное подтверждение пользователя")
            anchor = span(resolution["span"], text)
            interval = source_interval(anchor, segments)
            for identity in identities:
                target = issues[identity]
                if target["kind"] != "asr_discrepancy":
                    matches = intersects(anchor, target["span"])
                elif anchor["start"] == anchor["end"]:
                    # Пропуск можно вставить в точную привязку D, включая пустую.
                    matches = target["span"]["start"] <= anchor["start"] <= target["span"]["end"]
                else:
                    matches = (intersects(anchor, target["span"]) and
                               interval["end_seconds"] > target["interval"]["start_seconds"] and
                               interval["start_seconds"] < target["interval"]["end_seconds"])
                if not matches:
                    raise ValueError("Подтверждение не привязано к своему кандидату")
            resolved.update(identities)
            patches.append({"id": f"H{index + 1:03d}", "kind": "human_resolution", "span": anchor,
                            "to": resolution["confirmed_text"], "source": "user", "candidate_ids": identities,
                            "review_reason": "Дословный ответ пользователя", "interval": interval})
    validate_patches(patches)
    pending = set(issues) - resolved
    if verified and pending:
        raise ValueError("Остались неподтверждённые кандидаты: " + ", ".join(sorted(pending)))
    masks = []
    for identity in sorted(pending, key=lambda i: (issues[i]["span"]["start"], issues[i]["span"]["end"], i)):
        anchor = issues[identity]["span"]
        if masks and intersects(masks[-1]["span"], anchor):
            masks[-1]["span"]["end"] = max(masks[-1]["span"]["end"], anchor["end"])
            masks[-1]["ids"].append(identity)
        else:
            masks.append({"span": dict(anchor), "ids": [identity]})
    applied, skipped = [], []
    for patch in patches:
        (skipped if any(intersects(patch["span"], m["span"]) for m in masks) else applied).append(patch)
    operations = [{"span": m["span"], "to": "[НЕ ПОДТВЕРЖДЕНО: " + ", ".join(sorted(m["ids"])) + "]"}
                  for m in masks] + applied
    validate_patches(operations)
    clean = text
    for operation in sorted(operations, key=lambda p: (p["span"]["start"], p["span"]["end"]), reverse=True):
        anchor = operation["span"]
        clean = clean[:anchor["start"]] + operation["to"] + clean[anchor["end"]:]
    result_issues = [{**item, "status": "resolved_by_user" if identity in resolved else "pending_human_review"}
                     for identity, item in issues.items()]
    result = {"schema_version": 1, "manifest_file": str(manifest_path), "manifest_sha256": digest(manifest_path),
              "status": "verified" if verified else "draft", "base_engine": manifest["base_engine"],
              "pending_ids": sorted(pending), "findings": result_issues}
    changes = {"schema_version": 1, "manifest_file": str(manifest_path), "manifest_sha256": digest(manifest_path),
               "review": reference(review_path), "human_resolutions": reference(human_resolutions_path) if human_resolutions_path else None,
               "applied": applied, "skipped_due_to_pending": skipped, "pending_ids": sorted(pending)}
    lines = ["# Проверка формулировок", "", "Спорные факты подтверждает пользователь по исходной записи.", ""]
    for item in result_issues:
        timing = item["interval"]
        lines.extend([f"## {item['id']} · {item['status']}", "",
                      f"Интервал: {dual.fmt_ts(timing['start_seconds'])}–{dual.fmt_ts(timing['end_seconds'], ceil=True)} (точность: сегмент)", "",
                      item["review_reason"], ""])
        if item["kind"] == "asr_discrepancy":
            lines.extend([f"GigaAM: {item['versions']['gigaam'] or '—'}", "",
                          f"Whisper: {item['versions']['whisper'] or '—'}", ""])
        else:
            lines.extend([f"Исходная формулировка: {item['span']['from']}", ""])
    if not result_issues:
        lines.extend(["После полного review спорных формулировок не отмечено.", ""])
    raw_links = " · ".join(f"[{key}](<{manifest['refs'][key]['path']}>)" for key in ("gigaam", "whisper"))
    header = ("# Редакторская версия\n\nНе источник дословных цитат. "
              f"Статус: {result['status']}; нерешённых: {len(pending)}.\n\n"
              f"Исходные транскрипты: {raw_links}\n\n")
    destination = (output_dir or manifest_path.parent).expanduser().resolve()
    stem = manifest["stem"]
    paths = {"findings_json": destination / f"{stem}.quality-findings.json",
             "findings_md": destination / f"{stem}.quality-findings.md",
             "clean_md": destination / f"{stem}.clean.md",
             "changes_json": destination / f"{stem}.clean.changes.json"}
    protected = {Path(ref["path"]).resolve() for ref in manifest["refs"].values()} | {review_path, manifest_path}
    if human_resolutions_path:
        protected.add(human_resolutions_path)
    publish({paths["findings_json"]: dump(result), paths["findings_md"]: "\n".join(lines),
             paths["clean_md"]: header + clean + "\n", paths["changes_json"]: dump(changes)}, protected)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Полный материал для review, включая согласные участки ASR")
    prep.add_argument("comparison", type=Path)
    prep.add_argument("--discrepancy-review", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path)
    prep.add_argument("--base-engine", choices=("gigaam", "whisper"), default="gigaam")
    output = sub.add_parser("render", help="Проверить review и создать clean-версию с журналом")
    output.add_argument("review", type=Path)
    output.add_argument("--human-resolutions", type=Path)
    output.add_argument("--verified", action="store_true")
    output.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            paths = prepare(args.comparison, args.discrepancy_review, args.output_dir, args.base_engine)
            print("\n".join(map(str, paths)))
        else:
            paths = render(args.review, args.human_resolutions, args.verified, args.output_dir)
            print("\n".join(map(str, paths.values())))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
