#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_STUDENTS_JSON = Path("~/Downloads/students.json").expanduser()


def normalize_field(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\r", "").replace("\n", "\\n")


def resolve_tsv_path(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)

    plural = Path("resources/skills.tsv")
    singular = Path("resources/skill.tsv")

    if plural.exists():
        return plural
    return singular


def load_existing_lines(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()

    with tsv_path.open("r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f}


def load_existing_names(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()

    names: set[str] = set()
    with tsv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if "\t" not in line:
                continue
            name, _ = line.split("\t", 1)
            if name == "キャラ":
                continue
            names.add(name)
    return names


def append_lines(tsv_path: Path, lines: list[str]) -> None:
    if not lines:
        return

    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    needs_newline = False
    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        with tsv_path.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"

    with tsv_path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        for line in lines:
            f.write(line)
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append missing student EX skill rows to a TSV file."
    )
    parser.add_argument(
        "--students-json",
        default=str(DEFAULT_STUDENTS_JSON),
        help=f"Path to students.json (default: {DEFAULT_STUDENTS_JSON})",
    )
    parser.add_argument(
        "--tsv",
        default=None,
        help="Path to output TSV. Defaults to resources/skills.tsv if present, else resources/skill.tsv.",
    )
    args = parser.parse_args()

    students_json_path = Path(args.students_json).expanduser()
    tsv_path = resolve_tsv_path(args.tsv)

    with students_json_path.open("r", encoding="utf-8") as f:
        students = json.load(f)

    existing_lines = load_existing_lines(tsv_path)
    existing_names = load_existing_names(tsv_path)
    new_lines: list[str] = []

    for student in students.values():
        name = normalize_field(student.get("Name"))
        ex = student.get("Skills", {}).get("Ex", {})
        ex_name = normalize_field(ex.get("Name"))
        ex_desc = normalize_field(ex.get("Desc"))

        # Keep one row per character name. If a row starts with "{Name}\t",
        # treat it as already present regardless of skill fields.
        if name in existing_names:
            continue

        line = f"{name}\t{ex_name}\t{ex_desc}"
        if line in existing_lines:
            continue

        existing_lines.add(line)
        existing_names.add(name)
        new_lines.append(line)

    append_lines(tsv_path, new_lines)
    print(f"TSV path: {tsv_path}")
    print(f"Added {len(new_lines)} rows")


if __name__ == "__main__":
    main()
