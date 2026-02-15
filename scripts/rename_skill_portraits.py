#!/usr/bin/env python3
"""Copy and rename Skill Portrait images using student data from SchaleDB."""

import json
import os
import re
import shutil
import urllib.request

SCHALEDB_URL = "https://schaledb.com/data/jp/students.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DOWNLOAD_DIR = os.path.join(PROJECT_DIR, "resources", "download")
ASSETS_DIR = os.path.join(PROJECT_DIR, "resources", "assets")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch_json(url):
    """Fetch a URL and return parsed JSON."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_png_files():
    """Collect all PNG files from download directories."""
    files = []
    for dir_name in os.listdir(DOWNLOAD_DIR):
        dir_path = os.path.join(DOWNLOAD_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for filename in os.listdir(dir_path):
            if filename.endswith(".png"):
                files.append((filename, os.path.join(dir_path, filename)))
    return files


def resolve_dst_filename(filename, dev_name_to_name, path_name_to_name):
    """Determine the destination filename for a downloaded PNG file.

    1. Exact DevName match -> {Name}.png
    2. DevName prefix match (suffix not starting with lowercase) -> {Name}{suffix}.png
    3. PathName match (case-insensitive) -> {Name}.png
    4. PathName prefix match -> {Name}{suffix}.png
    5. Fallback -> strip Skill_Portrait_ prefix, keep English name
    """
    m = re.match(r"Skill_Portrait_(.+)\.png$", filename)
    if not m:
        return filename

    char_name = m.group(1)

    # Step 1: Exact match against DevName
    if char_name in dev_name_to_name:
        return f"{dev_name_to_name[char_name]}.png"

    # Step 2: Find longest DevName that is a prefix of char_name
    # with the constraint that the suffix doesn't start with a lowercase letter
    best_dev_name = None
    for dev_name in dev_name_to_name:
        if char_name.startswith(dev_name) and len(dev_name) < len(char_name):
            # Suffix must not start with lowercase to avoid e.g. "Hina" matching "Hinata"
            suffix_start = char_name[len(dev_name)]
            if suffix_start.islower():
                continue
            if best_dev_name is None or len(dev_name) > len(best_dev_name):
                best_dev_name = dev_name

    if best_dev_name:
        suffix = char_name[len(best_dev_name):]
        return f"{dev_name_to_name[best_dev_name]}{suffix}.png"

    # Step 3: Exact match against PathName (case-insensitive)
    char_lower = char_name.lower()
    if char_lower in path_name_to_name:
        return f"{path_name_to_name[char_lower]}.png"

    # Step 4: Find longest PathName that is a prefix of char_name
    best_path_name = None
    for path_name in path_name_to_name:
        if char_lower.startswith(path_name) and len(path_name) < len(char_lower):
            remaining = char_lower[len(path_name):]
            # Suffix should start with underscore or uppercase in original
            if remaining[0] not in ('_',):
                orig_suffix_start = char_name[len(path_name)]
                if orig_suffix_start.islower():
                    continue
            if best_path_name is None or len(path_name) > len(best_path_name):
                best_path_name = path_name

    if best_path_name:
        suffix = char_name[len(best_path_name):]
        return f"{path_name_to_name[best_path_name]}{suffix}.png"

    # Step 5: No match, keep English name
    return f"{char_name}.png"


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Fetching student data from SchaleDB...")
    students = fetch_json(SCHALEDB_URL)
    print(f"Found {len(students)} students")

    dev_name_to_name = {s["DevName"]: s["Name"] for s in students.values()}
    path_name_to_name = {s["PathName"]: s["Name"] for s in students.values() if s.get("PathName")}

    all_files = collect_png_files()
    print(f"Found {len(all_files)} PNG files in downloads")

    copied = 0
    for filename, src_path in all_files:
        dst_filename = resolve_dst_filename(filename, dev_name_to_name, path_name_to_name)
        dst_path = os.path.join(ASSETS_DIR, dst_filename)

        # Cache: skip if destination already exists
        if os.path.exists(dst_path):
            continue

        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename} -> {dst_filename}")
        copied += 1

    print(f"Done! Copied {copied} files.")


if __name__ == "__main__":
    main()
