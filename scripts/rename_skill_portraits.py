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
# Some downloaded portrait names do not match students.json keys.
DEV_NAME_RENAMES = {"CH0258_01": "CH0258"}
PATH_NAME_ALIAS_KEYS = {"reijo": ["rezyo", "reizyo"]}


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


def resolve_dst_filename(filename, dev_name_to_name, path_name_to_name_ci):
    """Determine the destination filename for a downloaded PNG file.

    1. Exact DevName match -> {Name}.png
    2. Case-insensitive PathName exact match -> {Name}.png
    3. DevName prefix match ({DevName}_{suffix}) -> {Name}/{suffix}.png
    4. Case-insensitive PathName prefix match ({PathName}_{suffix}) -> {Name}/{suffix}.png
    5. Fallback -> strip Skill_Portrait_ prefix, keep English name
    """
    m = re.match(r"Skill_Portrait_(.+)\.png$", filename)
    if not m:
        return filename

    char_name = m.group(1)
    char_name_lower = char_name.lower()

    # Step 1: Exact match against DevName
    if char_name in dev_name_to_name:
        return f"{dev_name_to_name[char_name]}.png"

    # Step 2: Case-insensitive exact match against PathName
    if char_name_lower in path_name_to_name_ci:
        return f"{path_name_to_name_ci[char_name_lower]}.png"

    # Step 3: Find longest DevName that is a prefix of char_name.
    # Suffix can be either "_X" or "X"; if it starts with "_", drop only the first one.
    best_dev_name = None
    for dev_name in dev_name_to_name:
        if char_name.startswith(dev_name) and len(dev_name) < len(char_name):
            if best_dev_name is None or len(dev_name) > len(best_dev_name):
                best_dev_name = dev_name

    if best_dev_name:
        suffix = char_name[len(best_dev_name) :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        return f"{dev_name_to_name[best_dev_name]}/{suffix}.png"

    # Step 4: Find longest case-insensitive PathName prefix.
    # Suffix can be either "_X" or "X"; if it starts with "_", drop only the first one.
    best_path_name = None
    for path_name in path_name_to_name_ci:
        if char_name_lower.startswith(path_name) and len(path_name) < len(char_name_lower):
            if best_path_name is None or len(path_name) > len(best_path_name):
                best_path_name = path_name

    if best_path_name:
        suffix = char_name[len(best_path_name) :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        return f"{path_name_to_name_ci[best_path_name]}/{suffix}.png"

    # Step 5: No match, keep English name
    return f"{char_name}.png"


def build_name_maps(students):
    """Build name maps with compatibility aliases for known naming mismatches."""
    dev_name_to_name = {}
    path_name_to_name_ci = {}

    for student in students.values():
        if "Name" not in student:
            continue

        name = student["Name"]

        dev_name = student.get("DevName")
        if isinstance(dev_name, str):
            normalized_dev_name = DEV_NAME_RENAMES.get(dev_name, dev_name)
            dev_name_to_name[normalized_dev_name] = name

        path_name = student.get("PathName")
        if isinstance(path_name, str):
            normalized_path_name = path_name.lower()
            path_name_to_name_ci[normalized_path_name] = name
            for alias in PATH_NAME_ALIAS_KEYS.get(normalized_path_name, []):
                path_name_to_name_ci[alias.lower()] = name

    return dev_name_to_name, path_name_to_name_ci


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Fetching student data from SchaleDB...")
    students = fetch_json(SCHALEDB_URL)
    print(f"Found {len(students)} students")

    dev_name_to_name, path_name_to_name_ci = build_name_maps(students)

    all_files = collect_png_files()
    print(f"Found {len(all_files)} PNG files in downloads")

    copied = 0
    for filename, src_path in all_files:
        dst_filename = resolve_dst_filename(filename, dev_name_to_name, path_name_to_name_ci)
        dst_path = os.path.join(ASSETS_DIR, dst_filename)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        # Cache: skip if destination already exists
        if os.path.exists(dst_path):
            continue

        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename} -> {dst_filename}")
        copied += 1

    print(f"Done! Copied {copied} files.")


if __name__ == "__main__":
    main()
