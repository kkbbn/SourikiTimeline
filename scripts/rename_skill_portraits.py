#!/usr/bin/env python3
"""Copy and rename Skill Portrait images using student data from SchaleDB."""

import json
import os
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


def build_file_map():
    """Build a map of filename -> full path for all files in download directories."""
    file_map = {}
    for dir_name in os.listdir(DOWNLOAD_DIR):
        dir_path = os.path.join(DOWNLOAD_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for filename in os.listdir(dir_path):
            file_map[filename] = os.path.join(dir_path, filename)
    return file_map


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    print("Fetching student data from SchaleDB...")
    students = fetch_json(SCHALEDB_URL)
    print(f"Found {len(students)} students")

    file_map = build_file_map()

    copied = 0
    not_found = 0
    for student_id, student in students.items():
        dev_name = student["DevName"]
        name = student["Name"]

        src_filename = f"Skill_Portrait_{dev_name}.png"
        dst_path = os.path.join(ASSETS_DIR, f"{name}.png")

        # Cache: skip if destination already exists
        if os.path.exists(dst_path):
            continue

        if src_filename in file_map:
            shutil.copy2(file_map[src_filename], dst_path)
            print(f"Copied: {src_filename} -> {name}.png")
            copied += 1
        else:
            print(f"WARNING: Could not find {src_filename} for {name} (DevName: {dev_name})")
            not_found += 1

    print(f"Done! Copied {copied} files. {not_found} not found.")


if __name__ == "__main__":
    main()
