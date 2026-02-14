#!/usr/bin/env python3
"""Download Skill Portrait images from spriters-resource.com for Blue Archive characters."""

import io
import os
import re
import sys
import time
import urllib.request
import zipfile

from bs4 import BeautifulSoup

BASE_URL = "https://www.spriters-resource.com"
INDEX_URL = f"{BASE_URL}/mobile/bluearchive/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DOWNLOAD_DIR = os.path.join(PROJECT_DIR, "resources", "download")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
SLEEP_INTERVAL = 0.1


def fetch(url):
    """Fetch a URL and return the response bytes."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as response:
        return response.read()


def fetch_text(url):
    """Fetch a URL and return the response as text."""
    return fetch(url).decode("utf-8")


def get_characters():
    """Parse the index page and return list of (name, download_url) for the Characters section."""
    html = fetch_text(INDEX_URL)
    soup = BeautifulSoup(html, "html.parser")

    # Find all section headers
    sections = soup.find_all("div", class_="section")
    characters_display = None
    for section in sections:
        text = section.get_text().strip()
        # Match "Characters" but not "Assembled Characters"
        if re.search(r"\]\s*Characters\s*$", text):
            characters_display = section.find_next_sibling("div", class_="icondisplay")
            break

    if not characters_display:
        print("ERROR: Could not find Characters section")
        sys.exit(1)

    characters = []
    for link in characters_display.find_all("a", class_="iconlink"):
        header = link.find("div", class_="iconheader")
        name = header["title"]
        # Extract game_id and asset_id from the icon image URL
        # e.g. /media/asset_icons/492/513218.png?updated=1769470953
        img = link.find("img")
        if img and img.get("src"):
            m = re.search(r"/asset_icons/(\d+)/(\d+)\.png", img["src"])
            if m:
                game_id, asset_id = m.group(1), m.group(2)
                download_url = f"{BASE_URL}/media/assets/{game_id}/{asset_id}.zip"
                characters.append((name, download_url))

    return characters


def download_and_extract(name, download_url):
    """Download zip and extract Skill_Portrait_*.png files to resources/download/{name}/."""
    target_dir = os.path.join(DOWNLOAD_DIR, name)
    zip_data = fetch(download_url)

    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        skill_files = [
            f for f in zf.namelist()
            if re.match(r"Skill_Portrait_.+\.png", os.path.basename(f))
        ]

        os.makedirs(target_dir, exist_ok=True)

        if not skill_files:
            print(f"  WARNING: No Skill_Portrait_*.png found for {name}")
            return

        for f in skill_files:
            filename = os.path.basename(f)
            with zf.open(f) as src:
                with open(os.path.join(target_dir, filename), "wb") as dst:
                    dst.write(src.read())
            print(f"  Extracted: {filename}")


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print("Fetching character list...")
    characters = get_characters()
    print(f"Found {len(characters)} characters")

    downloaded = 0
    for i, (name, download_url) in enumerate(characters):
        target_dir = os.path.join(DOWNLOAD_DIR, name)
        if os.path.isdir(target_dir):
            continue

        print(f"[{i + 1}/{len(characters)}] Downloading {name}...")
        try:
            download_and_extract(name, download_url)
            downloaded += 1
        except Exception as e:
            print(f"  ERROR: {e}")

        time.sleep(SLEEP_INTERVAL)

    print(f"Done! Downloaded {downloaded} new characters.")


if __name__ == "__main__":
    main()
