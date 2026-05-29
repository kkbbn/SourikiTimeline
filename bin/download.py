#!/usr/bin/env python3
import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENV_PYTHON = os.path.join(REPO_ROOT, "venv", "bin", "python3")
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON, __file__, *sys.argv[1:]])

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a SourikiTimeline project and download its video.",
    )
    parser.add_argument("url", help="YouTube URL to create a project from.")
    parser.add_argument(
        "--workspace",
        help="Workspace directory. Defaults to the workspace path in config.json.",
    )
    parser.add_argument(
        "--downloader",
        choices=["yt-dlp", "pytube"],
        help="Downloader to use. Defaults to the value in config.json.",
    )
    parser.add_argument(
        "--format",
        dest="download_format",
        choices=["mp4", "webm"],
        help="Video format to download.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from scripts.config_utils import AppConfig, ProjectConfig
    from scripts.project_utils import (
        ProjectAlreadyExistsError,
        ProjectCreationResult,
        create_project_from_url,
        download_project_video,
    )

    app_config = AppConfig.instance()

    if args.workspace:
        app_config.workspace_path = os.path.abspath(os.path.expanduser(args.workspace))
    if args.downloader:
        app_config.downloader = args.downloader
    if args.download_format:
        app_config.download_format = args.download_format

    try:
        created = create_project_from_url(args.url, app_config)
    except ProjectAlreadyExistsError as e:
        config = ProjectConfig.load(e.project_path)
        created = ProjectCreationResult(
            project_path=e.project_path,
            config=config,
            title=config.title,
            author=config.author,
            thumbnail_url="",
        )
        print("Project already exists; using existing project")
        print(f"  path: {created.project_path}")
    except Exception as e:
        print(f"Failed to create project: {e}", file=sys.stderr)
        return 1
    else:
        print("Project created")
        print(f"  path: {created.project_path}")
        print(f"  title: {created.title}")
        print(f"  author: {created.author}")

    try:
        downloaded = download_project_video(
            created.project_path,
            created.config,
            app_config,
        )
    except Exception as e:
        print(f"Failed to download video: {e}", file=sys.stderr)
        return 1

    print("Video downloaded")
    print(f"  path: {downloaded.output_path}")
    print(f"  duration: {downloaded.duration}")
    print(f"  size: {downloaded.width} x {downloaded.height}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
