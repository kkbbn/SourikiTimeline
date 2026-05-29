from dataclasses import dataclass
import os
from typing import Optional

import requests

from scripts.common_utils import convert_safe_filename
from scripts.config_utils import AppConfig, ProjectConfig
from scripts.media_utils import download_video, get_video_info, resize_image


class ProjectAlreadyExistsError(Exception):
    def __init__(self, project_path: str):
        super().__init__(f"Project already exists: {project_path}")
        self.project_path = project_path


@dataclass
class ProjectCreationResult:
    project_path: str
    config: ProjectConfig
    title: str
    author: str
    thumbnail_url: str


@dataclass
class ProjectVideoDownloadResult:
    project_path: str
    output_path: str
    config: ProjectConfig
    duration: float
    width: int
    height: int


def create_project_from_url(url: str, app_config: Optional[AppConfig] = None):
    app_config = app_config or AppConfig.instance()
    if url == "":
        raise ValueError("URLが入力されていません。")

    os.makedirs(app_config.workspace_path, exist_ok=True)

    title, author, thumbnail_url = get_video_info(url, app_config.downloader)
    project_name = convert_safe_filename(f"{author} - {title}")
    project_path = os.path.join(app_config.workspace_path, project_name)

    if os.path.exists(project_path):
        raise ProjectAlreadyExistsError(project_path)

    os.mkdir(project_path)

    config = ProjectConfig.load(project_path)
    config.movie_url = url
    config.title = title
    config.author = author
    config.save(project_path)

    thumbnail_path = os.path.join(project_path, config.movie_thumbnail_file_name)
    response = requests.get(thumbnail_url, timeout=30)
    with open(thumbnail_path, "wb") as file:
        file.write(response.content)

    resize_image(
        thumbnail_path,
        thumbnail_path,
        (app_config.thumbnail_width, app_config.thumbnail_height),
    )

    app_config.project_path = project_path
    app_config.save(".")

    return ProjectCreationResult(
        project_path=project_path,
        config=config,
        title=title,
        author=author,
        thumbnail_url=thumbnail_url,
    )


def download_project_video(
    project_path: str,
    config: Optional[ProjectConfig] = None,
    app_config: Optional[AppConfig] = None,
):
    app_config = app_config or AppConfig.instance()
    config = config or ProjectConfig.load(project_path)

    if config.movie_url == "":
        raise ValueError("URLを入力してください。")

    output_path = os.path.join(project_path, config.get_fixed_download_file_name())
    duration, width, height = download_video(
        config.movie_url,
        output_path,
        app_config.downloader,
    )

    config.movie_width = width
    config.movie_height = height
    config.movie_end_time = duration
    config.save(project_path)

    return ProjectVideoDownloadResult(
        project_path=project_path,
        output_path=output_path,
        config=config,
        duration=duration,
        width=width,
        height=height,
    )
