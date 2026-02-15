import json
import multiprocessing
import os
import inspect
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd

from scripts.common_utils import load_image, str_to_time

def get_default_workspace_path():
    path = os.path.join(os.path.expanduser("~"), "Documents", "SourikiTimeline")
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_timeline_columns():
    return [
        "発動時コスト",
        "残コスト",
        "キャラ名",
        "短縮キャラ名",
        "スキル名",
        "経過時間",
        "残り時間",
        "動画再生位置",
    ]

def adjust_mask_rect(mask_rect, mask_image_size, image_size, anchor=(0, 0)):
    x, y, w, h = mask_rect
    image_rate = image_size[0] / mask_image_size[0]
    x *= image_rate
    y *= image_rate
    w *= image_rate
    h *= image_rate
    #x += (image_size[0] - mask_image_size[0]) * anchor[0]
    y += (image_size[1] - mask_image_size[1] * image_rate) * anchor[1]
    return (int(x), int(y), int(w), int(h))

class JsonConfig:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def get_config_name(cls):
        return "config.json"

    @classmethod
    def get_config_path(cls, config_dir):
        return os.path.join(config_dir, cls.get_config_name())

    @classmethod
    def get_default_config_path(cls):
        return ""

    @classmethod
    def get_default_config(cls):
        config_path = cls.get_default_config_path()
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_json = json.load(f)
                config = cls.from_dict(config_json)
                return config
        return cls()

    @classmethod
    def from_dict(cls, dict):
        obj = cls(**{
            k: (pd.read_json(v, orient='split') if isinstance(v, str) and k.endswith('_dataframe') else v)
            for k, v in dict.items()
            if k in inspect.signature(cls).parameters
        })
        return obj

    @classmethod
    def get_parameters_size(cls):
        return len(inspect.signature(cls).parameters)

    @classmethod
    def load(cls, config_dir):
        config_path = cls.get_config_path(config_dir)
        if not os.path.exists(config_path):
            config_path = cls.get_default_config_path()
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_json = json.load(f)
                config = cls.from_dict(config_json)
                return config
        return cls()

    def save(self, config_dir):
        if not os.path.exists(config_dir):
            raise Exception("Project path not exists.")
        config_path = self.get_config_path(config_dir)
        with open(config_path, "w") as f:
            data_dict = asdict(self)
            for key, value in data_dict.items():
                if isinstance(value, pd.DataFrame):
                    data_dict[key] = value.to_json(orient="split", index=False, default_handler=str)
            json.dump(data_dict, f, indent=2)

    def update(self, *args, **kwargs):
        if args:
            for key, value in zip(self.__annotations__.keys(), args):
                setattr(self, key, value)

        # Update attributes with keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

cpu_count = multiprocessing.cpu_count()

@dataclass
class AppConfig(JsonConfig):
    project_path: str = ""
    workspace_path: str = field(default_factory=get_default_workspace_path)
    auto_save: bool = True
    download_format: str = "mp4"
    downloader: str = "yt-dlp"
    thumbnail_width: int = 640
    thumbnail_height: int = 480

    timeline_tsv_separator: str = ' '
    timeline_visible_columns: list[str] = field(default_factory=lambda: ["発動時コスト", "短縮キャラ名"])
    timeline_cost_omit_seconds: float = 3.0
    timeline_remain_cost_omit_value: float = 1.0
    timeline_newline_chara_names: list[str] = field(default_factory=lambda: [])
    timeline_newline_before_chara: bool = False
    timeline_newline_after_chara: bool = True

    _instance = None # Singleton instance

    def __post_init__(self):
        valid_columns = get_timeline_columns()
        self.timeline_visible_columns = [col for col in self.timeline_visible_columns if col in valid_columns]
        if not self.timeline_visible_columns:
            self.timeline_visible_columns = ["発動時コスト", "短縮キャラ名"]

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.load(".")
        return cls._instance

    def get_project_paths(self):
        if not os.path.isdir(self.workspace_path):
            return []

        files = os.listdir(self.workspace_path)
        project_paths = [os.path.join(self.workspace_path, f) for f in files if os.path.isdir(os.path.join(self.workspace_path, f))]
        return project_paths

    @classmethod
    def get_preimage(cls, project_path):
        preimage = os.path.join(project_path, "pre.jpg")
        if not os.path.exists(preimage):
            preimage = os.path.join("resources", "pre.jpg")
        return preimage

    def get_current_preimage(self):
        return AppConfig.get_preimage(self.project_path)

    @classmethod
    def get_movie(cls, project_path):
        movie_path = os.path.join(project_path, "movie.mp4")
        if not os.path.exists(movie_path):
            movie_path = None
        return movie_path

    def get_current_movie(self):
        return AppConfig.get_movie(self.project_path)

    def get_all_preimages(self):
        project_paths = self.get_project_paths()
        preimages = [AppConfig.get_preimage(f) for f in project_paths]
        return preimages

    def get_all_gallery(self):
        project_paths = self.get_project_paths()
        gallery = [(AppConfig.get_preimage(f), os.path.basename(f)) for f in project_paths]
        return gallery

app_config = AppConfig.instance()

@dataclass
class ProjectConfig(JsonConfig):
    title: str = ""
    author: str = ""

    movie_url: str = ""
    movie_download_file_name: str = "movie.mp4"
    movie_thumbnail_file_name: str = "pre.jpg"
    movie_start_time: float = 0.0
    movie_end_time: float = 100.0
    movie_x: int = 0
    movie_y: int = 0
    movie_width: int = 0
    movie_height: int = 0
    movie_frame_rate: float = 10.0
    movie_preview_time: float = 30.0

    mask_image_name: str = "mask_default.png"

    mask_image_w: int = 0
    mask_image_h: int = 0

    mask_skill_x: int = 0
    mask_skill_y: int = 0
    mask_skill_w: int = 0
    mask_skill_h: int = 0

    mask_cost_x: int = 0
    mask_cost_y: int = 0
    mask_cost_w: int = 0
    mask_cost_h: int = 0

    mask_time_x: int = 0
    mask_time_y: int = 0
    mask_time_w: int = 0
    mask_time_h: int = 0

    mask_cost_color1: str = '#00b4fa'
    mask_cost_color2: str = '#ffffff'
    mask_cost_color_threshold: int = 20

    mask_skill_color1: str = '#ffffff'
    mask_skill_color2: str = '#646464'
    mask_skill_color_threshold: int = 20
    mask_skill_color_fill_percentage: int = 30

    timeline_ignore_chara_names: list[str] = field(default_factory=lambda: [])
    timeline_max_time: int = 240

    @classmethod
    def get_config_name(cls):
        return "timeline_config.json"

    @classmethod
    def get_default_config_path(cls):
        return os.path.join("resources", cls.get_config_name())
   
    def get_fixed_download_file_name(self):
        base_name = os.path.splitext(self.movie_download_file_name)[0]
        return f"{base_name}.{app_config.download_format}"
    
    def get_mask_image_path(self):
        return os.path.join("resources", "mask", self.mask_image_name)

    def get_mask_image(self) -> np.ndarray:
        return load_image(self.get_mask_image_path())

    def get_movie_size(self):
        return (self.movie_width, self.movie_height)

    def get_mask_size(self):
        return (self.mask_image_w, self.mask_image_h)

    def get_skill_mask_rect(self):
        mask_rect = (
            int(self.mask_skill_x),
            int(self.mask_skill_y),
            int(self.mask_skill_w),
            int(self.mask_skill_h)
        )
        anchor = (0, 0)
        return adjust_mask_rect(mask_rect, self.get_mask_size(), self.get_movie_size(), anchor)

    def get_cost_mask_rect(self):
        mask_rect = (
            int(self.mask_cost_x),
            int(self.mask_cost_y),
            int(self.mask_cost_w),
            int(self.mask_cost_h)
        )
        anchor = (1, 1)
        return adjust_mask_rect(mask_rect, self.get_mask_size(), self.get_movie_size(), anchor)
    
    def get_time_mask_rect(self):
        mask_rect = (
            int(self.mask_time_x),
            int(self.mask_time_y),
            int(self.mask_time_w),
            int(self.mask_time_h)
        )
        anchor = (1, 0)
        return adjust_mask_rect(mask_rect, self.get_mask_size(), self.get_movie_size(), anchor)

    def get_mask_data(self):
        return [
            self.mask_image_w,
            self.mask_image_h,
            self.mask_skill_x,
            self.mask_skill_y,
            self.mask_skill_w,
            self.mask_skill_h,
            self.mask_cost_x,
            self.mask_cost_y,
            self.mask_cost_w,
            self.mask_cost_h,
            self.mask_time_x,
            self.mask_time_y,
            self.mask_time_w,
            self.mask_time_h,
        ]

    def convert_timeline(self, dataframe: pd.DataFrame):
        newline_chara_names = app_config.timeline_newline_chara_names
        newline_before_chara = app_config.timeline_newline_before_chara
        newline_after_chara = app_config.timeline_newline_after_chara
        columns = app_config.timeline_visible_columns
        cost_omit_seconds = app_config.timeline_cost_omit_seconds
        remain_cost_omit_value = app_config.timeline_remain_cost_omit_value

        if dataframe is None:
            return None

        new_rows = []
        prev_skill_time = 0
        all_columns = get_timeline_columns()

        columns = [column for column in columns if column in all_columns]

        for _, row in dataframe.iterrows():
            row = {key: row[key] if key in row else "" for key in all_columns}

            time_text = row["残り時間"]
            time = str_to_time(time_text)
            elapsed_time = prev_skill_time - time

            invoke_cost = row["発動時コスト"]
            remain_cost = row["残コスト"]

            if elapsed_time > 0 and cost_omit_seconds > 0 and elapsed_time < cost_omit_seconds + 0.01:
                invoke_cost = ""
                remain_cost = ""

            remain_cost_value = float(remain_cost) if remain_cost else 0
            if remain_cost_omit_value > 0 and remain_cost_value < remain_cost_omit_value + 0.01:
                remain_cost = ""

            new_row = []
            for column in columns:
                if column == "発動時コスト":
                    new_row.append(invoke_cost)
                elif column == "残コスト":
                    new_row.append(remain_cost)
                else:
                    new_row.append(row[column])

            if newline_before_chara and row["キャラ名"] in newline_chara_names:
                new_rows.append(["" for _ in columns])

            new_rows.append(new_row)

            if newline_after_chara and row["キャラ名"] in newline_chara_names:
                new_rows.append(["" for _ in columns])

            prev_skill_time = time

        new_dataframe = pd.DataFrame(new_rows, columns=columns, dtype=str)

        return new_dataframe

    def convert_timeline_and_tsv(self, dataframe: pd.DataFrame):
        if dataframe is None:
            return None, None

        new_dataframe = self.convert_timeline(dataframe)
        dataframe_tsv = new_dataframe.to_csv(index=False, header=True, sep=app_config.timeline_tsv_separator)
        return new_dataframe, dataframe_tsv
