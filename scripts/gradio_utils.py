from dataclasses import asdict
import datetime
import os
import gradio as gr
import numpy as np
import pandas as pd
import pyperclip
import requests
import ctypes
from moviepy import VideoFileClip

from scripts.chara_skill import CharaSkill
from scripts.common_utils import convert_safe_filename, load_memo, load_timeline, save_image, save_memo, save_timeline, str_to_time, time_to_str
from scripts.config_utils import AppConfig, ProjectConfig, get_timeline_columns
from scripts.debug_timer import DebugTimer
from scripts.debug_utils import debug_args
from scripts.media_utils import download_video, extract_video_frame, get_video_info, resize_image
from scripts.ocr_utils import crop_image, draw_image_line, draw_image_rect, draw_image_string, get_color_fill_percentage, get_image_bar_percentage, get_mask_image_rect, ocr_image
from scripts.platform_utils import get_folder_path

app_config = AppConfig.instance()

@debug_args
def auto_save(config: ProjectConfig, project_path: str):
    if not app_config.auto_save:
        return ""

    config.save(project_path)

    timestr = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    config_path = ProjectConfig.get_config_path(project_path)
    output_log = f"[{timestr}] 設定のセーブをしました。\n{config_path}"
    return output_log

@debug_args
def select_workspace_gr():
    workspace_path = get_folder_path(app_config.workspace_path)
    if workspace_path == '':
        return ["ワークスペースを選択してください", app_config.workspace_path]

    app_config.workspace_path = workspace_path
    app_config.save(".")

    output_log = f"ワークスペースを開きました。{workspace_path}\n\n"

    return [output_log, workspace_path]

@debug_args
def select_project_gr(evt: gr.SelectData):
    project_paths = app_config.get_project_paths()

    if evt.index >= len(project_paths):
        raise Exception(f"選択しているプロジェクトが見つかりません。 {evt.indexs}")

    project_path = project_paths[evt.index]
    config = ProjectConfig.load(project_path)

    app_config.project_path = project_path
    app_config.save(".")

    source_dataframe = load_timeline(project_path)
    dataframe, dataframe_tsv = config.convert_timeline_and_tsv(source_dataframe)

    memo = load_memo(project_path)

    output_log = f"プロジェクトをロードしました。\n\n{project_path}\n\n"

    return [
        output_log,
        project_path,
        app_config.get_current_preimage(),
        dataframe,
        dataframe_tsv,
        memo,
        *asdict(config).values(),
        *([None] * 13),
    ]

@debug_args
def create_project_gr(url: str):
    if not os.path.exists(app_config.workspace_path):
        os.mkdir(app_config.workspace_path)

    if url == "":
        config = ProjectConfig.load(app_config.project_path)
        return ["URLが入力されていません。", "", None, *asdict(config).values()]

    title, author, thumbnail_url = get_video_info(url, app_config.downloader)

    new_score_name = convert_safe_filename(f"{author} - {title}")

    project_path = os.path.join(app_config.workspace_path, new_score_name)
    if os.path.exists(project_path):
        config = ProjectConfig.load(app_config.project_path)
        return ["すでに同名ディレクトリが存在しています。", project_path, None, *asdict(config).values()]

    os.mkdir(project_path)

    config = ProjectConfig.load(project_path)
    config.movie_url = url
    config.title = title
    config.author = author
    config.save(project_path)

    # サムネDL
    thumbnail_file_name = config.movie_thumbnail_file_name
    thumbnail_path = os.path.join(project_path, thumbnail_file_name)

    response = requests.get(thumbnail_url)
    with open(thumbnail_path, "wb") as file:
        file.write(response.content)

    thumbnail_width = app_config.thumbnail_width
    thumbnail_height = app_config.thumbnail_height

    resize_image(thumbnail_path, thumbnail_path, (thumbnail_width, thumbnail_height))

    app_config.project_path = project_path
    app_config.save(".")

    output_log = "プロジェクト作成に成功しました。\n\n"
    output_log += '"ダウンロード"タブに進んでください。\n\n'

    return [
        output_log,
        project_path,
        app_config.get_current_preimage(),
        None,
        None,
        None,
        *asdict(config).values(),
        *([None] * 13),
    ]

@debug_args
def reload_workspace_gr():
    gallery = app_config.get_all_gallery()
    if len(gallery) == 0:
        return None

    return gallery

@debug_args
def _download_video_gr(config: ProjectConfig, project_path):
    url = config.movie_url
    output_file_name = config.get_fixed_download_file_name()
    output_path = os.path.join(project_path, output_file_name)
    downloader = app_config.downloader

    if url == "":
        raise Exception("URLを入力してください。")

    duration, width, height = download_video(
        url,
        output_path,
        downloader)

    config.movie_width = width
    config.movie_height = height
    config.movie_end_time = duration
    config.save(project_path)

    output_log = "動画のダウンロードに成功しました。\n\n"
    output_log += '"マスク調整"タブに進んでください。\n\n'

    output_log += f"- 再生時間: {duration}\n"
    output_log += f"- 解像度: {width} x {height}\n\n"

    auto_save(config, project_path)

    return [config, output_log, output_path, duration, width, height]

@debug_args
def parse_args(*args, project_path=None):
    app_config.update(*(args[:AppConfig.get_parameters_size()]))
    app_config.save(".")

    project_path = project_path or app_config.project_path
    config = ProjectConfig(*(args[AppConfig.get_parameters_size():]))

    return config, project_path

@debug_args
def reload_video_gr(*args, project_path=None):
    project_path = project_path or app_config.project_path
    config = ProjectConfig(*args)

    input_file_name = config.get_fixed_download_file_name()
    output_file_name = config.movie_output_file_name

    input_path = os.path.join(project_path, input_file_name)
    output_path = os.path.join(project_path, output_file_name)

    output_path = output_path if os.path.exists(output_path) else None

    output_log = "表示を更新しました。\n\n"

    auto_save(config, project_path)

    return [output_log, input_path, output_path]

@debug_args
def download_video_gr(*args, project_path=None):
    config, project_path = parse_args(*args, project_path=project_path)

    config, output_log, input_path, duration, width, height = _download_video_gr(config, project_path)

    config, _, mask_result = _load_mask_gr(config, project_path)

    _, input_path, preview_image, preview_slider = _mask_preview_gr(config, project_path)

    return [output_log, input_path, preview_image, duration, width, height, preview_slider, *mask_result]

@debug_args
def _mask_preview_gr(config: ProjectConfig, project_path):
    input_file_name = config.get_fixed_download_file_name()
    input_path = os.path.join(project_path, input_file_name)
    preview_time = config.movie_preview_time

    preview_image, skill_name, time_text, cost = create_test_image(config, project_path, preview_time)

    preview_slider = gr.update(minimum=config.movie_start_time, maximum=config.movie_end_time)

    if skill_name != "" and time_text != "" and cost > 0:
        output_log = "情報の取得に成功しました。\n\n"
        output_log += '"タイムライン生成"タブに進んでください。\n\n'
    else:
        output_log = "情報の取得に失敗しました。\n\n"
        output_log += '設定値を調整してください。\n\n'

    output_log += f"- スキル: {skill_name}\n"
    output_log += f"- 残り時間: {time_text}\n"
    output_log += f"- コスト: {cost}\n\n"

    auto_save(config, project_path)

    return [output_log, input_path, preview_image, preview_slider]

@debug_args
def mask_preview_gr(*args, project_path=None):
    config, project_path = parse_args(*args, project_path=project_path)

    output_log, input_path, preview_image, preview_slider = _mask_preview_gr(config, project_path)

    return [output_log, input_path, preview_image, preview_slider]

@debug_args
def get_scaled_coordinates(left, top, right, bottom):
    # ユーザーのディスプレイ設定からDPIスケーリングを取得
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()  # DPI仮想化を無効化（プログラムが実際のDPIを認識できるようにする）
    # スクリーンのDPIを取得
    dpi = user32.GetDpiForSystem()
    scaling_factor = dpi / 96  # デフォルトDPIは96

    # DPIスケーリングを考慮して座標を調整
    scaled_left = int(left * scaling_factor)
    scaled_top = int(top * scaling_factor)
    scaled_right = int(right * scaling_factor)
    scaled_bottom = int(bottom * scaling_factor)

    return scaled_left, scaled_top, scaled_right, scaled_bottom

def format_time_string(time_str: str):
    time_str = ''.join(filter(str.isdigit, time_str))
    if len(time_str) == 4:
        return f"{time_str[:2]}:{time_str[2:]}"
    elif len(time_str) == 7:
        return f"{time_str[:2]}:{time_str[2:4]}.{time_str[4:]}"
    else:
        return ""

@debug_args
def load_preview_image(config: ProjectConfig, project_path, target_time):
    input_name = config.get_fixed_download_file_name()
    input_path = os.path.join(project_path, input_name)
    movie_x = config.movie_x
    movie_y = config.movie_y
    movie_width = config.movie_width
    movie_height = config.movie_height

    image = extract_video_frame(input_path, target_time)
    image = crop_image(image, (movie_x, movie_y, movie_width, movie_height))
    return image

@debug_args
def create_test_image(config: ProjectConfig, project_path, target_time=0):
    cost_color1 = config.mask_cost_color1
    cost_color2 = config.mask_cost_color2
    cost_color_threshold = config.mask_cost_color_threshold
    skill_color1 = config.mask_skill_color1
    skill_color2 = config.mask_skill_color2
    skill_color_threshold = config.mask_skill_color_threshold
    ignore_chara_names = config.timeline_ignore_chara_names
    skill_mask_rect = config.get_skill_mask_rect()
    cost_mask_rect = config.get_cost_mask_rect()
    time_mask_rect = config.get_time_mask_rect()

    timer = DebugTimer()
    timer.start()

    input_image = load_preview_image(config, project_path, target_time)
    timer.print("load_image")

    chara_skills = CharaSkill.from_tsv()
    chara_skills = [cs for cs in chara_skills if cs.chara_name not in ignore_chara_names]
    timer.print("CharaSkill.from_tsv")

    fill_percentage = max(get_color_fill_percentage(input_image, skill_mask_rect, skill_color1, skill_color_threshold),
                            get_color_fill_percentage(input_image, skill_mask_rect, skill_color2, skill_color_threshold))
    timer.print("get_color_fill_percentage")

    skill_texts = ocr_image(input_image, skill_mask_rect)
    skill_text = "".join(skill_texts)

    time_texts = ocr_image(input_image, time_mask_rect, 'en')
    time_text = format_time_string("".join(time_texts))
    timer.print("ocr_image")

    chara_skill, similarity = CharaSkill.find_best_match(chara_skills, skill_text)
    timer.print("CharaSkill.find_best_match")

    cost = get_image_bar_percentage(
                input_image,
                cost_mask_rect,
                cost_color1,
                cost_color2,
                cost_color_threshold) / 10

    output_image = input_image.copy()

    draw_image_rect(output_image, skill_mask_rect, '#ff0000')
    draw_image_rect(output_image, cost_mask_rect, '#0000ff')
    draw_image_rect(output_image, time_mask_rect, '#00ff00')

    # costの位置に線を引く
    x, y, w, h = cost_mask_rect
    x += int(w * cost / 10)
    draw_image_line(output_image, (x, y), (x, y+h), '#ffffff')

    skill_info = f"{skill_text}: {chara_skill.chara_name} ({similarity:.2f}%)" if chara_skill is not None else "None"
    skill_info += f"\nスキル判定色の割合: {fill_percentage:.0f}%"

    def rect_to_position(rect, offset):
        x, y, w, h = rect
        ox, oy = offset
        return x + ox, y + oy

    image_scale = output_image.shape[1] / 1920
    font_size = 40 * image_scale
    stroke_width = 3 * image_scale
    output_image = draw_image_string(output_image, skill_info, rect_to_position(skill_mask_rect, (0, 50 * image_scale)), '#ff0000', font_size, stroke_width)
    output_image = draw_image_string(output_image, f"コスト: {cost}", rect_to_position(cost_mask_rect, (0, -50 * image_scale)), '#0000ff', font_size, stroke_width)
    output_image = draw_image_string(output_image, time_text, rect_to_position(time_mask_rect, (0, 50 * image_scale)), '#00ff00', font_size, stroke_width)

    skill_name = chara_skill.skill_name if chara_skill is not None else ""

    return [output_image, skill_name, time_text, cost]

@debug_args
def _timeline_generate_gr(config: ProjectConfig, project_path: str):
    input_name = config.get_fixed_download_file_name()
    input_path = os.path.join(project_path, input_name)
    start_time = config.movie_start_time
    end_time = config.movie_end_time
    cost_color1 = config.mask_cost_color1
    cost_color2 = config.mask_cost_color2
    cost_color_threshold = config.mask_cost_color_threshold
    skill_color1 = config.mask_skill_color1
    skill_color2 = config.mask_skill_color2
    skill_color_threshold = config.mask_skill_color_threshold
    skill_color_fill_percentage = config.mask_skill_color_fill_percentage
    ignore_chara_names = config.timeline_ignore_chara_names
    max_time = config.timeline_max_time
    movie_x = config.movie_x
    movie_y = config.movie_y
    movie_width = config.movie_width
    movie_height = config.movie_height
    frame_rate = config.movie_frame_rate
    skill_mask_rect = config.get_skill_mask_rect()
    cost_mask_rect = config.get_cost_mask_rect()
    time_mask_rect = config.get_time_mask_rect()

    chara_skills = CharaSkill.from_tsv()
    chara_skills = [cs for cs in chara_skills if cs.chara_name not in ignore_chara_names]

    rows = []
    columns = get_timeline_columns()

    class FrameData:
        chara_skill: CharaSkill = None
        time_image: np.ndarray = None
        movie_time: int = 0
        cost: float = 0

        remain_time: int = 0

        def __init__(self, chara_skill, time_image, movie_time, cost):
            self.chara_skill = chara_skill
            self.time_image = time_image
            self.movie_time = movie_time
            self.cost = cost

        def calculate(self):
            if self.time_image is None:
                return

            time_texts = ocr_image(self.time_image, None, 'en')
            remain_time_text = format_time_string("".join(time_texts))
            self.remain_time = str_to_time(remain_time_text)

            if self.remain_time == 0:
                self.cost = 0

            self.time_image = None

    prev_frame_chara_skill = None
    last_max_cost = 0
    is_executing_skill = False
    frame_datas: list[FrameData] = []
    used_skill_frame_ids = []

    with VideoFileClip(input_path) as clip:
        end_time = clip.duration if end_time == 0 else end_time
        end_time_text = time_to_str(end_time)
        with clip.subclipped(start_time, end_time).with_fps(frame_rate) as subclip:
            for frame_id, frame in enumerate(subclip.iter_frames()):
                movie_time = start_time + (frame_id / frame_rate)
                movie_time_text = time_to_str(movie_time)
                print(f"progress movie... {movie_time_text} / {end_time_text}")

                input_image = crop_image(frame, (movie_x, movie_y, movie_width, movie_height))

                fill_percentage = max(get_color_fill_percentage(input_image, skill_mask_rect, skill_color1, skill_color_threshold),
                                          get_color_fill_percentage(input_image, skill_mask_rect, skill_color2, skill_color_threshold))

                chara_skill = None

                if fill_percentage >= skill_color_fill_percentage:
                    if not is_executing_skill:
                        skill_texts = ocr_image(input_image, skill_mask_rect)
                        skill_text = "".join(skill_texts)

                        chara_skill, similarity = CharaSkill.find_best_match(chara_skills, skill_text)
                else:
                    is_executing_skill = False

                time_image = crop_image(input_image, time_mask_rect)

                cost = get_image_bar_percentage(
                        input_image,
                        cost_mask_rect,
                        cost_color1,
                        cost_color2,
                        cost_color_threshold) / 10
                
                frame_data = FrameData(
                    chara_skill,
                    time_image,
                    movie_time,
                    cost
                )

                frame_datas.append(frame_data)

                if chara_skill is not None and chara_skill == prev_frame_chara_skill:
                    used_skill_frame_ids.append(frame_id)

                    is_executing_skill = True

                prev_frame_chara_skill = chara_skill

    for skill_index, frame_id in enumerate(used_skill_frame_ids):
        frame_data = frame_datas[frame_id]
        frame_data.calculate()

        movie_time_text = time_to_str(frame_data.movie_time)
        print(f"progress frame... {movie_time_text} / {end_time_text}")

        prev_skill_frame_id = max(frame_id - 10, 0)
        if skill_index > 0:
            prev_skill_frame_id = max(used_skill_frame_ids[skill_index - 1], prev_skill_frame_id)

        last_max_cost = 0
        last_min_cost = 10
        for i in range(prev_skill_frame_id, frame_id):
            frame_datas[i].calculate()
            cost = frame_datas[i].cost
            if cost > 0:
                last_max_cost = max(cost, last_max_cost)
                last_min_cost = min(cost, last_min_cost)

        row = []
        row.append(f"{last_max_cost:.1f}") # 発動時コスト
        row.append(f"{last_min_cost:.1f}") # 残コスト
        row.append(frame_data.chara_skill.chara_name) # キャラ名
        row.append(frame_data.chara_skill.get_short_chara_name()) # 短縮キャラ名
        row.append(frame_data.chara_skill.skill_name) # スキル名
        row.append(time_to_str(max_time - frame_data.remain_time)) # 残り時間
        row.append(time_to_str(frame_data.remain_time)) # 経過時間
        row.append(time_to_str(frame_data.movie_time)) # 動画再生位置

        rows.append(row)

    source_dataframe = pd.DataFrame(data=rows, columns=columns, dtype=str)
    save_timeline(project_path, source_dataframe)

    dataframe, dataframe_tsv = config.convert_timeline_and_tsv(source_dataframe)
    pyperclip.copy(dataframe_tsv)

    auto_save(config, project_path)

    output_log = "タイムラインを生成しました。\n\n"
    output_log += "クリップボードにコピーしました。\n\n"

    return [output_log, dataframe, dataframe_tsv]

@debug_args
def _load_mask_gr(config: ProjectConfig, project_path):
    mask_image = config.get_mask_image()

    skill_mask_rect = get_mask_image_rect(mask_image, '#ff0000')
    cost_mask_rect = get_mask_image_rect(mask_image, '#0000ff')
    time_mask_rect = get_mask_image_rect(mask_image, '#00ff00')

    config.mask_image_w = mask_image.shape[1]
    config.mask_image_h = mask_image.shape[0]
    config.mask_skill_x, config.mask_skill_y, config.mask_skill_w, config.mask_skill_h = skill_mask_rect
    config.mask_cost_x, config.mask_cost_y, config.mask_cost_w, config.mask_cost_h = cost_mask_rect
    config.mask_time_x, config.mask_time_y, config.mask_time_w, config.mask_time_h = time_mask_rect

    auto_save(config, project_path)

    output_log = "マスクを更新しました。\n\n"

    return [
        config,
        output_log,
        config.get_mask_data(),
    ]

@debug_args
def _save_mask_gr(config: ProjectConfig, project_path, new_mask_image_name):
    if new_mask_image_name == "":
        return [config, "ファイル名を入力してください。", "", config.get_mask_data()]

    new_mask_image_name = os.path.splitext(new_mask_image_name)[0] + ".png"

    if new_mask_image_name == "mask_default.png":
        return [config, "mask_default.pngは上書きできません。", "", config.get_mask_data()]

    config.mask_image_name = new_mask_image_name

    preview_time = config.movie_preview_time
    mask_image_path = config.get_mask_image_path()
    skill_mask_rect = config.get_skill_mask_rect()
    cost_mask_rect = config.get_cost_mask_rect()
    time_mask_rect = config.get_time_mask_rect()

    image = load_preview_image(config, project_path, preview_time)

    draw_image_rect(image, skill_mask_rect, '#ff0000', -1)
    draw_image_rect(image, cost_mask_rect, '#0000ff', -1)
    draw_image_rect(image, time_mask_rect, '#00ff00', -1)

    save_image(image, mask_image_path)

    config, _, mask_result = _load_mask_gr(config, project_path)

    output_log = "マスクを保存しました。\n\n"
    output_log += f"{mask_image_path}\n\n"

    return [config, output_log, new_mask_image_name, mask_result]

@debug_args
def timeline_generate_gr(*args, project_path=None):
    config, project_path = parse_args(*args, project_path=project_path)

    output_log, dataframe, dataframe_tsv = _timeline_generate_gr(config, project_path)

    return [output_log, dataframe, dataframe_tsv]

@debug_args
def timeline_update_gr(*args, project_path=None):
    config, project_path = parse_args(*args, project_path=project_path)

    source_dataframe = load_timeline(app_config.project_path)
    dataframe, dataframe_tsv = config.convert_timeline_and_tsv(source_dataframe)
    pyperclip.copy(dataframe_tsv)

    auto_save(config, project_path)

    output_log = "タイムラインを更新しました。\n\n"
    output_log += "クリップボードにTSVをコピーしました。\n\n"

    return [output_log, dataframe, dataframe_tsv]

@debug_args
def load_mask_gr(*args, project_path=None):
    config, project_path = parse_args(*args, project_path=project_path)

    config, output_log, mask_result = _load_mask_gr(config, project_path)

    _, input_path, preview_image, preview_slider = _mask_preview_gr(config, project_path)

    return [output_log, input_path, preview_image, preview_slider, *mask_result]

@debug_args
def save_mask_gr(new_mask_image_name):
    project_path = app_config.project_path
    config = ProjectConfig.load(project_path)

    config, output_log, mask_image_name, mask_result = _save_mask_gr(config, project_path, new_mask_image_name)

    return [output_log, mask_image_name, *mask_result]

@debug_args
def save_memo_gr(memo: str):
    project_path = app_config.project_path
    save_memo(project_path, memo)

    output_log = "メモを保存しました。\n\n"

    return output_log
