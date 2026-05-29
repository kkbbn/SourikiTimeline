import os
import argparse
import gradio as gr
from gradio_modal import Modal

from scripts.chara_skill import CharaSkill
from scripts.common_utils import load_timeline, load_memo
from scripts.config_utils import AppConfig, ProjectConfig, get_timeline_columns
from scripts.gradio_utils import download_video_gr, mask_preview_gr, save_mask_gr, save_memo_gr, timeline_generate_gr, create_project_gr, delete_project_gr, load_app_gr, reload_workspace_gr, select_project_gr, select_workspace_gr, timeline_update_gr, load_mask_gr
from scripts.platform_utils import open_path_in_explorer


def apply_gradio_compatibility_patches():
    # gradio==4.44.x can receive boolean JSON schema nodes from newer pydantic.
    # Treat boolean schemas as bool types instead of raising TypeError.
    try:
        from gradio_client import utils as gradio_client_utils

        original_get_type = gradio_client_utils.get_type

        def safe_get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return original_get_type(schema)

        gradio_client_utils.get_type = safe_get_type
    except Exception:
        pass

    # Gradio checks URL reachability using the bind address. For 0.0.0.0 this can
    # fail even when the server is healthy, so probe localhost instead.
    try:
        from gradio import networking as gradio_networking

        original_url_ok = gradio_networking.url_ok

        def safe_url_ok(url: str) -> bool:
            if isinstance(url, str) and "://0.0.0.0" in url:
                url = url.replace("://0.0.0.0", "://127.0.0.1")
            return original_url_ok(url)

        gradio_networking.url_ok = safe_url_ok
    except Exception:
        pass


movie_downloaders = ["pytube", "yt-dlp"]
download_formats = ["mp4", "webm"]
chara_names = [x.chara_name for x in CharaSkill.from_tsv()]
chara_names.sort()
tsv_separators = [("タブ", "\t"), ("スペース", " "), ("カンマ", ",")]

app_config = AppConfig.instance()
config = ProjectConfig.load(app_config.project_path)

source_dataframe = load_timeline(app_config.project_path)
dataframe, dataframe_tsv = config.convert_timeline_and_tsv(source_dataframe)

source_memo = load_memo(app_config.project_path)

def get_mask_image_names():
    return [f for f in os.listdir(os.path.join("resources", "mask")) if f.endswith(".png")]

def add_space(space_count:int=1):
    for i in range(0, space_count):
        gr.HTML(value="")

js = """
function () {
    gradioURL = window.location.href
    if (!gradioURL.endsWith('?__theme=dark')) {
        window.location.replace(gradioURL + '?__theme=dark');
    }
}
"""

with gr.Blocks(title="総力戦タイムラインメーカー", js=js) as demo:
    gr.Markdown("総力戦タイムラインメーカー")

    with gr.Row():
        with gr.Column(scale=1, min_width=150):
            project_image = gr.Image(show_label=False, value=app_config.get_current_preimage(), width=150)
        with gr.Column(scale=2.5):
            title_textbox = gr.Textbox(label="タイトル", value=config.title, interactive=False)
        with gr.Column(scale=0.5, min_width=50):
            open_project_button = gr.Button("開く", variant="secondary", size="sm")
            delete_project_button = gr.Button("削除", variant="stop", size="sm")
        with gr.Column(scale=3):
            base_output = gr.Markdown(show_label=False)

    with gr.Tabs():
        with gr.TabItem("動画一覧"):
            with gr.Row():
                with gr.Column(scale=1.5):
                    add_space(1)
                    with gr.Row():
                        show_create_project_button = gr.Button("新規作成", variant="primary")
                        workspace_reload_button = gr.Button("リスト更新", variant="primary")
                    workspace_gallery = gr.Gallery(value=app_config.get_all_gallery(), columns=4, allow_preview=False)
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Column(scale=5):
                            workspace_path_textbox = gr.Textbox(label="ワークスペースパス", value=app_config.workspace_path, info="動画/タイムラインの保存先ディレクトリ", interactive=False)
                        with gr.Column(scale=1, min_width=50):
                            workspace_open_button = gr.Button("選択", variant="primary")

                    auto_save_checkbox = gr.Checkbox(value=app_config.auto_save, label="Auto Save", visible=False)
                    thumbnail_width_slider = gr.Number(value=app_config.thumbnail_width, label="Thumbnail Width", visible=False)
                    thumbnail_height_slider = gr.Number(value=app_config.thumbnail_height, label="Thumbnail Height", visible=False)
                    thumbnail_file_textbox = gr.Textbox(label="Thumbnail File Name", value=config.movie_thumbnail_file_name, visible=False)

                    movie_url_textbox = gr.Textbox(label="Youtube URL", value=config.movie_url, interactive=False)
                    artist_textbox = gr.Textbox(label="投稿者", value=config.author, interactive=False)
                    project_path_textbox = gr.Textbox(label="動画保存フォルダ", value=app_config.project_path, visible=False)

        with gr.TabItem("ダウンロード"):
            with gr.Row():
                with gr.Column():
                    add_space(1)
                    with gr.Row():
                        movie_download_button = gr.Button("動画ダウンロード", variant="primary")
                    with gr.Row():
                        downloader_dropdown = gr.Dropdown(movie_downloaders, value=app_config.downloader, label="ダウンローダー")
                        download_format_dropdown = gr.Dropdown(download_formats, value=app_config.download_format, label="フォーマット")
                        movie_download_file_textbox = gr.Textbox(label="ダウンロードファイル名", value=config.movie_download_file_name, visible=False)
                with gr.Column():
                    add_space(1)

        with gr.TabItem("マスク調整"):
            with gr.Row():
                with gr.Column():
                    add_space(1)
                    with gr.Row():
                        mask_preview_button = gr.Button("プレビュー更新", variant="primary")
                    with gr.Tabs():
                        with gr.TabItem("プレビュー設定"):
                            with gr.Row():
                                movie_preview_time_slider = gr.Slider(value=config.movie_preview_time, label="プレビュー時間(秒)", info="スキル発動時間に合わせて手動調整してください", minimum=config.movie_start_time, maximum=config.movie_end_time, step=0.1)
                            with gr.Accordion("動画クロップ設定"):
                                with gr.Row():
                                    movie_x_slider = gr.Number(value=config.movie_x, label="X")
                                    movie_y_slider = gr.Number(value=config.movie_y, label="Y")
                                with gr.Row():
                                    movie_width_slider = gr.Number(value=config.movie_width, label="Width")
                                    movie_height_slider = gr.Number(value=config.movie_height, label="Height")
                            with gr.Accordion("タイムライン対象時間(秒)"):
                                with gr.Row():
                                    movie_start_time_slider = gr.Number(value=config.movie_start_time, label="Start")
                                    movie_end_time_slider = gr.Number(value=config.movie_end_time, label="End")
                        with gr.TabItem("マスク設定"):
                            add_space(1)
                            with gr.Row():
                                show_load_mask_button = gr.Button("マスクのロード", variant="primary")
                                show_save_mask_button = gr.Button("マスクの保存", variant="primary")
                            with gr.Accordion("スキルマスク範囲(赤枠)", open=False):
                                with gr.Row():
                                    mask_skill_x_slider = gr.Number(value=config.mask_skill_x, label="X")
                                    mask_skill_y_slider = gr.Number(value=config.mask_skill_y, label="Y")
                                with gr.Row():
                                    mask_skill_w_slider = gr.Number(value=config.mask_skill_w, label="Width")
                                    mask_skill_h_slider = gr.Number(value=config.mask_skill_h, label="Height")
                            with gr.Accordion("コストマスク範囲(青枠)", open=False):
                                with gr.Row():
                                    mask_cost_x_slider = gr.Number(value=config.mask_cost_x, label="X")
                                    mask_cost_y_slider = gr.Number(value=config.mask_cost_y, label="Y")
                                with gr.Row():
                                    mask_cost_w_slider = gr.Number(value=config.mask_cost_w, label="Width")
                                    mask_cost_h_slider = gr.Number(value=config.mask_cost_h, label="Height")
                            with gr.Accordion("時間マスク範囲(緑枠)", open=False):
                                with gr.Row():
                                    mask_time_x_slider = gr.Number(value=config.mask_time_x, label="X")
                                    mask_time_y_slider = gr.Number(value=config.mask_time_y, label="Y")
                                with gr.Row():
                                    mask_time_w_slider = gr.Number(value=config.mask_time_w, label="Width")
                                    mask_time_h_slider = gr.Number(value=config.mask_time_h, label="Height")
                            with gr.Accordion("コストバー設定", open=False):
                                with gr.Row():
                                    mask_cost_color1_textbox = gr.ColorPicker(value=config.mask_cost_color1, label="基準色", info="コストバー判定に使用する基準色 デフォルト(0, 180, 250)")
                                    mask_cost_color2_textbox = gr.ColorPicker(value=config.mask_cost_color2, label="エフェクト色", info="コストバー判定に使用するエフェクト色 デフォルト(255, 255, 255)")
                                    mask_cost_color_threshold_slider = gr.Number(value=config.mask_cost_color_threshold, label="色の誤差許容値", info="デフォルト20")
                            with gr.Accordion("スキル判定設定", open=False):
                                with gr.Row():
                                    mask_skill_color1_textbox = gr.ColorPicker(value=config.mask_skill_color1, label="スキル判定色1", info="スキル判定に使用する色 デフォルト(255, 255, 255)")
                                    mask_skill_color2_textbox = gr.ColorPicker(value=config.mask_skill_color2, label="スキル判定色2", info="スキル判定に使用する色 デフォルト(100, 100, 100)")
                                with gr.Row():
                                    mask_skill_color_threshold_slider = gr.Number(value=config.mask_skill_color_threshold, label="色の誤差許容値", info="デフォルト20")
                                    mask_skill_color_fill_percentage_slider = gr.Number(value=config.mask_skill_color_fill_percentage, label="判定色の割合(%)", info="マスク範囲内の判定色が、この割合よりも多い場合にスキル判定を行う デフォルト30(%)")
                            mask_image_w_slider = gr.Number(value=config.mask_image_w, label="Width", visible=False)
                            mask_image_h_slider = gr.Number(value=config.mask_image_h, label="Height", visible=False)

                with gr.Column():
                    movie_preview_image = gr.Image(label="Preview Image", interactive=False)
                    movie_input_video = gr.Video(label="Download Video")

        with gr.TabItem("タイムライン生成"):
            with gr.Row():
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("生成設定"):
                            add_space(1)
                            with gr.Row():
                                timeline_generate_button = gr.Button("タイムライン生成", variant="primary")
                            with gr.Row():
                                movie_frame_rate_slider = gr.Slider(value=config.movie_frame_rate, label="読み取りフレームレート", minimum=0, maximum=60, step=1)
                            with gr.Row():
                                timeline_ignore_chara_names_dropdown = gr.Dropdown(chara_names, value=config.timeline_ignore_chara_names, multiselect=True, label="除外キャラ名一覧", info="ギミックの文字などで誤認識する場合に指定")
                            with gr.Row():
                                timeline_max_time_number = gr.Number(value=config.timeline_max_time, label="バトルの制限時間(秒)")
                            with gr.Row():
                                timeline_max_cost_slider = gr.Slider(value=app_config.timeline_max_cost, label="コスト最大値", info="コストバーの満タン時コスト(タイムライン生成時に使用)", minimum=10, maximum=11, step=0.5)
                        with gr.TabItem("表示設定(共通)"):
                            add_space(1)
                            with gr.Row():
                                timeline_update_button = gr.Button("表示更新", variant="primary")
                            with gr.Row():
                                timeline_visible_columns_checkbox = gr.CheckboxGroup(label="表示カラム", choices=get_timeline_columns(), value=app_config.timeline_visible_columns)
                            with gr.Row():
                                timeline_cost_omit_seconds_slider = gr.Slider(value=app_config.timeline_cost_omit_seconds, label="コストを省略する秒数", info="前回のスキルを発動してからの経過時間が指定秒数以下の場合、コストを省略する", minimum=0, maximum=10, step=0.1)
                            with gr.Row():
                                timeline_remain_cost_omit_value_slider = gr.Slider(value=app_config.timeline_remain_cost_omit_value, label="コストを省略する残コスト", info="残コストが指定値以下の場合、コストを省略する", minimum=0, maximum=10, step=0.1)
                            with gr.Row():
                                timeline_newline_chara_names_dropdown = gr.Dropdown(chara_names, value=app_config.timeline_newline_chara_names, multiselect=True, label="スキル発動時に改行するキャラ名一覧")
                            with gr.Row():
                                timeline_newline_before_chara = gr.Checkbox(value=app_config.timeline_newline_before_chara, label="発動キャラの前で改行する")
                                timeline_newline_after_chara = gr.Checkbox(value=app_config.timeline_newline_after_chara, label="発動キャラの後で改行する")
                            with gr.Row():
                                timeline_tsv_separator_dropdown = gr.Dropdown(choices=tsv_separators, value=app_config.timeline_tsv_separator, label="TSVの区切り文字")

                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Table"):
                            timeline_dataframe = gr.Dataframe(dataframe, interactive=False)
                        with gr.TabItem("TSV"):
                            timeline_dataframe_tsv = gr.Textbox(dataframe_tsv, show_label=False, interactive=False, lines=20)
                        with gr.TabItem("Memo"):
                            timeline_memo_textbox = gr.Textbox(source_memo, label="整形後のTLの保存などで使用できます", lines=20, interactive=True)
                            timeline_save_memo_button = gr.Button("保存", variant="primary")

    # プロジェクト作成モーダル
    create_project_modal = Modal("プロジェクト作成", visible=False)
    with create_project_modal:
        with gr.Row():
            with gr.Column(scale=1.5):
                create_project_url_textbox = gr.Textbox(label="タイムライン出力する動画のURL", value="", info="タイムラインの出力をしたいYouTubeのURLを入力してください")
                create_project_button = gr.Button("作成", variant="primary")
            with gr.Column(scale=1):
                create_project_output = gr.Markdown(show_label=False)

    # マスクのロードモーダル
    load_mask_modal = Modal("マスクのロード", visible=False)
    with load_mask_modal:
        with gr.Row():
            with gr.Column(scale=1.5):
                mask_image_name_dropdown = gr.Dropdown(get_mask_image_names(), value=config.mask_image_name, label="マスク画像名", info="OCRで読み取るマスク範囲を選択します。resources\maskに格納されている画像を使用できます")
                load_mask_button = gr.Button("ロード", variant="primary")
            with gr.Column(scale=1):
                load_mask_output = gr.Markdown(show_label=False)
        add_space(10)

    # マスクの保存モーダル
    save_mask_modal = Modal("マスクの保存", visible=False)
    with save_mask_modal:
        with gr.Row():
            with gr.Column(scale=1.5):
                save_mask_image_name_text = gr.Textbox(label="マスク画像名", info="保存するファイル名を入力してください。resources\maskに保存されます")
                save_mask_button = gr.Button("保存", variant="primary")
            with gr.Column(scale=1):
                save_mask_output = gr.Markdown(show_label=False)

    # 画像確認モーダル
    confirm_image_modal = Modal("画像確認", visible=False)
    with confirm_image_modal:
        confirm_image = gr.Image(show_label=False)

    app_config_inputs = [
        project_path_textbox,
        workspace_path_textbox,
        auto_save_checkbox,
        download_format_dropdown,
        downloader_dropdown,
        thumbnail_width_slider,
        thumbnail_height_slider,

        timeline_tsv_separator_dropdown,
        timeline_visible_columns_checkbox,
        timeline_cost_omit_seconds_slider,
        timeline_remain_cost_omit_value_slider,
        timeline_max_cost_slider,
        timeline_newline_chara_names_dropdown,
        timeline_newline_before_chara,
        timeline_newline_after_chara,
    ]

    inputs = [
        title_textbox,
        artist_textbox,

        movie_url_textbox,
        movie_download_file_textbox,
        thumbnail_file_textbox,
        movie_start_time_slider,
        movie_end_time_slider,
        movie_x_slider,
        movie_y_slider,
        movie_width_slider,
        movie_height_slider,
        movie_frame_rate_slider,
        movie_preview_time_slider,

        mask_image_name_dropdown,

        mask_image_w_slider,
        mask_image_h_slider,

        mask_skill_x_slider,
        mask_skill_y_slider,
        mask_skill_w_slider,
        mask_skill_h_slider,

        mask_cost_x_slider,
        mask_cost_y_slider,
        mask_cost_w_slider,
        mask_cost_h_slider,

        mask_time_x_slider,
        mask_time_y_slider,
        mask_time_w_slider,
        mask_time_h_slider,

        mask_cost_color1_textbox,
        mask_cost_color2_textbox,
        mask_cost_color_threshold_slider,

        mask_skill_color1_textbox,
        mask_skill_color2_textbox,
        mask_skill_color_threshold_slider,
        mask_skill_color_fill_percentage_slider,

        timeline_ignore_chara_names_dropdown,
        timeline_max_time_number,
    ]

    outputs = [
        movie_preview_image,
        movie_input_video,
    ]

    mask_outputs = [
        mask_image_w_slider,
        mask_image_h_slider,

        mask_skill_x_slider,
        mask_skill_y_slider,
        mask_skill_w_slider,
        mask_skill_h_slider,

        mask_cost_x_slider,
        mask_cost_y_slider,
        mask_cost_w_slider,
        mask_cost_h_slider,

        mask_time_x_slider,
        mask_time_y_slider,
        mask_time_w_slider,
        mask_time_h_slider,
    ]

    demo.load(load_app_gr,
              inputs=[],
              outputs=[
                    workspace_path_textbox,
                    workspace_gallery,
                    project_path_textbox,
                    project_image,
                    timeline_dataframe,
                    timeline_dataframe_tsv,
                    timeline_memo_textbox,
                    *inputs,
                    *outputs,
              ])

    show_create_project_button.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=create_project_modal
    )

    def open_project():
        prohect_path = app_config.project_path
        open_path_in_explorer(prohect_path)

    open_project_button.click(open_project,
                                inputs=None,
                                outputs=None
                            )

    delete_project_button.click(delete_project_gr,
                                inputs=[],
                                outputs=[
                                    base_output,
                                    workspace_gallery,
                                    project_path_textbox,
                                    project_image,
                                    timeline_dataframe,
                                    timeline_dataframe_tsv,
                                    timeline_memo_textbox,
                                    *inputs,
                                    *outputs,
                                ])

    show_load_mask_button.click(
        fn=lambda: [gr.update(visible=True), gr.update(choices=get_mask_image_names())],
        inputs=None,
        outputs=[load_mask_modal, mask_image_name_dropdown]
    )

    show_save_mask_button.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=save_mask_modal
    )

    movie_preview_image.select(
        fn=lambda image: [gr.update(visible=True), image],
        inputs=movie_preview_image,
        outputs=[confirm_image_modal, confirm_image]
    )

    workspace_open_button.click(select_workspace_gr,
                                  inputs=[],
                                  outputs=[
                                        base_output,
                                        workspace_path_textbox,
                                        workspace_gallery,
                                  ])

    workspace_gallery.select(select_project_gr,
                             inputs=[],
                             outputs=[
                                    base_output,
                                    project_path_textbox,
                                    project_image,
                                    timeline_dataframe,
                                    timeline_dataframe_tsv,
                                    timeline_memo_textbox,
                                    *inputs,
                                    *outputs,
                             ])

    create_project_button.click(create_project_gr,
                                  inputs=[
                                        create_project_url_textbox,
                                  ],
                                  outputs=[
                                        create_project_output,
                                        workspace_gallery,
                                        project_path_textbox,
                                        project_image,
                                        timeline_dataframe,
                                        timeline_dataframe_tsv,
                                        timeline_memo_textbox,
                                        *inputs,
                                        *outputs,
                                  ])

    workspace_reload_button.click(reload_workspace_gr,
                                  inputs=[],
                                  outputs=workspace_gallery)

    movie_download_button.click(download_video_gr,
                          inputs=[
                                *app_config_inputs,
                                *inputs,
                          ],
                          outputs=[
                                base_output,
                                movie_input_video,
                                movie_preview_image,
                                movie_end_time_slider,
                                movie_width_slider,
                                movie_height_slider,
                                movie_preview_time_slider,
                                *mask_outputs,
                          ])

    mask_preview_button.click(mask_preview_gr,
                          inputs=[
                                *app_config_inputs,
                                *inputs,
                          ],
                          outputs=[
                                base_output,
                                movie_input_video,
                                movie_preview_image,
                                movie_preview_time_slider,
                          ])

    load_mask_button.click(load_mask_gr,
                            inputs=[
                                *app_config_inputs,
                                *inputs,
                            ],
                            outputs=[
                                load_mask_output,
                                movie_input_video,
                                movie_preview_image,
                                movie_preview_time_slider,
                                *mask_outputs,
                            ])

    save_mask_button.click(save_mask_gr,
                            inputs=[
                                save_mask_image_name_text,
                            ],
                            outputs=[
                                save_mask_output,
                                mask_image_name_dropdown,
                                *mask_outputs,
                            ])

    timeline_generate_button.click(timeline_generate_gr,
                          inputs=[
                                *app_config_inputs,
                                *inputs,
                          ],
                          outputs=[
                                base_output,
                                timeline_dataframe,
                                timeline_dataframe_tsv,
                          ])

    timeline_update_button.click(timeline_update_gr,
                          inputs=[
                                *app_config_inputs,
                                *inputs,
                          ],
                          outputs=[
                                base_output,
                                timeline_dataframe,
                                timeline_dataframe_tsv,
                          ])

    timeline_save_memo_button.click(save_memo_gr,
                            inputs=[
                                    timeline_memo_textbox,
                            ],
                            outputs=base_output)

if __name__ == "__main__":
    apply_gradio_compatibility_patches()

    parser = argparse.ArgumentParser(description="Launch the SourikiTimeline Gradio app")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host/IP for Gradio server (e.g. 127.0.0.1, 0.0.0.0)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically on launch",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    args = parser.parse_args()

    demo.launch(
        server_name=args.host,
        inbrowser=not args.no_browser,
        share=args.share,
        allowed_paths=[app_config.workspace_path],
    )
