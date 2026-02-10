import os
import shutil
import subprocess
import urllib.parse
from moviepy import VideoFileClip
from moviepy.config import FFMPEG_BINARY
from pytubefix import YouTube
from PIL import Image
from yt_dlp import YoutubeDL

from scripts.common_utils import get_tmp_dir, get_tmp_file_name, get_tmp_file_path
from scripts.debug_utils import debug_args

@debug_args
def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    aspect_ratio = width / height
    
    # ターゲットのサイズとアスペクト比を設定
    target_width = int(size[0])
    target_height = int(size[1])
    target_ratio = target_width / target_height

    if aspect_ratio > target_ratio:
        # 元の画像の方が横長
        new_width = target_width
        new_height = round(target_width / aspect_ratio)
    else:
        # 元の画像の方が縦長またはアスペクト比が等しい
        new_height = target_height
        new_width = round(target_height * aspect_ratio)
    
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 新しい画像を作成して黒で塗りつぶす
    new_image = Image.new("RGB", (target_width, target_height), "black")
    # リサイズした画像を新しい画像の中央に配置
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    # 画像を保存
    new_image.save(output_image_path, quality = 85)

@debug_args
def get_video_info(url, downloader):
    if downloader == "pytube":
        yt = YouTube(url)

        title = yt.title
        author = yt.author
        thumbnail_url = yt.thumbnail_url

    elif downloader == "yt-dlp":
        option = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        with YoutubeDL(option) as ydl:
            info = ydl.extract_info(url, download=False)

            title = info.get('title', '')
            author = info.get('uploader', '')
            thumbnail_url = info.get('thumbnail', '')
    else:
        raise Exception(f"サポートされていないダウンローダーです。 downloader: {downloader}")

    return title, author, thumbnail_url

@debug_args
def format_youtube_url(url):
    # urlからvパラメータ以外を削除
    pr = urllib.parse.urlparse(url)
    qsl = urllib.parse.parse_qsl(pr.query)
    qsl = [q for q in qsl if q[0] == "v"]
    pr = pr._replace(query=urllib.parse.urlencode(qsl))
    url = urllib.parse.urlunparse(pr)
    return url

@debug_args
def pytube_download(url, output_dir, file_name, mime_type, order_key):
    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(output_path):
        os.remove(output_path)
    
    yt = YouTube(url)
    video = yt.streams.filter(mime_type=mime_type).order_by(order_key).desc().first()
    video.download(output_path=output_dir, filename=file_name)

@debug_args
def download_youtube_with_pytube(url, output_path):
    ext = os.path.splitext(output_path)[1]

    tmp_dir = get_tmp_dir()
    
    yt = YouTube(url)
    format_list = yt.streams
    for format in format_list:
        print(format)

    if ext == ".mp4":
        video_filename = get_tmp_file_name(".mp4")
        audio_filename = get_tmp_file_name(".m4a")
        pytube_download(url, tmp_dir, video_filename, "video/mp4", 'resolution')
        pytube_download(url, tmp_dir, audio_filename, "audio/mp4", 'abr')
    elif ext == ".webm":
        video_filename = get_tmp_file_name(".webm")
        audio_filename = get_tmp_file_name(".webm")
        pytube_download(url, tmp_dir, video_filename, "video/webm", 'resolution')
        pytube_download(url, tmp_dir, audio_filename, "audio/webm", 'abr')
    else:
        raise Exception(f"サポートされていない拡張子です。 ext: {ext}")

    merge_video_and_audio(
        os.path.join(tmp_dir, video_filename),
        os.path.join(tmp_dir, audio_filename),
        output_path,
        True
    )

@debug_args
def ydl_download(url, output_path, format):
    if os.path.exists(output_path):
        os.remove(output_path)

    option = {
        'outtmpl': output_path,
        'format': format,
        'ffmpeg_location': FFMPEG_BINARY,
        'cookiesfrombrowser': ('chrome',),
        'remote_components': ['ejs:github'],
    }
    with YoutubeDL(option) as ydl:
        result = ydl.download([url])
        if result != 0:
            raise Exception(f"ダウンロードに失敗しました。 url: {url}")

@debug_args
def download_youtube_with_yt_dlp(url, output_path):
    url = format_youtube_url(url)
    print(f"Download youtube. {url}")

    ext = os.path.splitext(output_path)[1]

    if ext == ".mp4":
        output_video_path = get_tmp_file_path(".mp4")
        output_audio_path = get_tmp_file_path(".m4a")
        ydl_download(url, output_video_path, 'bestvideo[ext=mp4]')
        ydl_download(url, output_audio_path, 'bestaudio[ext=m4a]')
    elif ext == ".webm":
        output_video_path = get_tmp_file_path(".webm")
        output_audio_path = get_tmp_file_path(".webm")
        ydl_download(url, output_video_path, 'bestvideo[ext=webm]')
        ydl_download(url, output_audio_path, 'bestaudio[ext=webm]')
    else:
        raise Exception(f"サポートされていない拡張子です。 ext: {ext}")
    
    merge_video_and_audio(
        output_video_path,
        output_audio_path,
        output_path,
        True
    )

@debug_args
def download_video(url, output_path, downloader):
    duration = 0

    if downloader == "pytube":
        download_youtube_with_pytube(url, output_path)
    elif downloader == "yt-dlp":
        download_youtube_with_yt_dlp(url, output_path)
    else:
        raise Exception(f"サポートされていないダウンローダーです。 downloader: {downloader}")

    print(f"Video download is complete. {output_path}")

    with VideoFileClip(output_path) as video:
        video_size = video.size
        duration = video.duration

    return duration, video_size[0], video_size[1]

@debug_args
def get_video_size(input_path):
    with VideoFileClip(input_path) as video:
        video_size = (video.w, video.h)
        video.close()

    return video_size

@debug_args
def trim_and_crop_video(input_path, output_path, start_time, end_time, width, height, bitrate):
    # mp4でパラメータが全て0の場合は何もしない
    ext = os.path.splitext(input_path)[1]
    if start_time == 0.0 and end_time == 0.0 and width == 0 and height == 0 and ext == ".mp4":
        print(f"Skip trim_and_crop_video. {input_path}")
        return input_path

    with VideoFileClip(input_path) as video:
        if start_time > 0.0 or end_time > 0.0:
            end_time = end_time if end_time > 0.0 else video.duration
            video = video.subclip(start_time, end_time)

        if width > 0 or height > 0:
            width = width if width > 0 else video.w
            height = height if height > 0 else video.h
            print(f"clipping {video.w}x{video.h} -> {width}x{height}")
            video = video.cropped(x_center=video.w/2, y_center=video.h/2, width=width, height=height)
        tmp_file = get_tmp_file_path(".m4a")
        video.write_videofile(output_path, temp_audiofile=tmp_file, codec="libx264", audio_codec="aac", audio_bitrate=bitrate)
        video.close()

    print(f"Video clipping is complete. {output_path}")
    return output_path

@debug_args
def get_audio_volume(audio_file):
    ffmpeg = FFMPEG_BINARY
    null_device = '/dev/null' if os.name == 'posix' else 'NUL'
    cmd = [ffmpeg, '-i', audio_file, '-af', 'volumedetect', '-f', 'null', null_device]
    print(" ".join(cmd))

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    _, stderr = process.communicate()
    print(stderr)

    source_dBFS = float(stderr.split("mean_volume: ")[1].split(" dB")[0])
    return source_dBFS

@debug_args
def normalize_audio(audio_file, target_dBFS, bitrate):
    ext = os.path.splitext(audio_file)[1]
    tmp_input_file = get_tmp_file_path(ext)
    tmp_output_file = get_tmp_file_path(ext)

    shutil.move(audio_file, tmp_input_file)

    source_dBFS = get_audio_volume(tmp_input_file)
    print(f"Normalize audio. {source_dBFS}dB -> {target_dBFS}dB")

    change_in_dBFS = target_dBFS - source_dBFS

    ffmpeg = FFMPEG_BINARY
    cmd = [ffmpeg, '-y', '-i', tmp_input_file, '-af', f'volume={change_in_dBFS}dB', '-ab', bitrate, tmp_output_file]
    print(" ".join(cmd))

    subprocess.run(cmd)

    os.remove(tmp_input_file)
    shutil.move(tmp_output_file, audio_file)

@debug_args
def extract_audio(input_path, output_path, target_dbfs, bitrate):
    with VideoFileClip(input_path) as video:
        audio = video.audio
        audio.write_audiofile(output_path, bitrate=bitrate)
        audio.close()

    if target_dbfs < 0.0:
        normalize_audio(output_path, target_dbfs, bitrate)

    print(f"Audio extract is complete. {output_path}")

@debug_args
def convert_audio(input_file, output_file, bitrate=None, remove_original=True):
    tmp_input_file = get_tmp_file_path(os.path.splitext(input_file)[1])
    tmp_output_file = get_tmp_file_path(os.path.splitext(output_file)[1])

    if remove_original:
        shutil.move(input_file, tmp_input_file)
    else:
        shutil.copy(input_file, tmp_input_file)

    ffmpeg = FFMPEG_BINARY
    cmd = [ffmpeg, '-y', '-i', tmp_input_file]

    if bitrate is not None:
        cmd.append('-ab')
        cmd.append(bitrate)

    cmd.append(tmp_output_file)

    print(" ".join(cmd))

    subprocess.run(cmd)

    os.remove(tmp_input_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.move(tmp_output_file, output_file)

@debug_args
def merge_video_and_audio(input_video_file, input_audio_file, output_file, remove_original=True):
    tmp_input_video_file = get_tmp_file_path(os.path.splitext(input_video_file)[1])
    tmp_input_audio_file = get_tmp_file_path(os.path.splitext(input_audio_file)[1])
    tmp_output_file = get_tmp_file_path(os.path.splitext(output_file)[1])

    shutil.copy(input_video_file, tmp_input_video_file)
    shutil.copy(input_audio_file, tmp_input_audio_file)

    ffmpeg = FFMPEG_BINARY
    cmd = [ffmpeg, '-y', '-i', tmp_input_video_file, '-i', tmp_input_audio_file, '-c:v', 'copy', '-c:a', 'copy', '-map', '0:v:0', '-map', '1:a:0', tmp_output_file]

    print(" ".join(cmd))

    subprocess.run(cmd)

    os.remove(tmp_input_video_file)
    os.remove(tmp_input_audio_file)

    if os.path.exists(output_file):
        os.remove(output_file)
    shutil.move(tmp_output_file, output_file)

    if remove_original:
        os.remove(input_video_file)
        os.remove(input_audio_file)

@debug_args
def get_video_duration(input_path):
    duration = 0.0
    with VideoFileClip(input_path) as video:
        duration = video.duration
        video.close()

    return duration

@debug_args
def extract_video_frame(video_path: str, time: float):
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(time)
    return frame
