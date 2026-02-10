import cv2
import numpy as np
import platform
from paddleocr import PaddleOCR
from PIL import ImageFont, ImageDraw, Image

from scripts.debug_utils import debug_args

def get_system_font():
    """Get a Japanese-capable font based on the operating system."""
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    elif system == 'Windows':
        font_paths = [
            'meiryo.ttc',
            'msgothic.ttc',
            'msmincho.ttc',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf',
        ]

    for font_path in font_paths:
        try:
            ImageFont.truetype(font_path, 10)
            return font_path
        except OSError:
            continue

    return None  # Will use default font

_system_font = get_system_font()

# ocr = PaddleOCR(use_angle_cls=False, lang='japan', show_log=False)
ocrs = {
    'japan': PaddleOCR(use_angle_cls=False, lang='japan', show_log=False),
    'en': PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
}

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def hex_to_hsv(hex_color):
    color = hex_to_rgb(hex_color)
    return cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

def crop_image(image: np.ndarray, image_rect):
    x, y, w, h = image_rect
    if w == 0:
        w = image.shape[1] - x
    if h == 0:
        h = image.shape[0] - y
    return image[int(y):int(y+h), int(x):int(x+w)]

@debug_args
def get_mask_image_rect(image: np.ndarray, hex_color):
    r, g, b = hex_to_rgb(hex_color)
    mask = (image[:, :, 0] == r) & (image[:, :, 1] == g) & (image[:, :, 2] == b)

    red_only = np.zeros_like(image)
    red_only[mask] = image[mask]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(max(contours, key=lambda x: cv2.contourArea(x)))
    print(f"mask x: {x}, y: {y}, w: {w}, h: {h}")

    return (x, y, w, h)

@debug_args
def draw_image_rect(image: np.ndarray, rect, hex_color, thickness=2):
    x, y, w, h = rect
    color = hex_to_rgb(hex_color)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)

@debug_args
def draw_image_line(image: np.ndarray, start, end, hex_color, thickness=2):
    color = hex_to_rgb(hex_color)
    cv2.line(image, start, end, color, thickness)

@debug_args
def draw_image_string(image: np.ndarray, text, position, hex_font_color, font_size=40, stroke_width=3, hex_stroke_color='#ffffff'):
    font_color = hex_to_rgb(hex_font_color)
    stroke_color = hex_to_rgb(hex_stroke_color)

    if _system_font:
        font = ImageFont.truetype(_system_font, int(font_size))
    else:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    draw.text(position, text, fill=font_color, font=font, stroke_width=int(stroke_width), stroke_fill=stroke_color)

    return np.array(img_pil)

def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return equalized

@debug_args
def ocr_image(image: np.ndarray, mask_rect, lang='japan'):
    if mask_rect is not None:
        image = crop_image(image, mask_rect)
    #image = histogram_equalization(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ocr = ocrs[lang]
    result = ocr.ocr(image, cls=False)
    print(result)

    if not result[0]:
        return []

    result_texts = []
    for line in result[0]:
        result_texts.append(line[1][0])

    return result_texts

@debug_args
def get_image_bar_percentage(image: np.ndarray, mask_rect, hex_color1, hex_color2, threshold):
    x, y, w, h = mask_rect
    cropped_image = image[y:y+h, x:x+w]

    rgb1 = hex_to_rgb(hex_color1)
    rgb2 = hex_to_rgb(hex_color2)

    lower_rgb = np.minimum(rgb1, rgb2) - threshold
    upper_rgb = np.maximum(rgb1, rgb2) + threshold

    lower_rgb = np.clip(lower_rgb, 0, 255)
    upper_rgb = np.clip(upper_rgb, 0, 255)

    #print(f"lower_rgb: {lower_rgb}, upper_rgb: {upper_rgb}")

    mask = np.all((cropped_image >= lower_rgb) & (cropped_image <= upper_rgb), axis=-1)

    # y方向において、50%以上のピクセルが有効であるxの位置を抽出
    y_valid = np.sum(mask, axis=0) > (h * 0.5)
    x_list = np.where(y_valid)[0]

    if len(x_list) == 0:
        return 0
    max_x = np.max(x_list)
    
    return int(max_x / w * 100)

# 特定の範囲内の色の割合を取得
@debug_args
def get_color_fill_percentage(image: np.ndarray, mask_rect, hex_color, threshold):
    x, y, w, h = mask_rect
    cropped_image = image[y:y+h, x:x+w]

    rgb = hex_to_rgb(hex_color)

    lower_rgb = np.array(rgb) - threshold
    upper_rgb = np.array(rgb) + threshold

    lower_rgb = np.clip(lower_rgb, 0, 255)
    upper_rgb = np.clip(upper_rgb, 0, 255)

    mask = np.all((cropped_image >= lower_rgb) & (cropped_image <= upper_rgb), axis=-1)

    return int(np.sum(mask) / (w * h) * 100)
