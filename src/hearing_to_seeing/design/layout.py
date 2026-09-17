from PIL import ImageFont

# 실제 설치 경로로 바꿔주세요 (otf 또는 ttf 파일)
FONT_PATH = r"C:\Users\cse_123\Downloads\Pretendard-1.3.9\public\static\Pretendard-ExtraBold.otf"

_font_cache: dict[int, "ImageFont.FreeTypeFont"] = {}


def _get_font(size_px: int) -> "ImageFont.FreeTypeFont":
    if size_px not in _font_cache:
        _font_cache[size_px] = ImageFont.truetype(FONT_PATH, size_px)
    return _font_cache[size_px]


def text_width(text: str, size_px: float) -> float:
    font = _get_font(max(1, round(size_px)))
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]