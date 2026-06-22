import os

# ─────────────────────────────────────────────
# 모델 설정
# ─────────────────────────────────────────────
BASE_MODEL = "/models/models/WAI-illustrious-SDXL_17.safetensors"
LORA_SCALE = 1.0

# ─────────────────────────────────────────────
# 생성 설정
# ─────────────────────────────────────────────
WIDTH      = 832
HEIGHT     = 1216
STEPS      = 28
CFG_SCALE  = 7.0
SEED       = 42
NUM_IMAGES = 4

# ─────────────────────────────────────────────
# 캐릭터 정의 — data/characters.yaml 가 단일 소스
# 표정 메뉴      — data/expressions.yaml (4컷 생성 시 GPT 가 선택)
# (수정은 prompts.py 가 아니라 data/*.yaml 에서 한다)
# ─────────────────────────────────────────────
try:
    import config_loader  # Modal: /root/config_loader.py (flat)
except ModuleNotFoundError:  # 로컬 패키지 컨텍스트
    from . import config_loader  # type: ignore

CHARS = config_loader.load_characters()
EXPRESSIONS = config_loader.load_expressions()
BACKGROUND_SETS = config_loader.load_background_sets()
resolve_expression = config_loader.resolve_expression
compose_char_tags = config_loader.compose_char_tags   # appearance + hair(default|flashback) + body
is_flashback = config_loader.is_flashback             # academic_uniform 이면 과거 회상

# ─────────────────────────────────────────────
# 배경 스타일 (SDXL background tags)
# ─────────────────────────────────────────────
COMIC_BG_STYLE = ["simple background", "white background"]

# ─────────────────────────────────────────────
# 복장 조합 — outfit(옷) + state(노출) + hair(머리) + props(소품) 를 합친다.
# data/characters.yaml 의 outfit 항목은 dict {outfit, state, props, hair_override} 형태.
# 씬(패널)별로 hair / props 를 갈아끼울 수 있다.
# 과거 문자열 outfit 도 그대로 지원(폴백).
# ─────────────────────────────────────────────
def compose_outfit(val, hair: str | None = None, props_extra: str | None = None) -> str:
    """outfit 값(dict 또는 str) → 최종 SDXL 태그 문자열.

    출력 순서: outfit, state, hair, props
    hair        : 씬 머리 오버라이드 — 주면 outfit 의 hair_override 보다 우선
    props_extra : 소품 추가 — 기존 props 에 덧붙임
    """
    if isinstance(val, dict):
        parts = [val.get("outfit", "")]
        state = val.get("state", "")
        if state:
            parts.append(state)
        h = hair or val.get("hair_override") or val.get("hair", "")
        if h:
            parts.append(h)
        props = val.get("props", "")
        if props_extra:
            props = f"{props}, {props_extra}" if props else props_extra
        if props:
            parts.append(props)
        return ", ".join(p for p in parts if p)
    # legacy 문자열
    s = str(val)
    if hair:
        s = f"{s}, {hair}"
    if props_extra:
        s = f"{s}, {props_extra}"
    return s

# ─────────────────────────────────────────────
# 캐릭터별 ElevenLabs 음성 ID
# ─────────────────────────────────────────────
# 계정별 ElevenLabs voice ID — 소스 하드코딩 시 레포 공개 시 유출된다.
# 환경변수(ELEVEN_VOICE_<KEY>)로 덮어쓸 수 있게 하고, 기존 ID 는 fallback 으로만 둔다.
def _voice(key: str, default: str) -> str:
    return os.getenv(f"ELEVEN_VOICE_{key.upper().replace('-', '_')}", default)

CHARACTER_VOICES: dict[str, str] = {
    "hanyoil":   _voice("hanyoil",   "EST9Ui6982FZPSi7gCHi"),  # Belle :  EST9Ui6982FZPSi7gCHi
    "ru-ha":     _voice("ru-ha",     "exsUS4vynmxd379XN4yO"),  # Blondie :  exsUS4vynmxd379XN4yO
    "so-ae":     _voice("so-ae",     "pjcYQlDFKMbcOUp6F5GD"),  # Brittney : pjcYQlDFKMbcOUp6F5GD
    "hanyuyeon": _voice("hanyuyeon", "XiPS9cXxAVbaIWtGDHDh"),  # Jessica
    "hyo-jeong": _voice("hyo-jeong", "7YaUDeaStRuoYg3FKsmU"),  # Callies
    "_default":  _voice("default",   "EXAVITQu4vr4xnSDxMaL"),  # Sarah
}
# ─────────────────────────────────────────────
# ★ 캐릭터 선택 ★
# ─────────────────────────────────────────────
CHAR_NAME   = "hanyoil"   # hanyoil / ru-ha / so-ae
OUTFIT_NAME = "workplace"  # workplace / street / athleisure / campus / minimal / uniform / swimwear / bare / sleepwear / costume

# 자동 추출 (수정 불필요)
_char        = CHARS[CHAR_NAME]
TRIGGER_WORD = CHAR_NAME
LORA_PATH    = _char["lora"]
_flashback   = is_flashback(OUTFIT_NAME)
_tags        = compose_char_tags(_char, flashback=_flashback).strip(", ")

# ─────────────────────────────────────────────
# 생성 프롬프트 (자동 조합)
# ─────────────────────────────────────────────
# 품질 태그 — 단일 소스. sd_generate.py(Modal) 와 아래 PROMPT/INPAINT_PROMPT 모두 참조.
QUALITY_TAGS         = "masterpiece, best quality, high-detailed, high contrast"
INPAINT_QUALITY_TAGS = "masterpiece, best quality, high-detailed, high contrast, detailed eyes"
OBJECT_PANEL_TAGS    = "no humans, no people, empty scene, object focus, still life"
OBJECT_NEGATIVE_TAGS = "1girl, girl, woman, face, body, hands, portrait, character focus"

PROMPT = (
    f"{TRIGGER_WORD}, {_tags}, "
    f"{_char['expression']}, "
    f"{_char['face_state']}, "
    f"{_char['action']}, "
    f"{compose_outfit(_char['outfits'].get(OUTFIT_NAME, ''))}, "
    f"{QUALITY_TAGS}, white background"
)
NEGATIVE_PROMPT = (
    "extra toes, lowres, (bad), text, error, fewer, extra, missing, worst quality, "
    "2girls, multiple girls, "
    "jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, "
    "signature, artistic error, username, bad feet, english text, shiny hair, "
)
REVIEW_CARD_NEGATIVE_TAGS = (
    "text, english text, japanese text, chinese text, korean text, letters, numbers, "
    "logo, watermark, signature, username, multiple girls, 2girls, extra person, "
    "duplicate, clone, multiple views, cropped head, cropped hands, bad hands, "
    "extra fingers, fewer fingers"
)

# ─────────────────────────────────────────────
# ADetailer (얼굴 인페인팅) 설정
# ─────────────────────────────────────────────
INPAINT_PROMPT = (
    f"{TRIGGER_WORD}, "
    f"{_char['appearance_tags']}, {config_loader.char_hair(_char, _flashback)}, "
    f"{_char['expression']}, "
    f"{INPAINT_QUALITY_TAGS}"
)
INPAINT_NEGATIVE = (
    "lowres, bad anatomy, bad face, ugly face, blurry face, "
    "bad eyes, asymmetrical eyes, worst quality, low quality"
)

INPAINT_STEPS    = 30
INPAINT_CFG      = 7.0
INPAINT_STRENGTH = 0.6
INPAINT_PADDING  = 40
INPAINT_SEED     = 42
CROP_SIZE        = 1024

# ─────────────────────────────────────────────
# Compel (프롬프트 가중치 + 77 토큰 제한 해제)
# ─────────────────────────────────────────────
# 사용 예: PROMPT 에 (word:1.3) , [word]^- 같은 가중치 문법 가능
# 자세한 문법: https://github.com/damian0815/compel
USE_COMPEL = True

# ─────────────────────────────────────────────
# 4컷 만화 설정
# ─────────────────────────────────────────────
COMIC_CHAR = "hanyoil"
COMIC_GRID = (2, 2)       # (cols, rows)
COMIC_GAP  = 12           # 패널 간 여백 (px)
COMIC_BG   = (30, 30, 30) # 그리드 배경색

# 패널별로 char 지정 — 다른 캐릭터끼리 번갈아 등장 가능
# 시나리오: 한요일 ↔ 루하  직장 상사/부하 4컷
COMIC_PANELS = [
    {
        "panel_type": "character",
        "char": "hanyoil",
        "outfit": "workplace_3",
        "action": "gesturing, leaning slightly forward",
        "subject": "",
        "expression": "smile face",
        "face_state": "looking at viewer",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "Hey So-ae, got a second?",
        "bubble_kr": "소애, 잠깐 시간 돼?",
        "seed_offset": 0
    },
    {
        "panel_type": "character",
        "char": "so-ae",
        "outfit": "workplace_1",
        "action": "sitting upright, hands resting on the table",
        "subject": "",
        "expression": "light smile face",
        "face_state": "looking at viewer",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "Sure, what do you need?",
        "bubble_kr": "응, 뭐 물어볼 거 있어?",
        "seed_offset": 7
    },
    {
        "panel_type": "character",
        "char": "hanyoil",
        "outfit": "workplace_3",
        "action": "hands together, polite gesture",
        "subject": "",
        "expression": "smile face",
        "face_state": "looking at viewer",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "Can I ask you a quick question?",
        "bubble_kr": "잠깐 질문 하나 드려도 될까요?",
        "seed_offset": 13
    },
    {
        "panel_type": "character",
        "char": "so-ae",
        "outfit": "workplace_1",
        "action": "hands raised slightly, emphasizing points",
        "subject": "",
        "expression": "serious face",
        "face_state": "none",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "Well, it all started with the agenda, so let's break it down.",
        "bubble_kr": "그게 말이야, 아젠다부터 시작해서 쭉 설명할게.",
        "seed_offset": 21
    },
    {
        "panel_type": "character",
        "char": "hanyoil",
        "outfit": "workplace_3",
        "action": "glancing at clock, slight sideways tilt",
        "subject": "",
        "expression": "serious face",
        "face_state": "looking at clock",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "(internally) This might take a while.",
        "bubble_kr": "(속으로) 오래 걸리겠는데.",
        "seed_offset": 29
    },
    {
        "panel_type": "character",
        "char": "so-ae",
        "outfit": "workplace_1",
        "action": "hands on table, finishing explanation",
        "subject": "",
        "expression": "serious face",
        "face_state": "none",
        "background": "meeting room, sticky notes, meeting notes",
        "location": "meeting_room",
        "used_in": "workplace",
        "target_sentence": "Can I ask you a quick question?",
        "bubble": "And that's how it connects perfectly!",
        "bubble_kr": "이렇게 해서 다 딱 맞아떨어지는 거지!",
        "seed_offset": 37
    }
]

# ─────────────────────────────────────────────
# 말풍선 (Speech Bubble)
# ─────────────────────────────────────────────
BUBBLE_DIR        = "/root/textbubble"
BUBBLE_FONT       = "/root/Font/DXMSubtitlesM-KSCpc-EUC-H.ttf"
BUBBLE_TEXT_COLOR = (25, 25, 25)
BUBBLE_FONT_SIZE  = 32          # 기준 폰트 크기 (텍스트 길어지면 자동 축소)
BUBBLE_MIN_FONT   = 18          # 자동 축소 하한
BUBBLE_WIDTH_RATIO = 0.55       # 패널 단변 대비 풍선 폭
BUBBLE_MARGIN     = 24          # 패널 가장자리 여유
BUBBLE_TEXT_INSET = 0.18        # 풍선 내부 텍스트 여백 (풍선 폭 대비)
BUBBLE_LINE_H     = 1.18        # 줄 간격 배수
