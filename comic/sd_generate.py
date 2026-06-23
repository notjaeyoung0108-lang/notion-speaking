"""sd_generate.py — Modal-runnable SDXL + LoRA + Compel + ADetailer + Speech Bubble.

Replaces the Colab notebook. Reads character/outfit/bubble config from `prompts.py`.

Speech-bubble math (face center / tail tip / bubble center are collinear) is ported
from comic/pipeline.py. Corner selection defaults to TOP bubble (body above face,
tail pointing down); switches to BOTTOM only when the top bubble would clip the panel.

Usage:
  modal run sd_generate.py                                        # all chars × outfits
  modal run sd_generate.py --char hanyoil --outfit business
  modal run sd_generate.py --char hanyoil --outfit business --bubble "오늘 회의 다 끝났네"

One-time setup (Modal volume must contain the model files):
  modal volume create comic-models
  modal volume put comic-models <local>/WAI-illustrious-SDXL_17.safetensors /WAI-illustrious-SDXL_17.safetensors
  modal volume put comic-models <local>/lora /lora
"""
from __future__ import annotations

import re
from pathlib import Path

import modal

HERE = Path(__file__).parent

# ─────────────────────────────────────────────────────────
# Modal image / app
# ─────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "diffusers==0.31.0",
        "transformers==4.46.3",        # diffusers 0.31 호환 (5.x 는 CLIPImageProcessor import 깨짐)
        "tokenizers==0.20.3",
        "huggingface_hub==0.26.2",     # 1.x 는 diffusers/transformers 와 비호환
        "accelerate==1.1.1",
        "safetensors",
        "peft==0.13.2",
        "compel==2.0.2",               # 2.4.x 는 empty_z 제거 → pad_conditioning 깨짐
        "xformers==0.0.27.post2",
        "ultralytics",
        "onnxruntime",
        "numpy<2",                     # numpy 2.x 회피 (검증된 1.26 라인)
        "opencv-python-headless",
        "pillow",
        "pyyaml",
    )
    .add_local_file(str(HERE / "prompts.py"), "/root/prompts.py")
    .add_local_file(str(HERE / "config_loader.py"), "/root/config_loader.py")
    .add_local_dir(str(HERE / "data"), "/root/data")
    .add_local_dir(str(HERE / "textbubble"), "/root/textbubble")
    .add_local_dir(str(HERE / "Font"), "/root/Font")
)

app = modal.App("sd-generate", image=image)
models_vol = modal.Volume.from_name("comic-models", create_if_missing=True)
output_vol = modal.Volume.from_name("comic-output", create_if_missing=True)

GPU = "A10G"
BG_IP_ADAPTER_REPO = "h94/IP-Adapter"
BG_IP_ADAPTER_SUBFOLDER = "sdxl_models"
BG_IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"


def _trim_tags(tags: str, limit: int = 3) -> str:
    """Keep scene tags sparse so backgrounds do not overpower character LoRAs."""
    parts = [p.strip() for p in str(tags or "").split(",") if p.strip()]
    return ", ".join(parts[:limit])


def _bg_key(value: str | None) -> str:
    return "_".join(str(value or "").strip().lower().replace("/", " ").split())


def _background_bundle(P, pcfg: dict) -> str:
    """SHORT SDXL 배경 = "장소, 소품 1개".

    배경은 로컬(generate_scenario)에서 locations.yaml 로 이미 해석돼 panel["background"]
    에 박혀 오므로(일원화), 여기서는 그 값을 그대로 쓴다(최대 2태그).
    """
    return _trim_tags(pcfg.get("background", ""), limit=2) or "simple background"


OBJECT_PANEL_SIZE = 1024


def _panel_render_size(P, is_object_panel: bool) -> tuple[int, int]:
    if is_object_panel:
        return OBJECT_PANEL_SIZE, OBJECT_PANEL_SIZE
    return P.WIDTH, P.HEIGHT


_FRAMING_TAGS = {
    "full_body": "full body",
    "waist_shot": "cowboy shot, knees up",
    "upper_body": "upper body, portrait",
    "close_up": "close-up, portrait, face focus, headshot",
    "object_close_up": "close-up, still life",
}

_FRAMING_NEGATIVE_TAGS = {
    "waist_shot": "feet",
    "upper_body": "full body, feet",
    "close_up": "full body, lower body, legs, feet",
    "object_close_up": "person, girl, face, body, hands",
}


def _framing_tags(pcfg: dict, is_object_panel: bool) -> str:
    key = str(pcfg.get("framing") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if is_object_panel:
        return _FRAMING_TAGS["object_close_up"]
    return _FRAMING_TAGS.get(key, "upper body")


def _framing_negative_tags(pcfg: dict, is_object_panel: bool) -> str:
    key = str(pcfg.get("framing") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if is_object_panel:
        return _FRAMING_NEGATIVE_TAGS["object_close_up"]
    return _FRAMING_NEGATIVE_TAGS.get(key, "")


_OBJECT_COLOR_HINTS = [
    (("phone", "smartphone", "cellphone"), "black smartphone, incoming call screen, white table"),
    (("menu",), "white menu card, white table"),
    (("receipt",), "white receipt, white table"),
    (("coffee", "cup"), "white coffee cup, brown table"),
    (("notebook", "notes"), "white paper, brown desk"),
]

_OBJECT_NEGATIVE_HINTS = [
    (("phone", "smartphone", "cellphone"), "laptop, keyboard, computer, monitor, tablet"),
]


def _object_subject_for_prompt(subject: str, action: str = "") -> str:
    """Strengthen object panels with simple color/material anchors.

    Illustrious often turns generic "phone on table" into any office device.
    A tiny color + surface hint keeps the object legible without overloading the
    scene prompt.
    """
    subject = (subject or "").strip()
    action = (action or "").strip()
    text = f"{subject}, {action}".lower()
    extras = []
    for keys, hint in _OBJECT_COLOR_HINTS:
        if any(key in text for key in keys):
            extras.extend(part.strip() for part in hint.split(",") if part.strip())
    merged = []
    for part in [subject, *extras]:
        if part and part.lower() not in {x.lower() for x in merged}:
            merged.append(part)
    return ", ".join(merged) or action or "still life objects"


def _object_negative_for_prompt(subject: str, action: str = "") -> str:
    text = f"{subject or ''}, {action or ''}".lower()
    extras = []
    for keys, hint in _OBJECT_NEGATIVE_HINTS:
        if any(key in text for key in keys):
            extras.append(hint)
    return ", ".join(extras)


def _scene_action_tags(pcfg: dict) -> str:
    parts = []
    for raw in (pcfg.get("action", ""), pcfg.get("prop_interaction", "")):
        for tag in str(raw or "").split(","):
            tag = tag.strip()
            if tag and tag.lower() not in {x.lower() for x in parts}:
                parts.append(tag)
    return ", ".join(parts)


def _object_narration_text(pcfg: dict) -> str:
    subject = str(pcfg.get("subject") or "").strip()
    action = str(pcfg.get("action") or "").strip()
    text = f"{subject}, {action}".lower()
    if any(word in text for word in ("phone", "smartphone", "cellphone")) and "ring" in text:
        return "A phone rings."
    if "upside-down" in text and "menu" in text:
        return "The menu is upside down."
    if subject:
        return subject.replace(",", " - ")
    if action:
        return action.replace(",", " - ")
    return ""


def _first_background_cfg(panels_cfg: list[dict]) -> dict:
    """Pick the first panel with set/room metadata as the episode background source."""
    for pcfg in panels_cfg:
        if pcfg.get("background_set") or pcfg.get("set"):
            return pcfg
    return panels_cfg[0] if panels_cfg else {}


def _mask_box_for_panel(P, pcfg: dict, idx: int) -> tuple[int, int, int, int]:
    """MVP placement mask for background-plate inpainting.

    The scenario writer can pass ``character_slot``/``slot`` later. Until then,
    alternate left/right for dialogue panels and use center for object panels.
    """
    W, H = P.WIDTH, P.HEIGHT
    panel_type = (pcfg.get("panel_type") or "").strip().lower()
    if panel_type in {"object", "setting"} or not pcfg.get("char"):
        return (int(W * 0.18), int(H * 0.48), int(W * 0.82), int(H * 0.88))

    slot = (pcfg.get("character_slot") or pcfg.get("slot") or "").strip().lower()
    if not slot:
        slot = "left" if idx % 2 == 0 else "right"
    boxes = {
        "left":   (int(W * 0.04), int(H * 0.18), int(W * 0.58), int(H * 0.96)),
        "right":  (int(W * 0.42), int(H * 0.18), int(W * 0.96), int(H * 0.96)),
        "center": (int(W * 0.22), int(H * 0.16), int(W * 0.78), int(H * 0.96)),
        "full":   (int(W * 0.06), int(H * 0.12), int(W * 0.94), int(H * 0.98)),
    }
    return boxes.get(slot, boxes["center"])


def _make_plate_mask(P, pcfg: dict, idx: int):
    """Return a soft-edged mask for character/object insertion on a fixed plate."""
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", (P.WIDTH, P.HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(_mask_box_for_panel(P, pcfg, idx), radius=48, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(18))


# ─────────────────────────────────────────────────────────
# Speech-bubble geometry (ported from comic/pipeline.py)
#
# Native PNG canvas is 500×500. tail / center / text coords are in that frame.
# Key naming = where the TAIL is on the canvas:
#   TL/TR: tail at top  → bubble body sits BELOW face (use for bottom-of-panel)
#   BL/BR: tail at bot  → bubble body sits ABOVE face (use for top-of-panel)  ← default
# File names are reversed (named for where bubble body sits, not the tail).
# ─────────────────────────────────────────────────────────
BUBBLE_NATIVE = (500, 500)
BUBBLE_GEOMETRY = {
    "TL": {"tail": (109,  83), "center": (252, 259), "text": (286, 210), "file": "bottom_left"},
    "TR": {"tail": (390,  83), "center": (247, 259), "text": (286, 210), "file": "bottom_right"},
    "BL": {"tail": (109, 416), "center": (252, 240), "text": (286, 210), "file": "top_left"},
    "BR": {"tail": (390, 416), "center": (247, 240), "text": (286, 210), "file": "top_right"},
}
THOUGHT_BUBBLE_GEOMETRY = {
    # Coordinates are normalized to the same 500x500 canvas used by speech bubbles.
    # "tail" is the center of the small thought circle closest to the speaker.
    "TL": {"tail": (166,  78), "center": (251, 319), "text": (320, 170), "file": "thought_bubble_top_left"},
    "TR": {"tail": (334,  78), "center": (249, 319), "text": (320, 170), "file": "thought_bubble_top_right"},
    "BL": {"tail": (166, 422), "center": (251, 196), "text": (320, 170), "file": "thought_bubble_bottom_left"},
    "BR": {"tail": (334, 422), "center": (249, 196), "text": (320, 170), "file": "thought_bubble_bottom_right"},
}


# ─────────────────────────────────────────────────────────
# Bubble helpers (CPU only — no torch dependency)
# ─────────────────────────────────────────────────────────
_BUBBLE_CACHE: dict = {}
_THOUGHT_BUBBLE_CACHE: dict = {}


def _load_bubble_assets(bubble_dir: str) -> dict:
    """Load 4 bubble PNGs into RGBA at native 500×500, keyed by corner (TL/TR/BL/BR)."""
    if _BUBBLE_CACHE:
        return _BUBBLE_CACHE
    from PIL import Image
    bdir = Path(bubble_dir)
    if not bdir.exists():
        raise FileNotFoundError(f"말풍선 폴더 없음: {bdir}")

    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())

    for corner, g in BUBBLE_GEOMETRY.items():
        target = _norm(g["file"])
        png = next((p for p in bdir.glob("*.png") if target in _norm(p.stem)), None)
        if png is None:
            continue
        img = Image.open(png).convert("RGBA")
        if img.size != BUBBLE_NATIVE:
            img = img.resize(BUBBLE_NATIVE, Image.LANCZOS)
        _BUBBLE_CACHE[corner] = {
            "img": img,
            "tip": g["tail"],
            "center": g["center"],
            "text_size": g["text"],
        }
    if not _BUBBLE_CACHE:
        raise RuntimeError("BUBBLE_GEOMETRY 에 등록된 PNG 가 textbubble/ 안에 없음")
    return _BUBBLE_CACHE


def _load_thought_bubble_assets(bubble_dir: str) -> dict:
    """Load thought bubble PNGs into RGBA at native 500x500, keyed by corner."""
    if _THOUGHT_BUBBLE_CACHE:
        return _THOUGHT_BUBBLE_CACHE
    from PIL import Image
    bdir = Path(bubble_dir)
    if not bdir.exists():
        raise FileNotFoundError(f"말풍선 폴더 없음: {bdir}")

    def _norm(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())

    for corner, g in THOUGHT_BUBBLE_GEOMETRY.items():
        target = _norm(g["file"])
        png = next((p for p in bdir.glob("*.png") if target == _norm(p.stem)), None)
        if png is None:
            continue
        img = Image.open(png).convert("RGBA")
        if img.size != BUBBLE_NATIVE:
            img = img.resize(BUBBLE_NATIVE, Image.LANCZOS)
        _THOUGHT_BUBBLE_CACHE[corner] = {
            "img": img,
            "tip": g["tail"],
            "center": g["center"],
            "text_size": g["text"],
        }
    if not _THOUGHT_BUBBLE_CACHE:
        raise RuntimeError("THOUGHT_BUBBLE_GEOMETRY 에 등록된 PNG 가 textbubble/ 안에 없음")
    return _THOUGHT_BUBBLE_CACHE


def _wrap_text(text: str, font, max_w: int, draw) -> list[str]:
    """공백 있으면 단어 단위, 없으면 글자 단위 줄바꿈."""
    has_space = " " in text
    tokens = text.split() if has_space else list(text)
    sep = " " if has_space else ""
    lines, cur = [], ""
    for tok in tokens:
        test = (cur + sep + tok).strip() if cur else tok
        if draw.textlength(test, font=font) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return lines


def _norm_sentence(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[.!?]+", "", text)
    return " ".join(text.split())


def _is_target_sentence(text: str, target_sentence: str = "") -> bool:
    return bool(target_sentence and _norm_sentence(text) == _norm_sentence(target_sentence))


def _italic_font_path(regular_path: str) -> str:
    """Return a real italic font path for target-sentence emphasis."""
    candidates = [
        str(Path(regular_path).with_name(Path(regular_path).stem + "-Italic" + Path(regular_path).suffix)),
        str(Path(regular_path).with_name(Path(regular_path).stem + "Italic" + Path(regular_path).suffix)),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    return regular_path


def _load_italic_font(regular_font, regular_path: str):
    from PIL import ImageFont

    return ImageFont.truetype(_italic_font_path(regular_path), regular_font.size)


def _target_spans(text: str, target_sentence: str = "") -> list[tuple[int, int]]:
    """Find exact target phrase spans while allowing final punctuation differences."""
    text = text or ""
    target = (target_sentence or "").strip()
    if not text or not target:
        return []
    hay = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    candidates = []
    for cand in (target, target.rstrip(".!?")):
        cand = cand.strip()
        if cand and cand.lower() not in {c.lower() for c in candidates}:
            candidates.append(cand)
    spans = []
    for cand in candidates:
        m = re.search(re.escape(cand), hay, flags=re.IGNORECASE)
        if m:
            end = m.end()
            while end < len(text) and text[end] in ".!?":
                end += 1
            spans.append((m.start(), end))
            break
    return spans


def _split_runs_by_target(text: str, target_sentence: str = "") -> list[tuple[str, bool]]:
    spans = _target_spans(text, target_sentence)
    if not spans:
        return [(text, False)]
    runs: list[tuple[str, bool]] = []
    cur = 0
    for start, end in spans:
        if start > cur:
            runs.append((text[cur:start], False))
        runs.append((text[start:end], True))
        cur = end
    if cur < len(text):
        runs.append((text[cur:], False))
    return [(s, italic) for s, italic in runs if s]


def _line_width(runs: list[tuple[str, bool]], font, italic_font, draw) -> float:
    return sum(draw.textlength(text, font=italic_font if italic else font) for text, italic in runs)


def _append_wrapped_token(lines, cur, token_runs, font, italic_font, max_w, draw):
    """Append one whitespace/word token to wrapped rich-text lines."""
    test = cur + token_runs
    if not cur or _line_width(test, font, italic_font, draw) <= max_w:
        return lines, test
    lines.append(cur)
    return lines, token_runs


def _wrap_rich_text(text: str, font, italic_font, max_w: int, draw,
                    target_sentence: str = "") -> list[list[tuple[str, bool]]]:
    runs = _split_runs_by_target(text, target_sentence)
    token_runs: list[list[tuple[str, bool]]] = []
    for run_text, italic in runs:
        for part in re.findall(r"\S+\s*|\s+", run_text):
            token_runs.append([(part, italic)])

    lines: list[list[tuple[str, bool]]] = []
    cur: list[tuple[str, bool]] = []
    for token in token_runs:
        lines, cur = _append_wrapped_token(lines, cur, token, font, italic_font, max_w, draw)
        if _line_width(cur, font, italic_font, draw) <= max_w:
            continue
        # Extremely long no-space token fallback: split by character.
        overflow = cur
        cur = []
        for text_part, italic in overflow:
            for ch in text_part:
                lines, cur = _append_wrapped_token(lines, cur, [(ch, italic)], font, italic_font, max_w, draw)
    if cur:
        lines.append(cur)
    return lines or [[("", False)]]


def _draw_rich_text_line(canvas, xy: tuple[int, int], runs: list[tuple[str, bool]],
                         font, italic_font, fill) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    x, y = xy
    for text, italic in runs:
        active_font = italic_font if italic else font
        draw.text((x, y), text, font=active_font, fill=fill)
        x += draw.textlength(text, font=active_font)


def _fit_bubble(text: str, asset: dict, font, draw, panel_w: int, panel_h: int,
                width_ratio: float, line_h_factor: float,
                max_width_ratio: float | None = None,
                italic_font=None, target_sentence: str = ""):
    """텍스트가 들어가는 최소 스케일 찾기. (scale, lines, line_h) 반환.

    min/max 분리: 평소(짧은 대사)는 ``width_ratio`` 기준 크기로 떨어지고, 대사가 길면
    ``max_width_ratio`` 상한까지 풍선이 '확' 커진다. max_width_ratio 가 None 이면 기존처럼
    width_ratio 가 곧 상한(=비율 고정).
    """
    nat_w, nat_h = BUBBLE_NATIVE
    tb_w_nat, tb_h_nat = asset["text_size"]
    short = min(panel_w, panel_h)
    nominal_scale = (short * width_ratio) / max(nat_w, nat_h)
    ceil_ratio = max_width_ratio if max_width_ratio is not None else width_ratio
    max_scale = (short * ceil_ratio) / max(nat_w, nat_h)
    # 하한은 평소 비율 기준 — 상한을 키워도 짧은 대사 크기는 그대로 유지된다.
    min_scale = nominal_scale * 0.55
    scale = min_scale
    lines: list[str] = []
    line_h = int(font.size * line_h_factor)
    italic_font = italic_font or font
    for _ in range(12):
        tbw = max(10, int(tb_w_nat * scale))
        tbh = max(10, int(tb_h_nat * scale))
        lines = _wrap_rich_text(text, font, italic_font, tbw, draw, target_sentence)
        req_h = line_h * len(lines)
        req_w = max((_line_width(ln, font, italic_font, draw) for ln in lines), default=0)
        if req_h <= tbh and req_w <= tbw:
            break
        sh = req_h / tbh if req_h > tbh else 1.0
        sw = req_w / tbw if req_w > tbw else 1.0
        scale = scale * max(sh, sw) * 1.04
        if scale >= max_scale:
            scale = max_scale
            tbw = max(10, int(tb_w_nat * scale))
            lines = _wrap_rich_text(text, font, italic_font, tbw, draw, target_sentence)
            break
    return min(max(scale, min_scale), max_scale), lines, line_h


def _compute_paste_xy(face_cx: int, face_cy: int, asset: dict, scale: float,
                      face_radius: float) -> tuple[int, int, float, float]:
    """face center · tail tip · bubble center 가 한 직선 위가 되도록 paste 좌표 계산.

    Returns (px, py, tip_xs, tip_ys) — tip_xs/ys 는 스케일 적용된 풍선 내부 tail 좌표.
    """
    body_cx, body_cy = asset["center"]
    tip_x, tip_y = asset["tip"]
    # 풍선 중심 → tail 단위 벡터 (방향만 — 스케일 무관)
    dvx, dvy = tip_x - body_cx, tip_y - body_cy
    dvlen = (dvx * dvx + dvy * dvy) ** 0.5 or 1.0
    ux, uy = dvx / dvlen, dvy / dvlen
    tip_xs, tip_ys = tip_x * scale, tip_y * scale
    # tail tip 은 face 경계 바깥 8px, face→bubble center 방향의 반대 (즉 face 쪽)
    tail_gap = face_radius + 8
    tail_cx = face_cx - ux * tail_gap
    tail_cy = face_cy - uy * tail_gap
    # paste 좌표: tail tip 이 (tail_cx, tail_cy) 에 놓이도록
    px = int(round(tail_cx - tip_xs))
    py = int(round(tail_cy - tip_ys))
    return px, py, tip_xs, tip_ys


def _rotate_vec(x: float, y: float, degrees: float) -> tuple[float, float]:
    import math

    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return x * c - y * s, x * s + y * c


def _compute_thought_paste_xy(face_cx: float, face_cy: float, asset: dict, scale: float,
                              face_radius: float, prefer_down: bool = True
                              ) -> tuple[int, int, float, float]:
    """Place thought bubble so bubble-center/tail-center/face makes a soft bend.

    The angle between tail->bubble_center and tail->face_center is about 120deg.
    This keeps thought bubbles from looking like a speech-tail line while still
    aiming the small circles toward the speaker.
    """
    body_cx, body_cy = asset["center"]
    tip_x, tip_y = asset["tip"]
    ux, uy = body_cx - tip_x, body_cy - tip_y
    length = (ux * ux + uy * uy) ** 0.5 or 1.0
    ux, uy = ux / length, uy / length

    if prefer_down and tip_x > body_cx:
        rotation = 120
    else:
        rotation = THOUGHT_BUBBLE_ROTATION_DEG if prefer_down else -THOUGHT_BUBBLE_ROTATION_DEG
    vx, vy = _rotate_vec(ux, uy, rotation)

    tail_gap = face_radius + 12
    tail_cx = face_cx - vx * tail_gap
    tail_cy = face_cy - vy * tail_gap
    tip_xs, tip_ys = tip_x * scale, tip_y * scale
    px = int(round(tail_cx - tip_xs))
    py = int(round(tail_cy - tip_ys + THOUGHT_BUBBLE_DOWN_NUDGE))
    return px, py, tip_xs, tip_ys


def _is_internal_thought_text(text: str) -> bool:
    return bool(re.match(r"^\s*\((?:internally|internal|thought|thinking|속으로)\)\s*", text or "", re.I))


def _strip_internal_thought_marker(text: str) -> str:
    return re.sub(r"^\s*\((?:internally|internal|thought|thinking|속으로)\)\s*", "", text or "", flags=re.I).strip()


def _is_narration_text(text: str) -> bool:
    return bool(re.match(r"^\s*(?:\((?:narration|caption|timecard)\)|\[(?:narration|caption|timecard)\])\s*", text or "", re.I))


def _strip_narration_marker(text: str) -> str:
    return re.sub(
        r"^\s*(?:\((?:narration|caption|timecard)\)|\[(?:narration|caption|timecard)\])\s*",
        "",
        text or "",
        flags=re.I,
    ).strip()


def _would_clip(px: int, py: int, bw: int, bh: int, W: int, H: int, M: int) -> bool:
    return px < M or py < M or px + bw > W - M or py + bh > H - M


def _visible_text_box(px: int, py: int, asset: dict, scale: float,
                      safe_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Return the text box clipped to the actually visible/safe area."""
    body_cx, body_cy = asset["center"]
    tb_w_nat, tb_h_nat = asset["text_size"]
    tb_l = px + int((body_cx - tb_w_nat // 2) * scale)
    tb_t = py + int((body_cy - tb_h_nat // 2) * scale)
    tb_r = tb_l + int(tb_w_nat * scale)
    tb_b = tb_t + int(tb_h_nat * scale)
    safe_l, safe_t, safe_r, safe_b = safe_box
    vis_l = max(tb_l, safe_l)
    vis_t = max(tb_t, safe_t)
    vis_r = min(tb_r, safe_r)
    vis_b = min(tb_b, safe_b)
    if vis_r - vis_l < 24 or vis_b - vis_t < 24:
        return tb_l, tb_t, tb_r, tb_b
    return vis_l, vis_t, vis_r, vis_b


def _pick_corner_keys(face_cx: int, panel_w: int) -> tuple[str, str]:
    """Default TOP (body 위, tail 아래). 클리핑되면 BOTTOM 으로 fallback.
    face 가 좌/우 어느 쪽인지로 left/right 코너만 결정."""
    is_left = face_cx < panel_w // 2
    top_key = "TL" if is_left else "TR"     # tail at bottom → body extends upward
    bottom_key = "BL" if is_left else "BR"  # tail at top    → body extends downward
    return top_key, bottom_key 


def add_speech_bubble(image, text: str, face_bbox: tuple | None = None,
                      yolo=None):
    """Paste speech-bubble PNG + Korean text on image.

    Args:
        image: PIL.Image
        text: 한국어 대사
        face_bbox: (x1,y1,x2,y2). None 이면 yolo 로 감지, 없으면 이미지 상단 중앙 가정.
        yolo: ultralytics YOLO (face_yolov8s.pt). None 이면 face_bbox 필수.

    Corner selection:
      1. Default → TOP bubble (body above face, tail pointing down)
      2. Fallback → BOTTOM bubble (only when TOP would clip the panel)
    """
    import sys
    sys.path.insert(0, "/root")
    import prompts as P  # noqa: E402
    from PIL import Image, ImageDraw, ImageFont

    canvas = image.convert("RGBA").copy()
    W, H = canvas.size
    M = P.BUBBLE_MARGIN

    # 1) Face bbox 결정
    if face_bbox is None and yolo is not None:
        import numpy as np
        res = yolo(np.array(canvas.convert("RGB")), conf=0.3, verbose=False)
        boxes = [b.xyxy[0].cpu().numpy().astype(int) for r in res for b in r.boxes]
        face_bbox = tuple(boxes[0]) if boxes else None
    if face_bbox is None:
        # 합리적 fallback — 상단 중앙 가짜 face
        r = min(W, H) // 10
        face_bbox = (W // 2 - r, H // 3 - r, W // 2 + r, H // 3 + r)

    fx1, fy1, fx2, fy2 = face_bbox
    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
    face_radius = max(fx2 - fx1, fy2 - fy1) / 2

    # 2) 코너 후보 (top-first / bottom-fallback)
    top_key, bot_key = _pick_corner_keys(fcx, W)
    assets = _load_bubble_assets(P.BUBBLE_DIR)
    font = ImageFont.truetype(P.BUBBLE_FONT, P.BUBBLE_FONT_SIZE)
    draw = ImageDraw.Draw(canvas)

    # 3) 각 코너에 대해 fit + 클립 검사 — top 먼저, 짤리면 bottom
    chosen = None
    for key in (top_key, bot_key):
        if key not in assets:
            continue
        asset = assets[key]
        scale, lines, line_h = _fit_bubble(
            text, asset, font, draw, W, H, P.BUBBLE_WIDTH_RATIO, P.BUBBLE_LINE_H,
        )
        nat_w, nat_h = BUBBLE_NATIVE
        bw, bh = max(8, int(nat_w * scale)), max(8, int(nat_h * scale))
        px, py, tip_xs, tip_ys = _compute_paste_xy(fcx, fcy, asset, scale, face_radius)
        if not _would_clip(px, py, bw, bh, W, H, M):
            chosen = (key, asset, scale, lines, line_h, bw, bh, px, py, tip_xs, tip_ys)
            break
    if chosen is None:
        # 둘 다 클립 — top 으로 두고 클램프 (라인얼라인 깨지지만 가시성 우선)
        key = top_key if top_key in assets else bot_key
        asset = assets[key]
        scale, lines, line_h = _fit_bubble(
            text, asset, font, draw, W, H, P.BUBBLE_WIDTH_RATIO, P.BUBBLE_LINE_H,
        )
        nat_w, nat_h = BUBBLE_NATIVE
        bw, bh = max(8, int(nat_w * scale)), max(8, int(nat_h * scale))
        px, py, tip_xs, tip_ys = _compute_paste_xy(fcx, fcy, asset, scale, face_radius)
        px = max(M, min(W - bw - M, px))
        py = max(M, min(H - bh - M, py))
        chosen = (key, asset, scale, lines, line_h, bw, bh, px, py, tip_xs, tip_ys)

    key, asset, scale, lines, line_h, bw, bh, px, py, _, _ = chosen
    print(f"  💬 bubble corner={key} (top_first={top_key}, fallback={bot_key})")

    # 4) 풍선 paste
    bubble = asset["img"].resize((bw, bh), Image.LANCZOS)
    canvas.alpha_composite(bubble, (px, py))

    # 5) 텍스트 영역은 꼬리 정렬용 박스가 아니라 실제 안전영역 기준으로 다시 잡는다.
    tb_l, tb_t, tb_r, tb_b = _visible_text_box(px, py, asset, scale, (M, M, W - M, H - M))
    tb_w = tb_r - tb_l
    tb_h = tb_b - tb_t
    lines = _wrap_text(text, font, max(10, tb_w), draw)
    while line_h * len(lines) > tb_h and font.size > P.BUBBLE_MIN_FONT:
        font = ImageFont.truetype(P.BUBBLE_FONT, max(P.BUBBLE_MIN_FONT, font.size - 2))
        line_h = int(font.size * P.BUBBLE_LINE_H)
        lines = _wrap_text(text, font, max(10, tb_w), draw)
    total_h = line_h * len(lines)
    text_y = tb_t + max(0, (tb_h - total_h) // 2)
    for ln in lines:
        lw = draw.textlength(ln, font=font)
        tx = tb_l + max(0, (tb_w - int(lw)) // 2)
        draw.text((tx, text_y), ln, font=font, fill=P.BUBBLE_TEXT_COLOR)
        text_y += line_h

    return canvas.convert("RGB")


# ─────────────────────────────────────────────────────────
# Webtoon framing
#
# Scene is drawn into a webtoon canvas. Academic panels use black gutters/rails;
# other learning contexts use white gutters/rails. The gutters/rails live outside
# the generated image so they do not leak into the model's scene prompt.
# The speech bubble floats in the top gutter with its tail pointing DOWN at the
# speaker (tail-at-bottom asset = BL/BR). face center · tail tip · bubble center
# stay collinear along the bubble's own tail axis; raising `lift` slides the
# bubble UP that same line so the three points never leave it.
# ─────────────────────────────────────────────────────────
WEBTOON_MARGIN_TOP_RATIO    = 0.20   # black gutter above the scene (holds bubble)
WEBTOON_MARGIN_BOTTOM_RATIO = 0.05   # black gutter below the scene
WEBTOON_LIFT_RATIO          = 0.02   # how far up the line the bubble floats (×scene h)
WEBTOON_BORDER_W            = 6      # black border / side-rail thickness
WEBTOON_BUBBLE_WIDTH_RATIO  = 0.50   # 평소(짧은 대사) 풍선 폭 — 비율 유지
WEBTOON_BUBBLE_MAX_RATIO    = 0.72   # 대사가 길면 여기까지 '확' 커진다(상한)
WEBTOON_GUTTER_BLACK        = (0, 0, 0)
WEBTOON_GUTTER_WHITE        = (255, 255, 255)
THOUGHT_BUBBLE_ROTATION_DEG = 240
THOUGHT_BUBBLE_DOWN_NUDGE   = 14
NARRATION_BOX_WIDTH_RATIO   = 0.62
NARRATION_BOX_MAX_RATIO     = 0.82
NARRATION_BOX_PAD_X         = 34
NARRATION_BOX_PAD_Y         = 18
NARRATION_BOX_BORDER_W      = 4
NARRATION_BOX_SCENE_OVERLAP = 12
NARRATION_BOX_SCENE_INSET   = 18
NARRATION_BOX_SCENE_TOP_INSET = 42


def _webtoon_gutter_color(pcfg: dict | None = None):
    used_in = ((pcfg or {}).get("used_in") or (pcfg or {}).get("used in") or "").strip().lower()
    return WEBTOON_GUTTER_BLACK if used_in == "academic" else WEBTOON_GUTTER_WHITE


def compose_webtoon_panel(image, text: str = "", face_bbox: tuple | None = None,
                          yolo=None, gutter_color=None, target_sentence: str = ""):
    """Frame a clean scene on webtoon gutters and float a tail-down bubble.

    Args:
        image: PIL.Image — the clean rendered scene (no bubble baked in).
        text: 한국어 대사. 빈 문자열이면 풍선 없이 액자만 씌운다(오브젝트 패널용).
        face_bbox: (x1,y1,x2,y2) in SCENE coords. None 이면 yolo 로 감지, 없으면 상단 중앙.
        yolo: ultralytics YOLO. face_bbox 가 None 일 때만 사용.

    Returns an RGB canvas LARGER than the scene. All panels share the scene width,
    so stacking them (see ``stack_webtoon_strip``) yields a continuous strip.
    """
    import sys
    sys.path.insert(0, "/root")
    import prompts as P  # noqa: E402
    from PIL import Image, ImageDraw, ImageFont

    scene = image.convert("RGB")
    sw, sh = scene.size
    m_top = int(sh * WEBTOON_MARGIN_TOP_RATIO)
    m_bot = int(sh * WEBTOON_MARGIN_BOTTOM_RATIO)
    bd = WEBTOON_BORDER_W

    # ── 풍선을 먼저 계산한다(없으면 액자만). 큰 풍선이 위로 넘쳐도 상단 여백을 그만큼
    #    늘려 캔버스에서 잘리지 않게 하기 위함. 좌표는 일단 SCENE 기준(py 가 음수면 씬
    #    위로 삐져나간다는 뜻)으로 잡고, 나중에 oy 만큼 더해 캔버스 좌표로 변환한다. ──
    bubble = None
    narration_box = None
    bubble_kind = "speech"
    display_text = text
    if text:
        if _is_narration_text(text):
            bubble_kind = "narration"
            display_text = _strip_narration_marker(text)
        else:
            bubble_kind = "thought" if _is_internal_thought_text(text) else "speech"
            display_text = _strip_internal_thought_marker(text) if bubble_kind == "thought" else text
    if text and bubble_kind == "narration":
        font = ImageFont.truetype(P.BUBBLE_FONT, P.BUBBLE_FONT_SIZE)
        italic_font = _load_italic_font(font, P.BUBBLE_FONT)
        tmp_draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        max_box_w = int(sw * NARRATION_BOX_MAX_RATIO)
        text_w = max(80, int(sw * NARRATION_BOX_WIDTH_RATIO))
        lines = _wrap_rich_text(display_text, font, italic_font, text_w, tmp_draw, "")
        line_h = int(font.size * P.BUBBLE_LINE_H)
        req_w = max((_line_width(ln, font, italic_font, tmp_draw) for ln in lines), default=0)
        while req_w > text_w and text_w < max_box_w:
            text_w = min(max_box_w, int(text_w * 1.12))
            lines = _wrap_rich_text(display_text, font, italic_font, text_w, tmp_draw, "")
            req_w = max((_line_width(ln, font, italic_font, tmp_draw) for ln in lines), default=0)
        box_w = min(max_box_w, max(int(req_w) + NARRATION_BOX_PAD_X * 2, int(sw * 0.26)))
        box_h = line_h * len(lines) + NARRATION_BOX_PAD_Y * 2
        box_x = bd + NARRATION_BOX_SCENE_INSET
        box_y = m_top + NARRATION_BOX_SCENE_TOP_INSET
        narration_box = (box_x, box_y, box_w, box_h, lines, line_h, font, italic_font)
    if text and bubble_kind != "narration":
        if face_bbox is None and yolo is not None:
            import numpy as np
            res = yolo(np.array(scene), conf=0.3, verbose=False)
            boxes = [b.xyxy[0].cpu().numpy().astype(int) for r in res for b in r.boxes]
            face_bbox = tuple(boxes[0]) if boxes else None
        if face_bbox is None:
            r = min(sw, sh) // 10
            face_bbox = (sw // 2 - r, sh // 3 - r, sw // 2 + r, sh // 3 + r)

        fx1, fy1, fx2, fy2 = face_bbox
        fcx_s = (fx1 + fx2) / 2
        fcy_s = (fy1 + fy2) / 2
        face_radius = max(fx2 - fx1, fy2 - fy1) / 2

        # 꼬리가 아래로 향하는 에셋(BL/BR) — 풍선 몸통이 얼굴 위에 뜬다.
        is_left = fcx_s < sw / 2
        key = "BL" if is_left else "BR"
        assets = _load_thought_bubble_assets(P.BUBBLE_DIR) if bubble_kind == "thought" else _load_bubble_assets(P.BUBBLE_DIR)
        if key not in assets:
            key = next(iter(assets))
        asset = assets[key]
        font = ImageFont.truetype(P.BUBBLE_FONT, P.BUBBLE_FONT_SIZE)
        italic_font = _load_italic_font(font, P.BUBBLE_FONT)
        tmp_draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))

        # 평소엔 WIDTH_RATIO 크기, 대사가 길면 MAX_RATIO 까지 확 커진다.
        scale, lines, line_h = _fit_bubble(
            display_text, asset, font, tmp_draw, sw, sh,
            WEBTOON_BUBBLE_WIDTH_RATIO, P.BUBBLE_LINE_H,
            max_width_ratio=WEBTOON_BUBBLE_MAX_RATIO,
            italic_font=italic_font,
            target_sentence=target_sentence,
        )
        nat_w, nat_h = BUBBLE_NATIVE
        b_w, b_h = max(8, int(nat_w * scale)), max(8, int(nat_h * scale))

        # 풍선을 상한까지 키워도 글이 안 들어가는 극단적 장문 → 폰트만 살짝 줄이는
        # 마지막 안전장치(BUBBLE_MIN_FONT 까지). 보통/중간 대사는 절대 줄지 않는다.
        tb_w0 = int(asset["text_size"][0] * scale)
        tb_h0 = int(asset["text_size"][1] * scale)
        fsz = P.BUBBLE_FONT_SIZE
        while line_h * len(lines) > tb_h0 and fsz > P.BUBBLE_MIN_FONT:
            fsz -= 2
            font = ImageFont.truetype(P.BUBBLE_FONT, fsz)
            italic_font = _load_italic_font(font, P.BUBBLE_FONT)
            line_h = int(fsz * P.BUBBLE_LINE_H)
            lines = _wrap_rich_text(display_text, font, italic_font, tb_w0, tmp_draw, target_sentence)
        if fsz != P.BUBBLE_FONT_SIZE:
            print(f"  💬 long line — font shrunk {P.BUBBLE_FONT_SIZE}→{fsz}")

        # 콜리니어 배치: 방향 = 풍선중심 → 꼬리끝(에셋 고유 꼬리축).
        body_cx, body_cy = asset["center"]
        tip_x, tip_y = asset["tip"]
        tip_xs, tip_ys = tip_x * scale, tip_y * scale
        dvx, dvy = tip_x - body_cx, tip_y - body_cy
        dvlen = (dvx * dvx + dvy * dvy) ** 0.5 or 1.0
        ux, uy = dvx / dvlen, dvy / dvlen
        gap = face_radius + 8 + sh * WEBTOON_LIFT_RATIO
        px = int(round(fcx_s - ux * gap - tip_xs))   # scene-coord top-left
        py = int(round(fcy_s - uy * gap - tip_ys))   # py<0 → 씬 위로 넘침
        if bubble_kind == "thought":
            px, py, _, _ = _compute_thought_paste_xy(
                fcx_s, fcy_s, asset, scale,
                face_radius + sh * WEBTOON_LIFT_RATIO,
                prefer_down=True,
            )
        bubble = asset["img"].resize((b_w, b_h), Image.LANCZOS)

        # 풍선이 씬 위로 넘치면 그만큼 상단 여백 확보(+여유 16px)
        if py < 0:
            m_top = max(m_top, -py + 16)

    cw, ch = sw + bd * 2, sh + m_top + m_bot
    ox, oy = bd, m_top

    gutter = gutter_color or WEBTOON_GUTTER_BLACK
    canvas = Image.new("RGB", (cw, ch), gutter)
    canvas.paste(scene, (ox, oy))

    draw = ImageDraw.Draw(canvas)
    black = (0, 0, 0)
    # Draw one integrated frame before bubbles: continuous side rails plus
    # scene separators, all using the same thickness.
    draw.rectangle([0, 0, bd - 1, ch - 1], fill=black)
    draw.rectangle([cw - bd, 0, cw - 1, ch - 1], fill=black)
    draw.rectangle([0, oy, cw - 1, oy + bd - 1], fill=black)
    draw.rectangle([0, oy + sh - bd, cw - 1, oy + sh - 1], fill=black)

    if narration_box is not None:
        box_x, box_y, box_w, box_h, lines, line_h, font, italic_font = narration_box
        draw.rectangle(
            [box_x, box_y, box_x + box_w - 1, box_y + box_h - 1],
            fill=(255, 255, 255),
            outline=black,
            width=NARRATION_BOX_BORDER_W,
        )
        text_y = box_y + NARRATION_BOX_PAD_Y
        for ln in lines:
            lw = _line_width(ln, font, italic_font, draw)
            tx = box_x + max(NARRATION_BOX_PAD_X, (box_w - int(lw)) // 2)
            _draw_rich_text_line(canvas, (tx, text_y), ln, font, italic_font, P.BUBBLE_TEXT_COLOR)
            text_y += line_h
        print("  💬 webtoon narration box")
        return canvas

    if bubble is None:
        return canvas

    # scene → canvas 좌표 변환(oy 만큼 하강) 후 합성
    px, py = px + ox, py + oy
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(bubble, (px, py))
    canvas = canvas_rgba.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    tb_l, tb_t, tb_r, tb_b = _visible_text_box(
        px, py, asset, scale, (bd + 8, 8, cw - bd - 8, ch - 8)
    )
    tb_w = tb_r - tb_l
    tb_h = tb_b - tb_t
    lines = _wrap_rich_text(display_text, font, italic_font, max(10, tb_w), draw, target_sentence)
    while line_h * len(lines) > tb_h and font.size > P.BUBBLE_MIN_FONT:
        font = ImageFont.truetype(P.BUBBLE_FONT, max(P.BUBBLE_MIN_FONT, font.size - 2))
        italic_font = _load_italic_font(font, P.BUBBLE_FONT)
        line_h = int(font.size * P.BUBBLE_LINE_H)
        lines = _wrap_rich_text(display_text, font, italic_font, max(10, tb_w), draw, target_sentence)
    total_h = line_h * len(lines)
    text_y = tb_t + max(0, (tb_h - total_h) // 2)
    for ln in lines:
        lw = _line_width(ln, font, italic_font, draw)
        tx = tb_l + max(0, (tb_w - int(lw)) // 2)
        _draw_rich_text_line(canvas, (tx, text_y), ln, font, italic_font, P.BUBBLE_TEXT_COLOR)
        text_y += line_h

    print(f"  💬 webtoon {bubble_kind} bubble corner={key} (lift={WEBTOON_LIFT_RATIO})")
    return canvas


def stack_webtoon_strip(panels: list, gutter_color=None):
    """Stack same-width framed panels into one continuous vertical webtoon strip."""
    from PIL import Image, ImageDraw
    if not panels:
        return None
    gutter = gutter_color or WEBTOON_GUTTER_BLACK
    cw = panels[0].width
    norm = [p if p.width == cw
            else p.resize((cw, int(p.height * cw / p.width)), Image.LANCZOS)
            for p in panels]
    total_h = sum(p.height for p in norm)
    strip = Image.new("RGB", (cw, total_h), gutter)
    y = 0
    for p in norm:
        strip.paste(p.convert("RGB"), (0, y))
        y += p.height
    # close the rails at the very top / bottom of the whole strip
    bd = WEBTOON_BORDER_W
    d = ImageDraw.Draw(strip)
    d.rectangle([0, 0, cw - 1, bd - 1], fill=gutter)
    d.rectangle([0, total_h - bd, cw - 1, total_h - 1], fill=gutter)
    return strip


def _square_face_crop(image, face_bbox: tuple | None = None, out_size: int = 832):
    """Make a square no-bubble cover crop centered on a detected face."""
    from PIL import Image

    src = image.convert("RGB")
    w, h = src.size
    if face_bbox:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        face = max(x2 - x1, y2 - y1)
        side = int(max(face * 3.4, min(w, h) * 0.56))
    else:
        cx, cy = w / 2, h * 0.42
        side = int(min(w, h) * 0.78)
    side = max(1, min(side, w, h))
    left = int(round(cx - side / 2))
    top = int(round(cy - side * 0.46))
    left = max(0, min(w - side, left))
    top = max(0, min(h - side, top))
    crop = src.crop((left, top, left + side, top + side))
    if crop.size != (out_size, out_size):
        crop = crop.resize((out_size, out_size), Image.LANCZOS)
    return crop


def _choose_cover_variant(panels_cfg: list[dict], variants: list[dict]):
    """Prefer Hanyoil's clean face crop; otherwise use the first character panel."""
    fallback = None
    for pcfg, variant in zip(panels_cfg, variants):
        if not variant.get("cover_src"):
            continue
        item = (variant["cover_src"], variant.get("face_box"))
        if fallback is None:
            fallback = item
        if (pcfg.get("char") or "").strip().lower() == "hanyoil":
            return item
    return fallback


# ─────────────────────────────────────────────────────────
# Generator class
# ─────────────────────────────────────────────────────────
@app.cls(gpu=GPU, volumes={"/models": models_vol, "/output": output_vol}, timeout=3600)
class Generator:
    @modal.enter()
    def setup(self):
        import os, sys, torch
        sys.path.insert(0, "/root")
        import prompts as P
        self.P = P
        self.torch = torch

        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionXLInpaintPipeline,
            DPMSolverMultistepScheduler,
            AutoencoderKL,
        )

        print(f"📦 Base 모델: {P.BASE_MODEL}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            P.BASE_MODEL, torch_dtype=torch.float16, use_safetensors=True,
        )
        # 체크포인트 내장 SDXL VAE 는 fp16 에서 오버플로 → NaN → 검정 이미지(간헐적)를 낸다.
        # fp16 에서도 오버플로하지 않도록 보정된 전용 VAE 로 교체해 검정 렌더를 제거한다.
        print("🩹 fp16-fix VAE 로드 (madebyollin/sdxl-vae-fp16-fix)")
        pipe.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++",
        )
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_attention_slicing()
        self.pipe = pipe.to("cuda")

        # inpaint pipe — base 의 컴포넌트 공유로 VRAM 절약
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(**self.pipe.components).to("cuda")
        self._bg_ip_adapter_loaded = False
        self._bg_ip_adapter_failed = False
        self._bg_ip_adapter_scale = 0.25

        from compel import Compel, ReturnedEmbeddingsType
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        # encoders 공유 → compel 하나 재사용
        self.compel_inpaint = self.compel

        # YOLO (face detector)
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
        yolo_path = hf_hub_download(repo_id="Bingsu/adetailer", filename="face_yolov8s.pt")
        self.yolo = YOLO(yolo_path)
        self.anime_seg_session = None
        self.anime_seg_model = ""
        try:
            import onnxruntime as ort
            anime_seg_path = hf_hub_download(repo_id="skytnt/anime-seg", filename="isnetis.onnx")
            providers = ["CPUExecutionProvider"]
            self.anime_seg_session = ort.InferenceSession(anime_seg_path, providers=providers)
            self.anime_seg_model = "skytnt/anime-seg/isnetis.onnx"
            print(f"✅ anime segmentation: {self.anime_seg_model}")
        except Exception as e:
            print(f"⚠ anime segmentation unavailable — subject masks will fallback: {e}")
        self.seg_yolo = None
        self.seg_yolo_name = ""
        for seg_model in ("yolo11x-seg.pt", "yolov8x-seg.pt", "yolov8n-seg.pt"):
            try:
                self.seg_yolo = YOLO(seg_model)
                self.seg_yolo_name = seg_model
                print(f"✅ segmentation YOLO: {seg_model}")
                break
            except Exception as e:
                print(f"⚠ segmentation YOLO load failed ({seg_model}): {e}")
        if self.seg_yolo is None:
            print("⚠ segmentation YOLO unavailable — subject masks will fallback")

        os.makedirs("/output", exist_ok=True)
        print("✅ Setup 완료")

    def _ensure_bg_ip_adapter(self, scale: float) -> bool:
        """Load a weak background IP-Adapter for plate-guided full-panel generation."""
        if scale <= 0 or self._bg_ip_adapter_failed:
            return False
        if not self._bg_ip_adapter_loaded:
            try:
                print(
                    f"  🧩 loading background IP-Adapter "
                    f"({BG_IP_ADAPTER_REPO}/{BG_IP_ADAPTER_SUBFOLDER}/{BG_IP_ADAPTER_WEIGHT})"
                )
                self.pipe.load_ip_adapter(
                    BG_IP_ADAPTER_REPO,
                    subfolder=BG_IP_ADAPTER_SUBFOLDER,
                    weight_name=BG_IP_ADAPTER_WEIGHT,
                )
                self._bg_ip_adapter_loaded = True
            except Exception as e:
                self._bg_ip_adapter_failed = True
                print(f"  ⚠ background IP-Adapter unavailable — fallback to plain inpaint: {e}")
                return False
        self._bg_ip_adapter_scale = float(scale)
        self.pipe.set_ip_adapter_scale(self._bg_ip_adapter_scale)
        return True

    def _unload_bg_ip_adapter(self) -> None:
        """Remove IP-Adapter before plain text-to-image calls on the shared UNet."""
        if not self._bg_ip_adapter_loaded:
            return
        for pipe in (self.inpaint_pipe, self.pipe):
            try:
                pipe.unload_ip_adapter()
            except Exception:
                pass
        self._bg_ip_adapter_loaded = False

    # ── LoRA 설정 정규화 ──
    @staticmethod
    def _normalize_lora(cfg) -> list[tuple[str, float]]:
        """characters.yaml 의 ``lora`` 값을 [(path, weight), ...] 로 정규화.

        지원 형태:
          - str                         → [(path, 1.0)]              (단일 LoRA)
          - list[str]                   → [(p, 1.0), ...]
          - list[{path, weight}]        → [(p, w), ...]              (블렌딩)
        """
        if isinstance(cfg, str):
            return [(cfg, 1.0)]
        out: list[tuple[str, float]] = []
        for item in cfg:
            if isinstance(item, str):
                out.append((item, 1.0))
            else:
                out.append((item["path"], float(item.get("weight", 1.0))))
        return out

    # ── LoRA 캐시 ──
    def _swap_lora(self, char_name: str, pipe, weights_override=None):
        import os
        # 이전 LoRA 정리 (unfuse → unload 순서 유지)
        try:
            pipe.unfuse_lora()
        except Exception:
            pass
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        specs = self._normalize_lora(self.P.CHARS[char_name]["lora"])  # prompts.py 가 단일 소스
        # weights_override(list[float]) 가 오면 앞에서부터 그만큼만 가중치 교체
        # 예) [1.0] → 첫 LoRA 만 1.0,  [0.8, 0.3] → 두 LoRA 0.8/0.3
        if weights_override:
            specs = [(specs[i][0], float(w)) for i, w in enumerate(weights_override)]
        adapter_names, adapter_weights = [], []
        for i, (lora_path, weight) in enumerate(specs):
            name = f"lora{i}"
            pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path),
                adapter_name=name,
            )
            adapter_names.append(name)
            adapter_weights.append(weight)
        # 다중 LoRA 가중 블렌딩 → fuse (단일 LoRA 도 동일 경로)
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        pipe.fuse_lora(adapter_names=adapter_names, lora_scale=self.P.LORA_SCALE)
        print(f"  🎨 LoRA[{char_name}]: "
              + ", ".join(f"{os.path.basename(p)}×{w}" for p, w in specs))

    def _clear_lora(self, pipe):
        """Object / setting panels use the base model without a character LoRA."""
        try:
            pipe.unfuse_lora()
        except Exception:
            pass
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        print("  🎨 LoRA cleared (object/setting panel)")

    # ── ADetailer face refine ──
    def _adetailer_face(self, image, prompt: str, seed: int, strength: float | None = None):
        import numpy as np
        from PIL import Image
        _strength = self.P.INPAINT_STRENGTH if strength is None else strength
        if _strength <= 0:
            # 강도 0 = 인페인팅 안 함 (베이스 그대로) — diffusers 는 0 스텝이면 에러남
            print("    ⏭ INPAINT_STRENGTH=0 → 인페인팅 스킵")
            return image, None
        P = self.P

        res = self.yolo(np.array(image), conf=0.3, verbose=False)
        faces = []
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                W, H = image.size
                pad = P.INPAINT_PADDING
                faces.append((max(0, x1 - pad), max(0, y1 - pad),
                              min(W, x2 + pad), min(H, y2 + pad)))
        if not faces:
            print("    ⚠ 얼굴 미감지 — 원본 반환")
            return image, None

        result = image.copy()
        for i, (x1, y1, x2, y2) in enumerate(faces):
            fw, fh = x2 - x1, y2 - y1
            face_up = result.crop((x1, y1, x2, y2)).resize(
                (P.CROP_SIZE, P.CROP_SIZE), Image.LANCZOS,
            )
            mask_up = Image.new("L", (P.CROP_SIZE, P.CROP_SIZE), 255)

            cond, pooled = self.compel_inpaint(prompt)
            neg_cond, neg_pooled = self.compel_inpaint(P.INPAINT_NEGATIVE)
            [cond, neg_cond] = self.compel_inpaint.pad_conditioning_tensors_to_same_length(
                [cond, neg_cond]
            )

            gen = self.torch.Generator(device="cuda").manual_seed(P.INPAINT_SEED + i)
            inpainted = self.inpaint_pipe(
                prompt_embeds=cond, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
                image=face_up, mask_image=mask_up,
                width=P.CROP_SIZE, height=P.CROP_SIZE,
                num_inference_steps=P.INPAINT_STEPS,
                guidance_scale=P.INPAINT_CFG,
                strength=_strength,
                generator=gen,
            ).images[0]

            down = inpainted.resize((fw, fh), Image.LANCZOS)

            # 페더링 마스크
            feather = max(8, P.INPAINT_PADDING // 2)
            blend = np.full((fh, fw), 255, dtype=np.float32)
            for e in range(feather):
                v = int(255 * (e / feather))
                blend[e, :] = v; blend[-(e + 1), :] = v
                blend[:, e] = v; blend[:, -(e + 1)] = v
            from PIL import Image as _I
            mask = _I.fromarray(blend.astype(np.uint8))
            result.paste(down, (x1, y1), mask)

        # 가장 큰 face bbox 를 말풍선용 anchor 로
        primary = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return result, primary

    def _generate_background_plate(self, panels_cfg: list[dict], seed: int):
        """Generate one empty recurring-set background to reuse across an episode."""
        P = self.P
        # IP-Adapter mutates the shared UNet; unload it before text-to-image plate generation.
        self._unload_bg_ip_adapter()
        pcfg = _first_background_cfg(panels_cfg)
        bg = _background_bundle(P, pcfg)
        prompt = (
            f"{bg}, empty scene, no humans, no people, clean readable background, "
            f"eye-level shot, {P.QUALITY_TAGS}"
        )
        negative_prompt = f"{P.NEGATIVE_PROMPT}, 1girl, girl, woman, face, body, hands, portrait"
        print(f"  🧱 background plate: {prompt}")

        # The plate should not inherit a character LoRA.
        self._clear_lora(self.pipe)
        cond, pooled = self.compel(prompt)
        neg_cond, neg_pooled = self.compel(negative_prompt)
        [cond, neg_cond] = self.compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])
        gen = self.torch.Generator(device="cuda").manual_seed(seed + 100_000)
        return self.pipe(
            prompt_embeds=cond, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
            width=P.WIDTH, height=P.HEIGHT,
            num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
            num_images_per_prompt=1, generator=gen,
        ).images[0]

    def _generate_ip_adapter_panel(self, plate, prompt: str, negative_prompt: str,
                                   seed: int, ip_adapter_image):
        """Generate a full panel guided by the fixed background plate."""
        P = self.P
        cond, pooled = self.compel(prompt)
        neg_cond, neg_pooled = self.compel(negative_prompt)
        [cond, neg_cond] = self.compel.pad_conditioning_tensors_to_same_length(
            [cond, neg_cond]
        )
        self.pipe.set_ip_adapter_scale(self._bg_ip_adapter_scale)
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        return self.pipe(
            prompt_embeds=cond, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
            width=P.WIDTH, height=P.HEIGHT,
            num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
            num_images_per_prompt=1, generator=gen,
            ip_adapter_image=ip_adapter_image,
        ).images[0]

    def _anime_subject_mask(self, image):
        """Return an anime foreground mask from SkyTNT anime-segmentation ONNX."""
        if self.anime_seg_session is None:
            return None

        import numpy as np
        from PIL import Image, ImageFilter

        raw = image.convert("RGB")
        raw_w, raw_h = raw.size
        inp = self.anime_seg_session.get_inputs()[0]
        out = self.anime_seg_session.get_outputs()[0]
        shape = inp.shape
        seg_h = int(shape[2]) if isinstance(shape[2], int) else 1024
        seg_w = int(shape[3]) if isinstance(shape[3], int) else 1024

        resized = raw.copy()
        resized.thumbnail((seg_w, seg_h), Image.LANCZOS)
        pad_x = (seg_w - resized.width) // 2
        pad_y = (seg_h - resized.height) // 2
        canvas = Image.new("RGB", (seg_w, seg_h), (0, 0, 0))
        canvas.paste(resized, (pad_x, pad_y))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr[:, :, ::-1].transpose(2, 0, 1)[None, ...]
        pred = self.anime_seg_session.run([out.name], {inp.name: arr})[0]
        mask_arr = np.squeeze(pred)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0] if mask_arr.shape[0] <= 4 else mask_arr[:, :, 0]
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
        if mask_arr.shape != (seg_h, seg_w):
            mask_img = Image.fromarray((mask_arr * 255).astype(np.uint8), "L")
            mask_img = mask_img.resize((seg_w, seg_h), Image.BILINEAR)
            mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0

        crop = mask_arr[pad_y:pad_y + resized.height, pad_x:pad_x + resized.width]
        if crop.size == 0 or float(crop.max()) < 0.08:
            return None

        mask = Image.fromarray((crop * 255).astype(np.uint8), "L")
        mask = mask.resize((raw_w, raw_h), Image.LANCZOS)
        mask = mask.point(lambda p: 255 if p > 72 else 0)

        # Keep meaningful foreground islands and drop speckle/background hallucinations.
        import cv2
        bin_arr = np.array(mask, dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_arr > 0).astype(np.uint8), 8)
        cleaned = np.zeros_like(bin_arr)
        min_area = max(300, int(raw_w * raw_h * 0.003))
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 255
        if cleaned.max() == 0:
            return None

        mask = Image.fromarray(cleaned, "L")
        mask = mask.filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(7))
        return mask

    def _extract_subject_mask(self, generated, plate, pcfg: dict, idx: int):
        """Extract the generated subject mask, falling back to a soft slot mask."""
        from PIL import Image, ImageChops, ImageFilter, ImageOps
        import numpy as np

        P = self.P
        panel_type = (pcfg.get("panel_type") or "").strip().lower()
        is_object_panel = panel_type in {"object", "setting"} or not pcfg.get("char")

        if not is_object_panel and self.anime_seg_session is not None:
            try:
                mask = self._anime_subject_mask(generated)
                if mask is not None and mask.getbbox():
                    area = np.array(mask).sum() / 255.0
                    coverage = area / float(P.WIDTH * P.HEIGHT)
                    if 0.015 <= coverage <= 0.72:
                        print(
                            f"  🧍 subject mask: anime segmentation "
                            f"({self.anime_seg_model}, coverage={coverage:.2%})"
                        )
                        return mask
                    print(f"  ⚠ anime segmentation coverage odd ({coverage:.2%}) — fallback")
            except Exception as e:
                print(f"  ⚠ anime segmentation failed — fallback mask: {e}")

        if not is_object_panel and self.seg_yolo is not None:
            try:
                arr = np.array(generated.convert("RGB"))
                res = self.seg_yolo(arr, conf=0.16, imgsz=1024, retina_masks=True, verbose=False)
                mask_arr = np.zeros((P.HEIGHT, P.WIDTH), dtype=np.uint8)
                for r in res:
                    if r.masks is None or r.boxes is None:
                        continue
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    masks = r.masks.data.cpu().numpy()
                    for c, m in zip(cls, masks):
                        if c != 0:  # COCO person
                            continue
                        mi = Image.fromarray((m * 255).astype(np.uint8), "L")
                        if mi.size != (P.WIDTH, P.HEIGHT):
                            mi = mi.resize((P.WIDTH, P.HEIGHT), Image.LANCZOS)
                        mask_arr = np.maximum(mask_arr, np.array(mi))
                if mask_arr.max() > 0:
                    mask = Image.fromarray(mask_arr, "L")
                    mask = mask.filter(ImageFilter.MaxFilter(31)).filter(ImageFilter.GaussianBlur(8))
                    print(f"  🧍 subject mask: YOLO segmentation ({self.seg_yolo_name})")
                    return mask
            except Exception as e:
                print(f"  ⚠ subject segmentation failed — fallback mask: {e}")

        diff = ImageChops.difference(generated.convert("RGB"), plate.convert("RGB"))
        diff = ImageOps.grayscale(diff)
        slot = _make_plate_mask(P, pcfg, idx)
        diff = Image.composite(diff, Image.new("L", diff.size, 0), slot)
        threshold = 22 if not is_object_panel else 28
        mask = diff.point(lambda p: 255 if p > threshold else 0)
        if mask.getbbox():
            label = "diff/person-fallback" if not is_object_panel else "diff/object"
            print(f"  🧍 subject mask: {label}")
            return mask.filter(ImageFilter.MaxFilter(25)).filter(ImageFilter.GaussianBlur(7))

        print("  ⚠ subject mask fallback: soft slot")
        return _make_plate_mask(P, pcfg, idx)

    def _composite_subject_on_plate(self, plate, generated, mask):
        """Keep the original plate, but paste only the generated subject."""
        final = plate.copy().convert("RGB")
        final.paste(generated.convert("RGB"), (0, 0), mask)
        return final

    def _inpaint_on_background_plate(self, plate, mask, prompt: str, negative_prompt: str,
                                     seed: int, strength: float = 0.92,
                                     ip_adapter_image=None):
        """Insert a character/object into a fixed background plate."""
        P = self.P
        cond, pooled = self.compel_inpaint(prompt)
        neg_cond, neg_pooled = self.compel_inpaint(negative_prompt)
        [cond, neg_cond] = self.compel_inpaint.pad_conditioning_tensors_to_same_length(
            [cond, neg_cond]
        )
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        kwargs = {"ip_adapter_image": ip_adapter_image} if ip_adapter_image is not None else {}
        if ip_adapter_image is not None:
            self.inpaint_pipe.set_ip_adapter_scale(self._bg_ip_adapter_scale)
        result = self.inpaint_pipe(
            prompt_embeds=cond, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
            image=plate, mask_image=mask,
            width=P.WIDTH, height=P.HEIGHT,
            num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
            strength=strength,
            generator=gen,
            **kwargs,
        ).images[0]
        # Preserve the original plate outside the mask exactly; only the insertion area blends in.
        final = plate.copy().convert("RGB")
        final.paste(result.convert("RGB"), (0, 0), mask)
        return final

    def _redraw_subject_in_ip_scene(self, generated, mask, prompt: str,
                                    negative_prompt: str, seed: int,
                                    strength: float = 0.68):
        """Redraw only the detected subject area while keeping the IP-Adapter scene."""
        P = self.P
        cond, pooled = self.compel_inpaint(prompt)
        neg_cond, neg_pooled = self.compel_inpaint(negative_prompt)
        [cond, neg_cond] = self.compel_inpaint.pad_conditioning_tensors_to_same_length(
            [cond, neg_cond]
        )
        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        result = self.inpaint_pipe(
            prompt_embeds=cond, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
            image=generated.convert("RGB"), mask_image=mask,
            width=P.WIDTH, height=P.HEIGHT,
            num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
            strength=strength,
            generator=gen,
        ).images[0]
        final = generated.copy().convert("RGB")
        final.paste(result.convert("RGB"), (0, 0), mask)
        return final

    # ── 한 캐릭터+한 의상 생성 ──
    @modal.method()
    def generate_one(self, char_name: str, outfit_name: str,
                     seed: int = -1, bubble_text: str = "",
                     lora_weights: str = "", inpaint_lora_weights: str = "",
                     outfit_weight: str = "", inpaint_strength: str = "",
                     tag: str = "") -> int:
        import os, random
        P = self.P
        if char_name not in P.CHARS:
            raise KeyError(f"unknown char: {char_name}")
        char = P.CHARS[char_name]
        if outfit_name not in char["outfits"]:
            raise KeyError(f"unknown outfit for {char_name}: {outfit_name}")

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        # "0.8,0.3" → [0.8, 0.3] (빈 문자열이면 characters.yaml 기본값 사용)
        weights_override = [float(x) for x in lora_weights.split(",") if x.strip()] or None
        # 인페인팅 전용 가중치 (지정 시 베이스 생성 후 LoRA 를 다시 바꿔 끼움)
        inpaint_weights_override = [float(x) for x in inpaint_lora_weights.split(",") if x.strip()] or None
        # 인페인팅 강도 오버라이드 (빈 문자열이면 prompts.py 의 INPAINT_STRENGTH 사용)
        strength_override = float(inpaint_strength) if inpaint_strength.strip() else None
        print(f"\n=== {char_name} | {outfit_name} | seed={seed}"
              + (f" | tag={tag}" if tag else "") + " ===")

        self._swap_lora(char_name, self.pipe, weights_override=weights_override)
        # inpaint pipe 는 base 와 컴포넌트 공유라 LoRA 자동 적용됨

        char_tags = char.get("tags") or P.compose_char_tags(char, flashback=P.is_flashback(outfit_name))
        face_state = char['face_state']
        if "looking at viewer" not in face_state:
            face_state = f"looking at viewer, {face_state}"
        # 의상 태그 — outfit_weight 주면 컴펠 가중치로 감쌈: (의상)0.8
        outfit_tags = P.compose_outfit(char['outfits'][outfit_name])
        if outfit_weight:
            outfit_tags = f"({outfit_tags}){outfit_weight}"
        # 순서: 퀄리티 → 트리거 → 외형태그(외모·머리·바디) → 아웃핏 → face_state → expression → 액션
        prompt = (
            f"{P.QUALITY_TAGS}, "
            f"{char_name}, {char_tags}, "
            f"{outfit_tags}, "
            f"{face_state}, {char['expression']}, {char['action']}"
        )
        print(f"  📝 prompt: {prompt}")

        cond, pooled = self.compel(prompt)
        neg_cond, neg_pooled = self.compel(P.NEGATIVE_PROMPT)
        [cond, neg_cond] = self.compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])

        gen = self.torch.Generator(device="cuda").manual_seed(seed)
        results = self.pipe(
            prompt_embeds=cond, pooled_prompt_embeds=pooled,
            negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
            width=P.WIDTH, height=P.HEIGHT,
            num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
            num_images_per_prompt=P.NUM_IMAGES, generator=gen,
        )

        out_dir = f"/output/{char_name}_{outfit_name}_{seed}" + (f"_{tag}" if tag else "")
        os.makedirs(out_dir, exist_ok=True)
        inpaint_prompt = (
            f"{char_name}, {char_tags}, {char['expression']}, "
            f"1girl, {P.INPAINT_QUALITY_TAGS}, blushing"
        )

        # 인페인팅 전용 가중치가 따로 지정되면 여기서 LoRA 를 다시 끼움
        # (inpaint pipe 는 base 와 unet 공유 → fuse 가 인페인팅에 그대로 반영됨)
        if inpaint_weights_override:
            print("  🔁 인페인팅용 LoRA 재설정")
            self._swap_lora(char_name, self.pipe, weights_override=inpaint_weights_override)

        for i, img in enumerate(results.images):
            refined, face_box = self._adetailer_face(img, inpaint_prompt, seed + i,
                                                     strength=strength_override)
            if bubble_text:
                refined = add_speech_bubble(refined, bubble_text,
                                            face_bbox=face_box, yolo=self.yolo)
            path = f"{out_dir}/{i:02d}.png"
            refined.save(path)
            print(f"  💾 {path}")
        output_vol.commit()
        return len(results.images)

    # ── 전체 (CHARS × outfits) 일괄 ──
    @modal.method()
    def generate_all(self, seed: int = -1, bubble_text: str = "") -> int:
        count = 0
        for char_name, char in self.P.CHARS.items():
            for outfit_name in char["outfits"]:
                count += self.generate_one.local(
                    char_name, outfit_name, seed=seed, bubble_text=bubble_text,
                )
        return count

    # ── 공통: 패널 목록 생성 (LoRA 상태 인수로 받아서 반환) ──
    def _render_panels(self, panels_cfg: list[dict], seed: int,
                       current_lora: str, use_bg_plate: bool = False,
                       bg_ip_adapter_scale: float = 0.25,
                       bg_base_lora: bool = False) -> tuple[list, str, list[dict]]:
        """panels_cfg 를 받아 이미지 리스트와 업데이트된 current_lora 를 반환."""
        P = self.P
        images = []
        variants = []
        background_plate = None
        bg_ip_adapter_image = None
        if use_bg_plate:
            background_plate = self._generate_background_plate(panels_cfg, seed)
            if self._ensure_bg_ip_adapter(bg_ip_adapter_scale):
                bg_ip_adapter_image = background_plate
                print(f"  🧩 background IP-Adapter scale={bg_ip_adapter_scale}")
            current_lora = ""
        for idx, pcfg in enumerate(panels_cfg):
            char_name = (pcfg.get("char") or "").strip()
            panel_type = (pcfg.get("panel_type") or "").strip().lower()
            is_object_panel = panel_type in {"object", "setting"} or not char_name
            panel_w, panel_h = _panel_render_size(P, is_object_panel)

            if is_object_panel:
                char = None
                if current_lora:
                    self._clear_lora(self.pipe)
                    current_lora = ""
            else:
                char = P.CHARS[char_name]
                if use_bg_plate and bg_ip_adapter_image is not None and not bg_base_lora:
                    if current_lora:
                        self._clear_lora(self.pipe)
                        current_lora = ""
                    print("  🎨 LoRA skipped for IP-Adapter base scene")
                elif char_name != current_lora:
                    self._swap_lora(char_name, self.pipe)
                    current_lora = char_name

            # 구버전(접두어 이전) 키 → 신규 numbered 키 매핑 (옛 scenario_data 호환)
            _OUTFIT_FALLBACK = {
                "office_formal":     "workplace_1",
                "office_edgy":       "workplace_2",
                "casual_chic":       "daily_outing_1",
                "casual_basic":      "daily_convenience_1",
                "casual_athleisure": "daily_sport_1",
                "casual_y2k":        "daily_outing_1",
                "active_sporty":     "daily_sport_1",
                "home_loungewear":   "daily_home_1",
                "home_sexy":         "daily_home_1",
                "date_chic":         "daily_dressup_1",
                "date_sexy":         "daily_dressup_1",
                "school_uniform":    "academic_1",
            }
            # action/expression 은 비주얼 GPT 가 이미 Danbooru 태그로 출력 — 그대로 사용
            action_tags = _scene_action_tags(pcfg)
            bg = _background_bundle(P, pcfg)
            framing_tags = _framing_tags(pcfg, is_object_panel)
            framing_negative = _framing_negative_tags(pcfg, is_object_panel)
            if is_object_panel:
                raw_subject = pcfg.get("subject") or ""
                subject = _object_subject_for_prompt(raw_subject, action_tags)
                prompt = (
                    f"{P.QUALITY_TAGS}, {subject}, {action_tags}, {P.OBJECT_PANEL_TAGS}, "
                    f"{framing_tags}, eye-level shot, {bg}"
                )
            else:
                _outfit_key = pcfg["outfit"]
                if _outfit_key not in char["outfits"]:
                    _outfit_key = _OUTFIT_FALLBACK.get(_outfit_key, "daily_outing_1")
                # outfit(dict|str) + 씬별 hair/props 오버라이드 조합
                outfit_tags = P.compose_outfit(
                    char["outfits"].get(_outfit_key, ""),
                    hair=pcfg.get("hair_override"),
                    props_extra=pcfg.get("props_extra"),
                )
                char_tags = char.get("tags") or P.compose_char_tags(char, flashback=P.is_flashback(_outfit_key))
                expr_tags = pcfg.get("expression", "")
                # 시선: 비었을 때만 정면 폴백. 명시적 시선(looking down/away/up …)은 그대로 둔다.
                #       (예전엔 무조건 "looking at viewer"를 붙여 'looking at viewer, looking down' 모순이 났다)
                panel_face = pcfg.get("face_state", "").strip() or "looking at viewer"
                # 카메라는 항상 eye-level 로 고정 (위/아래 앵글 방지). 퀄리티 태그는 맨 앞.
                prompt = (
                    f"{P.QUALITY_TAGS}, {char_name}, {action_tags}, "
                    f"{char_tags}, {outfit_tags}, {expr_tags}, {panel_face}, "
                    f"{framing_tags}, eye-level shot, {bg}"
                )
            negative_prompt = (
                P.NEGATIVE_PROMPT + ", " + P.OBJECT_NEGATIVE_TAGS
                if is_object_panel else P.NEGATIVE_PROMPT
            )
            if framing_negative:
                negative_prompt += ", " + framing_negative
            if is_object_panel:
                object_negative = _object_negative_for_prompt(pcfg.get("subject", ""), action_tags)
                if object_negative:
                    negative_prompt += ", " + object_negative

            panel_seed = seed + pcfg.get("seed_offset", idx)
            panel_variants = {}
            if background_plate is not None:
                plate = background_plate.copy().convert("RGB")
                if bg_ip_adapter_image is not None:
                    self._ensure_bg_ip_adapter(bg_ip_adapter_scale)
                    generated = self._generate_ip_adapter_panel(
                        plate, prompt, negative_prompt, panel_seed, bg_ip_adapter_image,
                    )
                    mask = self._extract_subject_mask(generated, plate, pcfg, idx)
                    panel_variants["ip_adapter"] = generated.copy().convert("RGB")
                    panel_variants["subject_mask"] = mask.copy()
                    self._unload_bg_ip_adapter()
                    if is_object_panel:
                        redraw_prompt = (
                            f"{subject}, {action_tags}, object focus, still life, {bg}, "
                            f"clean edges, natural contact shadows, {P.QUALITY_TAGS}"
                        )
                        result = self._redraw_subject_in_ip_scene(
                            generated, mask, redraw_prompt, negative_prompt,
                            panel_seed + 20_000, strength=0.62,
                        )
                    else:
                        if char_name != current_lora:
                            print("  🔁 LoRA enabled for subject redraw")
                            self._swap_lora(char_name, self.pipe)
                            current_lora = char_name
                        redraw_prompt = (
                            f"{char_name}, {char_tags}, {outfit_tags}, {expr_tags}, "
                            f"{panel_face}, {action_tags}, white background, "
                            f"{P.INPAINT_QUALITY_TAGS}, {P.QUALITY_TAGS}"
                        )
                        result = self._redraw_subject_in_ip_scene(
                            generated, mask, redraw_prompt, negative_prompt,
                            panel_seed + 20_000, strength=0.80,
                        )
                    panel_variants["person_redraw"] = result.copy().convert("RGB")
                else:
                    # If IP-Adapter cannot load, keep the old mask-inpaint fallback so
                    # background-plate renders still produce inspectable output.
                    mask = _make_plate_mask(P, pcfg, idx)
                    if is_object_panel:
                        fallback_prompt = (
                            f"{subject}, {action_tags}, object focus, still life, {bg}, "
                            f"natural contact shadows, {P.QUALITY_TAGS}"
                        )
                        result = self._inpaint_on_background_plate(
                            plate, mask, fallback_prompt, negative_prompt, panel_seed, strength=0.86,
                        )
                    else:
                        fallback_prompt = (
                            f"{char_name}, {char_tags}, {outfit_tags}, {expr_tags}, "
                            f"{panel_face}, {action_tags}, full body, natural contact shadows, "
                            f"standing in the scene, {P.QUALITY_TAGS}"
                        )
                        result = self._inpaint_on_background_plate(
                            plate, mask, fallback_prompt, negative_prompt, panel_seed, strength=0.94,
                        )
            else:
                cond, pooled = self.compel(prompt)
                neg_cond, neg_pooled = self.compel(negative_prompt)
                # compel 정식 헬퍼로 길이 정렬 (generate_one 과 동일) — empty-token 패딩 보장
                [cond, neg_cond] = self.compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])

                gen = self.torch.Generator(device="cuda").manual_seed(panel_seed)
                result = self.pipe(
                    prompt_embeds=cond, pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
                    width=panel_w, height=panel_h,
                    num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
                    num_images_per_prompt=1, generator=gen,
                ).images[0]

            if is_object_panel:
                refined, face_box = result, None
            else:
                inpaint_prompt = (
                    f"{char_name}, {char_tags}, {expr_tags}, "
                    f"1girl, {P.INPAINT_QUALITY_TAGS}"
                )
                refined, face_box = self._adetailer_face(
                    result, inpaint_prompt, panel_seed, strength=0.60,
                )
                panel_variants["cover_src"] = refined.copy().convert("RGB")
                panel_variants["face_box"] = face_box

            raw_bubble_text = pcfg.get("bubble", "")
            if is_object_panel:
                if _is_narration_text(raw_bubble_text):
                    bubble_text = raw_bubble_text
                elif not raw_bubble_text:
                    auto_caption = _object_narration_text(pcfg)
                    bubble_text = f"(narration) {auto_caption}" if auto_caption else ""
                else:
                    bubble_text = ""
            else:
                bubble_text = raw_bubble_text
            if bubble_text:
                print(f"  💬 말풍선: \"{bubble_text}\"")
            # 웹툰 액자: 모든 패널을 동일 프레임(좌우 레일+박스)으로 씌워 폭을 통일한다.
            # 풍선은 대사가 있을 때만 상단 여백에 띄운다(꼬리 아래, 세 점 콜리니어).
            # 폭이 같아야 stack_webtoon_strip 에서 레일이 끊김 없이 이어진다.
            refined = compose_webtoon_panel(
                refined, bubble_text, face_bbox=face_box,
                yolo=None if is_object_panel else self.yolo,
                gutter_color=_webtoon_gutter_color(pcfg),
                target_sentence=pcfg.get("target_sentence", ""),
            )

            images.append(refined)
            variants.append(panel_variants)
            label = "object" if is_object_panel else char_name
            print(f"  ✅ 패널 {idx + 1}/{len(panels_cfg)} ({label}) 완료")

        return images, current_lora, variants

    # ── 4컷 만화 단일 (prompts.py COMIC_PANELS 사용) ──
    @modal.method()
    def generate_comic(self, seed: int = -1, bg_plate: bool = False,
                       bg_ip_adapter_scale: float = 0.25,
                       bg_base_lora: bool = False) -> int:
        import os, random, shutil
        P = self.P

        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        chars_in_comic = [(pcfg.get("char") or "object") for pcfg in P.COMIC_PANELS]
        print(f"\n=== 4컷 만화 | chars={chars_in_comic} | seed={seed} ===")

        panels, _, variants = self._render_panels(
            P.COMIC_PANELS, seed, "", use_bg_plate=bg_plate,
            bg_ip_adapter_scale=bg_ip_adapter_scale,
            bg_base_lora=bg_base_lora,
        )

        out_dir = f"/output/comic_{'_'.join(dict.fromkeys(chars_in_comic))}_{seed}"
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, panel in enumerate(panels):
            path = f"{out_dir}/panel_{i + 1:02d}.png"
            panel.save(path)
            print(f"  💾 {path}")
        strip = stack_webtoon_strip(
            panels,
            gutter_color=_webtoon_gutter_color(P.COMIC_PANELS[0] if P.COMIC_PANELS else {}),
        )
        if strip is not None:
            strip_path = f"{out_dir}/strip.png"
            strip.save(strip_path)
            print(f"  💾 {strip_path}")
        cover_item = _choose_cover_variant(P.COMIC_PANELS, variants)
        if cover_item:
            cover_img = _square_face_crop(*cover_item)
            cover_path = f"{out_dir}/cover.png"
            cover_img.save(cover_path)
            print(f"  💾 {cover_path}")
        if bg_plate:
            variants_dir = f"{out_dir}/variants"
            os.makedirs(variants_dir, exist_ok=True)
            for i, panel_variants in enumerate(variants):
                for name, img in panel_variants.items():
                    path = f"{variants_dir}/panel_{i + 1:02d}_{name}.png"
                    img.save(path)
                    print(f"  💾 {path}")
        output_vol.commit()
        return len(panels)

    # ── 복습 카드 (단어별 단일 이미지, Notion 갤러리 썸네일용) ──
    @modal.method()
    def generate_review_cards(self, batch_json: str,
                              width: int = 832, height: int = 832) -> int:
        """단어별 복습 카드 1장씩 생성 (말풍선/텍스트 없음).

        generate_scenario.build_review_card 가 표현의 뉘앙스로 설계한 단일 패널을 panel 로
        넘긴다. character 카드는 한요일 단독 인출 단서, object 카드는 사람 없는 사물/장면
        단서로 렌더한다. width/height 는 Notion 갤러리 카드 비율에 맞춰 조정 가능
        (기본 정사각 832²).
        """
        import json, os, random
        P = self.P
        cards = json.loads(batch_json)
        current_lora = ""
        count = 0
        out_root = "/output/review_cards"
        os.makedirs(out_root, exist_ok=True)

        for i, card in enumerate(cards):
            pcfg = card.get("panel") or {}
            no   = card.get("no") or str(i + 1)
            seed = card.get("seed", -1)
            if seed is None or seed < 0:
                seed = random.randint(0, 2**32 - 1)

            char_name = (pcfg.get("char") or "").strip()
            panel_type = (pcfg.get("panel_type") or "").strip().lower()
            is_object_panel = panel_type in {"object", "setting"} or not char_name
            # 복습 카드는 '인출 단서'(키워드법) — 만화 씬 배경을 빼고 항상 흰배경으로
            # 고정한다(씬 배경을 쓰면 단서가 산만해지고 갤러리 썸네일 일관성도 깨진다).
            bg = "white background, simple background"
            action_tags = _scene_action_tags(pcfg)
            framing_tags = _framing_tags(pcfg, is_object_panel)
            framing_negative = _framing_negative_tags(pcfg, is_object_panel)

            if is_object_panel:
                if current_lora:
                    self._clear_lora(self.pipe)
                    current_lora = ""
                raw_subject = pcfg.get("subject") or ""
                subject = _object_subject_for_prompt(raw_subject, action_tags)
                prompt = (
                    f"{P.QUALITY_TAGS}, {subject}, {action_tags}, {P.OBJECT_PANEL_TAGS}, "
                    f"{framing_tags}, eye-level shot, {bg}"
                )
                negative_prompt = (
                    P.NEGATIVE_PROMPT + ", " + P.OBJECT_NEGATIVE_TAGS + ", "
                    + getattr(P, "REVIEW_CARD_NEGATIVE_TAGS", "")
                )
                if framing_negative:
                    negative_prompt += ", " + framing_negative
                object_negative = _object_negative_for_prompt(raw_subject, action_tags)
                if object_negative:
                    negative_prompt += ", " + object_negative
            else:
                char = P.CHARS[char_name]
                if char_name != current_lora:
                    self._swap_lora(char_name, self.pipe)
                    current_lora = char_name
                _outfit_key = pcfg.get("outfit", "")
                if _outfit_key not in char["outfits"]:
                    _outfit_key = "daily_outing_1"
                outfit_tags = P.compose_outfit(
                    char["outfits"].get(_outfit_key, ""),
                    hair=pcfg.get("hair_override"),
                    props_extra=pcfg.get("props_extra"),
                )
                char_tags = char.get("tags") or P.compose_char_tags(char, flashback=P.is_flashback(_outfit_key))
                expr_tags = pcfg.get("expression", "")
                panel_face = pcfg.get("face_state", "").strip() or "looking at viewer"
                prompt = (
                    f"{P.QUALITY_TAGS}, {char_name}, {action_tags}, "
                    f"{char_tags}, {outfit_tags}, {expr_tags}, {panel_face}, "
                    f"solo, {framing_tags}, centered, hands visible, eye-level shot, {bg}"
                )
                negative_prompt = (
                    P.NEGATIVE_PROMPT + ", " + getattr(P, "REVIEW_CARD_NEGATIVE_TAGS", "")
                )
                if framing_negative:
                    negative_prompt += ", " + framing_negative

            print(f"\n=== 복습카드 [{i+1}/{len(cards)}] word {no} | seed={seed} | "
                  f"{'object' if is_object_panel else char_name} ===")
            cond, pooled = self.compel(prompt)
            neg_cond, neg_pooled = self.compel(negative_prompt)
            [cond, neg_cond] = self.compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])
            gen = self.torch.Generator(device="cuda").manual_seed(seed)
            result = self.pipe(
                prompt_embeds=cond, pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_cond, negative_pooled_prompt_embeds=neg_pooled,
                width=width, height=height,
                num_inference_steps=P.STEPS, guidance_scale=P.CFG_SCALE,
                num_images_per_prompt=1, generator=gen,
            ).images[0]

            if not is_object_panel:
                inpaint_prompt = (
                    f"{char_name}, {char_tags}, {expr_tags}, "
                    f"1girl, {P.INPAINT_QUALITY_TAGS}"
                )
                result, _ = self._adetailer_face(result, inpaint_prompt, seed, strength=0.60)

            path = f"{out_root}/word_{no}_{seed}.png"
            result.save(path)
            print(f"  💾 {path}")
            count += 1

        output_vol.commit()
        return count

    # ── 4컷 만화 배치 (단어 N개를 모델 로딩 1회로 처리) ──
    @modal.method()
    def generate_batch_comic(self, batch_json: str, bg_plate: bool = False,
                             bg_ip_adapter_scale: float = 0.25,
                             bg_base_lora: bool = False) -> None:
        """여러 단어의 4컷 만화를 모델 로딩 1회로 순차 생성.

        batch_json 형식:
          [{"panels": [...], "seed": 1235}, ...]
        """
        import json, os, shutil
        batch = json.loads(batch_json)
        current_lora = ""

        for i, item in enumerate(batch):
            panels_cfg = item["panels"]
            seed       = item["seed"]
            chars      = "_".join(dict.fromkeys((p.get("char") or "object") for p in panels_cfg))
            print(f"\n=== [{i+1}/{len(batch)}] comic_{chars}_{seed} ===")
            for pi, p in enumerate(panels_cfg):
                print(f"  패널{pi+1} [{p.get('char')}] \"{p.get('bubble', '')}\"")


            images, current_lora, variants = self._render_panels(
                panels_cfg, seed, current_lora, use_bg_plate=bg_plate,
                bg_ip_adapter_scale=bg_ip_adapter_scale,
                bg_base_lora=bg_base_lora,
            )

            out_dir = f"/output/comic_{chars}_{seed}"
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)
            for j, img in enumerate(images):
                path = f"{out_dir}/panel_{j + 1:02d}.png"
                img.save(path)
                print(f"  💾 {path}")
            strip = stack_webtoon_strip(
                images,
                gutter_color=_webtoon_gutter_color(panels_cfg[0] if panels_cfg else {}),
            )
            if strip is not None:
                strip_path = f"{out_dir}/strip.png"
                strip.save(strip_path)
                print(f"  💾 {strip_path}")
            cover_item = _choose_cover_variant(panels_cfg, variants)
            if cover_item:
                cover_img = _square_face_crop(*cover_item)
                cover_path = f"{out_dir}/cover.png"
                cover_img.save(cover_path)
                print(f"  💾 {cover_path}")
            if bg_plate:
                variants_dir = f"{out_dir}/variants"
                os.makedirs(variants_dir, exist_ok=True)
                for j, panel_variants in enumerate(variants):
                    for name, img in panel_variants.items():
                        path = f"{variants_dir}/panel_{j + 1:02d}_{name}.png"
                        img.save(path)
                        print(f"  💾 {path}")
            output_vol.commit()
            print(f"  ✅ {out_dir} 저장 완료")

    @modal.method()
    def generate_batch_backgrounds(self, batch_json: str) -> None:
        """Generate and save only the empty recurring background plate per batch item."""
        import json, os
        batch = json.loads(batch_json)
        out_dir = "/output/background_plates"
        os.makedirs(out_dir, exist_ok=True)
        for i, item in enumerate(batch):
            panels_cfg = item["panels"]
            seed = item["seed"]
            pcfg = _first_background_cfg(panels_cfg)
            set_name = _bg_key(pcfg.get("background_set") or pcfg.get("set") or "background")
            room = _bg_key(pcfg.get("background_room") or pcfg.get("room") or "")
            label = "_".join(p for p in (set_name, room) if p)
            print(f"\n=== [{i+1}/{len(batch)}] background_{seed}_{label} ===")
            plate = self._generate_background_plate(panels_cfg, seed)
            path = f"{out_dir}/background_{seed}_{label}.png"
            plate.save(path)
            print(f"  💾 {path}")
        output_vol.commit()


# ─────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(char: str = "", outfit: str = "", seed: int = -1, bubble: str = "",
         comic: bool = False, batch_comic: bool = False, batch_json: str = "",
         lora_weights: str = "", inpaint_lora_weights: str = "",
         outfit_weight: str = "", inpaint_strength: str = "", tag: str = "",
         bg_plate: bool = False, batch_json_path: str = "",
         bg_ip_adapter_scale: float = 0.25, background_only: bool = False,
         bg_base_lora: bool = False, review_cards: bool = False,
         review_width: int = 832, review_height: int = 832):
    gen = Generator()
    if review_cards:
        if batch_json_path:
            batch_json = Path(batch_json_path).read_text(encoding="utf-8")
        n = gen.generate_review_cards.remote(
            batch_json=batch_json, width=review_width, height=review_height,
        )
        print(f"\n✅ 복습 카드 {n}장 생성 — review_cards/ 확인")
    elif background_only:
        if batch_json_path:
            batch_json = Path(batch_json_path).read_text(encoding="utf-8")
        gen.generate_batch_backgrounds.remote(batch_json=batch_json)
        print("\n✅ 배경 plate 생성 완료 — comic-output volume 확인")
    elif batch_comic:
        if batch_json_path:
            batch_json = Path(batch_json_path).read_text(encoding="utf-8")
        gen.generate_batch_comic.remote(
            batch_json=batch_json, bg_plate=bg_plate,
            bg_ip_adapter_scale=bg_ip_adapter_scale,
            bg_base_lora=bg_base_lora,
        )
        print("\n✅ 배치 만화 생성 완료 — comic-output volume 확인")
    elif comic:
        n = gen.generate_comic.remote(
            seed=seed, bg_plate=bg_plate,
            bg_ip_adapter_scale=bg_ip_adapter_scale,
            bg_base_lora=bg_base_lora,
        )
        print(f"\n✅ 4컷 만화 {n} 패널 생성 — comic-output volume 확인")
    elif char and outfit:
        n = gen.generate_one.remote(char, outfit, seed=seed, bubble_text=bubble,
                                    lora_weights=lora_weights,
                                    inpaint_lora_weights=inpaint_lora_weights,
                                    outfit_weight=outfit_weight,
                                    inpaint_strength=inpaint_strength, tag=tag)
        print(f"\n✅ {n} 장 생성 — comic-output volume 확인")
    else:
        n = gen.generate_all.remote(seed=seed, bubble_text=bubble)
        print(f"\n✅ 전체 {n} 장 생성 — comic-output volume 확인")
