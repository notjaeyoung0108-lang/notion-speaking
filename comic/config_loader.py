"""config_loader.py — data/*.yaml 로더.

prompts.py(Modal flat import) 와 generate_scenario.py(로컬 package import) 가 함께 쓴다.
YAML 파일은 이 파일과 같은 폴더의 data/ 안에 있다고 가정한다.
  - 로컬: notion_words/comic/data/*.yaml
  - Modal: /root/data/*.yaml  (sd_generate.py 가 add_local_dir 로 올림)

캐싱: 각 파일은 최초 1회만 파싱한다.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_DATA = Path(__file__).parent / "data"
_CACHE: dict[str, dict] = {}


def _load(name: str) -> dict:
    if name not in _CACHE:
        with (_DATA / name).open(encoding="utf-8") as f:
            _CACHE[name] = yaml.safe_load(f) or {}
    return _CACHE[name]


# ── 캐릭터 ───────────────────────────────────────────────
def load_characters() -> dict:
    """characters.yaml → CHARS dict (prompts.py CHARS 와 동일 형태)."""
    return _load("characters.yaml")


# academic_N 의상 = 고등학교 플래시백 (lore 참조)
# (구버전 호환: academic_uniform / school_uniform 도 인정)
FLASHBACK_OUTFITS = {"academic_1", "academic_uniform", "school_uniform"}


def is_flashback(outfit_name: str | None) -> bool:
    """이 의상이 과거 회상(교복 화)인가. academic_N 접두어 전부 인정."""
    name = (outfit_name or "")
    return name in FLASHBACK_OUTFITS or name.startswith("academic")


def char_hair(char: dict, flashback: bool = False) -> str:
    """캐릭터 머리 태그 (default / flashback). 구버전(hair 없음)이면 빈 문자열."""
    hair = char.get("hair", "")
    if isinstance(hair, dict):
        if flashback:
            return hair.get("flashback") or hair.get("default", "")
        return hair.get("default", "")
    return str(hair or "")


def compose_char_tags(char: dict, flashback: bool = False) -> str:
    """appearance_tags + hair(default|flashback) + body_tags → SDXL 태그 문자열.

    flashback=True 면 과거 회상용 머리(색/스타일)를 쓴다.
    """
    parts = [
        char.get("appearance_tags", ""),
        char_hair(char, flashback),
        char.get("body_tags", ""),
    ]
    return ", ".join(p for p in parts if p)


# ── 표정 ─────────────────────────────────────────────────
def load_expressions() -> dict:
    """expressions.yaml 의 expressions 매핑 (key → {tags, use_when})."""
    return _load("expressions.yaml").get("expressions", {})


def resolve_expression(key: str | None) -> str:
    """표정 key → SDXL tags.

    GPT 가 메뉴 key("deadpan")를 주면 tags 로 변환한다.
    이미 tags 문자열이거나 알 수 없는 key 면 그대로 반환(폴백),
    빈 값이면 neutral 로 폴백한다.
    """
    exprs = load_expressions()
    raw = (key or "").strip()
    norm = raw.lower().replace(" ", "_").replace("-", "_")
    if norm in exprs:
        return exprs[norm].get("tags", raw)
    if raw:
        return raw  # GPT 가 tags 를 직접 줬을 수도 있음 — 보존
    return exprs.get("neutral", {}).get("tags", "neutral")


def build_expression_menu() -> str:
    """expressions.yaml → GPT 프롬프트용 선택 메뉴 문자열."""
    lines = []
    for key, info in load_expressions().items():
        use_when = (info or {}).get("use_when", "")
        lines.append(f"- {key}: {use_when}")
    return "\n".join(lines)


# ── 포즈 ─────────────────────────────────────────────────
def load_poses() -> dict:
    """poses.yaml 의 poses 매핑 (key → {tags, use_when})."""
    return _load("poses.yaml").get("poses", {})


def load_body_poses() -> dict:
    return _load("poses.yaml").get("body_poses", {})


def load_gestures() -> dict:
    return _load("poses.yaml").get("gestures", {})


def resolve_pose(key: str | None) -> str:
    """포즈 key → SDXL/Danbooru tags. (resolve_expression 과 동일 동작)

    GPT 가 메뉴 key("arms_crossed")를 주면 tags("crossed arms")로 변환한다.
    콤마로 여러 key 를 줬으면 각각 변환해 합친다. 알 수 없는 key 면 그대로 보존,
    빈 값이면 standing 으로 폴백한다.
    """
    poses = load_poses()
    raw = (key or "").strip()
    if not raw:
        return poses.get("standing", {}).get("tags", "standing")
    out = []
    for part in raw.split(","):
        p = part.strip()
        norm = p.lower().replace(" ", "_").replace("-", "_")
        if norm in poses:
            out.append(poses[norm].get("tags", p))
        elif p:
            out.append(p)  # GPT 가 tags 를 직접 줬을 수도 있음 — 보존
    return ", ".join(dict.fromkeys(out)) or "standing"


def _resolve_motion_key(menu: dict, key: str | None) -> str:
    raw = (key or "").strip()
    if not raw:
        return ""
    norm = raw.lower().replace(" ", "_").replace("-", "_")
    if norm in menu:
        return (menu[norm].get("tags") or "").strip()
    return raw


def resolve_motion(body_pose: str | None, gesture: str | None) -> str:
    """Combine one body pose and one gesture into Danbooru action tags."""
    body = _resolve_motion_key(load_body_poses(), body_pose) or "standing"
    gesture_tags = _resolve_motion_key(load_gestures(), gesture)
    parts = [body]
    if gesture_tags:
        parts.append(gesture_tags)
    return ", ".join(dict.fromkeys(p for p in parts if p))


def build_pose_menu() -> str:
    """poses.yaml → GPT 프롬프트용 선택 메뉴 문자열."""
    lines = []
    for key, info in load_poses().items():
        use_when = (info or {}).get("use_when", "")
        lines.append(f"- {key}: {use_when}")
    return "\n".join(lines)


def build_motion_menu() -> str:
    body_lines = ["BODY_POSES (choose exactly one key):"]
    for key, info in load_body_poses().items():
        body_lines.append(f"- {key}: {(info or {}).get('use_when', '')}")
    gesture_lines = ["GESTURES (choose exactly one key; use none when unclear):"]
    for key, info in load_gestures().items():
        gesture_lines.append(f"- {key}: {(info or {}).get('use_when', '')}")
    return "\n".join(body_lines + [""] + gesture_lines)


# ── 참고용 어휘 (현재 GPT 자동 선택 안 함) ──────────────────
def load_hairstyles() -> dict:
    return _load("hairstyles.yaml").get("hairstyles", {})


def load_props() -> dict:
    return _load("props.yaml").get("props", {})


def load_background_sets() -> dict:
    """background_sets.yaml 의 recurring set/room SDXL tag bundles. (locations.yaml 로 일원화 — 레거시)"""
    return _load("background_sets.yaml").get("background_sets", {})


# ── 장소 (locations.yaml — 배경 단일 소스) ─────────────────
def load_locations() -> dict:
    """locations.yaml 의 location 매핑 (tag → {kr, object})."""
    return _load("locations.yaml").get("locations", {})


def load_location_domains() -> dict:
    """도메인별 사용 가능 location tag 화이트리스트."""
    return _load("locations.yaml").get("domains", {})


def resolve_location_outfit_setting(tag: str | None, domain: str = "") -> str:
    """location tag → locations.yaml 의 outfit_setting 값.

    장소 tag 를 모르면 resolve_location 과 동일하게 도메인 canonical 장소로 폴백한다.
    """
    locs = load_locations()
    key = (tag or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key not in locs:
        wl = load_location_domains().get((domain or "").strip().lower()) or []
        key = wl[0] if wl else ""
    if key not in locs:
        return ""
    return str((locs[key] or {}).get("outfit_setting") or "").strip().lower()


def resolve_location(tag: str | None, domain: str = "", prop: str | None = None) -> str:
    """location tag → SDXL 배경 문자열 "장소, 소품1개" (짧게).

    prop(장면에 맞게 생성된 소품 1개)이 있으면 그걸 쓰고, 없으면 locations.yaml 의
    기본 object 로 폴백한다. 장소 tag 를 모르면 도메인 canonical 로 폴백.
    """
    locs = load_locations()
    key = (tag or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key not in locs:
        wl = load_location_domains().get((domain or "").strip().lower()) or []
        key = wl[0] if wl else ""
    if key not in locs:
        return "simple background"
    bg_tag = key.replace("_", " ")
    # 소품 0~2개. prop 필드가 아예 없으면(None) yaml 기본 object 로 폴백,
    # 빈 문자열("")이면 planner 가 '소품 없음'을 고른 것 → 장소만.
    if prop is None:
        items = [o for o in [((locs[key] or {}).get("object") or "").strip()] if o]
    else:
        items = [p.strip() for p in str(prop).split(",") if p.strip()][:2]
    return ", ".join([bg_tag] + items)


def build_location_menu(domain: str) -> str:
    """planner 프롬프트용 — 해당 도메인에서 고를 수 있는 location 목록 (앞쪽=canonical)."""
    locs = load_locations()
    wl = load_location_domains().get((domain or "").strip().lower()) or []
    lines = []
    for tag in wl:
        kr = (locs.get(tag) or {}).get("kr", "")
        lines.append(f"- {tag} ({kr})")
    return "\n".join(lines)
