"""generate_scenario.py — GPT-4o로 4컷 시나리오 생성 → prompts.py 업데이트 → modal 실행.

Usage:
  python generate_scenario.py                  # 랜덤 시드
  python generate_scenario.py --seed 42        # 고정 시드
  python generate_scenario.py --word 1         # CSV 1번 단어 기반 시나리오
  python generate_scenario.py --word 1 --dry   # modal 실행 없이 시나리오만 확인
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from openai import OpenAI

if __package__:
    from .. import config
    from . import episode_log
    from .lore_sanitize import sanitize_character_bible
    from .config_loader import (
        build_expression_menu,
        build_location_menu,
        build_motion_menu,
        build_pose_menu,
        load_characters,
        load_expressions,
        load_location_domains,
        resolve_expression,
        resolve_motion,
        resolve_location,
        resolve_location_outfit_setting,
    )
    from .scenario_prompts import (
        CATEGORY_DEFAULT_SETTING,
        build_arc_prompt,
        build_planner_prompt,
        build_review_card_prompt,
        build_script_prompt,
        build_visual_prompt,
        build_word_block,
        build_word_rule,
        tone_rule_for,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from notion_speaking import config
    from notion_speaking.comic import episode_log
    from notion_speaking.comic.lore_sanitize import sanitize_character_bible
    from notion_speaking.comic.config_loader import (
        build_expression_menu,
        build_location_menu,
        build_motion_menu,
        build_pose_menu,
        load_characters,
        load_expressions,
        load_location_domains,
        resolve_expression,
        resolve_motion,
        resolve_location,
        resolve_location_outfit_setting,
    )
    from notion_speaking.comic.scenario_prompts import (
        CATEGORY_DEFAULT_SETTING,
        build_arc_prompt,
        build_planner_prompt,
        build_review_card_prompt,
        build_script_prompt,
        build_visual_prompt,
        build_word_block,
        build_word_rule,
        tone_rule_for,
    )

HERE         = Path(__file__).parent
LORE_DIR     = HERE.parent / "lore"        # 정본: notion_words/lore/*.md + relationship_state.yaml
ARC_STATE_PATH = LORE_DIR / "relationship_state.yaml"
FACET_STATE_PATH = LORE_DIR / "facet_state.yaml"
FACET_RECENT_N = int(os.getenv("FACET_RECENT_N", "5"))
# 프롬프트에 주입할 lore 본문 — 이 순서로 이어붙인다 (arc 상태는 별도 yaml)
# situation.md(고정 상황 라이브러리)는 주입하지 않는다 — 메뉴를 보면 플래너가 같은 상황을
# 재사용해 반복이 생겼다. 상황은 nuance 로부터 매번 자유 생성한다.
_LORE_FILES  = ["world.md", "characters.md", "episode_rules.md"]
PROMPTS_PATH = HERE / "prompts.py"


def load_lore() -> str:
    """lore/ 디렉터리의 마크다운 파일들을 한 본문으로 이어붙인다 (프롬프트 주입용)."""
    parts = []
    for fn in _LORE_FILES:
        p = LORE_DIR / fn
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(parts)


def load_domain_world(domain: str) -> str:
    """해당 도메인의 세계관 블록만 로드 (lore/domains/<domain>.md). 없으면 빈 문자열."""
    fn = _DOMAIN_FILE.get((domain or "").strip().lower())
    if not fn:
        return ""
    p = DOMAINS_DIR / fn
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _shared_lore() -> str:
    """도메인 무관 공용 lore — 톤/Found Family 등(world.md)과 작법(episode_rules.md)."""
    parts = []
    for fn in ("world.md", "episode_rules.md"):
        p = LORE_DIR / fn
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n---\n\n".join(parts)


def load_character_bible(names: list[str]) -> str:
    """characters.md 에서 '## <name>' 섹션 중 names 에 해당하는 것 + 캐릭터가 아닌
    공용 섹션(FACET 회전 규칙·관계 위계)만 남겨 슬라이스. 등장 인물 것만 주입해 컨텍스트 절감."""
    p = LORE_DIR / "characters.md"
    if not p.exists():
        return ""
    want = {n.strip().lower() for n in (names or []) if n}
    text = p.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^(?=## )", text)  # '## ' 헤더 단위로 분할 (선두 프리앰블 포함)
    kept = []
    for b in blocks:
        m = re.match(r"##\s+(\S+)", b)
        head = m.group(1).strip().lower() if m else ""
        is_char_section = head in _KNOWN_CHARS
        if not is_char_section:
            kept.append(b.rstrip())            # 공용 섹션(규칙/위계/프리앰블)은 항상 유지
        elif head in want:
            kept.append(b.rstrip())            # 등장 인물 섹션만 유지
    return sanitize_character_bible("\n\n".join(x for x in kept if x.strip()))


CSV_PATH     = config.STRUCTURED_CSV  # 시나리오 원천: 웹툰 대사로 덮이지 않은 원본 structured CSV

SEED_OFFSETS = [0, 7, 13, 21, 29, 37, 43, 51]  # 패널별 시드 분산 (가변 컷, 최대 8)
MAX_PANELS = 8

# ── 모델 티어링 ─────────────────────────────────────────────
# 단계별로 난이도가 다르다 → 기본은 gpt-4o(상위)/gpt-4o-mini(하위). env 로 덮어쓰기 가능.
#   ① Planner : 구조/추론 → 상위 모델
#   ② Script  : 학습자용 대사 품질이 결과물 핵심 → 상위 모델
#   ③ Visual  : expressions.yaml 메뉴 key 선택 + 태그 몇 개 (준기계적) → 하위 모델로 충분
# 전부 env 로 덮어쓸 수 있어 비용/품질을 측정하며 조정할 수 있다.
MODEL_PLAN   = os.getenv("MODEL_PLAN",   "gpt-4o")
MODEL_SCRIPT = os.getenv("MODEL_SCRIPT", "gpt-4o")
MODEL_VISUAL = os.getenv("MODEL_VISUAL", "gpt-4o-mini")
#   ⓪ Select : 콜로케이션+뉘앙스 → 도메인 내 최적 관계쌍 선택 (가벼운 판단) → 하위 모델
MODEL_SELECT = os.getenv("MODEL_SELECT", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "90"))

# 도메인별 세계관 파일 (lore/domains/*.md) — 해당 도메인 것만 주입해 컨텍스트를 줄인다.
DOMAINS_DIR = LORE_DIR / "domains"
_DOMAIN_FILE = {
    "workplace":        "workplace.md",
    "daily":            "daily.md",
    "customer/service": "customer_service.md",
    "academic":         "academic.md",
}
# 도메인별 등장 가능 캐스트 풀 (관계쌍 후보 필터 + 캐릭터 bible 슬라이스에 사용)
_DOMAIN_CAST = {
    "workplace":        {"hanyoil", "ru-ha", "hanyuyeon", "so-ae"},
    "daily":            {"hanyoil", "so-ae", "hyo-jeong"},
    "customer/service": {"hanyoil", "so-ae", "hyo-jeong"},
    "academic":         {"hanyoil", "so-ae", "hyo-jeong"},
}
_KNOWN_CHARS = {"hanyoil", "ru-ha", "so-ae", "hanyuyeon", "hyo-jeong"}
_REQUIRED_SENTENCE_METADATA = [
    "primary_used_in",
    "used_in",
    "speaker_role",
    "listener_role",
    "relationship",
    "power_dynamic",
    "speech_act",
    "politeness",
    "story_function",
    "character_fit",
    "avoid_with",
]

# 기계 검증 실패 시 LLM 비평 없이 재생성하는 최대 횟수 (0 = 로그만, 재생성 안 함)
MECH_MAX_RETRIES = int(os.getenv("MECH_MAX_RETRIES", "1"))
PLAN_CHECK_RETRIES = int(os.getenv("PLAN_CHECK_RETRIES", "2"))
SCRIPT_CHECK_RETRIES = int(os.getenv("SCRIPT_CHECK_RETRIES", "2"))
VISUAL_CHECK_RETRIES = int(os.getenv("VISUAL_CHECK_RETRIES", "1"))

# verify 게이트(mechanical/domain/nuance 검증 + 재생성) 사용 여부.
# 기본 OFF — 자주 헛다리 잡고 비싼 재생성을 유발해서, 품질은 프롬프트 강화 + 결정적
# sanitizer 로 앞단에서 보장한다. 되살리려면 VERIFY_ENABLED=1.
VERIFY_ENABLED = os.getenv("VERIFY_ENABLED", "0") == "1"

_PHASE_LABEL = {1: "현상유지", 2: "균열/긴장", 3: "전환", 4: "가까워짐"}

# GPT가 반환하는 캐릭터 이름 변형 → CHARS 키 정규화
_CHAR_CANONICAL: dict[str, str] = {
    "hanyoil":    "hanyoil",
    "ru-ha":      "ru-ha",  "ruha":       "ru-ha",  "ru_ha":      "ru-ha",  "ru ha":      "ru-ha",
    "so-ae":      "so-ae",  "soae":       "so-ae",  "so_ae":      "so-ae",  "so ae":      "so-ae",
    "hanyuyeon":  "hanyuyeon", "han yuyeon": "hanyuyeon", "han_yuyeon": "hanyuyeon",
    "hyo-jeong":  "hyo-jeong", "hyojeong":  "hyo-jeong", "hyo_jeong": "hyo-jeong", "hyo jeong": "hyo-jeong",
}


def _canonical_char(name: str, valid_chars: set[str]) -> str:
    key = name.lower().strip()
    canonical = _CHAR_CANONICAL.get(key, key)
    if canonical not in valid_chars:
        print(f"  ⚠️ 알 수 없는 캐릭터 '{name}' → 'hanyoil'로 대체")
        return "hanyoil"
    return canonical


def _first_domain_from_used_in(value) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw.split(",", 1)[0].strip()
        if isinstance(parsed, list):
            return str(parsed[0]).strip() if parsed else ""
    return ""


def get_primary_domain(word: dict | None) -> str:
    """Return the normalized primary domain from old or new sentence schemas."""
    word = word or {}
    primary = str(word.get("primary_used_in") or "").strip()
    if primary:
        return primary
    legacy = str(word.get("used in") or "").strip()
    if legacy:
        return legacy
    return _first_domain_from_used_in(word.get("used_in"))


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _metadata_is_missing(word: dict) -> bool:
    return any(not str(word.get(key) or "").strip() for key in _REQUIRED_SENTENCE_METADATA if key != "avoid_with")


def _backfill_sentence_metadata(word: dict) -> dict:
    """Fill new sentence metadata for old CSV rows without overwriting native JSON rows.

    This is a compatibility shim for existing sentence files created before
    SPEAKING_SENTENCE_PROMPT emitted relationship metadata. Fresh JSON-generated
    rows pass through unchanged and remain the source of truth.
    """
    out = dict(word)
    sentence = str(out.get("collocation unit") or out.get("sentence_unit") or out.get("sentence unit") or "").strip()
    situation = str(out.get("micro_situation") or out.get("micro situation") or out.get("nuance (Korean)") or "").strip()
    domain = get_primary_domain(out) or "daily"
    text = f"{sentence} {situation}".lower()

    defaults = {
        "speaker_role": "friend",
        "listener_role": "friend",
        "relationship": "friend_to_friend",
        "power_dynamic": "equal",
        "speech_act": "small_talk",
        "politeness": "softened",
        "story_function": "reveals_emotion",
        "character_fit": ["hanyoil", "ru-ha"],
        "avoid_with": [],
    }

    if domain == "customer/service":
        defaults.update({
            "speaker_role": "staff",
            "listener_role": "customer",
            "relationship": "staff_to_customer",
            "power_dynamic": "service_to_customer",
            "speech_act": "offer",
            "politeness": "polite",
            "story_function": "softens_conflict",
            "character_fit": ["hyo-jeong", "hanyoil"],
            "avoid_with": ["not_customer_to_staff", "staff_must_be_hyo-jeong"],
        })
    elif domain == "academic":
        defaults.update({
            "speaker_role": "student",
            "listener_role": "professor",
            "relationship": "student_to_professor",
            "power_dynamic": "upward",
            "speech_act": "clarification",
            "politeness": "polite",
            "story_function": "buys_time",
            "character_fit": ["so-ae", "hyo-jeong"],
            "avoid_with": ["too_casual_for_professor"],
        })
    elif domain == "workplace":
        defaults.update({
            "speaker_role": "coworker",
            "listener_role": "coworker",
            "relationship": "coworker_to_coworker",
            "power_dynamic": "equal",
            "speech_act": "update_status",
            "politeness": "softened",
            "story_function": "buys_time",
            "character_fit": ["hanyoil", "ru-ha"],
            "avoid_with": ["too_casual_for_boss"],
        })

    if "appreciate" in text or "thank" in text or "고마" in situation:
        defaults.update({
            "speaker_role": "employee" if domain == "workplace" else "friend",
            "listener_role": "boss" if domain == "workplace" else "friend",
            "relationship": "employee_to_boss" if domain == "workplace" else "friend_to_friend",
            "power_dynamic": "upward" if domain == "workplace" else "equal",
            "speech_act": "thanks",
            "politeness": "polite",
            "story_function": "softens_conflict",
            "character_fit": ["hanyoil", "hanyuyeon"] if domain == "workplace" else ["hanyoil", "hyo-jeong"],
        })
    elif "repeat" in text or "다시 한 번" in situation:
        defaults.update({
            "speaker_role": "employee" if domain == "workplace" else "student",
            "listener_role": "boss" if domain == "workplace" else "professor",
            "relationship": "employee_to_boss" if domain == "workplace" else "student_to_professor",
            "power_dynamic": "upward",
            "speech_act": "clarification",
            "politeness": "polite",
            "story_function": "buys_time",
            "character_fit": ["hanyoil", "hanyuyeon"] if domain == "workplace" else ["so-ae", "hyo-jeong"],
        })
    elif "running late" in text or "늦" in situation:
        defaults.update({
            "speech_act": "update_status",
            "story_function": "starts_conflict",
            "character_fit": ["hanyoil", "ru-ha"],
        })
    elif "interesting" in text or "glad to hear" in text or "재미" in situation or "좋은 소식" in situation:
        defaults.update({
            "speech_act": "small_talk" if "interesting" in text else "reassurance",
            "story_function": "reveals_emotion",
            "character_fit": ["ru-ha", "hyo-jeong"],
        })
    elif "check" in text or "get back" in text or "알려" in situation or "답변" in situation:
        defaults.update({
            "speech_act": "delay_answer" if "get back" in text else "update_status",
            "story_function": "buys_time",
            "character_fit": ["hanyoil", "so-ae"] if domain == "workplace" else ["hanyoil", "hyo-jeong"],
        })
    elif "not sure" in text or "get your point" in text or "확신" in situation or "납득" in situation:
        defaults.update({
            "speech_act": "disagreement" if "not sure" in text else "agreement",
            "story_function": "escalates_conflict" if "not sure" in text else "softens_conflict",
            "character_fit": ["hanyoil", "so-ae"] if domain == "workplace" else ["hanyoil", "hyo-jeong"],
        })
    elif "talk later" in text or "busy" in text or "go now" in text or "바쁠" in situation or "가야" in situation:
        defaults.update({
            "speech_act": "boundary_setting",
            "story_function": "resolves_conflict" if domain == "daily" else "softens_conflict",
            "character_fit": ["hanyoil", "ru-ha"] if domain == "workplace" else ["hanyoil", "hyo-jeong"],
        })
    elif "free tomorrow" in text or "catch up" in text or "약속" in situation or "다시 만나" in situation:
        defaults.update({
            "speech_act": "invitation",
            "story_function": "starts_conflict",
            "character_fit": ["ru-ha", "hyo-jeong"],
        })
    elif "up to you" in text:
        defaults.update({
            "speech_act": "suggestion",
            "story_function": "resolves_conflict",
            "character_fit": ["hanyoil", "hyo-jeong"],
        })
    elif "quick call" in text:
        defaults.update({
            "speech_act": "delay_answer",
            "story_function": "buys_time",
            "character_fit": ["hanyoil", "ru-ha"],
        })
    elif "handle it" in text:
        defaults.update({
            "speech_act": "offer",
            "story_function": "resolves_conflict",
            "character_fit": ["hanyoil", "ru-ha"],
        })
    elif "deadline" in text:
        defaults.update({
            "speaker_role": "employee",
            "listener_role": "boss",
            "relationship": "employee_to_boss",
            "power_dynamic": "upward",
            "speech_act": "request",
            "politeness": "polite",
            "story_function": "starts_conflict",
            "character_fit": ["hanyoil", "hanyuyeon"],
        })

    out.setdefault("primary_used_in", domain)
    if not _as_list(out.get("used_in")):
        out["used_in"] = [domain]
    for key, value in defaults.items():
        if key in {"character_fit", "avoid_with"}:
            if not _as_list(out.get(key)):
                out[key] = value
        elif not str(out.get(key) or "").strip():
            out[key] = value
    out["_metadata_source"] = "native" if not _metadata_is_missing(word) else "backfilled"
    out.setdefault("scenario metadata", {})
    meta = out["scenario metadata"] if isinstance(out["scenario metadata"], dict) else {}
    meta.update({
        "relationship context": out.get("relationship", ""),
        "speaker role": out.get("speaker_role", ""),
        "listener role": out.get("listener_role", ""),
        "power dynamic": out.get("power_dynamic", ""),
        "speech act": out.get("speech_act", ""),
        "service direction": out.get("relationship", ""),
        "story function": out.get("story_function", ""),
        "politeness": out.get("politeness", ""),
        "character_fit": json.dumps(_as_list(out.get("character_fit")), ensure_ascii=False),
        "avoid_with": json.dumps(_as_list(out.get("avoid_with")), ensure_ascii=False),
    })
    out["scenario metadata"] = meta
    return out


def _as_comic_word(row: dict) -> dict:
    """Normalize speaking sentence rows for the copied comic engine."""
    sentence = str(row.get("sentence_unit") or row.get("sentence unit") or row.get("collocation unit") or "").strip()
    trigger = str(row.get("korean_trigger") or row.get("Korean trigger") or row.get("meaning") or "").strip()
    situation = str(row.get("micro_situation") or row.get("micro situation") or row.get("nuance (Korean)") or "").strip()
    primary_used_in = get_primary_domain(row)

    adapted = dict(row)
    if sentence:
        adapted["collocation unit"] = sentence
        adapted.setdefault("sentence_unit", sentence)
    adapted.setdefault("sentence unit", sentence)
    adapted.setdefault("Korean trigger", trigger)
    adapted.setdefault("korean_trigger", trigger)
    adapted.setdefault("micro situation", situation)
    adapted.setdefault("micro_situation", situation)
    if primary_used_in:
        adapted["used in"] = primary_used_in
        adapted.setdefault("primary_used_in", primary_used_in)
    if not _as_list(adapted.get("used_in")) and primary_used_in:
        adapted["used_in"] = [primary_used_in]
    if trigger:
        adapted["meaning"] = trigger
    if situation or trigger:
        adapted["nuance (Korean)"] = situation or trigger
    adapted.setdefault("translation", trigger)
    adapted.setdefault("example sentence", sentence)
    adapted.setdefault("register", str(row.get("register", "")).strip())
    return _backfill_sentence_metadata(adapted)


# 플래너가 outfit_setting 으로 고를 수 있는 유효 접두어 (characters.yaml 공통 버킷)
_VALID_OUTFIT_SETTINGS = {
    "workplace", "academic",
    "daily_home", "daily_convenience", "daily_outing", "daily_dressup", "daily_sport",
}

def _resolve_outfit_setting(outfit_setting: str, domain: str, location: str | None = None) -> str:
    """장면의 의상 setting 접두어(workplace/daily_sport 등)를 정한다 — 한 화 전체 공통."""
    setting = (outfit_setting or "").strip().lower()
    dom = (domain or "").strip().lower()
    loc_setting = resolve_location_outfit_setting(location, dom)
    if dom == "workplace":
        return "workplace"
    if dom == "academic":
        return "academic"
    # customer/service: 손님은 '외출 상태'(가게/카운터)다. daily_home/sport/dressup 은 부적절 →
    # outing/convenience 로만 허용 (예: 손님이 집옷 daily_home 으로 나오던 문제 방지).
    if dom == "customer/service":
        if loc_setting in {"daily_convenience", "daily_outing", "daily_dressup"}:
            return "daily_convenience" if loc_setting == "daily_convenience" else "daily_outing"
        return setting if setting in {"daily_outing", "daily_convenience"} else "daily_outing"
    if dom == "daily" and loc_setting:
        if loc_setting == "daily_dressup" and setting == "daily_outing":
            return setting
        return loc_setting
    if setting not in _VALID_OUTFIT_SETTINGS:
        setting = CATEGORY_DEFAULT_SETTING.get(dom, "daily_outing")
    return setting


def _fixed_scene_background(plan: dict) -> str:
    """한 화 전체에 쓸 짧은 배경 태그 = "장소, 소품 1개".

    locations.yaml 단일 소스. planner 가 고른 location tag 를 그대로 해석하고,
    모르는 값이면 도메인 canonical(화이트리스트 첫 장소)로 폴백한다.
    """
    return resolve_location(plan.get("location"), plan.get("domain"), plan.get("background_prop"))


def _background_for_word(word: dict | None, plan: dict) -> str:
    """Domain-level background override before panel normalization."""
    # Keep the generated scene background tied to the planner's location. Webtoon
    # gutters/strip background are handled later during panel composition.
    return _fixed_scene_background(plan)


def _char_outfit_candidates(char: str, setting: str) -> list[str]:
    """캐릭터 자신의 의상 중 setting 접두어 후보 (없으면 daily_outing → 전체 폴백)."""
    keys = set((load_characters().get(char) or {}).get("outfits", {}).keys())
    cands = sorted(k for k in keys if k.startswith(setting))
    if not cands:
        cands = sorted(k for k in keys if k.startswith("daily_outing")) or sorted(keys)
    return cands


def _pick_char_outfit(char: str, setting: str, seed: int = 0) -> str:
    """캐릭터 자신의 의상 변형을 (seed+char) 결정적으로 고른다 — 캐릭터별 독립 랜덤."""
    cands = _char_outfit_candidates(char, setting)
    return random.Random(f"{seed}-{char}").choice(cands) if cands else "daily_outing_1"




def _sentence_match_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"(?<=[a-z])\?(?=[a-z])", "'", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _is_narration_bubble(text: str) -> bool:
    return bool(re.match(r"^\s*(?:\((?:narration|caption|timecard)\)|\[(?:narration|caption|timecard)\])\s*", text or "", re.I))


def _contains_exact_sentence(text: str, target_sentence: str) -> bool:
    target = _sentence_match_text(target_sentence)
    if not target:
        return False
    haystack = f" {_sentence_match_text(text)} "
    return f" {target} " in haystack


def _exact_sentence_count(text: str, target_sentence: str) -> int:
    target = _sentence_match_text(target_sentence)
    if not target:
        return 0
    haystack = f" {_sentence_match_text(text)} "
    return haystack.count(f" {target} ")


def _extract_example(panels: list[dict], collocation: str) -> tuple[str, str, str]:
    """Return (bubble, character, bubble_kr) for the panel containing the target sentence."""
    for p in reversed(panels):
        if _contains_exact_sentence(p.get("bubble", ""), collocation):
            return p["bubble"], p.get("char") or "_default", p.get("bubble_kr", "")
    print(f"  ⚠️ target sentence '{collocation}' not found in any bubble → using last panel as fallback")
    last = panels[-1]
    return last.get("bubble", ""), last.get("char") or "_default", last.get("bubble_kr", "")


# 콜로케이션 매칭 시 내용어 사이에 끼어들 수 있는 자연스러운 삽입어:
# 한정사(관사/지시/소유) + 목적격 대명사("ask YOU for a favor") — 임의 단어가 아닌 curated 집합.
_GAP_FILLER = (r"(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their|"
               r"me|you|him|us|them|it)")


def _collocation_present(panels: list[dict], collocation: str) -> bool:
    """Target sentence appears verbatim in spoken dialogue, punctuation-insensitive."""
    if not collocation:
        return True
    return any(_contains_exact_sentence(p.get("bubble", ""), collocation) for p in panels if p.get("char"))


def _find_collocation_index(panels: list[dict], collocation: str) -> int:
    """Target sentence panel index, punctuation-insensitive."""
    if collocation:
        for i in range(len(panels) - 1, -1, -1):
            if _contains_exact_sentence(panels[i].get("bubble", ""), collocation):
                return i
    return -1


_BANNED_JARGON_RE = re.compile(
    r"\b(?:deck|roi|kpi|okr|pivot|sync|bandwidth|alignment|stakeholder|leverage|"
    r"circle back|deep dive|touch base|fancam|bridge|bias|stan|algorithm|optimize)\b",
    re.I,
)


def _target_beats(plan: dict) -> list[tuple[int, dict]]:
    out = []
    for i, beat in enumerate(plan.get("beats") or []):
        if bool(beat.get("has_collocation")):
            out.append((i, beat))
    return out


def _plan_validation_issues(plan: dict, word: dict | None, selected_pair: str = "") -> list[str]:
    issues: list[str] = []
    correction = str(plan.get("metadata_correction") or "").strip()
    primary = get_primary_domain(word).strip().lower()
    plan_domain = str(plan.get("domain") or "").strip().lower()
    if primary and plan_domain != primary:
        issues.append(f"plan.domain={plan_domain or 'empty'} must equal primary_used_in={primary}")

    expected = {
        "target_sentence": (word or {}).get("collocation unit") or (word or {}).get("sentence_unit") or "",
        "primary_used_in": primary,
        "speaker_role": _metadata_value(word, "speaker_role", "speaker role"),
        "listener_role": _metadata_value(word, "listener_role", "listener role"),
        "relationship": _metadata_value(word, "relationship", "relationship context"),
        "power_dynamic": _metadata_value(word, "power_dynamic", "power dynamic"),
        "speech_act": _metadata_value(word, "speech_act", "speech act"),
        "politeness": _metadata_value(word, "politeness"),
        "story_function": _metadata_value(word, "story_function", "story function"),
    }
    for key, want in expected.items():
        got = str(plan.get(key) or "").strip()
        want = str(want or "").strip()
        if want and got != want and not correction:
            issues.append(f"plan.{key}='{got or 'empty'}' must match word metadata '{want}' or explain metadata_correction")

    loc = str(plan.get("location") or "").strip()
    allowed_locations = set(load_location_domains().get(primary, []) or [])
    if primary and allowed_locations and loc not in allowed_locations:
        issues.append(f"location='{loc}' is not allowed for domain '{primary}'")

    allowed_cast = set(_DOMAIN_CAST.get(primary, _KNOWN_CHARS))
    chars = {_canonical_known_char(c) for c in (plan.get("characters") or []) if str(c or "").strip()}
    bad_chars = sorted(c for c in chars if c and c not in allowed_cast)
    if bad_chars:
        issues.append(f"characters outside domain cast for {primary}: {', '.join(bad_chars)}")

    pair = selected_pair or plan.get("selected_pair") or ""
    if selected_pair and str(plan.get("selected_pair") or "").strip() != selected_pair:
        issues.append(f"plan.selected_pair must match preselected pair '{selected_pair}'")
    pair_names = _pair_names(pair)
    if pair_names and not pair_names <= chars:
        issues.append(f"selected_pair characters {sorted(pair_names)} not included in plan.characters {sorted(chars)}")

    target_speaker = _canonical_known_char(plan.get("target_speaker"))
    target_listener = _canonical_known_char(plan.get("target_listener"))
    delivery_mode = str(plan.get("delivery_mode") or "").strip().lower()
    if pair_names:
        if target_speaker not in pair_names:
            issues.append(f"target_speaker='{target_speaker or 'empty'}' must be one of selected_pair characters {sorted(pair_names)}")
        if target_listener:
            if target_listener not in pair_names:
                issues.append(f"target_listener='{target_listener}' must be one of selected_pair characters {sorted(pair_names)}")
            elif target_speaker and target_listener == target_speaker and delivery_mode not in {"action-led", "mixed"}:
                issues.append("target_listener should be the other selected_pair character unless monologue/action-led")

    relationship = _metadata_value(word, "relationship", "relationship context")
    speaker_role = _metadata_value(word, "speaker_role", "speaker role")
    listener_role = _metadata_value(word, "listener_role", "listener role")
    if relationship == "staff_to_customer" and (speaker_role != "staff" or listener_role != "customer"):
        issues.append("relationship=staff_to_customer requires speaker_role=staff and listener_role=customer")
    if relationship == "customer_to_staff" and (speaker_role != "customer" or listener_role != "staff"):
        issues.append("relationship=customer_to_staff requires speaker_role=customer and listener_role=staff")
    if primary == "customer/service":
        service_roles = resolve_service_roles(word, pair)
        staff_char = service_roles.get("staff", "")
        if relationship in {"staff_to_customer", "customer_to_staff"} and staff_char != "hyo-jeong":
            issues.append("customer/service staff relationship must map hyo-jeong to the staff side")
        if service_roles.get("target_speaker") and target_speaker != service_roles["target_speaker"]:
            issues.append(f"target_speaker must be service {relationship} speaker '{service_roles['target_speaker']}'")
        if service_roles.get("target_listener") and target_listener != service_roles["target_listener"]:
            issues.append(f"target_listener must be service {relationship} listener '{service_roles['target_listener']}'")

    targets = _target_beats(plan)
    beat_count = len(plan.get("beats") or [])
    if beat_count != 6:
        issues.append(f"plan must have exactly 6 beats, got {beat_count}")
    if len(targets) != 1:
        issues.append(f"exactly one beat must have has_collocation=true, got {len(targets)}")
    elif (str(targets[0][1].get("panel_type") or "").strip().lower() == "object"
          or not str(targets[0][1].get("speaker") or "").strip()):
        issues.append("has_collocation beat must be a character panel with a speaker")

    proof = plan.get("visible_proof_panel")
    if proof not in (None, "", "null"):
        by_no = {
            int(b.get("panel")): b
            for b in plan.get("beats") or []
            if str(b.get("panel", "")).isdigit()
        }
        try:
            proof_beat = by_no[int(proof)]
        except (TypeError, ValueError, KeyError):
            issues.append(f"visible_proof_panel={proof} does not point to a valid beat")
        else:
            if (str(proof_beat.get("panel_type") or "").strip().lower() != "object"
                    or str(proof_beat.get("speaker") or "").strip()):
                issues.append("visible_proof_panel must point to an object panel with no speaker")
    return issues


def _script_validation_issues(script_panels: list[dict], plan: dict, word: dict | None) -> list[str]:
    issues: list[str] = []
    beats = plan.get("beats") or []
    target = (
        (word or {}).get("collocation unit")
        or (word or {}).get("sentence_unit")
        or (word or {}).get("sentence unit")
        or ""
    )
    if len(script_panels) != len(beats):
        issues.append(f"script panel count {len(script_panels)} != plan beats {len(beats)}")

    target_beats = _target_beats(plan)
    marked_idx = target_beats[0][0] if len(target_beats) == 1 else -1
    marked_speaker = str(
        (plan.get("target_speaker") or (target_beats[0][1].get("speaker") if len(target_beats) == 1 else ""))
        or ""
    ).strip().lower()

    found_indices = []
    target_count = 0
    for i, panel in enumerate(script_panels):
        beat = beats[i] if i < len(beats) else {}
        char = str(panel.get("char") or "").strip()
        bubble = str(panel.get("bubble") or "").strip()
        bubble_kr = str(panel.get("bubble_kr") or "").strip()
        is_object = str(beat.get("panel_type") or "").strip().lower() == "object" or not str(beat.get("speaker") or "").strip()

        object_has_allowed_narration = is_object and not char and _is_narration_bubble(bubble) and not bubble_kr
        if is_object and (char or bubble or bubble_kr) and not object_has_allowed_narration:
            issues.append(f"panel {i + 1}: object panel must have empty char/bubble/bubble_kr")
        count_here = _exact_sentence_count(bubble, target) if target else 0
        if count_here:
            found_indices.append(i)
            target_count += count_here
        if bubble:
            words = bubble.split()
            splits = [part for part in re.split(r"[.!?]+", bubble) if part.strip()]
            # 한 버블 = 한 비트가 목표지만, 짧고 자연스러운 되묻기/감탄
            # ("Really? A penguin?", "Oh no! I'm late.")은 학습자에게도 부담이 아니다.
            # 진짜 문제는 (a) 긴 줄, (b) 두 문장 '이상'을 욱여넣은 경우뿐.
            too_long = len(words) > 16
            crammed = len(splits) >= 3 and len(words) > 12
            if too_long or crammed:
                issues.append(f"panel {i + 1}: bubble too complex ({len(words)} words, {len(splits)} punctuation splits)")
            if _BANNED_JARGON_RE.search(bubble):
                issues.append(f"panel {i + 1}: banned jargon in bubble")

    if target:
        if target_count != 1:
            issues.append(f"target sentence must appear exactly once, got {target_count}")
        else:
            if marked_idx >= 0 and found_indices[0] != marked_idx:
                issues.append(f"target sentence appears in panel {found_indices[0] + 1}, not marked beat {marked_idx + 1}")
            found_panel = script_panels[found_indices[0]]
            found_speaker = _canonical_known_char(found_panel.get("char"))
            planned_speaker = _canonical_known_char(marked_speaker)
            if planned_speaker and found_speaker != planned_speaker:
                issues.append(f"target sentence speaker {found_speaker} != planned speaker {planned_speaker}")
    return issues


def _visual_validation_issues(visual_panels: list[dict], script_panels: list[dict], plan: dict) -> list[str]:
    issues: list[str] = []
    if len(visual_panels) != len(script_panels):
        issues.append(f"visual panel count {len(visual_panels)} != script panel count {len(script_panels)}")
    for i, sp in enumerate(script_panels[:len(visual_panels)]):
        vp = visual_panels[i]
        is_object = not str(sp.get("char") or "").strip()
        has_motion = str(vp.get("action") or "").strip() or str(vp.get("body_pose") or "").strip()
        if not has_motion:
            issues.append(f"visual panel {i + 1}: missing action")
        if not is_object and not str(vp.get("expression") or "").strip():
            issues.append(f"visual panel {i + 1}: missing expression for character panel")
    return issues


def _deterministic_feedback(stage: str, issues: list[str]) -> str:
    return f"Deterministic {stage} validation failed. Fix these exactly:\n" + "\n".join(f"- {m}" for m in issues)


def _hanyoil_demeanor() -> str:
    """복습 카드 비주얼 프롬프트용 — 주인공 한요일의 baseline 표정(characters.yaml)."""
    base = (load_characters().get("hanyoil") or {}).get("expression", "")
    return f"- hanyoil: {base}" if base else ""


def _design_review_card_visual(word: dict | None) -> dict | None:
    """표현의 뉘앙스만 보고 복습 카드 1장의 비주얼을 GPT 로 새로 설계.

    GPT 가 mode 를 고른다: "character"(한요일 포즈·표정·소품) | "object"(사람 없는 사물/장면).
    만화 패널과 무관 — '인출 단서'(키워드법)로서 표현을 가장 잘 떠올리게 하는 한 장을 만든다.
    실패하면 None 을 돌려주고, 호출부가 만화 패널 재사용으로 폴백한다.
    """
    if not word:
        return None
    try:
        prompt = build_review_card_prompt(
            build_word_block(word),
            word.get("collocation unit", ""),
            expression_menu=build_expression_menu(),
            char_demeanor=_hanyoil_demeanor(),
        )
        vis = _gpt_json(prompt, "Design the review card now. Output the JSON only.", model=MODEL_VISUAL)
        # action 태그 정제(감정/호흡 태그 제거; object 면 사람 동작 태그도 제거) — 만화 새니타이저 재사용.
        is_object = (vis.get("mode") or "").strip().lower() == "object"
        clean = _sanitize_visual_actions([vis], [{"char": "" if is_object else "hanyoil"}])[0]
        vis["action"] = clean.get("action") or ("on table" if is_object else "standing")
        return vis
    except Exception as exc:
        print(f"  ⚠️ 복습 카드 비주얼 설계 실패({exc}) — 만화 패널 재사용으로 폴백")
        return None


_REVIEW_TEXT_PROP_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\b(certificate|diploma)\b", "medal, star badge"),
    (r"\b(document|contract|report|form|resume|application)\b", "blank folder, blank paper"),
    (r"\b(sign|label|poster|banner)\b", "arrow symbol"),
    (r"\b(chart|graph)\b", "simple bar chart, checkmarks"),
    (r"\b(checklist)\b", "clipboard with checkmarks"),
    (r"\b(receipt|invoice|bill)\b", "coins, blank paper"),
    (r"\b(screen with text|phone screen with text|computer screen with text)\b", "blank screen"),
)


def _safe_review_visual_tags(tags: str) -> str:
    """복습 카드에서 글자 렌더를 유발하는 소품을 텍스트 없는 상징으로 치환."""
    out = (tags or "").strip().replace("_", " ")
    for pattern, repl in _REVIEW_TEXT_PROP_REPLACEMENTS:
        out = re.sub(pattern, repl, out, flags=re.I)
    # GPT가 지시 문구를 태그처럼 섞는 경우를 방어한다.
    out = re.sub(r"\bwith labels?\b", "", out, flags=re.I)
    out = re.sub(r"\b(with )?(readable )?text\b", "", out, flags=re.I)
    out = re.sub(r"\bletters?\b|\bnumbers?\b|\blogo\b|\blabels?\b", "", out, flags=re.I)
    return re.sub(r"\s*,\s*,+", ",", out).strip(" ,")


def build_review_card(word: dict | None, plan: dict | None, panels: list[dict], seed: int) -> dict | None:
    """복습용 단일 카드(정사각 썸네일) 1장의 패널 설정을 만든다.

    학습 설계: 만화는 '이해'(왜 이 표현이 이 상황에 맞나), 단일 카드는 '인출 단서'(키워드법).
    이 카드는 만화 패널을 재사용하지 않고 **전용 GPT 호출로 뉘앙스를 새로 설계**한다 — 우연히
    그 패널에 있던 포즈가 아니라, 표현을 가장 잘 떠올리게 하는 한 장을 GPT 가 만든다.
    GPT 가 mode 를 고른다:
      · "character" → 항상 주인공 한요일(hanyoil). 복장/헤어는 만화의 한요일 패널 그대로(만화와
                      동일), 포즈·표정·소품으로 뉘앙스 표현. 한요일 패널이 없으면 복장은 렌더가 폴백.
      · "object"    → 사람 없는 사물/장면 1컷(heavy traffic → 꽉 막힌 도로 등). 복장 상속 불필요.
    만화 장면과의 '연관/연결'은 불필요(뉘앙스만 담으면 됨). GPT 설계가 실패하면 만화의 한요일
    패널 포즈를 재사용해 폴백한다(이때는 항상 character).
    """
    if not panels:
        return None
    collocation = (word or {}).get("collocation unit", "")

    def _is_hanyoil(p: dict) -> bool:
        return (p.get("char") or "").strip().lower() == "hanyoil"

    # 복장/헤어 상속용 소스 패널: 콜로케이션이 나온 한요일 패널 > 첫 한요일 패널.
    idx = _find_collocation_index(panels, collocation)
    src = (panels[idx] if (idx >= 0 and _is_hanyoil(panels[idx])) else None) \
        or next((p for p in panels if _is_hanyoil(p)), None)

    vis = _design_review_card_visual(word) or {}
    is_object = (vis.get("mode") or "").strip().lower() == "object"

    if is_object:
        # 사람 없는 사물/장면 카드 — 렌더의 object 분기가 흰 배경에 subject+action 만 그린다.
        subject = _safe_review_visual_tags(vis.get("subject") or "")
        action = _safe_review_visual_tags(vis.get("action") or "") or "on table"
        card = {
            "panel_type": "object",
            "char":       "",
            "outfit":     "",
            "subject":    subject,
            "action":     action,
            "expression": "",
            "face_state": "",
            "bubble":     "",
            "bubble_kr":  "",
            "seed_offset": 0,
        }
    else:
        outfit = (src or {}).get("outfit", "")  # 비면 렌더 단계가 daily_outing_1 로 폴백.
        hair_ov = (src or {}).get("hair_override")
        action = vis.get("action") or (src or {}).get("action") or "standing"
        expr_key = vis.get("expression") or "serious"
        face = (vis.get("face_state") or "").strip() or "looking at viewer"
        # props 는 Danbooru 태그(공백 구분) — GPT 가 가끔 넣는 언더스코어를 공백으로 정규화.
        props = _safe_review_visual_tags(vis.get("props") or "")
        card = {
            "panel_type": "character",
            "char":       "hanyoil",
            "outfit":     outfit,
            "subject":    "",
            "action":     action,
            "expression": resolve_expression(_safe_expression_key(expr_key)),
            "face_state": face,
            "bubble":     "",   # 복습 카드엔 말풍선/텍스트 없음 (이미지만)
            "bubble_kr":  "",
            "seed_offset": 0,
        }
        if hair_ov:
            card["hair_override"] = hair_ov
        if props:
            card["props_extra"] = props

    return {
        "no":          str((word or {}).get("No.", "")),
        "collocation": collocation,
        "seed":        seed,
        "panel":       card,
    }


# ─────────────────────────────────────────────────────────
# Relationship Arc State — relationship_state.yaml 파싱 & 업데이트
# ─────────────────────────────────────────────────────────

def _parse_arc_state(lore_text: str | None = None) -> dict:
    """relationship_state.yaml 을 파싱. (lore_text 인자는 하위호환용 — 무시)

    Returns: {"hanyoil ↔ ru-ha": {"phase": 1, "last_beat": None, "running_gag": None}, ...}
    """
    if not ARC_STATE_PATH.exists():
        return {}
    data = yaml.safe_load(ARC_STATE_PATH.read_text(encoding="utf-8")) or {}
    rels = data.get("relationships", data)  # 중첩 구조 우선, 없으면 평면(하위호환)
    result = {}
    for pair, st in rels.items():
        if not isinstance(st, dict):
            continue
        result[pair] = {
            "phase":         st.get("phase", 1),
            "comfort_level": st.get("comfort_level"),
            "last_beat":     st.get("last_beat"),
            "running_gag":   st.get("running_gag"),
            "signature_bit": st.get("signature_bit"),
            "dynamic":       st.get("dynamic") or [],
            "unresolved":    st.get("unresolved") or [],
        }
    return result


def _pair_names(pair: str) -> set[str]:
    """'hanyoil ↔ ru-ha' → {'hanyoil','ru-ha'} (구분자 ↔ 기준)."""
    return {n.strip().lower() for n in re.split(r"↔|<->|->|/", pair) if n.strip()}


def domain_pairs(domain: str, arc_state: dict | None = None) -> dict:
    """해당 도메인 캐스트 풀 안의 관계쌍만 추린다 (양쪽 인물이 모두 풀에 있는 쌍)."""
    dom = (domain or "").strip().lower()
    pool = _DOMAIN_CAST.get(dom)
    arc = arc_state if arc_state is not None else _parse_arc_state()
    if not pool:
        return arc
    out = {p: st for p, st in arc.items() if _pair_names(p) <= pool}
    return out


_METADATA_FIELDS = [
    "relationship context",
    "speaker role",
    "listener role",
    "power dynamic",
    "speech act",
    "service direction",
    "story function",
    "politeness",
    "character_fit",
    "avoid_with",
]

_METADATA_ALIASES = {
    "relationship context": ("relationship", "relationship context"),
    "speaker role": ("speaker_role", "speaker role"),
    "listener role": ("listener_role", "listener role"),
    "power dynamic": ("power_dynamic", "power dynamic"),
    "speech act": ("speech_act", "speech act"),
    "service direction": ("relationship", "service_direction", "service direction"),
    "story function": ("story_function", "story function"),
    "politeness": ("politeness",),
    "character_fit": ("character_fit",),
    "avoid_with": ("avoid_with",),
}


def _sentence_metadata(word: dict | None) -> dict[str, str]:
    """Return scenario-facing metadata from a speaking row, tolerating older CSVs."""
    word = word or {}
    nested = word.get("scenario metadata") or {}
    out = {}
    for field in _METADATA_FIELDS:
        value = ""
        for key in _METADATA_ALIASES.get(field, (field,)):
            value = word.get(key) or nested.get(key) or ""
            if str(value).strip():
                break
        out[field] = str(value).strip()
    return out


def _format_metadata(meta: dict[str, str]) -> str:
    lines = [f"- {field}: {meta.get(field) or '(infer from sentence)'}" for field in _METADATA_FIELDS]
    return "\n".join(lines)


def _metadata_value(word: dict | None, *keys: str) -> str:
    word = word or {}
    nested = word.get("scenario metadata") or {}
    for key in keys:
        value = word.get(key)
        if value is None:
            value = nested.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _metadata_list(word: dict | None, *keys: str) -> list[str]:
    word = word or {}
    nested = word.get("scenario metadata") or {}
    raw = None
    for key in keys:
        value = word.get(key)
        if value is None:
            value = nested.get(key)
        if isinstance(value, list) or str(value or "").strip():
            raw = value
            break
    if not raw:
        return []
    if isinstance(raw, list):
        parsed = raw
    else:
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            parsed = [part.strip().strip("'\"[] ") for part in str(raw).split(",") if part.strip()]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        name = str(item).strip()
        if not name:
            continue
        out.append(_CHAR_CANONICAL.get(name.lower(), name))
    return [name for name in out if name in _KNOWN_CHARS]


def _base_candidate_pairs(primary_domain: str, relationship: str) -> dict:
    arc = _parse_arc_state()
    dom = (primary_domain or "").strip().lower()
    pool = set(_DOMAIN_CAST.get(dom, _KNOWN_CHARS))
    rel = (relationship or "").strip()

    if rel == "friend_to_friend" and not dom:
        pool |= {"hanyoil", "ru-ha", "hyo-jeong", "so-ae"}
    if rel in {"staff_to_customer", "customer_to_staff"}:
        pool |= {"hyo-jeong", "hanyoil", "so-ae"}
    if rel in {"employee_to_boss", "boss_to_employee", "coworker_to_coworker"}:
        pool |= {"hanyoil", "ru-ha", "hanyuyeon", "so-ae"}

    return {p: st for p, st in arc.items() if _pair_names(p) <= pool}


def _pair_score(pair: str, word: dict | None, primary_domain: str, ignore_fit: bool = False) -> tuple[int, list[str]]:
    names = _pair_names(pair)
    relationship = _metadata_value(word, "relationship", "relationship context")
    speaker_role = _metadata_value(word, "speaker_role", "speaker role")
    listener_role = _metadata_value(word, "listener_role", "listener role")
    power_dynamic = _metadata_value(word, "power_dynamic", "power dynamic")
    character_fit = set() if ignore_fit else set(_metadata_list(word, "character_fit"))
    score = 0
    reasons: list[str] = []

    domain_pool = _DOMAIN_CAST.get((primary_domain or "").strip().lower())
    if domain_pool and names <= domain_pool:
        score += 2
        reasons.append("domain cast")

    fit_hits = names & character_fit
    if fit_hits:
        score += 6 * len(fit_hits)
        reasons.append("character_fit: " + ", ".join(sorted(fit_hits)))

    if relationship in {"staff_to_customer", "customer_to_staff"} and "hyo-jeong" in names:
        score += 10
        reasons.append("hyo-jeong as service staff")
    if relationship == "employee_to_boss" and "hanyuyeon" in names:
        score += 10
        reasons.append("hanyuyeon as boss")
    if relationship == "boss_to_employee" and "hanyuyeon" in names:
        score += 8
        reasons.append("hanyuyeon as boss")
    if relationship == "coworker_to_coworker":
        if names == {"hanyoil", "ru-ha"}:
            score += 7
            reasons.append("default coworker peers: hanyoil-ru-ha")
        if names == {"hanyoil", "so-ae"} and ("so-ae" in character_fit or "employee" in {speaker_role, listener_role}):
            score += 7
            reasons.append("detail/cautious coworker fit: hanyoil-so-ae")
    if relationship == "friend_to_friend" and names & {"hyo-jeong", "ru-ha"}:
        score += 5
        reasons.append("friend pool includes hyo-jeong/ru-ha")

    if power_dynamic == "upward" and "hanyuyeon" in names:
        score += 4
        reasons.append("upward hierarchy")
    if power_dynamic == "downward" and "hanyuyeon" in names:
        score += 4
        reasons.append("downward hierarchy")
    if power_dynamic in {"service_to_customer", "customer_to_service"} and "hyo-jeong" in names:
        score += 4
        reasons.append("service hierarchy")

    return score, reasons


def _filter_relationship_candidates(
    word: dict | None,
    primary_domain: str,
    ignore_fit: bool = False,
    narrow_top: bool = True,
) -> tuple[dict, dict[str, list[str]]]:
    relationship = _metadata_value(word, "relationship", "relationship context")
    speaker_role = _metadata_value(word, "speaker_role", "speaker role")
    listener_role = _metadata_value(word, "listener_role", "listener role")
    power_dynamic = _metadata_value(word, "power_dynamic", "power dynamic")
    # ignore_fit: 쿼터 재균형 시 character_fit(=확률적 힌트)을 무시해 더 넓은 후보를 얻는다.
    character_fit = set() if ignore_fit else set(_metadata_list(word, "character_fit"))
    cands = _base_candidate_pairs(primary_domain, relationship)

    def keep(pair: str) -> bool:
        names = _pair_names(pair)
        if character_fit and not (names & character_fit):
            return False
        if relationship in {"staff_to_customer", "customer_to_staff"}:
            return "hyo-jeong" in names
        if relationship == "employee_to_boss":
            return "hanyuyeon" in names and bool(names & {"hanyoil", "ru-ha", "so-ae"})
        if relationship == "boss_to_employee":
            return "hanyuyeon" in names and bool(names & {"hanyoil", "ru-ha", "so-ae"})
        if relationship == "coworker_to_coworker":
            preferred = [{"hanyoil", "ru-ha"}, {"hanyoil", "so-ae"}]
            if character_fit:
                return any(names == pair_names and names & character_fit for pair_names in preferred)
            return names in preferred
        if relationship == "friend_to_friend":
            if character_fit:
                return bool(names & character_fit)
            return bool(names & {"hyo-jeong", "ru-ha"})
        if relationship in {"student_to_professor", "professor_to_student"}:
            return "so-ae" in names or "hyo-jeong" in names
        if "boss" in {speaker_role, listener_role} or power_dynamic in {"upward", "downward"}:
            return "hanyuyeon" in names
        if "staff" in {speaker_role, listener_role} or power_dynamic in {"service_to_customer", "customer_to_service"}:
            return "hyo-jeong" in names
        return True

    filtered = {p: st for p, st in cands.items() if keep(p)}
    if not filtered:
        filtered = cands

    scored: list[tuple[int, str, list[str]]] = []
    for pair in filtered:
        score, reasons = _pair_score(pair, word, primary_domain, ignore_fit=ignore_fit)
        scored.append((score, pair, reasons))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if narrow_top and scored:
        top_score = scored[0][0]
        filtered = {pair: filtered[pair] for score, pair, _ in scored if score == top_score}
    reason_map = {pair: reasons for _, pair, reasons in scored}
    return filtered, reason_map


def _lead_from_metadata(pair: str, word: dict | None) -> str:
    names = _pair_names(pair)
    relationship = _metadata_value(word, "relationship", "relationship context")
    speaker_role = _metadata_value(word, "speaker_role", "speaker role")
    power_dynamic = _metadata_value(word, "power_dynamic", "power dynamic")

    if relationship == "staff_to_customer":
        return "hyo-jeong" if "hyo-jeong" in names else ""
    if relationship == "customer_to_staff":
        return next((name for name in sorted(names) if name != "hyo-jeong"), "hyo-jeong" if "hyo-jeong" in names else "")
    if relationship == "employee_to_boss":
        return next((name for name in sorted(names) if name != "hanyuyeon"), "")
    if relationship == "boss_to_employee":
        return "hanyuyeon" if "hanyuyeon" in names else ""
    if speaker_role in {"boss", "professor"} or power_dynamic == "downward":
        return "hanyuyeon" if "hanyuyeon" in names else ""
    if speaker_role == "staff" or power_dynamic == "service_to_customer":
        return "hyo-jeong" if "hyo-jeong" in names else ""
    if speaker_role == "customer" or power_dynamic == "customer_to_service":
        return next((name for name in sorted(names) if name != "hyo-jeong"), "")
    if speaker_role == "main_character" and "hanyoil" in names:
        return "hanyoil"
    return ""


def resolve_service_roles(word: dict | None, selected_pair: str) -> dict:
    names = sorted(_pair_names(selected_pair))
    relationship = _metadata_value(word, "relationship", "relationship context")
    staff = "hyo-jeong" if "hyo-jeong" in names else ""
    customer = next((name for name in names if name != staff), "")

    target_speaker = ""
    target_listener = ""
    if relationship == "staff_to_customer":
        target_speaker = staff
        target_listener = customer
    elif relationship == "customer_to_staff":
        target_speaker = customer
        target_listener = staff
    elif relationship == "staff_to_coworker":
        target_speaker = staff
        target_listener = next((name for name in names if name != staff), "")

    return {
        "staff": staff,
        "customer": customer,
        "target_speaker": target_speaker,
        "target_listener": target_listener,
    }


def _staff_character_for_scene(pair: str, word: dict | None, domain: str = "") -> str:
    """Return the character who should be treated as staff for service outfit logic."""
    if (domain or get_primary_domain(word)).strip().lower() != "customer/service":
        return ""
    relationship = _metadata_value(word, "relationship", "relationship context")
    if relationship not in {"staff_to_customer", "customer_to_staff", "staff_to_coworker"}:
        return ""
    return resolve_service_roles(word, pair).get("staff", "")


def _apply_cast_quota(
    word: dict | None,
    primary_domain: str,
    cands: dict,
    reason_map: dict[str, list[str]],
    used_chars: list[str] | None,
    used_pairs: list[str] | None,
    total: int,
) -> tuple[dict, dict[str, list[str]]]:
    """배치 쏠림 하드 가드. 이미 한도(supporting ≤45%, 단일 페어 ≤35%)에 닿은 캐릭터/페어가
    들어간 후보를 제거한다. 그러면 후보가 0이 되는 경우(주로 character_fit 핀 고정)엔
    fit 을 완화해 균형 잡힌 대체 후보를 찾는다. 그래도 없으면 원래 후보를 그대로 둔다(정합성 우선)."""
    from collections import Counter
    if not total or not cands:
        return cands, reason_map
    if total < 10:
        return cands, reason_map
    cap_char = int(0.45 * total)   # 이 수 '이상'이면 포화 → 다음 픽 차단 (count==cap 까진 허용)
    cap_pair = int(0.35 * total)
    char_counts = Counter(c for c in (used_chars or []) if c in _SUPPORTING)
    pair_counts = Counter(used_pairs or [])
    saturated_chars = {c for c in _SUPPORTING if char_counts.get(c, 0) >= cap_char}

    def within_quota(pair: str) -> bool:
        if _pair_names(pair) & saturated_chars:
            return False
        if pair_counts.get(pair, 0) >= cap_pair:
            return False
        return True

    balanced = {p: st for p, st in cands.items() if within_quota(p)}
    if balanced:
        return balanced, reason_map

    # 모든 후보가 포화 캐릭터/페어를 포함 → fit 완화로 더 넓게 탐색
    relaxed, relaxed_reasons = _filter_relationship_candidates(
        word, primary_domain, ignore_fit=True, narrow_top=False)
    balanced = {p: st for p, st in relaxed.items() if within_quota(p)}
    if balanced:
        print(f"  ⚖️ 쿼터 재균형: 포화({', '.join(sorted(saturated_chars)) or '페어'}) 회피 → fit 완화")
        merged = {p: (relaxed_reasons.get(p) or []) + ["quota rebalance"] for p in balanced}
        return balanced, {**reason_map, **merged}
    return cands, reason_map


def select_relationship(
    word: dict | None,
    domain: str,
    used_chars: list[str] | None = None,
    used_pairs: list[str] | None = None,
    total: int = 0,
) -> dict:
    """Select a relationship pair from sentence metadata, using LLM only as tie-breaker."""
    primary_domain = get_primary_domain(word) or domain
    meta = _sentence_metadata(word)
    relationship = _metadata_value(word, "relationship", "relationship context")
    speaker_role = _metadata_value(word, "speaker_role", "speaker role")
    listener_role = _metadata_value(word, "listener_role", "listener role")
    speech_act = _metadata_value(word, "speech_act", "speech act")
    story_function = _metadata_value(word, "story_function", "story function")
    cands, reason_map = _filter_relationship_candidates(word, primary_domain)
    cands, reason_map = _apply_cast_quota(
        word, primary_domain, cands, reason_map, used_chars, used_pairs, total)
    base = {
        "pair": "",
        "lead": "",
        "why": "",
        "candidates": cands,
        "speaker_role": speaker_role,
        "listener_role": listener_role,
        "relationship": relationship,
        "speech_act": speech_act,
        "story_function": story_function,
    }
    if not word or not cands:
        return base

    if len(cands) == 1:
        pair = next(iter(cands))
        lead = _lead_from_metadata(pair, word)
        why_bits = reason_map.get(pair) or ["metadata filters left one valid pair"]
        return {
            **base,
            "pair": pair,
            "lead": lead,
            "why": "; ".join(why_bits),
        }

    sentence = word.get("collocation unit", "")
    trigger = word.get("meaning", "")
    situation = word.get("nuance (Korean)", "")
    lines = []
    for p, st in cands.items():
        dyn = " / ".join(st.get("dynamic") or [])
        bit = st.get("signature_bit") or ""
        reasons = "; ".join(reason_map.get(p) or [])
        lines.append(f"- {p}: {dyn}" + (f" | {bit}" if bit else "") + (f" | filter reasons: {reasons}" if reasons else ""))
    menu = "\n".join(lines)

    sys = (
        "You pick the BEST character relationship for a character-driven English speaking "
        "webtoon. Choose ONLY from the filtered candidate pairs. The relationship metadata is "
        "the source of truth: do not contradict speaker_role, listener_role, relationship, "
        "power_dynamic, or speech_act. Return JSON only."
    )
    usr = (
        f'Target sentence: "{sentence}"\nKorean cue: {trigger}\nMicro situation: {situation}\n'
        f"Primary domain: {primary_domain}\nScenario metadata:\n{_format_metadata(meta)}\n\n"
        f"Filtered candidate relationships (pair: recurring dynamic):\n{menu}\n\n"
        'Return: {"pair": "<one pair string exactly as listed>", '
        '"lead": "<the character who speaks or initiates according to speaker_role/relationship>", '
        '"why": "<one sentence: why THIS pair, with this relationship/power/speech-act context, would naturally produce this exact sentence>"}'
    )
    try:
        out = _gpt_json(sys, usr, model=MODEL_SELECT)
    except Exception as exc:
        print(f"  ⚠️ 관계 선택 실패({exc}) — 최고점 후보로 폴백")
        pair = next(iter(cands))
        return {**base, "pair": pair, "lead": _lead_from_metadata(pair, word),
                "why": "; ".join(reason_map.get(pair) or ["top filtered candidate"])}

    pair = (out.get("pair") or "").strip()
    if pair not in cands:  # 모델이 변형/오타를 내면 이름 집합으로 매칭 시도
        want = _pair_names(pair)
        pair = next((p for p in cands if _pair_names(p) == want), "")
    if not pair:
        pair = next(iter(cands))
    lead = (out.get("lead") or "").strip().lower()
    if lead not in _pair_names(pair):
        lead = _lead_from_metadata(pair, word)
    return {
        **base,
        "pair": pair,
        "lead": lead,
        "why": (out.get("why") or "").strip() or "; ".join(reason_map.get(pair) or ["selected from filtered candidates"]),
    }


MILESTONES_PATH = LORE_DIR / "milestones.yaml"  # 굵직한 사건 전용 ledger (옴니버스 + 마일스톤 연속성)
_MILESTONE_CHANGE_RE = re.compile(
    r"\b(phase|comfort|trust|closer|distance|unresolved|tension|running gag|callback|"
    r"rapport|dynamic|relationship|shift|changes?|changed|turning point|opens up|softens)\b",
    re.I,
)
_MILESTONE_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "their", "his", "her", "they", "them", "through", "after", "during", "from",
    "that", "this", "showing", "revealing", "reveals", "realizes", "learns",
}


def _canonical_known_char(name: str | None) -> str:
    key = (name or "").strip().lower()
    return _CHAR_CANONICAL.get(key, key)


def _facet_rows_from_plan(plan: dict | None) -> list[dict]:
    rows = []
    for item in (plan or {}).get("character_filter_collision") or []:
        char = _canonical_known_char(item.get("character"))
        facet = str(item.get("facet") or "").strip()
        if char in _KNOWN_CHARS and facet:
            rows.append({
                "character": char,
                "facet": facet,
                "collision": str(item.get("collision") or "").strip(),
            })
    return rows


def _driver_from_plan(plan: dict | None) -> str:
    game = (plan or {}).get("comedic_game") or {}
    driver = _canonical_known_char(game.get("driver"))
    if driver in _KNOWN_CHARS:
        return driver
    return ""


def _load_facet_state() -> dict:
    if not FACET_STATE_PATH.exists():
        return {"usages": []}
    data = yaml.safe_load(FACET_STATE_PATH.read_text(encoding="utf-8")) or {}
    usages = data.get("usages") if isinstance(data, dict) else []
    return {"usages": usages if isinstance(usages, list) else []}


def _recent_facet_usages(chars: list[str], n: int = FACET_RECENT_N) -> dict[str, list[dict]]:
    want = {_canonical_known_char(c) for c in chars or [] if c}
    out = {c: [] for c in want if c in _KNOWN_CHARS}
    if not out:
        return {}
    for usage in reversed(_load_facet_state().get("usages") or []):
        char = _canonical_known_char(usage.get("character"))
        if char in out and len(out[char]) < n:
            out[char].append(usage)
    return out


def _facet_block(chars: list[str], n: int = FACET_RECENT_N) -> str:
    recent = _recent_facet_usages(chars, n=n)
    if not recent:
        return ""

    lines = [
        "\nRECENTLY USED FACETS — avoid repeating these unless strongly required.",
        f"Lookback: last {n} appearances per character.",
    ]
    bans: list[str] = []
    force: list[str] = []

    for char in sorted(recent):
        usages = recent[char]
        if not usages:
            continue
        facet_list = [str(u.get("facet") or "").strip() for u in usages if str(u.get("facet") or "").strip()]
        if not facet_list:
            continue
        lines.append(f"- {char}: " + " | ".join(facet_list))
        if len(facet_list) >= 2 and facet_list[0].lower() == facet_list[1].lower():
            bans.append(f"- {char}: BAN facet \"{facet_list[0]}\" for this appearance.")

        joined = " ".join(facet_list[:n]).lower()
        if char == "hanyuyeon" and re.search(r"organ|system|perfect|reorgan|tidy|arrang|standard|color.?coded", joined):
            force.append(
                "- hanyuyeon: recent organizing/system/perfection facets are saturated. "
                "Force a non-organization facet such as competition, scary-calm losing, or praise vulnerability."
            )
        if char == "so-ae" and re.search(r"mini.?lecture|actually|detail|correction|correct|debunk|expertise", joined):
            force.append(
                "- so-ae: recent mini-lecture/Actually/detail-correction facets are saturated. "
                "Force sulk, approval-seeking, or socially awkward helpfulness."
            )

    if bans:
        lines += ["", "BANNED FACETS FOR THIS EPISODE:"] + bans
    if force:
        lines += ["", "FORCED FACET ROTATION:"] + force
    return "\n".join(lines) + "\n"


def _record_facet_usage(word: dict | None, plan: dict | None, panels: list[dict]) -> None:
    plan = plan or {}
    facets = _facet_rows_from_plan(plan)
    if not facets:
        return
    game = plan.get("comedic_game") or {}
    record_base = {
        "word_no": str((word or {}).get("No.", "")),
        "date": config.YY_MM_DD,
        "domain": plan.get("domain") or get_primary_domain(word),
        "selected_pair": _pair_from_plan(plan),
        "characters": list(dict.fromkeys(p.get("char") for p in panels if p.get("char"))),
        "driver_character": _driver_from_plan(plan),
        "comedic_game_premise": game.get("premise", ""),
        "target_sentence": (word or {}).get("collocation unit") or (word or {}).get("sentence_unit") or "",
        "relationship": (word or {}).get("relationship") or (word or {}).get("relationship context", ""),
        "speech_act": (word or {}).get("speech_act") or (word or {}).get("speech act", ""),
        "story_function": (word or {}).get("story_function") or (word or {}).get("story function", ""),
    }
    data = _load_facet_state()
    usages = data.setdefault("usages", [])
    for facet in facets:
        usages.append({**record_base, **facet})
    data["usages"] = usages[-300:]
    FACET_STATE_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9가-힣↔ -]+", " ", (value or "").lower()).split())


def _normalize_pair(pair: str) -> str:
    names = sorted(_pair_names(pair))
    return " ↔ ".join(names) if names else _normalize_text(pair)


def _milestone_hash(pair: str, word_no: str, summary: str) -> str:
    raw = "|".join([_normalize_pair(pair), str(word_no).strip(), _normalize_text(summary)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _summary_tokens(summary: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[a-z0-9가-힣]+", (summary or "").lower())
        if len(tok) > 2 and tok not in _MILESTONE_STOPWORDS
    }


def _summary_similarity(a: str, b: str) -> float:
    ta, tb = _summary_tokens(a), _summary_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_milestones() -> dict:
    if not MILESTONES_PATH.exists():
        return {"milestones": []}
    data = yaml.safe_load(MILESTONES_PATH.read_text(encoding="utf-8")) or {}
    events = data.get("milestones") if isinstance(data, dict) else []
    return {"milestones": events if isinstance(events, list) else []}


def _milestone_changes_relationship(milestone: dict, summary: str) -> bool:
    keys = {
        "phase", "new_phase", "phase_up", "comfort", "comfort_level", "new_comfort",
        "unresolved", "new_unresolved", "resolved_unresolved", "running_gag",
        "new_running_gag", "relationship_change", "dynamic_change",
    }
    if any(k in milestone and milestone.get(k) not in ("", None, False, [], {}) for k in keys):
        return True
    return bool(_MILESTONE_CHANGE_RE.search(summary or ""))


def _milestone_allowed(
    word: dict | None,
    milestone: dict | None,
    plan: dict | None = None,
    milestone_state: dict | None = None,
) -> tuple[bool, str, dict | None]:
    m = milestone or {}
    if not m.get("is_milestone"):
        return False, "standalone episode", None
    summary = str(m.get("summary") or "").strip()
    if not summary or summary.lower() == "null":
        return False, "empty milestone summary", None

    pair = str(m.get("pair") or _pair_from_plan(plan) or "").strip()
    word_no = str((word or {}).get("No.", "")).strip()
    if not pair:
        return False, "missing milestone pair", None
    if not _milestone_changes_relationship(m, summary):
        return False, "summary does not describe relationship phase/comfort/tension/running-gag change", None

    data = _load_milestones()
    events = data.setdefault("milestones", [])
    norm_pair = _normalize_pair(pair)
    mh = _milestone_hash(pair, word_no, summary)

    for event in events:
        event_pair = _normalize_pair(str(event.get("pair") or ""))
        event_word_no = str(event.get("word_no") or "").strip()
        event_summary = str(event.get("summary") or "")
        event_hash = event.get("milestone_hash") or _milestone_hash(str(event.get("pair") or ""), event_word_no, event_summary)
        if event_hash == mh:
            return False, "duplicate milestone_hash", None
        if event_pair == norm_pair and event_word_no == word_no:
            return False, "same pair + word_no already recorded", None

    recent_same_pair = [
        e for e in reversed(events)
        if _normalize_pair(str(e.get("pair") or "")) == norm_pair
    ][:5]
    for event in recent_same_pair:
        if _summary_similarity(summary, str(event.get("summary") or "")) >= 0.55:
            return False, "summary semantically similar to recent milestone for same pair", None

    state = milestone_state if milestone_state is not None else {}
    pair_counts = state.setdefault("pairs", {})
    if pair_counts.get(norm_pair, 0) >= 1:
        return False, "max 1 milestone per pair per batch", None
    total = int(state.get("total") or 0)
    if total > 0:
        max_allowed = max(1, total // 10)
        if int(state.get("recorded") or 0) >= max_allowed:
            return False, "batch milestone rate cap reached", None

    event = {
        "date": config.YY_MM_DD,
        "pair": pair,
        "word_no": word_no,
        "summary": summary,
        "milestone_hash": mh,
    }
    return True, "allowed", event


def _record_milestone(
    word: dict | None,
    milestone: dict | None,
    plan: dict | None = None,
    milestone_state: dict | None = None,
) -> None:
    """Append only deterministic relationship-changing milestones; standalone episodes skip."""
    ok, reason, event = _milestone_allowed(word, milestone, plan=plan, milestone_state=milestone_state)
    if not ok:
        if (milestone or {}).get("is_milestone"):
            print(f"  ⏭️  Milestone skipped: {reason}")
        return

    data = _load_milestones()
    events = data.setdefault("milestones", [])
    events.append(event)
    MILESTONES_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    if milestone_state is not None:
        norm_pair = _normalize_pair(event["pair"])
        milestone_state["recorded"] = int(milestone_state.get("recorded") or 0) + 1
        pairs = milestone_state.setdefault("pairs", {})
        pairs[norm_pair] = int(pairs.get(norm_pair) or 0) + 1
    print(f"  🏛️  Milestone recorded [{event['pair']}]: {event['summary']}")


def _update_arc_state(arc_update: dict) -> None:
    """GPT가 반환한 arc_update를 relationship_state.yaml 에 반영."""
    pair = (arc_update or {}).get("pair", "").strip()
    if not pair:
        return

    if not ARC_STATE_PATH.exists():
        print("  ⚠️ relationship_state.yaml 없음 — arc 업데이트 건너뜀")
        return

    data = yaml.safe_load(ARC_STATE_PATH.read_text(encoding="utf-8")) or {}
    rels = data["relationships"] if isinstance(data.get("relationships"), dict) else data
    if pair not in rels:
        print(f"  ⚠️ Arc pair '{pair}' not found in relationship_state.yaml — skipping update")
        return

    st = rels[pair] if isinstance(rels[pair], dict) else {}
    current_phase = st.get("phase", 1)
    new_phase = min(4, current_phase + 1) if arc_update.get("phase_up") else current_phase

    raw_beat = arc_update.get("new_last_beat") or None
    raw_gag  = arc_update.get("new_running_gag") or None
    st["phase"]     = new_phase
    st["last_beat"] = raw_beat if raw_beat and str(raw_beat).lower() != "null" else None
    if raw_gag and str(raw_gag).lower() != "null":
        st["running_gag"] = raw_gag

    rels[pair] = st
    ARC_STATE_PATH.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    phase_label = _PHASE_LABEL.get(new_phase, str(new_phase))
    print(f"  📖 Arc updated [{pair}] phase {new_phase} ({phase_label}) | beat: {st['last_beat']}")


# ─────────────────────────────────────────────────────────
# .env 수동 로드
# ─────────────────────────────────────────────────────────
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# ─────────────────────────────────────────────────────────
# CSV 단어 로드
# ─────────────────────────────────────────────────────────
def _scenario_csv_path() -> Path:
    """시나리오 생성은 원본 structured CSV 를 우선 사용한다.

    CLEAN_CSV 는 이후 comic example/dialogue 로 덮이므로 재생성 입력으로 쓰면
    이전 웹툰 소재가 되먹임된다. structured 가 없을 때만 하위호환으로 CLEAN_CSV 를 쓴다.
    """
    return config.STRUCTURED_CSV if config.STRUCTURED_CSV.exists() else config.CLEAN_CSV


def _read_word_rows() -> list[dict]:
    path = _scenario_csv_path()
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, 1):
        row.setdefault("No.", str(i))
        if not str(row.get("No.", "")).strip():
            row["No."] = str(i)
    return [_as_comic_word(row) for row in rows]


def load_word(no: int) -> dict:
    for row in _read_word_rows():
        if int(row["No."]) == no:
            return row
    raise ValueError(f"CSV에서 No.{no}를 찾지 못했습니다.")


def list_words() -> list[dict]:
    return _read_word_rows()


# ─────────────────────────────────────────────────────────
# 캐릭터 정의 — data/characters.yaml 가 단일 소스
# ─────────────────────────────────────────────────────────
def _parse_chars() -> dict:
    return load_characters()


# ─────────────────────────────────────────────────────────
# GPT-4o 시나리오 생성
# ─────────────────────────────────────────────────────────
def _build_cast_directive(anchor: dict | None) -> str:
    """캐스팅 지시문 — 누가 이 화를 끌지 강제. 앙상블 균형용."""
    if not anchor:
        return ""
    pair = anchor.get("pair", "")
    lead = anchor.get("lead", "")
    lines = []
    if pair:
        lines.append(f"- This episode MUST be anchored by the pair: {pair}.")
    if lead:
        lines.append(f"- {lead} drives the scene and leads most panels.")
    if anchor.get("side_episode"):
        lines.append(
            "- This is a SIDE EPISODE: hanyoil appears in AT MOST 1 panel, or not at all. "
            "The anchor pair carries the scene entirely — reveal who they are when hanyoil isn't around."
        )
    else:
        lines.append("- hanyoil may appear, but the emotional weight is SHARED with the anchor pair above.")
    lines.append("- If you report a milestone, its pair MUST be this anchor pair.")
    # 연속극(A) — 같은 날/장면 연속성: 장소·시간·직전 상황을 고정
    scene = anchor.get("scene")
    if scene:
        lines.append(
            f"- CONTINUITY: this episode is part of one continuous storyline. Setting: {scene}. "
            "Keep the location and time-of-day consistent with that. Panel-specific background details may change "
            "when they show the nuance beat, but do not jump to a different place."
        )
    prev = anchor.get("prev")
    if prev:
        lines.append(f"- Earlier in this same continuous timeline: {prev}. This scene follows naturally from it.")
    return "\n".join(lines)


def _gpt_json(system_prompt: str, user_msg: str, model: str = MODEL_PLAN) -> dict:
    """단일 호출 → JSON. model 로 단계별 티어 지정."""
    client_kwargs = {"timeout": OPENAI_TIMEOUT}
    if not config.OPENAI_SSL_VERIFY:
        client_kwargs["http_client"] = httpx.Client(verify=False, timeout=OPENAI_TIMEOUT)
    resp = OpenAI(**client_kwargs).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _panels_from(obj: dict) -> list[dict]:
    return obj.get("panels") or next((v for v in obj.values() if isinstance(v, list)), [])


# ── ① Story Planner — 구조/기획 ──
def _repair_visible_proof_panel(plan: dict) -> dict:
    """Keep visible_proof_panel aligned with the actual silent object proof beat."""
    beats = plan.get("beats") or []
    by_panel = {
        int(b.get("panel")): b
        for b in beats
        if str(b.get("panel", "")).isdigit()
    }

    def _valid(panel_no) -> bool:
        try:
            b = by_panel[int(panel_no)]
        except (TypeError, ValueError, KeyError):
            return False
        return (
            (b.get("panel_type") or "").strip().lower() == "object"
            and not (b.get("speaker") or "").strip()
            and not bool(b.get("has_collocation"))
        )

    if _valid(plan.get("visible_proof_panel")):
        return plan

    proof = next(
        (
            int(b.get("panel"))
            for b in beats
            if str(b.get("panel", "")).isdigit()
            and (b.get("panel_type") or "").strip().lower() == "object"
            and not (b.get("speaker") or "").strip()
            and not bool(b.get("has_collocation"))
            and (b.get("nuance_role") or "").strip().lower() in {"confirmation", "aftermath"}
        ),
        None,
    )
    plan["visible_proof_panel"] = proof
    return plan


def plan_episode(word, anchor, lore, arc_prompt, showrunner_notes, feedback="", avoid_situations="", cast_note="") -> dict:
    wb = build_word_block(word) if word else ""
    wr = build_word_rule(word["collocation unit"]) if word else "- Weave one natural English sentence into the conversation"
    cast = _build_cast_directive(anchor)
    required_domain = get_primary_domain(word).strip().lower()
    location_menu = build_location_menu(required_domain)
    prompt = build_planner_prompt(lore, wb, wr, cast, arc_prompt, showrunner_notes, feedback,
                                  avoid_situations, cast_note, required_domain, location_menu)
    plan = _gpt_json(prompt, "Plan the episode now. Output the JSON plan only.", model=MODEL_PLAN)
    return _repair_visible_proof_panel(plan)


def render_scene_brief(plan: dict, word: dict | None) -> str:
    """plan(JSON) → 대사 작가용 '산문 장면 브리프'로 변환 (GPT 호출 0, 결정적).

    JSON 덩어리 대신 읽기 좋은 산문을 주면 대사 유창성이 올라간다(Dramatron식 Dialogue Prompt).
    단, 비트 스켈레톤(패널 번호·화자·역할·콜로 위치·사물 패널)은 그대로 박아 계약을 지킨다.
    """
    coll = (word or {}).get("collocation unit", "")
    ns = plan.get("nuance_structure", {}) or {}
    ctx = plan.get("target_sentence_context") or {}
    meta = _sentence_metadata(word)
    L = [
        f"LOGLINE: {plan.get('sitcom_conflict','')}",
        f"WHERE: {plan.get('location','')}",
        "",
        f'EXACT TARGET SENTENCE: "{coll}"',
        f"KOREAN CUE: {(word or {}).get('meaning','')}",
        f"MICRO SITUATION: {(word or {}).get('nuance (Korean)','')}",
    ]
    if ctx or any(meta.values()):
        L += [
            "",
            "TARGET SENTENCE SOCIAL CONTEXT:",
            f"  - Relationship context: {ctx.get('relationship_context') or meta.get('relationship context', '')}",
            f"  - Speaker role: {ctx.get('target_speaker_role') or meta.get('speaker role', '')}",
            f"  - Listener role: {ctx.get('target_listener_role') or meta.get('listener role', '')}",
            f"  - Power dynamic: {ctx.get('power_dynamic') or meta.get('power dynamic', '')}",
            f"  - Speech act: {ctx.get('speech_act') or meta.get('speech act', '')}",
            f"  - Service direction: {ctx.get('service_direction') or meta.get('service direction', '')}",
            f"  - Story function: {ctx.get('story_function') or meta.get('story function', '')}",
            "  The marked target-sentence beat must obey this social context.",
        ]
    L += [
        "",
        "HOW THE CHARACTER CONFLICT UNFOLDS:",
        f"  - Situation: {ns.get('situation','')}",
        f"  - Pressure: {ns.get('pressure','')}",
        f"  - Target sentence moment: {ns.get('expression','')}",
        f"  - Confirmation: {ns.get('confirmation','')}",
    ]
    g = plan.get("comedic_game") or {}
    if g.get("premise"):
        L += ["", "THE GAME (escalate this ONE thing — don't resolve it early):",
              f"  - Driver: {g.get('driver','')} keeps doing -> {g.get('premise','')}",
              f"  - Escalation: {g.get('escalation','')}",
              f"  - Button (ending twist, NOT a warm hug): {g.get('button','')}",
              "  Write SPECIFIC, weird details (exact objects/numbers/claims), not generic ones."]
    cfc = plan.get("character_filter_collision") or []
    if cfc:
        L += ["", "WHO'S IN IT (each reacts differently — they must NOT sound interchangeable):"]
        for c in cfc:
            L.append(f"  - {c.get('character')}: {c.get('facet','')} -> {c.get('collision','')}")
    L += ["", "BEAT-BY-BEAT — write exactly one bubble per CHARACTER beat; OBJECT beats are silent "
          "cutaways (no bubble). Keep this speaker order and the marked exact target-sentence beat:"]
    for b in plan.get("beats", []):
        n, role = b.get("panel"), b.get("nuance_role", "")
        if (b.get("panel_type") == "object") or not b.get("speaker"):
            L.append(f"  Panel {n} [SILENT OBJECT — {role}]: show {b.get('visual_focus','')}")
        else:
            star = "   <<< the target sentence lands HERE, spoken verbatim by this character" if b.get("has_collocation") else ""
            L.append(f"  Panel {n} — {b.get('speaker')} ({role}): {b.get('intent','')}{star}")
    return "\n".join(L)


# ── ② Script — 대사 ──
def write_script(plan, word, lore, register, feedback="") -> list[dict]:
    wb = build_word_block(word) if word else ""
    tr = tone_rule_for(register)
    brief = render_scene_brief(plan, word)
    prompt = build_script_prompt(lore, wb, brief, tr, feedback, register)
    return _panels_from(_gpt_json(prompt, "Write the dialogue now. Output the JSON only.", model=MODEL_SCRIPT))


# Visual GPT 가 메뉴에 없는 expression 을 창작할 때(determined/worried 등) 안전 폴백.
_EXPR_SYNONYM: dict[str, str] = {
    "determined": "serious", "thoughtful": "furrowed_brow", "worried": "frown",
    "nervous": "furrowed_brow", "shocked": "fear_kubrick", "surprised": "fear_kubrick",
    "confident": "smug", "excited": "happy", "relieved": "light_smile",
    "embarrassed": "blush", "neutral": "serious", "curious": "furrowed_brow",
}


def _safe_expression_key(key: str | None) -> str:
    """expression key 를 expressions.yaml 메뉴 키로 정규화. 모르는 값은 동의어→serious 폴백."""
    raw = (key or "").strip()
    norm = raw.lower().replace(" ", "_").replace("-", "_")
    menu = load_expressions()
    if norm in menu:
        return norm
    if norm in _EXPR_SYNONYM:
        return _EXPR_SYNONYM[norm]
    return "serious"


# ── ③ Visual (SDXL) — 그림 태그 ──
def _char_demeanor(script_panels) -> str:
    """씬 등장 캐릭터의 baseline 표정(characters.yaml)을 표정 선택 가이드로."""
    chars = list(dict.fromkeys(p.get("char") for p in script_panels if p.get("char")))
    defs = load_characters()
    lines = []
    for c in chars:
        base = (defs.get(c) or {}).get("expression", "")
        if base:
            lines.append(f"- {c}: {base}")
    return "\n".join(lines)


_ACTION_REWRITE_MAP: tuple[tuple[str, str], ...] = (
    ("looking worried", "holding paper, looking at screen"),
    ("looking relieved", "hand on chest"),
    ("looking confused", "head scratch"),
    ("looking happy", "hand wave"),
    ("looking sad", "hand on chest"),
    ("looking angry", "pointing"),
    ("focused", "reading, holding paper"),
    ("laughing", "hand over mouth"),
    ("smirking", "hand on hip"),
    ("sighing in relief", "hand on chest"),
    ("emphasizing importance", "pointing"),
    ("highlighting important points", "pointing at screen"),
    ("humorous touch", ""),
    ("kitchen chaos", "messy kitchen"),
    ("visible email confirmation", "email notification, unread message, document attachment"),
    ("email confirmation", "email notification, unread message, document attachment"),
    ("confirmation email", "email notification, unread message, document attachment"),
    ("showing response", "email notification"),
    ("new email", "email notification"),
)

_ACTION_DROP_RE = re.compile(
    r"\b(?:worried|happy|sad|angry|confused|relieved|important|importance|humorous)\b",
    re.I,
)

# 이미지에 입김/연기로 렌더되는 호흡 태그 — action 에서 무조건 제거 (verify.py 와 동일 집합)
_BANNED_ACTION_RE = re.compile(
    r"\b(?:breath|breathing|exhal\w*|inhal\w*|sigh\w*|steam|fog|mist|vapor|smoke|puff\w*)\b",
    re.I,
)
# object 패널에 들어오면 안 되는 '사람 동작' 동사 — 사물 패널엔 사람이 없다
_HUMAN_ACTION_RE = re.compile(
    r"\b(?:nod\w*|writ\w*|gestur\w*|point\w*|lean\w*|hold\w*|wav\w*|shrug\w*|"
    r"smil\w*|stand\w*|sit\w*|walk\w*|turn\w*|look\w*|rais\w*|cross\w*|fidget\w*)\b",
    re.I,
)


def _dedupe_action_tags(action: str) -> str:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in action.split(","):
        tag = raw.strip().lower()
        if not tag or tag == "none":
            continue
        if _ACTION_DROP_RE.search(tag):
            tag = _ACTION_DROP_RE.sub("", tag).strip()
            tag = re.sub(r"\s{2,}", " ", tag).strip(" ,")
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return ", ".join(tags)


def _fallback_visual_action(panel: dict, is_object_panel: bool) -> str:
    subject = (panel.get("subject") or "").lower()
    bubble = (panel.get("bubble") or "").lower()
    visual_focus = (panel.get("visual_focus") or "").lower()
    context = " ".join([subject, bubble, visual_focus])
    if is_object_panel:
        if re.search(r"\bscreen|email|notification|slide|phone|monitor\b", context):
            return "displayed on screen"
        if re.search(r"\breceipt|paper|document|report|checklist|plate|cake|table\b", context):
            return "on table"
        if re.search(r"\bflour|batter|mess|box|receipt|paper\b", context):
            return "scattered"
        return "on table"
    if re.search(r"\bphone|email|screen|notification\b", context):
        return "holding phone"
    if re.search(r"\bpoint|screen|slide|report|document|checklist|map\b", context):
        return "pointing"
    return "standing"


def _sanitize_visual_actions(visual_panels: list[dict], script_panels: list[dict]) -> list[dict]:
    """Remove face/emotion/abstract phrases from Visual GPT action tags."""
    cleaned: list[dict] = []
    for i, panel in enumerate(visual_panels):
        p = dict(panel)
        is_object_panel = not ((script_panels[i].get("char") if i < len(script_panels) else "") or "").strip()
        action = (p.get("action") or "").strip()
        subject = (p.get("subject") or "").strip()
        if not is_object_panel:
            body_pose = (p.get("body_pose") or "").strip()
            gesture = (p.get("gesture") or "").strip()
            if body_pose:
                action = resolve_motion(body_pose, gesture)
        for bad, good in _ACTION_REWRITE_MAP:
            if is_object_panel and re.search(r"\b" + re.escape(bad) + r"\b", subject, flags=re.I):
                action = good
                break
            action = re.sub(r"\b" + re.escape(bad) + r"\b", good, action, flags=re.I)
        # 태그(콤마) 단위 필터: 호흡/연기 태그는 무조건, object 패널이면 사람 동작 태그도 통째 제거.
        # (단어 일부만 지우면 "taking a deep breath"→"taking a deep" 같은 잔여물이 남으므로 태그째 버린다)
        kept = []
        for t in action.split(","):
            t = t.strip()
            if not t:
                continue
            if _BANNED_ACTION_RE.search(t):
                continue
            if is_object_panel and _HUMAN_ACTION_RE.search(t):
                continue
            kept.append(t)
        action = ", ".join(kept)
        action = _dedupe_action_tags(action)
        if not action:
            action = _fallback_visual_action(
                {
                    **p,
                    "bubble": script_panels[i].get("bubble", "") if i < len(script_panels) else "",
                    "visual_focus": script_panels[i].get("visual_focus", "") if i < len(script_panels) else "",
                },
                is_object_panel,
            )
        p["action"] = action
        cleaned.append(p)
    return cleaned


def write_visuals(plan, script_panels) -> list[dict]:
    beats = plan.get("beats") or []
    planner_context = json.dumps(
        {
            "location": plan.get("location"),
            "visible_learning_moment": plan.get("visible_learning_moment"),
            "visible_proof_panel": plan.get("visible_proof_panel"),
            "nuance_structure": plan.get("nuance_structure"),
            "beats": beats,
        },
        ensure_ascii=False, indent=2,
    )
    sp = json.dumps(
        [
            {
                "char": p.get("char"),
                "bubble": p.get("bubble"),
                "panel_type": (beats[i].get("panel_type") if i < len(beats) else None),
                "nuance_role": (beats[i].get("nuance_role") if i < len(beats) else None),
                "visual_focus": (beats[i].get("visual_focus") if i < len(beats) else None),
            }
            for i, p in enumerate(script_panels)
        ],
        ensure_ascii=False, indent=2,
    )
    # expression 은 expressions.yaml 메뉴에서 선택, action 은 GPT 가 danbooru 태그로 직접 출력.
    prompt = build_visual_prompt(
        plan.get("situation", ""), sp,
        expression_menu=build_expression_menu(),
        pose_menu=build_motion_menu(),
        char_demeanor=_char_demeanor(script_panels),
        planner_context=planner_context,
    )
    visual_panels = _panels_from(_gpt_json(prompt, "Write the visual tags now. Output the JSON only.", model=MODEL_VISUAL))
    return _sanitize_visual_actions(visual_panels, script_panels)


def _avoid_block(used_situations) -> str:
    """배치 내에서 이미 쓴 situation_id 를 회피하라는 지시 — 옴니버스 다양성."""
    used = [s for s in (used_situations or []) if s]
    if not used:
        return ""
    return ("\nIMPORTANT — variety: these situations were ALREADY used in this batch, "
            "so pick a DIFFERENT one:\n  " + ", ".join(sorted(set(used))) + "\n")


_SUPPORTING = ["ru-ha", "hanyuyeon", "so-ae", "hyo-jeong"]  # hanyoil 은 주인공이라 밸런싱 제외


def _cast_balance_block(used_chars) -> str:
    """배치 내 '상대역' 분포를 보여주고 덜 나온 supporting 캐릭터를 우선하게 한다.
    주인공 hanyoil 은 비중이 커도 되므로 카운트/지시에서 제외."""
    from collections import Counter
    c = Counter(x for x in (used_chars or []) if x in _SUPPORTING)
    if not used_chars:
        return ""
    tally = ", ".join(f"{ch}×{c.get(ch, 0)}" for ch in _SUPPORTING)
    return ("\nENSEMBLE BALANCE — supporting cast used so far this batch: " + tally + ".\n"
            "Within this episode's domain cast pool, FAVOR a supporting partner who has appeared "
            "the LEAST (hanyoil-as-lead is fine; just vary who she's with).\n")


def _omnibus_memory_block(pair: str = "", chars: list[str] | None = None) -> str:
    try:
        return episode_log.build_omnibus_memory(pair=pair, chars=chars or [], limit=6)
    except Exception as exc:
        print(f"  ⚠️ 옴니버스 메모리 로드 실패: {exc}")
        return ""


def _generate_core(
    word,
    seed,
    anchor,
    feedback,
    used_situations=None,
    plan_sink=None,
    used_chars=None,
    debug_sink=None,
    used_pairs=None,
    batch_total=0,
) -> tuple[list[dict], dict | None]:
    """3단 파이프라인: ⓪선택 → ①기획 → ②대사 → ③그림 → 병합. 로깅/milestone 갱신 없음."""
    hair_ov  = (anchor or {}).get("hair")
    props_ov = (anchor or {}).get("props_extra")
    register = word["register"].strip().lower() if word else ""
    domain   = get_primary_domain(word).strip().lower()

    # ⓪ SELECT — 콜로케이션+뉘앙스 → 도메인 내 최적 관계쌍 (값싼 LLM) + 배치 쿼터 가드
    sel = select_relationship(word, domain, used_chars=used_chars, used_pairs=used_pairs,
                              total=batch_total) if word else {"pair": "", "lead": "", "candidates": _parse_arc_state()}
    if debug_sink is not None:
        debug_sink["selected_relationship"] = {
            **{k: v for k, v in sel.items() if k != "candidates"},
            "candidate_pairs": list((sel.get("candidates") or {}).keys()),
        }
    if sel.get("pair"):
        print(f"  🔗 관계 선택: {sel['pair']} (lead={sel.get('lead') or '—'})")
        anchor = {**(anchor or {}), "pair": sel["pair"], "lead": sel.get("lead", "")}

    # 린 lore: 공용(톤·작법) + 해당 도메인 세계관 + 선택된 인물 bible 만 (전체 lore 주입 X)
    bible_names = list(_pair_names(sel["pair"])) if sel.get("pair") else list(_DOMAIN_CAST.get(domain, _KNOWN_CHARS))
    planner_lore = "\n\n---\n\n".join(x for x in [
        _shared_lore(), load_domain_world(domain), load_character_bible(bible_names),
    ] if x)
    arc_prompt = build_arc_prompt(sel.get("candidates") or domain_pairs(domain))
    sel_note = (
        f"\nRELATIONSHIP (pre-selected for THIS expression — anchor the episode here):\n"
        f"- pair: {sel['pair']}\n- {sel.get('lead') or 'either'} should drive/initiate the scene.\n"
        f"- why this fits the sentence: {sel.get('why', '')}\n"
        if sel.get("pair") else "")
    omnibus_note = _omnibus_memory_block(sel.get("pair", ""), bible_names)
    facet_note = _facet_block(bible_names)

    if __package__:
        from .lore_keeper import load_writer_memo
    else:
        from notion_speaking.comic.lore_keeper import load_writer_memo
    showrunner_notes = load_writer_memo()

    # ① 기획 → deterministic plan validation
    plan_feedback = feedback
    plan = {}
    plan_issues: list[str] = []
    for attempt in range(PLAN_CHECK_RETRIES + 1):
        plan = plan_episode(word, anchor, planner_lore, arc_prompt, showrunner_notes, plan_feedback,
                            _avoid_block(used_situations),
                            sel_note + omnibus_note + facet_note + _cast_balance_block(used_chars))
        if sel.get("pair"):
            plan["selected_pair"] = sel["pair"]
        plan_issues = _plan_validation_issues(plan, word, sel.get("pair", ""))
        if not plan_issues:
            break
        if attempt < PLAN_CHECK_RETRIES:
            print(f"  🔁 plan deterministic check failed ({len(plan_issues)}) → replan {attempt + 1}/{PLAN_CHECK_RETRIES}")
        for issue in plan_issues:
            print(f"     - {issue}")
        if attempt >= PLAN_CHECK_RETRIES:
            break
        plan_feedback = f"{feedback}\n\n{_deterministic_feedback('plan', plan_issues)}".strip()
    if plan_issues:
        print(f"  ⚠️ plan deterministic check still failing — best effort: {plan_issues}")
    if sel.get("pair"):
        plan["selected_pair"] = sel["pair"]
    # 배치 다양성: 이번에 고른 상황을 누적 목록에 기록(같은 list 객체를 배치가 공유)
    if used_situations is not None and plan.get("situation_id"):
        used_situations.append(plan["situation_id"])
    # 앙상블 밸런싱: 이번 화 등장인물(주인공 외 상대역)을 누적
    if used_chars is not None:
        used_chars.extend(c for c in (plan.get("characters") or []))
    # 배치 쿼터: 이번에 고른 관계쌍을 누적 (다음 화 select 단계가 쏠림을 회피)
    if used_pairs is not None and sel.get("pair"):
        used_pairs.append(sel["pair"])
    # plan 원본을 리뷰/디버그용으로 노출(재시도 시 최종 plan 으로 덮어씀)
    if plan_sink is not None:
        plan_sink["plan"] = plan
    if debug_sink is not None:
        debug_sink["planner_json"] = plan
        debug_sink["selected_facets"] = _facet_rows_from_plan(plan)
        debug_sink.setdefault("validation_result", {})["plan"] = {
            "ok": not plan_issues,
            "issues": plan_issues,
        }
    # 의상은 planner 가 고른 situation 의 location 에 묶는다 (CSV used_in 보다 우선).
    # 그래야 "사무실 장면인데 사복" 같은 배경↔의상 불일치가 안 생긴다.
    # 의상은 planner 가 보고한 domain 으로 정한다(상황은 이제 자유생성이라 situation_id 가 라이브러리에 없음).
    # domain 없으면 단어의 used in 으로 폴백.
    # 의상: anchor 가 옷을 못박으면 전원 그 옷. 아니면 setting(장면 종류)만 공통으로 정하고
    # 변형은 캐릭터별로(자기 옷장에서) 고른다 → 주인공의 많은 변형이 살아나고 둘이 안 겹친다.
    forced_outfit = (anchor or {}).get("outfit")
    outfit_setting = _resolve_outfit_setting(
        plan.get("outfit_setting", ""),
        plan.get("domain") or get_primary_domain(word),
        plan.get("location"),
    )
    # customer/service 도메인이라도 hyo-jeong 이 실제 staff 역할일 때만 알바복(service_1).
    staff_char = _staff_character_for_scene(sel.get("pair", ""), word, plan.get("domain") or get_primary_domain(word))
    # Script 단계는 등장 확정 인물의 bible 만 주입(plan.characters) — 더 슬림.
    script_lore = load_character_bible(plan.get("characters") or bible_names)

    # ② 대사 → deterministic script validation. 실패 시 대사만 재작성.
    collocation = (word or {}).get("collocation unit", "")
    script_feedback = feedback
    script_panels: list[dict] = []
    script_issues: list[str] = []
    for attempt in range(SCRIPT_CHECK_RETRIES + 1):
        script_panels = write_script(plan, word, script_lore, register, script_feedback)[:MAX_PANELS]
        script_issues = _script_validation_issues(script_panels, plan, word)
        if not script_issues:
            break
        if attempt < SCRIPT_CHECK_RETRIES:
            print(f"  🔁 script deterministic check failed ({len(script_issues)}) → rewrite script {attempt + 1}/{SCRIPT_CHECK_RETRIES}")
        for issue in script_issues:
            print(f"     - {issue}")
        if attempt >= SCRIPT_CHECK_RETRIES:
            break
        script_feedback = f"{feedback}\n\n{_deterministic_feedback('script', script_issues)}".strip()
    if script_issues:
        print(f"  ⚠️ script deterministic check still failing — best effort: {script_issues}")
    if debug_sink is not None:
        debug_sink["script_json"] = {"panels": script_panels}
        debug_sink.setdefault("validation_result", {})["script"] = {
            "ok": not script_issues,
            "issues": script_issues,
        }

    # ③ 그림 태그 → deterministic visual validation. 실패 시 visual만 재작성.
    vis_panels: list[dict] = []
    visual_issues: list[str] = []
    for attempt in range(VISUAL_CHECK_RETRIES + 1):
        vis_panels = write_visuals(plan, script_panels)
        visual_issues = _visual_validation_issues(vis_panels, script_panels, plan)
        if not visual_issues:
            break
        if attempt < VISUAL_CHECK_RETRIES:
            print(f"  🔁 visual deterministic check failed ({len(visual_issues)}) → rewrite visuals {attempt + 1}/{VISUAL_CHECK_RETRIES}")
        for issue in visual_issues:
            print(f"     - {issue}")
        if attempt >= VISUAL_CHECK_RETRIES:
            break
    if visual_issues:
        print(f"  ⚠️ visual deterministic check still failing — best effort: {visual_issues}")
    if debug_sink is not None:
        debug_sink["visual_json"] = {"panels": vis_panels}
        debug_sink.setdefault("validation_result", {})["visual"] = {
            "ok": not visual_issues,
            "issues": visual_issues,
        }
    fixed_background = _background_for_word(word, plan)

    # 병합 + 정규화
    valid_chars = set(_parse_chars().keys())
    outfit_cache: dict[str, str] = {}   # char → outfit (한 화 내 캐릭터별 일관성)
    panels: list[dict] = []
    for i, sp in enumerate(script_panels):
        vp = vis_panels[i] if i < len(vis_panels) else {}
        raw_char = (sp.get("char") or "").strip()
        is_object_panel = not raw_char or raw_char.lower() in {"none", "object", "caption"}
        if is_object_panel:
            char = ""
            outfit = ""
        else:
            char = _canonical_char(raw_char, valid_chars)
            if staff_char and char == staff_char:
                outfit = "service_1"
            elif forced_outfit:
                outfit = forced_outfit
            else:
                outfit = outfit_cache.get(char) or _pick_char_outfit(char, outfit_setting, seed)
                outfit_cache[char] = outfit
        raw_bubble = sp.get("bubble", "")
        raw_bubble_kr = sp.get("bubble_kr", "")
        keep_object_bubble = is_object_panel and _is_narration_bubble(raw_bubble)
        p = {
            "panel_type": "object" if is_object_panel else "character",
            "char":       char,
            "outfit":     outfit,
            # action 은 GPT 가 danbooru 태그 직접 출력 → 그대로. expression 은 메뉴 key → tags 변환.
            "action":     (vp.get("action") or "standing").strip(),
            "body_pose":  (vp.get("body_pose") or ("none" if is_object_panel else "standing")).strip(),
            "gesture":    (vp.get("gesture") or "none").strip(),
            "subject":    (vp.get("subject") or "").strip(),
            "expression": "" if is_object_panel else resolve_expression(_safe_expression_key(vp.get("expression", "serious"))),
            "face_state": "" if is_object_panel else vp.get("face_state", "looking at viewer"),
            "background": fixed_background,
            "location": plan.get("location", ""),
            "used_in": get_primary_domain(word),
            "target_sentence": (word or {}).get("collocation unit", ""),
            "bubble":     raw_bubble if (not is_object_panel or keep_object_bubble) else "",
            "bubble_kr":  "" if keep_object_bubble else ("" if is_object_panel else raw_bubble_kr),
            "seed_offset": SEED_OFFSETS[i] if i < len(SEED_OFFSETS) else i * 7,
        }
        if hair_ov:
            p["hair_override"] = hair_ov
        if props_ov:
            p["props_extra"] = props_ov
        panels.append(p)

    if debug_sink is not None:
        debug_sink["final_panels"] = panels
        debug_sink["milestone"] = plan.get("milestone") or {}
    return panels, plan.get("milestone")


def commit_episode(
    word,
    panels,
    milestone,
    plan: dict | None = None,
    milestone_state: dict | None = None,
) -> None:
    """검증 통과한 에피소드를 원장에 기록. relationship_state.yaml 은 매회 자동수정하지 않고,
    '굵직한 사건(milestone)'만 별도 ledger(milestones.yaml)에 적는다 (옴니버스 + 마일스톤 연속성)."""
    _record_milestone(word, milestone, plan=plan, milestone_state=milestone_state)
    # 원장에는 에피소드 자체를 남기되, 매회 beat/running_gag/phase_up 은 비운다(누적 강제 제거).
    try:
        episode_log.append_episode(
            word,
            panels,
            {"pair": (milestone or {}).get("pair", "") or _pair_from_plan(plan)},
            plan=plan,
        )
    except Exception as exc:
        print(f"  ⚠️ 에피소드 원장 기록 실패: {exc}")
    try:
        _record_facet_usage(word, plan, panels)
    except Exception as exc:
        print(f"  ⚠️ facet_state 기록 실패: {exc}")


def _pair_from_plan(plan: dict | None) -> str:
    if (plan or {}).get("selected_pair"):
        return str((plan or {}).get("selected_pair"))
    chars = [c for c in ((plan or {}).get("characters") or []) if c]
    return " ↔ ".join(chars[:2]) if len(chars) >= 2 else ""


def generate_scenario(
    word: dict | None = None,
    seed: int = 0,
    anchor: dict | None = None,
    feedback: str = "",
    log_episode: bool = True,
    mech_retries: int = MECH_MAX_RETRIES,
    used_situations=None,
    plans_out=None,
    used_chars=None,
    milestone_state=None,
    debug_sink=None,
    used_pairs=None,
    batch_total=0,
) -> list[dict]:
    collocation = (word or {}).get("collocation unit", "")
    used_in = get_primary_domain(word)
    plan_sink: dict = {}

    panels, milestone = _generate_core(word, seed, anchor, feedback, used_situations, plan_sink, used_chars, debug_sink,
                                       used_pairs=used_pairs, batch_total=batch_total)

    # verify 게이트: 기본 OFF. 품질은 프롬프트 강화 + 결정적 sanitizer 로 앞단에서 보장한다.
    # VERIFY_ENABLED=1 일 때만 검증→재생성을 돈다.
    if VERIFY_ENABLED:
        # verify 는 이 모듈을 import 하므로 순환 회피를 위해 지연 import.
        if __package__:
            from .verify import mechanical_check, domain_check, metadata_check, nuance_check
        else:
            from notion_speaking.comic.verify import mechanical_check, domain_check, metadata_check, nuance_check

        def _check(panels) -> list[str]:
            plan = plan_sink.get("plan") or {}
            return (mechanical_check(panels, collocation)
                    + domain_check(plan, used_in)
                    + nuance_check(plan, collocation)
                    + metadata_check(plan, word))

        issues = _check(panels)
        for attempt in range(1, mech_retries + 1):
            if not issues:
                break
            print(f"  ⚙️ 기계 검증 {len(issues)}건 → 재생성 {attempt}/{mech_retries}")
            for m in issues:
                print(f"     - {m}")
            mech_fb = "Mechanical problems to fix:\n" + "\n".join(f"- {m}" for m in issues)
            fb = f"{feedback}\n\n{mech_fb}".strip() if feedback else mech_fb
            panels, milestone = _generate_core(word, seed, anchor, fb, used_situations, plan_sink, used_chars, debug_sink,
                                               used_pairs=used_pairs, batch_total=batch_total)
            issues = _check(panels)
        if issues:
            print(f"  ⚠️ 기계 검증 미통과 {len(issues)}건 — 최선본으로 진행: {issues}")

    if plans_out is not None:
        plans_out[str((word or {}).get("No.", ""))] = plan_sink.get("plan")
    if debug_sink is not None:
        ok, reason, event = _milestone_allowed(word, milestone, plan=plan_sink.get("plan"), milestone_state=milestone_state)
        debug_sink["milestone_decision"] = {
            "allowed": ok,
            "reason": reason,
            "event": event,
        }

    if log_episode:
        commit_episode(word, panels, milestone, plan_sink.get("plan"), milestone_state=milestone_state)
    return panels


# ─────────────────────────────────────────────────────────
# prompts.py COMIC_PANELS 블록 교체
# ─────────────────────────────────────────────────────────
def patch_prompts(panels: list[dict]) -> None:
    src   = PROMPTS_PATH.read_text(encoding="utf-8")
    lines = src.splitlines()

    start = next(
        (i for i, l in enumerate(lines) if re.match(r"^COMIC_PANELS\s*=\s*\[", l)),
        None,
    )
    if start is None:
        raise RuntimeError("prompts.py에서 COMIC_PANELS 블록을 찾지 못했습니다.")

    end = next(
        i for i in range(start + 1, len(lines))
        if lines[i].strip() == "]"
    )

    new_block = "COMIC_PANELS = " + json.dumps(panels, ensure_ascii=False, indent=4)
    new_lines = lines[:start] + new_block.splitlines() + lines[end + 1:]
    PROMPTS_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# 프로그래밍 진입점 (comic_client.py 에서 호출)
# ─────────────────────────────────────────────────────────
def run(word: dict | None = None, seed: int = 42) -> str:
    """단일 단어: 시나리오 생성 → prompts.py 패치 → modal 실행.

    Returns the Modal volume subdirectory name (e.g. ``comic_hanyoil_ru-ha_42``).
    """
    panels = generate_scenario(word=word)
    patch_prompts(panels)
    subprocess.run(
        ["modal", "run", "sd_generate.py", "--comic", "--seed", str(seed)],
        check=True, cwd=HERE,
    )
    chars = "_".join(dict.fromkeys((p.get("char") or "object") for p in panels))
    return f"comic_{chars}_{seed}"


def _print_batch_diagnostics(words: list[dict], plans_out: dict) -> None:
    """Compact batch-level visibility into relationship/context variety."""
    from collections import Counter

    missing = []
    rels = Counter()
    powers = Counter()
    acts = Counter()
    services = Counter()
    story = Counter()
    pairs = Counter()
    target_roles = Counter()

    for i, word in enumerate(words, 1):
        no = str(word.get("No.", i))
        meta = _sentence_metadata(word)
        empty = [k for k, v in meta.items() if not v]
        if empty:
            missing.append(f"No.{no}: " + ", ".join(empty))
        rels.update([meta.get("relationship context") or "(blank)"])
        powers.update([meta.get("power dynamic") or "(blank)"])
        acts.update([meta.get("speech act") or "(blank)"])
        services.update([meta.get("service direction") or "(blank)"])
        story.update([meta.get("story function") or "(blank)"])

        plan = plans_out.get(no) or {}
        pairs.update([plan.get("selected_pair") or _pair_from_plan(plan) or "(none)"])
        ctx = plan.get("target_sentence_context") or {}
        target_roles.update([ctx.get("target_speaker_role") or meta.get("speaker role") or "(blank)"])

    def fmt(counter: Counter, limit: int = 6) -> str:
        return ", ".join(f"{k}×{v}" for k, v in counter.most_common(limit)) or "(none)"

    print("\n📊 sentence-context diagnostics")
    print(f"   relationships: {fmt(rels)}")
    print(f"   power dynamics: {fmt(powers)}")
    print(f"   speech acts: {fmt(acts)}")
    print(f"   service direction: {fmt(services)}")
    print(f"   story function: {fmt(story)}")
    print(f"   selected pairs: {fmt(pairs)}")
    print(f"   target speaker roles: {fmt(target_roles)}")
    if missing:
        print(f"   ⚠️ metadata blanks: {len(missing)} rows")
        for line in missing[:8]:
            print(f"      - {line}")
        if len(missing) > 8:
            print(f"      ... {len(missing) - 8} more")


def generate_scenarios_batch(
    words: list[dict],
    seeds: list[int],
    log_episode: bool = True,
    include_review_cards: bool = True,
    debug_items: list[dict] | None = None,
) -> dict:
    """GPT text only — generate all scenarios and extract example sentences.

    Returns:
        {
            "batch":    [{"panels": [...], "seed": int}, ...],
            "subdirs":  ["comic_charA_charB_seed", ...],
            "examples": [{"no": str, "example": str}, ...],
        }
    """
    _load_dotenv(HERE.parent.parent / ".env")
    words = [_as_comic_word(word) for word in (words or [])]
    print(f"🤖 GPT-4o 시나리오 {len(words)}개 생성 중...")

    batch     = []
    subdirs   = []
    examples  = []
    dialogues = []
    review_cards = []                 # 단어별 단일 복습 카드(정사각 썸네일) 설정
    used_situations: list[str] = []  # 배치 전체에서 공유 — 같은 상황 반복 방지(옴니버스 다양성)
    used_chars: list[str] = []        # 배치 전체에서 공유 — 상대역 앙상블 밸런싱
    used_pairs: list[str] = []        # 배치 전체에서 공유 — 관계쌍 쿼터(단일 페어 ≤35%) 가드
    batch_total = len(words)
    plans_out: dict = {}              # word No. → plan JSON (리뷰/판단용)
    milestone_state: dict = {"total": len(words), "recorded": 0, "pairs": {}}
    for i, (word, seed) in enumerate(zip(words, seeds), 1):
        label = word.get("collocation unit", f"word {i}")
        print(f"  [{i}/{len(words)}] {label}")
        debug_item: dict | None = None
        if debug_items is not None:
            debug_item = {
                "word_no": str(word.get("No.", i)),
                "seed": seed,
                "input_sentence_metadata": dict(word),
            }
        try:
            panels = generate_scenario(word=word, seed=seed, used_situations=used_situations,
                                       plans_out=plans_out, used_chars=used_chars,
                                       milestone_state=milestone_state,
                                       log_episode=log_episode,
                                       debug_sink=debug_item,
                                       used_pairs=used_pairs, batch_total=batch_total)
        except Exception as exc:
            if debug_item is None:
                raise
            err = repr(exc)
            print(f"    ⚠️ dry-run generation failed: {err}")
            panels = []
            no = str(word.get("No.", i))
            debug_item["error"] = err
            debug_item.setdefault("validation_result", {})["runtime"] = {"ok": False, "issues": [err]}
            plans_out.setdefault(no, {})
        if debug_item is not None:
            debug_items.append(debug_item)
        for p in panels:
            print(f"    {(p.get('char') or 'object'):10s} | {p['bubble']}")
        chars = "_".join(dict.fromkeys((p.get("char") or "object") for p in panels))
        batch.append({"panels": panels, "seed": seed})
        subdirs.append(f"comic_{chars}_{seed}")
        example, speaker, bubble_kr = _extract_example(panels, word.get("collocation unit", ""))
        examples.append({"no": str(word.get("No.", i)), "example": example, "speaker": speaker, "translation": bubble_kr})
        dialogue_lines = [
            f"{p.get('char')}: {p['bubble']} ({p.get('bubble_kr', '')})"
            for p in panels
            if p.get("char") and p.get("bubble")
        ]
        dialogues.append({"no": str(word.get("No.", i)), "dialogue": "\n".join(dialogue_lines)})
        if include_review_cards:
            card = build_review_card(word, plans_out.get(str(word.get("No.", i))), panels, seed)
            if card:
                review_cards.append(card)

    _print_batch_diagnostics(words, plans_out)

    # 배치 이미지 생성은 run_images_batch 가 --batch-json 으로 전체를 넘긴다(prompts.py 안 읽음).
    # 따라서 prompts.py COMIC_PANELS 를 마지막 단어로 덮어쓰던 호출은 불필요한 부작용이라 제거.
    return {"batch": batch, "subdirs": subdirs, "examples": examples,
            "dialogues": dialogues, "plans": plans_out, "review_cards": review_cards}


IMAGES_BATCH_CHUNK = int(os.getenv("IMAGES_BATCH_CHUNK", "12"))  # 청크당 최대 코믹 수 (Modal 3600s 타임아웃 회피)


def run_images_batch(batch: list[dict]) -> None:
    """Modal-only: generate images from pre-computed batch JSON.

    배치를 청크로 쪼개 modal 을 여러 번 호출한다 — 한 호출에 40개를 넣으면
    Modal 함수 타임아웃(3600s)에 걸리므로(코믹당 ~95s), 청크마다 새 호출(=새 타임아웃)로 분산.
    """
    if not batch:
        return
    chunks = [batch[i:i + IMAGES_BATCH_CHUNK] for i in range(0, len(batch), IMAGES_BATCH_CHUNK)]
    print(f"\n🚀 modal run (총 {len(batch)}개 → {len(chunks)}청크, 청크당 ≤{IMAGES_BATCH_CHUNK})...")
    for ci, chunk in enumerate(chunks, 1):
        print(f"  ── 청크 {ci}/{len(chunks)} ({len(chunk)}개) ──")
        subprocess.run(
            ["modal", "run", "sd_generate.py::main",
             "--batch-comic", "--batch-json", json.dumps(chunk, ensure_ascii=False)],
            cwd=HERE, check=True,
    )


REVIEW_CARDS_CHUNK = int(os.getenv("REVIEW_CARDS_CHUNK", "24"))  # 청크당 최대 복습 카드 수


def run_review_cards_batch(cards: list[dict]) -> None:
    """Modal-only: 단어별 단일 복습 카드(정사각 썸네일)를 batch 로 생성.

    카드당 1장(~25s)이라 만화보다 가벼워 청크를 더 크게 잡는다.
    """
    cards = [c for c in (cards or []) if c]
    if not cards:
        return
    chunks = [cards[i:i + REVIEW_CARDS_CHUNK] for i in range(0, len(cards), REVIEW_CARDS_CHUNK)]
    print(f"\n🖼️  복습 카드 {len(cards)}장 → {len(chunks)}청크 생성...")
    for ci, chunk in enumerate(chunks, 1):
        print(f"  ── 카드 청크 {ci}/{len(chunks)} ({len(chunk)}장) ──")
        subprocess.run(
            ["modal", "run", "sd_generate.py::main",
             "--review-cards", "--batch-json", json.dumps(chunk, ensure_ascii=False)],
            cwd=HERE, check=True,
        )


def run_batch(words: list[dict], seeds: list[int]) -> list[str]:
    """여러 단어: GPT 시나리오 전체 생성 → modal 1회 실행 (모델 로딩 1번)."""
    result = generate_scenarios_batch(words, seeds)
    run_images_batch(result["batch"])
    run_review_cards_batch(result.get("review_cards", []))
    return result["subdirs"]


def _counter_dict(counter) -> dict:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))))


def _diagnose_panels(word: dict, plan: dict, panels: list[dict]) -> dict:
    target = (
        word.get("collocation unit")
        or word.get("sentence_unit")
        or word.get("sentence unit")
        or ""
    )
    beats = plan.get("beats") or []
    target_beats = _target_beats(plan)
    marked_idx = target_beats[0][0] if len(target_beats) == 1 else -1
    marked_speaker = _canonical_known_char(plan.get("target_speaker") or (target_beats[0][1].get("speaker") if target_beats else ""))

    target_count = 0
    target_idx = -1
    repeated_speaker_violations = 0
    object_panel_bubble_violations = 0
    prev_char = ""
    for i, panel in enumerate(panels):
        char = str(panel.get("char") or "").strip()
        bubble = str(panel.get("bubble") or "").strip()
        bubble_kr = str(panel.get("bubble_kr") or "").strip()
        beat = beats[i] if i < len(beats) else {}
        is_object = str(panel.get("panel_type") or beat.get("panel_type") or "").strip().lower() == "object" or not char
        if is_object and (char or bubble or bubble_kr):
            object_panel_bubble_violations += 1
        if char:
            canon = _canonical_known_char(char)
            if prev_char and canon == prev_char:
                repeated_speaker_violations += 1
            prev_char = canon
        count_here = _exact_sentence_count(bubble, target) if target else 0
        if count_here:
            target_count += count_here
            if target_idx < 0:
                target_idx = i

    target_success = target_count == 1
    marked_success = target_success and target_idx == marked_idx
    speaker_success = True
    if target_success and marked_speaker:
        speaker_success = _canonical_known_char(panels[target_idx].get("char")) == marked_speaker
    return {
        "target_success": target_success,
        "marked_success": marked_success and speaker_success,
        "repeated_speaker_violations": repeated_speaker_violations,
        "object_panel_bubble_violations": object_panel_bubble_violations,
    }


def _premise_similarity_warnings(plans: dict) -> list[dict]:
    warnings = []
    seen: list[tuple[str, str, str]] = []
    for no, plan in plans.items():
        premise = str(((plan or {}).get("comedic_game") or {}).get("premise") or "").strip()
        if not premise:
            continue
        for other_no, other_premise, other_pair in seen:
            sim = _summary_similarity(premise, other_premise)
            if sim >= 0.55:
                warnings.append({
                    "word_no": no,
                    "similar_to": other_no,
                    "similarity": round(sim, 3),
                    "selected_pair": (plan or {}).get("selected_pair") or _pair_from_plan(plan),
                    "other_pair": other_pair,
                    "premise": premise,
                    "other_premise": other_premise,
                })
        seen.append((no, premise, (plan or {}).get("selected_pair") or _pair_from_plan(plan)))
    return warnings


def _service_role_inconsistencies(words: list[dict], plans: dict) -> list[dict]:
    issues = []
    for i, word in enumerate(words, 1):
        no = str(word.get("No.", i))
        plan = plans.get(no) or {}
        if (plan.get("domain") or get_primary_domain(word)) != "customer/service":
            continue
        relationship = _metadata_value(word, "relationship", "relationship context")
        if relationship not in {"staff_to_customer", "customer_to_staff", "staff_to_coworker"}:
            continue
        pair = plan.get("selected_pair") or _pair_from_plan(plan)
        roles = resolve_service_roles(word, pair)
        if relationship in {"staff_to_customer", "customer_to_staff"} and roles.get("staff") != "hyo-jeong":
            issues.append({"word_no": no, "issue": "staff is not hyo-jeong", "pair": pair, "roles": roles})
        if roles.get("target_speaker") and _canonical_known_char(plan.get("target_speaker")) != roles["target_speaker"]:
            issues.append({"word_no": no, "issue": "target_speaker mismatch", "expected": roles["target_speaker"], "actual": plan.get("target_speaker")})
        if roles.get("target_listener") and _canonical_known_char(plan.get("target_listener")) != roles["target_listener"]:
            issues.append({"word_no": no, "issue": "target_listener mismatch", "expected": roles["target_listener"], "actual": plan.get("target_listener")})
    return issues


def build_batch_diagnostic_report(words: list[dict], result: dict, debug_items: list[dict] | None = None) -> dict:
    from collections import Counter, defaultdict

    batch = result.get("batch") or []
    plans = result.get("plans") or {}
    total = len(batch)
    domain = Counter()
    relationship = Counter()
    selected_pair = Counter()
    chars = Counter()
    drivers = Counter()
    facet_counts = defaultdict(Counter)
    speaker_roles = Counter()
    speech_acts = Counter()
    story_functions = Counter()
    situations = Counter()
    metadata_sources = Counter()
    missing_metadata = []
    milestone_count = 0

    exact_ok = 0
    marked_ok = 0
    repeated_speaker_violations = 0
    object_panel_bubble_violations = 0

    facet_streaks: dict[str, list[str]] = defaultdict(list)

    for i, (word, item) in enumerate(zip(words, batch), 1):
        no = str(word.get("No.", i))
        plan = plans.get(no) or {}
        panels = item.get("panels") or []
        metadata_sources.update([word.get("_metadata_source") or "unknown"])
        blanks = [
            key for key in _REQUIRED_SENTENCE_METADATA
            if key != "avoid_with" and not str(word.get(key) or "").strip()
        ]
        if blanks:
            missing_metadata.append({"word_no": no, "missing": blanks})
        domain.update([plan.get("domain") or get_primary_domain(word) or "(blank)"])
        relationship.update([_metadata_value(word, "relationship", "relationship context") or plan.get("relationship") or "(blank)"])
        selected_pair.update([plan.get("selected_pair") or _pair_from_plan(plan) or "(none)"])
        episode_chars = list(dict.fromkeys(p.get("char") for p in panels if p.get("char")))
        chars.update(episode_chars)
        driver = _canonical_known_char(((plan.get("comedic_game") or {}).get("driver") or ""))
        drivers.update([driver or "(blank)"])
        speaker_roles.update([_metadata_value(word, "speaker_role", "speaker role") or plan.get("speaker_role") or "(blank)"])
        speech_acts.update([_metadata_value(word, "speech_act", "speech act") or plan.get("speech_act") or "(blank)"])
        story_functions.update([_metadata_value(word, "story_function", "story function") or plan.get("story_function") or "(blank)"])
        if plan.get("situation_id"):
            situations.update([plan["situation_id"]])
        if (plan.get("milestone") or {}).get("is_milestone"):
            milestone_count += 1

        for row in _facet_rows_from_plan(plan):
            facet_counts[row["character"]].update([row["facet"]])
            facet_streaks[row["character"]].append(row["facet"])

        panel_diag = _diagnose_panels(word, plan, panels)
        exact_ok += int(panel_diag["target_success"])
        marked_ok += int(panel_diag["marked_success"])
        repeated_speaker_violations += panel_diag["repeated_speaker_violations"]
        object_panel_bubble_violations += panel_diag["object_panel_bubble_violations"]

    service_issues = _service_role_inconsistencies(words, plans)
    repeated_situations = {k: v for k, v in situations.items() if v > 1}
    premise_warnings = _premise_similarity_warnings(plans)

    runtime_errors = [
        {"word_no": item.get("word_no"), "error": item.get("error")}
        for item in (debug_items or [])
        if item.get("error")
    ]
    failures = []
    if runtime_errors:
        failures.append(f"runtime errors: {len(runtime_errors)}")
    supporting = [ch for ch in _SUPPORTING if total and chars.get(ch, 0) / total > 0.45]
    if supporting:
        failures.append("supporting character over 45%: " + ", ".join(supporting))
    pair_over = [pair for pair, count in selected_pair.items() if total and count / total > 0.35]
    if pair_over:
        failures.append("selected pair over 35%: " + ", ".join(pair_over))
    for char, streak in facet_streaks.items():
        for idx in range(2, len(streak)):
            if streak[idx].lower() == streak[idx - 1].lower() == streak[idx - 2].lower():
                failures.append(f"same facet 3 times in a row: {char} -> {streak[idx]}")
                break
    milestone_rate = milestone_count / total if total else 0.0
    if milestone_rate > 0.10:
        failures.append(f"milestone rate >10%: {milestone_rate:.1%}")
    target_rate = exact_ok / total if total else 1.0
    marked_rate = marked_ok / total if total else 1.0
    if target_rate < 1.0:
        failures.append(f"target sentence success rate <100%: {target_rate:.1%}")
    if service_issues:
        failures.append(f"service role inconsistency exists: {len(service_issues)}")
    if missing_metadata:
        failures.append(f"sentence metadata missing required fields: {len(missing_metadata)} rows")

    return {
        "total_episodes": total,
        "domain_distribution": _counter_dict(domain),
        "relationship_distribution": _counter_dict(relationship),
        "selected_pair_distribution": _counter_dict(selected_pair),
        "character_appearance_counts": _counter_dict(chars),
        "driver_character_counts": _counter_dict(drivers),
        "facet_usage_counts_per_character": {char: _counter_dict(counter) for char, counter in sorted(facet_counts.items())},
        "target_sentence_speaker_role_distribution": _counter_dict(speaker_roles),
        "speech_act_distribution": _counter_dict(speech_acts),
        "story_function_distribution": _counter_dict(story_functions),
        "sentence_metadata_source_distribution": _counter_dict(metadata_sources),
        "sentence_metadata_missing": missing_metadata,
        "milestone_count": milestone_count,
        "milestone_rate": round(milestone_rate, 4),
        "repeated_situation_id_count": sum(v - 1 for v in repeated_situations.values()),
        "repeated_situation_ids": dict(sorted(repeated_situations.items())),
        "repeated_comedic_game_premise_similarity_warning": premise_warnings,
        "hyo_jeong_service_role_consistency": {
            "ok": not service_issues,
            "issues": service_issues,
        },
        "exact_target_sentence_success_rate": round(target_rate, 4),
        "marked_target_beat_success_rate": round(marked_rate, 4),
        "repeated_speaker_violations": repeated_speaker_violations,
        "object_panel_bubble_violations": object_panel_bubble_violations,
        "runtime_errors": runtime_errors,
        "failures": failures,
        "ok": not failures,
    }


def run_batch_dry(limit: int = 30, diagnostics: bool = False) -> int:
    words = list_words()[:limit]
    seeds = [1234 + i for i in range(1, len(words) + 1)]
    result = generate_scenarios_batch(words, seeds, log_episode=False, include_review_cards=False)
    if diagnostics:
        report = build_batch_diagnostic_report(words, result)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("ok") else 1
    print(json.dumps({"total_episodes": len(result.get("batch") or [])}, ensure_ascii=False, indent=2))
    return 0


def _dialogue_markdown(panels: list[dict]) -> str:
    lines = []
    for panel in panels or []:
        char = panel.get("char") or "object"
        bubble = panel.get("bubble") or ""
        if bubble:
            lines.append(f"- **{char}**: {bubble}")
    return "\n".join(lines) or "- _(no dialogue)_"


def _case_weakness_score(item: dict) -> int:
    score = 0
    if item.get("error"):
        score += 100
    validation = item.get("validation_result") or {}
    for stage in ("plan", "script", "visual"):
        score += 5 * len((validation.get(stage) or {}).get("issues") or [])
    if not (item.get("milestone_decision") or {}).get("allowed") and (item.get("milestone") or {}).get("is_milestone"):
        score += 1
    panels = item.get("final_panels") or []
    plan = item.get("planner_json") or {}
    word = item.get("input_sentence_metadata") or {}
    score += 4 * len(_script_validation_issues(item.get("script_json", {}).get("panels") or panels, plan, word))
    return score


def _write_weak_cases(path: Path, debug_items: list[dict]) -> None:
    ranked = sorted(debug_items, key=_case_weakness_score, reverse=True)[:10]
    lines = ["# Weak Cases", "", "Focus: failures and suspicious outputs, not showcases.", ""]
    for rank, item in enumerate(ranked, 1):
        word = item.get("input_sentence_metadata") or {}
        plan = item.get("planner_json") or {}
        validation = item.get("validation_result") or {}
        issues = []
        for stage in ("plan", "script", "visual"):
            issues += [f"{stage}: {x}" for x in ((validation.get(stage) or {}).get("issues") or [])]
        script_extra = _script_validation_issues(item.get("script_json", {}).get("panels") or [], plan, word)
        issues += [f"script_recheck: {x}" for x in script_extra]
        lines += [
            f"## {rank}. No.{item.get('word_no')} — {word.get('collocation unit') or word.get('sentence_unit', '')}",
            "",
            f"- domain: `{plan.get('domain') or get_primary_domain(word)}`",
            f"- pair: `{plan.get('selected_pair') or _pair_from_plan(plan)}`",
            f"- score: `{_case_weakness_score(item)}`",
            f"- situation_id: `{plan.get('situation_id', '')}`",
            f"- premise: {((plan.get('comedic_game') or {}).get('premise') or '')}",
            f"- runtime_error: `{item.get('error', '')}`",
            "- issues:",
        ]
        lines += [f"  - {issue}" for issue in issues] if issues else ["  - no deterministic issues; review for blandness/repetition"]
        lines += ["", "Dialogue:", _dialogue_markdown(item.get("final_panels") or []), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_domain_breakdown(path: Path, debug_items: list[dict]) -> None:
    domains = ["daily", "workplace", "customer/service", "academic"]
    by_domain = {d: [] for d in domains}
    for item in debug_items:
        word = item.get("input_sentence_metadata") or {}
        plan = item.get("planner_json") or {}
        dom = plan.get("domain") or get_primary_domain(word)
        if dom in by_domain:
            by_domain[dom].append(item)
    lines = ["# Domain Breakdown", "", "Generated 30 total; each section includes up to 10 outputs.", ""]
    for dom in domains:
        items = by_domain[dom][:10]
        lines += [f"## {dom}", ""]
        if len(items) < 10:
            lines.append(f"_Only {len(items)} outputs available for this domain in the 30-episode shakedown._")
            lines.append("")
        for item in items:
            word = item.get("input_sentence_metadata") or {}
            plan = item.get("planner_json") or {}
            lines += [
                f"### No.{item.get('word_no')} — {word.get('collocation unit') or word.get('sentence_unit', '')}",
                f"- pair: `{plan.get('selected_pair') or _pair_from_plan(plan)}`",
                f"- speaker_role: `{word.get('speaker_role', '')}` / speech_act: `{word.get('speech_act', '')}`",
                f"- facet(s): " + ", ".join(f"{f.get('character')}={f.get('facet')}" for f in item.get("selected_facets") or []),
                f"- premise: {((plan.get('comedic_game') or {}).get('premise') or '')}",
                "",
                _dialogue_markdown(item.get("final_panels") or []),
                "",
            ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ab_comparison(path: Path, debug_items: list[dict]) -> None:
    lines = [
        "# A/B Comparison",
        "",
        "Requested: same 20 sentence units with old vs new pipeline.",
        "",
        "Result: old pipeline was not run. The active `notion_speaking` path has been migrated in-place, "
        "and no safe frozen old-pipeline entry point exists for the new sentence metadata schema without risking "
        "mixed state or invalid inputs.",
        "",
        "## New Pipeline Sample (first 20)",
        "",
    ]
    for item in debug_items[:20]:
        word = item.get("input_sentence_metadata") or {}
        plan = item.get("planner_json") or {}
        lines += [
            f"- No.{item.get('word_no')}: `{word.get('collocation unit') or word.get('sentence_unit', '')}`",
            f"  - new pair: `{plan.get('selected_pair') or _pair_from_plan(plan)}`",
            f"  - new facet(s): " + ", ".join(f"{f.get('character')}={f.get('facet')}" for f in item.get("selected_facets") or []),
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_shakedown_summary(path: Path, report: dict) -> None:
    lines = [
        "# Shakedown Diagnostics Summary",
        "",
        f"- requested limit: {report.get('requested_limit')}",
        f"- available input rows: {report.get('available_input_rows')}",
        f"- total episodes: {report.get('total_episodes')}",
        f"- ok: {report.get('ok')}",
        f"- milestone rate: {report.get('milestone_rate')}",
        f"- exact target sentence success rate: {report.get('exact_target_sentence_success_rate')}",
        f"- marked target beat success rate: {report.get('marked_target_beat_success_rate')}",
        f"- repeated speaker violations: {report.get('repeated_speaker_violations')}",
        f"- object panel bubble violations: {report.get('object_panel_bubble_violations')}",
        f"- sentence metadata sources: {json.dumps(report.get('sentence_metadata_source_distribution', {}), ensure_ascii=False)}",
        f"- sentence metadata rows missing required fields: {len(report.get('sentence_metadata_missing') or [])}",
        "",
        "## Failures",
    ]
    failures = report.get("failures") or []
    lines += [f"- {failure}" for failure in failures] if failures else ["- none"]
    for key in [
        "domain_distribution", "relationship_distribution", "selected_pair_distribution",
        "character_appearance_counts", "driver_character_counts", "speech_act_distribution",
        "story_function_distribution",
    ]:
        lines += ["", f"## {key}", "```json", json.dumps(report.get(key, {}), ensure_ascii=False, indent=2), "```"]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_shakedown(limit: int = 30) -> int:
    base_dir = Path.cwd() / "debug" / ("shakedown_" + datetime.now().strftime("%Y%m%d_%H%M"))
    out_dir = base_dir
    suffix = 2
    while out_dir.exists():
        out_dir = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    words = list_words()[:limit]
    seeds = [1234 + i for i in range(1, len(words) + 1)]
    debug_items: list[dict] = []
    result = generate_scenarios_batch(
        words,
        seeds,
        log_episode=False,
        include_review_cards=False,
        debug_items=debug_items,
    )
    report = build_batch_diagnostic_report(words, result, debug_items=debug_items)
    report["requested_limit"] = limit
    report["available_input_rows"] = len(words)
    if len(words) < limit:
        report.setdefault("warnings", []).append(f"requested {limit} episodes but only {len(words)} input rows are available")

    (out_dir / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "all_intermediates.json").write_text(json.dumps(debug_items, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in debug_items:
        ep_dir = out_dir / f"episode_{str(item.get('word_no')).zfill(2)}"
        ep_dir.mkdir(exist_ok=True)
        for key, filename in [
            ("input_sentence_metadata", "input_sentence_metadata.json"),
            ("selected_relationship", "selected_relationship.json"),
            ("selected_facets", "selected_facets.json"),
            ("planner_json", "planner.json"),
            ("script_json", "script.json"),
            ("validation_result", "validation_result.json"),
            ("milestone_decision", "milestone_decision.json"),
        ]:
            (ep_dir / filename).write_text(json.dumps(item.get(key, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    _write_shakedown_summary(out_dir / "diagnostics_summary.md", report)
    _write_weak_cases(out_dir / "weak_cases.md", debug_items)
    _write_domain_breakdown(out_dir / "domain_breakdown.md", debug_items)
    _write_ab_comparison(out_dir / "ab_comparison.md", debug_items)
    print(json.dumps({"shakedown_dir": str(out_dir), "ok": report.get("ok"), "failures": report.get("failures")}, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  type=int, default=-1)
    parser.add_argument("--word",  type=int, default=None, help="CSV 학습 단어 번호 (No. 열)")
    parser.add_argument("--dry",   action="store_true",    help="시나리오 확인만 (modal 실행 안 함)")
    parser.add_argument("--batch-dry", action="store_true", help="batch text generation only; no logs, images, or uploads")
    parser.add_argument("--limit", type=int, default=30, help="batch dry row limit")
    parser.add_argument("--diagnostics", action="store_true", help="print batch diagnostic JSON report")
    parser.add_argument("--shakedown", action="store_true", help="write full dry-run shakedown artifacts under debug/")
    args = parser.parse_args()

    _load_dotenv(HERE.parent.parent / ".env")

    if args.batch_dry:
        raise SystemExit(run_batch_dry(limit=args.limit, diagnostics=args.diagnostics))
    if args.shakedown:
        raise SystemExit(run_shakedown(limit=args.limit))

    word = None
    if args.word is not None:
        word = load_word(args.word)
        print(f"📖 학습 단어: [{word['No.']}] {word['collocation unit']} — {word['meaning']}")
        print(f"   register: {word.get('register', '')} | used in: {get_primary_domain(word)}")
        print(f"   nuance: {word['nuance (Korean)']}\n")

    print("🤖 GPT-4o 시나리오 생성 중...")
    panels = generate_scenario(word=word)

    print("\n📋 생성된 시나리오:")
    for i, p in enumerate(panels):
        print(f"  [{i+1}] {(p.get('char') or 'object'):12s} | {p.get('outfit', ''):10s} | {p['bubble']}")

    if args.dry:
        print("\n[dry mode] prompts.py 업데이트 및 modal 실행 건너뜀.")
        return

    print("\n✏️  prompts.py COMIC_PANELS 업데이트...")
    patch_prompts(panels)
    print("   완료.")

    cmd = ["modal", "run", "sd_generate.py", "--comic"]
    if args.seed >= 0:
        cmd += ["--seed", str(args.seed)]
    print(f"\n🚀 실행: {' '.join(cmd)}\n{'─'*50}")
    subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
