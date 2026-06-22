"""verify.py — 에피소드 기계적 체크 (LLM 없음, 즉시·무료).

숨 태그·독백·콜로케이션 누락·빈 버블·화자 교대 등 episode_rules.md 의 구조 규칙을
LLM 없이 즉시 검출한다. 두 곳에서 쓰인다:
  - generate_scenario.py : 대사 생성 중 재시도 게이트
  - continuity.py        : 세계관 일관성 검증의 L1(결정적) 일부

품질 critic(LLM 점수화)은 세계관 일관성 검증(continuity.py)으로 대체되어 제거됨.
"""
from __future__ import annotations

import re

# 이미지에 입김/연기로 렌더되는 금지 action 태그
_BANNED_ACTION = re.compile(r"\b(breath|breathing|exhal|sigh|steam|fog|mist|vapor|smoke|puff)\w*", re.I)


def _sentence_match_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[.!?]+", "", text)
    return " ".join(text.split())


def _contains_exact_sentence(text: str, target_sentence: str) -> bool:
    target = _sentence_match_text(target_sentence)
    if not target:
        return False
    candidates = [text]
    candidates.extend(re.split(r"[.!?]+", text or ""))
    return any(_sentence_match_text(candidate) == target for candidate in candidates)


def mechanical_check(panels: list[dict], collocation: str = "") -> list[str]:
    issues: list[str] = []
    if len(panels) < 4:
        issues.append(f"패널이 {len(panels)}개뿐 (최소 4)")

    # 숨/연기 태그 (이미지에 입김 렌더)
    for i, p in enumerate(panels, 1):
        if _BANNED_ACTION.search(p.get("action") or ""):
            issues.append(f"p{i} action에 금지 태그(숨/연기): \"{p.get('action') or ''}\"")

    # 화자 교대 위반 — 같은 캐릭터가 연속 2컷 이상 발화 (대화는 번갈아야 함)
    for i in range(1, len(panels)):
        if panels[i].get("char") and panels[i].get("char") == panels[i - 1].get("char"):
            issues.append(f"p{i}-p{i+1}: 같은 캐릭터({panels[i].get('char')})가 연속 발화 — 화자 교대 필요")

    # 빈 버블 / 과장 길이
    for i, p in enumerate(panels, 1):
        if (p.get("panel_type") or "").strip().lower() in {"object", "setting"} or not p.get("char"):
            continue
        b = (p.get("bubble") or "").strip()
        if not b:
            issues.append(f"p{i} 빈 버블")
        elif len(b.split()) > 25:
            issues.append(f"p{i} 버블이 너무 김({len(b.split())}단어)")

    # sentence unit 누락 — 스피킹 학습용이라 정확한 문장 전체가 대사에 들어가야 한다.
    if collocation:
        found = any(_contains_exact_sentence(p.get("bubble", ""), collocation) for p in panels if p.get("char"))
        if not found:
            issues.append(f"sentence unit '{collocation}' 이 대사에 정확히 안 나옴")

    return issues


# CSV 'used in' (정본 도메인) → planner 가 골라도 되는 domain 값 집합
# 정확히 4개 도메인만 허용 — planner 가 social/personal 등 다른 값을 내면 재생성으로 교정
_DOMAIN_ALLOW: dict[str, set[str]] = {
    "workplace":        {"workplace"},
    "daily":            {"daily"},
    "customer/service": {"customer/service"},
    "academic":         {"academic"},
}

# 비-workplace 도메인인데 이 단어가 location 에 들어가면 '사무실 드리프트'로 간주
_OFFICE_LOC = re.compile(r"\b(office|meeting|plaoud|desk|conference|rooftop|\bhr\b|cubicle|boardroom)\w*", re.I)


def domain_check(plan: dict, used_in: str) -> list[str]:
    """CSV 'used in' 을 정본으로, planner 가 고른 domain/location 이 어긋났는지 검출.

    프롬프트만으로는 LLM 이 사무실로 드리프트하므로(통계상 daily/cs 절반이 사무실),
    여기서 위반을 잡아 재생성 피드백으로 되먹인다.
    """
    issues: list[str] = []
    csv_dom = (used_in or "").strip().lower()
    allow = _DOMAIN_ALLOW.get(csv_dom)
    if not allow:
        return issues  # 알 수 없는 도메인은 강제 안 함

    plan_dom = (plan.get("domain") or "").strip().lower()
    if plan_dom and plan_dom not in allow:
        issues.append(
            f"도메인 불일치: CSV 'used in'={csv_dom} 인데 plan.domain={plan_dom} — "
            f"반드시 {' 또는 '.join(sorted(allow))} 중 하나로 설정할 것")

    # workplace 가 아닌데 사무실 장소면 드리프트
    if csv_dom != "workplace":
        loc = (plan.get("location") or "").strip()
        if _OFFICE_LOC.search(loc):
            issues.append(
                f"장소 드리프트: '{csv_dom}' 장면인데 location='{loc}' 이 사무실 계열 — "
                f"사무실/회의실/플라우드 밖의 실제 장소로 옮길 것")
    return issues


def nuance_check(plan: dict, collocation: str) -> list[str]:
    """Ensure the planner anchored the exact sentence in the target-sentence beat."""
    issues: list[str] = []
    col = (collocation or "").strip().lower()
    if not col:
        return issues
    ns = plan.get("nuance_structure") or {}
    expr = " ".join(str(ns.get(k, "")) for k in ("pressure", "expression")).lower()
    beats = plan.get("beats") or []
    target_beats = [b for b in beats if b.get("has_collocation")]
    if not target_beats:
        issues.append("target sentence가 들어갈 beat(has_collocation=true)가 없음")
        return issues
    toks = col.split()
    head = " ".join(toks[:2]) if len(toks) >= 2 else col
    beat_text = " ".join(
        f"{b.get('nuance_role', '')} {b.get('intent', '')}"
        for b in target_beats
    ).lower()
    if col not in expr and head not in expr and col not in beat_text and head not in beat_text:
        issues.append(
            f"target sentence beat가 '{collocation}' 에 충분히 anchor 되지 않음 — "
            "nuance_structure.expression 또는 has_collocation beat intent에 정확한 순간을 명시할 것")
    return issues


_REQUIRED_CONTEXT_KEYS = {
    "relationship_context",
    "target_speaker_role",
    "target_listener_role",
    "power_dynamic",
    "speech_act",
    "service_direction",
    "story_function",
}

_STORY_FUNCTION_TO_ROLES = {
    "setup_problem": {"situation"},
    "reveal_pressure": {"pressure"},
    "escalate_conflict": {"pressure", "expression"},
    "state_decision": {"expression", "confirmation"},
    "request_solution": {"pressure", "expression"},
    "resist_pressure": {"pressure", "expression"},
    "soften_tension": {"confirmation"},
    "expose_mistake": {"pressure", "expression"},
    "confirm_result": {"confirmation"},
    "button_reaction": {"confirmation"},
    "starts_conflict": {"situation", "pressure"},
    "escalates_conflict": {"pressure", "expression"},
    "softens_conflict": {"confirmation"},
    "resolves_conflict": {"confirmation"},
    "creates_misunderstanding": {"pressure", "expression"},
    "reveals_emotion": {"expression", "confirmation"},
    "buys_time": {"pressure", "expression"},
    "sets_up_punchline": {"expression", "confirmation"},
}

_SERVICE_CONTEXT_ALLOW = {
    "customer_to_staff": {"customer/service"},
    "staff_to_customer": {"customer/service"},
    "customer_to_service": {"customer/service"},
    "service_to_customer": {"customer/service"},
    "teacher_to_student": {"academic"},
    "student_to_teacher": {"academic"},
}


def metadata_check(plan: dict, word: dict | None) -> list[str]:
    """Check that generated plan keeps sentence-level social/story metadata alive."""
    issues: list[str] = []
    word = word or {}
    ctx = plan.get("target_sentence_context") or {}
    if not ctx:
        issues.append("target_sentence_context 가 없음 — relationship/power/speech-act/story metadata를 plan에 포함할 것")
        return issues

    missing = sorted(k for k in _REQUIRED_CONTEXT_KEYS if not str(ctx.get(k, "")).strip())
    if missing:
        issues.append("target_sentence_context 필드 누락/공백: " + ", ".join(missing))

    beats = plan.get("beats") or []
    target_beats = [b for b in beats if b.get("has_collocation")]
    if len(target_beats) != 1:
        issues.append(f"has_collocation beat 수가 {len(target_beats)}개 — 정확히 1개여야 함")
    elif not str(target_beats[0].get("speaker", "")).strip():
        issues.append("target sentence beat가 object/caption beat임 — 반드시 캐릭터 발화여야 함")

    story_function = str(ctx.get("story_function") or word.get("story_function") or word.get("story function") or "").strip()
    if story_function and target_beats:
        role = str(target_beats[0].get("nuance_role") or "").strip().lower()
        allowed = _STORY_FUNCTION_TO_ROLES.get(story_function)
        if allowed and role not in allowed:
            issues.append(
                f"story_function={story_function} 인데 target beat nuance_role={role or 'empty'} — "
                f"{', '.join(sorted(allowed))} beat에 배치할 것")

    domain = str(plan.get("domain") or word.get("used in") or word.get("primary_used_in") or "").strip().lower()
    service_direction = str(
        ctx.get("service_direction")
        or word.get("relationship")
        or word.get("service_direction")
        or word.get("service direction")
        or ""
    ).strip()
    allowed_domains = _SERVICE_CONTEXT_ALLOW.get(service_direction)
    if allowed_domains and domain not in allowed_domains:
        issues.append(
            f"service_direction={service_direction} 는 {', '.join(sorted(allowed_domains))} 도메인에서만 자연스러움 "
            f"(현재 domain={domain or 'empty'})")

    return issues
