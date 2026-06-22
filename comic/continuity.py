"""continuity.py — 세계관 일관성(ConStory) 검증.

기존 verify.py(단일 편 품질 critic)를 대체하는 *검증 스테이지*. 한 에피소드가
"그 자체로 잘 만들어졌나"가 아니라 "확립된 세계관(canon)과 모순 없나"를 본다.

근거 (논문):
  - SCORE (arXiv 2503.23512): 상태 추적 + 구조화 사실을 LLM-judge에 함께 먹이면
    분석 정확도 90%↑. 결정적 상태검사 + 의미적 판정의 하이브리드.
  - Lost in Stories (arXiv 2603.05890): 연속성 오류 5범주/19유형 — 사실·관계·성격.
  - Guiding Storytelling with KG (arXiv 2505.24803): 구조화 캐논 주입이 충돌을 줄임.

2층 면역계:
  L1 결정적(무료): canon 파일 대조만으로 잡는 사실/규칙 위반.
       - 등장 캐스트 유효성            (characters.yaml)
       - 에피소드 구조 규칙            (episode_rules.md → mechanical_check 재사용)
       - hyo-jeong 비직원 규칙         (world.md)
       - 학원 = 회상, 캐스트 제한      (situation.md)
       - 의상 정합                     (characters.yaml outfits)
       - pair 유효성                   (relationship_state.yaml)
  L2 LLM-judge: 규칙으로 못 잡는 성격/관계 역행. 관련 캐논만 골라 먹인다.
       - 캐릭터가 triggers/behaviors/fear 에 역행 (#3: ru-ha가 sloppy work 방치)
       - pair 의 정본 dynamic 과 모순

출력: {"ok", "violations":[{layer,type,character,severity,evidence,conflicts_with,note}], ...}
  passed = high 심각도 위반이 하나도 없음.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from . import config_loader

HERE = Path(__file__).parent
LORE_DIR = HERE.parent / "lore"
CHARACTERS_MD = LORE_DIR / "characters.md"
ARC_STATE_PATH = LORE_DIR / "relationship_state.yaml"
EPISODES_PATH = HERE / "episodes.jsonl"

JUDGE_MODEL = os.getenv("CONTINUITY_MODEL", os.getenv("CRITIC_MODEL", "gpt-4o"))

# ── world.md / situation.md 에서 유도한 정적 캐논 규칙 ──
COMPANY_TEAM = {"hanyoil", "ru-ha", "hanyuyeon", "so-ae"}   # Plaoud HR 팀
NON_EMPLOYEES = {"hyo-jeong"}                               # world.md: 회사 소속 아님(알바)
ACADEMIC_CAST = {"hanyoil", "hyo-jeong", "so-ae"}           # situation.md: academic 은 이 셋
_OFFICE_BG = re.compile(r"\b(office|desk|workspace|meeting room|conference|cubicle|monitor)", re.I)
_SCHOOL_BG = re.compile(r"\b(classroom|library|cafeteria|hallway|school|lecture|chalkboard)", re.I)
_SOCIAL_BG = re.compile(r"\b(cafe|convenience store|apartment|rooftop|riverside|park|home)", re.I)


# ─────────────────────────────────────────────────────────
# canon 로더
# ─────────────────────────────────────────────────────────
def _parse_characters_md() -> dict[str, str]:
    """characters.md 를 {char_name: trait_block_text} 로 쪼갠다."""
    if not CHARACTERS_MD.exists():
        return {}
    text = CHARACTERS_MD.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    cur, buf = None, []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if cur:
                blocks[cur] = "\n".join(buf).strip()
            cur, buf = m.group(1).strip(), []
        elif cur:
            buf.append(line)
    if cur:
        blocks[cur] = "\n".join(buf).strip()
    return blocks


def _parse_arc_state() -> dict[str, dict]:
    """relationship_state.yaml → {pair: {phase, comfort_level, dynamic, unresolved, ...}}."""
    if not ARC_STATE_PATH.exists():
        return {}
    data = yaml.safe_load(ARC_STATE_PATH.read_text(encoding="utf-8")) or {}
    rels = data.get("relationships", data)
    return {p: st for p, st in rels.items() if isinstance(st, dict)}


def _phase_labels() -> dict[int, str]:
    if not ARC_STATE_PATH.exists():
        return {}
    data = yaml.safe_load(ARC_STATE_PATH.read_text(encoding="utf-8")) or {}
    return (data.get("meta") or {}).get("phase_scale", {}) or {}


def _recent_beats(pair_keys: list[str], limit: int = 3) -> list[str]:
    """episodes.jsonl 에서 해당 pair 의 최근 beat 들 (L2 맥락용)."""
    if not EPISODES_PATH.exists() or not pair_keys:
        return []
    want = set(pair_keys)
    beats: list[str] = []
    for line in EPISODES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("pair") in want and r.get("beat"):
            beats.append(f"[{r.get('date','?')}] {r['beat']}")
    return beats[-limit:]


# ─────────────────────────────────────────────────────────
# scene 메타 추출
# ─────────────────────────────────────────────────────────
def _scene_chars(panels: list[dict]) -> list[str]:
    """등장 캐릭터(중복 제거, 등장 순서 유지)."""
    return list(dict.fromkeys(p.get("char") for p in panels if p.get("char")))


def _pair_key(chars: list[str], arc: dict) -> str | None:
    """등장 2인 → relationship_state.yaml 의 정본 pair 키(순서 무관) 매칭."""
    if len(chars) != 2:
        return None
    a, b = chars
    for key in arc:
        parts = [x.strip() for x in re.split(r"↔|<->|/", key)]
        if {a, b} == set(parts):
            return key
    return None


def _is_academic(panels: list[dict]) -> bool:
    for p in panels:
        if config_loader.is_flashback(p.get("outfit")):
            return True
        if _SCHOOL_BG.search(p.get("background") or ""):
            return True
    return False


def _is_office(panels: list[dict]) -> bool:
    if _is_academic(panels):
        return False
    office = any(_OFFICE_BG.search(p.get("background") or "") for p in panels)
    social = any(_SOCIAL_BG.search(p.get("background") or "") for p in panels)
    return office and not social


def _v(layer, type_, severity, evidence, conflicts_with, note, character=None) -> dict:
    return {
        "layer": layer, "type": type_, "severity": severity,
        "character": character, "evidence": evidence,
        "conflicts_with": conflicts_with, "note": note,
    }


# ─────────────────────────────────────────────────────────
# L1 — 결정적 canon 검사 (무료)
# ─────────────────────────────────────────────────────────
def deterministic_check(word: dict | None, panels: list[dict]) -> list[dict]:
    from .verify import mechanical_check  # episode_rules.md 구조 규칙 재사용

    chars_yaml = config_loader.load_characters()
    arc = _parse_arc_state()
    chars = _scene_chars(panels)
    violations: list[dict] = []

    # 1) 캐스트 유효성 — 정의되지 않은 캐릭터
    for c in chars:
        if c not in chars_yaml:
            violations.append(_v(
                "deterministic", "unknown_character", "high",
                evidence=f"panel char = {c!r}",
                conflicts_with="characters.yaml (정의된 캐릭터 목록)",
                note=f"'{c}' 는 캐논에 없는 캐릭터.", character=c))

    # 2) 에피소드 구조 규칙 (episode_rules.md) — mechanical_check 재사용
    collocation = (word or {}).get("collocation unit", "")
    for issue in mechanical_check(panels, collocation):
        sev = "high" if "콜로케이션" in issue else "medium"
        violations.append(_v(
            "deterministic", "episode_rule", sev,
            evidence=issue, conflicts_with="episode_rules.md",
            note="에피소드 구조 규칙 위반."))

    # 3) hyo-jeong 비직원 규칙 (world.md)
    if _is_office(panels):
        for c in chars:
            if c in NON_EMPLOYEES:
                violations.append(_v(
                    "deterministic", "non_employee_at_work", "medium",
                    evidence=f"{c} 가 오피스 장면에 등장 (bg office)",
                    conflicts_with="world.md (hyo-jeong 은 Plaoud 소속 아님)",
                    note=f"{c} 가 사내 업무 장면의 동료처럼 등장 — 방문이면 정당.", character=c))

    # 4) 학원 = 회상 + 캐스트 제한 (situation.md)
    if _is_academic(panels):
        for c in chars:
            if c not in ACADEMIC_CAST:
                violations.append(_v(
                    "deterministic", "academic_cast", "high",
                    evidence=f"{c} 가 학원(회상) 장면에 등장",
                    conflicts_with="situation.md (academic 캐스트 = hanyoil/hyo-jeong/so-ae)",
                    note=f"{c} 는 학원 회상 캐스트가 아님.", character=c))

    # 5) 의상 정합 (characters.yaml outfits)
    for i, p in enumerate(panels, 1):
        c, outfit = p.get("char"), p.get("outfit")
        if c in chars_yaml and outfit:
            valid = (chars_yaml[c].get("outfits") or {})
            if outfit not in valid:
                violations.append(_v(
                    "deterministic", "invalid_outfit", "low",
                    evidence=f"p{i} {c} outfit={outfit!r}",
                    conflicts_with=f"characters.yaml {c}.outfits ({', '.join(valid) or '없음'})",
                    note="정의되지 않은 의상 세트.", character=c))

    # 6) pair 유효성 (2인 장면인데 정본 pair 가 아님)
    if len(chars) == 2 and arc and _pair_key(chars, arc) is None:
        violations.append(_v(
            "deterministic", "unknown_pair", "low",
            evidence=f"pair = {chars}",
            conflicts_with="relationship_state.yaml (정본 10쌍)",
            note="관계 정의가 없는 조합."))

    return violations


# ─────────────────────────────────────────────────────────
# L2 — LLM-judge (성격/관계 역행). 관련 캐논만 주입.
# ─────────────────────────────────────────────────────────
def _judge_prompt(word: dict | None, panels: list[dict]) -> str:
    chars = _scene_chars(panels)
    md = _parse_characters_md()
    arc = _parse_arc_state()
    labels = _phase_labels()

    script = "\n".join(
        f"  p{i} [{p.get('char')}] ({p.get('action','')}): {p.get('bubble','')}"
        for i, p in enumerate(panels, 1)
    )
    canon_blocks = "\n\n".join(
        f"### {c}\n{md.get(c, '(no canon block)')}" for c in chars
    )
    rel_lines = []
    pk = _pair_key(chars, arc)
    if pk and pk in arc:
        st = arc[pk]
        ph = st.get("phase")
        rel_lines.append(
            f"{pk}: phase {ph} ({labels.get(ph, '?')}), comfort {st.get('comfort_level')}\n"
            f"  canonical dynamic: {st.get('dynamic')}\n"
            f"  unresolved: {st.get('unresolved')}\n"
            f"  running_gag: {st.get('running_gag')}"
        )
    rel = "\n".join(rel_lines) or "(single-character or no canonical pair)"
    beats = _recent_beats([pk] if pk else [], limit=3)
    recent = "\n".join(f"  - {b}" for b in beats) or "  (none)"
    col = (word or {}).get("collocation unit", "")

    return f"""You are a continuity editor (story-bible keeper) for an ongoing slice-of-life workplace webtoon.
Your ONLY job is to detect CONTRADICTIONS between THIS new scene and the ESTABLISHED CANON below.
Do NOT judge writing quality, jokes, or clarity — only canon consistency.

TARGET PHRASE (must appear naturally, ignore for continuity): "{col}"

NEW SCENE:
{script}

=== ESTABLISHED CHARACTER CANON (characters.md) ===
{canon_blocks}

=== ESTABLISHED RELATIONSHIP (relationship_state.yaml) ===
{rel}

=== RECENT BEATS for this pair (episodes) ===
{recent}

Flag a violation when, and ONLY when, the scene clearly contradicts the canon, e.g.:
- a character acts AGAINST their established triggers / behaviors / fear / defense
  (e.g. a perfectionist whose trigger is "sloppy work" casually tolerating a sloppy mistake),
- the interaction contradicts the pair's canonical dynamic,
- a character suddenly knows/feels something the canon says they don't (relationship regression).
A character using a DIFFERENT-but-compatible trait is NOT a violation. When unsure, do NOT flag.

Return ONLY this JSON:
{{
  "ok": true,
  "violations": [
    {{"type": "character_reversal|dynamic_contradiction|relationship_regression",
      "character": "name or null",
      "severity": "low|medium|high",
      "evidence": "exact line quoted from the scene",
      "conflicts_with": "which canon field it violates (cite trait/dynamic)",
      "note": "one sentence: why it contradicts"}}
  ]
}}
"ok" must be false if violations is non-empty."""


def llm_judge(word: dict | None, panels: list[dict]) -> dict:
    from openai import OpenAI
    prompt = _judge_prompt(word, panels)
    resp = OpenAI().chat.completions.create(
        model=JUDGE_MODEL,
        seed=7,  # best-effort 재현성
        messages=[
            {"role": "system", "content": "You are a strict story-continuity editor. Reply with ONLY the JSON object."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    out = []
    for v in (data.get("violations") or []):
        out.append(_v(
            "llm", v.get("type", "character_reversal"), v.get("severity", "medium"),
            evidence=v.get("evidence", ""), conflicts_with=v.get("conflicts_with", ""),
            note=v.get("note", ""), character=v.get("character")))
    return {"violations": out}


# ─────────────────────────────────────────────────────────
# 통합 진입점
# ─────────────────────────────────────────────────────────
def continuity_check(word: dict | None, panels: list[dict], use_llm: bool = True) -> dict:
    """L1(결정적) + L2(LLM-judge) 합쳐 위반 목록과 통과 여부를 낸다.

    passed = high 심각도 위반이 하나도 없음.
    use_llm=False 면 L1 만 (무료·결정적).
    """
    violations = deterministic_check(word, panels)
    llm_error = None
    if use_llm:
        try:
            violations += llm_judge(word, panels)["violations"]
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"

    high = [v for v in violations if v.get("severity") == "high"]
    med = [v for v in violations if v.get("severity") == "medium"]
    out = {
        "ok": not high,
        "violations": violations,
        "n_high": len(high),
        "n_medium": len(med),
        "deterministic_ok": not any(v["layer"] == "deterministic" and v["severity"] in ("high", "medium") for v in violations),
        "llm_used": use_llm,
    }
    if llm_error:
        out["llm_error"] = llm_error
    return out


def format_feedback(check: dict) -> str:
    """검증 위반을 regen 프롬프트용 feedback 문자열로."""
    vs = check.get("violations") or []
    if not vs:
        return ""
    lines = ["Continuity violations to fix (keep the scene consistent with canon):"]
    for v in vs:
        lines.append(
            f"- [{v['severity']}] {v['type']}"
            + (f" ({v['character']})" if v.get("character") else "")
            + f": {v['note']}"
            + (f" — line: \"{v['evidence']}\"" if v.get("evidence") else "")
            + (f" — conflicts with {v['conflicts_with']}" if v.get("conflicts_with") else "")
        )
    return "\n".join(lines)
