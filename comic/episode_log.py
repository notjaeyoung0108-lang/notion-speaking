"""episode_log.py — 에피소드 원장(Archivist) + 패턴 감지.

세계관 확장의 *연료*. Writer가 4컷 대본을 만들 때마다 한 줄씩 append-only로 쌓고,
Lore Keeper(쇼러너)가 이 누적 기록을 읽어 떡밥·개그·관계 패턴을 감지한다.

LLM 없음 — 순수 집계. 단독 테스트 가능.

Usage:
    from .episode_log import append_episode, load_episodes, detect_patterns
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

LEDGER_PATH = Path(__file__).parent / "episodes.jsonl"


# ─────────────────────────────────────────────────────────
# 기록 (Archivist)
# ─────────────────────────────────────────────────────────
def append_episode(
    word: dict | None,
    panels: list[dict],
    arc_update: dict | None,
    plan: dict | None = None,
    date: str | None = None,
) -> dict:
    """한 에피소드를 원장에 append. 기록된 레코드를 반환."""
    date = date or datetime.now().strftime("%y.%m.%d")
    plan = plan or {}
    chars = list(dict.fromkeys(p.get("char") for p in panels if p.get("char")))
    bg = panels[0].get("background", "") if panels else ""
    location = bg.split(",")[0].strip().lower() if bg else ""
    game = plan.get("comedic_game") or {}
    selected_facets = [
        {
            "character": str(item.get("character") or "").strip(),
            "facet": str(item.get("facet") or "").strip(),
            "collision": str(item.get("collision") or "").strip(),
        }
        for item in (plan.get("character_filter_collision") or [])
        if str(item.get("character") or "").strip() and str(item.get("facet") or "").strip()
    ]

    record = {
        "date":        date,
        "word_no":     str((word or {}).get("No.", "")),
        "collocation": (word or {}).get("collocation unit", ""),
        "sentence_unit": (word or {}).get("sentence_unit") or (word or {}).get("sentence unit") or (word or {}).get("collocation unit", ""),
        "korean_trigger": (word or {}).get("korean_trigger") or (word or {}).get("Korean trigger") or (word or {}).get("meaning", ""),
        "micro_situation": (word or {}).get("micro_situation") or (word or {}).get("micro situation") or (word or {}).get("nuance (Korean)", ""),
        "used_in":     (word or {}).get("primary_used_in") or (word or {}).get("used in", ""),
        "relationship_context": (word or {}).get("relationship") or (word or {}).get("relationship context", ""),
        "speaker_role": (word or {}).get("speaker_role") or (word or {}).get("speaker role", ""),
        "listener_role": (word or {}).get("listener_role") or (word or {}).get("listener role", ""),
        "power_dynamic": (word or {}).get("power_dynamic") or (word or {}).get("power dynamic", ""),
        "speech_act": (word or {}).get("speech_act") or (word or {}).get("speech act", ""),
        "service_direction": (word or {}).get("relationship") or (word or {}).get("service direction", ""),
        "politeness": (word or {}).get("politeness", ""),
        "story_function": (word or {}).get("story_function") or (word or {}).get("story function", ""),
        "character_fit": (word or {}).get("character_fit", ""),
        "avoid_with": (word or {}).get("avoid_with", ""),
        "target_sentence_context": plan.get("target_sentence_context") or {},
        "pair":        (arc_update or {}).get("pair", ""),
        "chars":       chars,
        "lead":        panels[0].get("char", "") if panels else "",
        "location":    location,
        "outfit":      panels[0].get("outfit", "") if panels else "",
        "situation_id": plan.get("situation_id", ""),
        "sitcom_conflict": plan.get("sitcom_conflict", ""),
        "visible_learning_moment": plan.get("visible_learning_moment", ""),
        "comedic_game": game,
        "selected_facets": selected_facets,
        "driver_character": game.get("driver", ""),
        "comedic_game_premise": game.get("premise", ""),
        "callback_seed": _callback_seed(plan, panels),
        "character_filter_collision": plan.get("character_filter_collision") or [],
        "beat":        (arc_update or {}).get("new_last_beat", ""),
        "running_gag": (arc_update or {}).get("new_running_gag") or game.get("premise", ""),
        "phase_up":    bool((arc_update or {}).get("phase_up")),
        "dialogue":    [
            f'{p.get("char", "")}: {p.get("bubble", "")}'
            for p in panels
            if p.get("char") and p.get("bubble")
        ],
        "panels":      panels,  # 전체 SDXL 패널 — 언제든 재렌더 가능
    }
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _callback_seed(plan: dict, panels: list[dict]) -> str:
    """A compact future callback candidate from the episode."""
    game = plan.get("comedic_game") or {}
    bits = [
        game.get("premise", ""),
        game.get("button", ""),
        plan.get("visible_learning_moment", ""),
    ]
    for panel in reversed(panels or []):
        bubble = (panel.get("bubble") or "").strip()
        if bubble:
            bits.append(f"{panel.get('char', '')}: {bubble}")
            break
    return " / ".join(bit for bit in bits if bit)[:500]


# ─────────────────────────────────────────────────────────
# 로드
# ─────────────────────────────────────────────────────────
def load_episodes(limit: int | None = None) -> list[dict]:
    """원장 전체(또는 최근 limit개)를 시간순으로 반환."""
    if not LEDGER_PATH.exists():
        return []
    episodes: list[dict] = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            episodes.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # 손상된 줄은 건너뜀 — 원장 전체를 잃지 않는다
    return episodes[-limit:] if limit else episodes


# ─────────────────────────────────────────────────────────
# 패턴 감지 — 재미 엔진의 입력
# ─────────────────────────────────────────────────────────
def detect_patterns(episodes: list[dict]) -> dict:
    """누적 에피소드에서 반복 신호를 집계.

    Returns:
        locations / pairs / running_gags / appearances / leads — 빈도순 [(name, count), ...]
        wallflowers — 자주 등장하지만 좀처럼 장면을 *주도하지 않는* 캐릭터
                      (= streak break 후보. 이 캐릭터가 먼저 말하면 사건이 된다)
    """
    locations    = Counter()
    pairs        = Counter()
    gags         = Counter()
    appearances  = Counter()
    leads        = Counter()

    for ep in episodes:
        if ep.get("location"):
            locations[ep["location"]] += 1
        if ep.get("pair"):
            pairs[ep["pair"]] += 1
        if ep.get("running_gag"):
            gags[ep["running_gag"]] += 1
        for c in ep.get("chars", []):
            appearances[c] += 1
        if ep.get("lead"):
            leads[ep["lead"]] += 1

    # 등장은 잦은데 주도는 드문 캐릭터 → 정체성 반전(streak) 후보
    wallflowers = [
        c for c, n in appearances.most_common()
        if n >= 3 and leads.get(c, 0) <= max(1, n // 4)
    ]

    return {
        "episode_count": len(episodes),
        "locations":     locations.most_common(),
        "pairs":         pairs.most_common(),
        "running_gags":  gags.most_common(),
        "appearances":   appearances.most_common(),
        "leads":         leads.most_common(),
        "wallflowers":   wallflowers,
    }


def related_episodes(pair: str = "", chars: list[str] | None = None, limit: int = 6) -> list[dict]:
    """Recent episodes related to a pair or any of its characters."""
    chars_set = {c for c in (chars or []) if c}
    pair = (pair or "").strip()
    out = []
    for ep in reversed(load_episodes()):
        if pair and ep.get("pair") == pair:
            out.append(ep)
        elif chars_set and chars_set.intersection(ep.get("chars", [])):
            out.append(ep)
        if len(out) >= limit:
            break
    return list(reversed(out))


def build_omnibus_memory(pair: str = "", chars: list[str] | None = None, limit: int = 6) -> str:
    """Prompt block for Friends-like soft continuity.

    Each new episode must stand alone, but it may reward longtime readers with one
    callback, running gag escalation, or relationship beat from recent related episodes.
    """
    episodes = related_episodes(pair=pair, chars=chars, limit=limit)
    if not episodes:
        return ""

    lines = [
        "\n=== OMNIBUS MEMORY — optional soft continuity ===",
        "Use this like Friends-style memory: every episode must stand alone, but one callback",
        "or running-gag escalation can make it funnier for returning readers.",
        "Rules:",
        "- Use AT MOST ONE callback from this block.",
        "- Do not explain the callback; make it work as normal character behavior.",
        "- Prefer escalating a character habit over referencing plot lore.",
        "- Never let continuity block the exact target sentence.",
        "",
        "Recent related episodes:",
    ]
    for ep in episodes:
        dialogue = " / ".join(ep.get("dialogue", [])[:2])
        game = ep.get("comedic_game") or {}
        lines.append(
            f"- [{ep.get('date')}] {ep.get('pair') or ','.join(ep.get('chars', []))} "
            f"@{ep.get('location', '?')}: {ep.get('sitcom_conflict') or ep.get('micro_situation', '')}"
        )
        if game.get("premise"):
            lines.append(f"  running gag seed: {game.get('premise')}")
        if ep.get("callback_seed"):
            lines.append(f"  callback seed: {ep.get('callback_seed')}")
        if dialogue:
            lines.append(f"  sample dialogue: {dialogue}")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────
# 단독 실행 — 원장 요약 출력
# ─────────────────────────────────────────────────────────
def main() -> None:
    eps = load_episodes()
    if not eps:
        print(f"📭 원장이 비어있습니다: {LEDGER_PATH}")
        return
    pat = detect_patterns(eps)
    print(f"📚 에피소드 {pat['episode_count']}건  ({eps[0]['date']} ~ {eps[-1]['date']})")
    print(f"\n📍 장소     : {pat['locations']}")
    print(f"💑 페어     : {pat['pairs']}")
    print(f"🔁 개그     : {pat['running_gags']}")
    print(f"🎭 등장     : {pat['appearances']}")
    print(f"🎬 주도     : {pat['leads']}")
    print(f"🌻 반전후보 : {pat['wallflowers']}")


if __name__ == "__main__":
    main()
