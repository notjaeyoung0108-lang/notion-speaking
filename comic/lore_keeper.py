"""lore_keeper.py — 세계관 확장 에이전트(쇼러너).

Writer가 매일 4컷을 쓰고 Archivist가 원장에 쌓으면,
Lore Keeper는 *주기적으로* 그 누적 기록을 성찰해 "작가의 방 메모"를 만든다.

재미 엔진 4종:
  1. 떡밥 회수 (callbacks)        — 과거 대사/장면을 오늘 터뜨릴 큐
  2. 정체성 반전 (streak_breaks) — 잘 안 나서던 캐릭터가 처음 나서는 순간
  3. 개그 진화 (gag_escalations) — 반복 개그를 한 단계 키우기
  4. 떡밥 폭발 (earned_phase_ups)— 조용한 +1이 아니라 "터지는 순간"으로 승급

산출물:
  showrunner_notes.md — 사람이 읽는 제안서 (캐논은 손으로 쓴 자산이라 기본은 제안만)
  writer_memo         — Writer 프롬프트에 되먹이는 2~4문장 (재미 루프를 닫는다)

Usage:
  python -m notion_words.comic.lore_keeper            # 메모 생성 → showrunner_notes.md
  python -m notion_words.comic.lore_keeper --limit 20 # 최근 20개만 성찰
  python -m notion_words.comic.lore_keeper --show     # 현재 메모만 출력
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from . import episode_log
from .lore_sanitize import sanitize_character_bible

HERE       = Path(__file__).parent
LORE_DIR   = HERE.parent / "lore"        # 정본: notion_words/lore/
_LORE_FILES = ["world.md", "characters.md", "situation.md", "episode_rules.md", "relationship_state.yaml"]
NOTES_PATH = HERE / "showrunner_notes.md"


def load_lore() -> str:
    """lore/ 디렉터리를 한 본문으로 이어붙인다 (회고용 — arc 상태 yaml 포함)."""
    parts = []
    for fn in _LORE_FILES:
        p = LORE_DIR / fn
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            if fn == "characters.md":
                text = sanitize_character_bible(text)
            parts.append(f"=== {fn} ===\n" + text)
    return "\n\n".join(parts)

MODEL = os.getenv("LORE_MODEL", "gpt-4o")


# ─────────────────────────────────────────────────────────
# .env 수동 로드 (generate_scenario.py와 동일 패턴)
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
# 프롬프트
# ─────────────────────────────────────────────────────────
def _build_prompt(lore: str, patterns: dict, episodes: list[dict]) -> str:
    # 최근 에피소드 대사를 압축해서 보여줌 — 떡밥 회수의 재료
    recent = episodes[-12:]
    ep_lines = []
    for e in recent:
        gag = f" | gag: {e['running_gag']}" if e.get("running_gag") else ""
        ep_lines.append(
            f"[{e['date']}] ({e.get('pair','?')}) @{e.get('location','?')} "
            f"— beat: {e.get('beat','')}{gag}\n    " +
            " / ".join(e.get("dialogue", [])[:4])
        )
    episodes_block = "\n".join(ep_lines) if ep_lines else "(아직 에피소드 없음)"

    return f"""You are the SHOWRUNNER of a recurring slice-of-life webtoon — the keeper of its canon
and the brain of its writers' room. You do NOT write episodes. You read what has ALREADY
happened across episodes and find the moments that will make the world funnier and more alive.

Your job is NOT bookkeeping. Promoting "a location appeared 3 times" into a table is boring.
The fun lives in MEMORY, CALLBACKS, PAYOFFS, and ESCALATION. A throwaway line from two weeks
ago resurfacing is the single most satisfying device in episodic comedy. Hunt for those.

You must NEVER invent canon that contradicts the established world or characters. You only
elevate patterns that the episodes ALREADY demonstrate. Stay in-character for this world.

=== CURRENT CANON (lore/) ===

{lore}

=== PATTERN REPORT (mechanical counts across the ledger) ===

episode_count : {patterns['episode_count']}
locations     : {patterns['locations']}
pairs         : {patterns['pairs']}
running_gags  : {patterns['running_gags']}
appearances   : {patterns['appearances']}
leads (who drove the scene) : {patterns['leads']}
wallflowers (appear often, rarely lead — STREAK-BREAK candidates) : {patterns['wallflowers']}

=== RECENT EPISODES (newest last) ===

{episodes_block}

=== YOUR TASK ===

Produce a writers' room memo using the FOUR fun engines. Be specific and reference real
episodes/lines. Quality over quantity — 1-3 strong items per engine, skip an engine if nothing
is genuinely earned yet.

1. callbacks       — a setup planted in a past episode that should pay off soon. Cite the source.
2. streak_breaks   — a wallflower whose first time leading/speaking-first would LAND, *because*
                     of all the episodes they stayed quiet. Explain why it lands.
3. gag_escalations — a running gag that should level up (not just repeat). Give the next level.
4. earned_phase_ups— a pair whose accumulated beats have EARNED a relationship phase shift.
                     Describe the specific "moment", not a vague +1.

Then write `writer_memo`: a punchy 2-4 sentence note (Korean ok) that will be injected directly
into the episode Writer's next prompt. It should make the NEXT episode reward a longtime reader.

=== OUTPUT — return ONLY this JSON ===

{{
  "callbacks":        [{{"setup": "...", "from_date": "...", "pair": "...", "payoff_cue": "..."}}],
  "streak_breaks":    [{{"char": "...", "streak": "...", "when": "...", "why_it_lands": "..."}}],
  "gag_escalations":  [{{"gag": "...", "pair": "...", "current_level": 1, "next_level": "..."}}],
  "earned_phase_ups": [{{"pair": "...", "reason": "...", "the_moment": "..."}}],
  "new_canon":        [{{"section": "Recurring Locations|Characters|...", "addition": "...", "evidence": "..."}}],
  "writer_memo":      "2-4 sentence note injected into the Writer's next prompt"
}}
"""


# ─────────────────────────────────────────────────────────
# 메모 → 마크다운 렌더 (사람이 읽는 제안서)
# ─────────────────────────────────────────────────────────
def _render_notes(memo: dict, patterns: dict) -> str:
    def section(title: str, items: list, fmt) -> str:
        if not items:
            return f"## {title}\n\n_(없음)_\n"
        body = "\n".join(fmt(x) for x in items)
        return f"## {title}\n\n{body}\n"

    parts = [
        "# 🎬 Showrunner Notes",
        f"\n> 누적 에피소드 {patterns['episode_count']}건 기준 자동 생성. "
        "캐논은 손으로 쓴 자산이므로 **제안**입니다 — 검토 후 lore/에 반영하세요.\n",
        section("🪝 떡밥 회수 (Callbacks)", memo.get("callbacks", []),
                lambda c: f"- **{c.get('pair','')}** — {c.get('setup','')}  "
                          f"_(from {c.get('from_date','?')})_\n  → 회수 큐: {c.get('payoff_cue','')}"),
        section("🌻 정체성 반전 (Streak Breaks)", memo.get("streak_breaks", []),
                lambda s: f"- **{s.get('char','')}** — {s.get('streak','')}\n"
                          f"  → 언제: {s.get('when','')}\n  → 왜 터지나: {s.get('why_it_lands','')}"),
        section("📈 개그 진화 (Gag Escalations)", memo.get("gag_escalations", []),
                lambda g: f"- **{g.get('pair','')}** — \"{g.get('gag','')}\" "
                          f"(Lv.{g.get('current_level','?')})\n  → 다음 레벨: {g.get('next_level','')}"),
        section("💥 떡밥 폭발 (Earned Phase-Ups)", memo.get("earned_phase_ups", []),
                lambda p: f"- **{p.get('pair','')}** — {p.get('reason','')}\n"
                          f"  → 그 순간: {p.get('the_moment','')}"),
        section("📖 캐논 추가 제안 (New Canon — 검토 필요)", memo.get("new_canon", []),
                lambda n: f"- `{n.get('section','')}` ← {n.get('addition','')}  "
                          f"_(근거: {n.get('evidence','')})_"),
        "## ✏️ Writer Memo (다음 화 프롬프트에 주입)\n\n> "
        + memo.get("writer_memo", "_(없음)_"),
    ]
    return "\n".join(parts) + "\n"


# ─────────────────────────────────────────────────────────
# Lore Keeper 에이전트
# ─────────────────────────────────────────────────────────
class LoreKeeper:
    def __init__(self) -> None:
        self.client = OpenAI()

    # ── tools (결정론적 입력 수집) ──
    def read_lore(self) -> str:
        return load_lore()

    def gather(self, limit: int | None) -> tuple[list[dict], dict]:
        episodes = episode_log.load_episodes(limit=limit)
        return episodes, episode_log.detect_patterns(episodes)

    # ── 창작 reasoning (GPT-5) ──
    def reflect(self, lore: str, patterns: dict, episodes: list[dict]) -> dict:
        resp = self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _build_prompt(lore, patterns, episodes)},
                {"role": "user", "content":
                    "Reflect on the ledger and produce the writers' room memo. "
                    "Reference real episodes. Skip any engine that hasn't been earned yet."},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    # ── orchestration ──
    def run(self, limit: int | None = None) -> dict:
        episodes, patterns = self.gather(limit)
        if not episodes:
            print(f"📭 원장이 비어있습니다 ({episode_log.LEDGER_PATH}). "
                  "Writer를 먼저 돌려 에피소드를 쌓으세요.")
            return {}
        print(f"🎬 쇼러너 성찰 중... (에피소드 {patterns['episode_count']}건)")
        memo = self.reflect(self.read_lore(), patterns, episodes)
        NOTES_PATH.write_text(_render_notes(memo, patterns), encoding="utf-8")
        print(f"✏️  메모 작성 완료 → {NOTES_PATH}")
        return memo


# ─────────────────────────────────────────────────────────
# Writer 되먹임 — 다음 화 프롬프트에 주입할 메모 읽기
# ─────────────────────────────────────────────────────────
def load_writer_memo() -> str:
    """showrunner_notes.md에서 Writer Memo 섹션만 추출. 없으면 빈 문자열."""
    if not NOTES_PATH.exists():
        return ""
    text = NOTES_PATH.read_text(encoding="utf-8")
    marker = "## ✏️ Writer Memo"
    idx = text.find(marker)
    if idx == -1:
        return ""
    # 헤더 뒤 본문 추출: "(다음 화 프롬프트에 주입)" 부제목과 "> " 인용 마커를 정확히 제거.
    # (lstrip 은 문자 집합으로 동작하므로 접두어 제거에는 removeprefix 를 쓴다)
    tail = text[idx + len(marker):].strip()
    tail = tail.removeprefix("(다음 화 프롬프트에 주입)").strip()
    return tail.lstrip(">").strip()


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="최근 N개 에피소드만 성찰")
    parser.add_argument("--show", action="store_true", help="현재 Writer Memo만 출력")
    args = parser.parse_args()

    _load_dotenv(HERE.parent.parent / ".env")

    if args.show:
        print(load_writer_memo() or "(메모 없음)")
        return

    LoreKeeper().run(limit=args.limit)


if __name__ == "__main__":
    main()
