"""run_ablation_experiment.py — "재미는 어느 입력이 짊어지는가" ablation.

가설: 고빈도 기능문장(학습문장)이 시트콤을 평평하게 만든다. 세계관/캐릭터가
재미를 짊어진다.

같은 엔진(gpt-4o)·같은 출력 스키마로 입력만 바꿔 비교한다:
  · 변종 A (sentence-only) : 목표문장+메타데이터만. 세계관/고정 캐스트 제거.
  · 변종 B (world-only)    : world.md+characters.md+arc 전부. 목표문장 의무 제거.

A는 서로 다른 문장 여러 개로 → "기능문장은 같은 장면으로 수렴하나?"
B는 여러 번 → "세계관만으로 변주가 나오나?"

Usage (notion-speaking 또는 J_0 루트에서):
  py -m notion_speaking.comic.run_ablation_experiment
  py -m notion_speaking.comic.run_ablation_experiment --b-runs 3
  # SSL 가로채기 환경이면:  OPENAI_SSL_VERIFY=0 py -m ...
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__:
    from . import generate_scenario as gs
    from .generate_scenario import _gpt_json, load_lore, load_domain_world
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from notion_speaking.comic import generate_scenario as gs
    from notion_speaking.comic.generate_scenario import _gpt_json, load_lore, load_domain_world

MODEL = "gpt-4o"  # 실제 파이프라인 MODEL_PLAN/MODEL_SCRIPT 와 동일 티어
OUT_DIR = Path(__file__).parent / "data" / "state"

# 공통 출력 스키마 — 두 변종이 똑같은 모양을 내야 공정 비교가 된다.
_SHARED_SHAPE = """Return JSON ONLY in this exact shape:
{
  "situation": "one vivid sentence describing the scene",
  "characters": ["name", "name"],
  "comedic_game": {
    "driver": "who drives the funny pressure",
    "premise": "one playable comic idea",
    "escalation": "how it sharpens across the 6 panels",
    "button": "final character-specific turn or residue"
  },
  "panels": [
    {"char": "speaker", "bubble": "English line", "bubble_kr": "natural Korean", "expression_intent": "short visible emotion"}
  ]
}
Exactly 6 panels. Character speakers alternate (never the same speaker twice in a row).
Keep every English line simple enough for an English learner (one idea per bubble)."""


# ── 변종 A: 문장만 (세계관/캐스트 제거) ──────────────────────────────
def build_prompt_A_sentence_only() -> str:
    return f"""You are a sitcom writer. Write ONE funny 6-panel comic strip.

You are given a TARGET ENGLISH SENTENCE that a learner is studying. Your strip must
make that exact sentence the most natural thing to say at its moment.

You have NO fixed world and NO recurring cast. Invent whatever throwaway characters,
setting, and situation best make this sentence land and be funny. Comedy first, but the
target sentence must appear VERBATIM exactly once in a spoken bubble (not paraphrased).

Build it as: situation -> pressure -> escalation -> the target line becomes inevitable -> button.
{_SHARED_SHAPE}"""


def user_msg_A(sentence: str, meta: dict) -> str:
    lines = [f"- {k}: {v}" for k, v in meta.items() if v]
    return (
        "=== TARGET SENTENCE ===\n"
        f'- sentence: "{sentence}"\n'
        + "\n".join(lines)
        + "\n\nWrite the funniest 6-panel strip where this exact sentence is the natural line."
    )


# ── 변종 B: 세계관만 (목표문장 의무 제거) ────────────────────────────
def build_prompt_B_world_only(lore: str, arc_prompt: str, nudge: str = "") -> str:
    return f"""You are a sitcom story writer for a slice-of-life workplace sitcom.
Write ONE funny 6-panel comic strip using the WORLD BIBLE and CAST below.

There is NO required learning sentence. Your only job is to be genuinely funny in the
voice of this world. Comedy must come from CHARACTER FACET COLLISION: two people bring
different habits/fears/filters to the same tiny problem, one person's quirk ESCALATES,
the other reacts, and the warmth lands only at the final button (do not resolve every panel).

Pick the cast and the ONE facet each character activates from the bible. Follow the
facet-rotation and overuse rules in the bible. {nudge}

=== WORLD BIBLE ===
{lore}

=== RELATIONSHIP ARC STATE ===
{arc_prompt or "(none)"}

{_SHARED_SHAPE}"""


# ── 변종 C: 세계관(수정본) + 캐릭터 + 도메인 주입 (목표문장 여전히 없음) ──
def build_prompt_C_world_plus_domain(lore: str, arc_prompt: str, domain_lore: str,
                                     domain: str, nudge: str = "") -> str:
    domain_block = (
        f"\n=== DOMAIN BIBLE — {domain} ===\n"
        "This scene happens in this domain. Use its specific texture, but DO NOT let domain\n"
        "detail flatten the comedy — facet collision still drives the funny.\n"
        f"{domain_lore}\n"
        if domain_lore else f"\n(no domain bible found for '{domain}')\n"
    )
    return f"""You are a sitcom story writer for a slice-of-life workplace sitcom.
Write ONE funny 6-panel comic strip using the WORLD BIBLE, CAST, and DOMAIN BIBLE below.

There is NO required learning sentence. Your only job is to be genuinely funny in the
voice of this world. Comedy must come from CHARACTER FACET COLLISION: two people bring
different habits/fears/filters to the same tiny problem, one quirk ESCALATES, the other
reacts, warmth lands only at the final button.

Pick the cast and the ONE facet each character activates. Follow facet-rotation/overuse rules. {nudge}

=== WORLD BIBLE ===
{lore}

=== RELATIONSHIP ARC STATE ===
{arc_prompt or "(none)"}
{domain_block}
{_SHARED_SHAPE}"""


# ── 작가(Writer): 세계관+도메인+타겟문장 → 바로 6컷 대본 (플래너+대사 통합) ──
def build_writer_prompt(lore: str, domain_lore: str, domain: str, sentence: str, meta: dict) -> str:
    domain_block = f"\n=== DOMAIN BIBLE — {domain} ===\n{domain_lore}\n" if domain_lore else ""
    meta_lines = "\n".join(f"- {k}: {v}" for k, v in meta.items() if v)
    return f"""You are the WRITER for a slice-of-life workplace sitcom that teaches English.
Write ONE funny 6-panel episode AS A SCRIPT — real dialogue. Just write the scene; do NOT fill a
beat-sheet and do NOT plan slots.

Comedy FIRST, from CHARACTER FACET COLLISION (two facets from the bible clash over one tiny
problem; one quirk escalates, the other reacts, warmth only at the final button). Start from
character friction, NOT from the sentence. YOU choose the funniest cast pairing — do NOT default
to the explainer->confused-listener pairing.

The target sentence must appear VERBATIM exactly once. Let it fall WHEREVER it is most natural —
usually MID-SCENE at the peak of the collision (when a character would genuinely blurt it), NOT
forced onto the final button or a fixed slot. If it feels tacked on, rewrite the scene so the
line becomes the thing that character truly has to say right then.
EVEN IF it is a clarification/question, do NOT build a teaching scene — give a non-teaching reason
(absurd claim, prank, distraction, competition, refusing a ridiculous answer).
Keep every English line simple (one idea per bubble). Alternate speakers — NEVER the same speaker
twice in a row.

=== WORLD + CHARACTER BIBLE ===
{lore}
{domain_block}
=== TARGET SENTENCE ===
- sentence: "{sentence}"
{meta_lines}

Return JSON ONLY:
{{
  "situation": "one vivid sentence",
  "characters": ["name", "name"],
  "comedic_game": {{"driver": "", "premise": "", "escalation": "", "button": ""}},
  "panels": [
    {{"char": "speaker name, or empty for object panel",
      "panel_type": "character | object",
      "bubble": "English line; empty for silent object",
      "bubble_kr": "natural Korean; empty for object/narration",
      "expression_intent": "short visible emotion"}}
  ]
}}
Exactly 6 panels. The target sentence "{sentence}" must appear verbatim once, wherever it lands most naturally."""


# ── 연출(Direction): 완성된 대본 → 패널별 스테이징(표정/포즈/프레이밍/배경) ──
def build_direction_prompt(situation: str, script_json: str, expression_menu: str,
                           motion_menu: str, location_menu: str, domain: str) -> str:
    return f"""You are the DIRECTOR (연출) for a webtoon. You are given a FINISHED 6-panel script.
Do NOT change the dialogue or the story. For EACH panel, decide the staging so the image explains
why the line is said.

Scene: {situation}

FIRST choose ONE location tag for the WHOLE episode from this domain ({domain}); use that exact
tag in every panel:
{location_menu}

HARD RULES (the values are consumed by an image pipeline — they MUST be exact keys):
- expression: EXACTLY ONE key from the EXPRESSION MENU, verbatim (character panels). It must CHANGE
  with the mood panel to panel; do not repeat the resting face. "none" for object panels.
- body_pose: EXACTLY ONE key from BODY_POSES. gesture: EXACTLY ONE key from GESTURES (or "none").
  "none"/"none" for object panels. Do NOT invent keys or use free text.
- gaze: one of: looking at viewer | looking to the side | looking down | looking up | looking away.
- framing: full_body | waist_shot | upper_body | close_up | object_close_up.
- location: the SAME chosen location tag in EVERY panel.
- background_prop: 0-2 simple scene objects, or "".
- object panels: fill "subject" (visible object/state) and "action" (object-state tags like
  "placed on desk", "screen lit up"); no person, no body verbs.
- prop_interaction: short object tags only if the acting needs a visible prop, else "".
- visual_note: one short line — what the viewer should read.

=== EXPRESSION MENU (key: when to use) ===
{expression_menu}

=== MOTION MENU ===
{motion_menu}

FINISHED SCRIPT (panels in order):
{script_json}

Return JSON ONLY, panels aligned 1:1 with the script:
{{ "panels": [ {{"expression": "menu_key_or_none", "body_pose": "key_or_none",
  "gesture": "key_or_none", "gaze": "gaze tag", "framing": "framing_key",
  "location": "location tag", "background_prop": "objects or empty",
  "subject": "object-panel subject or empty", "action": "object-state tags (object panels) or empty",
  "prop_interaction": "tags or empty", "visual_note": "what the viewer reads"}} ] }}"""


def _to_render_panels(script: dict, direction: dict, domain: str, sentence: str, seed: int) -> list[dict]:
    """작가+연출 → sd_generate_local 이 먹는 렌더 패널 포맷 (프로덕션 리졸버 재사용)."""
    sp = script.get("panels") or []
    dp = direction.get("panels") or []
    # 한 화 = 한 장소: 연출이 고른 첫 location 으로 고정.
    loc = next((str(d.get("location") or "").strip() for d in dp if str(d.get("location") or "").strip()), "")
    bg_prop = next((str(d.get("background_prop") or "").strip() for d in dp if str(d.get("background_prop") or "").strip()), "")
    background = gs.resolve_location(loc, domain, bg_prop or None)
    setting = gs.resolve_location_outfit_setting(loc, domain) or "daily_outing"
    out = []
    for i, p in enumerate(sp):
        d = dp[i] if i < len(dp) else {}
        offset = gs.SEED_OFFSETS[i % len(gs.SEED_OFFSETS)]
        char_raw = str(p.get("char") or "").strip()
        is_obj = (p.get("panel_type") == "object") or not char_raw
        common = {
            "framing": d.get("framing") or ("object_close_up" if is_obj else "upper_body"),
            "background": background, "location": loc, "used_in": domain,
            "target_sentence": sentence,
            "bubble": p.get("bubble", ""), "bubble_kr": p.get("bubble_kr", ""),
            "seed_offset": offset,
        }
        if is_obj:
            out.append({**common, "panel_type": "object", "char": "", "outfit": "",
                        "action": (d.get("action") or "placed on table"),
                        "body_pose": "none", "gesture": "none",
                        "subject": d.get("subject") or "object",
                        "expression": "", "face_state": ""})
        else:
            char = gs._canonical_known_char(char_raw)
            out.append({**common, "panel_type": "character", "char": char,
                        "outfit": gs._pick_char_outfit(char, setting, seed),
                        "action": gs.resolve_motion(d.get("body_pose"), d.get("gesture")),
                        "body_pose": d.get("body_pose") or "standing",
                        "gesture": d.get("gesture") or "none",
                        "subject": "",
                        "expression": gs.resolve_expression(d.get("expression")),
                        "face_state": (d.get("gaze") or "looking at viewer").replace("_", " ")})
    return out


def _norm(t: str) -> str:
    return " ".join("".join(c for c in (t or "").lower() if c.isalnum() or c.isspace()).split())


def _print_script_direction(tag: str, script: dict, direction: dict, sentence: str = "") -> None:
    print(f"\n{'='*72}\n[{tag}]  {script.get('situation','')}")
    g = script.get("comedic_game") or {}
    print(f"  game: {g.get('driver','')} -> {g.get('premise','')} | button: {g.get('button','')}")
    sp = script.get("panels") or []
    dp = direction.get("panels") or []
    target = _norm(sentence)
    for i, p in enumerate(sp):
        d = dp[i] if i < len(dp) else {}
        is_target = bool(target) and target in _norm(p.get("bubble", ""))
        star = " ★TARGET" if is_target else ""
        who = p.get("char") or "(object)"
        print(f"\n  [{i+1}] {who}{star}")
        if p.get("bubble"):
            print(f"      대사: {p.get('bubble','')}")
            print(f"            {p.get('bubble_kr','')}")
        if p.get("panel_type") == "object" or not p.get("char"):
            print(f"      연출(object): subject={d.get('subject','')} | loc={d.get('location','')} | {d.get('framing','')}")
        else:
            print(f"      연출: expr={d.get('expression','')} | pose={d.get('body_pose','')}/{d.get('gesture','')}"
                  f" | gaze={d.get('gaze','')} | framing={d.get('framing','')}")
            print(f"            prop={d.get('prop_interaction','') or '-'} | loc={d.get('location','')}")
        if d.get("visual_note"):
            print(f"            ↳ {d.get('visual_note','')}")


# ── 수확(Harvest): 완성된 장면 → 재사용 가능한 고빈도 표현 추출 ──
def build_harvest_prompt() -> str:
    return """You are an English-learning editor for a Korean learner who wants to absorb
LOTS of natural, high-frequency everyday English (NOT a fixed test list). The learner's goal
is general fluency/intuition through volume of natural input.

You are given a finished 6-panel sitcom scene. HARVEST the 3-5 most useful everyday
expressions the learner should steal and reuse in real life.

Rules:
- Pick HIGH-FREQUENCY, reusable chunks/sentences people actually say a lot.
- Prefer transferable patterns over scene-specific nouns or one-off jokes.
- Give the CLEAN reusable form: strip scene-specific words, keep the usable pattern
  (e.g. "It looks... greenish." -> "It looks kind of [adjective]." ; keep "Any X is fine.").
- For each: natural Korean, a one-line "when to use it", and register.
- Then pick ONE headline expression (single most worth memorizing) for a review card.

Return JSON ONLY:
{
  "harvest": [
    {"expression": "clean reusable English", "korean": "자연스러운 한국어",
     "when_to_use": "이럴 때 쓴다 (한 줄)", "register": "casual|neutral|polite"}
  ],
  "headline": "the single most useful expression, verbatim"
}"""


def _scene_to_text(obj: dict) -> str:
    lines = [f"Situation: {obj.get('situation','')}"]
    for i, p in enumerate(obj.get("panels") or [], 1):
        lines.append(f'{i}. {p.get("char","")}: {p.get("bubble","")}')
    return "\n".join(lines)


def _print_harvest(harvest: dict) -> None:
    print("\n── 🌾 수확 결과 ──")
    for h in harvest.get("harvest") or []:
        print(f"  • {h.get('expression','')}   [{h.get('register','')}]")
        print(f"      {h.get('korean','')}")
        print(f"      쓰임: {h.get('when_to_use','')}")
    print(f"\n  ★ 복습카드 headline: {harvest.get('headline','')}")


# ── 변종 D: 학습문장 + 캐릭터 + 도메인 (원래 파이프라인 입력 조합 재현) ──
def build_prompt_D_full(lore: str, arc_prompt: str, domain_lore: str, domain: str,
                        sentence: str, meta: dict, nudge: str = "") -> str:
    domain_block = (
        f"\n=== DOMAIN BIBLE — {domain} ===\n{domain_lore}\n" if domain_lore
        else f"\n(no domain bible for '{domain}')\n"
    )
    meta_lines = "\n".join(f"- {k}: {v}" for k, v in meta.items() if v)
    return f"""You are a sitcom story writer for a slice-of-life workplace sitcom.
Write ONE funny 6-panel comic strip using the WORLD BIBLE, CAST, and DOMAIN BIBLE below.

There IS a target learning sentence. It must appear VERBATIM exactly once in a spoken
bubble (not paraphrased). But comedy comes FIRST and from CHARACTER FACET COLLISION —
do not let the sentence flatten the scene into a status-check or explanation exchange.
Pick the cast and the ONE facet each activates so this exact sentence becomes the
natural thing that character says under pressure. {nudge}

=== WORLD BIBLE ===
{lore}

=== RELATIONSHIP ARC STATE ===
{arc_prompt or "(none)"}
{domain_block}
=== TARGET SENTENCE ===
- sentence: "{sentence}"
{meta_lines}

{_SHARED_SHAPE}
The target sentence "{sentence}" MUST appear verbatim once in exactly one bubble."""


# A 변종용 목표문장들 — 실제 산출물에서 평평했던 고빈도 기능문장.
_SENTENCES = [
    ("I'll get back to you on that.", {
        "speech_act": "delay_answer", "politeness": "softened",
        "domain": "workplace", "relationship": "coworker_to_coworker",
        "korean_cue": "그건 다시 알려줄게.",
    }),
    ("Can you say that again?", {
        "speech_act": "clarification", "politeness": "polite",
        "domain": "academic", "relationship": "student_to_peer",
        "korean_cue": "다시 말해줄 수 있어?",
    }),
]


def _print_strip(tag: str, obj: dict) -> None:
    print(f"\n{'='*70}\n[{tag}]  situation: {obj.get('situation','')}")
    g = obj.get("comedic_game") or {}
    print(f"  driver={g.get('driver','')} | premise={g.get('premise','')}")
    print(f"  escalation={g.get('escalation','')}")
    print(f"  button={g.get('button','')}")
    print(f"  cast={obj.get('characters')}")
    for i, p in enumerate(obj.get("panels") or [], 1):
        print(f"   {i}. {p.get('char',''):10s} | {p.get('bubble','')}")
        print(f"      {'':10s} | {p.get('bubble_kr','')}  ({p.get('expression_intent','')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-runs", type=int, default=2, help="변종 B 반복 횟수")
    ap.add_argument("--c-runs", type=int, default=0, help="변종 C(세계관+도메인) 반복 횟수")
    ap.add_argument("--domain", default="academic",
                    help="변종 C에 주입할 도메인 (workplace|daily|academic|customer/service)")
    ap.add_argument("--skip-a", action="store_true", help="변종 A 생략")
    ap.add_argument("--skip-b", action="store_true", help="변종 B 생략")
    ap.add_argument("--run-d", action="store_true",
                    help="변종 D(학습문장+캐릭터+도메인) 실행 — 문장별 자기 도메인 주입")
    ap.add_argument("--demo-harvest", action="store_true",
                    help="C식 장면 1개 생성 → 수확 단계 시연")
    ap.add_argument("--writer-direction", action="store_true",
                    help="작가(자유 대본) → 연출(스테이징) 2단계 시연")
    args = ap.parse_args()

    lore = load_lore()                       # world.md + characters.md + episode_rules.md
    arc_prompt = ""
    try:
        from .generate_scenario import _parse_arc_state
        from .scenario_prompts import build_arc_prompt
        arc_prompt = build_arc_prompt(_parse_arc_state())
    except Exception as exc:
        print(f"  (arc state 생략: {exc})")

    # ── 작가 → 연출 시연 ──
    if args.writer_direction:
        expr_menu = gs.build_expression_menu()
        motion_menu = gs.build_motion_menu()
        out, render_batch = [], []
        for idx, (sentence, meta) in enumerate(_SENTENCES):
            dom = meta.get("domain", "")
            d_lore = load_domain_world(dom)
            loc_menu = gs.build_location_menu(dom)
            seed = 1235 + idx
            print(f"\n>>> [작가] 대본 생성: \"{sentence}\" (domain={dom}) ({MODEL})")
            script = _gpt_json(build_writer_prompt(lore, d_lore, dom, sentence, meta),
                               "Write the funniest 6-panel script now. Output JSON only.", model=MODEL)
            print(f">>> [연출] 스테이징 생성 ({MODEL})")
            direction = _gpt_json(
                build_direction_prompt(script.get("situation", ""),
                                       json.dumps(script.get("panels", []), ensure_ascii=False),
                                       expr_menu, motion_menu, loc_menu, dom),
                "Direct every panel now. Output JSON only.", model=MODEL)
            _print_script_direction(f"\"{sentence}\" · {dom}", script, direction, sentence)
            panels = _to_render_panels(script, direction, dom, sentence, seed)
            out.append({"sentence": sentence, "domain": dom, "script": script,
                        "direction": direction, "render_panels": panels})
            render_batch.append({"panels": panels, "seed": seed,
                                 "subdir": f"writer_{'_'.join(dict.fromkeys(p['char'] for p in panels if p.get('char')))}_{seed}"})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%y.%m.%d_%H%M%S")
        out_path = OUT_DIR / f"writer_direction_demo-{stamp}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        batch_path = OUT_DIR / f"writer_render_batch-{stamp}.json"
        batch_path.write_text(json.dumps(render_batch, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 대본+연출: {out_path}")
        print(f"💾 렌더 배치: {batch_path}")
        print(f"\n다음으로 이미지 렌더:\n  cd comic && py sd_generate_local.py --batch-json-path \"{batch_path}\"")
        return

    # ── 수확 시연: C 장면 생성 → 수확 ──
    if args.demo_harvest:
        d_lore = load_domain_world(args.domain)
        print(f"\n>>> [demo] C식 장면 생성 (domain={args.domain}) ({MODEL})")
        scene = _gpt_json(
            build_prompt_C_world_plus_domain(lore, arc_prompt, d_lore, args.domain, ""),
            "Write the funniest stand-alone 6-panel strip now.", model=MODEL)
        _print_strip(f"C scene · {args.domain}", scene)
        print(f"\n>>> [demo] 수확 호출 ({MODEL})")
        harvest = _gpt_json(build_harvest_prompt(), _scene_to_text(scene), model=MODEL)
        _print_harvest(harvest)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%y.%m.%d_%H%M%S")
        out_path = OUT_DIR / f"harvest_demo-{stamp}.json"
        out_path.write_text(json.dumps({"scene": scene, "harvest": harvest},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 저장: {out_path}")
        return

    results: dict = {
        "model": MODEL,
        "variant_A_sentence_only": [],
        "variant_B_world_only": [],
        "variant_C_world_plus_domain": [],
        "variant_D_sentence_char_domain": [],
    }

    # ── 변종 A ──
    sys_A = build_prompt_A_sentence_only()
    for sentence, meta in ([] if args.skip_a else _SENTENCES):
        print(f"\n>>> [A] sentence-only 호출: \"{sentence}\" ({MODEL})")
        try:
            obj = _gpt_json(sys_A, user_msg_A(sentence, meta), model=MODEL)
            obj["_target_sentence"] = sentence
            results["variant_A_sentence_only"].append(obj)
            _print_strip(f"A · \"{sentence}\"", obj)
        except Exception as exc:
            print(f"  ⚠️ A 실패: {exc}")

    # ── 변종 B ──
    nudges = [
        "",
        "Use a different location and a different pair than a typical office desk scene.",
        "Center hyo-jeong (the off-company friend) this time.",
    ]
    for r in range(0 if args.skip_b else args.b_runs):
        nudge = nudges[r] if r < len(nudges) else ""
        print(f"\n>>> [B] world-only 호출 #{r+1} ({MODEL})")
        try:
            obj = _gpt_json(build_prompt_B_world_only(lore, arc_prompt, nudge),
                            "Write the funniest stand-alone 6-panel strip now.", model=MODEL)
            results["variant_B_world_only"].append(obj)
            _print_strip(f"B · run {r+1}", obj)
        except Exception as exc:
            print(f"  ⚠️ B 실패: {exc}")

    # ── 변종 C: 세계관(수정본) + 캐릭터 + 도메인 주입 ──
    domain_lore = load_domain_world(args.domain)
    if args.c_runs and not domain_lore:
        print(f"  ⚠️ 도메인 '{args.domain}' lore를 못 찾음 — C는 도메인 없이 돈다")
    for r in range(args.c_runs):
        nudge = nudges[r] if r < len(nudges) else ""
        print(f"\n>>> [C] world+domain({args.domain}) 호출 #{r+1} ({MODEL})")
        try:
            obj = _gpt_json(
                build_prompt_C_world_plus_domain(lore, arc_prompt, domain_lore, args.domain, nudge),
                "Write the funniest stand-alone 6-panel strip now.", model=MODEL)
            obj["_domain"] = args.domain
            results["variant_C_world_plus_domain"].append(obj)
            _print_strip(f"C · {args.domain} · run {r+1}", obj)
        except Exception as exc:
            print(f"  ⚠️ C 실패: {exc}")

    # ── 변종 D: 학습문장 + 캐릭터 + 도메인 (문장별 자기 도메인) ──
    if args.run_d:
        for sentence, meta in _SENTENCES:
            dom = meta.get("domain", "")
            d_lore = load_domain_world(dom)
            print(f"\n>>> [D] full(문장+캐릭터+도메인:{dom}) 호출: \"{sentence}\" ({MODEL})")
            try:
                obj = _gpt_json(
                    build_prompt_D_full(lore, arc_prompt, d_lore, dom, sentence, meta),
                    "Write the funniest 6-panel strip where this exact sentence is the natural line.",
                    model=MODEL)
                obj["_target_sentence"] = sentence
                obj["_domain"] = dom
                results["variant_D_sentence_char_domain"].append(obj)
                _print_strip(f"D · \"{sentence}\" · {dom}", obj)
            except Exception as exc:
                print(f"  [!] D failed: {exc}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y.%m.%d_%H%M%S")
    out_path = OUT_DIR / f"ablation_experiment-{stamp}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n\n💾 저장: {out_path}")
    print(f"   A {len(results['variant_A_sentence_only'])}편 / "
          f"B {len(results['variant_B_world_only'])}편 / "
          f"C {len(results['variant_C_world_plus_domain'])}편 / "
          f"D {len(results['variant_D_sentence_char_domain'])}편")


if __name__ == "__main__":
    main()
