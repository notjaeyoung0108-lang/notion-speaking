"""scenario_prompts.py — GPT-4o 시나리오 생성용 프롬프트 템플릿."""

from __future__ import annotations


# ─────────────────────────────────────────────────────────
# Outfit Mapping
# ─────────────────────────────────────────────────────────

# 도메인(workplace/daily/academic …) → 의상 setting 접두어.
# (상황 라이브러리 폐지 — 의상은 이제 plan.domain 만으로 결정한다.)
# 선택 코드는 이 접두어로 시작하는 의상 중 '등장 캐릭터 전원 공통'인 것을 고른다.
CATEGORY_DEFAULT_SETTING: dict[str, str] = {
    "workplace": "workplace", "academic": "academic", "daily": "daily_outing",
    "customer/service": "daily_outing", "personal": "daily_dressup", "social": "daily_outing",
}


# ─────────────────────────────────────────────────────────
# Word Block
# ─────────────────────────────────────────────────────────

def build_word_block(word: dict) -> str:
    target = word.get("collocation unit") or word.get("sentence_unit") or word.get("sentence unit", "")
    meaning = word.get("meaning") or word.get("korean_trigger") or word.get("Korean trigger", "")
    nuance = word.get("nuance (Korean)") or word.get("micro_situation") or word.get("micro situation", "")
    translation = word.get("translation") or word.get("korean_trigger") or word.get("Korean trigger", "")
    metadata = word.get("scenario metadata") or {}
    primary_used_in = word.get("primary_used_in") or word.get("used in") or ""
    used_in = word.get("used_in") or primary_used_in
    speaker_role = word.get("speaker_role") or word.get("speaker role") or metadata.get("speaker role", "")
    listener_role = word.get("listener_role") or word.get("listener role") or metadata.get("listener role", "")
    relationship = word.get("relationship") or word.get("relationship context") or metadata.get("relationship context", "")
    power_dynamic = word.get("power_dynamic") or word.get("power dynamic") or metadata.get("power dynamic", "")
    speech_act = word.get("speech_act") or word.get("speech act") or metadata.get("speech act", "")
    politeness = word.get("politeness") or metadata.get("politeness", "")
    story_function = word.get("story_function") or word.get("story function") or metadata.get("story function", "")
    return f"""
=== Target Expression ===

- sentence        : {target}
- register        : {word.get('register', '')}
- primary_used_in : {primary_used_in}
- used_in         : {used_in}
- speaker_role    : {speaker_role}
- listener_role   : {listener_role}
- relationship    : {relationship}
- power_dynamic   : {power_dynamic}
- speech_act      : {speech_act}
- politeness      : {politeness}
- story_function  : {story_function}
- Korean cue      : {meaning}
- situation       : {nuance}
- translation     : {translation}

The relationship metadata is the source of truth. Do not contradict speaker_role, listener_role, relationship, power_dynamic, or speech_act.
"""


# ─────────────────────────────────────────────────────────
# Collocation Rules
# ─────────────────────────────────────────────────────────

def build_word_rule(collocation: str) -> str:
    return (
        f'- The exact target sentence "{collocation}" MUST appear verbatim in one spoken English bubble\n'
        f'- The sentence must be spoken by a character, never placed in an object/caption panel\n'
        f'- Choose the episode conflict so this exact sentence is something that character would genuinely say\n'
        f'- Do NOT paraphrase, shorten, split, or grammatically alter the sentence\n'
        f'- Do NOT directly explain the sentence\n'
        f'- Avoid textbook-style delivery\n'
        f'- Write natural spoken dialogue only'
    )


# ─────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────

def build_arc_prompt(arc_state: dict) -> str:
    """Format current relationship arc state into a system prompt block."""
    if not arc_state:
        return ""
    _PHASE_LABEL = {1: "현상유지", 2: "균열/긴장", 3: "전환", 4: "가까워짐"}

    def _fmt(v):
        return " / ".join(v) if isinstance(v, (list, tuple)) else str(v)

    lines = []
    for pair, state in arc_state.items():
        phase = state.get("phase", 1)
        comfort = state.get("comfort_level")
        last_beat = state.get("last_beat") or "none yet"
        running_gag = state.get("running_gag") or "none yet"
        signature_bit = state.get("signature_bit")
        dynamic = state.get("dynamic") or []
        unresolved = state.get("unresolved") or []
        head = f"{pair}  [phase {phase} — {_PHASE_LABEL.get(phase, phase)}"
        head += f", comfort {comfort}]" if comfort is not None else "]"
        lines.append(head)
        if dynamic:
            lines.append(f"  dynamic: {_fmt(dynamic)}")
        if signature_bit:
            lines.append(f"  signature_bit: {signature_bit}")
        lines.append(f"  last_beat: {last_beat}")
        lines.append(f"  running_gag: {running_gag}")
        if unresolved:
            lines.append(f"  unresolved: {_fmt(unresolved)}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# 3단 파이프라인 프롬프트 (Story Planner → Script → Visual)
# 관심사 분리 — 각 GPT 호출이 한 가지만 잘하도록
# ═════════════════════════════════════════════════════════════

_TONE_MAP = {
    # register 값 — collocations 의 'register' 컬럼(informal/standard/formal).
    # casual/sarcastic/emotional/blunt 는 현재 프롬프트가 생성하지 않지만 수동 지정 대비 유지.
    "informal":  "Overall tone: informal and relaxed — casual spoken language.",
    "standard":  "Overall tone: neutral, natural everyday speech.",
    "casual":    "Overall tone: casual and relaxed.",
    "formal":    "Overall tone: slightly formal/professional but still plain.",
    "sarcastic": "Overall tone: lightly dry/sarcastic — understated.",
    "emotional": "Overall tone: emotionally warm — speaking from feeling.",
    "blunt":     "Overall tone: blunt and direct, no softening.",
}


def tone_rule_for(register: str = "") -> str:
    if not register:
        return ""
    return _TONE_MAP.get(register.lower().strip(), f"Overall tone: {register}.")


def _revision_block(feedback: str) -> str:
    if not feedback:
        return ""
    return ("=== REVISION NOTES (previous draft FAILED review — fix these) ===\n\n"
            + feedback + "\n\n")


# ── ① Story Planner — 구조/기획만 ──────────────────────────────
def build_planner_prompt(
lore,
word_block,
word_rule,
cast_directive,
arc_prompt,
showrunner_notes="",
feedback="",
avoid_situations="",
cast_note="",
required_domain="",
location_menu="",
) -> str:
    _domain_lock = (
        (f"\n⛔ HARD REQUIREMENT — DOMAIN IS FIXED: this target sentence's domain is "
         f"\"{required_domain}\". The \"domain\" you return MUST be EXACTLY \"{required_domain}\" "
         f"(one of the four: workplace | daily | academic | customer/service). "
         f"Choose a location that genuinely belongs to "
         f"that domain. If the domain is NOT workplace, the scene must NOT happen at an office, "
         f"meeting room, Plaoud, or any desk/work setting — go to a real personal/shop/school place.\n")
        if required_domain else "")
    return f"""You are the STORY PLANNER for an ENGLISH-LEARNING webtoon.
{_domain_lock}


Your job is to plan ONE complete episode as a character-driven sitcom scene:
a tiny conflict that could only happen because these lore characters react in their own ways.

Output STRUCTURE ONLY.
DO NOT write dialogue.

Read the world bible carefully.

Priority order:

1. Episode Rules
2. Characters & World from the lore files — habits, fears, standards, relationships, and recurring dynamics
3. The exact target sentence — the episode must create a natural character move for that sentence
4. Target scenario metadata — relationship context, roles, power dynamic, speech act, service direction, story function
5. Relationship Arc State

{_revision_block(feedback)}

=== WORLD BIBLE ===

{lore}

{word_block}

Expression rules:
{word_rule}

{cast_directive}

=== RELATIONSHIP ARC STATE ===

{arc_prompt}

{("=== SHOWRUNNER NOTES ===" + chr(10) + showrunner_notes + chr(10)) if showrunner_notes else ""}

=== PLANNING RULES ===

Apply the world bible + episode_rules.md (above) — don't restate them. The relationship pair
and lead are PRE-SELECTED (see RELATIONSHIP note below); build the episode on that pair, using
only the domain's cast and locations from the domain world above.

Make the plan do this — and keep room to be creative within it:
- SEED THE SCENE FROM THE TARGET SENTENCE'S micro_situation (the "situation" field above). That real-life
  moment is the SPINE — the concrete reason a real person would say THIS exact sentence. Build the episode AS
  a vivid, specific realization of that moment, then layer the selected character's trait ON TOP: the comedy
  is how THIS character WARPS/heightens that same situation — NOT a different situation invented from their
  trait. Never swap the micro_situation for an unrelated game. (e.g. "What time do you finish?" / "친구의 일정을
  확인하고 싶을 때" → the scene is about checking when someone is free, NOT about critiquing a painting.)
- Use the target scenario metadata as constraints, not decoration:
  · relationship context = the social frame the selected pair must plausibly embody
  · speaker/listener roles = who says the exact target sentence and who receives it
  · power dynamic = who can pressure, refuse, correct, or decide
  · speech act = what the target sentence does in the exchange
  · service direction = whether the episode is customer→staff, staff→customer, internal team, teacher/student, or none
  · story function = where the target sentence belongs in the sitcom beat chain
- If a metadata field is blank because an older CSV is used, infer the missing value once and keep it consistent.
- Pick the clearest delivery_mode (character | object | action | setting | mixed) for the character conflict.
- 4–8 panels. Map the character-conflict beats onto a COMEDY shape (SREP), not a lesson-shaped scene:
    · situation  = SET-UP    (establish the micro_situation concretely + who's who)
    · pressure   = REVEAL    (the selected character's trait WARPS that situation — the GAME appears as a
                              heightening OF the micro_situation, never a new topic)
    · expression = LAND      (the EXACT TARGET SENTENCE is said here because the situation now DEMANDS it —
                              it performs its own speech_act naturally: a request asks, a refusal refuses, a
                              reassurance reassures. It may be the driver's line OR the other character's
                              line — whichever the speech_act + speaker_role require. Do NOT force it to be an
                              "exasperated reaction" when its speech_act is not a reaction.)
    · confirmation = PAYOFF  (the next honest consequence of the driver facet — resolve, dodge,
                              escalate, undercut, or leave awkward residue)
  Extra panels = more escalation, never a new topic.
- THE GAME OF THE SCENE: pick ONE character whose facet drives an absurd, SPECIFIC premise, and
  have them DOUBLE DOWN — each beat a bigger variation of the SAME premise — while the other
  reacts/condemns/tries to stop it. Comedy comes from heightening one thing, not many jokes.
- Do not plan by desired line type or scene-ending category.
  Plan the psychological cause: what does THIS character want, protect, avoid, or over-control
  in this exact moment? The target sentence should be the most character-plausible move available.
- The character clash is the SPINE: the exact target sentence must fall out of how THIS pair
  clashes. Never bolt the sentence on the side as a learning requirement.
- The speaker of the exact target sentence must match the implied speaker role and speech act.
  Do not give a customer complaint line to a staff helper, a junior request to a senior, or a
  button reaction to the setup beat unless the metadata says so.
- Echo the target metadata into the top-level JSON fields exactly. These fields must match the
  word metadata unless you explicitly explain a safe correction in "metadata_correction".
- target_speaker must be one of selected_pair's two characters.
- target_listener should be the other selected_pair character unless the scene is clearly monologue/action-led.
- BE SPECIFIC, not generic: a weird exact object/number/claim is funnier than a vague one
  ("Grandma's 1973 kimchi jar", "soda and a potato peeler" — not "something fragile", "some tool").
- Make the visible_learning_moment something a viewer literally SEES. Use an object/caption panel
  (speaker "") only when a screen/receipt/before-after state shows the character conflict faster than dialogue.
- The "situation" you invent must be a fresh, specific realization OF the micro_situation — not a stock
  setup, and never a substitute topic. The character bible + domain decide HOW it looks and who warps it;
  the micro_situation decides WHAT moment it is.

LOCATION — pick ONE "location" from THIS domain's allowed list (use the tag verbatim). The
earlier items are the canonical recurring stages; pick those by default, a later one only when
the character game truly fits it better (vary stages across episodes, don't always use the first):
{location_menu}
{cast_note}{avoid_situations}
Before returning, check:
- SITUATION MATCH: does the scene actually enact the micro_situation, or did you drift to an unrelated
  topic? (If the sentence is "몇 시에 끝나?" the scene MUST be about checking when someone is free.) If it
  drifted, the game is wrong — rebuild the game as a heightening of the micro_situation.
- COHERENCE: read the beats in order. Does each beat reply to the one before? Would a real person say the
  EXACT target sentence at its beat FOR THE REASON the micro_situation gives? If the target line is a
  non-sequitur in this conversation, do not ship it — rebuild around the micro_situation.
- Is the scene driven by the lore characters rather than a generic learner situation? For each character
  beat, can you answer "why would THIS character say/do this, instead of any generic speaker?"
- Does the visible_learning_moment actually show on screen? Is there ONE escalating game whose final beat
  follows from the driver facet? Are the details SPECIFIC, not generic? If not, revise.

=== RETURN JSON ONLY ===

{{
"target_sentence": "copy the exact target sentence",
"primary_used_in": "copy the word metadata primary_used_in / used in domain",
"speaker_role": "copy the word metadata speaker_role",
"listener_role": "copy the word metadata listener_role",
"relationship": "copy the word metadata relationship",
"power_dynamic": "copy the word metadata power_dynamic",
"speech_act": "copy the word metadata speech_act",
"politeness": "copy the word metadata politeness",
"story_function": "copy the word metadata story_function",
"selected_pair": "exact selected pair string from RELATIONSHIP note",
"target_speaker": "one character from selected_pair who speaks the exact target sentence",
"target_listener": "the other selected_pair character, unless monologue/action-led",
"metadata_correction": "empty string unless you safely corrected missing/contradictory metadata; explain the correction here",

"situation_id": "short slug for the situation you invented (e.g. clogged_sink_diy)",
"domain": "EXACTLY one of: workplace | daily | academic | customer/service",
"location": "ONE location TAG from the allowed list above, verbatim (e.g. cafe, office, classroom, park)",
"background_prop": "0 to 2 drawable background objects (comma-separated) that fit THIS scene's situation/game (generated, scene-specific — e.g. 'broken coffee machine', 'stack of receipts, sticky notes', 'whiteboard full of charts'). Objects only, no people. Use an empty string \"\" for a clean, prop-free location. Keep it short — at most two.",
"outfit_setting": "ONE wardrobe matching location & domain: workplace | academic | daily_home | daily_convenience | daily_outing | daily_dressup | daily_sport. Meanings: daily_home=at home only; daily_convenience=quick casual errand like convenience store/supermarket/street; daily_sport=exercise/outdoor activity; daily_outing=normal going out such as cafe/bookstore/bakery; daily_dressup=more dressed-up outing such as restaurant/movie/mall/salon/date-like place. Never use daily_home outside a home location.",
"situation": "one sentence: where, when, what problem",
"delivery_mode": "character-led | object-led | action-led | setting-led | mixed",

"target_sentence_context": {{
"relationship_context": "copy or infer one social frame for the episode",
"target_speaker_role": "which character fills the speaker role and why",
"target_listener_role": "which character fills the listener role and why",
"power_dynamic": "copy or infer the power dynamic and how it creates pressure",
"speech_act": "copy or infer the speech act performed by the exact target sentence",
"service_direction": "copy or infer: none | customer_to_staff | staff_to_customer | internal_team | teacher_to_student | student_to_teacher",
"story_function": "copy or infer the story function and ensure the collocation beat uses it"
}},

"nuance_structure": {{
"situation": "the ordinary state created by the selected character trait or relationship dynamic",
"pressure": "the specific pressure that makes the exact target sentence necessary",
"expression": "the concrete in-scene action/decision/realization where a character naturally says the exact target sentence (no meta commentary)",
"confirmation": "the visible result/reaction/aftermath that confirms the character conflict"
}},

"sitcom_conflict": "one sentence: the tiny warm character conflict that makes the target sentence necessary",

"comedic_game": {{
"driver": "the ONE character who drives the absurd premise",
"premise": "the single specific/absurd thing they insist on or keep doing (be concrete and weird)",
"escalation": "how it gets BIGGER beat by beat (variations of the SAME premise, not new jokes)",
"button": "the final line or image as the next honest move from the driver's facet. It may resolve, dodge, escalate, undercut, or leave awkward residue, but it must come from character causality, not scene-closing convenience."
}},

"character_filter_collision": [
{{"character": "name", "facet": "the habit/fear/standard this problem activates", "collision": "how it bumps against the other character/object/setting"}}
],

"visible_learning_moment": "the concrete visible before/after, decision, object state, or reaction that makes the target sentence feel inevitable without defining it",
"visible_proof_panel": "integer beat number (1..number of beats) or null. If chosen, that beat MUST be panel_type 'object', speaker '', has_collocation false, visual_focus a concrete drawable object/state. null if the conflict is purely interpersonal.",

"characters": ["0 to 3 names from the domain cast pool"],
"problem": "the single problem that drives the episode",

"beats": [
{{
"panel": 1,
"panel_type": "character | object",
"speaker": "name, or empty string for object/caption panels",
"nuance_role": "situation | pressure | expression | confirmation",
"visual_focus": "object panels: the exact object/state carrying the beat; character panels: the visible action",
"intent": "what this panel does for the character conflict + how the character uniquely reacts (one short line)",
"has_collocation": false
}}
],

"milestone": {{"is_milestone": false, "pair": "exact pair name or null", "summary": "one sentence if is_milestone else null"}}
}}
"""


# ── ② Script — 대사만 ─────────────────────────────────────────
_REGISTER_KR = {
    "informal": "Korean speech level: 반말 (casual) throughout — these people are close. Keep it consistent every bubble.",
    "standard": "Korean speech level: default to 반말 between these close friends/coworkers; use 존댓말 ONLY if the scene is clearly formal. Pick ONE level and keep it consistent every bubble.",
    "formal":   "Korean speech level: default to 반말 between these close friends/coworkers; use 존댓말 ONLY if the actual relationship/service situation clearly calls for polite speech. The English expression may be formal while the Korean relationship tone stays casual.",
}


def register_kr_rule(register: str = "") -> str:
    return _REGISTER_KR.get((register or "").lower().strip(), _REGISTER_KR["standard"])


def build_script_prompt(lore, word_block, plan_json, tone_rule="", feedback="", register="") -> str:
    return f"""You are the DIALOGUE WRITER for an ENGLISH-LEARNING webtoon (Korean learners).
You are given a SCENE BRIEF (prose). Write ONLY the dialogue. Read the brief for tone, character,
and what each beat means — then follow its BEAT-BY-BEAT skeleton EXACTLY (one bubble per character
beat, in that speaker order; object beats stay silent).

⭐ TOP PRIORITY: every line is instantly understandable to a learner.
- Use ONLY common, everyday English (textbook / daily-conversation level).
- NO jargon of any kind — not corporate (deck, ROI, pivot, sync), not hobby/fan
  ("bridge", "fancam", dance-move names), not obscure slang. A superfan/expert speaks in
  PLAIN words ("my favorite singer's old performance", NOT "the 2019 bridge").
- Short, clear sentences. One beat per bubble (no cramming greeting+situation+feeling).

{_revision_block(feedback)}Follow the brief:
- PANEL 1 must establish the situation from the dialogue alone — the reader sees only
  the bubbles + pictures, never the brief. Make who/where/what instantly clear in line 1.
- Each "Panel N — <speaker> (<role>)" line in the brief is one bubble spoken by that speaker.
  "[SILENT OBJECT]" beats are silent cutaways: emit char "", bubble "", bubble_kr "".
  Do NOT move the exact target sentence into a silent object beat.
- Each line REPLIES to the line before it. Stay on the ONE topic.
- Preserve the brief's beat order and roles: situation → pressure → expression → confirmation
  (extra panels deepen the same character pressure, never a new topic).
- The exact target sentence lands on the beat marked "<<< the target sentence lands HERE", spoken by
  THAT beat's character — naturally. NEVER write a line whose only job is to display the
  sentence; it must fall out of a real exchange. Do NOT move it to another character.
- Copy the target sentence verbatim, including contractions and word order. Do NOT paraphrase,
  shorten, split, tense-shift, or add words inside it.
- Character voice must make the sentence feel motivated by this specific character's personality
  and relationship, not by a generic learning scenario.
- For object-led panels, do not write captions. Let the object/state carry the meaning.
- The exact target sentence must appear in a character's spoken dialogue, not in an object panel.
- Final panel = the next honest character beat after the target sentence, legible at a glance.
  It may resolve, dodge, escalate, undercut, or leave small awkward residue. Choose the move THIS
  character would actually make.
- Characters use their behavior patterns (see bible) ONLY where it fits — never force a catchphrase.
- Each speaking character's chosen facet from character_filter_collision MUST appear at least
  once in what they say or how they respond. The facet can be subtle, but it must affect an
  actual line choice, timing, refusal, joke, correction, hesitation, or decision.

⭐ VOICE — same situation, different person, different line:
- Plain English and strong character are NOT in conflict. Keep words simple AND make each
  character sound like ONLY themselves. Voice lives in WHAT they choose to say, their
  attitude, and timing — not in fancy vocabulary.
- The two speakers must NOT sound interchangeable. Write each line the way THAT character
  (see bible) would react — e.g. one deflects with a dry one-liner while the other states
  the plan flatly; one over-explains a tiny detail while the other just wants to move on.
- You can be deadpan, fussy, blunt, or over-eager using only common words. Do that.

⭐ CHARACTER CAUSALITY — choose lines by inner pressure, not by ending shape:
- For every bubble, silently answer: What does this character want here? What are they protecting
  or avoiding? How does the other person or setting pressure that defense?
- Do not choose a line because it neatly ends the scene, teaches the lesson, or feels like the
  "right" closing answer. Any line is fine only when it is the most plausible thing this character
  would say under this pressure.
- If a bubble sounds like any character could say it, rewrite it from the chosen facet.

⭐ FUNNY — play THE GAME from the brief (this is what makes it worth reading):
- The driver keeps doubling down on the SAME absurd premise; each of their lines is a BIGGER,
  more specific version, while the other reacts/condemns/tries to stop it. Don't resolve early.
- BE SPECIFIC: a weird exact detail is funny; a vague one is not. "Grandma's 1973 kimchi jar"
  not "something old"; "I saw it on a cooking show at 3am" not "I heard somewhere".
- The TARGET SENTENCE often lands as the straight character's exasperated reaction to the
  escalation (e.g. "Can we just deal with the problem the normal way?").
- END ON A CHARACTER BUTTON: the last beat is whatever this character's facet naturally does next:
  a twist, deadpan topper, relapse, over-explanation, correction, dodge, reluctant softening, or
  small unresolved residue. Warmth is the frame; the character engine decides the line.
- Comedy comes from heightening ONE thing, not stacking random jokes. No memes, no puns for
  their own sake.

Before returning the script, self-check:
- Does the exact target sentence appear verbatim on the marked beat, spoken by that beat's character?
- Did every "[SILENT OBJECT]" beat stay silent (empty char/bubble/bubble_kr)?
- Does each character's reaction (from "WHO'S IN IT") visibly shape their lines?
- Can you explain every bubble through that character's want + defense + relationship pressure?
- Would swapping speaker names make the scene worse? If not, revise until each speaker's
  lines are recognizably tied to their character.

⭐ KOREAN TRANSLATION ("bubble_kr"): write how a real Korean speaker would actually say it,
NOT a word-for-word translation.
- Natural spoken Korean (구어체) — the way a friend/coworker really talks, contractions and all.
- Translate the MEANING and tone, not the grammar. Reorder, drop, or merge words freely so it
  sounds native. A stiff literal rendering (e.g. "디럭스를 유지하면 더 많은 비용이 청구될 거야") is WRONG;
  write it the way a person would say it (e.g. "그거 그냥 두면 돈 더 내야 돼").
- {register_kr_rule(register)}
  (Speech level is set by REGISTER, NOT by the setting — close coworkers at the office still speak casually.)
- Keep it short and punchy — same length feel as the English bubble.
{(tone_rule + chr(10)) if tone_rule else ""}
=== CHARACTER BIBLE (behavior patterns) ===
{lore}

{word_block}
=== THE SCENE BRIEF (read for tone/character; follow its BEAT-BY-BEAT skeleton exactly) ===
{plan_json}

Return ONLY this JSON. The panel count MUST match the brief's beat count exactly.
For object/caption panels, set "char" to an empty string and keep "bubble"/"bubble_kr" empty:
{{ "panels": [ {{"char": "name", "bubble": "natural everyday English", "bubble_kr": "natural spoken Korean (구어체 의역, NOT literal)"}} ] }}"""


# ── ③ Visual (SDXL) — Danbooru 태그 ──────────────────────────────
def build_visual_prompt(
    situation,
    script_panels_json,
    expression_menu="",
    pose_menu="",
    char_demeanor="",
    planner_context="",
) -> str:
    # action/expression 을 Danbooru 태그로 직접 출력시킨다(GPT 프롬프팅).
    # (expression_menu/pose_menu 인자는 하위호환용으로 남기지만 쓰지 않는다.)
    return f"""You are the VISUAL DIRECTOR for a webtoon, generating Danbooru tags for an anime
image model (Illustrious / NoobAI / Pony). For each panel, output image tags only — NO dialogue.

Scene situation: {situation}

=== PLANNER VISUAL CONTEXT ===
Use this as the source of truth for set, room, visible learning moment, character conflict, and beats:
{planner_context or "(none)"}

Danbooru tag rules (apply to action / expression / face_state / background):
- Real Danbooru tags, lowercase, words separated by SPACES (e.g. "crossed arms", NOT "crossed_arms").
- Comma-separate tags within a field. Decompose concepts into atomic tags
  ("leaning forward with hands on the table" -> "leaning forward, hands on table").
- Convert semantic/textual phrases into drawable objects or states:
  BAD: "incoming email response", "important issue", "confusion", "project delay"
  GOOD: "computer screen, email notification", "highlighted report", "map, red circle",
  "calendar deadline"

Per field:
- action: the character's BODY POSE / gesture as Danbooru tags ONLY (e.g. "crossed arms",
  "leaning forward, hands on table", "pointing", "head scratch", "hand on hip", "arm support").
  Body/hands only — NO facial/emotional tags here. NO subject count (1girl), hair, eyes,
  clothing. 1-3 tags. Character panel action must NEVER be "none".
  Do NOT put "smiling", "happy", "sad", "angry", "worried", "exaggerated expression",
  "confused", or any emotion/face word in action; those belong in expression.
  NEVER put breathing/air tags in action ("taking a deep breath", "breathing", "sigh",
  "exhale", "steam", "fog", "puff") — they render as visible breath clouds. Use a plain body
  pose instead (e.g. "hand on chest", "shoulders relaxed").
- expression: pick EXACTLY ONE key from the EXPRESSION MENU below. Output ONLY the key (e.g. "frown").
  ⭐ HARD RULE: the value MUST be one of the menu keys verbatim. Do NOT invent keys
  ("determined", "worried", "thoughtful", "shocked" are NOT valid). If the exact mood is not in
  the menu, pick the CLOSEST existing key (e.g. resolve→serious, worried→frown or furrowed_brow,
  determined→serious, shocked→fear_kubrick).
  ⭐ It is DRIVEN BY THIS LINE'S EMOTION — read the panel's dialogue and choose the menu key that
  matches THAT moment. It should CHANGE from panel to panel as the mood shifts; do NOT repeat the
  same expression every panel, and do NOT default to the character's resting face.
  The baseline demeanor below is only a CEILING (stay in character — don't give a shy person
  "naughty"/"seductive_smile"); it is NOT the default. The line's emotion comes first.
- face_state: gaze direction only (e.g. "looking at viewer", "looking away", "looking down", "looking up", "looking to the side"). Use a real Danbooru gaze tag.
- background: use Planner "set" + "room" as the source of truth.
  Use ONE coherent non-empty location tag-set for the entire episode, based on that set/room.
  Use 1-3 low-detail tags only (e.g. "office, desk", "meeting room, conference table",
  "cafe, table", "convenience store, storefront", "apartment, living room").
  Repeat the same background value in every panel, including object panels.
  NEVER output background as "none" or empty.
  Do NOT encode props, before/after states, lighting, lens, atmosphere, crowd, furniture
  lists, decorative detail, or panel-specific story beats in background.
  Put story-specific visible objects in "subject" (object panels) or "action", not background.
- visible_learning_moment must be visually encoded in the relevant panel's subject or action.
  If it involves a screen, report, map, receipt, box, notification, highlighted item, or
  before/after state, name that drawable object/state in subject or action.
- If Planner gives visible_proof_panel, that exact panel MUST encode visible_learning_moment
  in subject and/or action as a concrete drawable object/state. It must not be a character panel.
- Do NOT add camera-angle / framing tags anywhere (no "from above", "from below", "dutch angle",
  "close-up", "wide shot") — framing is fixed to eye-level elsewhere.

=== CHARACTER BASELINE (a CEILING for in-character range — NOT the default expression; the line's emotion drives each panel) ===
{char_demeanor or "(none)"}

=== EXPRESSION MENU (choose the "expression" value from these keys ONLY) ===
{expression_menu}

For object/caption panels (char is empty):
- Do NOT invent a person. No character, no face, no body, no clothing.
- Use "subject" to name the visible object/state/action clearly (e.g. "stack of receipts",
  "empty milk carton", "phone screen with unread message", "labeled boxes").
- action should describe object placement or state, not a body pose. It must be non-empty.
  NEVER use human-body verbs on an object panel ("nodding", "writing notes", "gesturing",
  "pointing", "leaning", "holding"); there is no person here. Use object-state words only
  (e.g. "placed on desk", "screen lit up", "scattered", "stacked", "displayed on screen").
- background must be the same non-empty set/room background used by the character panels.
- expression must be "none"; face_state must be "none".

Panels in order (char + their line):
{script_panels_json}

Before returning, self-check:
- Did every panel use the planned set/room?
- Is every background non-empty?
- Is visible_learning_moment visually encoded in subject or action?
- If visible_proof_panel exists, does that exact panel encode visible_learning_moment as a
  drawable subject/action object state?
- Are all action tags drawable body/object-state tags?
- Are face/emotion tags only in expression, not action?

Return ONLY this JSON (SAME number and order as the panels above).
"subject" = object/state for object panels, empty for character panels.
"action" = Danbooru pose tags for character panels OR object-state tags for object panels.
"expression" = ONE key from the EXPRESSION MENU for character panels, "none" for object panels.
face_state/background = Danbooru tags:
{{ "panels": [ {{"subject": "", "action": "pose or object-state tags", "expression": "menu_key_or_none", "face_state": "gaze tag or none", "background": "scene tags"}} ] }}"""


# ── ④ Review Card (SDXL) — 표현별 단일 인출 단서 카드 ──────────────
def build_review_card_prompt(
    word_block,
    collocation,
    expression_menu="",
    char_demeanor="",
) -> str:
    """복습 카드 전용 비주얼 프롬프트.

    만화 패널을 재사용하지 않고, sentence unit의 말하기 상황을 보고 단일 인출 단서(키워드법) 이미지를
    새로 설계한다. mode 를 GPT 가 고른다:
      · "character" → 주인공 한요일(hanyoil) 혼자, 포즈·표정·소품으로 뉘앙스 표현.
      · "object"    → 사람 없이 그 표현을 가장 잘 떠올리게 하는 사물/장면 1컷 (heavy traffic →
                      꽉 막힌 도로 등). 사람이 흉내내기 어려운 장면·사물성 표현일 때.
    둘 다 흰 배경. 만화 장면과의 연관 불필요.
    """
    return f"""You are the VISUAL DIRECTOR designing a single REVIEW-CARD image for an
ENGLISH-LEARNING webtoon. This card is a MEMORY RETRIEVAL CUE (keyword method) for ONE English
sentence — a learner glances at it later and the speaking moment comes back. Output
Danbooru tags only — NO dialogue, NO text in the image.

⭐ THIS CARD IS NOT A COMIC PANEL. Do NOT reconstruct any scene. Design ONE fresh, iconic image,
from scratch, whose whole job is to evoke the target sentence's speaking situation as vividly as possible.
The background is fixed to plain white elsewhere — do NOT output any background.

{word_block}
Target sentence: "{collocation}"

⭐ FIRST, CHOOSE THE MODE that makes the STRONGEST retrieval cue for THIS expression:
- "character": the protagonist Han Yoil (tag: hanyoil), alone, performs the speaking situation with her body
  pose / face / an optional prop. Use this when the expression is something a PERSON does or
  FEELS — an action, reaction, decision, or interpersonal moment (express gratitude, take up a
  challenge, weigh the pros and cons, express concern).
- "object": NO person at all — a single iconic OBJECT or SCENE that IS the expression. Use this
  when the concept is a thing/situation a person cannot embody by miming, and a learner would
  recognize it faster from the scene itself (heavy traffic → a jam of cars bumper to bumper;
  an empty wallet; a stack of overdue bills). Don't force Han Yoil to gesture at it — just show it.
Pick whichever a learner would decode fastest. When in doubt for a body gesture or feeling, use
character. When the expression is an abstract business/process idea (provide evidence, meet
requirements, pave the way, make arrangements, foster innovation), prefer object or a very simple
symbolic prop cue instead of a vague pose.

Danbooru tag rules (lowercase, words separated by SPACES, comma-separate tags within a field;
convert abstract ideas into concrete drawable objects — BAD: "rejection", "deadline";
GOOD: "raised hand, paper", "calendar, red circle"):

If mode = "character":
- action: Han Yoil's BODY POSE / gesture as Danbooru tags ONLY (e.g. "crossed arms",
  "pushing away, both hands", "covering ears", "reaching out", "holding phone"). Body/hands only.
  NO facial/emotion words (those go in expression). NO subject count (1girl), hair, eyes,
  clothing, background. 1-3 tags, never "none". NEVER use breathing/air tags
  ("sigh", "breath", "steam", "puff") — they render as visible clouds.
- expression: pick EXACTLY ONE key from the EXPRESSION MENU below, matching the speaking moment. Output
  ONLY the menu key verbatim. If the exact mood is absent, pick the CLOSEST key
  (worried→frown or furrowed_brow, determined→serious, shocked→fear_kubrick).
- face_state: gaze direction only ("looking at viewer", "looking away", "looking down",
  "looking up", "looking to the side").
- props: 0-1 drawable object that anchors the speaking moment (held/near her), comma-separated, or "".
  Prefer icon-like props that do not require readable writing. Use "medal", "star badge",
  "magnifying glass", "blank folder", "blank paper", "clipboard with checkmarks", "light bulb",
  "phone", "notebook", "pen", "calendar icon", "red circle".
  Avoid text-generating props: NO certificate, document, contract, report, form, resume, sign,
  label, poster, chart with labels, receipt, screen with text, book cover with text.
- subject: leave "".
- Stay in her character range (see baseline below) — don't make her act out of type.

If mode = "object":
- subject: the iconic object/scene that IS the cue, as concrete drawable Danbooru tags
  (e.g. "traffic jam, cars in a row, bumper to bumper, city street"; "empty wallet, coins").
  NO people, no body, no face, no clothing.
- action: object placement/state tags only (e.g. "lined up", "congested", "scattered on table",
  "stacked", "displayed on screen"). NEVER human-body verbs (holding, pointing, standing). Non-empty.
- expression: "none". face_state: "none". props: "".
  Prefer a single simple visual metaphor on a white background. If the cue needs a sign/label,
  replace it with a non-text symbol (arrow, red circle, checkmark, star, path, stepping stones).
  NO readable text, letters, numbers, logos, forms, posters, labeled charts, or documents.

Do NOT add camera-angle / framing tags (no "from above", "close-up", "wide shot").

=== HAN YOIL BASELINE (character mode only — a CEILING for in-character range, not the default) ===
{char_demeanor or "(none)"}

=== EXPRESSION MENU (character mode: choose the "expression" value from these keys ONLY) ===
{expression_menu}

Before returning, self-check:
- Would a learner who studied "{collocation}" remember when to say it from this single image?
- Is the chosen mode the FASTEST cue (a scene/object expression should be "object", not a person
  miming it)?
- Did you avoid any prop/object that invites readable text? If not, replace it with a blank or
  symbolic version.
- character: action is a drawable body pose (1-3 tags), no emotion/breath words; expression is one
  menu key, and the cue has only Han Yoil. object: subject names the scene/object, action is
  object-state only, no person.

Return ONLY this JSON:
{{ "mode": "character | object", "subject": "object/scene tags (object mode) or empty", "action": "pose tags or object-state tags", "expression": "menu_key or none", "face_state": "gaze tag or none", "props": "object tags or empty" }}"""
