"""Prompt templates for the English-learning webtoon pipeline."""

from __future__ import annotations


# ---------------------------------------------------------
# Outfit Mapping
# Outfit Mapping
# ---------------------------------------------------------
# Domain -> outfit setting prefix.
# The renderer chooses common outfits for all appearing characters.
# Outfit selection uses outfits shared by all appearing characters.
CATEGORY_DEFAULT_SETTING: dict[str, str] = {
    "workplace": "workplace", "academic": "academic", "daily": "daily_outing",
    "customer/service": "daily_outing", "personal": "daily_dressup", "social": "daily_outing",
}


# ---------------------------------------------------------
# Word Block
# Word Block
# ---------------------------------------------------------
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
=== Target Sentence ===

- sentence        : {target}
- Korean cue      : {meaning}
- register        : {word.get('register', '')}
- domain          : {used_in or primary_used_in}

Speech hints, not plot:
- speech_act      : {speech_act}
- politeness      : {politeness}
- speaker_role    : {speaker_role}
- listener_role   : {listener_role}
- relationship    : {relationship}
- power_dynamic   : {power_dynamic}
- story_function  : {story_function}
- micro_situation : {nuance}

Use these only to understand what the sentence does socially.
Do not copy the micro_situation as the episode plot.
Invent a fresh sitcom situation from the character/world friction where this sentence becomes useful.
"""


def build_planner_word_block(word: dict) -> str:
    target = word.get("collocation unit") or word.get("sentence_unit") or word.get("sentence unit", "")
    meaning = word.get("meaning") or word.get("korean_trigger") or word.get("Korean trigger", "")
    primary_used_in = word.get("primary_used_in") or word.get("used in") or ""
    used_in = word.get("used_in") or primary_used_in
    return f"""
=== Target Sentence Seed ===

- sentence   : {target}
- Korean cue : {meaning}
- register   : {word.get('register', '')}
- domain     : {used_in or primary_used_in}

Invent a sitcom situation where this exact sentence becomes useful.
Do not begin from the CSV micro_situation or role metadata.
"""


# ---------------------------------------------------------
# Collocation Rules
# Collocation Rules
# ---------------------------------------------------------
def build_word_rule(collocation: str) -> str:
    return (
        f'- The exact target sentence "{collocation}" MUST appear verbatim inside one spoken English bubble\n'
        f'- Natural short spoken wrappers are allowed around it when they make the line sound real '
        f'(e.g. "Yeah,", "Honestly,", "I mean,", "Yes, but", "Okay, so")\n'
        f'- The sentence must be spoken by a character, never placed in an object/caption panel\n'
        f'- Choose the episode conflict so this exact sentence is something that character would genuinely say\n'
        f'- Do NOT paraphrase, shorten, split, or grammatically alter the target sentence words\n'
        f'- Do NOT directly explain the sentence\n'
        f'- Avoid textbook-style delivery\n'
        f'- Write natural spoken dialogue only'
    )


# ---------------------------------------------------------
# System Prompt
# System Prompt
# ---------------------------------------------------------
def build_arc_prompt(arc_state: dict) -> str:
    """Format current relationship arc state into a system prompt block."""
    if not arc_state:
        return ""
    _PHASE_LABEL = {1: "stable", 2: "tension", 3: "turning_point", 4: "closer"}

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
        head = f"{pair}  [phase {phase} - {_PHASE_LABEL.get(phase, phase)}"
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


# ---------------------------------------------------------
# Three-stage prompt pipeline: Story Planner -> Script -> Visual
# Each GPT call focuses on one job.
# ---------------------------------------------------------

_TONE_MAP = {
    # register values from the collocations data.
    # Extra tone labels are kept for manual overrides.
    "informal":  "Overall tone: informal and relaxed - casual spoken language.",
    "standard":  "Overall tone: neutral, natural everyday speech.",
    "casual":    "Overall tone: casual and relaxed.",
    "formal":    "Overall tone: slightly formal/professional but still plain.",
    "sarcastic": "Overall tone: lightly dry/sarcastic - understated.",
    "emotional": "Overall tone: emotionally warm - speaking from feeling.",
    "blunt":     "Overall tone: blunt and direct, no softening.",
}


def tone_rule_for(register: str = "") -> str:
    if not register:
        return ""
    return _TONE_MAP.get(register.lower().strip(), f"Overall tone: {register}.")


def _revision_block(feedback: str) -> str:
    if not feedback:
        return ""
    return ("=== REVISION NOTES (previous draft FAILED review - fix these) ===\n\n"
            + feedback + "\n\n")


# Story Planner - structure only
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
        (f"\nHARD REQUIREMENT - DOMAIN IS FIXED: this target sentence's domain is "
         f"\"{required_domain}\". The \"domain\" you return MUST be EXACTLY \"{required_domain}\" "
         f"(one of the four: workplace | daily | academic | customer/service). "
         f"Choose a location that genuinely belongs to "
         f"that domain. If the domain is NOT workplace, the scene must NOT happen at an office, "
         f"meeting room, Plaoud, or any desk/work setting - go to a real personal/shop/school place.\n")
        if required_domain else "")
    return f"""You are a sitcom story breaker in a writers' room for an English-learning webtoon.
{_domain_lock}

Your job is to break one small 6-panel sitcom story, not to write jokes first
and not to write dialogue yet.

Start from CHARACTER FRICTION inside this world, NOT from the target sentence.
Pick characters whose facets (from the character bible) collide over one tiny
everyday problem in the chosen domain, and build the funny escalation FIRST.
The target sentence is simply the natural line one character lands on under that
pressure — it is NOT the seed and NOT the point of the scene.

Comedy is the engine; the world bible, character facets, and the domain are your
source of taste. The micro_situation and role metadata are only faint
speech-function hints — do NOT copy them as the plot, and do NOT let them flatten
the scene into people merely discussing status, readiness, or feelings.

HARD FAILURE MODE — do NOT produce a "status-check / explanation" episode where one
character over-explains and the other is confused or asks them to repeat. If your
premise reduces to "A explains, B doesn't get it," discard it and find a real
behavioral collision: an object that misbehaves, a small competition, a quirky
justification, a dodge, a visible social mistake. No two episodes may share the
same shape.

EVEN IF the target sentence is itself a question, a clarification, or a request to
repeat (e.g. "I didn't get that.", "Can you say that again?"), do NOT build a teaching/
explanation scene. Invent a NON-teaching reason the character says that line — they
double-take at an absurd claim, get drowned out by a noise/object, are distracted by
their own facet, mishear a prank, or refuse to accept a ridiculous answer. The line
should land because of a funny behavioral collision, never because someone is lecturing.

Break the story in this order:
1. WANT - what does one character want right now?
2. PRESSURE - what does the other character, setting, role, or object do that pressures it?
3. GAME - what is the one funny behavioral pattern created by that pressure?
4. ESCALATION - how does that same pattern get sharper across the panels?
5. TARGET MOMENT - where does the target sentence become the useful thing to say?
6. BUTTON - what is the final character-specific turn or residue?

A beat is not a feeling. A beat is an external event, choice, interruption,
mistake, object state, or social move that changes what the next character can do.

Creative goal:
- Start from character friction inside this world, not from the micro_situation wording.
- Make something specific happen in the room right now. Avoid abstract status-check
  conversations where characters only discuss progress, readiness, or feelings.
- Break external action, not internal mood. A sitcom beat should be playable on
  screen through behavior, interruption, object trouble, role pressure, or a
  visible social mistake. Do not build the episode mainly from inner thoughts.
- Avoid plans whose game is only "character feels nervous / tries to look ready /
  boss applies pressure." Turn that psychology into a visible behavior or event.
- Give the script writer a playable beat sheet: premise, turn, target moment, and button.
- Use only a few concrete details. Leave exact wording, acting, poses, and camera for later stages.

Pipeline needs:
- Return exactly 6 beats.
- Exactly one beat must have "has_collocation": true.
- That beat must be a character panel with a non-empty speaker.
- All other beats must have "has_collocation": false.
- The target_speaker must be the speaker of the true beat.

{_revision_block(feedback)}
=== WORLD BIBLE ===
{lore}

{word_block}

Target sentence rule:
{word_rule}

{cast_directive}

=== RELATIONSHIP ARC STATE ===
{arc_prompt}

{("=== SHOWRUNNER NOTES ===" + chr(10) + showrunner_notes + chr(10)) if showrunner_notes else ""}

=== AVAILABLE LOCATIONS ===
Pick one location tag from this domain:
{location_menu}
{cast_note}{avoid_situations}

Return JSON only. Keep this shape for the production pipeline, but fill it as a writer's room beat sheet:
{{
  "target_sentence": "exact target sentence",
  "primary_used_in": "target domain",
  "speaker_role": "metadata role or inferred role",
  "listener_role": "metadata role or inferred role",
  "relationship": "metadata relationship or inferred relationship",
  "power_dynamic": "metadata power dynamic or inferred dynamic",
  "speech_act": "what the sentence does in the scene",
  "politeness": "metadata or inferred",
  "story_function": "where the sentence lands emotionally",
  "selected_pair": "exact selected pair string",
  "target_speaker": "character who says the exact sentence",
  "target_listener": "character receiving it",
  "metadata_correction": "",

  "situation_id": "short slug",
  "domain": "workplace | daily | academic | customer/service",
  "location": "one allowed location tag",
  "background_prop": "0 to 2 simple objects, or empty string",
  "outfit_setting": "workplace | academic | daily_home | daily_convenience | daily_outing | daily_dressup | daily_sport",
  "situation": "one vivid sentence describing the scene pressure",
  "delivery_mode": "character-led | object-led | action-led | setting-led | mixed",

  "target_sentence_context": {{
    "relationship_context": "why these two are in this exchange",
    "target_speaker_role": "why this character owns the line",
    "target_listener_role": "why the other character receives it",
    "power_dynamic": "where the pressure comes from",
    "speech_act": "what the sentence accomplishes",
    "service_direction": "none | customer_to_staff | staff_to_customer | internal_team | teacher_to_student | student_to_teacher",
    "story_function": "the sentence's story job"
  }},

  "nuance_structure": {{
    "situation": "setup",
    "pressure": "turn/complication",
    "expression": "target sentence moment",
    "confirmation": "button/residue"
  }},

  "sitcom_conflict": "one sentence conflict",
  "comedic_game": {{
    "driver": "character driving the funny pressure",
    "premise": "one playable comic idea",
    "escalation": "how it changes across panels",
    "button": "ending beat"
  }},
  "character_filter_collision": [
    {{"character": "name", "facet": "activated trait", "collision": "how it meets the other character/setting"}}
  ],
  "emotional_engine": [
    {{"character": "name", "hidden_need": "what they want", "defense": "how they protect it", "soft_spot_or_contradiction": "what keeps them human"}}
  ],
  "visible_learning_moment": "why the target sentence feels inevitable",
  "visible_proof_panel": null,
  "characters": ["0 to 3 names"],
  "problem": "single problem driving the scene",
  "beat_count": 6,
  "beats": [
    {{
      "panel": 1,
      "panel_type": "character | object",
      "speaker": "name, or empty string for object/caption panel",
      "nuance_role": "situation | pressure | expression | confirmation",
      "visual_focus": "what matters in the panel",
      "intent": "writerly beat, not camera direction",
      "has_collocation": false
    }}
  ],
  "milestone": {{"is_milestone": false, "pair": "exact pair name or null", "summary": "one sentence if milestone else null"}}
}}"""


# Shared return schema (single-brace JSON) — reused by the free planner so downstream parsing
# stays identical. Embedded via f-string: f"...{_PLANNER_RETURN_SCHEMA}".
_PLANNER_RETURN_SCHEMA = """Return JSON only. Keep this shape for the production pipeline, but fill it as a writer's room beat sheet:
{
  "target_sentence": "exact target sentence",
  "primary_used_in": "target domain",
  "speaker_role": "metadata role or inferred role",
  "listener_role": "metadata role or inferred role",
  "relationship": "metadata relationship or inferred relationship",
  "power_dynamic": "metadata power dynamic or inferred dynamic",
  "speech_act": "what the sentence does in the scene",
  "politeness": "metadata or inferred",
  "story_function": "where the sentence lands emotionally",
  "selected_pair": "exact selected pair string",
  "target_speaker": "character who says the exact sentence",
  "target_listener": "character receiving it",
  "metadata_correction": "",

  "situation_id": "short slug",
  "domain": "workplace | daily | academic | customer/service",
  "location": "one allowed location tag",
  "background_prop": "0 to 2 simple objects, or empty string",
  "outfit_setting": "workplace | academic | daily_home | daily_convenience | daily_outing | daily_dressup | daily_sport",
  "situation": "one vivid sentence describing the scene pressure",
  "delivery_mode": "character-led | object-led | action-led | setting-led | mixed",

  "target_sentence_context": {
    "relationship_context": "why these two are in this exchange",
    "target_speaker_role": "why this character owns the line",
    "target_listener_role": "why the other character receives it",
    "power_dynamic": "where the pressure comes from",
    "speech_act": "what the sentence accomplishes",
    "service_direction": "none | customer_to_staff | staff_to_customer | internal_team | teacher_to_student | student_to_teacher",
    "story_function": "the sentence's story job"
  },

  "nuance_structure": {
    "situation": "setup",
    "pressure": "turn/complication",
    "expression": "target sentence moment",
    "confirmation": "button/residue"
  },

  "sitcom_conflict": "one sentence conflict",
  "comedic_game": {
    "driver": "character driving the funny pressure",
    "premise": "one playable comic idea",
    "escalation": "how it changes across panels",
    "button": "ending beat"
  },
  "character_filter_collision": [
    {"character": "name", "facet": "activated trait", "collision": "how it meets the other character/setting"}
  ],
  "emotional_engine": [
    {"character": "name", "hidden_need": "what they want", "defense": "how they protect it", "soft_spot_or_contradiction": "what keeps them human"}
  ],
  "visible_learning_moment": "why the target sentence feels inevitable",
  "visible_proof_panel": null,
  "characters": ["0 to 3 names"],
  "problem": "single problem driving the scene",
  "beat_count": 6,
  "beats": [
    {
      "panel": 1,
      "panel_type": "character | object",
      "speaker": "name, or empty string for object/caption panel",
      "nuance_role": "situation | pressure | expression | confirmation",
      "visual_focus": "what matters in the panel",
      "intent": "writerly beat, not camera direction",
      "has_collocation": false
    }
  ],
  "milestone": {"is_milestone": false, "pair": "exact pair name or null", "summary": "one sentence if milestone else null"}
}"""


def build_free_planner_prompt(
    lore,
    word_block,
    word_rule,
    arc_prompt,
    feedback="",
    avoid_situations="",
    required_domain="",
    location_menu="",
) -> str:
    """Clean D-style planner: domain + target sentence + full cast bible, comedy-first, free pairing.

    No pre-selected pair, no role/metadata anchoring — the experiment showed those force the
    'over-explainer + confused listener' template. Outputs the SAME schema as build_planner_prompt
    so the downstream script/visual stages are unchanged.
    """
    _domain_lock = (
        (f"\nHARD REQUIREMENT - DOMAIN IS FIXED: this target sentence's domain is "
         f"\"{required_domain}\". The \"domain\" you return MUST be EXACTLY \"{required_domain}\" "
         f"(workplace | daily | academic | customer/service). Choose a location that genuinely "
         f"belongs to that domain.\n")
        if required_domain else "")
    return f"""You are a sitcom story breaker for a slice-of-life workplace sitcom that teaches English.
{_domain_lock}
Write ONE small, funny 6-panel episode. Comedy comes FIRST and from CHARACTER FACET COLLISION:
two people bring different facets (habits, fears, filters) from the character bible to one tiny
everyday problem; one person's quirk ESCALATES while the other reacts, and warmth only lands at
the final button (do not resolve every panel).

Start from CHARACTER FRICTION inside this world, NOT from the target sentence. The target
sentence is simply the natural line one character lands on under that pressure — it is NOT the
seed and NOT the point of the scene. The world bible, character facets, and the domain are your
source of taste.

YOU choose which cast pairing makes the funniest collision. Do NOT default to the
explainer->confused-listener or teacher->student pairing.

HARD FAILURE MODE — do NOT produce a "status-check / explanation" episode where one character
over-explains and the other is confused or asks them to repeat. EVEN IF the target sentence is a
question / clarification / request to repeat (e.g. "I didn't get that.", "Can you say that again?"),
do NOT build a teaching scene. Invent a NON-teaching reason the line is said: a double-take at an
absurd claim, being drowned out by a noise/object, a prank, a distraction, refusing a ridiculous
answer, a competition. The line must land from a funny behavioral collision, never from lecturing.

Break the story so it is playable on screen through behavior, objects, interruptions, and social
mistakes — not inner mood. Use only a few concrete details; leave wording/acting/camera for later.

Write the actual DIALOGUE for every panel — this IS the script; there is no separate dialogue stage.

Pipeline needs:
- Exactly 6 beats; each beat is one panel.
- Character beats have ONE spoken bubble; object/caption beats use speaker "" and usually empty bubble.
- The exact target sentence appears VERBATIM in exactly ONE character beat; set that beat's
  has_collocation=true and all other beats false. Place it WHEREVER it lands most naturally
  (often mid-scene at the peak of the collision), NOT forced onto the final button.
- target_speaker = the speaker of that beat, and must be listed in "characters".
- Alternate speakers — never the same speaker twice in a row. Keep each bubble ONE simple idea.

Performance per beat uses ONLY these keys:
- body_pose: standing | sitting | leaning_forward | walking | crouching
- gesture: none | arms_crossed | hands_on_hips | pointing | hand_on_chin | hand_on_forehead |
  facepalm | head_in_hands | shrug | hand_raised | waving | hands_clasped | holding_cup |
  holding_phone | holding_paper | holding_laptop | typing
- gaze: looking_at_viewer | looking_to_side | looking_down | looking_up | looking_away
- expression_intent: short visible emotion (deadpan, confused, smug, awkward, ...)
- prop_use: "none", or one simple drawable prop

{_revision_block(feedback)}
=== WORLD + CHARACTER BIBLE ===
{lore}

=== RELATIONSHIP ARC STATE ===
{arc_prompt}

{word_block}

Target sentence rule:
{word_rule}

=== AVAILABLE LOCATIONS (pick ONE tag for the whole episode) ===
{location_menu}
{avoid_situations}

Return JSON ONLY:
{{
  "situation": "one vivid sentence describing the scene",
  "sitcom_conflict": "one sentence conflict",
  "domain": "{required_domain or 'workplace | daily | academic | customer/service'}",
  "location": "one allowed location tag",
  "background_prop": "0-2 simple objects, or empty string",
  "outfit_setting": "workplace | academic | daily_home | daily_convenience | daily_outing | daily_dressup | daily_sport",
  "characters": ["1-3 names"],
  "selected_pair": "the two leads as 'a ↔ b'",
  "target_speaker": "character who says the exact target sentence",
  "comedic_game": {{"driver": "", "premise": "", "escalation": "", "button": ""}},
  "character_filter_collision": [{{"character": "", "facet": "", "collision": ""}}],
  "visible_learning_moment": "why the target line feels inevitable",
  "visible_proof_panel": null,
  "beat_count": 6,
  "beats": [
    {{"panel": 1, "panel_type": "character | object", "speaker": "name, or empty for object panel",
      "bubble": "English line; empty for silent object",
      "bubble_kr": "natural Korean; empty for object/narration",
      "performance": {{"expression_intent": "", "body_pose": "", "gesture": "", "gaze": "", "prop_use": "none"}},
      "subject": "object panels: the visible object/state, else empty",
      "has_collocation": false}}
  ],
  "milestone": {{"is_milestone": false, "pair": null, "summary": null}}
}}"""


# Script - dialogue and minimal performance
_REGISTER_KR = {
    "informal": "Korean speech level: casual banmal throughout. Keep it consistent every bubble.",
    "standard": "Korean speech level: default to casual banmal between close friends/coworkers; use polite Korean only if the scene is clearly formal. Pick one level and keep it consistent.",
    "formal":   "Korean speech level: default to casual banmal for close relationships; use polite Korean only when the actual relationship/service situation calls for it.",
}


def register_kr_rule(register: str = "") -> str:
    return _REGISTER_KR.get((register or "").lower().strip(), _REGISTER_KR["standard"])


def build_script_prompt(lore, word_block, plan_json, tone_rule="", feedback="", register="") -> str:
    return f"""You are the SCRIPT WRITER for an English-learning webtoon.
You write the panel script: dialogue, Korean translation, and minimal performance direction.

Use the SCENE BRIEF as the source of truth. Do not invent a new plot, new scene, new speaker order,
or new target-sentence beat. The planner already decided the story; your job is to make each panel
play clearly.

Role:
- Write exactly one output panel for each beat in the brief.
- Character beats get one spoken bubble.
- Object/caption beats use char "" and usually empty bubble/bubble_kr.
- The exact target sentence must appear verbatim in the marked character beat.
- Do not move the target sentence to a different speaker or object/caption panel.
- Keep each panel's performance direction minimal and drawable.
- Do not replace the planned sitcom game with a new excuse or tangent.
- Each bubble must directly follow that panel's beat intent and the brief's comedic_game.
- If the brief says the excuse is a table leg, door, receipt, phone, order, time, or task,
  keep that exact everyday pressure. Do not swap in unrelated riddles, rare collections,
  trivia, fantasy logic, or new objects.

Allowed webtoon devices:
- Normal speech bubble: plain English in "bubble".
- Thought bubble: start English bubble with "(internally)", but use this only
  when the scene brief explicitly asks for an internal/thought beat. Do not use
  thought bubbles as the default way to show nervousness, pressure, or hesitation.
  If the brief does not literally ask for a thought bubble/internal beat, write
  a spoken line instead and let performance show the pressure.
- Narration card: start English bubble with "(narration)".
- Object/caption panel: char "", bubble/bubble_kr empty unless a short "(narration)" card is needed.

Performance direction rules:
- The performance is part of the script, not a later visual guess.
- Choose acting that helps the line read at a glance.
- Prefer face, posture, gaze, and simple hand gesture over literal prop handling.
- Use props only when the prop is visually central and easy to draw.
- If a prop is mentioned but the joke reads better through attitude, set prop_use to "none".
- Do not make a character hold an object just because the dialogue mentions it.
- For hard-to-draw physical jokes, let the dialogue/narration carry the idea and keep prop_use simple.
- Follow visual_policy from LOCKED PANEL PAYLOADS:
  attitude_only -> prop_use "none";
  show_prop -> one simple visible prop use;
  object_cutaway -> use an object/caption panel if that beat is object, otherwise keep prop_use simple;
  narration_card -> start bubble with "(narration)" only on caption/object beats;
  thought_reaction -> start bubble with "(internally)" on character beats.
- body_pose MUST be one of: standing, sitting, leaning_forward, walking, crouching.
- gesture MUST be one of: none, arms_crossed, hands_on_hips, pointing, pointing_at_screen,
  hand_on_chin, hand_on_forehead, facepalm, head_in_hands, shrug, hand_raised,
  waving, hands_clasped, holding_cup, holding_phone, holding_paper, holding_laptop, typing.
- gaze MUST be one of: looking_at_viewer, looking_to_side, looking_down, looking_up, looking_away.

Korean translation:
- bubble_kr should be natural spoken Korean, not literal translation.
- Do not include "(internally)" or "(narration)" in bubble_kr.
- {register_kr_rule(register)}

{_revision_block(feedback)}=== CHARACTER BIBLE ===
{lore}

{word_block}
=== SCENE BRIEF ===
{plan_json}

Return ONLY this JSON. The panel count MUST match the brief's beat count exactly:
{{
  "panels": [
    {{
      "char": "speaker name, or empty string for object/caption panels",
      "bubble": "English bubble; empty for silent object/caption panels",
      "bubble_kr": "natural Korean translation; empty for object/caption/narration panels",
      "performance": {{
        "expression_intent": "short visible emotion, e.g. deadpan, confused, calmly absurd, relieved",
        "body_pose": "one allowed body_pose key",
        "gesture": "one allowed gesture key",
        "gaze": "one allowed gaze key",
        "prop_use": "none, or one concrete easy-to-draw prop use",
        "visual_read": "what the viewer should understand from the acting"
      }}
    }}
  ]
}}"""
# Visual (SDXL) - Danbooru tags
def build_acting_prompt(
    situation,
    script_panels_json,
    planner_context="",
) -> str:
    return f"""You are the ACTING DIRECTOR for a character-driven webtoon.

Your job is NOT to write dialogue and NOT to write Danbooru tags yet.
For each panel, decide what the character is doing emotionally and physically
so the image explains why the line is being said.

Scene situation: {situation}

=== PLANNER VISUAL CONTEXT ===
{planner_context or "(none)"}

Panels in order:
{script_panels_json}

Rules:
- Keep the exact same number and order of panels.
- For character panels, make the acting match the line's speech function:
  asking, dodging, realizing, refusing, explaining, teasing, giving up, etc.
- For object panels, use a concrete visible object/state.
- Use props as story anchors. If the dialogue is about a notebook, receipt,
  phone, door, table, menu, screen, bag, or document, at least one relevant
  panel should visibly interact with that object.
- Do not make glamour poses. Prefer readable sitcom acting:
  reaching for an object, pointing at evidence, holding the prop, leaning in,
  hesitating, bracing, checking, blocking, hiding, or visibly giving up.
- Thought/inner beats should still have an external physical cue.
- Keep descriptions short and drawable.

Return ONLY this JSON:
{{ "panels": [ {{
  "acting_intent": "what this panel's performance communicates",
  "emotional_state": "specific line emotion, not generic mood",
  "visible_action": "drawable body action in plain English",
  "prop_interaction": "specific object interaction or empty string",
  "comic_function": "setup | pressure | target_sentence | button | object_proof"
}} ] }}"""


def build_visual_prompt(
    situation,
    script_panels_json,
    expression_menu="",
    pose_menu="",
    char_demeanor="",
    planner_context="",
    acting_context="",
) -> str:
    # expression/body_pose/gesture are menu-key selections. The renderer still
    # receives a legacy combined action string for compatibility.
    return f"""You are the VISUAL DIRECTOR for a webtoon, generating Danbooru tags for an anime
image model (Illustrious / NoobAI / Pony). For each panel, output image tags only - NO dialogue.

Scene situation: {situation}

=== PLANNER VISUAL CONTEXT ===
Use this as the source of truth for set, room, visible learning moment, character conflict, and beats:
{planner_context or "(none)"}

=== SCRIPT PERFORMANCE DIRECTIONS ===
Each script panel may include a performance object. Treat it as the source of truth
for expression, body_pose, gesture, gaze, and prop_use. Do not add prop handling
unless performance.prop_use asks for it.
{acting_context or "(none)"}

Danbooru tag rules (apply to action / expression / face_state / background):
- Real Danbooru tags, lowercase, words separated by SPACES (e.g. "crossed arms", NOT "crossed_arms").
- Comma-separate tags within a field. Decompose concepts into atomic tags
  ("leaning forward with hands on the table" -> "leaning forward, hands on table").
- Convert semantic/textual phrases into drawable objects or states:
  BAD: "incoming email response", "important issue", "confusion", "project delay"
  GOOD: "computer screen, email notification", "highlighted report", "map, red circle",
  "calendar deadline"
- First translate the script performance into tags. The action should explain the
  spoken line through posture, gesture, and gaze. A prop is optional; if
  performance.prop_use is "none", keep prop_interaction empty.

Per field:
- For character panels, choose motion by menu keys first:
  body_pose = exactly ONE key from BODY_POSES.
  gesture = exactly ONE key from GESTURES; use "none" when unclear.
  Do not invent body_pose or gesture keys. Do not stack multiple hand/arm/prop
  gestures. The renderer will combine body_pose + gesture into action tags.
  Use "none" sparingly: pressure, explanation, visible consequence, and reaction
  panels usually need one concrete gesture unless it would conflict with the body pose.
- action: the character's BODY POSE / gesture as Danbooru tags ONLY (e.g. "crossed arms",
  "leaning forward, hands on table", "pointing", "head scratch", "hand on hip", "arm support").
  Body/hands only - NO facial/emotional tags here. NO subject count (1girl), hair, eyes,
  clothing. 1-3 tags. Character panel action must NEVER be "none".
  Do NOT put "smiling", "happy", "sad", "angry", "worried", "exaggerated expression",
  "confused", or any emotion/face word in action; those belong in expression.
  NEVER put breathing/air tags in action ("taking a deep breath", "breathing", "sigh",
  "exhale", "steam", "fog", "puff") - they render as visible breath clouds. Use a plain body
  pose instead (e.g. "hand on chest", "shoulders relaxed").
- expression: pick EXACTLY ONE key from the EXPRESSION MENU below. Output ONLY the key (e.g. "frown").
  HARD RULE: the value MUST be one of the menu keys verbatim. Do NOT invent keys
  ("determined", "worried", "thoughtful", "shocked" are NOT valid). If the exact mood is not in
  the menu, pick the CLOSEST existing key (e.g. resolve -> serious, worried -> frown or furrowed_brow,
  determined -> serious, shocked -> fear_kubrick).
  It is DRIVEN BY THIS LINE'S EMOTION - read the panel's dialogue and choose the menu key that
  matches THAT moment. It should CHANGE from panel to panel as the mood shifts; do NOT repeat the
  same expression every panel, and do NOT default to the character's resting face.
  If script performance.expression_intent is present, map it actively:
  deadpan/resigned -> sigh; skeptical/confused -> furrowed_brow; annoyed -> annoyed;
  awkward/hopeful -> awkward_smile; playful/carefree/amiable -> light_smile or happy;
  calmly absurd/confident -> composed_smile. Do NOT default to serious unless the line is truly firm.
  The baseline demeanor below is only a CEILING (stay in character - don't give a shy person
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
- prop_interaction: output short object/action tags only when script performance.prop_use
  asks for a visible prop (e.g. "phone on table", "pointing at receipt").
  If prop_use is "none", this MUST be empty.
- If Planner gives visible_proof_panel, that exact panel MUST encode visible_learning_moment
  in subject and/or action as a concrete drawable object/state. It must not be a character panel.
- framing: choose ONE simple composition key from this menu:
  full_body | waist_shot | upper_body | close_up | object_close_up
  Keep the camera angle eye-level. Do not use from above, from below, dutch angle,
  dramatic lens, or wide shot. This is only for webtoon panel cropping:
  use full_body for establishing/physical action, waist_shot or upper_body for normal
  dialogue, close_up for a reaction or target-sentence beat, and object_close_up for
  object panels.

=== CHARACTER BASELINE (a CEILING for in-character range - NOT the default expression; the line's emotion drives each panel) ===
{char_demeanor or "(none)"}

=== EXPRESSION MENU (choose the "expression" value from these keys ONLY) ===
{expression_menu}

=== MOTION MENU ===
{pose_menu}

For object/caption panels (char is empty):
- Do NOT invent a person. No character, no face, no body, no clothing.
- Use "subject" to name the visible object/state/action clearly (e.g. "stack of receipts",
  "empty milk carton", "phone screen with unread message", "labeled boxes").
- Add simple color anchors for the main object and the surface when helpful:
  "black smartphone, white table", "white menu card, brown table",
  "red folder, white desk". Use color to clarify the object, not to decorate
  the whole scene.
- For phone beats, prefer "black smartphone" or "black cellphone" over plain
  "phone"; add "incoming call screen" when the phone is ringing.
- action should describe object placement or state, not a body pose. It must be non-empty.
  NEVER use human-body verbs on an object panel ("nodding", "writing notes", "gesturing",
  "pointing", "leaning", "holding"); there is no person here. Use object-state words only
  (e.g. "placed on desk", "screen lit up", "scattered", "stacked", "displayed on screen").
- background must be the same non-empty set/room background used by the character panels.
- expression must be "none"; face_state must be "none".
- body_pose must be "none"; gesture must be "none".

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
- Is framing one of the allowed keys and useful for this panel's storytelling beat?
- Does action + prop_interaction visibly explain why this exact line is said?

Return ONLY this JSON (SAME number and order as the panels above).
"subject" = object/state for object panels, empty for character panels.
"body_pose" = ONE key from BODY_POSES for character panels, "none" for object panels.
"gesture" = ONE key from GESTURES for character panels, "none" for object panels.
"action" = Danbooru pose tags for character panels OR object-state tags for object panels.
"prop_interaction" = concrete object/action tags from the ACTING PLAN, or empty string.
"expression" = ONE key from the EXPRESSION MENU for character panels, "none" for object panels.
"acting_intent" = copy the acting plan's intent in short plain English.
"framing" = ONE key: full_body | waist_shot | upper_body | close_up | object_close_up.
face_state/background = Danbooru tags:
{{ "panels": [ {{"subject": "", "body_pose": "body_pose_key_or_none", "gesture": "gesture_key_or_none", "action": "pose or object-state tags", "prop_interaction": "object interaction tags or empty", "expression": "menu_key_or_none", "face_state": "gaze tag or none", "acting_intent": "short intent", "framing": "upper_body", "background": "scene tags"}} ] }}"""


# Review Card (SDXL) - single retrieval cue image
def build_review_card_prompt(
    word_block,
    collocation,
    expression_menu="",
    char_demeanor="",
) -> str:
    """Build a visual prompt for a single review-card memory cue.

    This is not a comic panel. The model chooses either a character cue
    using Han Yoil, or an object-only cue when an object/scene is more
    memorable than a person acting the sentence out.
    """
    return f"""You are the VISUAL DIRECTOR designing a single REVIEW-CARD image for an
ENGLISH-LEARNING webtoon. This card is a MEMORY RETRIEVAL CUE (keyword method) for ONE English
sentence - a learner glances at it later and the speaking moment comes back. Output
Danbooru tags only - NO dialogue, NO text in the image.

IMPORTANT: THIS CARD IS NOT A COMIC PANEL. Do NOT reconstruct any scene. Design ONE fresh, iconic image,
from scratch, whose whole job is to evoke the target sentence's speaking situation as vividly as possible.
The background is fixed to plain white elsewhere - do NOT output any background.

{word_block}
Target sentence: "{collocation}"

IMPORTANT: FIRST, CHOOSE THE MODE that makes the STRONGEST retrieval cue for THIS expression:
- "character": the protagonist Han Yoil (tag: hanyoil), alone, performs the speaking situation with her body
  pose / face / an optional prop. Use this when the expression is something a PERSON does or
  FEELS - an action, reaction, decision, or interpersonal moment (express gratitude, take up a
  challenge, weigh the pros and cons, express concern).
- "object": NO person at all - a single iconic OBJECT or SCENE that IS the expression. Use this
  when the concept is a thing/situation a person cannot embody by miming, and a learner would
  recognize it faster from the scene itself (heavy traffic - a jam of cars bumper to bumper;
  an empty wallet; a stack of overdue bills). Don't force Han Yoil to gesture at it - just show it.
Pick whichever a learner would decode fastest. When in doubt for a body gesture or feeling, use
character. When the expression is an abstract business/process idea (provide evidence, meet
requirements, pave the way, make arrangements, foster innovation), prefer object or a very simple
symbolic prop cue instead of a vague pose.

Danbooru tag rules (lowercase, words separated by SPACES, comma-separate tags within a field;
convert abstract ideas into concrete drawable objects - BAD: "rejection", "deadline";
GOOD: "raised hand, paper", "calendar, red circle"):

If mode = "character":
- action: Han Yoil's BODY POSE / gesture as Danbooru tags ONLY (e.g. "crossed arms",
  "pushing away, both hands", "covering ears", "reaching out", "holding phone"). Body/hands only.
  NO facial/emotion words (those go in expression). NO subject count (1girl), hair, eyes,
  clothing, background. 1-3 tags, never "none". NEVER use breathing/air tags
  ("sigh", "breath", "steam", "puff") - they render as visible clouds.
- expression: pick EXACTLY ONE key from the EXPRESSION MENU below, matching the speaking moment. Output
  ONLY the menu key verbatim. If the exact mood is absent, pick the CLOSEST key
  (worried -> frown or furrowed_brow, determined -> serious, shocked -> fear_kubrick).
- face_state: gaze direction only ("looking at viewer", "looking away", "looking down",
  "looking up", "looking to the side").
- props: 0-1 drawable object that anchors the speaking moment (held/near her), comma-separated, or "".
  Prefer icon-like props that do not require readable writing. Use "medal", "star badge",
  "magnifying glass", "blank folder", "blank paper", "clipboard with checkmarks", "light bulb",
  "phone", "notebook", "pen", "calendar icon", "red circle".
  Avoid text-generating props: NO certificate, document, contract, report, form, resume, sign,
  label, poster, chart with labels, receipt, screen with text, book cover with text.
- subject: leave "".
- Stay in her character range (see baseline below) - don't make her act out of type.

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

=== HAN YOIL BASELINE (character mode only - a CEILING for in-character range, not the default) ===
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
