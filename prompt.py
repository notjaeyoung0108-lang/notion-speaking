SPEAKING_SENTENCE_PROMPT = """
## Instruction

You generate high-frequency English speaking sentence units for a Korean learner.

This is STAGE 1 of a 3-stage pipeline:
1. Generate only high-frequency spoken sentence units.
2. Add social/scenario metadata.
3. Build the webtoon/sitcom scene.

In this stage, do NOT optimize for sitcom story, character casting, or jokes.
Your only job is to choose sentences the learner will hear and need repeatedly.

---

## Target Learner

- Native language: Korean
- English level: B1-B2
- Goal: daily conversation and workplace conversation
- Current priority: speaking fluency over vocabulary depth

---

## Sentence Unit Definition

A sentence unit MUST be:
- a complete natural English sentence,
- short enough to say out loud easily,
- useful in realistic conversation,
- reusable through a simple pattern,
- connected to one clear speaking intention,
- common enough that a normal English speaker might say it many times in a month.

A sentence unit is NOT:
- an isolated word,
- a collocation by itself,
- a grammar explanation,
- a long written-style sentence,
- a rare joke line,
- a scene-specific one-off sentence.

---

## Good Examples

- I haven’t decided yet.
- I’m not sure how to explain it.
- Can I get back to you on that?
- I didn’t mean it that way.
- That makes sense.
- I see what you mean.
- I’m still working on it.
- I’ll check and let you know.
- Do you have a minute?
- Sorry, I missed that.
- What do you mean?
- That works for me.
- I’m running a little late.
- Let me think about it.
- It depends.
- I’ll take care of it.
- I didn’t catch that.

## Bad Examples

- Decision.
- Make a decision.
- The manager made an important decision at the meeting.
- It is what it is, bro.
- The quarterly compliance roadmap requires cross-functional alignment.
- I would be honored to accompany you to the dining establishment.
- The photocopier represents my emotional state.

---

## Input

Theme or situation:
{theme}

Already generated sentences to avoid:
{avoid}

If the avoid list is empty, generate fresh entries.

---

## Frequency Priority

Before choosing a sentence, silently ask:
"Would a normal English speaker plausibly say this sentence many times in a month?"

Prefer these high-frequency speaking jobs:
- asking for time or attention
- asking for clarification
- delaying an answer
- giving status
- agreeing or reacting
- soft disagreement
- making small requests
- setting boundaries
- offering help
- apologizing lightly
- confirming plans
- checking availability
- changing or confirming decisions

Avoid sentences that are:
- funny but rare,
- too scene-specific,
- too corporate,
- too dramatic,
- mainly useful for reading/writing, not speaking,
- natural once, but not reusable.

---

## Output Format

Return a JSON array only.
Do not wrap it in Markdown.
Do not include comments or extra text.

Every object must contain exactly these fields:

- sentence_unit: string
  A complete natural English sentence. Short and easy to say out loud.

- korean_trigger: string
  A natural Korean sentence that would make the learner want to say this English sentence.
  This should not be a literal translation only.

- speaking_intent: string
  A short English label for the speaking job, e.g. "ask for clarification", "set a boundary".

- frequency_reason: string
  One short English reason why this sentence is common and reusable.

Generate EXACTLY {n} objects.
"""


SPEAKING_METADATA_PROMPT = """
## Instruction

You add social/scenario metadata to already-selected high-frequency English speaking sentences.

This is STAGE 2 of a 3-stage pipeline:
1. Sentence units were already selected for frequency and usefulness.
2. You add metadata that helps a later webtoon/sitcom scene use the sentence naturally.
3. Another prompt will write the actual scene.

Do NOT replace the sentence_unit.
Do NOT make the sentence funnier or rarer.
Do NOT optimize for a punchline.
Your job is to identify who might say this sentence, to whom, under what social pressure.

---

## Input Sentence Units

{items}

---

## Domains

Allowed domains:
- daily
- workplace
- academic
- customer/service

For this learner, prefer:
- daily and workplace for most sentences,
- customer/service only when the sentence naturally fits short service interactions,
- academic only when the sentence naturally fits school/classroom flashback situations.

primary_used_in must be exactly ONE allowed domain.
used_in must be an array containing one or more allowed domains.

---

## Register

Register must be ONE of:
- informal
- standard
- formal

Most sentences should be "standard" unless clearly casual or clearly formal.

---

## Character Fit

character_fit is only a weak downstream hint.
Choose likely characters from:
- hanyoil: effortful, approval-seeking, over-prepares, wants to handle things well
- ru-ha: dry one-liner, deflects tension with jokes, quietly helps
- hanyuyeon: team lead, controls through planning, high standards
- so-ae: detail-oriented, corrects small errors, over-explains when interested
- hyo-jeong: off-angle sincere friend, service-job hopper, reacts oddly to the current object/situation

Do not force sitcom casting. If unsure, choose 1-2 plausible characters.

---

## Required JSON Fields

Return a JSON array with the SAME number and order as the input.
Every object must contain exactly these fields:

- sentence_unit: string
  Copy the exact input sentence_unit.

- korean_trigger: string
  Copy or lightly improve the input korean_trigger.

- register: "informal" | "standard" | "formal"

- primary_used_in: "daily" | "workplace" | "academic" | "customer/service"

- used_in: array of allowed domains

- speaker_role: one of:
  "main_character", "friend", "roommate", "coworker", "boss", "employee", "staff",
  "customer", "professor", "student", "stranger"

- listener_role: same enum as speaker_role

- relationship: one of:
  "friend_to_friend",
  "roommate_to_roommate",
  "employee_to_boss",
  "boss_to_employee",
  "coworker_to_coworker",
  "staff_to_customer",
  "customer_to_staff",
  "student_to_professor",
  "professor_to_student",
  "stranger_to_stranger"

- power_dynamic: "equal" | "upward" | "downward" | "service_to_customer" | "customer_to_service"

- speech_act: one of:
  "request", "refusal", "apology", "clarification", "agreement", "disagreement",
  "delay_answer", "update_status", "suggestion", "invitation", "reassurance",
  "boundary_setting", "small_talk", "complaint", "offer", "thanks"

- politeness: "direct" | "softened" | "polite" | "very_polite"

- micro_situation: Korean string
  A concrete Korean speaking situation. Keep it ordinary and reusable.

- story_function: one of:
  "starts_conflict",
  "escalates_conflict",
  "softens_conflict",
  "resolves_conflict",
  "creates_misunderstanding",
  "reveals_emotion",
  "buys_time",
  "sets_up_punchline"

- character_fit: array of likely characters from:
  "hanyoil", "ru-ha", "hanyuyeon", "so-ae", "hyo-jeong"

- avoid_with: array of short warnings, e.g. ["too_casual_for_boss", "not_for_customer_service"]

---

## Metadata Principles

- Match relationship and power_dynamic logically.
- employee_to_boss usually means upward.
- boss_to_employee usually means downward.
- staff_to_customer usually means service_to_customer.
- customer_to_staff usually means customer_to_service.
- Do not make every workplace sentence formal.
- Do not force all service sentences to be staff_to_customer; include customer_to_staff too.
- story_function should describe how the sentence naturally works in a tiny scene.
- character_fit should reflect sentence/social fit, not domain quota.
- avoid_with should be an empty array when there is no meaningful warning.

---

## Output Format

Return a JSON array only.
Do not wrap it in Markdown.
Do not include comments or extra text.
"""
