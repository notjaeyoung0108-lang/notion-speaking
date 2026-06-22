SPEAKING_SENTENCE_PROMPT = """
## Instruction

You are generating an English speaking sentence unit for a Korean learner.

The goal is:
- To help the learner speak naturally in real conversations.
- To build short, reusable sentence units.
- To improve active speaking, not passive vocabulary knowledge.
- To prioritize sentences the learner is likely to actually say.
- To make each sentence feel connected to a real-life situation.
- To avoid textbook-like, stiff, overly formal, or unnatural sentences.

The output should feel like:
- something a native speaker might actually say,
- something useful in daily conversation,
- something useful in workplace conversation,
- something the learner can say within 3 seconds,
- something reusable with small changes.

---

## Target Learner

- Native language: Korean
- English level: B1–B2
- Goal: daily conversation and workplace conversation
- Current priority: speaking fluency over vocabulary depth

---

## Definition of Sentence Unit

A sentence unit is NOT:
- an isolated word,
- a collocation by itself,
- a grammar explanation,
- a long written-style sentence,
- a memorized textbook phrase with limited use.

A sentence unit MUST be:
- a complete natural English sentence,
- short enough to say out loud easily,
- useful in realistic conversation,
- reusable through a simple pattern,
- connected to a clear speaking intention.

Good examples:
- I haven’t decided yet.
- I’m not sure how to explain it.
- Can I get back to you on that?
- I didn’t mean it that way.
- I was going to, but I changed my mind.
- That makes sense.
- I see what you mean.
- I’m still working on it.

Bad examples:
- Decision.
- Make a decision.
- I would like to hereby express my uncertainty regarding the matter.
- The manager made an important decision at the meeting.
- If I had known that this would happen, I would have prepared more thoroughly for the situation.

---

## Input

Theme or situation:
{theme}

Already generated sentences to avoid:
{avoid}

If the avoid list is empty, generate fresh entries.

---

## Generation Principles

1. Focus on SENTENCE UNITS, not collocations.
2. Each sentence must be something the learner can actually say in conversation.
3. Prefer short, natural spoken English.
4. Prefer first-person and second-person sentences.
5. Each sentence should express one clear speaking intention.
6. Avoid long, complex, written-style sentences.
7. Avoid rare idioms, slang-heavy expressions, and overly formal phrases.
8. Avoid sentences that are grammatically correct but unlikely in real conversation.
9. Avoid generic textbook subjects like "The manager", "The company", or "The committee".
10. Do NOT force collocations.
11. Collocations may appear only if they naturally belong in the sentence.
12. Do NOT over-explain grammar.
13. Do NOT generate near-duplicate sentences.
14. Do NOT artificially balance daily and workplace contexts.
15. Choose the most natural context for each sentence.
16. Give each sentence enough social/story metadata for a sitcom scene.
17. Do not leave the relationship/context fields generic. Make clear who would say it, to whom,
    under what pressure, and what story job the sentence performs.

---

## Register

Register must be ONE of:
- informal
- standard
- formal

Most sentences should be "standard" unless the sentence is clearly casual or clearly formal.

---

## Domains

Allowed domains:
- daily
- workplace
- academic
- customer/service

primary_used_in must be exactly ONE allowed domain.
used_in must be an array containing one or more allowed domains.
used_in can contain multiple domains, but primary_used_in must be exactly one.

---

## Required JSON Fields

Every object must contain exactly these fields:

- sentence_unit: string
  A complete natural English sentence. Short and easy to say out loud.

- korean_trigger: string
  A natural Korean sentence that would make the learner want to say the English sentence.
  This should NOT be a literal translation only.

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

- Do not make every workplace sentence formal.
- Do not force all service sentences to be staff_to_customer; include customer_to_staff too.
- Match relationship and power_dynamic logically:
  employee_to_boss usually means upward.
  boss_to_employee usually means downward.
  staff_to_customer usually means service_to_customer.
  customer_to_staff usually means customer_to_service.
- character_fit should reflect character personality, not just domain.
- avoid_with should be an empty array when there is no meaningful warning.

---

## Output Format

Return a JSON array only.
Do not wrap it in Markdown.
Do not include comments or extra text.

---

Generate EXACTLY {n} objects.
"""
