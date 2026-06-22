"""Generate structured speaking sentence units."""
from __future__ import annotations

import json
import re
import time

import openai
import pandas as pd

from . import config
from .prompt import SPEAKING_SENTENCE_PROMPT

_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

COLUMNS = [
    "sentence_unit",
    "korean_trigger",
    "register",
    "primary_used_in",
    "used_in",
    "speaker_role",
    "listener_role",
    "relationship",
    "power_dynamic",
    "speech_act",
    "politeness",
    "micro_situation",
    "story_function",
    "character_fit",
    "avoid_with",
]

VALID_REGISTERS = {"informal", "standard", "formal"}
VALID_USED_IN = {"daily", "workplace", "academic", "customer/service"}
VALID_ROLES = {
    "main_character",
    "friend",
    "roommate",
    "coworker",
    "boss",
    "employee",
    "staff",
    "customer",
    "professor",
    "student",
    "stranger",
}
VALID_RELATIONSHIPS = {
    "friend_to_friend",
    "roommate_to_roommate",
    "employee_to_boss",
    "boss_to_employee",
    "coworker_to_coworker",
    "staff_to_customer",
    "customer_to_staff",
    "student_to_professor",
    "professor_to_student",
    "stranger_to_stranger",
}
VALID_POWER_DYNAMICS = {
    "equal",
    "upward",
    "downward",
    "service_to_customer",
    "customer_to_service",
}
VALID_SPEECH_ACTS = {
    "request",
    "refusal",
    "apology",
    "clarification",
    "agreement",
    "disagreement",
    "delay_answer",
    "update_status",
    "suggestion",
    "invitation",
    "reassurance",
    "boundary_setting",
    "small_talk",
    "complaint",
    "offer",
    "thanks",
}
VALID_POLITENESS = {"direct", "softened", "polite", "very_polite"}
VALID_STORY_FUNCTIONS = {
    "starts_conflict",
    "escalates_conflict",
    "softens_conflict",
    "resolves_conflict",
    "creates_misunderstanding",
    "reveals_emotion",
    "buys_time",
    "sets_up_punchline",
}
VALID_CHARACTERS = {"hanyoil", "ru-ha", "hanyuyeon", "so-ae", "hyo-jeong"}
_PROMPT_AVOID_MAX = 800

_HEADER_ALIASES = {
    "sentence": "sentence_unit",
    "sentence unit": "sentence_unit",
    "english sentence": "sentence_unit",
    "korean trigger": "korean_trigger",
    "korean prompt": "korean_trigger",
    "korean cue": "korean_trigger",
    "used in": "primary_used_in",
    "context": "primary_used_in",
    "micro situation": "micro_situation",
    "situation": "micro_situation",
    "speaker role": "speaker_role",
    "listener role": "listener_role",
    "power dynamic": "power_dynamic",
    "speech act": "speech_act",
    "story function": "story_function",
    "story role": "story_function",
    "story beat": "story_function",
    "character fit": "character_fit",
    "avoid with": "avoid_with",
}


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def load_history() -> list[str]:
    if not config.HISTORY_PATH.exists():
        return []
    return [
        line.strip()
        for line in config.HISTORY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _append_history(sentences: list[str]) -> None:
    config.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.HISTORY_PATH.open("a", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.split("|")[1:-1]]


def _normalize_header(cell: str) -> str | None:
    key = cell.strip().lower().replace("-", "_")
    if key in _HEADER_ALIASES:
        return _HEADER_ALIASES[key]
    for col in COLUMNS:
        if key == col.lower():
            return col
    return None


def _json_cell(value) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "").strip()


def _coerce_record(item: dict) -> list[str]:
    record = {col: "" for col in COLUMNS}
    for col in COLUMNS:
        record[col] = _json_cell(item.get(col, ""))
    if not record["primary_used_in"] and isinstance(item.get("used_in"), list) and item["used_in"]:
        record["primary_used_in"] = str(item["used_in"][0]).strip()
    return [record[col] for col in COLUMNS]


def _extract_json_array(text: str):
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, list) else None


def _parse_json_rows(text: str) -> list[list[str]]:
    data = _extract_json_array(text)
    if data is None:
        return []
    rows = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            print(f"  ⚠️ JSON item {i}: object가 아님")
            continue
        rows.append(_coerce_record(item))
    return rows


def _parse_table(text: str) -> list[list[str]]:
    table_lines = [ln for ln in text.splitlines() if "|" in ln and "---" not in ln]
    if not table_lines:
        return []

    header_idx = next(
        (i for i, ln in enumerate(table_lines)
         if "sentence unit" in [cell.lower() for cell in _split_row(ln)]),
        None,
    )
    if header_idx is None:
        print("  ⚠️ 표 헤더('sentence unit')를 찾지 못했습니다.")
        return []

    pos_to_col = [_normalize_header(cell) for cell in _split_row(table_lines[header_idx])]
    rows: list[list[str]] = []
    for line in table_lines[header_idx + 1:]:
        cells = _split_row(line)
        if not cells or "sentence unit" in [cell.lower() for cell in cells]:
            continue
        record = {col: "" for col in COLUMNS}
        for pos, col in enumerate(pos_to_col):
            if col and pos < len(cells):
                record[col] = cells[pos]
        rows.append([record[col] for col in COLUMNS])
    return rows


def _parse_output(text: str) -> list[list[str]]:
    rows = _parse_json_rows(text)
    if rows:
        return rows
    return _parse_table(text)


def _list_cell(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        data = [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in data] if isinstance(data, list) else []


# GPT가 enum 근처로 빗나갈 때 가장 가까운 유효값으로 흡수한다.
# (관측: assurance/comment, shows_character, customer_to_service …)
_ENUM_ALIASES: dict[str, dict[str, str]] = {
    "speech_act": {
        "assurance": "reassurance",
        "reassure": "reassurance",
        "comment": "small_talk",
        "remark": "small_talk",
        "observation": "small_talk",
        "decline": "refusal",
        "agree": "agreement",
        "disagree": "disagreement",
        "ask": "request",
        "thank": "thanks",
        "invite": "invitation",
        "suggest": "suggestion",
        "apologize": "apology",
        "complain": "complaint",
        "update": "update_status",
        "status_update": "update_status",
        "delay": "delay_answer",
        "set_boundary": "boundary_setting",
    },
    "story_function": {
        "shows_character": "reveals_emotion",
        "reveals_character": "reveals_emotion",
        "reveal_emotion": "reveals_emotion",
        "starts_conflict ": "starts_conflict",
        "start_conflict": "starts_conflict",
        "escalate_conflict": "escalates_conflict",
        "soften_conflict": "softens_conflict",
        "resolve_conflict": "resolves_conflict",
        "punchline": "sets_up_punchline",
        "sets_punchline": "sets_up_punchline",
        "buy_time": "buys_time",
    },
    "relationship": {
        "customer_to_service": "customer_to_staff",
        "service_to_customer": "staff_to_customer",
        "staff_to_client": "staff_to_customer",
        "client_to_staff": "customer_to_staff",
    },
    "power_dynamic": {
        "service_to_client": "service_to_customer",
        "client_to_service": "customer_to_service",
    },
}

# 매핑 불가한 값일 때 떨어뜨릴 안전 기본값.
_ENUM_FALLBACK: dict[str, str] = {
    "register": "standard",
    "primary_used_in": "daily",
    "speaker_role": "friend",
    "listener_role": "friend",
    "relationship": "friend_to_friend",
    "power_dynamic": "equal",
    "speech_act": "small_talk",
    "politeness": "softened",
    "story_function": "reveals_emotion",
}

_ENUM_CHECKS: dict[str, set] = {
    "register": VALID_REGISTERS,
    "primary_used_in": VALID_USED_IN,
    "speaker_role": VALID_ROLES,
    "listener_role": VALID_ROLES,
    "relationship": VALID_RELATIONSHIPS,
    "power_dynamic": VALID_POWER_DYNAMICS,
    "speech_act": VALID_SPEECH_ACTS,
    "politeness": VALID_POLITENESS,
    "story_function": VALID_STORY_FUNCTIONS,
}


def _coerce_enum_rows(rows: list[list[str]]) -> None:
    """잘못된 enum 값을 alias→유효값, 그래도 안 되면 기본값으로 in-place 교정."""
    for i, row in enumerate(rows, 1):
        for col, allowed in _ENUM_CHECKS.items():
            idx = COLUMNS.index(col)
            raw = row[idx].strip()
            value = raw.lower()
            if not value or value in allowed:
                row[idx] = value if value else raw
                continue
            mapped = _ENUM_ALIASES.get(col, {}).get(value)
            if mapped and mapped in allowed:
                print(f"  🔧 row {i}: {col}='{raw}' → '{mapped}'")
                row[idx] = mapped
            else:
                fallback = _ENUM_FALLBACK.get(col, "")
                print(f"  🔧 row {i}: {col}='{raw}' 매핑 불가 → 기본값 '{fallback}'")
                row[idx] = fallback


# relationship 이 도메인을 단정하는 경우 — primary_used_in 이 어긋나면 relationship 을 정본으로 삼아
# 교정한다. (관측: customer_to_staff 인데 primary=workplace → service 롤 로직이 hyo-jeong 을 끌어와
# workplace 캐스트와 충돌 → 에피소드 붕괴. friend/roommate/stranger 는 도메인 자유라 강제하지 않는다.)
_RELATIONSHIP_DOMAIN: dict[str, str] = {
    "staff_to_customer": "customer/service",
    "customer_to_staff": "customer/service",
    "student_to_professor": "academic",
    "professor_to_student": "academic",
}


def _coerce_domain_consistency(rows: list[list[str]]) -> None:
    """relationship 이 도메인을 단정하면 primary_used_in/used_in 을 거기 맞춰 in-place 교정."""
    rel_idx = COLUMNS.index("relationship")
    dom_idx = COLUMNS.index("primary_used_in")
    used_idx = COLUMNS.index("used_in")
    for i, row in enumerate(rows, 1):
        rel = row[rel_idx].strip().lower()
        required = _RELATIONSHIP_DOMAIN.get(rel)
        if not required:
            continue
        if row[dom_idx].strip().lower() != required:
            print(f"  🔧 row {i}: primary_used_in='{row[dom_idx]}' ⊥ relationship='{rel}' → '{required}'")
            row[dom_idx] = required
        used = _list_cell(row[used_idx])
        if required not in used:
            used = [required] + [u for u in used if u != required]
            row[used_idx] = json.dumps(used, ensure_ascii=False)


def _validate_rows(rows: list[list[str]]) -> None:
    checks = {
        "register": VALID_REGISTERS,
        "primary_used_in": VALID_USED_IN,
        "speaker_role": VALID_ROLES,
        "listener_role": VALID_ROLES,
        "relationship": VALID_RELATIONSHIPS,
        "power_dynamic": VALID_POWER_DYNAMICS,
        "speech_act": VALID_SPEECH_ACTS,
        "politeness": VALID_POLITENESS,
        "story_function": VALID_STORY_FUNCTIONS,
    }
    for i, row in enumerate(rows, 1):
        for col, allowed in checks.items():
            idx = COLUMNS.index(col)
            value = row[idx].strip().lower()
            if value and value not in allowed:
                print(f"  ⚠️ row {i}: {col}='{row[idx]}'")
        used_in = _list_cell(row[COLUMNS.index("used_in")])
        if not used_in:
            print(f"  ⚠️ row {i}: used_in 비어 있음")
        for value in used_in:
            if value not in VALID_USED_IN:
                print(f"  ⚠️ row {i}: used_in contains '{value}'")
        primary = row[COLUMNS.index("primary_used_in")].strip().lower()
        if primary and used_in and primary not in used_in:
            print(f"  ⚠️ row {i}: primary_used_in='{primary}' not in used_in={used_in}")
        for value in _list_cell(row[COLUMNS.index("character_fit")]):
            if value not in VALID_CHARACTERS:
                print(f"  ⚠️ row {i}: character_fit contains '{value}'")
        for col in COLUMNS:
            idx = COLUMNS.index(col)
            if col == "avoid_with":
                continue
            if not row[idx].strip():
                print(f"  ⚠️ row {i}: {col} 비어 있음")


def _ask_gpt(theme: str, n: int, avoid: list[str]) -> str | None:
    avoid_block = "\n".join(f"- {item}" for item in avoid[-_PROMPT_AVOID_MAX:]) or "(none yet)"
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You generate concise structured English speaking practice material.",
                },
                {
                    "role": "user",
                    "content": SPEAKING_SENTENCE_PROMPT.format(
                        theme=theme,
                        avoid=avoid_block,
                        n=n,
                    ),
                },
            ],
        )
        return resp.choices[0].message.content
    except (openai.AuthenticationError, openai.PermissionDeniedError):
        raise
    except Exception as exc:
        print(f"⚠️ OpenAI error (speaking sentences): {exc}")
        return None


def generate_set(
    n: int = config.DEFAULT_COUNT,
    theme: str = config.DEFAULT_THEME,
    max_rounds: int = 4,
) -> bool:
    history = load_history()
    seen = {_norm(item) for item in history}
    sentence_idx = COLUMNS.index("sentence_unit")
    collected: list[list[str]] = []

    for rnd in range(1, max_rounds + 1):
        need = n - len(collected)
        if need <= 0:
            break
        print(f"  🔸 speaking 생성 라운드 {rnd}: {need}개 요청 (누적 {len(collected)}/{n})")
        text = _ask_gpt(theme, need, history + [row[sentence_idx] for row in collected])
        if not text:
            time.sleep(1)
            continue
        for row in _parse_output(text):
            sentence = _norm(row[sentence_idx])
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            collected.append(row)
            if len(collected) >= n:
                break
        time.sleep(1)

    if not collected:
        print("❌ 생성 실패: 새 speaking sentence를 얻지 못했습니다.")
        return False
    if len(collected) < n:
        print(f"  ⚠️ {n}개 목표 중 {len(collected)}개만 확보 — 최선본으로 진행")

    _coerce_enum_rows(collected)
    _coerce_domain_consistency(collected)
    _validate_rows(collected)
    pd.DataFrame(collected, columns=COLUMNS).to_csv(
        config.STRUCTURED_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    _append_history([row[sentence_idx] for row in collected])
    print(f"✅ 스피킹 구조화 CSV: {config.STRUCTURED_CSV} ({len(collected)}개)")
    return True


def clean() -> None:
    df = pd.read_csv(config.STRUCTURED_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[df["sentence_unit"] != "sentence_unit"].reset_index(drop=True)
    df.insert(0, "No.", range(1, len(df) + 1))
    df.to_csv(config.CLEAN_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 스피킹 정리 CSV: {config.CLEAN_CSV}")


def generate_structured(
    n: int = config.DEFAULT_COUNT,
    theme: str = config.DEFAULT_THEME,
) -> bool:
    return generate_set(n=n, theme=theme)
