"""Notion upload helpers for the speaking sentence database."""
from __future__ import annotations

from pathlib import Path
import os
import time

import pandas as pd
import requests
from notion_client import Client

from .. import config

if not config.NOTION_API_KEY:
    raise RuntimeError("NOTION_API_KEY 가 설정되지 않았습니다 — .env 를 확인하세요.")
if not config.NOTION_SPEAKING_DATABASE_ID:
    raise RuntimeError("NOTION_SPEAKING_DATABASE_ID 가 설정되지 않았습니다 — .env 를 확인하세요.")

_notion = Client(auth=config.NOTION_API_KEY)
_DB_PROPS: dict | None = None

SPEAKING_SCHEMA = {
    "No.": {"rich_text": {}},
    "sentence_unit": {"rich_text": {}},
    "korean_trigger": {"rich_text": {}},
    "sentence unit": {"rich_text": {}},
    "Korean trigger": {"rich_text": {}},
    "register": {
        "select": {
            "options": [
                {"name": "informal", "color": "yellow"},
                {"name": "standard", "color": "blue"},
                {"name": "formal", "color": "purple"},
            ],
        },
    },
    "used in": {
        "select": {
            "options": [
                {"name": "daily", "color": "green"},
                {"name": "workplace", "color": "orange"},
                {"name": "academic", "color": "red"},
                {"name": "customer/service", "color": "pink"},
            ],
        },
    },
    "micro situation": {"rich_text": {}},
    "primary_used_in": {
        "select": {
            "options": [
                {"name": "daily", "color": "green"},
                {"name": "workplace", "color": "orange"},
                {"name": "academic", "color": "red"},
                {"name": "customer/service", "color": "pink"},
            ],
        },
    },
    "used_in": {"rich_text": {}},
    "speaker_role": {"rich_text": {}},
    "listener_role": {"rich_text": {}},
    "relationship": {"rich_text": {}},
    "power_dynamic": {"rich_text": {}},
    "speech_act": {"rich_text": {}},
    "politeness": {"rich_text": {}},
    "micro_situation": {"rich_text": {}},
    "story_function": {"rich_text": {}},
    "character_fit": {"rich_text": {}},
    "avoid_with": {"rich_text": {}},
    "relationship context": {
        "select": {
            "options": [
                {"name": "close_friend", "color": "blue"},
                {"name": "coworker_peer", "color": "green"},
                {"name": "senior_to_junior", "color": "purple"},
                {"name": "junior_to_senior", "color": "pink"},
                {"name": "customer_to_staff", "color": "orange"},
                {"name": "staff_to_customer", "color": "yellow"},
                {"name": "teacher_to_student", "color": "red"},
                {"name": "student_to_teacher", "color": "gray"},
                {"name": "family", "color": "brown"},
                {"name": "stranger", "color": "default"},
            ],
        },
    },
    "speaker role": {"rich_text": {}},
    "listener role": {"rich_text": {}},
    "power dynamic": {
        "select": {
            "options": [
                {"name": "equal", "color": "blue"},
                {"name": "speaker_has_power", "color": "green"},
                {"name": "listener_has_power", "color": "purple"},
                {"name": "service_staff_to_customer", "color": "yellow"},
                {"name": "customer_to_staff", "color": "orange"},
                {"name": "mentor_to_learner", "color": "red"},
                {"name": "learner_to_mentor", "color": "pink"},
            ],
        },
    },
    "speech act": {
        "select": {
            "options": [
                {"name": "ask_for_help", "color": "blue"},
                {"name": "refuse", "color": "red"},
                {"name": "apologize", "color": "pink"},
                {"name": "reassure", "color": "green"},
                {"name": "warn", "color": "orange"},
                {"name": "suggest", "color": "yellow"},
                {"name": "decide", "color": "purple"},
                {"name": "correct", "color": "brown"},
                {"name": "request_action", "color": "blue"},
                {"name": "report_problem", "color": "red"},
                {"name": "check_understanding", "color": "gray"},
                {"name": "express_feeling", "color": "pink"},
                {"name": "set_boundary", "color": "orange"},
                {"name": "negotiate", "color": "purple"},
                {"name": "confirm", "color": "green"},
            ],
        },
    },
    "service direction": {
        "select": {
            "options": [
                {"name": "none", "color": "default"},
                {"name": "customer_to_staff", "color": "orange"},
                {"name": "staff_to_customer", "color": "yellow"},
                {"name": "internal_team", "color": "blue"},
                {"name": "teacher_to_student", "color": "red"},
                {"name": "student_to_teacher", "color": "gray"},
            ],
        },
    },
    "story function": {
        "select": {
            "options": [
                {"name": "setup_problem", "color": "blue"},
                {"name": "reveal_pressure", "color": "orange"},
                {"name": "escalate_conflict", "color": "red"},
                {"name": "state_decision", "color": "purple"},
                {"name": "request_solution", "color": "yellow"},
                {"name": "resist_pressure", "color": "brown"},
                {"name": "soften_tension", "color": "pink"},
                {"name": "expose_mistake", "color": "red"},
                {"name": "confirm_result", "color": "green"},
                {"name": "button_reaction", "color": "gray"},
            ],
        },
    },
    "웹툰번역": {"rich_text": {}},
    "음성": {"files": {}},
    "날짜": {"date": {}},
}


def _db_props() -> dict:
    global _DB_PROPS
    if _DB_PROPS is None:
        db = _notion.databases.retrieve(database_id=config.NOTION_SPEAKING_DATABASE_ID)
        _DB_PROPS = db.get("properties", {})
    return _DB_PROPS


def _refresh_db_props() -> dict:
    global _DB_PROPS
    _DB_PROPS = None
    return _db_props()


def ensure_speaking_schema() -> None:
    props = _db_props()
    missing = {
        name: schema
        for name, schema in SPEAKING_SCHEMA.items()
        if name not in props
    }
    if not missing:
        return
    _notion.databases.update(
        database_id=config.NOTION_SPEAKING_DATABASE_ID,
        properties=missing,
    )
    _refresh_db_props()
    print(f"  ✅ Notion 컬럼 생성: {', '.join(missing)}")


def _title_prop_name() -> str:
    for name, prop in _db_props().items():
        if prop.get("type") == "title":
            return name
    raise RuntimeError("스피킹 Notion DB에 title 속성이 없습니다.")


def _row_get(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _page_title(row: dict) -> str:
    sentence = _row_get(row, "sentence_unit", "sentence unit")
    return sentence.rstrip(".")


def upload_speaking() -> None:
    ensure_speaking_schema()
    df = pd.read_csv(config.CLEAN_CSV, dtype=str).fillna("")
    for _, row in df.iterrows():
        page_ids = _find_speaking_pages(str(row["No."]))
        if page_ids:
            for page_id in page_ids:
                _notion.pages.update(page_id=page_id, properties=_speaking_properties(row))
            print(f"  ✅ SPEAKING #{row['No.']} 업데이트 ({len(page_ids)} page)")
            continue

        payload = {
            "parent": {"database_id": config.NOTION_SPEAKING_DATABASE_ID},
            "properties": _speaking_properties(row),
        }
        r = requests.post(
            "https://api.notion.com/v1/pages",
            headers=config.NOTION_HEADERS,
            json=payload,
            timeout=(10, 60),
        )
        if r.status_code == 200:
            print(f"  ✅ SPEAKING #{row['No.']}")
        else:
            print(f"  ⚠️ SPEAKING #{row['No.']} {r.status_code}: {r.text[:300]}")


def archive_today_pages() -> int:
    """Archive all speaking pages dated with today's pipeline date."""
    ensure_speaking_schema()
    if "날짜" not in _db_props():
        print("  ⚠️ 날짜 컬럼 없음 — archive 생략")
        return 0

    archived = 0
    cur = None
    while True:
        kwargs = {
            "database_id": config.NOTION_SPEAKING_DATABASE_ID,
            "filter": {"property": "날짜", "date": {"equals": config.TODAY_DATE}},
            "page_size": 100,
        }
        if cur:
            kwargs["start_cursor"] = cur
        q = _notion.databases.query(**kwargs)
        for page in q.get("results", []):
            _notion.pages.update(page_id=page["id"], archived=True)
            archived += 1
        if not q.get("has_more"):
            break
        cur = q.get("next_cursor")
    print(f"  🗑️ SPEAKING 오늘 페이지 archive: {archived}개")
    return archived


def _speaking_properties(row: dict) -> dict:
    props_meta = _db_props()
    props = {
        _title_prop_name(): {"title": [{"text": {"content": _page_title(row)}}]},
    }
    if "No." in props_meta:
        props["No."] = {"rich_text": [{"text": {"content": str(row["No."])}}]}
    for col in ("sentence_unit", "korean_trigger", "used_in", "speaker_role", "listener_role",
                "relationship", "power_dynamic", "speech_act", "politeness", "micro_situation",
                "story_function", "character_fit", "avoid_with"):
        if row.get(col) and col in props_meta:
            props[col] = {"rich_text": [{"text": {"content": str(row[col])[:1900]}}]}
    if "sentence unit" in props_meta and props_meta["sentence unit"].get("type") != "title":
        props["sentence unit"] = {"rich_text": [{"text": {"content": _row_get(row, "sentence unit", "sentence_unit")}}]}
    if "Korean trigger" in props_meta:
        props["Korean trigger"] = {"rich_text": [{"text": {"content": _row_get(row, "Korean trigger", "korean_trigger")}}]}
    if "micro situation" in props_meta:
        props["micro situation"] = {"rich_text": [{"text": {"content": _row_get(row, "micro situation", "micro_situation")}}]}
    for col in ("speaker role", "listener role"):
        if row.get(col) and col in props_meta:
            props[col] = {"rich_text": [{"text": {"content": str(row[col])}}]}
    if row.get("웹툰번역") and "웹툰번역" in props_meta:
        props["웹툰번역"] = {"rich_text": [{"text": {"content": str(row["웹툰번역"])[:1900]}}]}
    if "날짜" in props_meta:
        props["날짜"] = {"date": {"start": config.TODAY_DATE}}
    if row.get("register") and "register" in props_meta:
        props["register"] = {"select": {"name": row["register"]}}
    if row.get("used in") and "used in" in props_meta:
        props["used in"] = {"select": {"name": row["used in"]}}
    if row.get("primary_used_in") and "primary_used_in" in props_meta:
        props["primary_used_in"] = {"select": {"name": row["primary_used_in"]}}
    for col in ("relationship context", "power dynamic", "speech act", "service direction", "story function"):
        if row.get(col) and col in props_meta:
            props[col] = {"select": {"name": row[col]}}
    return props


def _row_for_no(no: str) -> dict | None:
    df = pd.read_csv(config.CLEAN_CSV, dtype=str).fillna("")
    matches = df[df["No."].astype(str) == str(no)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _find_speaking_pages(no: str) -> list[str]:
    row = _row_for_no(no)
    if not row:
        return []
    title_prop = _title_prop_name()
    filters = []
    if "No." in _db_props():
        filters.append({"property": "No.", "rich_text": {"equals": str(no)}})
    title = _page_title(row)
    filters.append({"property": title_prop, "title": {"equals": title}})
    old_title = f"{str(row.get('No.', '')).strip()}. {_row_get(row, 'sentence unit', 'sentence_unit')}"
    if old_title != title:
        filters.append({"property": title_prop, "title": {"equals": old_title}})
    results = []
    seen = set()
    for page_filter in filters:
        cur = None
        while True:
            kwargs = {
                "database_id": config.NOTION_SPEAKING_DATABASE_ID,
                "filter": page_filter,
                "page_size": 100,
            }
            if cur:
                kwargs["start_cursor"] = cur
            q = _notion.databases.query(**kwargs)
            for page in q.get("results", []):
                pid = page["id"]
                if pid not in seen:
                    seen.add(pid)
                    results.append(pid)
            if not q.get("has_more"):
                break
            cur = q.get("next_cursor")
    return results


def _upload_file_to_notion(path: Path, content_type: str) -> str | None:
    headers = {
        "Authorization": f"Bearer {config.NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(
                "https://api.notion.com/v1/file_uploads",
                headers={**headers, "Content-Type": "application/json"},
                json={"filename": path.name, "content_type": content_type},
                timeout=(10, 120),
            )
            if r.status_code not in (200, 201):
                print(f"  ⚠️ file_upload 생성 실패 {path.name}: {r.status_code} {r.text[:200]}")
                return None
            upload_id = r.json()["id"]
            with path.open("rb") as f:
                r2 = requests.post(
                    f"https://api.notion.com/v1/file_uploads/{upload_id}/send",
                    headers=headers,
                    files={"file": (path.name, f, content_type)},
                    timeout=(10, 180),
                )
            if r2.status_code not in (200, 201):
                print(f"  ⚠️ 파일 전송 실패 {path.name}: {r2.status_code} {r2.text[:200]}")
                return None
            return upload_id
        except requests.exceptions.RequestException as exc:
            if attempt == 3:
                print(f"  ⚠️ 파일 업로드 실패 {path.name}: {exc}")
                return None
            print(f"  ↻ 파일 업로드 재시도 {attempt}/3 ({path.name})")
            time.sleep(2 * attempt)
    return None


def _upload_image_to_notion(path: Path) -> str | None:
    return _upload_file_to_notion(path, "image/png")


def attach_tts_direct(no: str, file_path: Path) -> bool:
    ensure_speaking_schema()
    page_ids = _find_speaking_pages(no)
    if not page_ids:
        print(f"  ⚠️ SPEAKING TTS #{no}: 페이지 없음")
        return False
    fid = _upload_file_to_notion(file_path, "audio/mpeg")
    if not fid:
        return False
    ok = False
    for page_id in page_ids:
        try:
            _notion.pages.update(
                page_id=page_id,
                properties={"음성": {"files": [{"name": file_path.name, "file_upload": {"id": fid}}]}},
            )
            ok = True
        except Exception as exc:
            print(f"  ⚠️ SPEAKING TTS #{no} 연결 실패: {exc}")
    return ok


def upload_tts_directory(upload_reel: bool = False) -> None:
    mp3s = sorted(
        [f for f in os.listdir(config.TTS_DAY_DIR) if f.endswith(".mp3") and f[:-4].isdigit()],
        key=lambda x: int(x[:-4]),
    )
    if not mp3s:
        print("  ⚠️ SPEAKING TTS mp3 파일 없음")
        return
    for f in mp3s:
        no = f[:-4]
        ok = attach_tts_direct(no, config.TTS_DAY_DIR / f)
        print(f"  {'✅' if ok else '⚠️'} SPEAKING TTS #{no}")


def attach_panels(
    no: str,
    panel_paths: list[Path],
    cover_path: Path | None = None,
    webtoon_translation: str = "",
) -> bool:
    ensure_speaking_schema()
    page_ids = _find_speaking_pages(no)
    if not page_ids:
        print(f"  ⚠️ SPEAKING #{no}: 페이지 없음")
        return False

    children = []
    for panel_path in panel_paths:
        fid = _upload_image_to_notion(panel_path)
        if not fid:
            continue
        children.append({
            "object": "block",
            "type": "image",
            "image": {"type": "file_upload", "file_upload": {"id": fid}},
        })

    if not children:
        print(f"  ⚠️ SPEAKING #{no}: 업로드된 이미지 없음")
        return False

    cover_fid = None
    if cover_path and cover_path.exists():
        cover_fid = _upload_image_to_notion(cover_path)

    ok = False
    for page_id in page_ids:
        try:
            if cover_fid:
                _notion.pages.update(
                    page_id=page_id,
                    cover={"type": "file_upload", "file_upload": {"id": cover_fid}},
                )
            if webtoon_translation and "웹툰번역" in _db_props():
                _notion.pages.update(
                    page_id=page_id,
                    properties={
                        "웹툰번역": {
                            "rich_text": [{"text": {"content": webtoon_translation[:1900]}}],
                        },
                    },
                )
            _notion.blocks.children.append(block_id=page_id, children=children)
            ok = True
        except Exception as exc:
            print(f"  ⚠️ attach_panels #{no}: {exc}")
    return ok
