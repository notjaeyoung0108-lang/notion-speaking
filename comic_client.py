"""comic_client.py — 만화 파이프라인 오케스트레이터.

generate_scenarios(): GPT 시나리오 생성 (text-only, GPU 없음).
                      반환값: list[dict] — 각 단어의 시나리오 + example sentence
render_images()     : Modal GPU 이미지 생성 + 로컬 다운로드.
                      반환값: list[tuple[str, list[Path], str]]  — (word_no, [panel PNGs], webtoon translation)
upload_and_attach() : Notion 직접 업로드 → 페이지 컨텐츠에 이미지 블록 추가
render_all()        : generate_scenarios + render_images 한번에 (하위 호환용)
"""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from . import config


def _first(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _scenario_module():
    """Lazy import — openai / modal 미설치 환경에서 import 오류 방지."""
    from .comic import generate_scenario as gs
    return gs


def _load_word_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, 1):
        row.setdefault("No.", str(i))
        if not str(row.get("No.", "")).strip():
            row["No."] = str(i)
    return [_as_comic_word(row) for row in rows]


def _as_comic_word(row: dict) -> dict:
    """Expose speaking rows through the legacy comic word shape.

    The copied comic engine still expects collocation-era keys, so this adapter
    fills those keys in memory while preserving the richer speaking metadata.
    """
    sentence = _first(row, "sentence_unit", "sentence unit", "collocation unit")
    trigger = _first(row, "korean_trigger", "Korean trigger", "translation")
    situation = _first(row, "micro_situation", "micro situation", "nuance (Korean)")
    primary_used_in = _first(row, "primary_used_in", "used in")

    adapted = dict(row)
    adapted.setdefault("sentence unit", sentence)
    adapted.setdefault("Korean trigger", trigger)
    adapted.setdefault("micro situation", situation)
    adapted.setdefault("used in", primary_used_in)
    adapted.setdefault("collocation unit", sentence)
    adapted.setdefault("meaning", trigger)
    adapted.setdefault("nuance (Korean)", situation or trigger)
    adapted.setdefault("translation", trigger)
    adapted.setdefault("example sentence", sentence)
    adapted.setdefault("scenario metadata", {
        "relationship context": _first(row, "relationship", "relationship context"),
        "speaker role": _first(row, "speaker_role", "speaker role"),
        "listener role": _first(row, "listener_role", "listener role"),
        "power dynamic": _first(row, "power_dynamic", "power dynamic"),
        "speech act": _first(row, "speech_act", "speech act"),
        "service direction": _first(row, "relationship", "service direction"),
        "story function": _first(row, "story_function", "story function"),
        "politeness": _first(row, "politeness"),
        "character_fit": _first(row, "character_fit"),
        "avoid_with": _first(row, "avoid_with"),
    })
    return adapted


def generate_scenarios(
    csv_path: Path | str | None = None,
    seed_base: int = 1234,
) -> list[dict]:
    """GPT 시나리오 생성만 (GPU 없음). render_images()에 넘길 scenario_data 반환.

    Returns:
        [{"word_no": str, "subdir": str, "batch_item": dict, "example": str}, ...]
    """
    # 시나리오 생성은 comic 대사로 덮이지 않은 structured CSV 를 우선 사용한다.
    csv_path = Path(csv_path or (config.STRUCTURED_CSV if config.STRUCTURED_CSV.exists() else config.CLEAN_CSV))
    if not csv_path.exists():
        print(f"⚠️ CSV 없음: {csv_path}")
        return []

    word_rows = _load_word_rows(csv_path)
    if not word_rows:
        return []

    gs    = _scenario_module()
    seeds = [seed_base + i for i in range(1, len(word_rows) + 1)]

    result = gs.generate_scenarios_batch(words=word_rows, seeds=seeds)

    return [
        {
            "word_no":    str(word.get("No.", "?")),
            "subdir":     subdir,
            "batch_item": batch_item,
            "example":     ex["example"],
            "speaker":     ex["speaker"],
            "translation": ex.get("translation", ""),
            "panels":      batch_item["panels"],
            "dialogue":    dlg["dialogue"],
            "plan":        result.get("plans", {}).get(str(word.get("No.", "?"))),
            # 커버(복습 카드) spec — 뉘앙스 전용 GPT 호출로 새로 설계(복장만 만화에서 상속). images 단계가 사용.
            "review_card": gs.build_review_card(
                word,
                result.get("plans", {}).get(str(word.get("No.", "?"))),
                batch_item["panels"],
                batch_item["seed"],
            ),
        }
        for word, subdir, batch_item, ex, dlg in zip(
            word_rows, result["subdirs"], result["batch"],
            result["examples"], result["dialogues"],
        )
    ]


def _webtoon_translation(item: dict) -> str:
    lines = []
    for panel in item.get("panels") or []:
        char = str(panel.get("char") or "").strip()
        kr = str(panel.get("bubble_kr") or "").strip()
        if not char or not kr:
            continue
        lines.append(f"{char}: {kr}")
    return "\n".join(lines)


def render_images(scenario_data: list[dict]) -> list[tuple[str, list[Path], str]]:
    """Modal GPU 이미지 생성 + 로컬 다운로드. generate_scenarios() 반환값을 받는다."""
    if not scenario_data:
        return []
    if config.MODAL_DISABLED:
        print("ℹ️ MODAL_DISABLED=1 — 이미지 렌더를 건너뜁니다.")
        return []

    gs    = _scenario_module()
    batch = [d["batch_item"] for d in scenario_data]

    try:
        gs.run_images_batch(batch)
    except Exception as exc:
        print(f"  ✗ 이미지 생성 실패: {exc}")
        return []

    results: list[tuple[str, list[Path], str]] = []
    for d in scenario_data:
        word_no   = d["word_no"]
        subdir    = d["subdir"]
        local_dir = config.COMIC_DAY_DIR / subdir
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # subdir 은 GPT/CSV 유래 값이라 shell=True + f-string 보간은 명령어 인젝션 위험.
        # 리스트 인자로 전달해 셸을 거치지 않는다.
        dl = subprocess.run(
            ["modal", "volume", "get", "--force", "comic-output",
             str(subdir), str(local_dir.parent)],
            cwd=gs.HERE,
        )
        if dl.returncode != 0:
            print(f"  ⚠️ volume get 실패 No.{word_no} (returncode={dl.returncode})")

        panels = sorted(local_dir.glob("panel_*.png"))
        if panels:
            print(f"  ✓ No.{word_no} {len(panels)}컷 → {subdir}/")
        results.append((word_no, panels, _webtoon_translation(d)))

    # ── 커버용 복습 카드(단어별 정사각 썸네일) 생성 + 다운로드 ──
    #    GPU 렌더만 sd_generate(--review-cards) 가 하고, Notion 커버 설정은 stage_images 가 한다.
    review_cards = [d["review_card"] for d in scenario_data if d.get("review_card")]
    if review_cards:
        try:
            gs.run_review_cards_batch(review_cards)
            rc_local = config.COMIC_DAY_DIR / "review_cards"
            dl = subprocess.run(
                ["modal", "volume", "get", "--force", "comic-output",
                 "review_cards", str(rc_local.parent)],
                cwd=gs.HERE,
            )
            if dl.returncode != 0:
                print(f"  ⚠️ 복습카드 volume get 실패 (returncode={dl.returncode})")
            else:
                n = len(sorted(rc_local.glob("word_*.png"))) if rc_local.exists() else 0
                print(f"  ✓ 복습카드 {n}장 → review_cards/")
        except Exception as exc:
            print(f"  ⚠️ 복습 카드 생성 실패(커버 생략): {exc}")

    return results


def render_all(
    csv_path: Path | str | None = None,
    seed_base: int = 1234,
) -> list[tuple[str, list[Path], str]]:
    """generate_scenarios + render_images 한번에 (하위 호환용)."""
    scenario_data = generate_scenarios(csv_path=csv_path, seed_base=seed_base)
    return render_images(scenario_data)


def upload_and_attach(results: list[tuple]) -> None:
    """Upload panel PNGs and attach them to today's speaking Notion pages."""
    if not results:
        return
    from .integrations import notion

    for result in results:
        if len(result) == 2:
            word_no, panels = result
            webtoon_translation = ""
        else:
            word_no, panels, webtoon_translation = result[:3]
        if not panels:
            continue
        cover_path = panels[0].parent / "cover.png"
        ok = notion.attach_panels(
            word_no,
            panels,
            cover_path=cover_path if cover_path.exists() else None,
            webtoon_translation=webtoon_translation,
        )
        if ok:
            print(f"  ✅ SPEAKING No.{word_no} comic {len(panels)}컷 페이지에 추가됨")
