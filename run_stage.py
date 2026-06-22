"""Stage runner for the speaking sentence + comic pipeline.

Usage:
  python -m notion_speaking.run_stage sentences
  python -m notion_speaking.run_stage scenarios
  python -m notion_speaking.run_stage notion
  python -m notion_speaking.run_stage tts
  python -m notion_speaking.run_stage ttsup
  python -m notion_speaking.run_stage images
  python -m notion_speaking.run_stage lore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import comic_client, config

STATE_DIR = config.DATA_DIR / "state"
SCENARIO_DATA_JSON = STATE_DIR / f"scenario_data-{config.YY_MM_DD}.json"


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def stage_sentences(n: int, theme: str) -> dict:
    from . import sentences

    if not sentences.generate_structured(n=n, theme=theme):
        return {"ok": False, "error": "speaking sentence generation failed"}
    sentences.clean()
    return {"ok": True, "clean_csv": str(config.CLEAN_CSV)}


def stage_scenarios() -> dict:
    _ensure_state_dir()
    data = comic_client.generate_scenarios()
    SCENARIO_DATA_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(data), "scenario_data": str(SCENARIO_DATA_JSON)}


def _load_scenario_data() -> list[dict]:
    if not SCENARIO_DATA_JSON.exists():
        raise FileNotFoundError(f"{SCENARIO_DATA_JSON} 없음 — 먼저 scenarios 단계를 실행하세요.")
    return json.loads(SCENARIO_DATA_JSON.read_text(encoding="utf-8"))


def stage_notion() -> dict:
    from .integrations import notion

    notion.upload_speaking()
    return {"ok": True, "database_id": config.NOTION_SPEAKING_DATABASE_ID}


def stage_images(attach: bool = True) -> dict:
    results = comic_client.render_images(_load_scenario_data())
    if attach:
        comic_client.upload_and_attach(results)
    return {"ok": True, "rendered": sum(1 for result in results if result[1]), "attached": attach}


def stage_tts() -> dict:
    from . import tts

    data = _load_scenario_data()
    tts.generate(scenario_data=data)
    reel = tts.build_episode_reel()
    return {"ok": True, "tts_dir": str(config.TTS_DAY_DIR), "reel": str(reel) if reel else None}


def stage_ttsup() -> dict:
    from .integrations import notion

    notion.upload_tts_directory()
    return {"ok": True, "tts_dir": str(config.TTS_DAY_DIR)}


def stage_lore(limit: int | None = None) -> dict:
    from .comic.lore_keeper import LoreKeeper

    memo = LoreKeeper().run(limit=limit)
    return {"ok": bool(memo), "memo_keys": list(memo.keys()) if memo else []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sentences", "scenarios", "notion", "tts", "ttsup", "images", "lore"])
    parser.add_argument("--n", type=int, default=config.DEFAULT_COUNT)
    parser.add_argument("--theme", default=config.DEFAULT_THEME)
    parser.add_argument("--no-attach", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "sentences":
        result = stage_sentences(args.n, args.theme)
    elif args.stage == "scenarios":
        result = stage_scenarios()
    elif args.stage == "notion":
        result = stage_notion()
    elif args.stage == "tts":
        result = stage_tts()
    elif args.stage == "ttsup":
        result = stage_ttsup()
    elif args.stage == "lore":
        result = stage_lore(limit=args.limit)
    else:
        result = stage_images(attach=not args.no_attach)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
