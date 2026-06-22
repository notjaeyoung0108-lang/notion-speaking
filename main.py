"""Daily speaking sentence + comic pipeline."""
from __future__ import annotations

import argparse
import traceback

from . import comic_client, config, sentences


def run(
    n: int = config.DEFAULT_COUNT,
    theme: str = config.DEFAULT_THEME,
    images: bool = True,
    tts_enabled: bool = True,
    notion: bool = True,
) -> None:
    print("=" * 50)
    print("🚀 스피킹 문장 + 만화 자동화")
    print(f"📅 {config.YY_MM_DD}    count: {n}")
    print("=" * 50)

    if not sentences.generate_structured(n=n, theme=theme):
        return
    sentences.clean()

    if notion:
        from .integrations import notion as notion_integration
        notion_integration.upload_speaking()

    scenario_data = comic_client.generate_scenarios()

    if tts_enabled:
        from . import tts
        tts.generate(scenario_data=scenario_data)
        tts.build_episode_reel()
        if notion:
            from .integrations import notion as notion_integration
            notion_integration.upload_tts_directory()

    if images:
        results = comic_client.render_images(scenario_data)
        if notion:
            comic_client.upload_and_attach(results)

    try:
        from .comic.lore_keeper import LoreKeeper
        LoreKeeper().run()
    except Exception as exc:
        print(f"  ⚠️ 쇼러너 메모 생성 실패: {exc}")

    print("\n" + "=" * 50)
    print("🎉 완료")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.DEFAULT_COUNT)
    parser.add_argument("--theme", default=config.DEFAULT_THEME)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-notion", action="store_true")
    args = parser.parse_args()

    try:
        run(
            n=args.n,
            theme=args.theme,
            images=not args.no_images,
            tts_enabled=not args.no_tts,
            notion=not args.no_notion,
        )
    except Exception as exc:
        print(f"\n❌ {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
