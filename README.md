# Notion Speaking

English speaking sentence generation plus short webtoon-style scenario, TTS,
image, and Notion upload pipeline.

## Quick Start

Run commands from this repository root:

```powershell
cd C:\Users\user\J_0\english\notion-speaking
python -m notion_speaking.run_stage --help
```

Generate only speaking sentence CSV:

```powershell
python -m notion_speaking.run_stage sentences --n 10 --theme "daily situations I can actually say out loud"
```

Generate text-only scenarios from the latest structured CSV:

```powershell
python -m notion_speaking.run_stage scenarios
```

Skip GPU image rendering with:

```powershell
$env:MODAL_DISABLED = "1"
```

## Pipeline Stages

- `sentences`: generate short reusable speaking sentences into `data/sentences/<yy.mm>/`.
- `scenarios`: generate planner/script/visual metadata into `data/state/`.
- `notion`: upload the cleaned sentence CSV to Notion.
- `tts`: generate per-sentence dialogue audio from scenario data.
- `ttsup`: upload TTS files to Notion.
- `images`: render webtoon panels and attach them to Notion.
- `lore`: update continuity/lore notes from recent episodes.

## Environment

Expected `.env` values:

```dotenv
OPENAI_API_KEY=
ELEVENLABS_API_KEY=
NOTION_API_KEY=
NOTION_SPEAKING_DATABASE_ID=
```

Optional:

```dotenv
PIPELINE_DATE=26.06.22
SPEAKING_COUNT=20
SPEAKING_THEME=daily and workplace situations a Korean B1-B2 learner is likely to speak in
MODEL_PLAN=gpt-4o
MODEL_SCRIPT=gpt-4o
MODEL_VISUAL=gpt-4o-mini
MODEL_SELECT=gpt-4o-mini
MODAL_DISABLED=1
VERIFY_ENABLED=1
```

## Prompting Notes

The main prompt surfaces are:

- `prompt.py`: sentence-unit generation.
- `comic/scenario_prompts.py`: story planner, dialogue writer, visual tags, review card.
- `lore/episode_rules.md`: global scene rules.
- `lore/characters.md`: character voice and facets.
- `lore/domains/*.md`: allowed locations and domain-specific cast.

The sentence prompt is tuned for high-frequency spoken building blocks first:
phrases people actually say all the time, reusable across daily/workplace
contexts.

The scenario prompts are tuned for speaking memory: tiny social pressure, one
clean comic turn, plain repeatable English, and a final button that makes the
target sentence easier to recall.
