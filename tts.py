"""TTS generation for speaking sentence webtoon episodes."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import openai
import pandas as pd
from elevenlabs.client import ElevenLabs
from elevenlabs.types import VoiceSettings

from . import config
from .comic.prompts import CHARACTER_VOICES

if not config.ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY 가 설정되지 않았습니다 — .env 를 확인하세요.")

_openai = openai.OpenAI(api_key=config.OPENAI_API_KEY)
_eleven = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)

ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
ELEVENLABS_EMOTION_TAGS = os.getenv("ELEVENLABS_EMOTION_TAGS", "1") != "0"
SPEEDS = {"slow": 0.9, "normal": 1.0, "fast": 1.2}
PANEL_GAP_MS = 800
EPISODE_GAP_MS = 1200
_VOICE_SETTINGS = VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.0, use_speaker_boost=True)

_CHAR_CANONICAL: dict[str, str] = {
    "hanyoil": "hanyoil",
    "ru-ha": "ru-ha", "ruha": "ru-ha", "ru_ha": "ru-ha", "ru ha": "ru-ha",
    "so-ae": "so-ae", "soae": "so-ae", "so_ae": "so-ae", "so ae": "so-ae",
    "hanyuyeon": "hanyuyeon", "han yuyeon": "hanyuyeon", "han_yuyeon": "hanyuyeon",
    "hyo-jeong": "hyo-jeong", "hyojeong": "hyo-jeong", "hyo_jeong": "hyo-jeong", "hyo jeong": "hyo-jeong",
}


def _supports_dialogue_api() -> bool:
    model = (ELEVENLABS_MODEL or "").lower()
    return model in {"eleven_v3", "eleven_multilingual_v3"} or model.endswith("_v3")


def _emotion_tag(expression: str = "", tone: str = "") -> str:
    e = f"{expression or ''} {tone or ''}".lower()
    rules = [
        (("laugh", "giggl"), "[laughs]"),
        (("excited", "thrill", "grin", "eager", "delighted"), "[excited]"),
        (("embarrass", "blush", "shy", "flustered", "nervous", "worried", "anxious", "panic", "timid"), "[nervous]"),
        (("sigh", "resigned", "exhaust", "tired", "reluctant", "defeated"), "[sighs]"),
        (("sad", "teary", "gloomy", "melancholy", "hurt", "downcast"), "[sad]"),
        (("angry", "annoyed", "irritat", "frustrat", "grumpy", "scowl"), "[annoyed]"),
        (("sarcastic", "smug", "dry", "deadpan", "unimpressed", "flat", "expressionless", "blank"), "[sarcastic]"),
        (("surprised", "shock", "startled", "wide-eyed", "astonished"), "[surprised]"),
        (("whisper", "murmur", "hushed"), "[whispers]"),
    ]
    for keys, tag in rules:
        if any(k in e for k in keys):
            return tag + " "
    return ""


def _canonical_char(name: str) -> str:
    key = (name or "").lower().strip()
    if not key:
        return "_default"
    return _CHAR_CANONICAL.get(key, key)


def _make_silence(duration_ms: int, out: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", str(duration_ms / 1000),
            "-q:a", "9", "-acodec", "libmp3lame", str(out),
        ],
        check=True,
        capture_output=True,
    )


def _ffmpeg_concat(files: list[Path], out: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(f"file '{p}'" for p in files))
        list_path = Path(f.name)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-q:a", "4", str(out)],
        check=True,
        capture_output=True,
    )
    list_path.unlink()


def _tts_eleven(
    text: str,
    voice_id: str,
    out: Path,
    previous_text: str | None = None,
    next_text: str | None = None,
) -> bool:
    try:
        kwargs = {
            "voice_id": voice_id,
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "output_format": "mp3_44100_128",
        }
        if not _supports_dialogue_api():
            kwargs.update(
                voice_settings=_VOICE_SETTINGS,
                previous_text=previous_text,
                next_text=next_text,
            )
        audio = _eleven.text_to_speech.convert(**kwargs)
        out.write_bytes(b"".join(audio))
        return True
    except Exception as exc:
        if getattr(exc, "status_code", None) == 401:
            raise
        print(f"  ⚠️ ElevenLabs TTS error {out.name}: {exc}")
        return False


def _tts_openai(text: str, speed: float, out: Path) -> bool:
    try:
        resp = _openai.audio.speech.create(model="tts-1", voice="alloy", speed=speed, input=text)
        resp.stream_to_file(str(out))
        return True
    except openai.BadRequestError as exc:
        print(f"  ⚠️ OpenAI TTS error {out.name}: {exc}")
        return False


def generate_dialogue_mp3(panels: list[dict], out: Path) -> bool:
    lines: list[tuple[str, str, str, str]] = []
    for panel in panels:
        bubble = str(panel.get("bubble", "")).strip()
        if not bubble:
            continue
        char = _canonical_char(str(panel.get("char", "_default")))
        voice = CHARACTER_VOICES.get(char, CHARACTER_VOICES["_default"])
        if char not in CHARACTER_VOICES:
            print(f"    ⚠️ 알 수 없는 캐릭터 '{panel.get('char')}' → _default 목소리 사용")
        lines.append((char, voice, bubble, str(panel.get("expression", ""))))

    if not lines:
        return False

    if _supports_dialogue_api():
        inputs = []
        for char, voice, bubble, expr in lines:
            tag = _emotion_tag(expr) if ELEVENLABS_EMOTION_TAGS and char != "_default" else ""
            inputs.append({"text": f"{tag}{bubble}", "voice_id": voice})
            print(f"    [{char}] {tag}{bubble[:40]}")
        try:
            audio = _eleven.text_to_dialogue.convert(inputs=inputs, model_id=ELEVENLABS_MODEL)
            out.write_bytes(b"".join(audio))
            return True
        except Exception as exc:
            print(f"  ⚠️ text_to_dialogue error {out.name}: {exc}")
            return False

    texts = [bubble for _, _, bubble, _ in lines]
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        gap = tdir / "gap.mp3"
        _make_silence(PANEL_GAP_MS, gap)
        parts: list[Path] = []
        for i, (char, voice, bubble, _expr) in enumerate(lines):
            seg = tdir / f"panel_{i:02d}.mp3"
            prev_t = texts[i - 1] if i > 0 else None
            next_t = texts[i + 1] if i + 1 < len(texts) else None
            if not _tts_eleven(bubble, voice, seg, previous_text=prev_t, next_text=next_t):
                return False
            print(f"    [{char}] {bubble[:40]}")
            parts.append(seg)
            if i < len(lines) - 1:
                parts.append(gap)
        _ffmpeg_concat(parts, out)
    return True


def build_episode_reel(out: Path | None = None) -> Path | None:
    mp3s = sorted(
        (f for f in config.TTS_DAY_DIR.glob("*.mp3") if f.stem.isdigit()),
        key=lambda p: int(p.stem),
    )
    if not mp3s:
        print("  ⚠️ 스피킹 에피소드 mp3 없음 — reel 생략")
        return None

    out = out or config.TTS_DAY_DIR / f"{config.YY_MM_DD}_speaking_episodes.mp3"
    with tempfile.TemporaryDirectory() as tmp:
        gap = Path(tmp) / "gap.mp3"
        _make_silence(EPISODE_GAP_MS, gap)
        seq: list[Path] = []
        for i, mp3 in enumerate(mp3s):
            seq.append(mp3)
            if i < len(mp3s) - 1:
                seq.append(gap)
        _ffmpeg_concat(seq, out)
    print(f"  🎞️ 스피킹 에피소드 합본 {len(mp3s)}화 → {out.name}")
    return out


def generate(scenario_data: list[dict] | None = None, char_map: dict[str, str] | None = None) -> None:
    df = pd.read_csv(config.CLEAN_CSV, encoding="utf-8-sig").fillna("")
    aac_dir = config.TTS_DAY_DIR / "temp_aac"
    if config.BUILD_AAC:
        aac_dir.mkdir(exist_ok=True)

    panels_map = {
        str(d.get("word_no", "")): d.get("panels", [])
        for d in (scenario_data or [])
    }

    for _, row in df.iterrows():
        no = str(row["No."]).strip()
        sentence = str(row.get("sentence_unit") or row.get("sentence unit", "")).strip()
        if not sentence:
            continue

        out_mp3 = config.TTS_DAY_DIR / f"{no}.mp3"
        panels = panels_map.get(no)
        if panels:
            print(f"  🎭 SPEAKING #{no} 대화형 TTS ({len(panels)}컷)...")
            ok = generate_dialogue_mp3(panels, out_mp3)
            if not ok:
                print("    ⚠️ 대화형 실패 → sentence unit 단일 fallback")
                _tts_eleven(sentence, CHARACTER_VOICES["_default"], out_mp3)
        else:
            char = (char_map or {}).get(no, "_default").lower().strip()
            voice = CHARACTER_VOICES.get(char, CHARACTER_VOICES["_default"])
            print(f"  🎤 SPEAKING #{no} [{char}] {sentence[:40]}...")
            _tts_eleven(sentence, voice, out_mp3)

        if config.BUILD_AAC:
            for name, speed in SPEEDS.items():
                _tts_openai(sentence, speed, aac_dir / f"{no}_{name}.mp3")
        time.sleep(0.3)

    print("✅ SPEAKING TTS 완료")
