"""Render the same Hanyoil scene with LoRA blend choices.

This is a local comparison helper, not part of the main pipeline.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sd_generate_local


SEEDS = [6240, 6241, 6242]
OUTPUT_ROOT = Path("comic/output/hanyoil_lora_scene_test")


def _lora(path: Path, weight: float) -> dict:
    return {"path": str(path), "weight": weight}


def _comparison_canvas(items: list[tuple[str, int, Image.Image]]) -> Image.Image:
    if not items:
        raise ValueError("no images to compare")
    thumb_w = 300
    padding = 24
    label_h = 82
    thumbs = []
    for label, seed, img in items:
        scale = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * scale)), Image.LANCZOS)
        thumbs.append((label, seed, thumb))
    labels = list(dict.fromkeys(label for label, _, _ in thumbs))
    seed_values = list(dict.fromkeys(seed for _, seed, _ in thumbs))
    width = padding * (len(seed_values) + 1) + thumb_w * len(seed_values)
    row_h = label_h + max(t.height for _, _, t in thumbs)
    height = padding * (len(labels) + 1) + row_h * len(labels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    by_key = {(label, seed): thumb for label, seed, thumb in thumbs}
    for row, label in enumerate(labels):
        y = padding + row * row_h
        for col, seed in enumerate(seed_values):
            x = padding + col * (thumb_w + padding)
            draw.text((x, y), f"{label}\nseed {seed}", fill=(0, 0, 0), font=font)
            canvas.paste(by_key[(label, seed)], (x, y + label_h))
    return canvas


def main() -> None:
    sd_generate_local._apply_local_ssl_patch()
    model_root = sd_generate_local._default_model_root().resolve()
    prompts = sd_generate_local.load_patched_prompts(model_root)
    sd = sd_generate_local.import_sd_generate()
    gen = sd_generate_local.setup_generator(sd, prompts, model_root)

    lora_dir = model_root / "lora"
    combos = [
        (
            "hanyoil 0.8 + hanyoil_2 0.3",
            [
                _lora(lora_dir / "hanyoil_lora.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.3),
            ],
        ),
        (
            "hanyoil_2 0.8 + hanyoil_3 0.3",
            [
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_3.safetensors", 0.3),
            ],
        ),
        (
            "hanyoil_3 0.8 + hanyoil_2 0.3",
            [
                _lora(lora_dir / "hanyoil_lora_3.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.3),
            ],
        ),
        (
            "hanyoil_3 0.8 + hanyoil_2 0.5",
            [
                _lora(lora_dir / "hanyoil_lora_3.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.5),
            ],
        ),
    ]

    panel = {
        "panel_type": "character",
        "char": "hanyoil",
        "outfit": "daily_convenience_5",
        "action": "standing",
        "body_pose": "standing",
        "gesture": "none",
        "subject": "",
        "expression": "frown face",
        "face_state": "looking at viewer",
        "background": "convenience store, snack pile",
        "location": "convenience_store",
        "used_in": "daily",
        "target_sentence": "",
        "bubble": "",
        "bubble_kr": "",
        "seed_offset": 0,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, int, Image.Image]] = []
    for index, (label, loras) in enumerate(combos, 1):
        prompts.CHARS["hanyoil"]["lora"] = loras
        safe_label = f"blend_{index:02d}"
        for seed in SEEDS:
            panels, _current_lora, _variants = gen._render_panels([panel], seed, "")
            image = panels[0]
            out_path = OUTPUT_ROOT / f"{safe_label}_seed_{seed}.png"
            image.save(out_path)
            rendered.append((label, seed, image))
            print(f"saved: {out_path.resolve()}")

    preview = _comparison_canvas(rendered)
    preview_path = OUTPUT_ROOT / "_comparison.png"
    preview.save(preview_path)
    print(f"preview: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
