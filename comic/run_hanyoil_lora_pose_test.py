"""Render narrowed Hanyoil LoRA blends with one outfit and varied compositions."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import sd_generate_local


OUTPUT_ROOT = Path("comic/output/hanyoil_lora_pose_test")


def _lora(path: Path, weight: float) -> dict:
    return {"path": str(path), "weight": weight}


def _comparison_canvas(items: list[tuple[str, str, Image.Image]]) -> Image.Image:
    if not items:
        raise ValueError("no images to compare")
    thumb_w = 320
    padding = 24
    label_h = 86
    thumbs = []
    for label, variant, img in items:
        scale = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * scale)), Image.LANCZOS)
        thumbs.append((label, variant, thumb))
    labels = list(dict.fromkeys(label for label, _, _ in thumbs))
    variants = list(dict.fromkeys(variant for _, variant, _ in thumbs))
    width = padding * (len(variants) + 1) + thumb_w * len(variants)
    row_h = label_h + max(t.height for _, _, t in thumbs)
    height = padding * (len(labels) + 1) + row_h * len(labels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        font = ImageFont.load_default()
    by_key = {(label, variant): thumb for label, variant, thumb in thumbs}
    for row, label in enumerate(labels):
        y = padding + row * row_h
        for col, variant in enumerate(variants):
            x = padding + col * (thumb_w + padding)
            draw.text((x, y), f"{label}\n{variant}", fill=(0, 0, 0), font=font)
            canvas.paste(by_key[(label, variant)], (x, y + label_h))
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
            "hanyoil_2 0.8 + hanyoil_3 0.3",
            [
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_3.safetensors", 0.3),
            ],
        ),
        (
            "hanyoil_2 0.8 + hanyoil 0.3",
            [
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora.safetensors", 0.3),
            ],
        ),
        (
            "hanyoil_3 0.8 + hanyoil_2 0.3",
            [
                _lora(lora_dir / "hanyoil_lora_3.safetensors", 0.8),
                _lora(lora_dir / "hanyoil_lora_2.safetensors", 0.3),
            ],
        ),
    ]

    variants = [
        (
            "front upper body",
            6440,
            {
                "action": "standing, hand on hip",
                "gesture": "hands_on_hips",
                "expression": "serious face",
                "face_state": "looking at viewer",
                "framing": "upper_body",
                "background": "city street, storefront",
            },
        ),
        (
            "side gaze waist shot",
            6441,
            {
                "action": "standing, hand on chest",
                "gesture": "hand_on_chest",
                "expression": "light smile face",
                "face_state": "looking to the side",
                "framing": "waist_shot",
                "background": "cafe, window",
            },
        ),
        (
            "reaction close-up",
            6442,
            {
                "action": "standing",
                "gesture": "none",
                "expression": "frown face",
                "face_state": "looking at viewer",
                "framing": "close_up",
                "background": "apartment, living room",
            },
        ),
    ]

    base_panel = {
        "panel_type": "character",
        "char": "hanyoil",
        "outfit": "daily_outing_1",
        "body_pose": "standing",
        "subject": "",
        "location": "pose_test",
        "used_in": "daily",
        "target_sentence": "",
        "bubble": "",
        "bubble_kr": "",
        "seed_offset": 0,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, str, Image.Image]] = []
    for index, (label, loras) in enumerate(combos, 1):
        prompts.CHARS["hanyoil"]["lora"] = loras
        safe_label = f"blend_{index:02d}"
        for variant_label, seed, fields in variants:
            panel = {**base_panel, **fields}
            panels, _current_lora, _variants = gen._render_panels([panel], seed, "")
            image = panels[0]
            safe_variant = "_".join(variant_label.lower().split())
            out_path = OUTPUT_ROOT / f"{safe_label}_{safe_variant}.png"
            image.save(out_path)
            rendered.append((label, variant_label, image))
            print(f"saved: {out_path.resolve()}")

    preview = _comparison_canvas(rendered)
    preview_path = OUTPUT_ROOT / "_comparison.png"
    preview.save(preview_path)
    print(f"preview: {preview_path.resolve()}")


if __name__ == "__main__":
    main()
