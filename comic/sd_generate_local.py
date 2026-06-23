"""Local SDXL comic runner.

This reuses the rendering logic in ``sd_generate.py`` without requiring Modal.

Examples:
  python sd_generate_local.py --comic
  python sd_generate_local.py --comic --seed 42 --model-root C:/path/to/models
  python sd_generate_local.py --batch-json-path data/state/scenario_data-26.06.22.json

Expected local model layout:
  <model-root>/WAI-illustrious-SDXL_17.safetensors
  <model-root>/lora/*.safetensors
  <model-root>/detect/face_yolov8n.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


COMIC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = COMIC_DIR / "output" / "local_sd"
DEFAULT_PLANNER_MODEL_ROOT = (
    COMIC_DIR.parents[2]
    / "notion_planner"
    / "notion_words"
    / "comic"
    / "models"
)


def _apply_local_ssl_patch() -> None:
    """Reuse the local SSL workaround from notion_planner when available."""
    candidates = [
        COMIC_DIR.parents[2] / "notion_planner",
        Path(os.getenv("NOTION_PLANNER_ROOT", "")) if os.getenv("NOTION_PLANNER_ROOT") else None,
    ]
    for root in candidates:
        if not root:
            continue
        patch = root / "src" / "ssl_patch.py"
        if not patch.exists():
            continue
        sys.path.insert(0, str(root))
        try:
            import src.ssl_patch  # noqa: F401
            print(f"SSL patch applied: {patch}")
        except Exception as exc:
            print(f"SSL patch import failed ({patch}): {exc}")
        return


def _install_modal_stub() -> None:
    """Make sd_generate.py importable when Modal is not installed."""
    if "modal" in sys.modules:
        return

    class _Image:
        @classmethod
        def debian_slim(cls, *args, **kwargs):
            return cls()

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def add_local_file(self, *args, **kwargs):
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

    class _Volume:
        @classmethod
        def from_name(cls, *args, **kwargs):
            return cls()

        def commit(self):
            return None

    class _App:
        def __init__(self, *args, **kwargs):
            pass

        def cls(self, *args, **kwargs):
            def deco(obj):
                return obj

            return deco

        def local_entrypoint(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

    def _identity_decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def deco(obj):
            return obj

        return deco

    stub = types.ModuleType("modal")
    stub.Image = _Image
    stub.Volume = _Volume
    stub.App = _App
    stub.enter = _identity_decorator
    stub.method = _identity_decorator
    sys.modules["modal"] = stub


def _load_module(name: str, path: Path):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _candidate_model_roots() -> list[Path]:
    roots: list[Path] = []
    if os.getenv("COMIC_MODEL_ROOT"):
        roots.append(Path(os.environ["COMIC_MODEL_ROOT"]))
    roots.append(DEFAULT_PLANNER_MODEL_ROOT)
    roots.append(COMIC_DIR / "models")
    return roots


def _default_model_root() -> Path:
    for root in _candidate_model_roots():
        if (root / "WAI-illustrious-SDXL_17.safetensors").exists():
            return root
    return DEFAULT_PLANNER_MODEL_ROOT


def _resolve_lora_path(path_value: str, model_root: Path) -> str:
    raw = Path(str(path_value))
    lora_dir = model_root / "lora"
    direct = raw if raw.exists() else lora_dir / raw.name
    if direct.exists():
        return str(direct)

    aliases = {
        "hanyoil_01.safetensors": "hanyoil_lora.safetensors",
        "hanyoil_02.safetensors": "hanyoil_lora_2.safetensors",
        "hyo_jeong.safetensors": "hyo-jeong.safetensors",
    }
    alias = aliases.get(raw.name)
    if alias and (lora_dir / alias).exists():
        return str(lora_dir / alias)

    raise FileNotFoundError(
        f"LoRA not found for {path_value!r}. Looked under {lora_dir}"
    )


def _patch_lora_cfg(value, model_root: Path):
    if isinstance(value, str):
        return _resolve_lora_path(value, model_root)
    patched = []
    for item in value:
        if isinstance(item, str):
            patched.append(_resolve_lora_path(item, model_root))
        else:
            new_item = dict(item)
            new_item["path"] = _resolve_lora_path(new_item["path"], model_root)
            patched.append(new_item)
    return patched


def load_patched_prompts(model_root: Path, no_inpaint: bool = False):
    prompts = _load_module("prompts", COMIC_DIR / "prompts.py")
    prompts.BASE_MODEL = str(model_root / "WAI-illustrious-SDXL_17.safetensors")
    prompts.BUBBLE_DIR = str(COMIC_DIR / "textbubble")
    prompts.BUBBLE_FONT = str(COMIC_DIR / "Font" / "DXMSubtitlesM-KSCpc-EUC-H.ttf")
    if no_inpaint:
        prompts.INPAINT_STRENGTH = 0.0

    for char in prompts.CHARS.values():
        char["lora"] = _patch_lora_cfg(char["lora"], model_root)
    return prompts


def import_sd_generate():
    _install_modal_stub()
    return _load_module("sd_generate_modal_source", COMIC_DIR / "sd_generate.py")


def setup_generator(sd, prompts, model_root: Path, yolo_path: Path | None = None):
    import torch
    import torch.nn.functional as F
    from compel import Compel, ReturnedEmbeddingsType
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

    gen = sd.Generator()
    gen.P = prompts
    gen.torch = torch
    gen._bg_ip_adapter_loaded = False
    gen._bg_ip_adapter_failed = False
    gen._bg_ip_adapter_scale = 0.25
    gen.anime_seg_session = None
    gen.anime_seg_model = ""
    gen.seg_yolo = None
    gen.seg_yolo_name = ""

    print(f"Loading base model: {prompts.BASE_MODEL}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        prompts.BASE_MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    if not os.getenv("COMIC_SKIP_FP16_VAE"):
        print("Loading fp16-fix VAE: madebyollin/sdxl-vae-fp16-fix")
        pipe.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled")
    except Exception:
        pipe.enable_attention_slicing()
        print("attention slicing enabled")

    gen.pipe = pipe.to("cuda")
    gen.inpaint_pipe = StableDiffusionXLInpaintPipeline(**gen.pipe.components).to("cuda")
    gen.compel = Compel(
        tokenizer=[gen.pipe.tokenizer, gen.pipe.tokenizer_2],
        text_encoder=[gen.pipe.text_encoder, gen.pipe.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )
    def _pad_conditioning_tensors_to_same_length(tensors=None, conditionings=None):
        tensors = tensors if tensors is not None else conditionings
        target = max(t.shape[1] for t in tensors)
        out = []
        for tensor in tensors:
            if tensor.shape[1] < target:
                tensor = F.pad(tensor, (0, 0, 0, target - tensor.shape[1]))
            out.append(tensor)
        return out

    gen.compel.pad_conditioning_tensors_to_same_length = _pad_conditioning_tensors_to_same_length
    gen.compel_inpaint = gen.compel

    yolo_path = yolo_path or model_root / "detect" / "face_yolov8n.pt"
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO face model not found: {yolo_path}")
    gen.yolo = YOLO(str(yolo_path))
    print(f"YOLO loaded: {yolo_path}")
    return gen


def save_rendered_comic(sd, panels_cfg: list[dict], gen, seed: int, output_root: Path,
                        bg_plate: bool = False, bg_ip_adapter_scale: float = 0.25,
                        bg_base_lora: bool = False) -> Path:
    chars = "_".join(dict.fromkeys((p.get("char") or "object") for p in panels_cfg))
    out_dir = output_root / f"comic_{chars}_{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels, _current_lora, variants = gen._render_panels(
        panels_cfg,
        seed,
        "",
        use_bg_plate=bg_plate,
        bg_ip_adapter_scale=bg_ip_adapter_scale,
        bg_base_lora=bg_base_lora,
    )

    for index, panel in enumerate(panels, 1):
        path = out_dir / f"panel_{index:02d}.png"
        panel.save(path)
        print(f"  saved: {path}")

    strip = sd.stack_webtoon_strip(
        panels,
        gutter_color=sd._webtoon_gutter_color(panels_cfg[0] if panels_cfg else {}),
    )
    if strip is not None:
        strip_path = out_dir / "strip.png"
        strip.save(strip_path)
        print(f"  saved: {strip_path}")

    cover_item = sd._choose_cover_variant(panels_cfg, variants)
    if cover_item:
        cover = sd._square_face_crop(*cover_item)
        cover_path = out_dir / "cover.png"
        cover.save(cover_path)
        print(f"  saved: {cover_path}")
    return out_dir


def _load_batch_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("batch"), list):
            return data["batch"]
        if isinstance(data.get("items"), list):
            return data["items"]
    raise ValueError(f"Unsupported batch JSON shape: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sd_generate.py locally without Modal.")
    parser.add_argument("--comic", action="store_true", help="Render prompts.COMIC_PANELS.")
    parser.add_argument("--batch-json-path", type=Path, help="Render batch items with panels/seed.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many batch items before rendering.")
    parser.add_argument("--limit", type=int, default=0, help="Limit batch item count.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed; default random when negative.")
    parser.add_argument("--model-root", type=Path, default=_default_model_root())
    parser.add_argument("--yolo-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-inpaint", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Validate local paths without loading SDXL.")
    parser.add_argument("--bg-plate", action="store_true")
    parser.add_argument("--bg-ip-adapter-scale", type=float, default=0.25)
    parser.add_argument("--bg-base-lora", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _apply_local_ssl_patch()
    args = parse_args()
    seed = args.seed if args.seed >= 0 else random.randint(0, 2**32 - 1)
    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    prompts = load_patched_prompts(model_root, no_inpaint=args.no_inpaint)
    if args.check_only:
        required = [
            Path(prompts.BASE_MODEL),
            Path(prompts.BUBBLE_DIR),
            Path(prompts.BUBBLE_FONT),
            args.yolo_path or model_root / "detect" / "face_yolov8n.pt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        for cname, cdata in prompts.CHARS.items():
            specs = cdata["lora"] if isinstance(cdata["lora"], list) else [cdata["lora"]]
            for spec in specs:
                path = Path(spec["path"] if isinstance(spec, dict) else spec)
                if not path.exists():
                    missing.append(f"{cname}: {path}")
        if missing:
            print("Missing local assets:")
            for item in missing:
                print(f"  - {item}")
            raise SystemExit(1)
        print("Local SD assets OK")
        print(f"  model_root: {model_root}")
        print(f"  output_root: {output_root}")
        print(f"  characters: {', '.join(prompts.CHARS)}")
        return

    sd = import_sd_generate()
    gen = setup_generator(sd, prompts, model_root, yolo_path=args.yolo_path)

    if args.batch_json_path:
        items = _load_batch_items(args.batch_json_path)
        if args.offset > 0:
            items = items[args.offset :]
        if args.limit > 0:
            items = items[: args.limit]
        for index, item in enumerate(items, 1):
            panels_cfg = item["panels"]
            item_seed = int(item.get("seed") or seed + index - 1)
            print(f"\n=== [{index}/{len(items)}] seed={item_seed} panels={len(panels_cfg)} ===")
            save_rendered_comic(
                sd,
                panels_cfg,
                gen,
                item_seed,
                output_root,
                bg_plate=args.bg_plate,
                bg_ip_adapter_scale=args.bg_ip_adapter_scale,
                bg_base_lora=args.bg_base_lora,
            )
        return

    if not args.comic:
        args.comic = True

    print(f"\n=== local comic | seed={seed} | panels={len(prompts.COMIC_PANELS)} ===")
    save_rendered_comic(
        sd,
        prompts.COMIC_PANELS,
        gen,
        seed,
        output_root,
        bg_plate=args.bg_plate,
        bg_ip_adapter_scale=args.bg_ip_adapter_scale,
        bg_base_lora=args.bg_base_lora,
    )


if __name__ == "__main__":
    main()
