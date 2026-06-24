from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8")

COMIC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = COMIC_DIR.parents[1]
PROMPTS_PATH = COMIC_DIR / "prompts.py"
DEFAULT_OUTPUT_ROOT = COMIC_DIR / "output" / "regional_cafe_test"
DEFAULT_YOLO_PATH = COMIC_DIR / "models" / "detect" / "face_yolov8n.pt"
DEFAULT_WIDTH = 1216
DEFAULT_HEIGHT = 832
LORA_DIR = COMIC_DIR / "models" / "lora"

HANYOIL_BLEND_LORAS = (
    (LORA_DIR / "hanyoil_lora_2.safetensors", 0.3),
    (LORA_DIR / "hanyoil_lora_3.safetensors", 0.8),
)
HYOJEONG_LORA = LORA_DIR / "hyo-jeong.safetensors"

# Apply the local SSL patch before diffusers/huggingface_hub need network access.
sys.path.insert(0, str(PROJECT_DIR))
try:
    import src.ssl_patch  # noqa: F401,E402
except Exception:
    pass


def load_runtime_dependencies() -> None:
    global Compel
    global DPMSolverMultistepScheduler
    global F
    global Image
    global np
    global ReturnedEmbeddingsType
    global StableDiffusionXLInpaintPipeline
    global StableDiffusionXLPipeline
    global torch
    global YOLO

    import numpy as _np
    import torch as _torch
    import torch.nn.functional as _F
    from compel import Compel as _Compel
    from compel import ReturnedEmbeddingsType as _ReturnedEmbeddingsType
    from diffusers import DPMSolverMultistepScheduler as _DPMSolverMultistepScheduler
    from diffusers import StableDiffusionXLInpaintPipeline as _StableDiffusionXLInpaintPipeline
    from diffusers import StableDiffusionXLPipeline as _StableDiffusionXLPipeline
    from PIL import Image as _Image
    from ultralytics import YOLO as _YOLO

    Compel = _Compel
    DPMSolverMultistepScheduler = _DPMSolverMultistepScheduler
    F = _F
    Image = _Image
    np = _np
    ReturnedEmbeddingsType = _ReturnedEmbeddingsType
    StableDiffusionXLInpaintPipeline = _StableDiffusionXLInpaintPipeline
    StableDiffusionXLPipeline = _StableDiffusionXLPipeline
    torch = _torch
    YOLO = _YOLO


def load_prompts_module():
    sys.modules.pop("prompts", None)
    spec = importlib.util.spec_from_file_location("prompts", PROMPTS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prompts.py: {PROMPTS_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["prompts"] = module
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SDXL regional-mask cafe scene with Hanyoil and Hyo-jeong."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Defaults to a random uint32.")
    parser.add_argument("--width", type=int, default=None, help=f"Image width. Defaults to {DEFAULT_WIDTH}.")
    parser.add_argument("--height", type=int, default=None, help=f"Image height. Defaults to {DEFAULT_HEIGHT}.")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps. Defaults to prompts.py STEPS.")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale. Defaults to prompts.py CFG_SCALE.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory.")
    parser.add_argument("--left-strength", type=float, default=1.0, help="Left region prompt strength.")
    parser.add_argument("--right-strength", type=float, default=1.0, help="Right region prompt strength.")
    parser.add_argument("--base-strength", type=float, default=0.35, help="Base cafe prompt strength.")
    parser.add_argument("--mask-feather", type=int, default=96, help="Mask feather width in image pixels.")
    parser.add_argument("--no-inpaint", action="store_true", help="Disable per-character face inpainting.")
    parser.add_argument("--yolo-path", type=Path, default=DEFAULT_YOLO_PATH, help="Face YOLO model path.")
    parser.add_argument(
        "--scene",
        choices=("conversation", "coffee_chaos"),
        default="conversation",
        help="Regional scene preset.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def pad_cond(cond: torch.Tensor, neg_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cond.shape[1] == neg_cond.shape[1]:
        return cond, neg_cond

    target = max(cond.shape[1], neg_cond.shape[1])

    def zpad(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] >= target:
            return tensor
        return F.pad(tensor, (0, 0, 0, target - tensor.shape[1]))

    return zpad(cond), zpad(neg_cond)


def soft_box_mask(
    width: int,
    height: int,
    box: tuple[float, float, float, float],
    feather: int,
    device,
    dtype,
) -> torch.Tensor:
    x1, y1, x2, y2 = box
    xs = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    ys = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    px1, py1, px2, py2 = x1 * width, y1 * height, x2 * width, y2 * height

    inside_x = torch.minimum(xs - px1, px2 - xs)
    inside_y = torch.minimum(ys - py1, py2 - ys)
    signed_distance = torch.minimum(inside_x, inside_y)
    mask = ((signed_distance + feather) / max(feather, 1)).clamp(0, 1)
    return mask[None, None].to(dtype=dtype)


def downsample_mask(mask: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    return F.interpolate(mask, size=(latent_h, latent_w), mode="bilinear", align_corners=False).clamp(0, 1)


def vertical_split_masks(
    width: int,
    height: int,
    split: float,
    feather: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(width, device=device, dtype=torch.float32)[None, :]
    feather = max(feather, 1)
    center = split * width
    left = ((center + feather - xs) / (2 * feather)).clamp(0, 1)
    left = left.expand(height, width)
    right = 1.0 - left
    return left[None, None].to(dtype=dtype), right[None, None].to(dtype=dtype)


@dataclass(frozen=True)
class Region:
    name: str
    prompt: str
    adapter_names: tuple[str, ...]
    adapter_weights: tuple[float, ...]
    mask: torch.Tensor
    strength: float


class RegionalCafeGenerator:
    def __init__(self, prompts_module, yolo_path: Path) -> None:
        self.P = prompts_module
        self.current_adapters: tuple[tuple[str, ...], tuple[float, ...]] | None = None

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")

        require_file(Path(self.P.BASE_MODEL), "Base model")
        for path, _weight in HANYOIL_BLEND_LORAS:
            require_file(path, "Hanyoil LoRA")
        require_file(HYOJEONG_LORA, "Hyo-jeong LoRA")
        require_file(yolo_path, "YOLO face model")

        print(f"Loading base model: {self.P.BASE_MODEL}")
        self.pipe = StableDiffusionXLPipeline.from_single_file(
            self.P.BASE_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xformers enabled")
        except Exception:
            self.pipe.enable_attention_slicing()
            print("attention slicing enabled")

        self.pipe = self.pipe.to("cuda")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(**self.pipe.components).to("cuda")
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        self.load_character_loras()
        print("Model loaded")
        self.yolo = YOLO(str(yolo_path))
        print(f"YOLO loaded: {yolo_path}")

    def load_character_loras(self) -> None:
        try:
            self.pipe.unfuse_lora()
            self.pipe.unload_lora_weights()
        except Exception:
            pass

        for index, (path, _weight) in enumerate(HANYOIL_BLEND_LORAS):
            self.pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=f"hanyoil_{index}",
                low_cpu_mem_usage=False,
            )

        self.pipe.load_lora_weights(
            str(HYOJEONG_LORA.parent),
            weight_name=HYOJEONG_LORA.name,
            adapter_name="hyojeong",
            low_cpu_mem_usage=False,
        )
        print("LoRAs loaded: hanyoil blend + hyo-jeong")

    def set_adapters(self, names: tuple[str, ...], weights: tuple[float, ...]) -> None:
        key = (names, weights)
        if self.current_adapters == key:
            return
        self.pipe.set_adapters(list(names), adapter_weights=list(weights))
        self.current_adapters = key

    def encode_prompt(self, prompt: str, negative_prompt: str):
        cond, pooled = self.compel(prompt)
        neg_cond, neg_pooled = self.compel(negative_prompt)
        cond, neg_cond = pad_cond(cond, neg_cond)
        return cond, pooled, neg_cond, neg_pooled

    def feather_mask(self, width: int, height: int) -> Image.Image:
        feather = max(8, getattr(self.P, "INPAINT_PADDING", 40) // 2)
        blend = np.full((height, width), 255, dtype=np.float32)
        for edge in range(feather):
            value = int(255 * (edge / feather))
            blend[edge, :] = value
            blend[-(edge + 1), :] = value
            blend[:, edge] = value
            blend[:, -(edge + 1)] = value
        return Image.fromarray(blend.astype(np.uint8))

    def inpaint_faces(self, image: Image.Image, seed: int, single_character: str | None = None) -> Image.Image:
        result = self.yolo(np.array(image), conf=0.3, verbose=False)
        faces: list[tuple[int, int, int, int]] = []
        width, height = image.size
        padding = getattr(self.P, "INPAINT_PADDING", 40)

        for item in result:
            for box in item.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                faces.append(
                    (
                        max(0, x1 - padding),
                        max(0, y1 - padding),
                        min(width, x2 + padding),
                        min(height, y2 + padding),
                    )
                )

        if not faces:
            print("no face detected; keeping original")
            return image

        faces.sort(key=lambda item: (item[0] + item[2]) / 2)
        refined = image.copy()
        crop_size = getattr(self.P, "CROP_SIZE", 1024)
        inpaint_negative = getattr(self.P, "INPAINT_NEGATIVE", "")
        inpaint_steps = getattr(self.P, "INPAINT_STEPS", 30)
        inpaint_cfg = getattr(self.P, "INPAINT_CFG", 7.0)
        inpaint_strength = getattr(self.P, "INPAINT_STRENGTH", 0.6)

        hanyoil_prompt = (
            "hanyoil, black-haired woman, black hair, dark gray eyes, smile, "
            "1girl, masterpiece, best quality, detailed eyes, highly detailed face"
        )
        hyojeong_prompt = (
            "hyo-jeong, red hair, low side ponytail, black eyes, gentle smile, "
            "1girl, masterpiece, best quality, detailed eyes, highly detailed face"
        )

        for index, (x1, y1, x2, y2) in enumerate(faces):
            center_x = (x1 + x2) / 2
            is_left_character = (
                single_character == "hanyoil"
                or (index == 0 if len(faces) >= 2 else center_x < width / 2)
            )
            if is_left_character:
                label = "hanyoil"
                adapter_names = tuple(f"hanyoil_{i}" for i in range(len(HANYOIL_BLEND_LORAS)))
                adapter_weights = tuple(weight for _path, weight in HANYOIL_BLEND_LORAS)
                inpaint_prompt = hanyoil_prompt
            else:
                label = "hyojeong"
                adapter_names = ("hyojeong",)
                adapter_weights = (1.0,)
                inpaint_prompt = hyojeong_prompt

            print(f"  inpaint face {index + 1}/{len(faces)} as {label}")
            self.set_adapters(adapter_names, adapter_weights)

            face_width = x2 - x1
            face_height = y2 - y1
            face_up = refined.crop((x1, y1, x2, y2)).resize(
                (crop_size, crop_size),
                Image.LANCZOS,
            )
            mask_up = Image.new("L", (crop_size, crop_size), 255)
            cond, pooled, neg_cond, neg_pooled = self.encode_prompt(inpaint_prompt, inpaint_negative)
            generator = torch.Generator(device="cuda").manual_seed(seed + index)
            inpainted = self.inpaint_pipe(
                prompt_embeds=cond,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_cond,
                negative_pooled_prompt_embeds=neg_pooled,
                image=face_up,
                mask_image=mask_up,
                width=crop_size,
                height=crop_size,
                num_inference_steps=inpaint_steps,
                guidance_scale=inpaint_cfg,
                strength=inpaint_strength,
                generator=generator,
            ).images[0]

            down = inpainted.resize((face_width, face_height), Image.LANCZOS)
            mask = self.feather_mask(face_width, face_height)
            refined.paste(down, (x1, y1), mask)

        return refined

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_bundle,
        add_time_ids: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        cond, pooled, neg_cond, neg_pooled = prompt_bundle
        latent_input = torch.cat([latents, latents])
        latent_input = self.pipe.scheduler.scale_model_input(latent_input, timestep)
        encoder_states = torch.cat([neg_cond, cond])
        text_embeds = torch.cat([neg_pooled, pooled])
        time_ids = torch.cat([add_time_ids, add_time_ids])

        noise_pred = self.pipe.unet(
            latent_input,
            timestep,
            encoder_hidden_states=encoder_states,
            added_cond_kwargs={
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
            return_dict=False,
        )[0]
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_text - noise_uncond)

    def decode_latents(self, latents: torch.Tensor):
        needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
        if needs_upcasting:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        image = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return self.pipe.image_processor.postprocess(image.detach(), output_type="pil")

    def generate(
        self,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        left_strength: float,
        right_strength: float,
        base_strength: float,
        mask_feather: int,
        scene: str,
    ):
        device = self.pipe._execution_device
        dtype = self.pipe.unet.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        self.pipe.scheduler.set_timesteps(steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        num_channels_latents = self.pipe.unet.config.in_channels
        latents = self.pipe.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            None,
        )
        latent_h, latent_w = latents.shape[-2:]

        base_mask = torch.ones((1, 1, latent_h, latent_w), device=device, dtype=dtype)

        negative_prompt = getattr(self.P, "NEGATIVE_PROMPT", "")

        hanyoil_names = tuple(f"hanyoil_{i}" for i in range(len(HANYOIL_BLEND_LORAS)))
        hanyoil_weights = tuple(weight for _path, weight in HANYOIL_BLEND_LORAS)
        base_names = hanyoil_names + ("hyojeong",)
        base_weights = (0.15, 0.25, 0.25)

        if scene == "coffee_chaos":
            person_mask = downsample_mask(
                soft_box_mask(width, height, (0.02, 0.12, 0.58, 1.0), mask_feather, device, dtype),
                latent_h,
                latent_w,
            )
            machine_mask = downsample_mask(
                soft_box_mask(width, height, (0.45, 0.10, 0.98, 0.78), mask_feather, device, dtype),
                latent_h,
                latent_w,
            )
            spray_mask = downsample_mask(
                soft_box_mask(width, height, (0.28, 0.05, 0.95, 0.72), mask_feather, device, dtype),
                latent_h,
                latent_w,
            )
            cafe_prompt = (
                "cozy modern cafe interior, coffee counter, espresso machine, warm lighting, "
                "chaotic accident scene, dynamic anime illustration, masterpiece, best quality, high detail"
            )
            hanyoil_tags = (
                "hanyoil, black-haired woman, 1girl, black hair, dark gray eyes, slim body, light skin, "
                "standing on the left side, cowboy shot, startled, flustered, panicking expression, "
                "wide eyes, open mouth, recoiling backward, raised hands, reacting to coffee spraying everywhere"
            )
            machine_tags = (
                "large espresso coffee machine on the right side, broken coffee machine, "
                "malfunctioning, pressure burst, coffee machine shooting coffee everywhere, "
                "countertop, cups knocked over, chaotic cafe counter"
            )
            spray_tags = (
                "brown coffee spray blasting across the scene, arcing streams of coffee, "
                "splash droplets everywhere, liquid motion, explosive messy splash, "
                "coffee flying from right to left toward the panicking woman"
            )
            base_bundle = self.encode_prompt(cafe_prompt, negative_prompt)
            hanyoil_bundle = self.encode_prompt(f"{cafe_prompt}, BREAK, {hanyoil_tags}", negative_prompt)
            machine_bundle = self.encode_prompt(f"{cafe_prompt}, BREAK, {machine_tags}", negative_prompt)
            spray_bundle = self.encode_prompt(f"{cafe_prompt}, BREAK, {spray_tags}", negative_prompt)
            regions = (
                Region("base", cafe_prompt, base_names, base_weights, base_mask, base_strength),
                Region("hanyoil", hanyoil_tags, hanyoil_names, hanyoil_weights, person_mask, left_strength),
                Region("machine", machine_tags, base_names, base_weights, machine_mask, 1.1),
                Region("spray", spray_tags, base_names, base_weights, spray_mask, 1.25),
            )
            bundles = {
                "base": base_bundle,
                "hanyoil": hanyoil_bundle,
                "machine": machine_bundle,
                "spray": spray_bundle,
            }
        else:
            left_mask, right_mask = vertical_split_masks(width, height, 0.5, mask_feather, device, dtype)
            left_mask = downsample_mask(left_mask, latent_h, latent_w)
            right_mask = downsample_mask(right_mask, latent_h, latent_w)
            cafe_prompt = (
                "two women standing side by side and talking in a cozy modern cafe, "
                "cowboy shot, medium-long shot from head to upper thighs, knees cropped out, "
                "two-shot composition, evenly spaced across the frame, "
                "black-haired woman on the left and red-haired woman on the right, "
                "coffee counter, small round table, warm interior lighting, "
                "eye-level medium full shot, conversational body language, "
                "anime style, masterpiece, best quality, high-detailed, high contrast"
            )
            hanyoil_tags = (
                "hanyoil, black-haired woman, 1girl, black hair, dark gray eyes, slim body, light skin, "
                "halterneck white top, black jeans, torn jeans, midriff, navel, "
                "left woman, standing on the left side of the image, cowboy shot, foreground, "
                "turned slightly toward the red-haired woman on the right, "
                "smile, speaking, cafe interior"
            )
            hyojeong_tags = (
                "hyo-jeong, 1girl, red hair, long hair, low side ponytail, sidelocks, "
                "hair scrunchie, black eyes, medium breasts, cream colored blouse, "
                "long brown skirt, right woman, standing on the right side of the image, cowboy shot, "
                "turned slightly toward the black-haired woman on the left, "
                "gentle smile, listening and talking, cafe interior"
            )
            base_bundle = self.encode_prompt(cafe_prompt, negative_prompt)
            left_bundle = self.encode_prompt(f"{cafe_prompt}, BREAK, {hanyoil_tags}", negative_prompt)
            right_bundle = self.encode_prompt(f"{cafe_prompt}, BREAK, {hyojeong_tags}", negative_prompt)
            regions = (
                Region("base", cafe_prompt, base_names, base_weights, base_mask, base_strength),
                Region("hanyoil", hanyoil_tags, hanyoil_names, hanyoil_weights, left_mask, left_strength),
                Region("hyojeong", hyojeong_tags, ("hyojeong",), (1.0,), right_mask, right_strength),
            )
            bundles = {
                "base": base_bundle,
                "hanyoil": left_bundle,
                "hyojeong": right_bundle,
            }

        text_encoder_projection_dim = int(base_bundle[1].shape[-1])
        add_time_ids = self.pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=base_bundle[0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)

        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, 0.0)
        print(
            f"Generating regional scene '{scene}': seed={seed}, size={width}x{height}, "
            f"steps={steps}, cfg={guidance_scale}"
        )
        with torch.no_grad(), self.pipe.progress_bar(total=len(timesteps)) as progress_bar:
            for timestep in timesteps:
                mixed_noise = torch.zeros_like(latents)
                mixed_weight = torch.zeros_like(latents[:, :1])

                for region in regions:
                    self.set_adapters(region.adapter_names, region.adapter_weights)
                    noise = self.predict_noise(
                        latents,
                        timestep,
                        bundles[region.name],
                        add_time_ids,
                        guidance_scale,
                    )
                    weight = region.mask * region.strength
                    mixed_noise = mixed_noise + noise * weight
                    mixed_weight = mixed_weight + weight

                mixed_noise = mixed_noise / mixed_weight.clamp(min=1e-5)
                latents = self.pipe.scheduler.step(
                    mixed_noise,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                progress_bar.update()

        return self.decode_latents(latents)[0]


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    P = load_prompts_module()
    width = args.width or DEFAULT_WIDTH
    height = args.height or DEFAULT_HEIGHT
    steps = args.steps or P.STEPS
    cfg = args.cfg or P.CFG_SCALE

    print(f"prompts.py loaded: {PROMPTS_PATH}")
    print(f"seed: {seed}")
    print(f"output: {output_root}")

    load_runtime_dependencies()
    generator = RegionalCafeGenerator(P, args.yolo_path)
    image = generator.generate(
        seed=seed,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=cfg,
        left_strength=args.left_strength,
        right_strength=args.right_strength,
        base_strength=args.base_strength,
        mask_feather=args.mask_feather,
        scene=args.scene,
    )
    if not args.no_inpaint:
        single_character = "hanyoil" if args.scene == "coffee_chaos" else None
        image = generator.inpaint_faces(image, seed + 10_000, single_character=single_character)

    output_stem = "hanyoil_coffee_chaos" if args.scene == "coffee_chaos" else "hanyoil_hyojeong_cafe"
    path = output_root / f"{output_stem}_{seed}.png"
    image.save(path)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
