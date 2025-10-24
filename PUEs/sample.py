# QPS 采样：在已加载的代理模型 (UNet + Text Encoder) 上做 1000 次参数噪声采样，每次生成 4 张图
import argparse
from pathlib import Path
import numpy as np
# 当前 .py 脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
import os

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="None")
# parser.add_argument("--round", type=str, default="0")
parser.add_argument("--device_n", type=str, default="0")
parser.add_argument("--sigma", type=float, default=0.001)
parser.add_argument("--num_samples", type=int, default=1000)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_n
sigma = args.sigma
SIGMA_LIST = [sigma]
num_samples = args.num_samples

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from pathlib import Path
from contextlib import ExitStack

import torch
from diffusers import StableDiffusionPipeline

# ===== 你的现有配置 =====
PROMPT = "a photo of sks person"
EXPS_DIR = args.target
target_ids = [i for i in range(50)]
# target_ids = [7]
EXP_NAME_LIST = os.listdir(EXPS_DIR)
# SIGMA_LIST = [0.001]
# SIGMA_LIST = np.logspace(np.log10(0.0001), np.log10(0.1), num=20)[11:]

# ===== 采样与推理超参 =====
N_SAMPLES   = 1000     # 采样次数（每次 4 张）
IMAGES_PER  = 4        # 每次生成的张数
STEPS       = 100      # 推理步数
GUIDANCE    = 7.5      # CFG
WIDTH, HEIGHT = 512, 512
start_steps = 0
# ===== 是否排除某些参数（更稳）=====
EXCLUDE_BIAS_NORM = True       # 跳过 bias / norm(ln/gn/bn) 参数
EXCLUDE_EMBEDDINGS = True      # 跳过嵌入层(embedding/positional)等

# ===== 随机性（可复现）=====
BASE_SEED = 20250829
for EXP_NAME in EXP_NAME_LIST:
    if "exp_data" not in EXP_NAME:
        continue
    print(f"now EXP_NAME is {EXP_NAME}")
    id = int(EXP_NAME.split("-")[1][2:])
    if id not in target_ids:
        print(f"id {id} not in target_ids, skip")
        continue
    EXP_DIR = os.path.join(EXPS_DIR, EXP_NAME)
    TEXT_ENCODER_PTH = os.path.join(EXP_DIR, "models/50/text_encoder.pth")
    UNET_PTH        = os.path.join(EXP_DIR, "models/50/unet.pth")
    MODEL_DIR       = "/data/home/yekai/github/DiffAdvPerturbationBench/SD/stable-diffusion-2-1-base"
    # ===== 噪声控制：二选一 =====
    NOISE_MODE   = "fixed"   # "rms" 或 "fixed"
    ALPHA_UNET   = 0.03    # NOISE_MODE="rms" 时生效：sigma_layer = ALPHA * RMS(W)
    ALPHA_TEXT   = 0.01

    # SIGMA_UNET   = 1e-3*1    # NOISE_MODE="fixed" 时生效：sigma_layer = SIGMA
    # SIGMA_TEXT   = 5e-4*1
    for sigma in SIGMA_LIST:
        print(f"now sigma is {sigma}")
        SIGMA_UNET  = sigma    # NOISE_MODE="fixed" 时生效：sigma_layer = SIGMA
        SIGMA_TEXT  = sigma
        OUT_DIR = os.path.join(SCRIPT_DIR,"samples",os.path.basename(EXPS_DIR),f"outputs_sd21_base_replaced_qps{N_SAMPLES}_unet{SIGMA_UNET:.5f}_text{SIGMA_TEXT:.5f}",EXP_NAME)

        # ---------- 工具函数 ----------
        def _strip_module_prefix(state_dict):
            if any(k.startswith("module.") for k in state_dict.keys()):
                return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            return state_dict

        @torch.no_grad()
        def replace_unet_and_text_encoder(pipe, unet_pth=None, text_encoder_pth=None, device="cuda"):
            orig_unet_dtype = next(pipe.unet.parameters()).dtype
            orig_te_dtype   = next(pipe.text_encoder.parameters()).dtype

            # UNet
            if unet_pth:
                obj = torch.load(unet_pth, map_location="cpu")
                sd = _strip_module_prefix(obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj)
                new_unet = pipe.unet.__class__.from_config(pipe.unet.config)
                missing, unexpected = new_unet.load_state_dict(sd, strict=False)
                print(f"[UNet] missing: {len(missing)}, unexpected: {len(unexpected)}")
                pipe.unet = new_unet.to(device=device, dtype=orig_unet_dtype)

            # Text Encoder
            if text_encoder_pth:
                obj = torch.load(text_encoder_pth, map_location="cpu")
                sd = _strip_module_prefix(obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj)
                new_te = pipe.text_encoder.__class__(pipe.text_encoder.config)
                missing, unexpected = new_te.load_state_dict(sd, strict=False)
                print(f"[TextEncoder] missing: {len(missing)}, unexpected: {len(unexpected)}")
                pipe.text_encoder = new_te.to(device=device, dtype=orig_te_dtype)

            return pipe

        def tensor_rms(t: torch.Tensor):
            if t.numel() == 0:
                return torch.tensor(0.0, device=t.device, dtype=t.dtype)
            return (t.float().pow(2).mean().sqrt().clamp_min(1e-12)).to(t.dtype)

        def _should_skip(name: str):
            lname = name.lower()
            if EXCLUDE_BIAS_NORM and ("bias" in lname or "norm" in lname or "ln" in lname or "layernorm" in lname or "groupnorm" in lname or "batchnorm" in lname):
                return True
            if EXCLUDE_EMBEDDINGS and ("embed" in lname or "pos" in lname and "embedding" in lname):
                return True
            return False

        @torch.no_grad()
        def add_noise_inplace(module: torch.nn.Module, *, mode: str, alpha: float, sigma_fixed: float, g: torch.Generator):
            """
            对 module 的每个参数添加各向同性高斯噪声：
            NOISE_MODE="rms":    sigma_layer = alpha * RMS(W)
            NOISE_MODE="fixed":  sigma_layer = sigma_fixed
            返回 restore() 闭包用于恢复原值。
            """
            backups = []
            for name, p in module.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                if _should_skip(name):
                    continue
                if mode == "rms":
                    sigma = (alpha * tensor_rms(p.data)).to(p.dtype)
                elif mode == "fixed":
                    sigma = torch.as_tensor(sigma_fixed, device=p.device, dtype=p.dtype)
                else:
                    raise ValueError(f"Unknown NOISE_MODE: {mode}")
                if float(sigma) == 0.0:
                    continue
                noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=g) * sigma
                backups.append((p, p.data.detach().clone()))
                p.add_(noise)

            def restore():
                for p, buf in backups:
                    p.data.copy_(buf)
                backups.clear()

            return restore

        def ensure_dir(path: str):
            Path(path).mkdir(parents=True, exist_ok=True)

        # ---------- 加载 Pipeline 与代理参数 ----------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        ).to(device)

        pipe = replace_unet_and_text_encoder(
            pipe,
            unet_pth=UNET_PTH,
            text_encoder_pth=TEXT_ENCODER_PTH,
            device=device,
        )

        # 省显存
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

        # ---------- 进行 N 次采样 ----------
        ensure_dir(OUT_DIR)

        for i in range(start_steps,N_SAMPLES):
            sample_dir = Path(OUT_DIR) / f"sample_{i:04d}"
            ensure_dir(sample_dir)

            # 为本次采样构造子种子（可复现），并分别给 UNet/TextEncoder 噪声与图像生成用
            g_unet = torch.Generator(device=device).manual_seed(BASE_SEED * 100000 + i * 2 + 0)
            g_text = torch.Generator(device=device).manual_seed(BASE_SEED * 100000 + i * 2 + 1)

            # 加噪 & 生成（try/finally 确保恢复）
            restore_unet = add_noise_inplace(
                pipe.unet, mode=NOISE_MODE, alpha=ALPHA_UNET, sigma_fixed=SIGMA_UNET, g=g_unet
            )
            restore_text = add_noise_inplace(
                pipe.text_encoder, mode=NOISE_MODE, alpha=ALPHA_TEXT, sigma_fixed=SIGMA_TEXT, g=g_text
            )

            try:
                for j in range(IMAGES_PER):
                    g_img = torch.Generator(device=device).manual_seed(BASE_SEED * 10_000 + i * IMAGES_PER + j)
                    image = pipe(
                        PROMPT,
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE,
                        width=WIDTH,
                        height=HEIGHT,
                        generator=g_img,
                    ).images[0]
                    image.save(sample_dir / f"{j:03d}.png")
            finally:
                # 无论成功与否都恢复参数，避免噪声累积
                restore_text()
                restore_unet()

        print(f"All done. Root output dir: {OUT_DIR}")