#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA-based generative tuning for VLM disease detection.

This is intentionally separate from adapter_train.py/proj_train.py. Those
scripts are classifier-based; this script keeps the text-generation interface:

    image -> vision encoder -> projector -> LLM attention + LoRA -> generate

The base VLM weights remain frozen. Only LoRA matrices inserted into LLM
attention projection layers are trainable. By default, LoRA is restricted to
language_model self-attention q_proj/v_proj, matching the structures in:
internvl.txt, lingshu.txt, medgemma.txt, and qwen3.txt.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from repro_utils import (
    add_reproducibility_args,
    build_repro_metadata,
    configure_hf_cache as configure_hf_cache_portable,
    hf_revision_kwargs,
    prepare_dataframe_paths,
    resolved_model_revision,
    seed_everything as seed_everything_portable,
    seed_worker,
    torch_save_atomic,
    validate_resume_config,
)


MODEL_ID_BY_BACKEND = {
    "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen35": "Qwen/Qwen3.5-9B",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu": "lingshu-medical-mllm/Lingshu-7B",
    "lingshu_i": "lingshu-medical-mllm/Lingshu-I-8B",
}

PROMPT_BY_DATASET = {
    "mri": (
        "This is a brain MRI scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
    ),
    "oct": (
        "This is a retinal OCT scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
    ),
    "xray": (
        "This is a chest X-ray image.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
    ),
    "fundus": (
        "This is a retinal fundus photograph.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
    ),
}

SYSTEM_PROMPT_SHORT = (
    "Answer with exactly one word: normal or disease. "
    "Do not output punctuation or any extra words."
)

LORA_PRESETS = {
    "qv": ["q_proj", "v_proj"],
    "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qv_linear": ["q_proj", "v_proj", "in_proj_qkv"],
    "qkvo_linear": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"],
}

LANGUAGE_MODEL_PREFIXES = ("model.language_model.", "language_model.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INTERNVL_LIKE_BACKENDS = {"internvl", "lingshu_i"}
QWEN_LIKE_BACKENDS = {"qwen3", "qwen35"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train LoRA modules in frozen VLM LLM attention layers using generative "
            "normal/disease supervision."
        )
    )
    p.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["qwen3", "qwen35", "medgemma", "internvl", "lingshu", "lingshu_i"],
    )
    p.add_argument("--model_id", type=str, default=None, help="Override the default model id for the backend.")
    p.add_argument("--clean_csv", type=str, required=True)
    p.add_argument("--artefact_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./soft_prompt_ckpt")
    p.add_argument("--tag", type=str, default="lora_gen")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--load_4bit",
        action="store_true",
        help="Load the frozen base VLM in bitsandbytes 4-bit for single-GPU QLoRA-style training.",
    )
    p.add_argument(
        "--device_map",
        type=str,
        default="single",
        choices=["single", "auto"],
        help=(
            "Model placement during from_pretrained(). 'single' places the full model on the "
            "current CUDA device. 'auto' uses Accelerate-style placement and can reduce load-time OOM."
        ),
    )
    p.add_argument(
        "--max_gpu_memory",
        type=str,
        default=None,
        help="Optional max_memory value for each visible GPU when --device_map auto is used, e.g. 76GiB.",
    )
    p.add_argument(
        "--max_cpu_memory",
        type=str,
        default=None,
        help="Optional CPU max_memory value when --device_map auto is used, e.g. 120GiB.",
    )
    p.add_argument(
        "--offload_folder",
        type=str,
        default=None,
        help="Optional offload folder passed to from_pretrained() when --device_map auto is used.",
    )
    p.add_argument(
        "--use_flash_attn",
        action="store_true",
        help=(
            "Pass use_flash_attn=True to InternVL-like from_pretrained(). "
            "Use this only when flash-attn is installed in the active environment."
        ),
    )
    p.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional Transformers attention implementation passed to from_pretrained().",
    )
    p.add_argument(
        "--allow_plain_auto_model",
        action="store_true",
        help=(
            "For InternVL-like backends, try transformers.AutoModel as a last-resort fallback. "
            "This matches the adapter loader but only works for generative training if the "
            "loaded class exposes LM logits."
        ),
    )
    p.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="Quantization type used when --load_4bit is enabled.",
    )
    p.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype used by bitsandbytes 4-bit layers.",
    )
    p.add_argument(
        "--no_bnb_4bit_double_quant",
        action="store_true",
        help="Disable nested/double quantization when --load_4bit is enabled.",
    )
    p.add_argument(
        "--lora_targets",
        type=str,
        default="qv",
        help=(
            "LoRA target preset/name list. Use 'qv', 'qkvo', 'qv_linear', "
            "'qkvo_linear', or comma-separated module names."
        ),
    )
    p.add_argument("--max_weaks", type=int, default=5)
    p.add_argument(
        "--max_train_weaks",
        type=int,
        default=1,
        help=(
            "Override --max_weaks for training groups. Defaults to 1 to keep memory low; "
            "training samples this many weak views randomly from each group."
        ),
    )
    p.add_argument(
        "--max_val_weaks",
        type=int,
        default=None,
        help="Override --max_weaks for validation groups. Lower this to reduce GPU memory.",
    )
    p.add_argument("--w_clean", type=float, default=1.0)
    p.add_argument("--w_weak", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable model gradient checkpointing. Keeping it enabled saves memory for LoRA training.",
    )
    p.add_argument(
        "--empty_cache_steps",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache() every N optimizer steps. 0 disables it.",
    )
    p.add_argument(
        "--logits_to_keep",
        type=str,
        default="auto",
        help=(
            "Ask supported models to return only final logits for LM loss. "
            "Use 'auto', a positive integer, or 0 to disable."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument(
        "--eval_generate_every",
        type=int,
        default=1,
        help="Run generate() validation every N epochs. Use 0 to skip generation metrics.",
    )
    p.add_argument(
        "--eval_max_batches",
        type=int,
        default=0,
        help="Limit validation generation batches. 0 means all validation batches.",
    )
    p.add_argument("--normal_text", type=str, default="normal")
    p.add_argument("--disease_text", type=str, default="disease")
    run_mode = p.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", action="store_true", help="resume from this run's last checkpoint")
    run_mode.add_argument("--overwrite", action="store_true", help="replace checkpoints from a previous run")
    p.add_argument(
        "--best_ckpt_name",
        type=str,
        default="{backend}_{tag}_lora_best.pt",
        help="Supports {backend}, {tag}.",
    )
    p.add_argument(
        "--last_ckpt_name",
        type=str,
        default="{backend}_{tag}_lora_last.pt",
        help="Supports {backend}, {tag}.",
    )
    add_reproducibility_args(p)
    return p.parse_args()


def set_seed(seed: int, *, deterministic: bool) -> torch.Generator:
    return seed_everything_portable(seed, deterministic=deterministic)


def capture_rng_state(generator: torch.Generator) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "generator": generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]], generator: torch.Generator) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("generator") is not None:
        generator.set_state(state["generator"])


def load_trusted_checkpoint(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_torch_dtype_name(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def build_quantization_config(args: argparse.Namespace):
    if not bool(getattr(args, "load_4bit", False)):
        return None
    if DEVICE != "cuda":
        raise RuntimeError("--load_4bit requires a CUDA device.")
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "--load_4bit requires bitsandbytes in the active environment. "
            "Install it in the conda env, for example: pip install -U bitsandbytes"
        ) from exc

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(args.bnb_4bit_quant_type),
        bnb_4bit_compute_dtype=parse_torch_dtype_name(args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=not bool(args.no_bnb_4bit_double_quant),
    )


def module_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float):
        super().__init__()
        if int(r) <= 0:
            raise ValueError("LoRA rank r must be positive.")
        if not (hasattr(base, "in_features") and hasattr(base, "out_features")):
            raise TypeError(f"LoRA base module has no Linear feature metadata: {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.dropout = nn.Dropout(float(dropout))
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        weight = getattr(base, "weight", None)
        if weight is not None and hasattr(weight, "device"):
            device = weight.device
        else:
            device = next(base.parameters()).device
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_lora = self.dropout(x).to(dtype=torch.float32)
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling
        return base_out + delta.to(device=base_out.device, dtype=base_out.dtype)


def set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def parse_lora_targets(target_spec: str) -> List[str]:
    spec = str(target_spec).strip().lower()
    if spec in LORA_PRESETS:
        return list(LORA_PRESETS[spec])
    targets = [part.strip() for part in spec.split(",") if part.strip()]
    if not targets:
        raise ValueError("--lora_targets must be 'qv', 'qkvo', or comma-separated module names.")
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj"}
    bad = [target for target in targets if target not in allowed]
    if bad:
        raise ValueError(f"Unsupported LoRA target(s): {bad}. Allowed: {sorted(allowed)}")
    return targets


def is_language_lora_target(module_name: str, target_names: List[str]) -> bool:
    if not module_name.startswith(LANGUAGE_MODEL_PREFIXES):
        return False
    leaf = module_name.rsplit(".", 1)[-1]
    target_set = set(target_names)
    if ".self_attn." in module_name:
        return leaf in target_set
    if ".linear_attn." in module_name:
        return leaf in target_set and leaf in {"in_proj_qkv", "out_proj"}
    return False


def is_lora_wrappable_linear(module: nn.Module) -> bool:
    if isinstance(module, LoRALinear):
        return False
    if isinstance(module, nn.Linear):
        return True
    cls_name = module.__class__.__name__
    return (
        cls_name in {"Linear4bit", "Linear8bitLt"}
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def inject_lora_adapters(
    base_model: nn.Module,
    target_names: List[str],
    r: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    for p in base_model.parameters():
        p.requires_grad = False

    replacements: List[Tuple[str, nn.Module]] = []
    for name, module in base_model.named_modules():
        if is_lora_wrappable_linear(module) and is_language_lora_target(name, target_names):
            replacements.append((name, module))

    if not replacements:
        raise RuntimeError(
            "No LLM attention Linear modules matched for LoRA. "
            f"Targets={target_names}, prefixes={LANGUAGE_MODEL_PREFIXES}"
        )

    for name, module in replacements:
        set_submodule(base_model, name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))

    replaced_names = [name for name, _ in replacements]
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"[LoRA] targets={target_names} rank={r} alpha={alpha} dropout={dropout}")
    print(f"[LoRA] injected modules={len(replaced_names)} trainable_params={trainable:,}")
    for name in replaced_names[:8]:
        print(f"[LoRA]   {name}")
    if len(replaced_names) > 8:
        print(f"[LoRA]   ... ({len(replaced_names) - 8} more)")
    return replaced_names


def iter_lora_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def set_lora_train_mode(model: nn.Module, train: bool) -> None:
    for module in iter_lora_modules(model):
        module.train(train)


def lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def collect_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu()
        for name, p in model.named_parameters()
        if p.requires_grad and (name.endswith(".lora_A") or name.endswith(".lora_B"))
    }


def build_auto_device_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"device_map": "auto"}
    max_memory: Dict[Any, str] = {}
    if getattr(args, "max_gpu_memory", None):
        for idx in range(torch.cuda.device_count()):
            max_memory[idx] = str(args.max_gpu_memory)
    if getattr(args, "max_cpu_memory", None):
        max_memory["cpu"] = str(args.max_cpu_memory)
    if max_memory:
        kwargs["max_memory"] = max_memory
    if getattr(args, "offload_folder", None):
        offload_folder = str(args.offload_folder)
        os.makedirs(offload_folder, exist_ok=True)
        kwargs["offload_folder"] = offload_folder
    return kwargs


def cleanup_after_failed_load() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_safetensors_mmap_error(message: str) -> bool:
    text = str(message).lower()
    return "unable to mmap" in text and (".safetensors" in text or "safetensors" in text or "from file <" in text)


def add_load_failure_hint(message: str) -> str:
    if not is_safetensors_mmap_error(message):
        return message
    return (
        f"{message}\n"
        "[load] safetensors mmap failed while mapping a checkpoint shard. "
        "This is host RAM / virtual-memory pressure during model loading, not a CUDA allocator OOM. "
        "Request more CPU memory for the job, remove any strict ulimit/container memory cap, or use a "
        "smaller checkpoint. --device_map auto/offload can help GPU placement but cannot avoid mapping "
        "at least one safetensors shard."
    )


def from_pretrained_compat(cls, model_id: str, **kwargs):
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as exc:
        message = str(exc)
        if "dtype" in kwargs and "unexpected keyword argument" in message and "dtype" in message:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            print("[load] transformers build does not accept dtype; retrying with legacy torch_dtype.")
            try:
                return from_pretrained_compat(cls, model_id, **fallback_kwargs)
            except Exception as fallback_exc:
                raise fallback_exc from None
        for optional_key in ["use_flash_attn", "attn_implementation"]:
            if optional_key in kwargs and "unexpected keyword argument" in message and optional_key in message:
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop(optional_key, None)
                print(f"[load] {cls.__name__} does not accept {optional_key}; retrying without it.")
                try:
                    return from_pretrained_compat(cls, model_id, **fallback_kwargs)
                except Exception as fallback_exc:
                    raise fallback_exc from None
        raise


def load_model_on_device(
    cls,
    model_id: str,
    common: Dict[str, Any],
    args: argparse.Namespace,
    trust_remote_code: bool = False,
) -> nn.Module:
    kwargs = dict(common)
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    if DEVICE == "cuda":
        try:
            cuda_index = torch.cuda.current_device()
            if str(getattr(args, "device_map", "single")) == "auto":
                placement_kwargs = build_auto_device_kwargs(args)
                model = from_pretrained_compat(cls, model_id, **placement_kwargs, **kwargs)
                device_map = getattr(model, "hf_device_map", None)
                print(f"[load] {cls.__name__} loaded with device_map=auto")
                if device_map is not None:
                    print(f"[load] hf_device_map={device_map}")
            else:
                model = from_pretrained_compat(cls, model_id, device_map={"": cuda_index}, **kwargs)
                print(f"[load] {cls.__name__} loaded directly on cuda:{cuda_index}")
            return model
        except Exception as exc:
            placement = "device_map=auto" if str(getattr(args, "device_map", "single")) == "auto" else "direct CUDA"
            message = add_load_failure_hint(f"{placement} load failed for {cls.__name__}: {exc}")
            cleanup_after_failed_load()
            raise RuntimeError(message) from None

    model = from_pretrained_compat(cls, model_id, **kwargs)
    model.to(DEVICE)
    return model


def load_backend(backend: str, model_id: str, args: argparse.Namespace):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    if backend in ["medgemma", "internvl", "qwen3", "qwen35", "lingshu", "lingshu_i"] and DEVICE == "cuda":
        torch_dtype = torch.bfloat16

    quantization_config = build_quantization_config(args)
    revision_kwargs = hf_revision_kwargs(args.model_revision)
    common: Dict[str, Any] = {"low_cpu_mem_usage": True, **revision_kwargs}
    if quantization_config is None:
        common["dtype"] = torch_dtype
    else:
        common["quantization_config"] = quantization_config
        print(
            "[load] 4-bit base model loading enabled "
            f"(type={args.bnb_4bit_quant_type}, compute={args.bnb_4bit_compute_dtype}, "
            f"double_quant={not bool(args.no_bnb_4bit_double_quant)})"
        )

    if backend in QWEN_LIKE_BACKENDS:
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)
        class_names = {
            "qwen3": ["Qwen3VLForConditionalGeneration", "AutoModelForMultimodalLM"],
            "qwen35": ["Qwen3_5ForConditionalGeneration", "AutoModelForMultimodalLM"],
        }[backend]
        model = None
        last_err: Optional[Exception] = None
        for cls_name in class_names:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = load_model_on_device(cls, model_id, common, args)
                print(f"[{backend}] loaded with {cls_name}")
                break
            except Exception as exc:
                last_err = exc
                model = None
        if model is None:
            raise RuntimeError(f"Failed to load {backend} model with {class_names}: {last_err}")
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **revision_kwargs)
        model = load_model_on_device(
            Qwen2_5_VLForConditionalGeneration,
            model_id,
            common,
            args,
            trust_remote_code=True,
        )
        model.eval()
        return model, processor

    if backend in INTERNVL_LIKE_BACKENDS:
        from transformers import AutoProcessor as AP2

        processor = AP2.from_pretrained(model_id, trust_remote_code=True, **revision_kwargs)
        internvl_common = dict(common)
        if bool(getattr(args, "use_flash_attn", False)):
            internvl_common["use_flash_attn"] = True
        if getattr(args, "attn_implementation", None):
            internvl_common["attn_implementation"] = str(args.attn_implementation)
        model = None
        load_errors: List[str] = []
        # InternVL-family HF checkpoints, including Lingshu-I-8B, expose a
        # conditional-generation class. LoRA training needs logits, so plain
        # AutoModel is only attempted behind an explicit fallback flag.
        class_names = [
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]
        if bool(getattr(args, "allow_plain_auto_model", False)):
            class_names = ["AutoModel"] + class_names
        for cls_name in class_names:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                load_errors.append(
                    f"{cls_name}: class is not available in transformers {getattr(transformers, '__version__', 'unknown')}"
                )
                continue
            try:
                model = load_model_on_device(cls, model_id, internvl_common, args, trust_remote_code=True)
                print(f"[{backend}] loaded with {cls_name}")
                break
            except Exception as exc:
                err_text = str(exc)
                load_errors.append(f"{cls_name}: {err_text}")
                model = None
                cleanup_after_failed_load()
                if is_safetensors_mmap_error(err_text):
                    break
        if model is None:
            detail = "\n".join(load_errors) if load_errors else "No conditional-generation AutoModel class is available."
            raise RuntimeError(
                f"Failed to load {backend} with a conditional-generation class. "
                "Plain AutoModel is not supported for generative LoRA training because it "
                "does not expose LM logits. Tried:\n"
                f"{detail}"
            )
        model.eval()

        img_seq_len = getattr(getattr(model, "config", None), "image_seq_length", None)
        if img_seq_len is not None:
            for obj in [processor, getattr(processor, "tokenizer", None)]:
                if obj is None:
                    continue
                for attr in ["num_image_tokens", "num_image_token", "image_seq_len", "image_seq_length"]:
                    if hasattr(obj, attr):
                        try:
                            setattr(obj, attr, int(img_seq_len))
                        except Exception:
                            pass
            print(f"[{backend}] processor image tokens set to image_seq_length={int(img_seq_len)}")
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id, **revision_kwargs)

        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = load_model_on_device(AutoImageTextToText, model_id, common, args)
            model.eval()
            return model, processor

        AutoMultimodalLM = getattr(transformers, "AutoModelForMultimodalLM", None)
        if AutoMultimodalLM is not None:
            model = load_model_on_device(AutoMultimodalLM, model_id, common, args)
            model.eval()
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = load_model_on_device(AutoVision2Seq, model_id, common, args)
            model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM

        model = load_model_on_device(AutoModelForCausalLM, model_id, common, args)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


def has_output_head(model: nn.Module) -> bool:
    if hasattr(model, "lm_head") and isinstance(getattr(model, "lm_head"), nn.Module):
        return True
    for attr_path in [
        ("language_model", "lm_head"),
        ("model", "language_model", "lm_head"),
        ("model", "lm_head"),
    ]:
        obj: Any = model
        for attr in attr_path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return True
    if hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
        try:
            return model.get_output_embeddings() is not None
        except Exception:
            return False
    return False


def validate_generative_model(model: nn.Module, backend: str) -> None:
    if has_output_head(model):
        return
    if backend in INTERNVL_LIKE_BACKENDS:
        raise RuntimeError(
            f"The loaded {backend} class has no lm_head/output embeddings, so LoRA "
            "generative LM loss cannot be computed. Use an HF-format conditional-generation "
            "class such as AutoModelForMultimodalLM. Plain AutoModel/chat paths are "
            "inference-oriented and do not expose the logits needed here."
        )
    raise RuntimeError(
        f"The loaded {backend} model has no detectable lm_head/output embeddings; "
        "cannot compute generative LM loss."
    )


def configure_memory_saving(model: nn.Module, gradient_checkpointing: bool) -> None:
    for obj in [getattr(model, "config", None), getattr(model, "generation_config", None)]:
        if obj is not None and hasattr(obj, "use_cache"):
            try:
                setattr(obj, "use_cache", False)
            except Exception:
                pass

    if not gradient_checkpointing:
        print("[memory] gradient checkpointing disabled")
        return

    enabled = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        enabled = True
    else:
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    module.gradient_checkpointing_enable()
                enabled = True

    if enabled:
        print("[memory] gradient checkpointing enabled; use_cache=False")
    else:
        print("[memory] gradient checkpointing not available; use_cache=False")


def load_split_csv(
    path: str,
    data_root: Optional[str],
    default_severity: Optional[str],
    *,
    check_paths: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    required = ["dataset", "filepath", "fileindex", "use_type", "binarylabel"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    df["use_type_norm"] = df["use_type"].astype(str).str.lower()
    df = df[df["use_type_norm"].isin(["train", "val"])].copy()
    if df.empty:
        raise ValueError(f"{path}: no rows with use_type=train or use_type=val")
    if "severity" in df.columns:
        df["severity_norm"] = df["severity"].fillna(default_severity or "clean").astype(str).str.lower()
    elif default_severity is not None:
        df["severity_norm"] = str(default_severity).lower()
    else:
        raise ValueError(f"{path}: severity column missing")

    df = prepare_dataframe_paths(
        df,
        csv_path=path,
        data_root=data_root,
        check_exists=check_paths,
    )
    df["fileindex"] = df["fileindex"].astype(str)
    return df.reset_index(drop=True)


class CleanMultiWeakDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_by_dataset: Dict[str, str],
        system_prompt: Optional[str],
        max_weaks: int,
        normal_text: str,
        disease_text: str,
        random_sample_weaks: bool = False,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt
        self.max_weaks = int(max_weaks)
        self.normal_text = str(normal_text)
        self.disease_text = str(disease_text)
        self.random_sample_weaks = bool(random_sample_weaks)

        if "severity_norm" not in self.df.columns:
            self.df["severity_norm"] = self.df.get("severity", "clean").fillna("clean").astype(str).str.lower()
        if "dataset_norm" not in self.df.columns:
            self.df["dataset_norm"] = self.df["dataset"].astype(str).str.lower()

        self.groups = []
        self.group_modalities: List[str] = []
        for (dataset_name, fid), group in self.df.groupby(["dataset_norm", "fileindex"], sort=False):
            group = group.reset_index(drop=True)
            clean_rows = group[group["severity_norm"].eq("clean")]
            weak_rows = group[group["severity_norm"].eq("weak")]
            if len(clean_rows) == 0 or len(weak_rows) == 0:
                continue
            clean_row = clean_rows.iloc[0]
            if not self.random_sample_weaks:
                weak_rows = weak_rows.iloc[: self.max_weaks]
            dataset_name = str(dataset_name).lower()
            self.groups.append((f"{dataset_name}::{fid}", dataset_name, clean_row, weak_rows))
            self.group_modalities.append(dataset_name)

        sample_mode = "random" if self.random_sample_weaks else "first"
        print(
            f"Dataset - found {len(self.groups)} fileindex groups "
            f"(clean + <= {self.max_weaks} weaks, {sample_mode} weak sampling)."
        )
        if self.group_modalities:
            print("Dataset - group modalities:", pd.Series(self.group_modalities).value_counts().to_dict())

    def _target_text(self, label: int) -> str:
        return self.disease_text if int(label) == 1 else self.normal_text

    def _make_sample(self, row: pd.Series) -> Dict[str, Any]:
        img_path = str(row["filepath"])
        img = Image.open(img_path).convert("RGB")
        modality = str(row["dataset_norm"]).lower()
        label = int(row["binarylabel"])
        task_prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\n",
        )
        full_prompt = (self.system_prompt + "\n\n" + task_prompt) if self.system_prompt else task_prompt
        return {
            "image": img,
            "input_text": full_prompt,
            "target_text": self._target_text(label),
            "label": label,
            "path": img_path,
            "fileindex": str(row.get("fileindex", img_path)),
            "modality": modality,
            "corruption": str(row.get("corruption", "clean")).lower(),
        }

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fid, modality, clean_row, weak_rows = self.groups[idx]
        clean = self._make_sample(clean_row)
        if self.random_sample_weaks and self.max_weaks < len(weak_rows):
            selected_idx = random.sample(range(len(weak_rows)), k=max(0, self.max_weaks))
            weak_rows = weak_rows.iloc[selected_idx]
        weaks = [self._make_sample(r) for _, r in weak_rows.iterrows()]
        return {"fileindex": fid, "modality": modality, "clean": clean, "weaks": weaks}


def get_tokenizer(processor):
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        tok = processor
    return tok


def tokenizer_len(tokenizer, text: str) -> int:
    try:
        ids = tokenizer(text, add_special_tokens=False).input_ids
    except TypeError:
        ids = tokenizer(text).input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def apply_chat_template(
    processor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
    **template_kwargs: Any,
) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
    except TypeError:
        if template_kwargs:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        raise


def build_prompt_and_full_text(
    processor,
    backend: str,
    image: Image.Image,
    prompt_text: str,
    target_text: str,
) -> Tuple[str, str]:
    if backend == "qwen35":
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
        prefix = apply_chat_template(
            processor,
            [user],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenizer = get_tokenizer(processor)
        eos_token = getattr(tokenizer, "eos_token", None) or ""
        full = f"{prefix}{target_text}{eos_token}"
        return prefix, full

    if backend in ["qwen3", "lingshu", "internvl", "lingshu_i"]:
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
        prefix = apply_chat_template(processor, [user], add_generation_prompt=True)
        full = apply_chat_template(
            processor,
            [user, {"role": "assistant", "content": target_text}],
            add_generation_prompt=False,
        )
        return prefix, full

    if backend == "medgemma":
        user = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
        prefix = apply_chat_template(processor, [user], add_generation_prompt=True)
        full = apply_chat_template(
            processor,
            [user, {"role": "assistant", "content": target_text}],
            add_generation_prompt=False,
        )
        return prefix, full

    return prompt_text, f"{prompt_text}\n{target_text}"


def merge_processor_outputs(single_inputs: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, Any]:
    model_inputs: Dict[str, Any] = {}
    keys = set()
    for one in single_inputs:
        keys.update(one.keys())

    for key in keys:
        vals = [one[key] for one in single_inputs if key in one]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            if vals[0].dim() == 2 and key in {"input_ids", "attention_mask", "token_type_ids"}:
                max_t = max(int(v.shape[1]) for v in vals)
                pad_value = pad_token_id if key == "input_ids" else 0
                padded = []
                for v in vals:
                    if int(v.shape[1]) < max_t:
                        pad = torch.full(
                            (int(v.shape[0]), max_t - int(v.shape[1])),
                            pad_value,
                            dtype=v.dtype,
                        )
                        v = torch.cat([v, pad], dim=1)
                    padded.append(v)
                model_inputs[key] = torch.cat(padded, dim=0)
            else:
                model_inputs[key] = torch.cat(vals, dim=0)
        else:
            merged = []
            for v in vals:
                if isinstance(v, list):
                    merged.extend(v)
                else:
                    merged.append(v)
            model_inputs[key] = merged
    return model_inputs


def encode_text_image_batch(
    processor,
    backend: str,
    texts: List[str],
    images: List[Image.Image],
) -> Dict[str, Any]:
    tokenizer = get_tokenizer(processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    if backend in ["internvl", "lingshu_i", "medgemma"]:
        single_inputs = []
        patch_counts = []
        for text, image in zip(texts, images):
            one = dict(processor(text=[text], images=[image], padding=True, return_tensors="pt"))
            single_inputs.append(one)
            pv = one.get("pixel_values", None)
            patch_counts.append(int(pv.shape[0]) if torch.is_tensor(pv) else 0)
        model_inputs = merge_processor_outputs(single_inputs, int(pad_token_id))
        if backend in INTERNVL_LIKE_BACKENDS:
            model_inputs["num_patches_list"] = patch_counts
        return model_inputs

    return dict(processor(text=texts, images=images, padding=True, return_tensors="pt"))


def build_lm_labels(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    prefix_lens: List[int],
) -> torch.Tensor:
    labels = input_ids.clone()
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels[attention_mask == 0] = -100

    for row_idx, prefix_len in enumerate(prefix_lens):
        valid_idx = torch.nonzero(attention_mask[row_idx] > 0, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue
        n_mask = min(int(prefix_len), int(valid_idx.numel()))
        if n_mask > 0:
            labels[row_idx, valid_idx[:n_mask]] = -100
    return labels


def input_lengths_from_attention(model_inputs: Dict[str, Any]) -> List[int]:
    attention_mask = model_inputs.get("attention_mask", None)
    if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
        return attention_mask.to(torch.long).sum(dim=1).cpu().tolist()
    input_ids = model_inputs.get("input_ids", None)
    if torch.is_tensor(input_ids) and input_ids.dim() == 2:
        return [int(input_ids.shape[1])] * int(input_ids.shape[0])
    raise RuntimeError("Cannot infer input lengths from processor output.")


def make_generative_collate_fn(processor, backend: str, include_prompt_inputs: bool):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images: List[Image.Image] = []
        prompts: List[str] = []
        targets: List[str] = []
        labels_cls: List[int] = []
        paths: List[str] = []
        fileindices: List[str] = []
        group_ids: List[int] = []
        is_clean: List[bool] = []
        modalities: List[str] = []
        corruptions: List[str] = []

        for group_idx, item in enumerate(batch):
            views = [item["clean"]] + list(item["weaks"])
            for view_idx, sample in enumerate(views):
                images.append(sample["image"])
                prompts.append(sample["input_text"])
                targets.append(sample["target_text"])
                labels_cls.append(int(sample["label"]))
                paths.append(str(sample.get("path", "<unknown>")))
                fileindices.append(str(item.get("fileindex", sample.get("fileindex", "<unknown>"))))
                group_ids.append(group_idx)
                is_clean.append(view_idx == 0)
                modalities.append(str(sample.get("modality", item.get("modality", "unknown"))).lower())
                corruptions.append(str(sample.get("corruption", "clean" if view_idx == 0 else "unknown")).lower())

        prefix_texts: List[str] = []
        full_texts: List[str] = []
        for image, prompt, target in zip(images, prompts, targets):
            prefix, full = build_prompt_and_full_text(processor, backend, image, prompt, target)
            prefix_texts.append(prefix)
            full_texts.append(full)

        model_inputs = encode_text_image_batch(processor, backend, full_texts, images)
        if "input_ids" not in model_inputs:
            raise RuntimeError("Processor output has no input_ids; cannot build LM labels.")
        # Prefix lengths must use the same multimodal processor path as the
        # full inputs. Tokenizer-only lengths are wrong when image placeholders
        # expand into backend-specific image token sequences.
        prefix_model_inputs = encode_text_image_batch(processor, backend, prefix_texts, images)
        prefix_lens = input_lengths_from_attention(prefix_model_inputs)
        attention_mask = model_inputs.get("attention_mask", None)
        labels_lm = build_lm_labels(model_inputs["input_ids"], attention_mask, prefix_lens)

        out = dict(model_inputs)
        out["labels"] = labels_lm
        out["labels_cls"] = torch.tensor(labels_cls, dtype=torch.long)
        out["paths"] = paths
        out["fileindices"] = fileindices
        out["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
        out["is_clean"] = torch.tensor(is_clean, dtype=torch.bool)
        out["modality"] = modalities
        out["corruption"] = corruptions
        out["target_texts"] = targets
        out["prefix_texts"] = prefix_texts

        if include_prompt_inputs:
            out["prompt_inputs"] = prefix_model_inputs

        return out

    return collate


def get_model_float_dtype(model: nn.Module) -> Optional[torch.dtype]:
    for p in model.parameters():
        if torch.is_floating_point(p):
            return p.dtype
    return None


def get_model_device(model: nn.Module) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        try:
            device = torch.device(model_device)
            if device.type != "meta":
                return device
        except Exception:
            pass

    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device(DEVICE)


def to_device(x: Any, target_device: torch.device, float_dtype: Optional[torch.dtype] = None):
    if torch.is_tensor(x):
        if float_dtype is not None and torch.is_floating_point(x):
            return x.to(device=target_device, dtype=float_dtype, non_blocking=True)
        return x.to(device=target_device, non_blocking=True)
    if isinstance(x, dict):
        out = {}
        keep = {
            "paths",
            "fileindices",
            "modality",
            "corruption",
            "target_texts",
            "prefix_texts",
        }
        for key, value in x.items():
            out[key] = value if key in keep else to_device(value, target_device, float_dtype)
        return out
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device, float_dtype) for v in x)
    return x


MODEL_METADATA_KEYS = {
    "labels_cls",
    "paths",
    "fileindices",
    "group_ids",
    "is_clean",
    "modality",
    "corruption",
    "target_texts",
    "prefix_texts",
    "prompt_inputs",
}


def model_kwargs_from_batch(batch: Dict[str, Any], include_labels: bool) -> Dict[str, Any]:
    kwargs = {
        key: value
        for key, value in batch.items()
        if key not in MODEL_METADATA_KEYS and (include_labels or key != "labels")
    }
    return kwargs


def call_model(base_model: nn.Module, kwargs: Dict[str, Any]):
    configs = [
        {"return_dict": True, "use_cache": False},
        {"return_dict": True},
        {"use_cache": False},
        {},
    ]
    last_err: Optional[Exception] = None
    for drop_num_patches in [True, False]:
        for drop_logits_to_keep in [False, True]:
            work_kwargs = dict(kwargs)
            if drop_num_patches:
                work_kwargs.pop("num_patches_list", None)
            if drop_logits_to_keep:
                work_kwargs.pop("logits_to_keep", None)
            for cfg in configs:
                fkwargs = dict(work_kwargs)
                fkwargs.update(cfg)
                try:
                    return base_model(**fkwargs)
                except TypeError as exc:
                    last_err = exc
                    continue
                except Exception as exc:
                    last_err = exc
                    break
    raise RuntimeError(f"base_model forward failed: {last_err}")


def get_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if isinstance(outputs, dict) and outputs.get("logits", None) is not None:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise RuntimeError(
        "Model output has no logits. This backend may not expose a generative forward path."
    )


def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_clean: torch.Tensor,
    w_clean: float,
    w_weak: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:2] != labels.shape:
        if logits.shape[0] != labels.shape[0] or logits.shape[1] >= labels.shape[1]:
            raise RuntimeError(f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
    if logits.shape[1] < 2:
        raise RuntimeError("Need at least two sequence positions for causal LM loss.")

    if logits.shape[1] < labels.shape[1]:
        # Some models support logits_to_keep and only return the final sequence
        # positions. The last logit is unused by causal LM loss, so align the
        # remaining logits with the final labels.
        n_pred = int(logits.shape[1]) - 1
        shift_logits = logits[:, :n_pred, :]
        shift_labels = labels[:, -n_pred:]
    else:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

    vocab = int(shift_logits.shape[-1])
    token_mask = shift_labels.ne(-100)
    valid = token_mask.any(dim=1)
    if not bool(valid.any().item()):
        raise RuntimeError("No supervised target tokens found in labels.")

    selected_logits = shift_logits[token_mask].float()
    selected_labels = shift_labels[token_mask]
    selected_loss = F.cross_entropy(
        selected_logits.view(-1, vocab),
        selected_labels.view(-1),
        reduction="none",
    )

    row_ids = torch.arange(
        int(shift_labels.shape[0]),
        device=shift_labels.device,
        dtype=torch.long,
    ).unsqueeze(1).expand_as(shift_labels)[token_mask]
    per_example_sum = torch.zeros(int(shift_labels.shape[0]), device=selected_loss.device, dtype=selected_loss.dtype)
    per_example_count = torch.zeros_like(per_example_sum)
    per_example_sum.scatter_add_(0, row_ids, selected_loss)
    per_example_count.scatter_add_(0, row_ids, torch.ones_like(selected_loss))
    per_example = per_example_sum / per_example_count.clamp_min(1)

    weights = torch.where(
        is_clean.to(device=per_example.device),
        torch.tensor(float(w_clean), device=per_example.device, dtype=per_example.dtype),
        torch.tensor(float(w_weak), device=per_example.device, dtype=per_example.dtype),
    )
    loss = (per_example[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1.0)
    return loss, per_example.detach()


def resolve_logits_to_keep(spec: str, labels: torch.Tensor) -> Optional[int]:
    value = str(spec).strip().lower()
    if value in {"", "none", "false", "off", "0"}:
        return None
    if value == "auto":
        supervised = labels[:, 1:].ne(-100).sum(dim=1)
        keep = int(supervised.max().item()) + 1
        return max(2, keep)
    keep = int(value)
    return keep if keep > 0 else None


NORMAL_RE = re.compile(r"\bnormal\b", re.IGNORECASE)
DISEASE_RE = re.compile(r"\bdisease\b", re.IGNORECASE)


def parse_generated_label(text: str) -> int:
    normal = NORMAL_RE.search(text)
    disease = DISEASE_RE.search(text)
    if normal and disease:
        return 0 if normal.start() < disease.start() else 1
    if disease:
        return 1
    if normal:
        return 0
    return -1


def call_generate(base_model: nn.Module, prompt_inputs: Dict[str, Any], max_new_tokens: int) -> torch.Tensor:
    kwargs = {
        key: value
        for key, value in prompt_inputs.items()
        if key not in {"labels", "labels_cls", "paths", "fileindices", "group_ids", "is_clean", "modality", "corruption"}
    }
    target_device = get_model_device(base_model)
    kwargs = to_device(kwargs, target_device, get_model_float_dtype(base_model))
    for key, value in list(kwargs.items()):
        if torch.is_tensor(value) and value.device != target_device:
            kwargs[key] = value.to(device=target_device, non_blocking=True)
    configs = [
        {"max_new_tokens": int(max_new_tokens), "do_sample": False, "use_cache": True},
        {"max_new_tokens": int(max_new_tokens), "do_sample": False},
    ]
    last_err: Optional[Exception] = None
    for drop_num_patches in [True, False]:
        work_kwargs = dict(kwargs)
        if drop_num_patches:
            work_kwargs.pop("num_patches_list", None)
        for cfg in configs:
            try:
                return base_model.generate(**work_kwargs, **cfg)
            except TypeError as exc:
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                break
    raise RuntimeError(f"generate() failed: {last_err}")


def generate_batch_predictions(
    base_model: nn.Module,
    processor,
    batch: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[List[str], List[int]]:
    if "prompt_inputs" not in batch:
        return [], []
    tokenizer = get_tokenizer(processor)
    prompt_inputs = batch["prompt_inputs"]
    generated = call_generate(base_model, prompt_inputs, max_new_tokens=max_new_tokens)
    if not torch.is_tensor(generated):
        sequences = getattr(generated, "sequences", None)
        if sequences is None:
            raise RuntimeError(f"Unsupported generate output type: {type(generated)}")
        generated = sequences

    prompt_width = int(prompt_inputs["input_ids"].shape[1])
    new_tokens = generated[:, prompt_width:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    preds = [parse_generated_label(text) for text in texts]
    return texts, preds


def compute_binary_metrics_from_preds(y_true: List[int], pred_raw: List[int]) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "acc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "invalid": 0,
            "n": 0,
        }

    yt = np.asarray(y_true, dtype=np.int64)
    pr = np.asarray(pred_raw, dtype=np.int64)
    invalid = pr < 0
    pred = pr.copy()
    pred[invalid] = 1 - yt[invalid]
    score = pred.astype(np.float32)
    score[invalid] = 0.5

    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    eps = 1e-12
    acc = (tp + tn) / max(1, len(yt))
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    if np.unique(yt).size < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(yt, score))

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "invalid": int(invalid.sum()),
        "invalid_rate": float(invalid.mean()),
        "n": int(len(yt)),
    }


def compute_split_metrics(
    y_true: List[int],
    pred_raw: List[int],
    is_clean: List[bool],
    modalities: List[str],
) -> Dict[str, Any]:
    metrics = {"overall": compute_binary_metrics_from_preds(y_true, pred_raw)}

    for split_name, keep_value in [("clean", True), ("weak", False)]:
        idx = [i for i, value in enumerate(is_clean) if bool(value) is keep_value]
        metrics[split_name] = compute_binary_metrics_from_preds(
            [y_true[i] for i in idx],
            [pred_raw[i] for i in idx],
        )

    by_modality: Dict[str, Any] = {}
    f1s, aurocs = [], []
    for modality in sorted(set(str(m).lower() for m in modalities)):
        idx = [i for i, m in enumerate(modalities) if str(m).lower() == modality]
        m = compute_binary_metrics_from_preds(
            [y_true[i] for i in idx],
            [pred_raw[i] for i in idx],
        )
        by_modality[modality] = m
        if np.isfinite(m.get("f1", float("nan"))):
            f1s.append(float(m["f1"]))
        if np.isfinite(m.get("auroc", float("nan"))):
            aurocs.append(float(m["auroc"]))

    metrics["by_modality"] = by_modality
    metrics["modality_macro"] = {
        "f1": float(np.mean(f1s)) if f1s else float("nan"),
        "auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "n_modalities": int(len(by_modality)),
    }
    return metrics


def run_epoch(
    base_model: nn.Module,
    processor,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    run_generation: bool,
) -> Dict[str, Any]:
    train = optimizer is not None
    base_model.eval()
    set_lora_train_mode(base_model, train)

    lora_params = lora_trainable_parameters(base_model)
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found.")
    target_device = next(iter(lora_params)).device
    input_float_dtype = get_model_float_dtype(base_model)
    total_loss = 0.0
    total_examples = 0
    n_steps = 0

    all_y: List[int] = []
    all_pred: List[int] = []
    all_is_clean: List[bool] = []
    all_modalities: List[str] = []
    generated_samples: List[Dict[str, Any]] = []

    for step, batch in enumerate(loader):
        batch = to_device(batch, target_device, input_float_dtype)
        labels_lm = batch["labels"]
        labels_cls = batch["labels_cls"]
        is_clean = batch["is_clean"]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            kwargs = model_kwargs_from_batch(batch, include_labels=False)
            logits_to_keep = resolve_logits_to_keep(str(args.logits_to_keep), labels_lm)
            if logits_to_keep is not None:
                kwargs["logits_to_keep"] = int(logits_to_keep)
            outputs = call_model(base_model, kwargs)
            logits = get_logits(outputs)
            if labels_lm.device != logits.device:
                labels_lm = labels_lm.to(device=logits.device, non_blocking=True)
            if is_clean.device != logits.device:
                is_clean = is_clean.to(device=logits.device, non_blocking=True)
            loss, per_example = weighted_lm_loss(
                logits,
                labels_lm,
                is_clean,
                w_clean=float(args.w_clean),
                w_weak=float(args.w_weak),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite LM loss")

            if train:
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, float(args.grad_clip))
                optimizer.step()
                if DEVICE == "cuda" and int(args.empty_cache_steps) > 0:
                    if (step + 1) % int(args.empty_cache_steps) == 0:
                        torch.cuda.empty_cache()

        batch_size = int(labels_cls.numel())
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        n_steps += 1

        if (not train) and run_generation:
            if int(args.eval_max_batches) > 0 and step >= int(args.eval_max_batches):
                continue
            with torch.no_grad():
                texts, preds = generate_batch_predictions(
                    base_model,
                    processor,
                    batch,
                    max_new_tokens=int(args.max_new_tokens),
                )
            y_list = labels_cls.detach().cpu().to(torch.int64).tolist()
            clean_list = batch["is_clean"].detach().cpu().tolist()
            modalities = [str(m).lower() for m in batch["modality"]]
            all_y.extend([int(y) for y in y_list])
            all_pred.extend([int(p) for p in preds])
            all_is_clean.extend([bool(v) for v in clean_list])
            all_modalities.extend(modalities)
            for i, text in enumerate(texts[: max(0, 12 - len(generated_samples))]):
                generated_samples.append(
                    {
                        "target": int(y_list[i]),
                        "pred": int(preds[i]),
                        "is_clean": bool(clean_list[i]),
                        "modality": modalities[i],
                        "generated": text,
                    }
                )

    out: Dict[str, Any] = {
        "loss": total_loss / max(1, total_examples),
        "n_steps": int(n_steps),
        "n_examples": int(total_examples),
    }
    if run_generation and len(all_y) > 0:
        out.update(compute_split_metrics(all_y, all_pred, all_is_clean, all_modalities))
        out["generated_samples"] = generated_samples
    return out


def save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    configure_hf_cache_portable(args.hf_cache_dir)
    generator = set_seed(int(args.seed), deterministic=not args.allow_nondeterministic)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = save_dir / "metrics_json"
    results_dir.mkdir(parents=True, exist_ok=True)
    name_ctx = {"backend": args.backend, "tag": args.tag}
    best_ckpt = save_dir / args.best_ckpt_name.format(**name_ctx)
    last_ckpt = save_dir / args.last_ckpt_name.format(**name_ctx)
    if args.resume and not last_ckpt.exists():
        raise FileNotFoundError(f"--resume was requested but no checkpoint exists: {last_ckpt}")
    if not args.resume and not args.overwrite and (last_ckpt.exists() or best_ckpt.exists()):
        raise FileExistsError(
            f"Checkpoint already exists for backend={args.backend}, tag={args.tag}. "
            "Pass --resume, --overwrite, or choose another --tag/--save_dir."
        )
    if args.overwrite:
        for checkpoint_path in (best_ckpt, last_ckpt):
            if checkpoint_path.is_file():
                checkpoint_path.unlink()
    resume_ckpt = load_trusted_checkpoint(str(last_ckpt)) if args.resume else None

    print("Device:", DEVICE)
    model_id = args.model_id or MODEL_ID_BY_BACKEND[args.backend]
    if resume_ckpt is not None:
        if resume_ckpt.get("backend") not in {None, args.backend}:
            raise ValueError(
                f"--backend does not match checkpoint: cli={args.backend!r}, checkpoint={resume_ckpt.get('backend')!r}"
            )
        checkpoint_model_id = resume_ckpt.get("model_id")
        if args.model_id and checkpoint_model_id and args.model_id != checkpoint_model_id:
            raise ValueError(
                f"--model_id does not match checkpoint: cli={args.model_id!r}, checkpoint={checkpoint_model_id!r}"
            )
        model_id = str(checkpoint_model_id or model_id)
        checkpoint_revision = (
            resume_ckpt.get("reproducibility", {}).get("model", {}).get("resolved_revision")
            or resume_ckpt.get("config", {}).get("model_revision")
        )
        if args.model_revision and checkpoint_revision and args.model_revision != checkpoint_revision:
            raise ValueError(
                "--model_revision does not match checkpoint: "
                f"cli={args.model_revision!r}, checkpoint={checkpoint_revision!r}"
            )
        args.model_revision = args.model_revision or checkpoint_revision
        run_id = str(resume_ckpt.get("run_id", run_id))
        validate_resume_config(
            resume_ckpt.get("config"),
            vars(args),
            ["backend", "seed", "lora_r", "lora_alpha", "lora_targets"],
        )
    clean_df = load_split_csv(
        args.clean_csv,
        args.data_root,
        default_severity="clean",
        check_paths=not args.skip_path_check,
    )
    weak_df = load_split_csv(
        args.artefact_csv,
        args.data_root,
        default_severity="weak",
        check_paths=not args.skip_path_check,
    )
    clean_df["severity_norm"] = "clean"
    weak_df["severity_norm"] = "weak"

    train_df = pd.concat(
        [
            clean_df[clean_df["use_type_norm"].eq("train")],
            weak_df[weak_df["use_type_norm"].eq("train")],
        ],
        ignore_index=True,
    )
    val_df = pd.concat(
        [
            clean_df[clean_df["use_type_norm"].eq("val")],
            weak_df[weak_df["use_type_norm"].eq("val")],
        ],
        ignore_index=True,
    )
    print("train_df shape:", train_df.shape, "val_df shape:", val_df.shape)

    base_model, processor = load_backend(args.backend, model_id, args)
    repro = build_repro_metadata(
        script_path=__file__,
        args=args,
        input_files=[args.clean_csv, args.artefact_csv],
        model_id=model_id,
        requested_revision=args.model_revision,
        resolved_revision=resolved_model_revision(base_model, args.model_revision),
        prompts={"system": SYSTEM_PROMPT_SHORT, "by_dataset": PROMPT_BY_DATASET},
    )
    validate_generative_model(base_model, args.backend)
    print("base_model device:", next(base_model.parameters()).device, flush=True)
    lora_targets = parse_lora_targets(args.lora_targets)
    lora_module_names = inject_lora_adapters(
        base_model,
        target_names=lora_targets,
        r=int(args.lora_r),
        alpha=float(args.lora_alpha),
        dropout=float(args.lora_dropout),
    )
    configure_memory_saving(base_model, gradient_checkpointing=not bool(args.no_gradient_checkpointing))

    train_max_weaks = int(args.max_train_weaks) if args.max_train_weaks is not None else int(args.max_weaks)
    val_max_weaks = int(args.max_val_weaks) if args.max_val_weaks is not None else int(args.max_weaks)
    print(
        "[memory] effective images per group: "
        f"train<=1+{train_max_weaks}, val<=1+{val_max_weaks}; "
        "lower --max_train_weaks if CUDA OOM persists."
    )

    train_ds = CleanMultiWeakDataset(
        train_df,
        PROMPT_BY_DATASET,
        SYSTEM_PROMPT_SHORT,
        max_weaks=train_max_weaks,
        normal_text=args.normal_text,
        disease_text=args.disease_text,
        random_sample_weaks=True,
    )
    val_ds = CleanMultiWeakDataset(
        val_df,
        PROMPT_BY_DATASET,
        SYSTEM_PROMPT_SHORT,
        max_weaks=val_max_weaks,
        normal_text=args.normal_text,
        disease_text=args.disease_text,
        random_sample_weaks=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.bs),
        shuffle=True,
        collate_fn=make_generative_collate_fn(processor, args.backend, include_prompt_inputs=False),
        pin_memory=(DEVICE == "cuda"),
        generator=generator,
        num_workers=int(args.num_workers),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.bs),
        shuffle=False,
        collate_fn=make_generative_collate_fn(processor, args.backend, include_prompt_inputs=True),
        pin_memory=(DEVICE == "cuda"),
        generator=generator,
        num_workers=int(args.num_workers),
        worker_init_fn=seed_worker,
    )

    lora_params = lora_trainable_parameters(base_model)
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found.")
    optimizer = AdamW(lora_params, lr=float(args.lr))

    metrics_path = results_dir / f"{run_id}_{args.backend}_{args.tag}_lora_gen_metrics.json"

    start_epoch = 0
    best_val_f1 = -1.0
    epoch_logs: List[Dict[str, Any]] = []

    if args.resume:
        ckpt = resume_ckpt
        if isinstance(ckpt.get("lora"), dict):
            missing, unexpected = base_model.load_state_dict(ckpt["lora"], strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected LoRA checkpoint keys: {unexpected}")
            if len(ckpt["lora"]) == 0:
                raise RuntimeError("LoRA checkpoint has no parameters.")
        if isinstance(ckpt.get("optimizer"), dict):
            optimizer.load_state_dict(ckpt["optimizer"])
        restore_rng_state(ckpt.get("rng_state"), generator)
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        epoch_logs = ckpt.get("epoch_logs", [])
        run_id = ckpt.get("run_id", run_id)
        metrics_path = results_dir / f"{run_id}_{args.backend}_{args.tag}_lora_gen_metrics.json"
        print(f"[resume] Loaded {last_ckpt} | start_epoch={start_epoch} | best_val_f1={best_val_f1:.4f}")
    else:
        print("[resume] Start fresh.")

    for epoch in range(start_epoch, int(args.epochs)):
        train_m = run_epoch(
            base_model,
            processor,
            train_loader,
            optimizer=optimizer,
            args=args,
            run_generation=False,
        )
        run_gen = int(args.eval_generate_every) > 0 and ((epoch + 1) % int(args.eval_generate_every) == 0)
        val_m = run_epoch(
            base_model,
            processor,
            val_loader,
            optimizer=None,
            args=args,
            run_generation=run_gen,
        )

        val_overall = val_m.get("overall", {})
        val_clean = val_m.get("clean", {})
        val_weak = val_m.get("weak", {})
        val_f1 = float(val_overall.get("f1", float("nan")))
        val_acc = float(val_overall.get("acc", float("nan")))
        invalid_rate = float(val_overall.get("invalid_rate", float("nan")))

        print(
            f"[{args.backend}] Epoch {epoch + 1}/{args.epochs}\n"
            f"Train: loss={train_m['loss']:.4f} n={train_m['n_examples']}\n"
            f"Val  : loss={val_m['loss']:.4f} "
            f"GenF1={val_f1:.3f} Acc={val_acc:.3f} Invalid={invalid_rate:.3f} "
            f"CleanF1={float(val_clean.get('f1', float('nan'))):.3f} "
            f"WeakF1={float(val_weak.get('f1', float('nan'))):.3f}"
        )

        epoch_logs.append({"epoch": epoch + 1, "train": train_m, "val": val_m})
        if np.isfinite(val_f1) and val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            torch_save_atomic(
                {
                    "run_id": run_id,
                    "epoch": epoch + 1,
                    "lora": collect_lora_state(base_model),
                    "backend": args.backend,
                    "model_id": model_id,
                    "lora_targets": lora_targets,
                    "lora_module_names": lora_module_names,
                    "best_val_f1": best_val_f1,
                    "config": vars(args),
                    "reproducibility": repro,
                    "rng_state": capture_rng_state(generator),
                },
                best_ckpt,
            )
            print(f"BEST saved: {best_ckpt} (GenF1={best_val_f1:.4f})")

        payload = {
            "run_id": run_id,
            "backend": args.backend,
            "model_id": model_id,
            "lora_targets": lora_targets,
            "lora_module_names": lora_module_names,
            "tag": args.tag,
            "best_val_f1": best_val_f1,
            "logs": epoch_logs,
            "config": vars(args),
            "reproducibility": repro,
        }
        save_json_atomic(str(metrics_path), payload)

        torch_save_atomic(
            {
                "run_id": run_id,
                "epoch": epoch + 1,
                "lora": collect_lora_state(base_model),
                "optimizer": optimizer.state_dict(),
                "backend": args.backend,
                "model_id": model_id,
                "lora_targets": lora_targets,
                "lora_module_names": lora_module_names,
                "best_val_f1": best_val_f1,
                "epoch_logs": epoch_logs,
                "config": vars(args),
                "reproducibility": repro,
                "rng_state": capture_rng_state(generator),
            },
            last_ckpt,
        )
        print(f"[metrics] wrote {metrics_path} | epochs_logged={len(epoch_logs)}")

    del base_model, processor, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
