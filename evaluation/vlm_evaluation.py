import os
import gc
import json
import argparse
import re
import random
from typing import Optional, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# EffNet
import torchvision.transforms as T
import torchvision.models as tvm

from sklearn.metrics import roc_auc_score

from repro_utils import (
    add_reproducibility_args,
    build_repro_metadata,
    configure_hf_cache,
    hf_revision_kwargs,
    prepare_dataframe_paths,
    resolved_model_revision,
    save_json_atomic,
    seed_everything as seed_everything_portable,
    seed_worker,
)

BACKEND_ALIASES = {
    "qwen3.5": "qwen35",
    "qwen3.5-9b": "qwen35",
    "qwen3_5": "qwen35",
    "qwen_3.5": "qwen35",
    "qwen_3_5": "qwen35",
    "qwen35-9b": "qwen35",
    "qwen35_9b": "qwen35",
    "qwen3.6": "qwen36",
    "qwen3.6-27b": "qwen36",
    "qwen3_6": "qwen36",
    "qwen_3.6": "qwen36",
    "qwen_3_6": "qwen36",
    "qwen36_27b": "qwen36",
    "lingshu-i": "lingshu_i",
    "lingshu-i-8b": "lingshu_i",
    "lingshu_i8b": "lingshu_i",
    "lingshu_i-8b": "lingshu_i",
}
CANONICAL_BACKENDS = ["qwen3", "qwen35", "qwen36", "medgemma", "internvl", "lingshu", "lingshu_i"]
BACKEND_CHOICES = sorted(set(CANONICAL_BACKENDS + list(BACKEND_ALIASES)))
QWEN_STYLE_BACKENDS = {"qwen3", "qwen35", "qwen36", "lingshu", "lingshu_i"}
QWEN_THINKING_BACKENDS = {"qwen3", "qwen35", "qwen36"}
QWEN_MIN_NEW_TOKENS = 24
AUTO_IMAGE_TEXT_BACKENDS = {"qwen35", "qwen36", "lingshu_i"}
TRUST_REMOTE_CODE_BACKENDS = {"internvl", "medgemma", "lingshu_i", "qwen35", "qwen36"}


def normalize_backend(backend: str) -> str:
    backend = str(backend).lower()
    return BACKEND_ALIASES.get(backend, backend)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="single CSV containing both clean and artefact/weak samples (uses the severity column)",
    )
    p.add_argument("--clean_csv", type=str, default=None,
                   help="CSV containing clean samples (legacy two-CSV input)")
    p.add_argument("--artefact_csv", type=str, default=None,
                   help="CSV containing artefact/weak samples (legacy two-CSV input)")

    p.add_argument("--out_json", type=str, required=True, help="output json path")
    p.add_argument("--thr", type=float, default=0.5, help="threshold for Acc/F1 + flip rate")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducible evaluation")

    # Run mode flags
    p.add_argument("--run_vlm", action="store_true", help="evaluate VLM adapter+classifier")
    p.add_argument("--run_vlm_baseline", action="store_true", help="evaluate VLM prompt baseline (no ckpt)")
    p.add_argument("--run_effnet", action="store_true", help="evaluate EfficientNet classifier")

    # VLM settings
    p.add_argument("--backend", type=normalize_backend, default=None, choices=BACKEND_CHOICES)
    p.add_argument("--layer", type=str, default="last", choices=["last", "hs_-4"])
    p.add_argument("--vlm_ckpt", type=str, default=None,
                   help="path to VLM best.pt (supports adapter_train / adapter_train_clean_only style ckpts)")

    p.add_argument("--adapter_on_clean", action="store_true",
                   help="apply adapter to clean images too (recommended for fair evaluation)")
    p.add_argument(
        "--baseline_max_new_tokens", type=int, default=24,
        help=f"max_new_tokens for prompt baseline generation; Qwen3-family backends use at least {QWEN_MIN_NEW_TOKENS}",
    )

    # EfficientNet settings
    p.add_argument("--eff_ckpt", type=str, default=None, help="path to EffNet best.pt (contains model state)")
    p.add_argument("--eff_variant", type=str, default="effv2_l",
                   choices=["effv2_s", "effv2_m", "effv2_l",
                            "eff_b0", "eff_b1", "eff_b2", "eff_b3", "eff_b4", "eff_b5", "eff_b6", "eff_b7"])
    p.add_argument("--eff_img_size", type=int, default=224, help="EffNet eval resize")
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing output files")
    add_reproducibility_args(p)
    return p.parse_args()


args = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int, *, deterministic: bool):
    return seed_everything_portable(seed, deterministic=deterministic)


def load_trusted_checkpoint(path: str):
    # PyTorch 2.6 changed torch.load's default weights_only to True, which can
    # reject older checkpoints containing numpy/Python objects in metadata.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# -------------------------
# Prompts (VLM)
# -------------------------
# PROMPT_BY_DATASET = {
#     "mri": (
#         "This is a brain MRI scan.\n"
#         "Question: Does this image show normal anatomy or signs of disease?\n"
#         "Answer:"
#     ),
#     "oct": (
#         "This is a retinal OCT scan.\n"
#         "Question: Does this image show normal anatomy or signs of disease?\n"
#         "Answer:"
#     ),
#     "xray": (
#         "This is a chest X-ray image.\n"
#         "Question: Does this image show normal anatomy or signs of disease?\n"
#         "Answer:"
#     ),
#     "fundus": (
#         "This is a retinal fundus photograph.\n"
#         "Question: Does this image show normal anatomy or signs of disease?\n"
#         "Answer:"
#     ),
# }

PROMPT_BY_DATASET = {
    "mri": (
        "This is a brain MRI scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
        "Answer:"
    ),
    "oct": (
        "This is a retinal OCT scan.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
        "Answer:"
    ),
    "xray": (
        "This is a chest X-ray image.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
        "Answer:"
    ),
    "fundus": (
        "This is a retinal fundus photograph.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n"
        "Answer:"
    ),
}

PROMPT_MODALITY_ALIASES = {
    "retina": "fundus",
    "color fundus": "fundus",
    "colour fundus": "fundus",
}


def get_task_prompt(modality: str) -> str:
    prompt_modality = PROMPT_MODALITY_ALIASES.get(str(modality).lower(), str(modality).lower())
    return PROMPT_BY_DATASET.get(
        prompt_modality,
        "This is a medical image.\n"
        "Question: Does this image show normal anatomy or signs of disease?\n\n",
    )


SYSTEM_PROMPT_SHORT = (
    "Output MUST be exactly one word: normal or disease.\n"
    "Do not output punctuation, quotes, JSON, or any extra words."
)


MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "qwen35":   "Qwen/Qwen3.5-9B",
    "qwen36":   "Qwen/Qwen3.6-27B",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
    "lingshu_i": "lingshu-medical-mllm/Lingshu-I-8B",
}

# -------------------------
# Common utils
# -------------------------
def compute_binary_metrics_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().cpu().to(torch.int64)
    y_pred = y_pred.detach().cpu().to(torch.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("paths", "path", "meta", "severities", "modalities", "fileindices", "corruptions"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out

    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)

    return x

def load_split_csv(
    path: str,
    data_root: Optional[str],
    default_severity: Optional[str] = None,
    *,
    check_paths: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # label column name normalize
    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    # use_type == test only (both csv)
    if "use_type" in df.columns:
        df["use_type_norm"] = df["use_type"].astype(str).str.lower()
        df = df[df["use_type_norm"] == "test"].copy()
    else:
        # use_type 컬럼 없으면 그냥 전체 사용 (원하면 여기서 에러로 바꿔도 됨)
        df["use_type_norm"] = "unknown"

    # dataset normalize
    if "dataset" in df.columns:
        df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    else:
        # dataset 없으면 modality 계산이 안되니까 에러로 두는 게 안전
        raise ValueError(f"{path} needs dataset column")

    df = prepare_dataframe_paths(
        df,
        csv_path=path,
        data_root=data_root,
        check_exists=check_paths,
    )

    # fileindex required
    if "fileindex" not in df.columns:
        raise ValueError(f"{path} needs fileindex column")

    # severity normalize: 없으면 default_severity로 채움
    if "severity" in df.columns:
        df["severity_norm"] = df["severity"].fillna(default_severity or "clean").astype(str).str.lower()
    else:
        if default_severity is None:
            raise ValueError(f"{path} needs severity column or pass default_severity")
        df["severity_norm"] = str(default_severity).lower()

    # corruption normalize: 없으면 unknown
    if "corruption" in df.columns:
        c = df["corruption"]
    elif "corrupt_detail" in df.columns:
        c = df["corrupt_detail"]
    else:
        c = pd.Series(["unknown"] * len(df), index=df.index)
    df["corruption_norm"] = c.fillna("unknown").astype(str).str.lower()

    return df.reset_index(drop=True)

def count_valid_pairs(df_pair: pd.DataFrame) -> int:
    n = 0
    for _, g in df_pair.groupby(["dataset_norm", "fileindex"], sort=False):
        sev = g["severity_norm"].astype(str).str.lower()

        n_clean = int((sev == "clean").sum())
        n_weak = int((sev == "weak").sum())

        # flexible condition: clean 1 + weak >= 1
        if n_clean == 1 and n_weak >= 1:
            n += 1

    return int(n)


# =============================================================================
# VLM PART
# =============================================================================
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up   = nn.Linear(bottleneck, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.ln(h)
        delta = self.up(self.drop(self.act(self.down(x))))
        g = torch.sigmoid(self.gate)
        return h + g * delta


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_projection = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x.to(dtype=self.query_projection.weight.dtype, device=self.query_projection.weight.device)
        attn_logits = self.query_projection(x)
        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=x.dtype)
            attn_logits = attn_logits.masked_fill(m == 0, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled, attn_weights


class VLMAdapterWrapper(nn.Module):
    def __init__(
        self,
        base_model,
        backend: str,
        layer_choice: str = "last",
        adapter_bottleneck: int = 512,
    ):
        super().__init__()
        self.base_model = base_model
        self.backend = backend
        self.layer_choice = layer_choice

        for p in self.base_model.parameters():
            p.requires_grad = False

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim
        self.pooler = AttentionPooling(hidden_dim).to(device=embed_device, dtype=torch.float32)
        self.adapter = VisualAdapter(
            hidden_dim,
            bottleneck=int(adapter_bottleneck),
        ).to(device=embed_device, dtype=torch.float32)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=torch.float32)
        self.use_attention_pooling = False

        self.base_model.eval()

    def _get_text_embeddings(self, m: nn.Module) -> nn.Module:
        if hasattr(m, "get_input_embeddings") and callable(m.get_input_embeddings):
            emb = m.get_input_embeddings()
            if emb is not None:
                return emb

        for attr in ["language_model", "lm", "model", "text_model"]:
            if hasattr(m, attr):
                sub = getattr(m, attr)
                if hasattr(sub, "get_input_embeddings") and callable(sub.get_input_embeddings):
                    emb = sub.get_input_embeddings()
                    if emb is not None:
                        return emb

        candidates = [
            ("language_model", "model", "embed_tokens"),
            ("model", "embed_tokens"),
            ("text_model", "embed_tokens"),
            ("language_model", "embed_tokens"),
        ]
        for path in candidates:
            cur: Any = m
            ok = True
            for p in path:
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, nn.Module) and hasattr(cur, "weight"):
                return cur

        raise ValueError("Cannot locate text input embeddings to infer hidden_dim.")

    def extract_features(self, outputs, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        H = self.hidden_dim

        def get(obj, name: str):
            if obj is None:
                return None
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
            if isinstance(obj, dict) and name in obj and obj[name] is not None:
                return obj[name]
            return None

        inner = get(outputs, "model_output")

        img_hs = get(outputs, "image_hidden_states")
        if img_hs is None and inner is not None:
            img_hs = get(inner, "image_hidden_states")

        hs = get(outputs, "hidden_states")
        if hs is None:
            hs = get(outputs, "decoder_hidden_states")
        if hs is None and inner is not None:
            hs = get(inner, "hidden_states")
        if hs is None and inner is not None:
            hs = get(inner, "decoder_hidden_states")

        t  = get(outputs, "last_hidden_state")
        if t is None:
            t = get(outputs, "decoder_last_hidden_state")
        if t is None and inner is not None:
            t = get(inner, "last_hidden_state")
        if t is None and inner is not None:
            t = get(inner, "decoder_last_hidden_state")

        x = None
        used_image_hidden = False

        if torch.is_tensor(img_hs) and img_hs.dim() == 3 and img_hs.shape[-1] == H:
            x = img_hs
            used_image_hidden = True
        elif isinstance(img_hs, (list, tuple)) and len(img_hs) > 0:
            cand = img_hs[-1]
            if torch.is_tensor(cand) and cand.dim() == 3 and cand.shape[-1] == H:
                x = cand
                used_image_hidden = True

        if x is None:
            if self.layer_choice == "hs_-4":
                if isinstance(hs, (list, tuple)) and len(hs) >= 5 and torch.is_tensor(hs[-4]) and hs[-4].dim() == 3 and hs[-4].shape[-1] == H:
                    x = hs[-4]
                elif torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                    x = t
                else:
                    raise RuntimeError(f"{self.backend}: cannot use hs_-4")
            else:
                if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]) and hs[-1].dim() == 3 and hs[-1].shape[-1] == H:
                    x = hs[-1]
                elif torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
                    x = t
                else:
                    raise RuntimeError(f"{self.backend}: cannot find usable hidden state")

        if self.use_attention_pooling:
            pooled, _ = self.pooler(x, attention_mask if not used_image_hidden else None)
        else:
            if used_image_hidden or attention_mask is None or attention_mask.dim() != 2:
                pooled = x.mean(dim=1)
            else:
                Tm = min(x.shape[1], attention_mask.shape[1])
                x = x[:, :Tm, :]
                m = attention_mask[:, :Tm].to(dtype=x.dtype, device=x.device).unsqueeze(-1)
                x_sum = (x * m).sum(dim=1)
                denom = m.sum(dim=1).clamp_min(1.0)
                pooled = x_sum / denom

        if self.backend in ["internvl", "lingshu_i"] and attention_mask is not None and pooled.dim() == 2:
            batch_size = attention_mask.shape[0]
            if pooled.shape[0] != batch_size:
                if pooled.shape[0] % batch_size == 0:
                    n_tiles = pooled.shape[0] // batch_size
                    pooled = pooled.view(batch_size, n_tiles, pooled.shape[-1]).mean(dim=1)
                else:
                    raise RuntimeError(
                        f"{self.backend} pooled feature count mismatch: pooled.shape={tuple(pooled.shape)}, "
                        f"batch_size={batch_size}"
                    )

        return pooled


def _coerce_pred_int(p: Any, y_true: int) -> int:
    """
    Parse 실패(None/기타)면 '무조건 오답'이 되도록 강제로 예측 라벨을 채움.
    - p in {0,1}이면 그대로
    - 그 외는 1-y_true 로 채워서 항상 틀리게 만듦
    """
    if p in (0, 1):
        return int(p)
    return 1 - int(y_true)

def load_backend(backend: str, model_id: str, model_revision: Optional[str]):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["medgemma", "internvl", "qwen35", "qwen36", "lingshu_i"] and device == "cuda":
        torch_dtype = torch.bfloat16

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None

    revision_kwargs = hf_revision_kwargs(model_revision)
    common = dict(
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        **revision_kwargs,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, **revision_kwargs)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
        return model, processor

    if backend in AUTO_IMAGE_TEXT_BACKENDS:
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is None:
            raise RuntimeError(
                f"{backend} requires transformers.AutoModelForImageTextToText. "
                "Please update transformers."
            )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )
        model = AutoImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **common
        )
        model.to(device); model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, **revision_kwargs)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, **common)
        img_seq_len = getattr(getattr(model, "config", None), "image_seq_length", None)
        if img_seq_len is not None:
            for attr in ["num_image_tokens", "num_image_token", "image_seq_len", "image_seq_length"]:
                if hasattr(processor, attr):
                    try:
                        setattr(processor, attr, int(img_seq_len))
                    except Exception:
                        pass

            tok = getattr(processor, "tokenizer", None)
            if tok is not None:
                for attr in ["num_image_tokens", "num_image_token", "image_seq_len", "image_seq_length"]:
                    if hasattr(tok, attr):
                        try:
                            setattr(tok, attr, int(img_seq_len))
                        except Exception:
                            pass

            print(f"[internvl] processor image tokens set to image_seq_length={int(img_seq_len)}")
        model.to(device); model.eval()
        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, **revision_kwargs)
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device); model.eval()
            print("[medgemma] loaded class:", type(model).__name__)
            print("[medgemma] has generate:", hasattr(model, "generate"))
            return model, processor

        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, cache_dir=cache_dir, **common)
            model.to(device); model.eval()
            return model, processor

        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, **common)
        model.to(device); model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


def load_backend_for_prompt(backend: str, model_id: str, model_revision: Optional[str]):
    """
    Prompt baseline loader.
    For InternVL, use a generation-capable class (AutoModelForImageTextToText),
    because AutoModel(InternVLModel) may not expose .generate().
    """
    if backend != "internvl":
        return load_backend(backend, model_id, model_revision)

    from transformers import AutoProcessor, AutoModelForImageTextToText

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch_dtype = torch.bfloat16

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None
    revision_kwargs = hf_revision_kwargs(model_revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        **revision_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
    )
    model.to(device)
    model.eval()
    return model, processor


def _parse_normal_disease(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    if not t:
        return None

    # 1) JSON-first parsing: {"label":"normal|disease"}
    m_json = re.search(
        r'\{\s*"label"\s*:\s*"(normal|disease)"\s*\}',
        t,
        flags=re.IGNORECASE,
    )
    if m_json:
        lbl = m_json.group(1).lower()
        return 0 if lbl == "normal" else 1

    # 1.5) Accept partial JSON when generation was cut:
    # e.g. {"label":"normal
    m_json_partial = re.search(
        r'"label"\s*:\s*"(normal|disease)\b',
        t,
        flags=re.IGNORECASE,
    )
    if m_json_partial:
        lbl = m_json_partial.group(1).lower()
        return 0 if lbl == "normal" else 1

    # Prefer the model's final answer span when prompt text is echoed.
    ans = t
    for marker in ["answer:", "assistant:"]:
        if marker in ans:
            ans = ans.split(marker)[-1].strip()

    # Evaluate only answer block before explanation tail.
    lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    answer_lines = []
    for ln in lines:
        if ln.startswith("explanation:"):
            break
        answer_lines.append(ln)
    if not answer_lines:
        answer_lines = lines if lines else [ans]

    block = " ".join(answer_lines).strip()
    if not block:
        return None

    # Some VLMs, especially MedGemma, answer with the label first and then keep
    # explaining despite the one-word instruction. In that case the leading
    # label is the prediction; later label words may be part of the explanation.
    m_leading_label = re.match(r'^\s*["\']?\s*(normal|disease)\b', block)
    if m_leading_label:
        return 0 if m_leading_label.group(1) == "normal" else 1

    # Handle explicit negation phrases before plain token matching.
    if re.search(r"\bnot\s+normal\b", block):
        return 1
    if re.search(r"\bno\s+disease\b", block):
        return 0

    # If both labels appear in answer block, treat as ambiguous.
    has_normal_block = re.search(r"\bnormal\b", block) is not None
    has_disease_block = re.search(r"\bdisease\b", block) is not None
    if has_normal_block and has_disease_block:
        return None

    # First valid answer line wins (e.g., "Answer: normal" then "Explanation: ...").
    for ln in answer_lines:
        m_exact = re.fullmatch(r'["\']?\s*(normal|disease)\s*["\']?[.!?]?\s*', ln)
        if m_exact:
            return 0 if m_exact.group(1) == "normal" else 1
        if re.search(r"\bnormal\b", ln) is not None:
            return 0
        if re.search(r"\bdisease\b", ln) is not None:
            return 1

    # Fallback on answer block
    if has_normal_block:
        return 0
    if has_disease_block:
        return 1
    return None


@torch.no_grad()
def forward_vlm_prompt_pred_one(
    base_model,
    processor,
    backend: str,
    img: Image.Image,
    text: str,
    device: str,
    max_new_tokens: int = 8,
) -> Tuple[Optional[int], str]:
    if not hasattr(base_model, "generate"):
        raise RuntimeError(
            f"{backend} model class does not expose generate(); "
            "use a generation-capable loader for prompt baseline."
        )

    def _apply_chat_template(messages, *, add_generation_prompt: bool) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        if backend in QWEN_THINKING_BACKENDS:
            try:
                return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
            except TypeError as e:
                if "enable_thinking" not in str(e):
                    raise
        return processor.apply_chat_template(messages, **kwargs)

    def _render_prompt_text(base_text: str, add_generation_prompt: bool = True) -> str:
        if backend in QWEN_STYLE_BACKENDS:
            messages = [[{
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": base_text}],
            }]]
            return _apply_chat_template(messages[0], add_generation_prompt=add_generation_prompt)
        if backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            return f"{image_tok}\n{base_text}"
        if backend == "medgemma":
            messages = [[{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": base_text}],
            }]]
            return processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=add_generation_prompt)
        return base_text

    def _build_inputs_from_rendered(rendered_text: str) -> Dict[str, Any]:
        model_inputs = processor(text=[rendered_text], images=[img], padding=True, return_tensors="pt")
        out: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    prompt_rendered = _render_prompt_text(text, add_generation_prompt=True)
    inputs = _build_inputs_from_rendered(prompt_rendered)

    # MedGemma may start with markdown/code-fence + JSON, so very short generation
    # can truncate at '{"label":"'. Keep a backend-specific minimum budget.
    gen_max_new_tokens = int(max_new_tokens)
    if backend in QWEN_THINKING_BACKENDS:
        gen_max_new_tokens = max(gen_max_new_tokens, QWEN_MIN_NEW_TOKENS)
    elif backend == "medgemma":
        gen_max_new_tokens = min(gen_max_new_tokens, 4)

    tok = getattr(processor, "tokenizer", None)
    gen_kwargs = {
        "max_new_tokens": gen_max_new_tokens,
        "do_sample": False,
    }
    if tok is not None:
        eos_token_id = getattr(tok, "eos_token_id", None)
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id

    seq = base_model.generate(**inputs, **gen_kwargs)
    decoded = processor.batch_decode(seq, skip_special_tokens=True)[0]

    # Parse only newly generated continuation when possible (avoid prompt-echo leakage).
    decoded_new = ""
    in_ids = inputs.get("input_ids", None)
    if torch.is_tensor(in_ids) and torch.is_tensor(seq) and seq.dim() == 2 and in_ids.dim() == 2:
        prompt_len = int(in_ids.shape[1])
        if seq.shape[1] > prompt_len:
            new_tok = seq[:, prompt_len:]
            decoded_new = processor.batch_decode(new_tok, skip_special_tokens=True)[0]

    parse_text = decoded_new if (decoded_new or "").strip() else decoded
    pred = _parse_normal_disease(parse_text)
    return pred, parse_text.strip()


def _to_nullable_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _extract_best_threshold_from_ckpt(ckpt: Dict[str, Any]) -> Optional[float]:
    if not isinstance(ckpt, dict):
        return None

    def _as_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
        except Exception:
            return None
        if not np.isfinite(f):
            return None
        return f

    # 1) Common top-level keys
    for k in ["best_thr", "best_val_thr"]:
        if k in ckpt:
            out = _as_float(ckpt.get(k))
            if out is not None:
                return out

    # 2) adapter_train.py BEST_CKPT layout
    vo = ckpt.get("val_overall_best")
    if isinstance(vo, dict):
        out = _as_float(vo.get("best_thr"))
        if out is not None:
            return out

    return None


def _infer_adapter_bottleneck_from_ckpt(ckpt: Dict[str, Any], fallback: int = 512) -> int:
    if not isinstance(ckpt, dict):
        return int(fallback)

    adapter_sd = ckpt.get("adapter")
    if not isinstance(adapter_sd, dict):
        return int(fallback)

    # Allow eval to reconstruct adapters trained with non-default bottlenecks.
    down_w = adapter_sd.get("down.weight")
    if torch.is_tensor(down_w) and down_w.ndim == 2 and int(down_w.shape[0]) > 0:
        return int(down_w.shape[0])

    return int(fallback)


def _extract_layer_choice_from_ckpt(ckpt: Dict[str, Any], fallback: str) -> str:
    if isinstance(ckpt, dict):
        lc = ckpt.get("layer_choice", None)
        if isinstance(lc, str) and lc in {"last", "hs_-4"}:
            return lc
    return fallback


def _is_adapter_train_ckpt(ckpt: Dict[str, Any]) -> bool:
    if not isinstance(ckpt, dict):
        return False
    return ("val_overall_best" in ckpt) or ("val_clean_best" in ckpt) or ("val_artefact_best" in ckpt)


def _infer_ckpt_train_mode(ckpt: Dict[str, Any]) -> Optional[str]:
    if not isinstance(ckpt, dict):
        return None

    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        mode = str(cfg.get("mode", "")).strip().lower()
        if mode:
            return mode

    if _is_adapter_train_ckpt(ckpt):
        return "paired"

    if "best_val_f1_overall" in ckpt:
        return "clean_only"
    if "best_val_acc" in ckpt:
        return "weak_only"
    if "best_val_auroc_overall" in ckpt:
        return "both"

    adapter_bottleneck = _infer_adapter_bottleneck_from_ckpt(ckpt, fallback=-1)
    if (
        adapter_bottleneck == 256
        and "best_val_thr" not in ckpt
        and "layer_choice" not in ckpt
        and "best_val_score" in ckpt
    ):
        return "clean_only"

    return None


def _should_auto_apply_adapter_on_clean(ckpt: Dict[str, Any]) -> bool:
    mode = _infer_ckpt_train_mode(ckpt)
    return mode in {"paired", "clean_only", "both"}


def _compute_balanced_accuracy_and_f1_from_optional_preds(
    y_true_list: List[int],
    y_pred_list: List[Optional[int]],
) -> Dict[str, Optional[float]]:
    if len(y_true_list) == 0:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None,
            "tp": None,
            "tn": None,
            "fp": None,
            "fn": None,
        }

    yt = [int(y) for y in y_true_list]
    yp = [_coerce_pred_int(p, y) for y, p in zip(yt, y_pred_list)]

    yt_t = torch.tensor(yt, dtype=torch.long)
    yp_t = torch.tensor(yp, dtype=torch.long)
    m = compute_binary_metrics_from_preds(yt_t, yp_t)

    tp = int(m["tp"]); tn = int(m["tn"]); fp = int(m["fp"]); fn = int(m["fn"])
    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (tpr + tnr)

    return {
        "accuracy": _to_nullable_float(acc),
        "balanced_accuracy": _to_nullable_float(bacc),
        "f1_score": _to_nullable_float(m.get("f1")),
        "precision": _to_nullable_float(m.get("precision")),
        "recall": _to_nullable_float(m.get("recall")),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

def _mean_optional(vals: List[Optional[float]]) -> Optional[float]:
    kept = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if len(kept) == 0:
        return None
    return _to_nullable_float(np.mean(kept))


def _compute_auroc_from_records(records: List[Dict[str, Any]]) -> Optional[float]:
    ys: List[int] = []
    ps: List[float] = []
    for r in records:
        if "y_true" not in r or "p_disease" not in r:
            continue
        yv = r.get("y_true")
        pv = r.get("p_disease")
        try:
            yi = int(yv)
            pf = float(pv)
        except Exception:
            continue
        if yi not in (0, 1):
            continue
        if not np.isfinite(pf):
            continue
        ys.append(yi)
        ps.append(pf)

    if len(ys) == 0:
        return None
    y_np = np.asarray(ys, dtype=np.int64)
    p_np = np.asarray(ps, dtype=np.float64)
    if np.unique(y_np).size < 2:
        return None
    try:
        return _to_nullable_float(float(roc_auc_score(y_np, p_np)))
    except Exception:
        return None


def build_records_from_prob_outputs(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    severities: List[str],
    modalities: List[str],
    paths: List[str],
    fileindices: List[str],
    thr: float,
    corruptions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    yt = y_true.detach().cpu().tolist()
    yp = y_prob.detach().cpu().tolist()
    eps = 1e-12
    for i in range(len(yt)):
        pd = float(np.clip(float(yp[i]), 0.0, 1.0))
        pn = float(1.0 - pd)
        p1 = float(np.clip(pd, eps, 1.0 - eps))
        p0 = float(np.clip(pn, eps, 1.0 - eps))
        ent = float(-(p0 * np.log(p0) + p1 * np.log(p1)))
        out.append({
            "path": str(paths[i]) if i < len(paths) else None,
            "fileindex": str(fileindices[i]) if i < len(fileindices) else None,
            "severity": str(severities[i]).lower() if i < len(severities) else None,
            "modality": str(modalities[i]).lower() if i < len(modalities) else None,
            "corruption": str(corruptions[i]).lower() if (corruptions is not None and i < len(corruptions)) else "unknown",
            "y_true": int(yt[i]),
            "pred_label": int(1 if pd >= thr else 0),
            "p_normal": pn,
            "p_disease": pd,
            "entropy": ent,
        })
    return out


def compute_metrics_from_records(records: List[Dict[str, Any]], pred_key: str = "pred_label") -> Dict[str, Any]:
    clean_recs = [r for r in records if str(r.get("severity", "")).lower() == "clean"]
    art_recs = [r for r in records if str(r.get("severity", "")).lower() == "weak"]

    y_clean = [int(r["y_true"]) for r in clean_recs]
    y_art = [int(r["y_true"]) for r in art_recs]
    pred_clean = [_coerce_pred_int(r.get(pred_key), int(r["y_true"])) for r in clean_recs]
    pred_art   = [_coerce_pred_int(r.get(pred_key), int(r["y_true"])) for r in art_recs]
    clean_lbl = _compute_balanced_accuracy_and_f1_from_optional_preds(y_clean, pred_clean)
    art_lbl = _compute_balanced_accuracy_and_f1_from_optional_preds(y_art, pred_art)
    clean_lbl["auroc"] = _compute_auroc_from_records(clean_recs)
    art_lbl["auroc"] = _compute_auroc_from_records(art_recs)
    clean_bacc = clean_lbl["balanced_accuracy"]
    art_bacc = art_lbl["balanced_accuracy"]
    performance_gap = None if (clean_bacc is None or art_bacc is None) else _to_nullable_float(clean_bacc - art_bacc)

    grouped: Dict[str, Dict[str, Any]] = {}
    for r in records:
        fid = str(r.get("fileindex"))
        pair_key = (str(r.get("modality", "unknown")).lower(), fid)
        if pair_key not in grouped:
            grouped[pair_key] = {"clean": None, "weak": []}
        sev = str(r.get("severity", "")).lower()
        if sev == "clean" and grouped[pair_key]["clean"] is None:
            grouped[pair_key]["clean"] = r
        elif sev == "weak":
            grouped[pair_key]["weak"].append(r)

    consistency_vals: List[float] = []
    all_same_vals: List[float] = []
    rel_num = 0
    rel_any_num = 0
    rel_den = 0
    cwc_num = 0
    cwc_den = 0
    flip_maj_num = 0
    flip_maj_den = 0
    aier_harmful_flip_count = 0
    aier_eligible_artefact_count = 0
    aier_clean_correct_count = 0
    for _, g in grouped.items():
        c = g["clean"]
        ws = g["weak"]
        if c is None or len(ws) == 0:
            continue

        # Impute parse fails as wrong (no group drop)
        y = int(c["y_true"])
        c_pred = _coerce_pred_int(c.get(pred_key), y)

        weak_preds = [_coerce_pred_int(w.get(pred_key), int(w["y_true"])) for w in ws]  # each weak has same y_true usually
        n0 = sum(1 for p in weak_preds if p == 0)
        n1 = len(weak_preds) - n0

        consistency_vals.append(float(max(n0, n1) / max(1, len(weak_preds))))
        all_same_vals.append(float(1.0 if (n0 == 0 or n1 == 0) else 0.0))
        maj_pred = 1 if n1 >= (len(weak_preds) / 2) else 0

        # robustness: compare majority(weak) vs clean pred
        cwc_den += 1
        if int(maj_pred) == int(c_pred):
            cwc_num += 1

        # flip majority vs clean pred
        flip_maj_den += 1
        if int(maj_pred) != int(c_pred):
            flip_maj_num += 1

        # reliability flip rate: only count when clean pred was correct (as before)
        if int(c_pred) == y:
            rel_den += 1
            aier_clean_correct_count += 1
            aier_eligible_artefact_count += len(weak_preds)
            aier_harmful_flip_count += sum(int(p) != y for p in weak_preds)
            if int(maj_pred) != y:
                rel_num += 1
            if any(int(p) != y for p in weak_preds):
                rel_any_num += 1


    prediction_consistency = _mean_optional([_to_nullable_float(v) for v in consistency_vals])
    all_same_rate = _mean_optional([_to_nullable_float(v) for v in all_same_vals])
    flip_rate = None if rel_den == 0 else _to_nullable_float(rel_num / rel_den)
    flip_rate_any = None if rel_den == 0 else _to_nullable_float(rel_any_num / rel_den)
    aier = (
        None
        if aier_eligible_artefact_count == 0
        else _to_nullable_float(aier_harmful_flip_count / aier_eligible_artefact_count)
    )
    consistency_with_clean = None if cwc_den == 0 else _to_nullable_float(cwc_num / cwc_den)
    flip_majority_vs_clean_pred = None if flip_maj_den == 0 else _to_nullable_float(flip_maj_num / flip_maj_den)

    return {
        "label_based_metrics": {
            "clean": clean_lbl,
            "artefact": {
                "accuracy": art_lbl["accuracy"],
                "balanced_accuracy": art_lbl["balanced_accuracy"],
                "f1_score": art_lbl["f1_score"],
                "precision": art_lbl["precision"],
                "recall": art_lbl["recall"],
                "auroc": art_lbl["auroc"],
                "tp": art_lbl["tp"],
                "tn": art_lbl["tn"],
                "fp": art_lbl["fp"],
                "fn": art_lbl["fn"],
                "performance_gap": performance_gap,
            },
            "robustness": {
                "prediction_consistency": prediction_consistency,
                "consistency_with_clean": consistency_with_clean,
                "all_same_rate": all_same_rate,
            },
            "reliability": {
                "aier": aier,
                "aier_harmful_flip_count": int(aier_harmful_flip_count),
                "aier_eligible_artefact_count": int(aier_eligible_artefact_count),
                "aier_clean_correct_count": int(aier_clean_correct_count),
                "flip_rate": flip_rate,
                "flip_rate_any": flip_rate_any,
                "flip_majority_vs_clean_pred": flip_majority_vs_clean_pred,
            },
        },
    }


def _pick_metrics(mm: Dict[str, Any]) -> Dict[str, Any]:
    lb = mm.get("label_based_metrics", {}) if isinstance(mm, dict) else {}
    clean = lb.get("clean", {}) if isinstance(lb, dict) else {}
    art = lb.get("artefact", {}) if isinstance(lb, dict) else {}
    robustness = lb.get("robustness", {}) if isinstance(lb, dict) else {}
    reliability = lb.get("reliability", {}) if isinstance(lb, dict) else {}
    return {
        "clean_accuracy": _to_nullable_float(clean.get("accuracy")),
        "clean_f1_score": _to_nullable_float(clean.get("f1_score")),
        "clean_precision": _to_nullable_float(clean.get("precision")),
        "clean_recall": _to_nullable_float(clean.get("recall")),
        "clean_auroc": _to_nullable_float(clean.get("auroc")),
        "clean_tp": clean.get("tp"),
        "clean_tn": clean.get("tn"),
        "clean_fp": clean.get("fp"),
        "clean_fn": clean.get("fn"),
        "artefact_accuracy": _to_nullable_float(art.get("accuracy")),
        "artefact_f1_score": _to_nullable_float(art.get("f1_score")),
        "artefact_precision": _to_nullable_float(art.get("precision")),
        "artefact_recall": _to_nullable_float(art.get("recall")),
        "artefact_auroc": _to_nullable_float(art.get("auroc")),
        "artefact_tp": art.get("tp"),
        "artefact_tn": art.get("tn"),
        "artefact_fp": art.get("fp"),
        "artefact_fn": art.get("fn"),
        "prediction_consistency": _to_nullable_float(robustness.get("prediction_consistency")),
        "consistency_with_clean": _to_nullable_float(robustness.get("consistency_with_clean")),
        "aier": _to_nullable_float(reliability.get("aier")),
        "aier_harmful_flip_count": reliability.get("aier_harmful_flip_count"),
        "aier_eligible_artefact_count": reliability.get("aier_eligible_artefact_count"),
        "aier_clean_correct_count": reliability.get("aier_clean_correct_count"),
        "flip_rate": _to_nullable_float(reliability.get("flip_rate")),
        "flip_rate_any": _to_nullable_float(reliability.get("flip_rate_any")),
    }


def compute_metrics_by_corruption(records: List[Dict[str, Any]], pred_key: str = "pred_label") -> Dict[str, Dict[str, Any]]:
    clean_records = [
        r for r in records if str(r.get("severity", "")).lower() == "clean"
    ]
    cors = sorted({
        str(r.get("corruption", "unknown")).lower()
        for r in records
        if str(r.get("severity", "")).lower() == "weak"
        if str(r.get("corruption", "unknown")).strip() != ""
    })
    out: Dict[str, Dict[str, Any]] = {}
    for c in cors:
        weak_records = [
            r
            for r in records
            if str(r.get("severity", "")).lower() == "weak"
            and str(r.get("corruption", "unknown")).lower() == c
        ]
        sub = clean_records + weak_records
        out[c] = _pick_metrics(compute_metrics_from_records(sub, pred_key=pred_key))
    return out


def compute_metrics_by_modality(records: List[Dict[str, Any]], pred_key: str = "pred_label") -> Dict[str, Dict[str, Any]]:

    mods = sorted({
        str(r.get("modality", "")).lower()
        for r in records
        if str(r.get("modality", "")).strip() != ""
    })
    out: Dict[str, Dict[str, Any]] = {}
    for m in mods:
        sub = [r for r in records if str(r.get("modality", "")).lower() == m]
        out[m] = _pick_metrics(compute_metrics_from_records(sub, pred_key=pred_key))
        out[m]["by_corruption"] = compute_metrics_by_corruption(sub, pred_key=pred_key)
    return out


def _build_modality_metrics_rows(metrics_by_modality: Dict[str, Any], model_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for modality, mod_metrics in metrics_by_modality.items():
        if not isinstance(mod_metrics, dict):
            continue
        for metric_name, value in mod_metrics.items():
            if metric_name == "by_corruption":
                continue
            rows.append({
                "model_key": str(model_key),
                "scope": "modality",
                "modality": str(modality),
                "group": "summary",
                "metric": str(metric_name),
                "value": value,
            })
        by_corruption = mod_metrics.get("by_corruption", {})
        if isinstance(by_corruption, dict):
            for corruption, cor_metrics in by_corruption.items():
                if not isinstance(cor_metrics, dict):
                    continue
                for metric_name, value in cor_metrics.items():
                    rows.append({
                        "model_key": str(model_key),
                        "scope": "modality_by_corruption",
                        "modality": str(modality),
                        "group": f"corruption:{str(corruption)}",
                        "metric": str(metric_name),
                        "value": value,
                    })
    return rows


def _sanitize_filename_piece(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def _save_csv_outputs(results: Dict[str, Any], out_dir: str, base_name: str) -> None:
    models = results.get("models", {})
    if not isinstance(models, dict):
        return

    for model_key, mr in models.items():
        if not isinstance(mr, dict):
            continue
        safe_key = _sanitize_filename_piece(str(model_key))
        prefix = os.path.join(out_dir, f"{base_name}__{safe_key}")

        per_image = mr.get("per_image", [])
        pd.DataFrame(per_image if isinstance(per_image, list) else []).to_csv(
            prefix + "__per_image.csv", index=False, encoding="utf-8"
        )

        rows_by_modality = _build_modality_metrics_rows(
            mr.get("metrics_by_modality", {}), model_key=str(model_key)
        )
        pd.DataFrame(rows_by_modality).to_csv(
            prefix + "__metrics_by_modality.csv", index=False, encoding="utf-8"
        )

        parse_diag = mr.get("parse_diagnostics", None)
        if isinstance(parse_diag, dict):
            pd.DataFrame([parse_diag]).to_csv(
                prefix + "__parse_diagnostics.csv", index=False, encoding="utf-8"
            )


def _write_results_outputs(results: Dict[str, Any], out_json: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_json))
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(out_json))[0]
    out_path = os.path.abspath(out_json)
    save_json_atomic(out_path, results)
    print(f"[saved] {out_path}", flush=True)
    _save_csv_outputs(results, out_dir=out_dir, base_name=base_name)
    print(f"[saved] csv files under {out_dir} (prefix: {base_name}__<model_key>)", flush=True)


def eval_prompt_baseline_metrics(
    base_model,
    processor,
    backend: str,
    df_all: pd.DataFrame,
    max_new_tokens: int,
    partial_csv_path: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[float]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    if partial_csv_path:
        partial_dir = os.path.dirname(os.path.abspath(partial_csv_path))
        os.makedirs(partial_dir, exist_ok=True)
        if os.path.exists(partial_csv_path):
            os.remove(partial_csv_path)

    def _append_partial_record(record: Dict[str, Any]) -> None:
        if not partial_csv_path:
            return
        pd.DataFrame([record]).to_csv(
            partial_csv_path,
            mode="a",
            header=not os.path.exists(partial_csv_path),
            index=False,
            encoding="utf-8",
        )

    # 밖에서 한번만
    checked_medgemma = False
    def _run_df(df_part: pd.DataFrame):
        nonlocal records, checked_medgemma
        for i in range(len(df_part)):
            row = df_part.iloc[i]
            img = Image.open(row["filepath"]).convert("RGB")
            modality = str(row["dataset_norm"]).lower()
            y_true = int(row["binarylabel"])
            severity = str(row["severity_norm"]).lower()
            corruption = str(row.get("corruption_norm", "unknown")).lower()
            path = str(row["filepath"])
            fileindex = str(row["fileindex"])

            print(
                f"[baseline] backend={backend} severity={severity} "
                f"{i + 1}/{len(df_part)} | fileindex={fileindex} | modality={modality}",
                flush=True,
            )

            task_prompt = get_task_prompt(modality)
            full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

            if backend == "medgemma" and not checked_medgemma:
                # tmp = processor(text=["test"], images=[img], return_tensors="pt", padding=True)
                # print("[medgemma] processor keys:", list(tmp.keys()))
                checked_medgemma = True

            pred, raw = forward_vlm_prompt_pred_one(
                base_model=base_model,
                processor=processor,
                backend=backend,
                img=img,
                text=full_text,
                device=device,
                max_new_tokens=int(max_new_tokens),
            )
            pred_parse = (int(pred) if pred in (0, 1) else None)

            record = {
                "path": path,
                "fileindex": fileindex,
                "severity": severity,
                "corruption": corruption,
                "modality": modality,
                "y_true": y_true,
                "pred_label_parse": pred_parse,
                "pred_label": pred_parse,
                "raw_text": str(raw),
            }
            records.append(record)
            _append_partial_record(record)

            raw_short = re.sub(r"\s+", " ", str(raw)).strip()
            if len(raw_short) > 120:
                raw_short = raw_short[:117] + "..."
            print(
                f"[baseline] done | pred={pred_parse} | raw={raw_short!r}",
                flush=True,
            )

    # Run the combined CSV in its original row order. Clean/weak are still
    # distinguished in the output and metrics by each row's severity value.
    _run_df(df_all)

    # modality별 parse(label-based)만 계산
    metrics_by_modality_parse = compute_metrics_by_modality(records, pred_key="pred_label_parse")

    n_total = int(len(records))
    n_parse_success = int(sum(1 for r in records if r.get("pred_label_parse") in (0, 1)))
    n_parse_fail = int(n_total - n_parse_success)
    parse_diagnostics = {
        "n_total": n_total,
        "n_parse_success": n_parse_success,
        "n_parse_fail": n_parse_fail,
        "parse_success_ratio": _to_nullable_float(n_parse_success / max(1, n_total)),
        "parse_fail_ratio": _to_nullable_float(n_parse_fail / max(1, n_total)),
    }

    return metrics_by_modality_parse, parse_diagnostics, records

class SingleImageDatasetVLM(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

        modality = str(row["dataset_norm"]).lower()
        label = int(row["binarylabel"])
        sev = str(row["severity_norm"]).lower()
        corruption = str(row.get("corruption_norm", "unknown")).lower()

        task_prompt = get_task_prompt(modality)
        full_text = SYSTEM_PROMPT_SHORT + "\n\n" + task_prompt

        return {
            "image": img,
            "input_text": full_text,
            "label": label,
            "path": img_path,
            "severity": sev,
            "fileindex": str(row["fileindex"]),
            "modality": modality,
            "corruption": corruption,
        }



def make_single_collate_fn_vlm(processor, backend: str):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts  = [b["input_text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        paths  = [b["path"] for b in batch]
        sevs   = [b["severity"] for b in batch]
        fids = [str(b["fileindex"]) for b in batch]
        mods = [b["modality"] for b in batch]
        cors = [b.get("corruption", "unknown") for b in batch]


        if backend in QWEN_STYLE_BACKENDS:
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                         for m in messages_list]
            model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
            model_inputs = processor(text=inline_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "medgemma":
            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError("medgemma needs apply_chat_template")
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                         for m in messages_list]
            model_inputs = processor(text=chat_texts, images=images, padding=True, return_tensors="pt")
        else:
            model_inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

        out = dict(model_inputs)
        out["labels_cls"] = labels
        out["paths"] = paths
        out["severities"] = sevs
        out["fileindices"] = fids
        out["modalities"] = mods
        out["corruptions"] = cors
        return out
    return collate


@torch.no_grad()
def forward_vlm_prob(vlm: VLMAdapterWrapper, batch: Dict[str, Any], apply_adapter: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    target_device = next(vlm.adapter.parameters()).device
    batch = to_device(batch, target_device)

    y = batch["labels_cls"]
    d = dict(batch)
    for k in ["labels_cls", "paths", "severities", "fileindices", "labels_token", "modalities", "corruptions"]:
        d.pop(k, None)

    if vlm.backend in ["internvl", "lingshu_i"] and device == "cuda":
        for k in ["pixel_values", "images", "image", "vision_x"]:
            if k in d and torch.is_tensor(d[k]):
                d[k] = d[k].to(dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            outputs = vlm.base_model(**d, output_hidden_states=True, return_dict=True)
    else:
        outputs = vlm.base_model(**d, output_hidden_states=True, return_dict=True)

    attn = d.get("attention_mask", None)
    h_base = vlm.extract_features(outputs, attention_mask=attn).float()
    h_base = F.layer_norm(h_base, (h_base.shape[-1],))

    h = vlm.adapter(h_base) if apply_adapter else h_base
    logits = vlm.classifier(h)
    prob = torch.softmax(logits.float(), dim=1)[:, 1]
    return y.detach().cpu(), prob.detach().cpu()


# =============================================================================
# EFFNET PART
# =============================================================================
class EfficientNetBinary(nn.Module):
    def __init__(self, variant: str, hidden: int = 0, freeze_backbone: bool = False):
        super().__init__()

        if variant == "effv2_s":
            self.backbone = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_m":
            self.backbone = tvm.efficientnet_v2_m(weights=tvm.EfficientNet_V2_M_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "effv2_l":
            self.backbone = tvm.efficientnet_v2_l(weights=tvm.EfficientNet_V2_L_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b0":
            self.backbone = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b1":
            self.backbone = tvm.efficientnet_b1(weights=tvm.EfficientNet_B1_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b2":
            self.backbone = tvm.efficientnet_b2(weights=tvm.EfficientNet_B2_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b3":
            self.backbone = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b4":
            self.backbone = tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b5":
            self.backbone = tvm.efficientnet_b5(weights=tvm.EfficientNet_B5_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b6":
            self.backbone = tvm.efficientnet_b6(weights=tvm.EfficientNet_B6_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        elif variant == "eff_b7":
            self.backbone = tvm.efficientnet_b7(weights=tvm.EfficientNet_B7_Weights.DEFAULT)
            feat_dim = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if hidden and hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.head = nn.Linear(feat_dim, 2)

        self.feat_dim = feat_dim
        self.hidden = int(hidden)
        self.freeze_backbone = bool(freeze_backbone)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


def make_eff_eval_tf(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class SingleImageDatasetEff(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.copy().reset_index(drop=True)
        self.tf = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["filepath"]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = int(row["binarylabel"])
        sev = str(row["severity_norm"]).lower()
        modality = str(row["dataset_norm"]).lower()
        corruption = str(row.get("corruption_norm", "unknown")).lower()
        return {
            "x": x,
            "y": torch.tensor(y, dtype=torch.long),
            "severity": sev,
            "fileindex": str(row["fileindex"]),
            "path": path,
            "modality": modality,
            "corruption": corruption,
        }

def collect_single_vlm(vlm: VLMAdapterWrapper, loader: DataLoader, adapter_on: bool):
    all_y, all_p = [], []
    all_sev, all_mod, all_cor = [], [], []
    all_paths, all_fids = [], []
    for batch in loader:
        y, p = forward_vlm_prob(vlm, batch, apply_adapter=adapter_on)
        all_y.append(y); all_p.append(p)
        all_sev.extend(batch["severities"])
        all_mod.extend(batch["modalities"])
        all_cor.extend(batch.get("corruptions", ["unknown"] * len(batch["modalities"])))
        all_paths.extend(batch["paths"])
        all_fids.extend(batch["fileindices"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    return y_true, y_prob, all_sev, all_mod, all_paths, all_fids, all_cor

def collect_single_eff(model: nn.Module, loader: DataLoader):
    model.eval()
    all_y, all_p = [], []
    all_sev, all_mod, all_cor = [], [], []
    all_paths, all_fids = [], []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        prob = torch.softmax(logits.float(), dim=1)[:, 1]

        all_y.append(y.detach().cpu())
        all_p.append(prob.detach().cpu())
        all_sev.extend(batch["severity"])
        all_mod.extend(batch["modality"])
        all_cor.extend(batch.get("corruption", ["unknown"] * len(batch["modality"])))
        all_paths.extend(batch["path"])
        all_fids.extend(batch["fileindex"])

    y_true = torch.cat(all_y, dim=0)
    y_prob = torch.cat(all_p, dim=0)
    return y_true, y_prob, all_sev, all_mod, all_paths, all_fids, all_cor


# =============================================================================
# Main
# =============================================================================
def main():
    global args
    args = parse_args()
    loader_generator = seed_everything(
        args.seed, deterministic=not args.allow_nondeterministic
    )
    print("Device:", device)
    if args.run_vlm or args.run_vlm_baseline:
        configure_hf_cache(args.hf_cache_dir)
    if os.path.exists(args.out_json) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out_json}. Pass --overwrite to replace it.")
    print(f"Seed: {args.seed}")

    # Validate run mode flags.
    if not args.run_vlm and not args.run_vlm_baseline and not args.run_effnet:
        raise ValueError("You must set at least one of --run_vlm / --run_vlm_baseline / --run_effnet")

    if args.run_vlm or args.run_vlm_baseline:
        if args.backend is None:
            raise ValueError("--run_vlm / --run_vlm_baseline requires --backend")

    if args.run_vlm:
        if args.vlm_ckpt is None:
            raise ValueError("--run_vlm requires --vlm_ckpt")


    if args.run_effnet:
        if args.eff_ckpt is None:
            raise ValueError("--run_effnet requires --eff_ckpt")

    # Prefer one combined CSV. The legacy clean/artefact pair remains supported.
    if args.data_csv:
        if args.clean_csv or args.artefact_csv:
            raise ValueError("Use either --data_csv or --clean_csv/--artefact_csv, not both")
        df_all = load_split_csv(
            args.data_csv,
            args.data_root,
            check_paths=not args.skip_path_check,
        )
        df_clean = df_all[df_all["severity_norm"] == "clean"].copy().reset_index(drop=True)
        df_pair_weak = df_all[df_all["severity_norm"] == "weak"].copy().reset_index(drop=True)
    else:
        if not args.clean_csv or not args.artefact_csv:
            raise ValueError(
                "Pass --data_csv, or pass both --clean_csv and --artefact_csv"
            )
        df_clean = load_split_csv(
            args.clean_csv,
            args.data_root,
            default_severity="clean",
            check_paths=not args.skip_path_check,
        )
        df_pair_weak = load_split_csv(
            args.artefact_csv,
            args.data_root,
            default_severity="weak",
            check_paths=not args.skip_path_check,
        )

        # The legacy files represent fixed splits, regardless of an existing column.
        df_clean["severity_norm"] = "clean"
        df_pair_weak["severity_norm"] = "weak"
        df_all = pd.concat([df_clean, df_pair_weak], ignore_index=True)

    n_pairs = count_valid_pairs(df_all)

    results = {
        "schema_version": "1.0",
        "reproducibility": build_repro_metadata(
            script_path=__file__,
            args=args,
            input_files=[args.data_csv, args.clean_csv, args.artefact_csv, args.vlm_ckpt, args.eff_ckpt],
            model_id=MODEL_ID_BY_BACKEND.get(args.backend) if args.backend else None,
            requested_revision=args.model_revision,
            prompts={"system": SYSTEM_PROMPT_SHORT, "by_dataset": PROMPT_BY_DATASET},
        ),
        "thr": float(args.thr),
        "data": {
            "data_csv": args.data_csv,
            "clean_csv": args.clean_csv,
            "artefact_csv": args.artefact_csv,
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_pair_weak)),
            "n_pairs": int(n_pairs),
            "use_type": "test",
            "seed": int(args.seed),
        },
        "models": {}
    }
    # VLM evaluation
    if args.run_vlm:
        backends = [args.backend]
        for backend in backends:
            model_id = MODEL_ID_BY_BACKEND[backend]
            vlm_ckpt = args.vlm_ckpt
            use_thr = float(args.thr)
            eval_layer = str(args.layer)
            adapter_on_clean_eval = bool(args.adapter_on_clean)
            adapter_bottleneck = 512
            ckpt_train_mode = None

            print("\n==============================")
            print(f"== EVAL VLM backend={backend} layer={args.layer}")
            print(f"== ckpt={vlm_ckpt}")
            print("==============================")

            ckpt = load_trusted_checkpoint(vlm_ckpt)
            checkpoint_model_id = ckpt.get("model_id")
            if checkpoint_model_id:
                model_id = str(checkpoint_model_id)
            checkpoint_revision = (
                ckpt.get("reproducibility", {}).get("model", {}).get("resolved_revision")
                or ckpt.get("config", {}).get("model_revision")
            )
            if args.model_revision and checkpoint_revision and args.model_revision != checkpoint_revision:
                raise ValueError(
                    "--model_revision does not match the revision stored in the adapter checkpoint: "
                    f"cli={args.model_revision!r}, checkpoint={checkpoint_revision!r}"
                )
            load_revision = args.model_revision or checkpoint_revision
            adapter_bottleneck = _infer_adapter_bottleneck_from_ckpt(ckpt, fallback=512)
            ckpt_train_mode = _infer_ckpt_train_mode(ckpt)
            eval_layer = _extract_layer_choice_from_ckpt(ckpt, fallback=eval_layer)
            if eval_layer != args.layer:
                print(f"[layer] override cli_layer={args.layer} -> ckpt_layer_choice={eval_layer}")
            print(f"[adapter] inferred bottleneck={adapter_bottleneck} train_mode={ckpt_train_mode}")

            if not adapter_on_clean_eval and _should_auto_apply_adapter_on_clean(ckpt):
                adapter_on_clean_eval = True
                print("[adapter_on_clean] auto-enabled from ckpt training mode")

            base_model, processor = load_backend(backend, model_id, load_revision)
            results["reproducibility"]["model"]["id"] = model_id
            results["reproducibility"]["model"]["requested_revision"] = load_revision
            results["reproducibility"]["model"]["resolved_revision"] = resolved_model_revision(
                base_model, load_revision
            )
            vlm = VLMAdapterWrapper(
                base_model,
                backend=backend,
                layer_choice=eval_layer,
                adapter_bottleneck=adapter_bottleneck,
            )

            ckpt_best_thr = _extract_best_threshold_from_ckpt(ckpt)
            if ckpt_best_thr is not None:
                try:
                    use_thr = float(ckpt_best_thr)
                except Exception:
                    pass
            print(f"[thr] using threshold={use_thr} (ckpt_best_thr={ckpt_best_thr}, cli_thr={args.thr})")
            vlm.adapter.load_state_dict(ckpt["adapter"], strict=True)
            vlm.classifier.load_state_dict(ckpt["classifier"], strict=True)
            if "pooler" in ckpt:
                vlm.pooler.load_state_dict(ckpt["pooler"], strict=True)
                vlm.use_attention_pooling = True
                print("== detected pooler in ckpt: attention pooling enabled")
            vlm.eval()

            ds_clean = SingleImageDatasetVLM(df_clean)
            ds_weak  = SingleImageDatasetVLM(df_pair_weak)

            collate = make_single_collate_fn_vlm(processor, backend)
            dl_clean = DataLoader(
                ds_clean,
                batch_size=1,
                shuffle=False,
                collate_fn=collate,
                pin_memory=(device == "cuda"),
                num_workers=int(args.num_workers),
                worker_init_fn=seed_worker,
                generator=loader_generator,
            )
            dl_weak = DataLoader(
                ds_weak,
                batch_size=1,
                shuffle=False,
                collate_fn=collate,
                pin_memory=(device == "cuda"),
                num_workers=int(args.num_workers),
                worker_init_fn=seed_worker,
                generator=loader_generator,
            )

            y_c, p_c, sev_c, mod_c, path_c, fid_c, cor_c = collect_single_vlm(
                vlm, dl_clean, adapter_on=adapter_on_clean_eval
            )
            y_w, p_w, sev_w, mod_w, path_w, fid_w, cor_w = collect_single_vlm(
                vlm, dl_weak, adapter_on=True
            )

            y_all   = torch.cat([y_c, y_w], dim=0)
            p_all   = torch.cat([p_c, p_w], dim=0)
            sev_all = sev_c + sev_w
            mod_all = mod_c + mod_w
            path_all = path_c + path_w
            fid_all = fid_c + fid_w
            cor_all = cor_c + cor_w

            model_records = build_records_from_prob_outputs(
                y_true=y_all,
                y_prob=p_all,
                severities=sev_all,
                modalities=mod_all,
                paths=path_all,
                fileindices=fid_all,
                corruptions=cor_all,
                thr=use_thr,
            )
            model_metrics_by_modality = compute_metrics_by_modality(model_records)

            mr = {
                "model_type": "vlm",
                "id": f"{backend}:{eval_layer}",
                "ckpt": vlm_ckpt,
                "config": {
                    "backend": backend,
                    "model_id": model_id,
                    "model_revision": resolved_model_revision(base_model, load_revision),
                    "layer": eval_layer,
                    "ckpt_train_mode": ckpt_train_mode,
                    "adapter_bottleneck": int(adapter_bottleneck),
                    "adapter_on_clean": adapter_on_clean_eval,
                    "eff_variant": None,
                    "eff_img_size": None,
                    "hidden": None,
                    "freeze_backbone": None,
                    "thr_used": float(use_thr),
                },
                "metrics_by_modality": model_metrics_by_modality,
                "per_image": model_records,
            }


            results["models"][f"vlm:{backend}:{eval_layer}"] = mr
            _write_results_outputs(results, args.out_json)

            del vlm, base_model, processor
            gc.collect()
            torch.cuda.empty_cache()

    # VLM prompt baseline (no ckpt)
    if args.run_vlm_baseline:
        backends = [args.backend]
        for backend in backends:
            model_id = MODEL_ID_BY_BACKEND[backend]

            print("\n==============================")
            print(f"== EVAL VLM PROMPT baseline backend={backend}")
            print("==============================")

            base_model, processor = load_backend_for_prompt(backend, model_id, args.model_revision)
            results["reproducibility"]["model"]["resolved_revision"] = resolved_model_revision(
                base_model, args.model_revision
            )
            partial_out_dir = os.path.dirname(os.path.abspath(args.out_json))
            partial_base_name = os.path.splitext(os.path.basename(args.out_json))[0]
            partial_model_key = _sanitize_filename_piece(f"vlm_prompt:{backend}")
            partial_csv_path = os.path.join(
                partial_out_dir,
                f"{partial_base_name}__{partial_model_key}__partial_per_image.csv",
            )
            print(f"[baseline] partial per-image CSV: {partial_csv_path}", flush=True)

            prompt_metrics_by_modality, parse_diagnostics, prompt_per_image = eval_prompt_baseline_metrics(
                base_model=base_model,
                processor=processor,
                backend=backend,
                df_all=df_all,
                max_new_tokens=int(args.baseline_max_new_tokens),
                partial_csv_path=partial_csv_path,
            )
            mr = {
                "model_type": "vlm_prompt_baseline",
                "id": f"{backend}:prompt",
                "ckpt": None,
                "config": {
                    "backend": backend,
                    "model_id": model_id,
                    "model_revision": resolved_model_revision(base_model, args.model_revision),
                    "layer": "prompt",
                    "adapter_on_clean": None,
                    "eff_variant": None,
                    "eff_img_size": None,
                    "hidden": None,
                    "freeze_backbone": None,
                    "max_new_tokens": int(args.baseline_max_new_tokens),
                    "effective_max_new_tokens": int(
                        max(int(args.baseline_max_new_tokens), QWEN_MIN_NEW_TOKENS)
                        if backend in QWEN_THINKING_BACKENDS
                        else min(int(args.baseline_max_new_tokens), 4)
                        if backend == "medgemma"
                        else int(args.baseline_max_new_tokens)
                    ),
                },
                "metrics_by_modality": prompt_metrics_by_modality,
                "parse_diagnostics": parse_diagnostics,
                "per_image": prompt_per_image,
            }

            results["models"][f"vlm_prompt:{backend}"] = mr
            _write_results_outputs(results, args.out_json)

            del base_model, processor
            gc.collect()
            torch.cuda.empty_cache()

    # EfficientNet evaluation
    if args.run_effnet:
        print("\n==============================")
        print(f"== EVAL EffNet variant={args.eff_variant} img_size={args.eff_img_size}")
        print(f"== ckpt={args.eff_ckpt}")
        print("==============================")

        ckpt = load_trusted_checkpoint(args.eff_ckpt)
        use_thr = float(args.thr)
        ckpt_best_thr = _extract_best_threshold_from_ckpt(ckpt)
        if ckpt_best_thr is not None:
            try:
                use_thr = float(ckpt_best_thr)
            except Exception:
                pass
        print(f"[thr] using threshold={use_thr} (ckpt_best_thr={ckpt_best_thr}, cli_thr={args.thr})")

        variant = ckpt.get("variant", args.eff_variant)
        hidden = int(ckpt.get("hidden", 0))
        freeze_backbone = bool(ckpt.get("freeze_backbone", False))

        model = EfficientNetBinary(variant=variant, hidden=hidden, freeze_backbone=freeze_backbone).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        eff_tf = make_eff_eval_tf(args.eff_img_size)

        ds_clean = SingleImageDatasetEff(df_clean, transform=eff_tf)
        ds_weak  = SingleImageDatasetEff(df_pair_weak, transform=eff_tf)

        dl_clean = DataLoader(
            ds_clean,
            batch_size=32,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device == "cuda"),
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
        dl_weak = DataLoader(
            ds_weak,
            batch_size=32,
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device == "cuda"),
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

        y_c, p_c, sev_c, mod_c, path_c, fid_c, cor_c = collect_single_eff(model, dl_clean)
        y_w, p_w, sev_w, mod_w, path_w, fid_w, cor_w = collect_single_eff(model, dl_weak)

        y_all   = torch.cat([y_c, y_w], dim=0)
        p_all   = torch.cat([p_c, p_w], dim=0)
        sev_all = sev_c + sev_w
        mod_all = mod_c + mod_w
        path_all = path_c + path_w
        fid_all = fid_c + fid_w
        cor_all = cor_c + cor_w

        model_records = build_records_from_prob_outputs(
            y_true=y_all,
            y_prob=p_all,
            severities=sev_all,
            modalities=mod_all,
            paths=path_all,
            fileindices=fid_all,
            corruptions=cor_all,
            thr=use_thr,
        )
        model_metrics_by_modality = compute_metrics_by_modality(model_records)

        mr = {
            "model_type": "effnet",
            "id": f"{variant}:{int(args.eff_img_size)}",
            "ckpt": args.eff_ckpt,
            "config": {
                "backend": None,
                "layer": None,
                "adapter_on_clean": None,
                "eff_variant": variant,
                "eff_img_size": int(args.eff_img_size),
                "hidden": hidden,
                "freeze_backbone": freeze_backbone,
                "thr_used": float(use_thr),
            },
            "metrics_by_modality": model_metrics_by_modality,
            "per_image": model_records,
        }

        results["models"][f"effnet:{variant}:{int(args.eff_img_size)}"] = mr
        _write_results_outputs(results, args.out_json)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    _write_results_outputs(results, args.out_json)



if __name__ == "__main__":
    main()

