#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
if os.environ.get("HF_HOME"):
    hf_home = os.environ["HF_HOME"]
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))

import random
import gc
import json
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
import argparse
import transformers

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

BACKEND_ALIASES = {
    "qwen3.5": "qwen35",
    "qwen3.5-9b": "qwen35",
    "qwen3_5": "qwen35",
    "qwen35-9b": "qwen35",
    "qwen35_9b": "qwen35",
    "qwen3.6": "qwen36",
    "qwen3.6-27b": "qwen36",
    "qwen3_6": "qwen36",
    "qwen36_27b": "qwen36",
    "lingshu-i": "lingshu_i",
    "lingshu-i-8b": "lingshu_i",
    "lingshu_i8b": "lingshu_i",
    "lingshu_i-8b": "lingshu_i",
}
CANONICAL_BACKENDS = ["qwen3", "qwen35", "qwen36", "medgemma", "internvl", "lingshu", "lingshu_i"]
BACKEND_CHOICES = sorted(set(CANONICAL_BACKENDS + list(BACKEND_ALIASES)))
QWEN_STYLE_BACKENDS = {"qwen3", "qwen35", "qwen36", "lingshu", "lingshu_i"}
AUTO_IMAGE_TEXT_BACKENDS = {"qwen35", "qwen36", "lingshu_i"}


def normalize_backend(backend: str) -> str:
    backend = str(backend).lower()
    return BACKEND_ALIASES.get(backend, backend)


# ============================================================
# Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=".", help="Root for relative image paths in CSVs")
    p.add_argument("--backend", type=str.lower, required=True, choices=BACKEND_CHOICES)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--tag",type=str, default="adapter")
    p.add_argument("--clean_csv", type=str, required=True)     # ✅ clean.csv
    p.add_argument("--artefact_csv", type=str, required=True)  # ✅ artefact.csv
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_cons", type=float, default=0.01)
    p.add_argument("--w_clean", type=float, default=1.0)
    p.add_argument("--w_art", type=float, default=1.0)
    p.add_argument("--adapter_bottleneck", type=int, default=256, choices=[128, 256])
    p.add_argument("--pooling", type=str, default="masked_mean", choices=["masked_mean", "last_token"])
    p.add_argument("--max_art_per_clean", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="./soft_prompt_ckpt")
    p.add_argument("--debug_pool", action="store_true",
                   help="print token/pooling related debug info")
    p.add_argument("--debug_pool_steps", type=int, default=3,
                   help="number of forward steps to print pooling debug info")
    p.add_argument("--debug_structure", action="store_true",
               help="print model output structure for pooling design")
    p.add_argument("--debug_structure_steps", type=int, default=3,
                help="number of forward steps to print structure debug info")
    return p.parse_args()

args = parse_args()
args.backend = normalize_backend(args.backend)
DEBUG_POOL_COUNTER = 0
DEBUG_STRUCTURE_COUNTER = 0
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# 0. Common
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


MODEL_ID_BY_BACKEND = {
    "qwen3":    "Qwen/Qwen3-VL-8B-Instruct",
    "qwen35":   "Qwen/Qwen3.5-9B",
    "qwen36":   "Qwen/Qwen3.6-27B",
    "medgemma": "google/medgemma-1.5-4b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
    "lingshu_i": "lingshu-medical-mllm/Lingshu-I-8B",
}

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
SYSTEM_PROMPT_SHORT = (
    "Output MUST be exactly one word: normal or disease.\n"
    "Do not output punctuation, quotes, JSON, or any extra words."
)

BACKEND = args.backend

# loss weights
w_clean = float(args.w_clean)
w_art   = float(args.w_art)
lambda_cons = float(args.lambda_cons)
ADAPTER_BOTTLENECK = int(args.adapter_bottleneck)
POOLING_STRATEGY = str(args.pooling)
ADAPTER_OBJECTIVE = "ce_plus_cosine_consistency_v1"
SELECTION_METRIC = "mean_clean_artefact_f1_thr05"

LR = args.lr
EPOCHS = args.epochs
POOL_LAYER_CHOICES = ["last"]

BATCH_SIZE_DEFAULT = 1
BATCH_SIZE_BY_BACKEND = {
    "internvl": 1,
    "lingshu": 1,
    "lingshu_i": 1,
    "qwen3": 1,
    "qwen35": 1,
    "qwen36": 1,
    "medgemma": 1,
}

SEED = int(args.seed)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(SEED)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "generator": g.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
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
        g.set_state(state["generator"])


def load_trusted_checkpoint(path: str) -> Dict[str, Any]:
    # PyTorch 2.6 defaults torch.load(weights_only=True), which can reject
    # checkpoints containing RNG states and JSON-like metadata.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ============================================================
# 1. Adapter + Classifier
# ============================================================
class VisualAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck=256, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up   = nn.Linear(bottleneck, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gate = nn.Parameter(torch.tensor(-4.0))
  # scalar gate (start near-identity)

    def forward(self, h):
        x = self.ln(h)
        delta = self.up(self.drop(self.act(self.down(x))))
        g = torch.sigmoid(self.gate)   # 0~1
        return h + g * delta

class VLMAdapterWrapper(nn.Module):
    def __init__(self, base_model, backend: str, layer_choice: str = "last",
                 adapter_bottleneck: int = 256, pooling_strategy: str = "masked_mean"):
        super().__init__()
        self.base_model = base_model
        self.backend = backend
        self.pooling_strategy = str(pooling_strategy)
        if self.pooling_strategy not in {"masked_mean", "last_token"}:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")
        if layer_choice != "last":
            raise ValueError("Only layer_choice='last' is supported.")
        self.layer_choice = "last"
        self.base_model.eval()

        for p in self.base_model.parameters():
            p.requires_grad = False

        emb = self._get_text_embeddings(self.base_model)
        hidden_dim = emb.weight.shape[1]
        embed_device = emb.weight.device

        self.hidden_dim = hidden_dim
        # ✅ 학습 파트 fp32 고정
        self.adapter = VisualAdapter(hidden_dim, bottleneck=int(adapter_bottleneck)).to(device=embed_device, dtype=torch.float32)
        self.classifier = nn.Linear(hidden_dim, 2).to(device=embed_device, dtype=torch.float32)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()   # backbone은 항상 eval 유지
        return self

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

        t = get(outputs, "last_hidden_state")
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

        if x is None and (
            isinstance(hs, (list, tuple))
            and len(hs) > 0
            and torch.is_tensor(hs[-1])
            and hs[-1].dim() == 3
            and hs[-1].shape[-1] == H
        ):
            x = hs[-1]
        elif x is None and torch.is_tensor(t) and t.dim() == 3 and t.shape[-1] == H:
            x = t

        if x is None:
            raise ValueError(
                f"No usable hidden representation found for last layer. "
                f"hs_len={len(hs) if isinstance(hs, (list, tuple)) else None}, "
                f"last_shape={None if t is None else tuple(t.shape)}"
            )

        if used_image_hidden or attention_mask is None or not torch.is_tensor(attention_mask) or attention_mask.dim() != 2:
            pooled = x.mean(dim=1)
        elif self.pooling_strategy == "last_token":
            T = min(x.shape[1], attention_mask.shape[1])
            x = x[:, :T, :]
            mask = attention_mask[:, :T] > 0
            last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
            gather_idx = last_idx.to(device=x.device).view(-1, 1, 1).expand(-1, 1, x.shape[-1])
            pooled = x.gather(dim=1, index=gather_idx).squeeze(1)
        else:
            T = min(x.shape[1], attention_mask.shape[1])
            x = x[:, :T, :]
            mask = (attention_mask[:, :T] > 0).to(device=x.device, dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=1) / denom

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


# ============================================================
# 2. Backend loader
# ============================================================
def load_backend(backend: str, model_id: str):
    import transformers
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["medgemma", "internvl", "qwen35", "qwen36", "lingshu_i"] and device == "cuda":
        torch_dtype = torch.bfloat16

    common = dict(
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend in ["qwen35", "qwen36"]:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                f"{backend} loads {model_id}, whose config uses model_type='qwen3_5'. "
                f"Your transformers version ({transformers.__version__}) does not include Qwen3.5 support.\n"
                "Install a newer Transformers build in the same conda env, for example:\n"
                "  pip install -U 'transformers[torch]'\n"
                "If PyPI is still behind for this model:\n"
                "  pip install -U 'transformers @ git+https://github.com/huggingface/transformers.git@main'"
            ) from e

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend in AUTO_IMAGE_TEXT_BACKENDS:
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is None:
            raise RuntimeError(
                f"{backend} requires transformers.AutoModelForImageTextToText. "
                "Please upgrade transformers."
            )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoImageTextToText.from_pretrained(model_id, trust_remote_code=True, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    if backend == "internvl":
        from transformers import AutoModel, AutoProcessor as AP2
        processor = AP2.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, **common)
        model.to(device)
        model.eval()

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

        return model, processor

    if backend == "medgemma":
        processor = AutoProcessor.from_pretrained(model_id)
        AutoImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
        if AutoImageTextToText is not None:
            model = AutoImageTextToText.from_pretrained(model_id, **common)
            model.to(device)
            model.eval()
            return model, processor
        AutoVision2Seq = getattr(transformers, "AutoModelForVision2Seq", None)
        if AutoVision2Seq is not None:
            model = AutoVision2Seq.from_pretrained(model_id, **common)
            model.to(device)
            model.eval()
            return model, processor
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
        model.to(device)
        model.eval()
        return model, processor

    raise ValueError(f"Unknown backend: {backend}")


# ============================================================
# 3. CSV load (✅ clean.csv / artefact.csv, ✅ use_type으로 train/val 분리)
# ============================================================
def load_csv(path: str, split: str) -> pd.DataFrame:
    split_norm = split.lower()
    if split_norm not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got: {split}")
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    if "use_type" in df.columns:
        df = df[df["use_type"].astype(str).str.lower() == split_norm].copy()
    else:
        raise ValueError(f"{path}: use_type column missing (need train/val split).")

    required_cols = ["dataset", "severity", "binarylabel", "fileindex", "filepath"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing required columns after split filtering: {missing_cols}")

    df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()

    df["fileindex"] = df["fileindex"].astype(str)   # 추가

    def resolve_image_path(value):
        path = os.path.expanduser(str(value)).replace("\\", "/")
        return path if os.path.isabs(path) else os.path.join(args.data_root, path)
    df["filepath"] = df["filepath"].map(resolve_image_path)

    return df.reset_index(drop=True)


train_clean_df = load_csv(args.clean_csv, "train")
train_art_df   = load_csv(args.artefact_csv, "train")
val_clean_df   = load_csv(args.clean_csv, "val")
val_art_df     = load_csv(args.artefact_csv, "val")

print("train_clean:", train_clean_df.shape, "train_art:", train_art_df.shape)
print("val_clean  :", val_clean_df.shape,   "val_art  :", val_art_df.shape)


# ============================================================
# 4. Pair Dataset (fileindex로 매칭)
# ============================================================
class CleanArtefactGroupDataset(Dataset):
    def __init__(self, clean_df, art_df, prompt_by_dataset, system_prompt=None, max_art_per_clean: int = 5):
        self.prompt_by_dataset = prompt_by_dataset
        self.system_prompt = system_prompt
        self.max_art_per_clean = int(max_art_per_clean)

        clean_df = clean_df.copy().reset_index(drop=True)
        art_df   = art_df.copy().reset_index(drop=True)

        duplicate_clean = clean_df.duplicated(["dataset_norm", "fileindex"], keep=False)
        if duplicate_clean.any():
            n_dup = int(duplicate_clean.sum())
            print(f"[warn] duplicate clean rows for (dataset_norm, fileindex): {n_dup}; keeping first row per key.")
            clean_df = clean_df.drop_duplicates(["dataset_norm", "fileindex"], keep="first").reset_index(drop=True)

        # clean: (dataset_norm, fileindex) -> 1 row
        clean_map = {}
        for _, row in clean_df.iterrows():
            key = (str(row["dataset_norm"]), str(row["fileindex"]))
            clean_map[key] = row

        self.groups: List[Tuple[pd.Series, List[pd.Series]]] = []
        for (dataset_norm, fid), g in art_df.groupby(["dataset_norm", "fileindex"], sort=False):
            key = (str(dataset_norm), str(fid))
            if key not in clean_map:
                continue

            clean_row = clean_map[key]
            g2 = g.sort_values(["severity_norm", "filepath"]).reset_index(drop=True)  # 순서 고정
            rows = [g2.iloc[i] for i in range(len(g2))]

            if self.max_art_per_clean > 0:
                rows = rows[: self.max_art_per_clean]

            if len(rows) == 0:
                continue

            self.groups.append((clean_row, rows))

        print(f"Dataset - found {len(self.groups)} clean groups (each with up to {self.max_art_per_clean} artefacts).")

    def _make_sample(self, row: pd.Series):
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")

        modality = str(row["dataset_norm"])
        label = int(row["binarylabel"])

        task_prompt = self.prompt_by_dataset.get(
            modality,
            "This is a medical image.\nQuestion: Does this image show normal anatomy or signs of disease?\nAnswer:",
        )
        full_text = (self.system_prompt + "\n\n" + task_prompt) if self.system_prompt else task_prompt
        return {"image": img, "input_text": full_text, "label": label, "path": img_path}

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        clean_row, art_rows = self.groups[idx]
        clean = self._make_sample(clean_row)
        artefacts = [self._make_sample(r) for r in art_rows]
        return {"clean": clean, "artefacts": artefacts}

def make_group_collate_fn(processor, backend: str):
    def encode(images, texts):
        if backend in QWEN_STYLE_BACKENDS:
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                          for m in messages_list]
            return processor(text=chat_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "internvl":
            image_tok = getattr(processor, "image_token", None) or "<image>"
            inline_texts = [f"{image_tok}\n{txt}" for txt in texts]
            return processor(text=inline_texts, images=images, padding=True, return_tensors="pt")

        elif backend == "medgemma":
            if not hasattr(processor, "apply_chat_template"):
                raise RuntimeError("medgemma needs processor.apply_chat_template.")
            messages_list = []
            for img, txt in zip(images, texts):
                messages_list.append([{
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": txt}],
                }])
            chat_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                          for m in messages_list]

            # ✅ Gemma 계열 안정적으로: images를 nested로 주는 게 안전함
            images_nested = [[img] for img in images]
            return processor(text=chat_texts, images=images_nested, padding=True, return_tensors="pt")

        else:
            return processor(text=texts, images=images, padding=True, return_tensors="pt")

    def collate(batch):
        # batch: list of {"clean": sample, "artefacts": [samples...]}

        # ---- clean (B items) ----
        clean_images, clean_texts, clean_labels, clean_paths = [], [], [], []
        for item in batch:
            s = item["clean"]
            clean_images.append(s["image"])
            clean_texts.append(s["input_text"])
            clean_labels.append(int(s["label"]))
            clean_paths.append(s.get("path", "<unknown>"))

        clean_inputs = dict(encode(clean_images, clean_texts))
        clean_inputs["labels_cls"] = torch.tensor(clean_labels, dtype=torch.long)
        clean_inputs["paths"] = clean_paths

        # ---- artefacts (flatten B*K items) ----
        art_images, art_texts, art_labels, art_paths, parent_idx = [], [], [], [], []
        for bi, item in enumerate(batch):
            for s in item["artefacts"]:
                art_images.append(s["image"])
                art_texts.append(s["input_text"])
                art_labels.append(int(s["label"]))
                art_paths.append(s.get("path", "<unknown>"))
                parent_idx.append(bi)  # which clean this artefact belongs to

        if len(art_images) == 0:
            # edge case: no artefacts
            art_inputs = {"labels_cls": torch.empty((0,), dtype=torch.long),
                          "paths": [],
                          "parent_idx": torch.empty((0,), dtype=torch.long)}
        else:
            art_inputs = dict(encode(art_images, art_texts))
            art_inputs["labels_cls"] = torch.tensor(art_labels, dtype=torch.long)
            art_inputs["paths"] = art_paths
            art_inputs["parent_idx"] = torch.tensor(parent_idx, dtype=torch.long)

        return {"clean": clean_inputs, "artefact": art_inputs}

    return collate

# ============================================================
# Helpers
# ============================================================
def to_device(x, target_device: torch.device):
    if torch.is_tensor(x):
        return x.to(target_device, non_blocking=True)
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in ("paths", "path", "meta"):
                out[k] = v
            else:
                out[k] = to_device(v, target_device)
        return out
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return x
        return type(x)(to_device(v, target_device) for v in x)
    return x


def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, thr: float = 0.5):
    y_true = y_true.detach().cpu().to(torch.int64)
    y_score = y_score.detach().cpu().to(torch.float32)

    y_pred = (y_score >= thr).to(torch.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = int(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = int(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true == 1)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # AUROC (Mann–Whitney U)
    pos = (y_true == 1)
    neg = (y_true == 0)
    if int(pos.sum()) == 0 or int(neg.sum()) == 0:
        auroc = float("nan")
    else:
        order = torch.argsort(y_score)  # ascending
        sorted_score = y_score[order]
        sorted_y = y_true[order]
        _, counts = torch.unique_consecutive(sorted_score, return_counts=True)
        counts = counts.to(torch.float32)
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.float32), counts[:-1]]),
            dim=0,
        )
        ends = starts + counts - 1.0
        avg_ranks = ((starts + 1.0) + (ends + 1.0)) / 2.0
        ranks_sorted = torch.repeat_interleave(avg_ranks, counts.to(torch.int64))
        sum_ranks_pos = ranks_sorted[sorted_y == 1].sum()
        n_pos = float(pos.sum().item())
        n_neg = float(neg.sum().item())
        u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
        auroc = float((u / (n_pos * n_neg)).item())

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def find_best_f1_threshold(y_true: torch.Tensor, y_score: torch.Tensor):
    """
    y_true: (N,) int64 {0,1} on CPU
    y_score: (N,) float32 prob for class1 on CPU
    Returns: best_thr (float), best_f1 (float), best_precision (float), best_recall (float)
    """
    y_true = y_true.to(torch.int64).cpu()
    y_score = y_score.to(torch.float32).cpu()

    n = y_true.numel()
    if n == 0:
        return 0.5, float("nan"), float("nan"), float("nan")

    pos_total = int((y_true == 1).sum().item())
    neg_total = n - pos_total
    if pos_total == 0 or neg_total == 0:
        # F1 정의가 의미 없거나 한쪽만 있음
        return 0.5, float("nan"), float("nan"), float("nan")

    # score 내림차순 정렬
    order = torch.argsort(y_score, descending=True)
    s = y_score[order]
    y = y_true[order]

    # 누적 TP/FP
    tp_cum = torch.cumsum((y == 1).to(torch.int64), dim=0)
    fp_cum = torch.cumsum((y == 0).to(torch.int64), dim=0)

    # 각 cutoff k에서 예측양성 = 0..k까지
    # threshold는 s[k] (>= threshold => 양성)
    tp = tp_cum
    fp = fp_cum
    fn = pos_total - tp

    eps = 1e-12
    precision = tp.to(torch.float32) / (tp + fp).to(torch.float32).clamp_min(1)  # 분모 0 방지
    recall    = tp.to(torch.float32) / (tp + fn).to(torch.float32).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 같은 score가 여러 개면 마지막(가장 낮은 threshold) 지점만 대표로 봄 (중복 제거)
    # -> unique score의 마지막 인덱스만 남김
    # mask: 다음 score와 다를 때 True (마지막 원소는 True)
    diff = torch.ones_like(s, dtype=torch.bool)
    diff[:-1] = s[:-1] != s[1:]
    idxs = torch.nonzero(diff, as_tuple=False).squeeze(1)

    f1_u = f1[idxs]
    best_i = int(torch.argmax(f1_u).item())
    k = int(idxs[best_i].item())

    best_thr = float(s[k].item())
    best_f1 = float(f1[k].item())
    best_p = float(precision[k].item())
    best_r = float(recall[k].item())
    return best_thr, best_f1, best_p, best_r

# ============================================================
# Train/Eval (✅ 둘 다 adapter 통과 + consistency는 adapted 공간에서)
# ============================================================
def forward_base(vlm_adapt: VLMAdapterWrapper, inputs: Dict[str, Any]) -> torch.Tensor:
    d = dict(inputs)
    for k in ["labels_cls", "labels_token", "paths", "path", "meta", "parent_idx"]:
        d.pop(k, None)

    target_device = next(vlm_adapt.adapter.parameters()).device

    with torch.no_grad():
        forward_kwargs = dict(d)
        if vlm_adapt.backend in ["internvl", "lingshu_i"] and device == "cuda":
            for k in ["pixel_values", "images", "image", "vision_x"]:
                if k in forward_kwargs and torch.is_tensor(forward_kwargs[k]):
                    forward_kwargs[k] = forward_kwargs[k].to(dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)
        else:
            outputs = vlm_adapt.base_model(**forward_kwargs, output_hidden_states=True, return_dict=True)

    attn = d.get("attention_mask", None)

    h = vlm_adapt.extract_features(outputs, attention_mask=attn)
    h = h.to(target_device, dtype=torch.float32)   # device/dtype 명시
    h = F.layer_norm(h, (h.shape[-1],))
    if not torch.isfinite(h).all():
        raise RuntimeError("NaN/Inf in base features")
    return h



def run_epoch(vlm_adapt: VLMAdapterWrapper, loader, optimizer=None):
    train = optimizer is not None
    vlm_adapt.train() if train else vlm_adapt.eval()
    target_device = next(vlm_adapt.adapter.parameters()).device

    total_loss = total_ce = total_cons = 0.0
    total_lc = total_la = total_lcos = 0.0
    n_steps = 0

    # overall
    all_y, all_score = [], []

    # split metrics
    clean_y, clean_score = [], []
    art_y, art_score = [], []

    with torch.set_grad_enabled(train):
        for batch in loader:
            clean = to_device(batch["clean"], target_device)
            art   = to_device(batch["artefact"], target_device)

            y_c = clean["labels_cls"]  # (B,)
            y_a = art["labels_cls"]    # (M,)
            parent_idx = art["parent_idx"].to(device=y_c.device) if y_a.numel() > 0 else None
            if y_a.numel() > 0:
                expected_y_a = y_c[parent_idx]
                assert torch.equal(y_a, expected_y_a), (
                    "clean/art label mismatch: "
                    f"clean={y_c.detach().cpu().tolist()} "
                    f"art={y_a.detach().cpu().tolist()} "
                    f"parent_idx={parent_idx.detach().cpu().tolist()}"
                )

            # ----- base features -----
            h_c_base = forward_base(vlm_adapt, clean)   # (B,H)
            if y_a.numel() > 0:
                h_a_base = forward_base(vlm_adapt, art) # (M,H)
            else:
                h_a_base = torch.empty((0, h_c_base.shape[1]), device=h_c_base.device, dtype=h_c_base.dtype)

            # ----- adapter -----
            h_c = vlm_adapt.adapter(h_c_base)  # (B,H)
            h_a = vlm_adapt.adapter(h_a_base)  # (M,H)

            # ----- classifier -----
            logits_c = vlm_adapt.classifier(h_c)  # (B,2)
            L_c = F.cross_entropy(logits_c, y_c)

            if y_a.numel() > 0:
                logits_a = vlm_adapt.classifier(h_a)  # (M,2)
                L_a = F.cross_entropy(logits_a, y_a)
            else:
                logits_a = torch.empty((0, 2), device=logits_c.device, dtype=logits_c.dtype)
                L_a = torch.tensor(0.0, device=logits_c.device)

            L_ce = w_clean * L_c + w_art * L_a

            # ----- consistency (artefact -> corresponding clean) -----
            if y_a.numel() > 0:
                parent_idx = parent_idx.to(device=h_c.device)
                h_c_rep = h_c[parent_idx].detach()   # (M,H)

                ha_n = F.normalize(h_a, dim=1)
                hc_n = F.normalize(h_c_rep, dim=1)
                L_cos = (1 - (ha_n * hc_n).sum(dim=1)).mean()

                L_cons = L_cos
            else:
                L_cos = torch.tensor(0.0, device=logits_c.device)
                L_cons = torch.tensor(0.0, device=logits_c.device)

            loss = L_ce + lambda_cons * L_cons

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # ----- metrics -----
            with torch.no_grad():
                prob_c = torch.softmax(logits_c.float(), dim=1)[:, 1]
                all_y.append(y_c.detach().cpu())
                all_score.append(prob_c.detach().cpu())
                clean_y.append(y_c.detach().cpu())
                clean_score.append(prob_c.detach().cpu())

                if y_a.numel() > 0:
                    prob_a = torch.softmax(logits_a.float(), dim=1)[:, 1]
                    all_y.append(y_a.detach().cpu())
                    all_score.append(prob_a.detach().cpu())
                    art_y.append(y_a.detach().cpu())
                    art_score.append(prob_a.detach().cpu())

            total_loss += float(loss.item())
            total_ce   += float(L_ce.item())
            total_cons += float(L_cons.item())
            total_lc   += float(L_c.item())
            total_la   += float(L_a.item())
            total_lcos += float(L_cos.item())
            n_steps += 1

    def build_metrics(y_list, s_list):
        if len(y_list) == 0:
            return {
                "thr05": {"acc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                          "f1": float("nan"), "auroc": float("nan"),
                          "tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "best": {"acc": float("nan"), "precision": float("nan"), "recall": float("nan"),
                         "f1": float("nan"), "auroc": float("nan"),
                         "tp": 0, "tn": 0, "fp": 0, "fn": 0,
                         "best_thr": float("nan"), "best_f1": float("nan"),
                         "best_precision": float("nan"), "best_recall": float("nan")}
            }

        y_true = torch.cat(y_list, dim=0)
        y_score = torch.cat(s_list, dim=0)

        m05 = compute_binary_metrics(y_true, y_score, thr=0.5)
        best_thr, best_f1, best_p, best_r = find_best_f1_threshold(y_true, y_score)
        mbest = compute_binary_metrics(y_true, y_score, thr=best_thr)
        mbest = {
            **mbest,
            "best_thr": float(best_thr),
            "best_f1": float(best_f1),
            "best_precision": float(best_p),
            "best_recall": float(best_r),
        }
        return {"thr05": m05, "best": mbest}

    overall_metrics = build_metrics(all_y, all_score)
    clean_metrics   = build_metrics(clean_y, clean_score)
    art_metrics     = build_metrics(art_y, art_score)

    return {
        "loss": total_loss / max(n_steps, 1),
        "ce": total_ce / max(n_steps, 1),
        "cons": total_cons / max(n_steps, 1),
        "clean_ce": total_lc / max(n_steps, 1),
        "artefact_ce": total_la / max(n_steps, 1),
        "cons_cos": total_lcos / max(n_steps, 1),

        "overall_thr05": overall_metrics["thr05"],
        "overall_best": overall_metrics["best"],

        "clean_thr05": clean_metrics["thr05"],
        "clean_best": clean_metrics["best"],

        "artefact_thr05": art_metrics["thr05"],
        "artefact_best": art_metrics["best"],
    }


# ============================================================
# Main
# ============================================================
SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(SAVE_DIR, "metrics_json")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n==============================")
print(f"== Training backend: {BACKEND}")
print("==============================")

model_id = MODEL_ID_BY_BACKEND[BACKEND]
base_model, processor = load_backend(BACKEND, model_id)
print("base_model device:", next(base_model.parameters()).device, flush=True)

for layer_choice in POOL_LAYER_CHOICES:
    print(
        f"\n=== BACKEND={BACKEND} | LAYER={layer_choice} | "
        f"POOLING={POOLING_STRATEGY} | BOTTLENECK={ADAPTER_BOTTLENECK} ==="
    )
    def set_seed(s=SEED):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        g.manual_seed(s)

    set_seed(SEED)

    vlm_adapt = VLMAdapterWrapper(
        base_model,
        backend=BACKEND,
        layer_choice=layer_choice,
        adapter_bottleneck=ADAPTER_BOTTLENECK,
        pooling_strategy=POOLING_STRATEGY,
    )

    collate_fn = make_group_collate_fn(processor, BACKEND)
    bs = args.bs if args.bs is not None else BATCH_SIZE_BY_BACKEND.get(BACKEND, BATCH_SIZE_DEFAULT)

    train_ds = CleanArtefactGroupDataset(
        train_clean_df,
        train_art_df,
        PROMPT_BY_DATASET,
        SYSTEM_PROMPT_SHORT,
        max_art_per_clean=args.max_art_per_clean,
    )
    val_ds = CleanArtefactGroupDataset(
        val_clean_df,
        val_art_df,
        PROMPT_BY_DATASET,
        SYSTEM_PROMPT_SHORT,
        max_art_per_clean=args.max_art_per_clean,
    )

    print(f"[{BACKEND}] Using batch_size = {bs}")
    print(f"[{BACKEND}] #train_pairs={len(train_ds)} #val_pairs={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate_fn, pin_memory=(device=="cuda"), generator=g,)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate_fn, pin_memory=(device=="cuda"), generator=g,)

    optimizer = AdamW(list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()), lr=LR)

    EXP_TAG = args.tag
    pooling_suffix = "" if POOLING_STRATEGY == "masked_mean" else f"_{POOLING_STRATEGY}"
    BEST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{EXP_TAG}{pooling_suffix}_best.pt")
    LAST_CKPT = os.path.join(SAVE_DIR, f"{BACKEND}_{EXP_TAG}{pooling_suffix}_last.pt")

    metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{EXP_TAG}{pooling_suffix}_adapter_metrics.json")

    # ✅ resume
    start_epoch = 0
    best_val_score = -1.0
    best_val_thr = 0.5
    epoch_logs = []

    if os.path.exists(LAST_CKPT):
        ckpt = load_trusted_checkpoint(LAST_CKPT)
        resume_training_state = True
        ckpt_config = ckpt.get("config", {})
        if not isinstance(ckpt_config, dict):
            ckpt_config = {}
        if str(ckpt.get("adapter_objective", ckpt_config.get("adapter_objective", ""))) != ADAPTER_OBJECTIVE:
            resume_training_state = False
            print("[resume] adapter objective changed; restarting optimizer/history.")
        if int(ckpt_config.get("adapter_bottleneck", ADAPTER_BOTTLENECK)) != ADAPTER_BOTTLENECK:
            resume_training_state = False
            print("[resume] adapter bottleneck changed; restarting optimizer/history.")
        if str(ckpt_config.get("pooling_strategy", POOLING_STRATEGY)) != POOLING_STRATEGY:
            resume_training_state = False
            print("[resume] pooling strategy changed; restarting optimizer/history.")
        if str(ckpt_config.get("selection_metric", SELECTION_METRIC)) != SELECTION_METRIC:
            resume_training_state = False
            print("[resume] selection metric changed; restarting optimizer/history.")

        try:
            vlm_adapt.adapter.load_state_dict(ckpt["adapter"], strict=True)
            vlm_adapt.classifier.load_state_dict(ckpt["classifier"], strict=True)
        except RuntimeError as e:
            resume_training_state = False
            print(f"[resume] adapter/classifier shape mismatch; restarting optimizer/history: {e}")
            vlm_adapt = VLMAdapterWrapper(
                base_model,
                backend=BACKEND,
                layer_choice=layer_choice,
                adapter_bottleneck=ADAPTER_BOTTLENECK,
                pooling_strategy=POOLING_STRATEGY,
            )

        if resume_training_state and ckpt.get("selection_metric") != SELECTION_METRIC:
            raise ValueError("Checkpoint selection criterion differs; use a new save_dir/tag for the F1 run")

        if resume_training_state and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                resume_training_state = False
                print(f"[resume] optimizer state mismatch; restarting optimizer/history: {e}")

        if resume_training_state:
            restore_rng_state(ckpt.get("rng_state"))
            best_val_score = float(ckpt.get("best_val_score", -1.0))
            start_epoch = int(ckpt.get("epoch", 0))
            epoch_logs = ckpt.get("epoch_logs", [])
            best_val_thr = float(ckpt.get("best_val_thr", 0.5))
            RUN_ID = ckpt.get("run_id", RUN_ID)
            metrics_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{BACKEND}_{EXP_TAG}{pooling_suffix}_adapter_metrics.json")
            print(f"[resume] Loaded {LAST_CKPT} | start_epoch={start_epoch} | best_val_score={best_val_score:.4f}")
        else:
            optimizer = AdamW(list(vlm_adapt.adapter.parameters()) + list(vlm_adapt.classifier.parameters()), lr=LR)
            print("[resume] Loaded compatible weights where possible; restarted from epoch 0.")
    else:
        print("[resume] No last checkpoint found. Start fresh.")

    for epoch in range(start_epoch, EPOCHS):
        tr_m  = run_epoch(vlm_adapt, train_loader, optimizer)
        val_m = run_epoch(vlm_adapt, val_loader, optimizer=None)

        tr_f1_05 = tr_m["overall_thr05"].get("f1", float("nan"))
        va_f1_05 = val_m["overall_thr05"].get("f1", float("nan"))
        tr_best = tr_m["overall_best"]
        va_best = val_m["overall_best"]
        tr_f1_best = tr_best.get("best_f1", float("nan"))
        va_f1_best = va_best.get("best_f1", float("nan"))
        va_thr_best = va_best.get("best_thr", 0.5)
        tr_auroc = tr_m["overall_thr05"].get("auroc", float("nan"))
        va_auroc = val_m["overall_thr05"].get("auroc", float("nan"))

        tr_over_05 = tr_m["overall_thr05"]
        va_over_05 = val_m["overall_thr05"]

        tr_over_best = tr_m["overall_best"]
        va_over_best = val_m["overall_best"]

        tr_clean_05 = tr_m["clean_thr05"]
        va_clean_05 = val_m["clean_thr05"]
        tr_clean_best = tr_m["clean_best"]
        va_clean_best = val_m["clean_best"]

        tr_art_05 = tr_m["artefact_thr05"]
        va_art_05 = val_m["artefact_thr05"]
        tr_art_best = tr_m["artefact_best"]
        va_art_best = val_m["artefact_best"]

        va_thr_best = va_best.get("best_thr", 0.5)

        print(
            f"[{BACKEND}] Epoch {epoch+1}/{EPOCHS}\n"
            f"Train: loss={tr_m['loss']:.4f} CE={tr_m['ce']:.4f} Cons={tr_m['cons']:.4f} "
            f"(Lc={tr_m['clean_ce']:.4f}, La={tr_m['artefact_ce']:.4f}, COS={tr_m['cons_cos']:.4f})\n"
            f"  Overall : F1@0.5={tr_over_05['f1']:.3f} F1@best={tr_over_best['best_f1']:.3f} AUROC={tr_over_05['auroc']:.3f}\n"
            f"  Clean   : F1@0.5={tr_clean_05['f1']:.3f} F1@best={tr_clean_best['best_f1']:.3f} "
            f"Rec@0.5={tr_clean_05['recall']:.3f} AUROC={tr_clean_05['auroc']:.3f}\n"
            f"  Artefact: F1@0.5={tr_art_05['f1']:.3f} F1@best={tr_art_best['best_f1']:.3f} "
            f"Rec@0.5={tr_art_05['recall']:.3f} AUROC={tr_art_05['auroc']:.3f}\n"
            f"Val  : loss={val_m['loss']:.4f} CE={val_m['ce']:.4f} Cons={val_m['cons']:.4f} "
            f"(Lc={val_m['clean_ce']:.4f}, La={val_m['artefact_ce']:.4f}, COS={val_m['cons_cos']:.4f})\n"
            f"  Overall : F1@0.5={va_over_05['f1']:.3f} F1@best={va_over_best['best_f1']:.3f} "
            f"(thr={va_over_best['best_thr']:.3f}) AUROC={va_over_05['auroc']:.3f}\n"
            f"  Clean   : F1@0.5={va_clean_05['f1']:.3f} F1@best={va_clean_best['best_f1']:.3f} "
            f"Rec@0.5={va_clean_05['recall']:.3f} AUROC={va_clean_05['auroc']:.3f}\n"
            f"  Artefact: F1@0.5={va_art_05['f1']:.3f} F1@best={va_art_best['best_f1']:.3f} "
            f"Rec@0.5={va_art_05['recall']:.3f} AUROC={va_art_05['auroc']:.3f}\n"
            f"  Selection({SELECTION_METRIC}): mean F1={(va_clean_05['f1'] + va_art_05['f1']) / 2:.3f}, "
            f"F1@0.5={va_over_05['f1']:.3f}, F1@best={va_over_best['best_f1']:.3f}"
        )

        epoch_logs.append({"epoch": epoch + 1, "train": tr_m, "val": val_m})

        va_thr_best = va_over_best.get("best_thr", 0.5)
        va_f1_best  = va_over_best.get("best_f1", float("nan"))
        score = 0.5 * (float(va_clean_05["f1"]) + float(va_art_05["f1"]))

        if not np.isnan(score) and score > best_val_score:
            best_val_score = float(score)
            best_val_thr = 0.5

            torch.save(
                {
                    "adapter": vlm_adapt.adapter.state_dict(),
                    "classifier": vlm_adapt.classifier.state_dict(),
                    "hidden_dim": vlm_adapt.hidden_dim,
                    "backend": BACKEND,
                    "model_id": model_id,
                    "layer_choice": vlm_adapt.layer_choice,
                    "epoch": epoch + 1,
                    "adapter_objective": ADAPTER_OBJECTIVE,
                    "selection_metric": SELECTION_METRIC,
                    "best_val_score": best_val_score,
                    "best_val_thr": best_val_thr,
                    "val_overall_thr05": va_over_05,
                    "val_overall_best": va_over_best,
                    "val_clean_best": va_clean_best,
                    "val_artefact_best": va_art_best,
                    "config": {
                        "mode": "paired",
                        "lambda_cons": lambda_cons,
                        "w_clean": w_clean,
                        "w_art": w_art,
                        "adapter_bottleneck": ADAPTER_BOTTLENECK,
                        "pooling_strategy": POOLING_STRATEGY,
                        "adapter_objective": ADAPTER_OBJECTIVE,
                        "selection_metric": SELECTION_METRIC,
                        "max_art_per_clean": int(args.max_art_per_clean),
                    },
                    "rng_state": capture_rng_state(),
                },
                BEST_CKPT,
            )
            print(
                f"BEST saved: {BEST_CKPT} "
                f"({SELECTION_METRIC}={best_val_score:.4f}, "
                f"F1@0.5={va_over_05['f1']:.4f}, bestF1={va_f1_best:.4f}, thr={best_val_thr:.4f})"
            )


        payload = {
            "backend": BACKEND,
            "layer_choice": layer_choice,
            "pooling_strategy": POOLING_STRATEGY,
            "model_id": model_id,
            "seed": SEED,
            "epochs": EPOCHS,
            "adapter_objective": ADAPTER_OBJECTIVE,
            "selection_metric": SELECTION_METRIC,
            "logs": epoch_logs,
            "best_val_score": best_val_score,
            "best_val_thr": float(best_val_thr),

        }
        payload["env"] = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "model_class": type(base_model).__name__,
            "processor_class": type(processor).__name__,
        }
        payload["config"] = {
            "backend": BACKEND,
            "model_id": model_id,
            "seed": SEED,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": bs,
            "lambda_cons": lambda_cons,
            "w_clean": w_clean,
            "w_art": w_art,
            "adapter_bottleneck": ADAPTER_BOTTLENECK,
            "pooling_strategy": POOLING_STRATEGY,
            "adapter_objective": ADAPTER_OBJECTIVE,
            "selection_metric": SELECTION_METRIC,
            "max_art_per_clean": int(args.max_art_per_clean),
            "clean_csv": args.clean_csv,
            "artefact_csv": args.artefact_csv,
            "system_prompt": SYSTEM_PROMPT_SHORT,
            "prompt_by_dataset": PROMPT_BY_DATASET,
            "mode": "paired",
        }
        tmp_path = metrics_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, metrics_path)

        torch.save(
            {
                "run_id": RUN_ID,
                "epoch": epoch + 1,
                "adapter": vlm_adapt.adapter.state_dict(),
                "classifier": vlm_adapt.classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "backend": BACKEND,
                "model_id": model_id,
                "layer_choice": vlm_adapt.layer_choice,
                "adapter_objective": ADAPTER_OBJECTIVE,
                "selection_metric": SELECTION_METRIC,
                "epoch_logs": epoch_logs,
                "best_val_score": best_val_score,
                "best_val_thr": float(best_val_thr),
                "val_overall_thr05": va_over_05,
                "val_overall_best": va_over_best,
                "val_clean_best": va_clean_best,
                "val_artefact_best": va_art_best,
                "config": payload["config"],
                "rng_state": capture_rng_state(),
            },
            LAST_CKPT,
        )
        print(f"[metrics] wrote {metrics_path} | epochs_logged={len(epoch_logs)}")

    del vlm_adapt, optimizer, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

del base_model, processor
gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.ipc_collect()
