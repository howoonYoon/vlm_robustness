#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, gc, json, re, argparse
from typing import Optional, Any, Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import time

import torch

from repro_utils import (
    add_reproducibility_args,
    build_repro_metadata,
    configure_hf_cache,
    hf_revision_kwargs,
    prepare_dataframe_paths,
    resolved_model_revision,
    save_json_atomic,
    seed_everything,
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
CANONICAL_BACKENDS = ["qwen3", "qwen35", "qwen36", "internvl", "lingshu", "lingshu_i", "medgemma"]
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
    p.add_argument("--backend", type=str.lower, required=True, choices=BACKEND_CHOICES)
    p.add_argument("--demo_csv", type=str, required=True, help="few_shot_examples.csv")

    p.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="single target CSV containing both clean and artefact/weak samples",
    )
    p.add_argument("--clean_csv", type=str, default=None, help="clean target CSV (legacy two-CSV input)")
    p.add_argument("--artefact_csv", type=str, default=None, help="artefact/weak target CSV (legacy two-CSV input)")
    p.add_argument(
        "--max_new_tokens", type=int, default=24,
        help=f"requested generation budget; Qwen3-family backends use at least {QWEN_MIN_NEW_TOKENS}",
    )
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_csv", type=str, default=None, help="optional per-image CSV output path")
    p.add_argument("--parse_mode", type=str, default="strict", choices=["strict", "relaxed"])
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing output files")
    add_reproducibility_args(p, add_seed=True, add_workers=False)
    return p.parse_args()

args = None
device = "cuda" if torch.cuda.is_available() else "cpu"
DBG_ONCE = {"medgemma": False, "internvl": False}


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
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",
    "lingshu":  "lingshu-medical-mllm/Lingshu-7B",
    "lingshu_i": "lingshu-medical-mllm/Lingshu-I-8B",
    "medgemma": "google/medgemma-1.5-4b-it",
}

# -------------------------
# CSV load
# -------------------------
def _stable_fileindex_from_path(p: str) -> str:
    b = os.path.basename(str(p)).replace("\\", "/")
    stem, _ = os.path.splitext(b)
    return stem


def load_csv(
    path: str,
    split: Optional[str] = None,
    *,
    data_root: Optional[str] = None,
    check_paths: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    if split is not None:
        split_norm = str(split).strip().lower()
        if "use_type" not in df.columns:
            raise RuntimeError(f"'use_type' column missing in {path} (required for split={split_norm})")
        df = df[df["use_type"].astype(str).str.lower() == split_norm].copy()

    # if severity missing, default clean (but NOTE: pair csv SHOULD have severity to separate clean/weak)
    if "severity" not in df.columns:
        df["severity"] = "clean"

    df = prepare_dataframe_paths(
        df,
        csv_path=path,
        data_root=data_root,
        check_exists=check_paths,
    )

    # fileindex
    if "fileindex" not in df.columns:
        df["fileindex"] = df["filepath"].map(_stable_fileindex_from_path)

    df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()
    if "dataset" not in df.columns:
        raise RuntimeError(f"'dataset' column missing in {path}")
    df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    return df.reset_index(drop=True)


def count_valid_pairs(df_pair: pd.DataFrame) -> int:
    n = 0
    for _, g in df_pair.groupby(["dataset_norm", "fileindex"], sort=False):
        sev = g["severity_norm"].astype(str).str.lower()
        n_clean = int((sev == "clean").sum())
        n_weak = int((sev == "weak").sum())
        if n_clean == 1 and n_weak >= 1:
            n += 1
    return int(n)

def _mean_optional(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))

def _parse_normal_disease(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    if not t:
        return None

    m_json = re.search(
        r'\{\s*"label"\s*:\s*"(normal|disease)"\s*\}',
        t,
        flags=re.IGNORECASE,
    )
    if m_json:
        lbl = m_json.group(1).lower()
        return 0 if lbl == "normal" else 1

    m_json_partial = re.search(
        r'"label"\s*:\s*"(normal|disease)\b',
        t,
        flags=re.IGNORECASE,
    )
    if m_json_partial:
        lbl = m_json_partial.group(1).lower()
        return 0 if lbl == "normal" else 1

    ans = t
    for marker in ["answer:", "assistant:"]:
        if marker in ans:
            ans = ans.split(marker)[-1].strip()

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

    m_leading_label = re.match(r'^\s*["\']?\s*(normal|disease)\b', block)
    if m_leading_label:
        return 0 if m_leading_label.group(1) == "normal" else 1

    if re.search(r"\bnot\s+normal\b", block):
        return 1
    if re.search(r"\bno\s+disease\b", block):
        return 0

    has_normal_block = re.search(r"\bnormal\b", block) is not None
    has_disease_block = re.search(r"\bdisease\b", block) is not None
    if has_normal_block and has_disease_block:
        return None

    for ln in answer_lines:
        m_exact = re.fullmatch(r'["\']?\s*(normal|disease)\s*["\']?[.!?]?\s*', ln)
        if m_exact:
            return 0 if m_exact.group(1) == "normal" else 1
        if re.search(r"\bnormal\b", ln) is not None:
            return 0
        if re.search(r"\bdisease\b", ln) is not None:
            return 1

    if has_normal_block:
        return 0
    if has_disease_block:
        return 1
    return None

# -------------------------
# Backend loader
# -------------------------
def load_backend(backend: str, model_id: str, model_revision: Optional[str]):
    from transformers import AutoProcessor

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    if backend in ["internvl", "medgemma", "qwen35", "qwen36", "lingshu_i"] and device == "cuda":
        torch_dtype = torch.bfloat16  # 둘 다 bf16이 더 안전한 경우 많음

    cache_dir = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or None
    revision_kwargs = hf_revision_kwargs(model_revision)
    common = dict(torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=True, **revision_kwargs)

    if backend == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, **revision_kwargs)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        return model.to(device).eval(), processor

    if backend in ["qwen35", "qwen36"]:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as e:
            import transformers

            raise RuntimeError(
                f"{backend} loads {model_id}, whose config uses model_type='qwen3_5'. "
                f"Your transformers version ({transformers.__version__}) does not include Qwen3.5 support.\n"
                "Install a newer Transformers build in the same conda env, for example:\n"
                "  python -m pip install -U 'transformers[torch]'\n"
                "If PyPI is still behind for this model:\n"
                "  python -m pip install -U 'transformers @ git+https://github.com/huggingface/transformers.git@main'"
            ) from e

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **common
        )
        return model.to(device).eval(), processor

    if backend in AUTO_IMAGE_TEXT_BACKENDS:
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None, **revision_kwargs,
        )
        return model.to(device).eval(), processor

    if backend == "lingshu":
        from transformers import Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, **revision_kwargs)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir, **common)
        return model.to(device).eval(), processor

    if backend == "internvl":
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
            torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None, **revision_kwargs,
        )
        return model.to(device).eval(), processor

    if backend == "medgemma":
        # ✅ MedGemma는 환경에 따라 모델 클래스가 다를 수 있어서 try/fallback이 안전
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir, **revision_kwargs
        )

        model = None
        last_err = None

        # 1) 가장 흔한 멀티모달 text-generation 계열
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=cache_dir,
                torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None, **revision_kwargs,
            )
        except Exception as e:
            last_err = e

        # 2) Vision2Seq 계열로 들어오는 경우
        if model is None:
            try:
                from transformers import AutoModelForVision2Seq
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id, trust_remote_code=True, cache_dir=cache_dir,
                    torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None, **revision_kwargs,
                )
                last_err = None
            except Exception as e:
                last_err = e

        # 3) 마지막 fallback (그래도 안되면 에러를 명확히)
        if model is None:
            raise RuntimeError(f"Failed to load medgemma model. Last error: {repr(last_err)}")

        return model.to(device).eval(), processor

    raise ValueError(backend)

# -------------------------
# Few-shot demo builder (모달리티별 고정)
# -------------------------
def build_demos_by_modality(df_demo: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    demos_by_mod = {}
    for mod, g in df_demo.groupby("dataset_norm"):
        demos = []
        for _, row in g.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            y = int(row["binarylabel"])
            ans = "disease" if y == 1 else "normal"
            prompt = SYSTEM_PROMPT_SHORT + "\n\n" + get_task_prompt(mod)
            demos.append({"image": img, "text": prompt, "answer": ans})
        demos_by_mod[str(mod)] = demos
    return demos_by_mod

def _build_qwen_style_messages_with_demos(demos, query_img, query_text):
    messages = []
    for ex in demos:
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": ex["image"]}, {"type": "text", "text": ex["text"]}],
        })
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]})
    messages.append({
        "role": "user",
        "content": [{"type": "image", "image": query_img}, {"type": "text", "text": query_text}],
    })
    return messages

def _apply_chat_template(processor, messages, *, add_generation_prompt: bool, backend: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if backend in QWEN_THINKING_BACKENDS:
        try:
            return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError as e:
            if "enable_thinking" not in str(e):
                raise
    return processor.apply_chat_template(messages, **kwargs)

def _build_internvl_single_prompt_with_demos(processor, demos, query_text):
    image_tok = getattr(processor, "image_token", None) or "<image>"
    parts = []
    for ex in demos:
        parts.append(f"{image_tok}\n{ex['text']}\n{ex['answer']}\n")
    parts.append(f"{image_tok}\n{query_text}\n")
    return "\n".join(parts)

def _decode_generated_text(tok, inputs: Dict[str, Any], sequences: torch.Tensor, max_new_tokens: int) -> str:
    """
    Prefer: decode only generated part (sequences[:, input_len:])
    Fallback: decode tail max_new_tokens
    """
    input_ids = inputs.get("input_ids", None)

    # 1) generated-part decode
    if input_ids is not None and torch.is_tensor(input_ids) and sequences.ndim == 2:
        in_len = int(input_ids.shape[1])
        if sequences.shape[1] > in_len:
            gen = sequences[:, in_len:]
            t = tok.batch_decode(gen, skip_special_tokens=True)[0].strip()
            if t:
                # take last non-empty line
                parts = [p.strip() for p in re.split(r"[\n\r]+", t) if p.strip()]
                return parts[-1] if parts else t

    # 2) fallback: tail
    tail = sequences[:, -max_new_tokens:] if sequences.ndim == 2 else sequences
    t = tok.batch_decode(tail, skip_special_tokens=True)[0].strip()
    if not t:
        t = tok.batch_decode(sequences, skip_special_tokens=True)[0].strip()
    parts = [p.strip() for p in re.split(r"[\n\r]+", t) if p.strip()]
    return parts[-1] if parts else t

def _dbg_print_inputs_once(tag: str, inputs: Dict[str, Any], max_items: int = 50):
    # tag: "medgemma" or "internvl"
    if DBG_ONCE.get(tag, False):
        return
    DBG_ONCE[tag] = True

    print(f"\n[{tag} debug] processor outputs keys = {list(inputs.keys())}", flush=True)

    # 텐서들 shape/dtype/device
    for k in list(inputs.keys())[:max_items]:
        v = inputs[k]
        if torch.is_tensor(v):
            print(
                f"[{tag} debug] {k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device} "
                f"min={float(v.min()) if v.numel() else 'NA'} max={float(v.max()) if v.numel() else 'NA'}",
                flush=True
            )
        elif isinstance(v, (list, tuple)) and len(v) and torch.is_tensor(v[0]):
            print(f"[{tag} debug] {k}: list[{len(v)}] of tensors, first shape={tuple(v[0].shape)} dtype={v[0].dtype}", flush=True)
        else:
            # 너무 길면 잘라서
            s = str(v)
            if len(s) > 200:
                s = s[:200] + " ... (truncated)"
            print(f"[{tag} debug] {k}: {type(v).__name__} = {s}", flush=True)

    print(f"[{tag} debug] end\n", flush=True)


@torch.no_grad()
def infer_one(model, processor, backend: str, demos, img: Image.Image, text: str,
              max_new_tokens: int, parse_mode: str):

    model_id = MODEL_ID_BY_BACKEND[backend]
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=(backend in TRUST_REMOTE_CODE_BACKENDS),
            **hf_revision_kwargs(args.model_revision),
        )

    def _move_and_cast(model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

    def _build_inputs(
        generation_mode: bool,
        answer_suffix: Optional[str] = None,
    ) -> Dict[str, Any]:
        all_images = [ex["image"] for ex in demos] + [img]

        if backend in QWEN_STYLE_BACKENDS:
            msgs = _build_qwen_style_messages_with_demos(demos, img, text)
            if generation_mode:
                chat_text = _apply_chat_template(
                    processor, msgs, add_generation_prompt=True, backend=backend
                )
            else:
                msgs_sc = list(msgs)
                msgs_sc.append({"role": "assistant", "content": [{"type": "text", "text": answer_suffix or ""}]})
                chat_text = _apply_chat_template(
                    processor, msgs_sc, add_generation_prompt=False, backend=backend
                )

            out = _move_and_cast(processor(text=[chat_text], images=all_images, padding=True, return_tensors="pt"))

            return out

        if backend == "internvl":
            prompt = _build_internvl_single_prompt_with_demos(processor, demos, text)
            if (not generation_mode) and answer_suffix is not None:
                prompt = prompt + answer_suffix
            inp = _move_and_cast(
                processor(text=[prompt], images=all_images, padding=True, return_tensors="pt")
            )

            _dbg_print_inputs_once(backend, inp)

            return inp

        if backend == "medgemma":
            messages = []
            for ex in demos:
                messages.append({
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": ex["text"]}],
                })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["answer"]}],
                })
            messages.append({
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            })
            if not generation_mode:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_suffix or ""}],
                })

            chat_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(generation_mode),
            )

            inp = _move_and_cast(processor(text=[chat_text], images=all_images, padding=True, return_tensors="pt"))

            _dbg_print_inputs_once("medgemma", inp)
            return inp

        raise RuntimeError(backend)

    inputs = _build_inputs(generation_mode=True)

    gen_max_new_tokens = int(max_new_tokens)
    if backend in QWEN_THINKING_BACKENDS:
        gen_max_new_tokens = max(gen_max_new_tokens, QWEN_MIN_NEW_TOKENS)
    elif backend == "medgemma":
        gen_max_new_tokens = min(gen_max_new_tokens, 4)

    print(f"[infer] before generate | backend={backend}", flush=True)
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

    seq = model.generate(**inputs, **gen_kwargs)

    print(f"[infer] after generate", flush=True)

    decoded = processor.batch_decode(seq, skip_special_tokens=True)[0]
    decoded_new = ""
    in_ids = inputs.get("input_ids", None)
    if torch.is_tensor(in_ids) and torch.is_tensor(seq) and seq.dim() == 2 and in_ids.dim() == 2:
        prompt_len = int(in_ids.shape[1])
        if seq.shape[1] > prompt_len:
            new_tok = seq[:, prompt_len:]
            decoded_new = processor.batch_decode(new_tok, skip_special_tokens=True)[0]

    raw = (decoded_new if (decoded_new or "").strip() else decoded).strip()

    pred_label = _parse_normal_disease(raw)

    lg = {"mode": "parse_only"}
    return raw, (None if pred_label is None else int(pred_label)), lg





# -------------------------
# Metrics
# -------------------------
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


def _coerce_pred_int(p: Any, y_true: int) -> int:
    if p in (0, 1):
        return int(p)
    return 1 - int(y_true)


def _compute_balanced_accuracy_and_f1_from_optional_preds(
    y_true_list: List[int],
    y_pred_list: List[Optional[int]],
) -> Dict[str, Optional[float]]:
    if len(y_true_list) == 0:
        return {
            "balanced_accuracy": None,
            "f1_score": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "sensitivity": None,
            "specificity": None,
            "tp": None,
            "tn": None,
            "fp": None,
            "fn": None,
        }

    yt_values = [int(y) for y in y_true_list]
    yp_values = [_coerce_pred_int(p, y) for y, p in zip(yt_values, y_pred_list)]
    yt = torch.tensor(yt_values, dtype=torch.long)
    yp = torch.tensor(yp_values, dtype=torch.long)
    m = compute_binary_metrics_from_preds(yt, yp)

    tp = int(m["tp"]); tn = int(m["tn"]); fp = int(m["fp"]); fn = int(m["fn"])
    eps = 1e-12
    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    bacc = 0.5 * (tpr + tnr)

    return {
        "balanced_accuracy": _to_nullable_float(bacc),
        "f1_score": _to_nullable_float(m.get("f1")),
        "accuracy": _to_nullable_float(m.get("acc")),
        "precision": _to_nullable_float(m.get("precision")),
        "recall": _to_nullable_float(m.get("recall")),
        "sensitivity": _to_nullable_float(tpr),
        "specificity": _to_nullable_float(tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_metrics_from_records(records: List[Dict[str, Any]], pred_key: str = "pred_label") -> Dict[str, Any]:
    clean_recs = [r for r in records if str(r.get("severity", "")).lower() == "clean"]
    art_recs = [r for r in records if str(r.get("severity", "")).lower() == "weak"]

    y_clean = [int(r["y_true"]) for r in clean_recs if r.get("y_true") is not None]
    y_art = [int(r["y_true"]) for r in art_recs if r.get("y_true") is not None]
    pred_clean = [r.get(pred_key) for r in clean_recs if r.get("y_true") is not None]
    pred_art = [r.get(pred_key) for r in art_recs if r.get("y_true") is not None]
    clean_lbl = _compute_balanced_accuracy_and_f1_from_optional_preds(y_clean, pred_clean)
    art_lbl = _compute_balanced_accuracy_and_f1_from_optional_preds(y_art, pred_art)
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
    rel_num = 0
    rel_den = 0
    cwc_num = 0
    cwc_den = 0
    for _, g in grouped.items():
        c = g["clean"]
        ws = g["weak"]
        if c is None or len(ws) == 0:
            continue

        y = int(c["y_true"])
        c_pred = _coerce_pred_int(c.get(pred_key), y)
        weak_preds = [
            _coerce_pred_int(w.get(pred_key), int(w["y_true"]))
            for w in ws
        ]
        n0 = sum(1 for p in weak_preds if p == 0)
        n1 = len(weak_preds) - n0
        consistency_vals.append(float(max(n0, n1) / max(1, len(weak_preds))))
        maj_pred = 1 if n1 >= (len(weak_preds) / 2) else 0

        cwc_den += 1
        if int(maj_pred) == int(c_pred):
            cwc_num += 1

        if int(c_pred) == y:
            rel_den += 1
            if int(maj_pred) != y:
                rel_num += 1

    prediction_consistency = _mean_optional([_to_nullable_float(v) for v in consistency_vals])
    flip_rate = None if rel_den == 0 else _to_nullable_float(rel_num / rel_den)
    consistency_with_clean = None if cwc_den == 0 else _to_nullable_float(cwc_num / cwc_den)

    return {
        "label_based_metrics": {
            "clean": clean_lbl,
            "artefact": {
                "balanced_accuracy": art_lbl["balanced_accuracy"],
                "f1_score": art_lbl["f1_score"],
                "accuracy": art_lbl["accuracy"],
                "precision": art_lbl["precision"],
                "recall": art_lbl["recall"],
                "sensitivity": art_lbl["sensitivity"],
                "specificity": art_lbl["specificity"],
                "tp": art_lbl["tp"],
                "tn": art_lbl["tn"],
                "fp": art_lbl["fp"],
                "fn": art_lbl["fn"],
                "performance_gap": performance_gap,
            },
            "robustness": {
                "prediction_consistency": prediction_consistency,
                "consistency_with_clean": consistency_with_clean,
            },
            "reliability": {
                "flip_rate": flip_rate,
            },
        }
    }


def compute_metrics_by_modality(records: List[Dict[str, Any]], pred_key: str = "pred_label") -> Dict[str, Dict[str, Any]]:
    mods = sorted({
        str(r.get("modality", "")).lower()
        for r in records
        if str(r.get("modality", "")).strip() != ""
    })
    out: Dict[str, Dict[str, Any]] = {}
    for m in mods:
        sub = [r for r in records if str(r.get("modality", "")).lower() == m]
        out[m] = compute_metrics_from_records(sub, pred_key=pred_key)
    return out


def _build_overall_metrics_rows(label_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group_name in ["clean", "artefact", "robustness", "reliability"]:
        group = label_metrics.get(group_name, {})
        if not isinstance(group, dict):
            continue
        for metric_name, value in group.items():
            rows.append({
                "scope": "overall",
                "modality": "all",
                "group": str(group_name),
                "metric": str(metric_name),
                "value": value,
            })
    return rows


def _build_modality_metrics_rows(metrics_by_modality: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for modality, payload in metrics_by_modality.items():
        label_metrics = payload.get("label_based_metrics", {})
        for group_name in ["clean", "artefact", "robustness", "reliability"]:
            group = label_metrics.get(group_name, {})
            if not isinstance(group, dict):
                continue
            for metric_name, value in group.items():
                rows.append({
                    "scope": "modality",
                    "modality": str(modality),
                    "group": str(group_name),
                    "metric": str(metric_name),
                    "value": value,
                })
    return rows


# -------------------------
# Run query inference (single list)
# -------------------------
def run_query_df(model, processor, backend: str, demos_by_mod: dict, dfq: pd.DataFrame, split_name: str):
    outputs = []

    for i, row in dfq.iterrows():
        mod = str(row["dataset_norm"]).lower()
        if mod not in demos_by_mod:
            raise RuntimeError(f"demo missing for modality={mod} in demo_csv")

        print(
            f"[sample] {i+1}/{len(dfq)} | fileindex={row['fileindex']} | modality={mod}",
            flush=True
        )

        demos = demos_by_mod[mod]
        img = Image.open(row["filepath"]).convert("RGB")
        y_true = None if ("binarylabel" not in row or pd.isna(row["binarylabel"])) else int(row["binarylabel"])
        prompt = SYSTEM_PROMPT_SHORT + "\n\n" + get_task_prompt(mod)

        raw, y_pred, _ = infer_one(
            model, processor, backend, demos, img, prompt,
            max_new_tokens=args.max_new_tokens, parse_mode=args.parse_mode
        )

        # print(
        #     f"[step] done | idx={i+1}/{len(dfq)} | fileindex={row['fileindex']} | pred={y_pred} | raw={repr(raw)}",
        #     flush=True
        # )

        outputs.append({
            "split": split_name,
            "idx": int(i),
            "fileindex": str(row["fileindex"]),
            "path": str(row["filepath"]),
            "modality": mod,
            "severity": str(row.get("severity_norm", "clean")),
            "y_true": None if y_true is None else int(y_true),
            "pred_label": (int(y_pred) if y_pred in (0, 1) else None),
            "raw_text": raw,
        })

    metrics = compute_metrics_from_records(outputs, pred_key="pred_label")
    metrics_by_modality = compute_metrics_by_modality(outputs, pred_key="pred_label")

    return {
        "split": split_name,
        "n_total": int(len(dfq)),
        "label_based_metrics": metrics["label_based_metrics"],
        "metrics_by_modality": metrics_by_modality,
        "per_image": outputs,
    }

# -------------------------
# Main
# -------------------------
def main():
    global args
    args = parse_args()
    args.backend = normalize_backend(args.backend)
    backend = args.backend
    print("Device:", device, flush=True)
    configure_hf_cache(args.hf_cache_dir)
    seed_everything(args.seed, deterministic=not args.allow_nondeterministic)
    if os.path.exists(args.out_json) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out_json}. Pass --overwrite to replace it.")
    if args.data_csv:
        if args.clean_csv or args.artefact_csv:
            raise ValueError("Use either --data_csv or --clean_csv/--artefact_csv, not both")
    elif not args.clean_csv or not args.artefact_csv:
        raise ValueError("Pass --data_csv, or pass both --clean_csv and --artefact_csv")

    model_id = MODEL_ID_BY_BACKEND[backend]
    model, processor = load_backend(backend, model_id, args.model_revision)

    df_demo = load_csv(
        args.demo_csv,
        data_root=args.data_root,
        check_paths=not args.skip_path_check,
    )
    demos_by_mod = build_demos_by_modality(df_demo)
    for mod, demos in demos_by_mod.items():
        print(f"[demo] {mod}: {len(demos)} images", flush=True)

    if args.data_csv:
        # Preserve the combined target CSV's original order and severity values.
        df_pair = load_csv(
            args.data_csv,
            split="test",
            data_root=args.data_root,
            check_paths=not args.skip_path_check,
        )
        df_clean = df_pair[df_pair["severity_norm"] == "clean"].copy().reset_index(drop=True)
        df_weak = df_pair[df_pair["severity_norm"] == "weak"].copy().reset_index(drop=True)
    else:
        # Legacy two-CSV input.
        df_clean = load_csv(
            args.clean_csv,
            split="test",
            data_root=args.data_root,
            check_paths=not args.skip_path_check,
        )
        df_weak = load_csv(
            args.artefact_csv,
            split="test",
            data_root=args.data_root,
            check_paths=not args.skip_path_check,
        )
        df_clean["severity_norm"] = "clean"
        df_weak["severity_norm"] = "weak"
        df_pair = pd.concat([df_clean, df_weak], ignore_index=True)

    res_pair_all = run_query_df(model, processor, backend, demos_by_mod, df_pair, "pair_set_test")
    n_pairs = count_valid_pairs(df_pair)

    model_key = f"vlm_few_shot:{backend}"
    model_result = {
        "model_type": "vlm_few_shot",
        "id": f"{backend}:few_shot",
        "ckpt": None,
        "config": {
            "backend": backend,
            "layer": "few_shot",
            "adapter_on_clean": None,
            "eff_variant": None,
            "eff_img_size": None,
            "hidden": None,
            "freeze_backbone": None,
            "demo_csv": args.demo_csv,
            "max_new_tokens": int(args.max_new_tokens),
            "effective_max_new_tokens": int(
                max(int(args.max_new_tokens), QWEN_MIN_NEW_TOKENS)
                if backend in QWEN_THINKING_BACKENDS
                else min(int(args.max_new_tokens), 4)
                if backend == "medgemma"
                else int(args.max_new_tokens)
            ),
            "parse_mode": args.parse_mode,
            "demo_summary": {k: len(v) for k, v in demos_by_mod.items()},
        },
        "label_based_metrics": res_pair_all["label_based_metrics"],
        "metrics_by_modality": res_pair_all["metrics_by_modality"],
        "per_image": res_pair_all["per_image"],
    }

    results = {
        "schema_version": "1.0",
        "reproducibility": build_repro_metadata(
            script_path=__file__,
            args=args,
            input_files=[args.demo_csv, args.data_csv, args.clean_csv, args.artefact_csv],
            model_id=model_id,
            requested_revision=args.model_revision,
            resolved_revision=resolved_model_revision(model, args.model_revision),
            prompts={"system": SYSTEM_PROMPT_SHORT, "by_dataset": PROMPT_BY_DATASET},
        ),
        "thr": 0.5,
        "data": {
            "data_csv": args.data_csv,
            "clean_csv": args.clean_csv,
            "artefact_csv": args.artefact_csv,
            "n_clean": int(len(df_clean)),
            "n_art": int(len(df_weak)),
            "n_pairs": int(n_pairs),
        },
        "models": {
            model_key: model_result
        },
    }

    save_json_atomic(args.out_json, results)
    print(f"[saved] {args.out_json}", flush=True)

    out_csv = args.out_csv
    if not out_csv:
        root, _ = os.path.splitext(args.out_json)
        out_csv = root + ".csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(res_pair_all["per_image"]).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[saved] {out_csv}", flush=True)

    out_root, _ = os.path.splitext(out_csv)
    out_metrics_csv = out_root + "_metrics.csv"
    out_metrics_by_mod_csv = out_root + "_metrics_by_modality.csv"

    rows_overall = _build_overall_metrics_rows(res_pair_all["label_based_metrics"])
    rows_by_mod = _build_modality_metrics_rows(res_pair_all["metrics_by_modality"])
    pd.DataFrame(rows_overall).to_csv(out_metrics_csv, index=False, encoding="utf-8")
    pd.DataFrame(rows_by_mod).to_csv(out_metrics_by_mod_csv, index=False, encoding="utf-8")
    print(f"[saved] {out_metrics_csv}", flush=True)
    print(f"[saved] {out_metrics_by_mod_csv}", flush=True)

    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
