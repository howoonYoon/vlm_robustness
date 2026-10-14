#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import gc
import json
import mimetypes
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CLEAN_CSV = None
DEFAULT_ARTEFACT_CSV = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=".", help="Root for relative image paths in CSVs")
    p.add_argument("--model", type=str, default="gpt-5.5",
                   help="OpenAI vision-capable model name, e.g. gpt-5.5")
    p.add_argument("--query_csv", type=str, default=None,
                   help="single test csv. If set, --clean_csv/--artefact_csv are not used")
    p.add_argument("--clean_csv", type=str, default=DEFAULT_CLEAN_CSV,
                   help="clean csv (severity clean)")
    p.add_argument("--artefact_csv", type=str, default=DEFAULT_ARTEFACT_CSV,
                   help="artefact/weak csv (severity weak)")
    p.add_argument("--mode", type=str, default="sync",
                   choices=["sync", "batch_submit", "batch_status", "batch_collect"],
                   help="sync calls the API per image; batch_submit creates OpenAI Batch jobs; batch_collect downloads completed results")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_csv", type=str, default=None, help="optional per-image CSV output path")
    p.add_argument("--batch_id", type=str, default=None,
                   help="single OpenAI batch id for batch_status/batch_collect")
    p.add_argument("--batch_metadata_json", type=str, default=None,
                   help="batch job metadata json path. Defaults to <out_json root>_batch_jobs.json")
    p.add_argument("--batch_input_prefix", type=str, default=None,
                   help="prefix for generated batch JSONL shards. Defaults to <out_json root>_batch_input")
    p.add_argument("--batch_output_jsonl", type=str, default=None,
                   help="combined downloaded batch output JSONL path. Defaults to <out_json root>_batch_output.jsonl")
    p.add_argument("--batch_error_jsonl", type=str, default=None,
                   help="combined downloaded batch error JSONL path. Defaults to <out_json root>_batch_errors.jsonl")
    p.add_argument("--batch_max_file_mb", type=float, default=180.0,
                   help="max generated JSONL shard size before creating another batch input file")
    p.add_argument("--batch_completion_window", type=str, default="24h",
                   help="OpenAI Batch completion window. Currently 24h")
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--reasoning_effort", type=str, default="low",
                   choices=["low", "medium", "high", "xhigh"],
                   help="lower values reduce hidden reasoning tokens before the visible answer")
    p.add_argument("--text_verbosity", type=str, default="low",
                   choices=["low", "medium", "high"],
                   help="lower values encourage shorter visible answers")
    p.add_argument("--parse_mode", type=str, default="strict", choices=["strict", "relaxed"])
    p.add_argument("--image_detail", type=str, default="auto", choices=["auto", "low", "high"])
    p.add_argument("--only_modality", type=str, default="mri",
                   help="optional modality filter, e.g. mri/oct/xray, or all for every modality. Default is mri for quick testing")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between API calls")
    p.add_argument("--limit", type=int, default=None, help="optional limit for quick smoke tests")
    return p.parse_args()


args = parse_args()


PROMPT_BY_DATASET = {
    "mri": "This is a brain MRI scan.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "oct": "This is a retinal OCT scan.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "xray": "This is a chest X-ray image.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
    "fundus": "This is a retinal fundus photograph.\nQuestion: Does this image show normal anatomy or signs of disease?\n\n",
}

SYSTEM_PROMPT_SHORT = 'Answer with ONE WORD only: "normal" or "disease". Do not explain.'


def _stable_fileindex_from_path(p: str) -> str:
    b = os.path.basename(str(p).replace("\\", "/"))
    stem, _ = os.path.splitext(b)
    return stem


def normalize_filepath(path: str) -> str:
    raw = os.path.expanduser(str(path)).replace("\\", "/")
    if os.path.isabs(raw):
        return raw
    return os.path.join(args.data_root, raw)



def load_csv(path: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "binarylab" in df.columns and "binarylabel" not in df.columns:
        df = df.rename(columns={"binarylab": "binarylabel"})

    if split is not None:
        split_norm = str(split).strip().lower()
        if "use_type" not in df.columns:
            raise RuntimeError(f"'use_type' column missing in {path} (required for split={split_norm})")
        df = df[df["use_type"].astype(str).str.lower() == split_norm].copy()

    if "severity" not in df.columns:
        df["severity"] = "clean"

    if "filepath" not in df.columns:
        raise RuntimeError(f"'filepath' column missing in {path}")

    df["filepath"] = df["filepath"].astype(str).map(normalize_filepath)

    if "fileindex" not in df.columns:
        df["fileindex"] = df["filepath"].map(_stable_fileindex_from_path)

    if "dataset" not in df.columns:
        raise RuntimeError(f"'dataset' column missing in {path}")

    df["severity_norm"] = df["severity"].fillna("clean").astype(str).str.lower()
    df["dataset_norm"] = df["dataset"].astype(str).str.lower()
    return df.reset_index(drop=True)


def filter_modality(df: pd.DataFrame, modality: Optional[str], name: str) -> pd.DataFrame:
    if modality is None or str(modality).strip() == "":
        return df.reset_index(drop=True)

    mod = str(modality).strip().lower()
    if mod in ("all", "none", "*"):
        return df.reset_index(drop=True)

    out = df[df["dataset_norm"].astype(str).str.lower() == mod].copy().reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError(f"No rows left in {name} after --only_modality {mod}")
    print(f"[filter] {name}: modality={mod}, rows={len(out)}", flush=True)
    return out


def count_valid_pairs(df_pair: pd.DataFrame) -> int:
    n = 0
    for _, g in df_pair.groupby("fileindex"):
        sev = g["severity_norm"].astype(str).str.lower()
        n_clean = int((sev == "clean").sum())
        n_weak = int((sev == "weak").sum())
        if n_clean >= 1 and n_weak == 5:
            n += 1
    return int(n)


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/png"


def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(path)};base64,{encoded}"


def build_openai_input(query_image_url: str, query_text: str, image_detail: str) -> List[Dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": query_image_url, "detail": image_detail},
            {"type": "input_text", "text": query_text},
        ],
    }]


def _parse_normal_disease(text: str) -> Optional[int]:
    t = (text or "").strip().lower()
    if not t:
        return None

    m_json = re.search(r'\{\s*"label"\s*:\s*"(normal|disease)"\s*\}', t, flags=re.IGNORECASE)
    if m_json:
        return 0 if m_json.group(1).lower() == "normal" else 1

    m_json_partial = re.search(r'"label"\s*:\s*"(normal|disease)\b', t, flags=re.IGNORECASE)
    if m_json_partial:
        return 0 if m_json_partial.group(1).lower() == "normal" else 1

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


def infer_one_openai(client, model: str, image_path: str, text: str) -> Tuple[str, Optional[int], Dict[str, Any]]:
    query_image_url = image_to_data_url(image_path)

    kwargs = {
        "model": model,
        "instructions": SYSTEM_PROMPT_SHORT,
        "input": build_openai_input(
            query_image_url=query_image_url,
            query_text=text,
            image_detail=args.image_detail,
        ),
        "max_output_tokens": int(args.max_output_tokens),
        "text": {"verbosity": args.text_verbosity},
    }
    if args.reasoning_effort:
        kwargs["reasoning"] = {"effort": args.reasoning_effort}

    response = client.responses.create(**kwargs)
    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw and hasattr(response, "model_dump"):
        dumped = response.model_dump()
        raw = _extract_text_from_response_dump(dumped).strip()

    pred_label = _parse_normal_disease(raw)
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return raw, (None if pred_label is None else int(pred_label)), {
        "usage": usage,
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "incomplete_details": _model_dump_or_none(getattr(response, "incomplete_details", None)),
    }


def _model_dump_or_none(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, dict):
        return x
    return {"value": str(x)}


def _extract_text_from_response_dump(dumped: Dict[str, Any]) -> str:
    texts: List[str] = []
    for item in dumped.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in ("output_text", "text"):
                txt = content.get("text")
                if txt:
                    texts.append(str(txt))
    return "\n".join(texts)


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


def _compute_balanced_accuracy_and_f1_from_optional_preds(
    y_true_list: List[int],
    y_pred_list: List[Optional[int]],
) -> Dict[str, Optional[float]]:
    valid = [(int(y), int(p)) for y, p in zip(y_true_list, y_pred_list) if p in (0, 1)]
    if len(valid) == 0:
        return {
            "balanced_accuracy": None,
            "f1_score": None,
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "tp": None,
            "tn": None,
            "fp": None,
            "fn": None,
        }

    tp = sum(1 for y, p in valid if y == 1 and p == 1)
    tn = sum(1 for y, p in valid if y == 0 and p == 0)
    fp = sum(1 for y, p in valid if y == 0 and p == 1)
    fn = sum(1 for y, p in valid if y == 1 and p == 0)
    eps = 1e-12
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    sensitivity = recall
    specificity = tn / (tn + fp + eps)
    bacc = 0.5 * (sensitivity + specificity)

    return {
        "balanced_accuracy": _to_nullable_float(bacc),
        "f1_score": _to_nullable_float(f1),
        "accuracy": _to_nullable_float(acc),
        "sensitivity": _to_nullable_float(sensitivity),
        "specificity": _to_nullable_float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _mean_optional(vals: List[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


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
        if fid not in grouped:
            grouped[fid] = {"clean": None, "weak": []}
        sev = str(r.get("severity", "")).lower()
        if sev == "clean" and grouped[fid]["clean"] is None:
            grouped[fid]["clean"] = r
        elif sev == "weak":
            grouped[fid]["weak"].append(r)

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

        weak_preds = [w.get(pred_key) for w in ws]
        weak_valid = [int(p) for p in weak_preds if p in (0, 1)]
        if len(weak_valid) == len(ws):
            n0 = sum(1 for p in weak_valid if p == 0)
            n1 = len(weak_valid) - n0
            consistency_vals.append(float(max(n0, n1) / max(1, len(weak_valid))))
            n_pos = sum(1 for p in weak_valid if p == 1)
            maj_pred = 1 if n_pos >= (len(weak_valid) / 2) else 0
        else:
            maj_pred = None

        c_pred = c.get(pred_key)
        y = c.get("y_true")
        if c_pred in (0, 1) and maj_pred in (0, 1):
            cwc_den += 1
            if int(maj_pred) == int(c_pred):
                cwc_num += 1

        if y in (0, 1) and c_pred in (0, 1) and int(c_pred) == int(y) and maj_pred in (0, 1):
            rel_den += 1
            if int(maj_pred) != int(y):
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


def _is_missing_value(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _clean_loaded_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in record.items():
        out[k] = None if _is_missing_value(v) else v

    for k in ["idx", "y_true", "pred_label"]:
        if out.get(k) is not None:
            out[k] = int(out[k])
    return out


def _append_record_csv(out_csv: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    pd.DataFrame([record]).to_csv(
        out_csv,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def _load_existing_records(out_csv: str, target_paths: set) -> List[Dict[str, Any]]:
    if not out_csv or not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return []

    df_prev = pd.read_csv(out_csv, on_bad_lines="skip")
    if "path" not in df_prev.columns:
        return []

    df_prev = df_prev[df_prev["path"].astype(str).isin(target_paths)].copy()
    by_path: Dict[str, Dict[str, Any]] = {}
    for record in df_prev.to_dict("records"):
        clean_record = _clean_loaded_record(record)
        path = clean_record.get("path")
        if path is not None:
            by_path[str(path)] = clean_record
    return list(by_path.values())


def _default_out_csv() -> str:
    if args.out_csv:
        return args.out_csv
    root, _ = os.path.splitext(args.out_json)
    return root + ".csv"


def _batch_paths() -> Dict[str, str]:
    root, _ = os.path.splitext(args.out_json)
    return {
        "jobs": args.batch_metadata_json or root + "_batch_jobs.json",
        "manifest": root + "_batch_manifest.jsonl",
        "input_prefix": args.batch_input_prefix or root + "_batch_input",
        "output": args.batch_output_jsonl or root + "_batch_output.jsonl",
        "errors": args.batch_error_jsonl or root + "_batch_errors.jsonl",
    }


def _json_line(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _build_response_body(image_path: str, prompt: str) -> Dict[str, Any]:
    body = {
        "model": args.model,
        "instructions": SYSTEM_PROMPT_SHORT,
        "input": build_openai_input(
            query_image_url=image_to_data_url(image_path),
            query_text=prompt,
            image_detail=args.image_detail,
        ),
        "max_output_tokens": int(args.max_output_tokens),
        "text": {"verbosity": args.text_verbosity},
    }
    if args.reasoning_effort:
        body["reasoning"] = {"effort": args.reasoning_effort}
    return body


def _batch_request_for_row(custom_id: str, image_path: str, prompt: str) -> Dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": _build_response_body(image_path=image_path, prompt=prompt),
    }


def _write_batch_inputs(dfq: pd.DataFrame, out_csv: str, paths: Dict[str, str]) -> List[Dict[str, Any]]:
    if args.limit is not None:
        dfq = dfq.head(int(args.limit)).copy()

    target_paths = set(dfq["filepath"].astype(str).tolist())
    completed = {str(r["path"]) for r in _load_existing_records(out_csv, target_paths) if r.get("path") is not None}
    if completed:
        print(f"[resume] skipping {len(completed)} rows already in {out_csv}", flush=True)

    os.makedirs(os.path.dirname(paths["manifest"]) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(paths["input_prefix"]) or ".", exist_ok=True)

    max_bytes = max(1, int(float(args.batch_max_file_mb) * 1024 * 1024))
    shard_paths: List[str] = []
    shard_file = None
    shard_bytes = 0
    shard_rows = 0
    shard_idx = -1
    total_rows = 0

    def _open_next_shard():
        nonlocal shard_file, shard_bytes, shard_rows, shard_idx
        if shard_file is not None:
            shard_file.close()
        shard_idx += 1
        shard_path = f"{paths['input_prefix']}_{shard_idx:04d}.jsonl"
        shard_paths.append(shard_path)
        shard_file = open(shard_path, "w", encoding="utf-8")
        shard_bytes = 0
        shard_rows = 0

    with open(paths["manifest"], "w", encoding="utf-8") as manifest_file:
        for i, row in dfq.iterrows():
            image_path = str(row["filepath"])
            if image_path in completed:
                continue

            mod = str(row["dataset_norm"]).lower()
            y_true = None if ("binarylabel" not in row or pd.isna(row["binarylabel"])) else int(row["binarylabel"])
            custom_id = f"request-{total_rows:08d}"
            prompt = PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])
            request_obj = _batch_request_for_row(custom_id, image_path, prompt)
            line = _json_line(request_obj)
            line_bytes = len(line.encode("utf-8")) + 1

            if shard_file is None or (shard_rows > 0 and shard_bytes + line_bytes > max_bytes):
                _open_next_shard()

            shard_file.write(line + "\n")
            shard_bytes += line_bytes
            shard_rows += 1

            manifest = {
                "custom_id": custom_id,
                "split": "pair_set_test",
                "idx": int(i),
                "fileindex": str(row["fileindex"]),
                "path": image_path,
                "modality": mod,
                "severity": str(row.get("severity_norm", "clean")),
                "y_true": None if y_true is None else int(y_true),
            }
            manifest_file.write(_json_line(manifest) + "\n")
            total_rows += 1

    if shard_file is not None:
        shard_file.close()

    shard_meta = []
    for path in shard_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            shard_meta.append({"path": path, "bytes": int(os.path.getsize(path))})

    print(f"[batch] wrote {total_rows} pending requests into {len(shard_meta)} JSONL shard(s)", flush=True)
    return shard_meta


def submit_batch_jobs(client, dfq: pd.DataFrame, out_csv: str) -> Dict[str, Any]:
    paths = _batch_paths()
    shards = _write_batch_inputs(dfq, out_csv, paths)
    jobs: List[Dict[str, Any]] = []

    for shard in shards:
        print(f"[batch] uploading {shard['path']}", flush=True)
        with open(shard["path"], "rb") as f:
            input_file = client.files.create(file=f, purpose="batch")

        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window=args.batch_completion_window,
            metadata={
                "script": "openai_zero_shot_test.py",
                "model": args.model,
            },
        )
        job = {
            "batch_id": batch.id,
            "input_file_id": input_file.id,
            "input_jsonl": shard["path"],
            "input_bytes": shard["bytes"],
            "status": getattr(batch, "status", None),
            "batch": _model_dump_or_none(batch),
        }
        jobs.append(job)
        print(f"[batch] submitted {batch.id} status={getattr(batch, 'status', None)}", flush=True)

    payload = {
        "schema_version": "1.0",
        "mode": "batch_submit",
        "endpoint": "/v1/responses",
        "completion_window": args.batch_completion_window,
        "manifest_jsonl": paths["manifest"],
        "out_json": args.out_json,
        "out_csv": out_csv,
        "jobs": jobs,
    }

    os.makedirs(os.path.dirname(paths["jobs"]) or ".", exist_ok=True)
    with open(paths["jobs"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {paths['jobs']}", flush=True)
    return payload


def _load_batch_jobs(paths: Dict[str, str]) -> Dict[str, Any]:
    if args.batch_id:
        return {"jobs": [{"batch_id": args.batch_id}], "manifest_jsonl": paths["manifest"]}

    if not os.path.exists(paths["jobs"]):
        raise RuntimeError(f"Batch metadata file not found: {paths['jobs']}")
    with open(paths["jobs"], "r", encoding="utf-8") as f:
        return json.load(f)


def show_batch_status(client) -> None:
    paths = _batch_paths()
    payload = _load_batch_jobs(paths)
    changed = False
    for job in payload.get("jobs", []):
        batch_id = job["batch_id"]
        batch = client.batches.retrieve(batch_id)
        batch_dump = _model_dump_or_none(batch)
        job["status"] = getattr(batch, "status", None)
        job["output_file_id"] = getattr(batch, "output_file_id", None)
        job["error_file_id"] = getattr(batch, "error_file_id", None)
        job["batch"] = batch_dump
        changed = True
        print(
            f"[batch] {batch_id} status={getattr(batch, 'status', None)} "
            f"output_file_id={getattr(batch, 'output_file_id', None)} "
            f"error_file_id={getattr(batch, 'error_file_id', None)}",
            flush=True,
        )
        request_counts = getattr(batch, "request_counts", None)
        if request_counts is not None:
            print(f"[batch] request_counts={json.dumps(_model_dump_or_none(request_counts), ensure_ascii=False)}", flush=True)
        errors = getattr(batch, "errors", None)
        if errors is not None:
            print(f"[batch] errors={json.dumps(_model_dump_or_none(errors), ensure_ascii=False)}", flush=True)

    if changed and not args.batch_id:
        with open(paths["jobs"], "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _file_content_text(client, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(content)


def _load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise RuntimeError(f"Batch manifest file not found: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            out[str(record["custom_id"])] = record
    return out


def _record_from_batch_line(line_obj: Dict[str, Any], manifest: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    custom_id = str(line_obj.get("custom_id", ""))
    base = manifest.get(custom_id)
    if base is None:
        return None

    error = line_obj.get("error")
    response = line_obj.get("response") or {}
    body = response.get("body") or {}
    raw = ""
    y_pred = None
    if not error:
        raw = (body.get("output_text") or "").strip()
        if not raw:
            raw = _extract_text_from_response_dump(body).strip()
        y_pred = _parse_normal_disease(raw)

    return {
        "split": base.get("split"),
        "idx": int(base["idx"]),
        "fileindex": str(base["fileindex"]),
        "path": str(base["path"]),
        "modality": str(base["modality"]),
        "severity": str(base["severity"]),
        "y_true": base.get("y_true"),
        "pred_label": (int(y_pred) if y_pred in (0, 1) else None),
        "raw_text": raw,
        "response_id": body.get("id"),
        "response_status": body.get("status") or response.get("status_code"),
        "incomplete_details": json.dumps(body.get("incomplete_details") or error, ensure_ascii=False),
        "usage": json.dumps(body.get("usage"), ensure_ascii=False),
    }


def collect_batch_results(client, dfq: pd.DataFrame, out_csv: str) -> Optional[Dict[str, Any]]:
    paths = _batch_paths()
    payload = _load_batch_jobs(paths)
    manifest = _load_manifest(payload.get("manifest_jsonl") or paths["manifest"])
    target_paths = set(dfq["filepath"].astype(str).tolist())
    completed_paths = {str(r["path"]) for r in _load_existing_records(out_csv, target_paths) if r.get("path") is not None}

    all_complete = True
    os.makedirs(os.path.dirname(paths["output"]) or ".", exist_ok=True)
    output_seen = set()
    error_seen = set()

    with open(paths["output"], "w", encoding="utf-8") as out_f, open(paths["errors"], "w", encoding="utf-8") as err_f:
        for job in payload.get("jobs", []):
            batch_id = job["batch_id"]
            batch = client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            print(f"[batch] {batch_id} status={status}", flush=True)
            if status != "completed":
                all_complete = False
                continue

            output_file_id = getattr(batch, "output_file_id", None)
            if output_file_id and output_file_id not in output_seen:
                output_seen.add(output_file_id)
                text = _file_content_text(client, output_file_id)
                out_f.write(text.rstrip() + "\n")
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    record = _record_from_batch_line(json.loads(line), manifest)
                    if record is None or record["path"] in completed_paths:
                        continue
                    _append_record_csv(out_csv, record)
                    completed_paths.add(record["path"])

            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id and error_file_id not in error_seen:
                error_seen.add(error_file_id)
                text = _file_content_text(client, error_file_id)
                err_f.write(text.rstrip() + "\n")
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    record = _record_from_batch_line(json.loads(line), manifest)
                    if record is None or record["path"] in completed_paths:
                        continue
                    _append_record_csv(out_csv, record)
                    completed_paths.add(record["path"])

    if not all_complete:
        print("[batch] not all jobs are completed yet; run batch_collect again later", flush=True)
        return None

    outputs = _load_existing_records(out_csv, target_paths)
    metrics = compute_metrics_from_records(outputs, pred_key="pred_label")
    metrics_by_modality = compute_metrics_by_modality(outputs, pred_key="pred_label")
    return {
        "split": "pair_set_test",
        "n_total": int(len(dfq)),
        "label_based_metrics": metrics["label_based_metrics"],
        "metrics_by_modality": metrics_by_modality,
        "per_image": outputs,
    }


def run_query_df(client, dfq: pd.DataFrame, split_name: str, out_csv: str):
    outputs = []

    if args.limit is not None:
        dfq = dfq.head(int(args.limit)).copy()

    target_paths = set(dfq["filepath"].astype(str).tolist())
    existing_outputs = _load_existing_records(out_csv, target_paths)
    completed_paths = {str(r["path"]) for r in existing_outputs if r.get("path") is not None}
    outputs.extend(existing_outputs)
    if completed_paths:
        print(f"[resume] loaded {len(completed_paths)} existing rows from {out_csv}", flush=True)

    for i, row in dfq.iterrows():
        mod = str(row["dataset_norm"]).lower()
        image_path = str(row["filepath"])
        if image_path in completed_paths:
            continue

        print(
            f"[sample] {len(outputs)+1}/{len(dfq)} | fileindex={row['fileindex']} | modality={mod}",
            flush=True,
        )

        y_true = None if ("binarylabel" not in row or pd.isna(row["binarylabel"])) else int(row["binarylabel"])
        prompt = PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])

        raw, y_pred, meta = infer_one_openai(
            client=client,
            model=args.model,
            image_path=image_path,
            text=prompt,
        )

        outputs.append({
            "split": split_name,
            "idx": int(i),
            "fileindex": str(row["fileindex"]),
            "path": image_path,
            "modality": mod,
            "severity": str(row.get("severity_norm", "clean")),
            "y_true": None if y_true is None else int(y_true),
            "pred_label": (int(y_pred) if y_pred in (0, 1) else None),
            "raw_text": raw,
            "response_id": meta.get("response_id"),
            "response_status": meta.get("status"),
            "incomplete_details": json.dumps(meta.get("incomplete_details"), ensure_ascii=False),
            "usage": json.dumps(meta.get("usage"), ensure_ascii=False),
        })
        _append_record_csv(out_csv, outputs[-1])
        completed_paths.add(image_path)
        print(f"[append] {out_csv}", flush=True)

        if args.sleep > 0:
            time.sleep(float(args.sleep))

    metrics = compute_metrics_from_records(outputs, pred_key="pred_label")
    metrics_by_modality = compute_metrics_by_modality(outputs, pred_key="pred_label")

    return {
        "split": split_name,
        "n_total": int(len(dfq)),
        "label_based_metrics": metrics["label_based_metrics"],
        "metrics_by_modality": metrics_by_modality,
        "per_image": outputs,
    }


def save_final_results(
    res_pair_all: Dict[str, Any],
    df_pair: pd.DataFrame,
    n_clean: int,
    n_weak: int,
    clean_csv: Optional[str],
    artefact_csv: Optional[str],
    query_csv: Optional[str],
    out_csv: str,
) -> None:
    n_pairs = count_valid_pairs(df_pair)

    model_key = f"openai_zero_shot:{args.model}"
    model_result = {
        "model_type": "openai_zero_shot",
        "id": f"{args.model}:zero_shot",
        "ckpt": None,
        "config": {
            "model": args.model,
            "layer": "zero_shot",
            "adapter_on_clean": None,
            "eff_variant": None,
            "eff_img_size": None,
            "hidden": None,
            "freeze_backbone": None,
            "max_output_tokens": int(args.max_output_tokens),
            "reasoning_effort": args.reasoning_effort,
            "text_verbosity": args.text_verbosity,
            "parse_mode": args.parse_mode,
            "image_detail": args.image_detail,
            "only_modality": args.only_modality,
            "mode": args.mode,
        },
        "label_based_metrics": res_pair_all["label_based_metrics"],
        "metrics_by_modality": res_pair_all["metrics_by_modality"],
        "per_image": res_pair_all["per_image"],
    }

    results = {
        "schema_version": "1.0",
        "thr": 0.5,
        "data": {
            "query_csv": query_csv,
            "clean_csv": clean_csv,
            "artefact_csv": artefact_csv,
            "n_clean": int(n_clean),
            "n_art": int(n_weak),
            "n_total": int(len(df_pair)),
            "n_pairs": int(n_pairs),
            "only_modality": args.only_modality,
        },
        "models": {
            model_key: model_result,
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out_json}", flush=True)
    print(f"[saved] {out_csv}", flush=True)


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI()

    if args.query_csv:
        df_pair = load_csv(args.query_csv, split="test")
        df_pair = filter_modality(df_pair, args.only_modality, "query_csv")
        n_clean = int((df_pair["severity_norm"].astype(str).str.lower() == "clean").sum())
        n_weak = int((df_pair["severity_norm"].astype(str).str.lower() == "weak").sum())
        clean_csv = None
        artefact_csv = None
        query_csv = args.query_csv
    else:
        df_clean = load_csv(args.clean_csv, split="test")
        df_weak = load_csv(args.artefact_csv, split="test")

        df_clean["severity_norm"] = "clean"
        df_weak["severity_norm"] = "weak"

        df_clean = filter_modality(df_clean, args.only_modality, "clean_csv")
        df_weak = filter_modality(df_weak, args.only_modality, "artefact_csv")

        df_pair = pd.concat([df_clean, df_weak], ignore_index=True)
        n_clean = int(len(df_clean))
        n_weak = int(len(df_weak))
        clean_csv = args.clean_csv
        artefact_csv = args.artefact_csv
        query_csv = None

    out_csv = _default_out_csv()

    if args.mode == "batch_submit":
        submit_batch_jobs(client, df_pair, out_csv)
        gc.collect()
        return

    if args.mode == "batch_status":
        show_batch_status(client)
        gc.collect()
        return

    if args.mode == "batch_collect":
        res_pair_all = collect_batch_results(client, df_pair, out_csv)
        if res_pair_all is None:
            gc.collect()
            return
    else:
        res_pair_all = run_query_df(client, df_pair, "pair_set_test", out_csv)

    save_final_results(
        res_pair_all=res_pair_all,
        df_pair=df_pair,
        n_clean=n_clean,
        n_weak=n_weak,
        clean_csv=clean_csv,
        artefact_csv=artefact_csv,
        query_csv=query_csv,
        out_csv=out_csv,
    )

    gc.collect()


if __name__ == "__main__":
    main()
