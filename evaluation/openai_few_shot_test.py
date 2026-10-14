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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=".", help="Root for relative image paths in CSVs")
    p.add_argument("--model", type=str, default="gpt-5.5",
                   help="OpenAI vision-capable model name, e.g. gpt-5.5")
    p.add_argument("--demo_csv", type=str, required=True, help="few_shot_examples.csv")
    p.add_argument("--query_csv", type=str, default=None,
                   help="single test csv. If set, --clean_csv/--artefact_csv are not required")
    p.add_argument("--clean_csv", type=str, default=None, help="clean csv (severity clean)")
    p.add_argument("--artefact_csv", type=str, default=None, help="artefact/weak csv (severity weak)")
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--out_csv", type=str, default=None, help="optional per-image CSV output path")
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
                   help="optional modality filter, e.g. mri/oct/xray. Default is mri for quick testing")
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


def build_demos_by_modality(df_demo: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    demos_by_mod: Dict[str, List[Dict[str, Any]]] = {}
    for mod, g in df_demo.groupby("dataset_norm"):
        demos = []
        for _, row in g.iterrows():
            y = int(row["binarylabel"])
            ans = "disease" if y == 1 else "normal"
            prompt = SYSTEM_PROMPT_SHORT + "\n" + PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])
            demos.append({
                "image_url": image_to_data_url(str(row["filepath"])),
                "path": str(row["filepath"]),
                "text": prompt,
                "answer": ans,
            })
        demos_by_mod[str(mod)] = demos
    return demos_by_mod


def build_openai_input_with_demos(
    demos: List[Dict[str, Any]],
    query_image_url: str,
    query_text: str,
    image_detail: str,
    assistant_content_style: str = "output_text",
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for ex in demos:
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": ex["image_url"], "detail": image_detail},
                {"type": "input_text", "text": ex["text"]},
            ],
        })
        if assistant_content_style == "string":
            messages.append({"role": "assistant", "content": ex["answer"]})
        else:
            messages.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": ex["answer"]}],
            })

    messages.append({
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": query_image_url, "detail": image_detail},
            {"type": "input_text", "text": query_text},
        ],
    })
    return messages


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


def infer_one_openai(client, model: str, demos, image_path: str, text: str) -> Tuple[str, Optional[int], Dict[str, Any]]:
    query_image_url = image_to_data_url(image_path)

    def _create(style: str, max_output_tokens: int):
        kwargs = {
            "model": model,
            "instructions": SYSTEM_PROMPT_SHORT,
            "input": build_openai_input_with_demos(
                demos=demos,
                query_image_url=query_image_url,
                query_text=text,
                image_detail=args.image_detail,
                assistant_content_style=style,
            ),
            "max_output_tokens": int(max_output_tokens),
            "text": {"verbosity": args.text_verbosity},
        }
        if args.reasoning_effort:
            kwargs["reasoning"] = {"effort": args.reasoning_effort}
        return client.responses.create(**kwargs)

    try:
        response = _create("output_text", int(args.max_output_tokens))
        assistant_style = "output_text"
    except Exception as e:
        # Some SDK/API versions accept prior assistant messages more reliably as plain strings.
        if "output_text" not in str(e).lower() and "assistant" not in str(e).lower():
            raise
        response = _create("string", int(args.max_output_tokens))
        assistant_style = "string"

    raw = (getattr(response, "output_text", None) or "").strip()
    if not raw and hasattr(response, "model_dump"):
        dumped = response.model_dump()
        raw = _extract_text_from_response_dump(dumped).strip()

    pred_label = _parse_normal_disease(raw)
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return raw, (None if pred_label is None else int(pred_label)), {
        "assistant_content_style": assistant_style,
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


def _normalize_resume_path(path: Any) -> str:
    return os.path.normcase(os.path.normpath(str(path).strip()))


def _resume_key(fileindex: Any, severity: Any, path: Any) -> Tuple[str, str, str]:
    return (
        str(fileindex).strip(),
        str(severity).strip().lower(),
        _normalize_resume_path(path),
    )


def _resume_key_from_record(record: Dict[str, Any]) -> Tuple[str, str, str]:
    return _resume_key(
        record.get("fileindex", ""),
        record.get("severity", ""),
        record.get("path", ""),
    )


def _resume_key_from_row(row: pd.Series) -> Tuple[str, str, str]:
    return _resume_key(
        row.get("fileindex", ""),
        row.get("severity_norm", row.get("severity", "")),
        row.get("filepath", ""),
    )


def _load_resume_records(out_csv: str, wanted_keys: set) -> Tuple[List[Dict[str, Any]], set]:
    if not out_csv or not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return [], set()

    df_prev = pd.read_csv(out_csv)
    records: List[Dict[str, Any]] = []
    seen = set()
    for rec in df_prev.to_dict("records"):
        key = _resume_key_from_record(rec)
        if key not in wanted_keys or key in seen:
            continue

        if rec.get("pred_label") is not None and not pd.isna(rec.get("pred_label")):
            rec["pred_label"] = int(rec["pred_label"])
        else:
            rec["pred_label"] = None

        if rec.get("y_true") is not None and not pd.isna(rec.get("y_true")):
            rec["y_true"] = int(rec["y_true"])
        else:
            rec["y_true"] = None

        records.append(rec)
        seen.add(key)

    print(f"[resume] loaded {len(records)} completed rows from {out_csv}", flush=True)
    return records, seen


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


def run_query_df(client, demos_by_mod: dict, dfq: pd.DataFrame, split_name: str, out_csv: str):
    outputs: List[Dict[str, Any]] = []

    if args.limit is not None:
        dfq = dfq.head(int(args.limit)).copy()

    wanted_keys = {_resume_key_from_row(row) for _, row in dfq.iterrows()}
    outputs, completed_keys = _load_resume_records(out_csv, wanted_keys)
    n_skipped = 0

    for i, row in dfq.iterrows():
        row_key = _resume_key_from_row(row)
        if row_key in completed_keys:
            n_skipped += 1
            print(
                f"[skip] {n_skipped}/{len(completed_keys)} | fileindex={row['fileindex']} | severity={row.get('severity_norm', row.get('severity', ''))}",
                flush=True,
            )
            continue

        mod = str(row["dataset_norm"]).lower()
        if mod not in demos_by_mod:
            raise RuntimeError(f"demo missing for modality={mod} in demo_csv")

        print(
            f"[sample] {len(outputs)+1}/{len(dfq)} | fileindex={row['fileindex']} | modality={mod}",
            flush=True,
        )

        demos = demos_by_mod[mod]
        image_path = str(row["filepath"])
        y_true = None if ("binarylabel" not in row or pd.isna(row["binarylabel"])) else int(row["binarylabel"])
        prompt = SYSTEM_PROMPT_SHORT + "\n" + PROMPT_BY_DATASET.get(mod, PROMPT_BY_DATASET["mri"])

        raw, y_pred, meta = infer_one_openai(
            client=client,
            model=args.model,
            demos=demos,
            image_path=image_path,
            text=prompt,
        )

        record = {
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
            "assistant_content_style": meta.get("assistant_content_style"),
            "usage": json.dumps(meta.get("usage"), ensure_ascii=False),
        }
        outputs.append(record)
        completed_keys.add(row_key)
        _append_record_csv(out_csv, record)

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


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI()

    df_demo = load_csv(args.demo_csv)
    df_demo = filter_modality(df_demo, args.only_modality, "demo_csv")
    demos_by_mod = build_demos_by_modality(df_demo)
    for mod, demos in demos_by_mod.items():
        print(f"[demo] {mod}: {len(demos)} images", flush=True)

    if args.query_csv:
        df_pair = load_csv(args.query_csv, split="test")
        df_pair = filter_modality(df_pair, args.only_modality, "query_csv")
        n_clean = int((df_pair["severity_norm"].astype(str).str.lower() == "clean").sum())
        n_weak = int((df_pair["severity_norm"].astype(str).str.lower() == "weak").sum())
        clean_csv = None
        artefact_csv = None
        query_csv = args.query_csv
    else:
        if args.clean_csv is None or args.artefact_csv is None:
            raise RuntimeError("Set either --query_csv or both --clean_csv and --artefact_csv")

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

    out_csv = args.out_csv
    if not out_csv:
        root, _ = os.path.splitext(args.out_json)
        out_csv = root + ".csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    res_pair_all = run_query_df(client, demos_by_mod, df_pair, "pair_set_test", out_csv)
    n_pairs = count_valid_pairs(df_pair)

    model_key = f"openai_few_shot:{args.model}"
    model_result = {
        "model_type": "openai_few_shot",
        "id": f"{args.model}:few_shot",
        "ckpt": None,
        "config": {
            "model": args.model,
            "layer": "few_shot",
            "adapter_on_clean": None,
            "eff_variant": None,
            "eff_img_size": None,
            "hidden": None,
            "freeze_backbone": None,
            "demo_csv": args.demo_csv,
            "max_output_tokens": int(args.max_output_tokens),
            "reasoning_effort": args.reasoning_effort,
            "text_verbosity": args.text_verbosity,
            "parse_mode": args.parse_mode,
            "image_detail": args.image_detail,
            "only_modality": args.only_modality,
            "demo_summary": {k: len(v) for k, v in demos_by_mod.items()},
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

    gc.collect()


if __name__ == "__main__":
    main()
