"""Shared helpers for portable and reproducible experiment scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch


DATASET_MARKER = "vlm_prompt_dataset"


def add_reproducibility_args(
    parser,
    *,
    add_seed: bool = False,
    add_workers: bool = True,
    add_model_revision: bool = True,
) -> None:
    """Add the CLI options shared by the five public experiment scripts."""
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=(
            "Root directory containing the images. Relative CSV paths are resolved "
            "under this directory; legacy paths are stripped through the "
            f"'{DATASET_MARKER}' directory."
        ),
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory. HF_HOME is also honoured.",
    )
    if add_model_revision:
        parser.add_argument(
            "--model_revision",
            type=str,
            default=None,
            help=(
                "Hugging Face git commit/tag to load. A commit hash is recommended; "
                "the resolved commit is recorded in every result."
            ),
        )
    if add_workers:
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="DataLoader workers. The portable deterministic default is 0.",
        )
    if add_seed:
        parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_nondeterministic",
        action="store_true",
        help="Allow nondeterministic PyTorch kernels (faster on some GPUs).",
    )
    parser.add_argument(
        "--skip_path_check",
        action="store_true",
        help="Skip the early check that all image paths in each CSV exist.",
    )


def configure_hf_cache(cache_dir: Optional[str]) -> Optional[str]:
    """Configure an optional cache without requiring machine-specific variables."""
    raw = cache_dir or os.environ.get("HF_HOME")
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))
    return str(root)


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy, PyTorch and a dedicated DataLoader generator."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=False)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python in a DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def hf_revision_kwargs(revision: Optional[str]) -> Dict[str, str]:
    return {"revision": revision} if revision else {}


def _looks_absolute(raw: str) -> bool:
    return Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()


def resolve_image_path(raw: Any, *, csv_path: str, data_root: Optional[str]) -> str:
    """Resolve portable relative paths and the project's historical absolute paths."""
    value = os.path.expandvars(os.path.expanduser(str(raw).strip()))
    native = Path(value.replace("\\", os.sep).replace("/", os.sep))
    if native.exists():
        return str(native.resolve())

    root = Path(data_root).expanduser().resolve() if data_root else None
    csv_dir = Path(csv_path).expanduser().resolve().parent
    normalised = value.replace("\\", "/")
    parts = [part for part in normalised.split("/") if part not in {"", "."}]

    candidates = []
    marker_indices = [i for i, part in enumerate(parts) if part.lower() == DATASET_MARKER]
    if root is not None and marker_indices:
        tail = parts[marker_indices[-1] + 1 :]
        candidates.append(root.joinpath(*tail))
    if not _looks_absolute(value):
        relative_parts = [part for part in parts if part != ".."]
        if root is not None:
            candidates.append(root.joinpath(*relative_parts))
        candidates.append(csv_dir.joinpath(*relative_parts))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    if candidates:
        return str(candidates[0].resolve(strict=False))
    return str(native)


def prepare_dataframe_paths(
    df: pd.DataFrame,
    *,
    csv_path: str,
    data_root: Optional[str],
    check_exists: bool,
    column: str = "filepath",
) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"{csv_path}: missing required column '{column}'")
    out = df.copy()
    out[column] = out[column].map(
        lambda value: resolve_image_path(value, csv_path=csv_path, data_root=data_root)
    )
    if check_exists:
        missing = [path for path in out[column].astype(str) if not Path(path).is_file()]
        if missing:
            preview = "\n  - ".join(missing[:5])
            raise FileNotFoundError(
                f"{csv_path}: {len(missing)} image path(s) do not exist. "
                "Store paths relative to --data_root or pass the correct --data_root."
                f"\n  - {preview}"
            )
    return out


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file to avoid partial results."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, destination)


def torch_save_atomic(payload: Any, path: str | Path) -> None:
    """Write a PyTorch checkpoint atomically on the destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def resolved_model_revision(model: Any, requested: Optional[str]) -> Optional[str]:
    config = getattr(model, "config", None)
    return getattr(config, "_commit_hash", None) or requested


def _distribution_versions() -> Dict[str, Optional[str]]:
    names = [
        "accelerate",
        "bitsandbytes",
        "numpy",
        "pandas",
        "Pillow",
        "safetensors",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
    ]
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_state(script_path: str | Path) -> Dict[str, Any]:
    directory = Path(script_path).resolve().parent
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "tracked_files_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_files_dirty": None}


def build_repro_metadata(
    *,
    script_path: str | Path,
    args: Any,
    input_files: Iterable[Optional[str]],
    model_id: Optional[str] = None,
    requested_revision: Optional[str] = None,
    resolved_revision: Optional[str] = None,
    prompts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    files = []
    for raw in input_files:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        files.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    prompt_payload = dict(prompts or {})
    prompt_json = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {
            "name": Path(script_path).name,
            "sha256": sha256_file(script_path),
        },
        "git": _git_state(script_path),
        "command": [Path(sys.executable).name, *sys.argv],
        "arguments": dict(vars(args)),
        "inputs": files,
        "model": {
            "id": model_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
        },
        "prompts": prompt_payload,
        "prompts_sha256": hashlib.sha256(prompt_json).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _distribution_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_names": gpu_names,
        },
        "determinism": {
            "seed": getattr(args, "seed", None),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }


def validate_resume_config(
    saved: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    """Reject a checkpoint whose experiment-defining options changed."""
    if not saved:
        return
    mismatches = []
    for key in keys:
        if key in saved and saved.get(key) != current.get(key):
            mismatches.append(f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}")
    if mismatches:
        raise ValueError("Resume configuration mismatch:\n  - " + "\n  - ".join(mismatches))
