import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

import cv2
import skimage as sk
from scipy.ndimage import zoom as scizoom
import random


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# Utils
# -------------------------
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def resize_keep_ratio_no_pad(pil_img, max_side=384):
    w, h = pil_img.size

    if max(w, h) <= max_side:
        return pil_img

    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return pil_img.resize((new_w, new_h), Image.BILINEAR)


def clipped_zoom_rect(img, zoom_factor):
    """
    Zoom helper that works for non-square images (H,W,C).
    Keeps output size identical to input size.
    """
    h, w = img.shape[:2]
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))

    top = (h - ch) // 2
    left = (w - cw) // 2
    cropped = img[top:top + ch, left:left + cw]

    zoomed = scizoom(cropped, (zoom_factor, zoom_factor, 1), order=1)

    zh, zw = zoomed.shape[:2]
    top2 = max(0, (zh - h) // 2)
    left2 = max(0, (zw - w) // 2)
    out = zoomed[top2:top2 + h, left2:left2 + w]

    # pad if rounding makes it slightly smaller
    oh, ow = out.shape[:2]
    if oh != h or ow != w:
        pad_h = h - oh
        pad_w = w - ow
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        out = np.pad(out, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")

    return out


# -------------------------
# BaseCorruption + Corruptions
# -------------------------
class BaseCorruption:
    def __init__(self, severity_params, seed=1234):
        self.severity_params = severity_params
        self.rng = np.random.default_rng(seed)

    def apply(self, img, severity=-1, augmentation=False):
        raise NotImplementedError


class ImpulseNoise(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            c = float(self.rng.uniform(low=range_min, high=range_max, size=None))
        else:
            c = self.severity_params[severity]

        noisy_image = sk.util.random_noise(
            np.array(img) / 255.0, mode="s&p", amount=c, rng=self.rng
        )
        noisy_image = np.clip(noisy_image, 0, 1)
        return (noisy_image * 255).astype(np.uint8)


class Brightness(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            range_min, range_max = self.severity_params[0], self.severity_params[-1]
            brightness_factor = float(self.rng.uniform(low=range_min, high=range_max, size=None))
        else:
            brightness_factor = self.severity_params[severity]

        enhancer = ImageEnhance.Brightness(img)
        brightened_img = enhancer.enhance(brightness_factor)
        return np.array(brightened_img).astype(np.uint8)


class DefocusBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            radius_min, radius_max = self.severity_params[0][0], self.severity_params[-1][0]
            alias_min, alias_max = self.severity_params[0][1], self.severity_params[-1][1]
            radius = float(self.rng.uniform(low=radius_min, high=radius_max, size=None))
            alias = float(self.rng.uniform(low=alias_min, high=alias_max, size=None))
        else:
            radius, alias = self.severity_params[severity]

        img = np.array(img).astype(np.float32) / 255.0
        kernel = disk(radius=radius, alias_blur=alias)

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))

        out = np.clip(channels, 0, 1) * 255.0
        return out.astype(np.uint8)


class ZoomBlur(BaseCorruption):
    def apply(self, img, severity=-1, augmentation=False):
        if augmentation:
            min_factor = 1.0
            max_factor = self.severity_params[-1][-1]
            min_step = self.severity_params[0][1] - self.severity_params[0][0]
            max_step = self.severity_params[-1][1] - self.severity_params[-1][0]
            max_factor_sampled = float(self.rng.uniform(low=min_factor, high=max_factor, size=None))
            step = float(self.rng.uniform(low=min_step, high=max_step, size=None))
            zoom_factors = np.arange(min_factor, max_factor_sampled, step)
            if len(zoom_factors) == 0:
                zoom_factors = np.array([1.01], dtype=np.float32)
        else:
            zoom_factors = self.severity_params[severity]

        img = (np.array(img).astype(np.float32) / 255.0)
        out = np.zeros_like(img, dtype=np.float32)

        for zf in zoom_factors:
            out += clipped_zoom_rect(img, float(zf))

        img2 = (img + out) / (len(zoom_factors) + 1)
        img2 = np.clip(img2, 0, 1) * 255.0
        return img2.astype(np.uint8)


corruptions = {
    "impulse_noise": ImpulseNoise([0.01, 0.03, 0.05, 0.07, 0.1]),
    "brightness": Brightness([0.9, 0.8, 0.6, 0.4, 0.3]),
    "defocus_blur": DefocusBlur([(2, 0.1), (3, 0.1), (4, 0.1), (6, 0.1), (8, 0.1)]),
    "zoom_blur": ZoomBlur([
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ]),
}

#strong
# corruptions = {
#     "impulse_noise": ImpulseNoise([0.03, 0.05, 0.07, 0.10, 0.15]),
#     "brightness": Brightness([0.3, 0.4, 0.5, 0.6, 0.7]),
#     "defocus_blur": DefocusBlur([(3, 0.1), (5, 0.1), (7, 0.1), (9, 0.1), (11, 0.1)]),
#     "zoom_blur": ZoomBlur([
#         np.arange(1, 1.16, 0.01),
#         np.arange(1, 1.21, 0.01),
#         np.arange(1, 1.26, 0.02),
#         np.arange(1, 1.31, 0.02),
#         np.arange(1, 1.41, 0.03),
#     ]),
# }



def apply_artefact_from_csv(
    input_csv: str,
    output_csv: str,
    save_root: str,
    severity: int = 2,
    severity_name: str = "weak",
    max_side: int = 384,
    data_root: str | None = None,
):
    """
    Reads clean CSV (corruption can be 'clean') and creates 4 artefact images per row.
    Saves images to:
      save_root/<dataset>/<corruption>/<label>/<fileindex>_<corruption>_s<severity>.jpg
    Writes output CSV with updated filepath/filename/corruption/severity/resize.
    ``severity`` selects the corruption preset (0-4), while ``severity_name``
    stores the clean/weak condition expected by the evaluation scripts.
    """
    df = pd.read_csv(input_csv)

    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    data_root_path = Path(data_root).resolve() if data_root is not None else None

    if data_root_path is not None:
        try:
            save_root.resolve().relative_to(data_root_path)
        except ValueError as exc:
            raise ValueError("save_root must be inside data_root") from exc

    new_rows = []

    for _, row in df.iterrows():
        img_path = Path(str(row["filepath"]))
        if not img_path.is_absolute() and data_root_path is not None:
            img_path = data_root_path / img_path
        img = Image.open(img_path).convert("RGB")
        img = resize_keep_ratio_no_pad(img, max_side=max_side)

        base = str(row["fileindex"])
        dataset = str(row.get("dataset", "unknown")).strip()
        binarylabel = str(row.get("binarylabel", "unknown")).strip().lower()

        # 2) create 4 artefacts
        for corruption_name, corruptor in corruptions.items():
            corrupted = corruptor.apply(img, severity=severity)
            corrupted_img = Image.fromarray(corrupted)

            # 폴더: dataset/corruption/label
            save_dir = save_root / dataset / binarylabel
            save_dir.mkdir(parents=True, exist_ok=True)

            new_name = f"{base}_{corruption_name}_s{severity}.jpg"
            new_path = save_dir / new_name

            corrupted_img.save(new_path, quality=95)

            new_row = row.copy()
            new_row["filename"] = new_name
            if data_root_path is None:
                metadata_path = new_path.as_posix()
            else:
                metadata_path = new_path.resolve().relative_to(data_root_path).as_posix()
            new_row["filepath"] = metadata_path
            new_row["corruption"] = corruption_name
            new_row["severity"] = severity_name
            new_row["corruption_severity"] = int(severity)
            new_row["resize"] = f"max_side_{max_side}"

            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"done: saved {len(new_df)} rows -> {output_csv}")


# -----------------------------
# run
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate impulse-noise, brightness, defocus-blur, and zoom-blur artefacts."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--data_root",
        default=".",
        help="Root used to resolve input paths and write relative output paths",
    )
    parser.add_argument("--severity_level", type=int, choices=range(5), default=2)
    parser.add_argument("--severity_name", default="weak")
    parser.add_argument("--max_side", type=int, default=384)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_artefact_from_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        save_root=args.output_dir,
        severity=args.severity_level,
        severity_name=args.severity_name,
        max_side=args.max_side,
        data_root=args.data_root,
    )
