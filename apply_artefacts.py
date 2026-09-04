import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torchio as tio
from torchio import Subject, ScalarImage
from scipy.ndimage import rotate
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


# ----------------------------
# Custom transforms
# ----------------------------
class FixedRotation(tio.Transform):
    def __init__(self, angle: float, fill_value: float = 0.0):
        super().__init__()
        self.angle = float(angle)
        self.fill_value = float(fill_value)

    def apply_transform(self, subject):
        image = subject.image
        data = np.squeeze(image.data.numpy())  # (H,W)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={data.shape}")

        rotated = rotate(
            data,
            self.angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=self.fill_value,
            prefilter=False
        )

        rotated = rotated[None, None, ...]
        return tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(rotated).float(), affine=image.affine)
        )
    
class RandomResizedCrop2D(tio.Transform):
    """
    Random crop (scale_factor) then resize back to original (H,W).
    Works on 2D grayscale stored as (1,1,H,W).
    """
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = float(scale_factor)

    def apply_transform(self, subject):
        image = subject.image
        data = image.data.squeeze().numpy()  # (H,W), typically float in 0..1

        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={data.shape}")

        h, w = data.shape
        crop_h = max(1, min(int(round(h * self.scale_factor)), h))
        crop_w = max(1, min(int(round(w * self.scale_factor)), w))

        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)

        cropped = data[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        out_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return tio.Subject(image=tio.ScalarImage(tensor=out_tensor, affine=image.affine))


class RandomRotation(tio.Transform):
    def __init__(self, max_abs_angle: float, fill_value: float = 0.0):
        super().__init__()
        self.max_abs_angle = float(max_abs_angle)
        self.fill_value = float(fill_value)
        self.last_angle = None

    def apply_transform(self, subject):
        image = subject.image
        data = np.squeeze(image.data.numpy())  # (H,W)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape={data.shape}")

        angle = float(np.random.uniform(-self.max_abs_angle, self.max_abs_angle))
        self.last_angle = angle

        rotated = rotate(
            data,
            angle,
            reshape=False,
            order=1,               # bilinear
            mode="constant",       # ✅ 바깥 영역 상수로 채움
            cval=self.fill_value,  # ✅ 0이면 검정
            prefilter=False
        )

        rotated = rotated[None, None, ...]
        return tio.Subject(
            image=tio.ScalarImage(tensor=torch.from_numpy(rotated).float(), affine=image.affine)
        )

# ----------------------------
# Save helper (expects 0..255 or uint8)
# ----------------------------
def save_grayscale(arr: np.ndarray, out_path: str):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)

def save_color(arr: np.ndarray, out_path: str):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(out_path)


# ----------------------------
# Apply augmentation with proper scaling:
# - input orig_arr is 0..255 float32
# - normalize to 0..1 before feeding TorchIO
# - output from TorchIO assumed ~0..1 (or at least float)
# - convert back to 0..255 for saving
# ----------------------------
def apply_aug_to_array(original_array_0_255: np.ndarray, aug):
    """
    original_array_0_255: (H,W) float32 in 0..255
    aug: TorchIO transform (or custom)
    returns: (aug_img_0_255, rotate_angle_str)
    """
    # normalize to 0..1
    x = np.asarray(original_array_0_255, dtype=np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)

    original_tensor = torch.from_numpy(x)  # (H,W) float32 0..1

    rotate_angle = ""

    # build subject as (1,1,H,W)
    subject = Subject(image=ScalarImage(tensor=original_tensor.unsqueeze(0).unsqueeze(0)))

    # motion special-case (keep your original trick, but in 0..1 scale)
    if isinstance(aug, tio.RandomMotion):
        augmented_subject = aug(subject)
        random_motion = augmented_subject.history[0]  # sampled transform
        augmented_subject2 = random_motion(subject)
        arr01 = augmented_subject2.image.data.numpy()[0, 0, :, :]
    else:
        augmented_subject = aug(subject)
        # if isinstance(aug, FixedRotation):
        #     rotate_angle = str(aug.angle)
        # elif isinstance(aug, RandomRotation):
        #     rotate_angle = str(aug.last_angle if aug.last_angle is not None else "")
        arr01 = augmented_subject.image.data.numpy()[0, 0, :, :]


    # back to 0..255 float32
    arr01 = np.asarray(arr01, dtype=np.float32)
    arr255 = np.clip(arr01, 0.0, 1.0) * 255.0
    return arr255, rotate_angle

def apply_aug_to_color_array(original_array_0_255: np.ndarray, aug):
    """
    original_array_0_255: (H,W,3) float32 RGB in 0..255
    aug: TorchIO transform (or custom)
    returns: (aug_img_0_255, rotate_angle_str)
    """
    x = np.asarray(original_array_0_255, dtype=np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)

    h, w, channels = x.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image, got shape={x.shape}")

    if isinstance(aug, RandomResizedCrop2D):
        crop_h = max(1, min(int(round(h * aug.scale_factor)), h))
        crop_w = max(1, min(int(round(w * aug.scale_factor)), w))
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
        cropped = x[start_h:start_h + crop_h, start_w:start_w + crop_w, :]
        arr01 = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    elif isinstance(aug, FixedRotation):
        arr01 = rotate(
            x,
            aug.angle,
            axes=(0, 1),
            reshape=False,
            order=1,
            mode="constant",
            cval=aug.fill_value,
            prefilter=False,
        )
    else:
        tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(1)
        subject = Subject(image=ScalarImage(tensor=tensor))

        if isinstance(aug, tio.RandomMotion):
            augmented_subject = aug(subject)
            sampled_motion = augmented_subject.history[0]
            augmented_subject = sampled_motion(subject)
        else:
            augmented_subject = aug(subject)

        arr01 = (
            augmented_subject.image.data.numpy()[:, 0, :, :]
            .transpose(1, 2, 0)
        )

    arr255 = np.clip(np.asarray(arr01, dtype=np.float32), 0.0, 1.0) * 255.0
    return arr255, ""


# ----------------------------
# Factory for augmentations
# ----------------------------
def build_augmentations(severity_name: str):
    presets = {
        "weak": {
            "Noise": tio.RandomNoise(mean=0, std=0.20),
            "Motion": tio.RandomMotion(degrees=30, translation=30, num_transforms=8),
            "BiasField": tio.RandomBiasField(coefficients=0.7),
            "Crop": RandomResizedCrop2D(0.8),
            "Rotate": FixedRotation(30),
        },
        "strong": {
            "Noise": tio.RandomNoise(mean=0, std=0.35),
            "Motion": tio.RandomMotion(degrees=50, translation=50, num_transforms=10),
            "BiasField": tio.RandomBiasField(coefficients=1.2),
            "Crop": RandomResizedCrop2D(0.6),
            "Rotate": FixedRotation(45),
        }
    }
    if severity_name not in presets:
        raise ValueError(f"Unknown severity: {severity_name}")
    return presets[severity_name]


# ----------------------------
# Worker: process ONE source image row -> multiple artefact rows
# ----------------------------
def process_one_row(
    row_dict,
    artefacts,
    out_img_root,
    severity_name,
    preserve_color=False,
    data_root=None,
):
    src_p = Path(str(row_dict["filepath"])).expanduser()
    if not src_p.is_absolute() and data_root is not None:
        src_p = Path(data_root).expanduser() / src_p
    if not src_p.exists():
        return []

    image_mode = "RGB" if preserve_color else "L"
    original_image = Image.open(src_p).convert(image_mode)
    original_image = resize_keep_ratio_no_pad(
        original_image,
        max_side=384,
        image_mode=image_mode,
    )
    orig_arr = np.array(original_image, dtype=np.float32)

    dataset = str(row_dict.get("dataset", "UNKNOWN"))
    # dataset_detail = str(row_dict.get("dataset_detail", "unknown"))
    binarylabel = str(row_dict.get("binarylabel", "unknown")).lower()

    source_stem = src_p.stem
    row_fileindex = row_dict.get("fileindex")
    base_fileindex = (
        source_stem
        if row_fileindex is None or pd.isna(row_fileindex) or not str(row_fileindex).strip()
        else str(row_fileindex)
    )
    ext = src_p.suffix.lower() if src_p.suffix else ".png"

    aug_map = build_augmentations(severity_name)

    out_rows = []
    for art in artefacts:
        if art not in aug_map:
            raise KeyError(f"Artefact '{art}' not in aug_map keys: {list(aug_map.keys())}")

        aug = aug_map[art]
        if preserve_color:
            aug_img_255, rotate_angle = apply_aug_to_color_array(orig_arr, aug)
        else:
            aug_img_255, rotate_angle = apply_aug_to_array(orig_arr, aug)

        save_dir = Path(out_img_root) / dataset / binarylabel
        save_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{source_stem}__{art}__{severity_name}{ext}"
        out_path = save_dir / out_name

        # overwrite ok
        if preserve_color:
            save_color(aug_img_255, str(out_path))
        else:
            save_grayscale(aug_img_255, str(out_path))

        row_out = dict(row_dict)
        row_out.update({
            "corruption": art,
            "severity": severity_name,
            "fileindex": base_fileindex,
            "filename": out_name,
            "filepath": str(out_path),
            "binarylabel": binarylabel,
            # "rotate_angle": rotate_angle if art in ["Rotate_pos", "Rotate_neg"] else "",
        })
        out_rows.append(row_out)

    return out_rows

def resize_keep_ratio_no_pad(pil_img, max_side=384, image_mode="L"):
    pil_img = pil_img.convert(image_mode)
    w, h = pil_img.size

    if max(w, h) <= max_side:
        return pil_img

    scale = max_side / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return pil_img.resize((new_w, new_h), Image.BILINEAR)


# ----------------------------
# Main generator
# ----------------------------
def generate_artefacts_from_csv(
    in_csv: str,
    out_csv: str,
    out_img_root: str,
    severity_name: str = "weak",
    artefacts=None,
    num_workers: int = 0,   # 0 = single; >0 = parallel
    preserve_color: bool = False,
    data_root: str | None = None,
):
    if artefacts is None:
        artefacts = ["Noise", "BiasField", "Crop", "Motion", "Rotate"]
    df = pd.read_csv(in_csv)

    # test 데이터만
    # df = df[df["use_type"].astype(str).str.lower() == "test"].copy()

    out_img_root = Path(out_img_root)
    out_img_root.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict(orient="records")
    all_out = []

    if num_workers and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(
                    process_one_row,
                    r,
                    artefacts,
                    str(out_img_root),
                    severity_name,
                    preserve_color,
                    data_root,
                )
                for r in rows
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    all_out.extend(res)
    else:
        for r in rows:
            res = process_one_row(
                r,
                artefacts,
                str(out_img_root),
                severity_name,
                preserve_color,
                data_root,
            )
            if res:
                all_out.extend(res)

    out_df = pd.DataFrame(all_out)

    # IDX: 있으면 덮어쓰기, 없으면 생성
    if len(out_df) > 0:
        out_df["IDX"] = np.arange(len(out_df), dtype=int)
        out_df = out_df[["IDX"] + [c for c in out_df.columns if c != "IDX"]]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] rows={len(out_df)} saved -> {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate classic medical-image artefacts from a metadata CSV."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional root used to resolve relative image paths in the input CSV",
    )
    parser.add_argument("--severity", choices=["weak", "strong"], default="weak")
    parser.add_argument(
        "--artefacts",
        nargs="+",
        choices=["Noise", "BiasField", "Crop", "Motion", "Rotate"],
        default=["Noise", "BiasField", "Crop", "Motion", "Rotate"],
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preserve_color", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_artefacts_from_csv(
        in_csv=args.input_csv,
        out_csv=args.output_csv,
        out_img_root=args.output_dir,
        severity_name=args.severity,
        artefacts=args.artefacts,
        num_workers=args.num_workers,
        preserve_color=args.preserve_color,
        data_root=args.data_root,
    )
