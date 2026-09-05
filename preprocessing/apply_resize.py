import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image


# -------------------------
# resize function (no padding)
# -------------------------
def resize_keep_ratio_no_pad(pil_img, max_side=384):
    """
    Keep aspect ratio.
    - If max(w,h) <= max_side: return original size (converted to L).
    - Else: downscale so that max(w,h) == max_side.
    No padding.
    """
    pil_img = pil_img.convert("L")
    w, h = pil_img.size

    if max(w, h) <= max_side:
        return pil_img

    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return pil_img.resize((new_w, new_h), Image.BILINEAR)


# -------------------------
# gaussian noise function
# -------------------------
def add_gaussian_noise(pil_img, mean=0.0, std=20.0, rng=None):
    """
    Add Gaussian noise to a grayscale image.
    mean/std are pixel intensity values on the 0..255 scale.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    noise = rng.normal(loc=mean, scale=std, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy, mode="L")


# -------------------------
# main resize + gaussian noise function
# -------------------------
def resize_images_from_csv(
    csv_path,
    output_root,
    out_csv_path,
    max_side=384,
    gaussian_mean=0.0,
    gaussian_std=20.0,
    seed=1234,
    severity_name="weak",
    data_root=None,
):

    df = pd.read_csv(csv_path)
    output_root = Path(output_root)
    rng = np.random.default_rng(seed)

    new_rows = []

    for _, row in df.iterrows():

        img_path = Path(str(row["filepath"])).expanduser()
        if not img_path.is_absolute() and data_root is not None:
            img_path = Path(data_root).expanduser() / img_path
        dataset = str(row["dataset"])
        label = str(row["binarylabel"]).lower()

        if not img_path.exists():
            print("skip:", img_path)
            continue

        # load image
        img = Image.open(img_path)

        # resize (no padding)
        img_resized = resize_keep_ratio_no_pad(img, max_side=max_side)

        # add gaussian noise
        img_noisy = add_gaussian_noise(
            img_resized,
            mean=gaussian_mean,
            std=gaussian_std,
            rng=rng,
        )

        # save path
        save_dir = output_root / dataset / label
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / img_path.name

        # (optional) keep format consistent
        # If you want to force PNG output, uncomment next line and change suffix.
        # save_path = save_path.with_suffix(".png")

        img_noisy.save(save_path)

        # ---- new csv row ----
        new_row = row.copy()
        new_row["filepath"] = str(save_path)
        new_row["corruption"] = "gaussian_noise"
        new_row["severity"] = severity_name
        new_row["corruption_severity"] = f"std_{gaussian_std:g}"
        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    print(f"Done. images: {len(new_df)}")
    print(f"Gaussian noise: mean={gaussian_mean}, std={gaussian_std}, seed={seed}")
    print(f"CSV saved -> {out_csv_path}")


# -------------------------
# run
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Resize images and add Gaussian noise from a metadata CSV."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--data_root",
        default=None,
        help="Optional root used to resolve relative image paths in the input CSV",
    )
    parser.add_argument("--max_side", type=int, default=384)
    parser.add_argument("--gaussian_mean", type=float, default=0.0)
    parser.add_argument("--gaussian_std", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--severity_name", default="weak")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resize_images_from_csv(
        csv_path=args.input_csv,
        output_root=args.output_dir,
        out_csv_path=args.output_csv,
        max_side=args.max_side,
        gaussian_mean=args.gaussian_mean,
        gaussian_std=args.gaussian_std,
        seed=args.seed,
        severity_name=args.severity_name,
        data_root=args.data_root,
    )
