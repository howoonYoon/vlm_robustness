# VLM Robustness to Medical Image Artefacts

Code for evaluating vision-language models (VLMs) on clean and artefact-corrupted medical images and applying training-free and parameter-efficient adaptation methods.

Supported methods include:

- Zero-shot prompting
- Few-shot prompting
- Vision-language projector tuning
- Low-Rank Adaptation (LoRA)
- Final-representation adapter tuning

All commands below assume that they are executed from the repository root.

## 1. Overview

The code provides pipelines for:

- preparing clean and artefact-corrupted image metadata,
- generating image artefacts,
- running zero-shot and few-shot VLM inference,
- training parameter-efficient adaptation methods, and
- evaluating classification performance and robustness.

Datasets are supplied through metadata CSV files, allowing the same scripts to be used with different medical image datasets.

## 2. Installation

Python 3.11 or later is recommended.

```bash
python -m pip install -r Codes/reproducible/requirements.txt
```

For experiments requiring 4-bit model loading:

```bash
python -m pip install -r Codes/reproducible/requirements-4bit.txt
```

For GPU execution, use a PyTorch build compatible with the CUDA version installed on your system.

## 3. Data Preparation

Original medical images are not included in this repository. Organise images under a local data root, for example:

```text
data/
├── mri/
├── oct/
└── xray/
```

The directory names above are examples only. Supply the root directory when running a script:

```text
--data_root data/
```

Image paths in metadata CSV files should be relative to `--data_root` rather than machine-specific absolute paths. For example, use `mri/image_001.png` instead of `/home/user/project/data/mri/image_001.png`.

Clean and artefact-corrupted versions of the same source image must remain in the same dataset split. Create patient-level splits before generating artefacts to avoid data leakage.

## 4. Metadata CSV Format

Training and evaluation data are specified using CSV metadata files. See [example_train_metadata.csv](example_train_metadata.csv) for a complete example.

```csv
dataset,filepath,fileindex,use_type,binarylabel,severity,corruption
mri,mri/001_clean.png,001,train,0,clean,none
mri,mri/001_noise.png,001,train,0,weak,gaussian_noise
mri,mri/001_motion.png,001,train,0,weak,motion
```

| Column | Description |
|---|---|
| `dataset` | Dataset or modality identifier, such as `mri`, `oct`, `xray`, or `fundus` |
| `filepath` | Image path relative to `--data_root` |
| `fileindex` | Identifier of the original source image |
| `use_type` | Dataset split: `train`, `val`, or `test` |
| `binarylabel` | `0` for normal and `1` for disease |
| `severity` | Image condition, normally `clean` or `weak` |
| `corruption` | Artefact type where applicable |

Clean and corrupted versions of the same source image are linked using `(dataset, fileindex)`.

### Few-shot metadata

Few-shot demonstration metadata requires at least:

- `dataset`
- `filepath`
- `binarylabel`

See [example_few_shot_metadata.csv](example_few_shot_metadata.csv). The CSV row order determines the order of demonstrations supplied to the model.

## 5. Artefact Generation

The following scripts generate corrupted versions of clean images while retaining their labels:

- [apply_artefacts.py](apply_artefacts.py): Gaussian noise, rotation, random crop, motion, and bias field
- [apply_new_artefacts.py](apply_new_artefacts.py): impulse noise, brightness change, defocus blur, and zoom blur
- [apply_resize.py](apply_resize.py): aspect-ratio-preserving resize with Gaussian noise

Weak and strong settings are defined in the scripts. For example, the classic artefacts can be generated with:

```bash
python Codes/reproducible/apply_artefacts.py \
  --input_csv metadata/clean.csv \
  --output_csv metadata/artefact.csv \
  --output_dir data/generated_artefacts \
  --data_root data \
  --severity weak
```

The additional artefacts can be generated with:

```bash
python Codes/reproducible/apply_new_artefacts.py \
  --input_csv metadata/clean.csv \
  --output_csv metadata/new_artefacts.csv \
  --output_dir data/new_artefacts \
  --data_root data \
  --severity_level 2 \
  --severity_name weak
```

Run any artefact script with `--help` to see its available settings.

Generated rows should retain the source image's `fileindex` and `binarylabel` so that clean and corrupted views remain paired.

## 6. Prompting Strategies

### Zero-shot prompting

Zero-shot inference and prompt-template code is provided in [adapter_test.py](adapter_test.py). Prompt templates can be edited in the source code for different zero-shot cue formulations. Precomputed zero-shot results are not included.

The base task asks the VLM to return one of two labels:

```text
normal
disease
```

### Few-shot prompting

Few-shot inference is implemented in [vlm_few_shot.py](vlm_few_shot.py). Labelled images are supplied as in-context demonstrations through a metadata CSV. The selected rows determine which images and labels are used and the order in which they are provided to the model.

## 7. Parameter-Efficient Adaptation

Three trainable adaptation methods are provided.

### Vision-language projector tuning

The vision encoder and language model remain frozen while the vision-language projector is updated. See [proj_gen_train.py](proj_gen_train.py).

### LoRA tuning

The pretrained VLM remains frozen while low-rank trainable parameters are added to selected language-model attention projections. See [lora_gen_train.py](lora_gen_train.py).

### Final-representation adapter tuning

The VLM backbone remains frozen while a lightweight adapter and binary classifier are trained on the final model representation. See [adapter_train.py](adapter_train.py).

Example adapter training using Qwen3-VL:

```bash
python Codes/reproducible/adapter_train.py \
  --backend qwen3 \
  --clean_csv metadata/clean.csv \
  --artefact_csv metadata/artefact.csv \
  --data_root data/ \
  --model_revision <HF_COMMIT_SHA> \
  --save_dir outputs/adapter \
  --tag seed42 \
  --seed 42
```

Add `--resume` to continue an interrupted run. Using an exact Hugging Face commit SHA with `--model_revision` is recommended.

## 8. Evaluation

Evaluation can be performed on any dataset following the metadata format above.

```bash
python Codes/reproducible/adapter_test.py \
  --run_vlm \
  --backend qwen3 \
  --vlm_ckpt outputs/adapter/qwen3_seed42_best.pt \
  --data_csv metadata/test.csv \
  --data_root data/ \
  --out_json outputs/test_results.json \
  --seed 42
```

The evaluation code reports metrics including:

- F1 score
- Accuracy and balanced accuracy
- Precision and recall (sensitivity)
- AUROC when probability scores are available
- Prediction consistency and flip rates
- Artefact-Induced Error Rate (AIER)

AIER is the proportion of incorrect artefact predictions among artefact views whose paired clean source image was classified correctly:

```text
AIER = harmful artefact errors / eligible artefact views
```

Lower AIER indicates greater conditional prediction stability. Results are saved to the path specified by `--out_json`. Use `--overwrite` only when intentionally replacing an existing result file.
