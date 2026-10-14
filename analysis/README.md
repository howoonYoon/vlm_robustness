# Bootstrap analysis

Run from the repository root after inference:

```bash
python -m analysis.bootstrap_metrics --manifest analysis/example_manifest.json \
  --output outputs/bootstrap_metrics.csv --n_boot 2000 --seed 42
```

Replace the example paths with saved predictions. Manifest paths are relative to the manifest file. Each entry supplies `setting` (internal/external/unseen), `family`, `model`, `method`, and `path`. Methods compare with `baseline` for the same model. `path` can be a list to combine separate modality files; duplicate conditions/views are rejected. Multi-model JSON requires `model_selector` matching its exact `models` key.

Use a separate `family` for each analysis: `adaptation`, `prompt_refinement`, `fewshot_configuration`, or `proprietary`. Holm correction includes all non-baseline conditions in each setting/family/modality, separately for each metric. Complete manuscript families contain 24, 18, 24, and 2 comparisons respectively; the two-condition example is only a template. Include a baseline for every model in every family.

Inputs may be CSV, a JSON record list, `per_image` JSON, or the release inference scripts' `models/{key}/per_image` JSON. Accepted label fields include `y_true`/`binarylabel`; prediction fields include `pred`/`pred_label`/`pred_label_parse`/`prediction`. Modality, source-image `fileindex`, severity, and corruption identify each view. Invalid predictions, ambiguous sources, conflicting labels, duplicate views, missing artefacts, and unequal comparison cohorts raise errors. Image extensions are stripped from source IDs to match CSV and JSON conventions; resulting collisions are rejected.

Unseen entries require `clean_reference`: the saved internal predictions for the same model/method. Its clean rows replace any clean rows in the unseen file. Internal/external analyses expect five designated artefacts per source image; unseen expects four. Explicit corruption is recommended; missing corruption can only be inferred from an unambiguous artefact directory in the image path.

The output gives disease-class clean F1, pooled artefact F1, and AIER, all on a 0–1 scale. AIER is harmful corrupted predictions divided by corrupted views whose source clean prediction was correct. Bootstrap draws retain all views of each sampled source image. The paired comparison uses identical sampled source indices for method and baseline; confidence intervals are 95% percentiles and two-sided sign-tail p-values use a +1 correction. Undefined AIER replicates are excluded and valid replicate counts are reported. Undefined comparison p-values stop Holm correction rather than reducing its family silently.

The adjacent `.run.json` records bootstrap settings. Seed 42 is reproducible for a fixed manifest/input order, but newly generated confidence intervals need not be bit-identical to historical tables calculated with a different random draw order. Published tables must be checked against their archived predictions.
