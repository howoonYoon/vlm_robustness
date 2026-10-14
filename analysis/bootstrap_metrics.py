"""Source-image cluster bootstrap, paired comparisons, and Holm correction.

Run: python -m analysis.bootstrap_metrics --manifest analysis/example_manifest.json
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

MODALITIES = {'brain mri': 'mri', 'retina oct': 'oct', 'retinal oct': 'oct',
              'chest x-ray': 'xray', 'chest xray': 'xray'}
ARTEFACTS = {'noise': 'gaussian_noise', 'biasfield': 'bias_field',
             'crop': 'cropping', 'rotate': 'rotation', 'motion': 'motion'}
SEEN = {'gaussian_noise', 'bias_field', 'cropping', 'rotation', 'motion'}
UNSEEN = {'brightness', 'defocus_blur', 'impulse_noise', 'zoom_blur'}
METRICS = ('clean_f1', 'artefact_f1', 'aier')


def binary(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {'normal', '0', '0.0'}:
        return 0
    if text in {'disease', '1', '1.0'}:
        return 1
    return None


def first(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value):
            return value
    return None


def read_predictions(path, model_selector=None):
    """Read CSV or per_image JSON; reject ambiguous multi-model JSON."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        records = pd.read_csv(path, dtype={'fileindex': str}).to_dict('records')
    else:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(payload, dict) and 'models' in payload:
            models = payload['models']
            if 'per_image' in models:
                payload = models
            elif model_selector is not None:
                payload = models[model_selector]
            elif len(models) == 1:
                payload = next(iter(models.values()))
            else:
                raise ValueError(f'{path}: specify model_selector for multi-model JSON')
        records = payload if isinstance(payload, list) else payload.get('per_image')
        if not isinstance(records, list):
            raise ValueError(f'{path}: expected a list or per_image JSON')
    rows = []
    for row in records:
        y = binary(first(row, ('y_true', 'binarylabel', 'binarylab', 'label')))
        pred = binary(first(row, ('pred', 'pred_label', 'pred_label_parse', 'prediction')))
        if y is None or pred is None:
            raise ValueError(f'{path}: invalid/missing binary target or prediction; no rows dropped')
        mod = str(first(row, ('modality_key', 'modality', 'dataset', 'dataset_norm'))).lower().strip()
        mod = MODALITIES.get(mod, mod)
        if mod not in {'mri', 'oct', 'xray'}:
            raise ValueError(f'{path}: unknown modality {mod}')
        fid = first(row, ('fileindex', 'source_image_id'))
        if fid is None:
            raise ValueError(f'{path}: source-image fileindex is required')
        fid = re.sub(r'\.(png|jpe?g|bmp|tiff?)$', '', str(fid).strip(), flags=re.I)
        severity = str(first(row, ('severity', 'severity_norm'))).lower()
        if severity in {'clean', 'original', 'none'}:
            corruption = 'clean'
            severity = 'clean'
        elif severity in {'weak', 'artefact', 'artifact', 'corrupted'}:
            severity = 'weak'
            corruption = first(row, ('corruption', 'artefact', 'artifact'))
            if corruption is None:
                image_path = str(first(row, ('filepath', 'path'))).replace('\\', '/').lower()
                match = re.search(r'__([^/]+?)__weak', image_path)
                stem = Path(image_path).stem
                suffix = re.sub(r'_s\d+$', '', stem[len(fid)+1:]) if stem.startswith(fid.lower()+'_') else ''
                candidates = SEEN | UNSEEN | set(ARTEFACTS)
                found = [c for c in candidates if c in image_path.split('/')
                         or c == (match.group(1) if match else '') or c == suffix]
                if len(found) != 1:
                    raise ValueError(f'{path}: corrupted rows need explicit corruption')
                corruption = found[0]
            corruption = str(corruption).strip().lower()
            corruption = ARTEFACTS.get(corruption, corruption)
        else:
            raise ValueError(f'{path}: unsupported severity {severity}')
        rows.append(dict(modality=mod, fileindex=fid, severity=severity,
                         corruption=corruption, y_true=y, pred=pred))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f'{path}: no prediction records')
    keys = ['modality', 'fileindex', 'severity', 'corruption']
    if frame.duplicated(keys).any():
        raise ValueError(f'{path}: duplicate source-image/view keys')
    if (frame.groupby(['modality', 'fileindex']).y_true.nunique() != 1).any():
        raise ValueError(f'{path}: conflicting labels for a source image')
    return frame


def contributions(frame, expected_artefacts):
    """One row per source image: clean/pooled confusion counts, AIER num/den."""
    clean = frame[frame.severity.eq('clean')].set_index('fileindex').sort_index()
    weak = frame[frame.severity.eq('weak')].copy()
    if clean.empty or weak.empty or set(clean.index) != set(weak.fileindex):
        raise ValueError('Clean and corrupted source-image cohorts must match exactly')
    sets = weak.groupby('fileindex').corruption.agg(set)
    if any(value != expected_artefacts for value in sets):
        raise ValueError('Every source image must have exactly the designated artefact views')
    labels = weak.fileindex.map(clean.y_true)
    if not (labels.to_numpy() == weak.y_true.to_numpy()).all():
        raise ValueError('Clean and corrupted source-image labels differ')
    def counts(part):
        out = part[['fileindex']].copy()
        out['tp'] = (part.y_true.eq(1) & part.pred.eq(1)).astype(int)
        out['fp'] = (part.y_true.eq(0) & part.pred.eq(1)).astype(int)
        out['fn'] = (part.y_true.eq(1) & part.pred.eq(0)).astype(int)
        return out.groupby('fileindex')[['tp', 'fp', 'fn']].sum().reindex(clean.index)
    correct = weak.fileindex.map(clean.pred.eq(clean.y_true))
    weak['num'] = (correct & weak.pred.ne(weak.y_true)).astype(int)
    weak['den'] = correct.astype(int)
    harm = weak.groupby('fileindex')[['num', 'den']].sum().reindex(clean.index)
    array = np.column_stack([counts(clean.reset_index()), counts(weak), harm])
    views = frame.sort_values(['fileindex', 'severity', 'corruption'])
    return clean.index, array, views[['fileindex', 'severity', 'corruption', 'y_true']]


def values(totals):
    totals = np.asarray(totals, dtype=float)
    def f1(start):
        tp, fp, fn = [totals[..., start + i] for i in range(3)]
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
    den = totals[..., 7]
    aier = np.divide(totals[..., 6], den, out=np.full_like(den, np.nan), where=den > 0)
    return np.stack([f1(0), f1(3), aier], axis=-1)


def draws(array, n_boot, rng, baseline=None):
    result = []
    for start in range(0, n_boot, 100):
        indices = rng.integers(0, len(array), size=(min(100, n_boot-start), len(array)))
        sample = values(array[indices].sum(axis=1))
        if baseline is not None:
            sample -= values(baseline[indices].sum(axis=1))
        result.append(sample)
    return np.concatenate(result)


def holm(p_values):
    p = np.asarray(p_values, float)
    if not np.isfinite(p).all():
        raise ValueError('Undefined comparison p-value; cannot silently shrink the Holm family')
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def analyse(manifest_path, output, n_boot=2000, seed=42):
    manifest_path = Path(manifest_path)
    entries = json.loads(manifest_path.read_text(encoding='utf-8'))
    rng = np.random.default_rng(seed)
    panels, rows = {}, []
    def load(entry, field):
        paths = entry[field] if isinstance(entry[field], list) else [entry[field]]
        return pd.concat([read_predictions(manifest_path.parent / p, entry.get('model_selector'))
                          for p in paths], ignore_index=True)
    for entry in entries:
        frame = load(entry, 'path')
        setting = entry['setting']
        expected = UNSEEN if setting == 'unseen' else SEEN
        if setting not in {'internal', 'external', 'unseen'}:
            raise ValueError(f'Unknown setting {setting}')
        if 'clean_reference' in entry:
            reference = load(entry, 'clean_reference')
            frame = pd.concat([reference[reference.severity.eq('clean')],
                               frame[frame.severity.eq('weak')]], ignore_index=True)
        if setting == 'unseen' and 'clean_reference' not in entry:
            raise ValueError('Unseen entries require saved internal clean_reference')
        for modality, part in frame.groupby('modality', sort=True):
            key = (setting, entry['family'], entry['model'], entry['method'], modality)
            if key in panels:
                raise ValueError(f'Duplicate manifest condition {key}; use a path list')
            if part.duplicated(['fileindex', 'severity', 'corruption']).any():
                raise ValueError(f'Duplicate views across source files for {key}')
            units, array, views = contributions(part, expected)
            panels[key] = (units, array, views)
            estimate = values(array.sum(axis=0))
            boot = draws(array, n_boot, rng)
            row = dict(setting=setting, family=entry['family'], model=entry['model'],
                       method=entry['method'], modality=modality, n_source_images=len(units),
                       n_artefacts=len(expected), artefact_f1_aggregation='pooled_across_artefacts')
            for i, metric in enumerate(METRICS):
                finite = boot[:, i][np.isfinite(boot[:, i])]
                low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                row.update({metric: estimate[i], metric+'_ci_low': low,
                            metric+'_ci_high': high, metric+'_valid_bootstrap': len(finite)})
            rows.append(row)
    table = pd.DataFrame(rows)
    for index, row in table.iterrows():
        if row.method == 'baseline':
            continue
        key = (row.setting, row.family, row.model, row.method, row.modality)
        baseline_key = (row.setting, row.family, row.model, 'baseline', row.modality)
        if baseline_key not in panels:
            raise ValueError(f'Missing same-model baseline for {key}')
        units, array, views = panels[key]
        base_units, base_array, base_views = panels[baseline_key]
        if not units.equals(base_units) or not views.reset_index(drop=True).equals(base_views.reset_index(drop=True)):
            raise ValueError(f'Paired comparison cohorts/views differ for {key}')
        boot = draws(array, n_boot, rng, base_array)
        diff = values(array.sum(axis=0)) - values(base_array.sum(axis=0))
        for i, metric in enumerate(METRICS):
            finite = boot[:, i][np.isfinite(boot[:, i])]
            p = min(1, 2 * min((np.count_nonzero(finite <= 0)+1)/(len(finite)+1),
                              (np.count_nonzero(finite >= 0)+1)/(len(finite)+1))) if len(finite) else np.nan
            table.loc[index, metric+'_diff_vs_base'] = diff[i]
            table.loc[index, metric+'_p_raw'] = p
    for _, group in table[table.method.ne('baseline')].groupby(['setting', 'family', 'modality']):
        for metric in METRICS:
            table.loc[group.index, metric+'_p_adj'] = holm(group[metric+'_p_raw'])
        table.loc[group.index, 'holm_family_size'] = len(group)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    output.with_suffix('.run.json').write_text(json.dumps(dict(n_boot=n_boot, seed=seed,
        manifest=str(manifest_path.resolve()), ci='95% percentile',
        p_value='two-sided paired bootstrap sign-tail with +1 correction',
        holm='per metric, setting, modality, explicit analysis family'), indent=2), encoding='utf-8')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/bootstrap_metrics.csv'))
    parser.add_argument('--n_boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error('--n_boot must be positive')
    print(analyse(args.manifest, args.output, args.n_boot, args.seed).to_string(index=False))
