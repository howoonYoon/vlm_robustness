import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from analysis.bootstrap_metrics import read_predictions, contributions, values, holm, analyse, SEEN


class BootstrapTests(unittest.TestCase):
    def records(self):
        rows = []
        for fid, y, clean_pred, wrong in [('a', 1, 1, 2), ('b', 0, 0, 1), ('c', 1, 0, 0)]:
            rows.append(dict(dataset='MRI', fileindex=fid+'.png', severity='clean',
                             y_true=y, pred_label=clean_pred, corruption='clean'))
            for i, corruption in enumerate(sorted(SEEN)):
                rows.append(dict(dataset='MRI', fileindex=fid, severity='weak',
                                 y_true=y, pred_label=1-y if i < wrong else y, corruption=corruption))
        return rows

    def test_metrics_pairing_and_full_cli_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'prediction.json'
            path.write_text(json.dumps({'models': {'model': {'per_image': self.records()}}}))
            frame = read_predictions(path)
            _, array, _ = contributions(frame, SEEN)
            np.testing.assert_allclose(values(array.sum(axis=0)), [2/3, 16/19, .3])
            entries = [dict(setting='internal', family='adaptation', model='test',
                            method=m, path='prediction.json') for m in ['baseline', 'fewshot']]
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(entries))
            out = analyse(manifest, root/'metrics.csv', n_boot=200, seed=42)
            np.testing.assert_allclose(out.loc[1, ['clean_f1_diff_vs_base', 'artefact_f1_diff_vs_base', 'aier_diff_vs_base']].astype(float), 0)
            np.testing.assert_allclose(out.loc[1, ['clean_f1_p_adj', 'artefact_f1_p_adj', 'aier_p_adj']].astype(float), 1)
            second = analyse(manifest, root/'again.csv', n_boot=200, seed=42)
            self.assertTrue(out.equals(second))

    def test_holm(self):
        np.testing.assert_allclose(holm([.03, .001, .04]), [.06, .003, .06])

    def test_duplicates_and_ambiguous_models_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pred.json'
            path.write_text(json.dumps(self.records()+[self.records()[0]]))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_predictions(path)
            path.write_text(json.dumps({'models': {'a': {'per_image': self.records()}, 'b': {'per_image': self.records()}}}))
            with self.assertRaisesRegex(ValueError, 'model_selector'):
                read_predictions(path)


if __name__ == '__main__':
    unittest.main()
