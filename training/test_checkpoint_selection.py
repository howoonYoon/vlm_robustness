import unittest

from training.checkpoint_selection import validation_f1, validate_selection_resume, SELECTION_METRIC


class CheckpointSelectionTests(unittest.TestCase):
    def test_equal_weighting_can_select_different_epoch_from_pooled_f1(self):
        # Five corrupted views per clean image must not give artefacts fivefold weight.
        first = {"clean": {"f1": .95}, "weak": {"f1": .65}, "overall": {"f1": .70}}
        second = {"clean": {"f1": .60}, "weak": {"f1": .90}, "overall": {"f1": .85}}
        self.assertGreater(validation_f1(first), validation_f1(second))
        self.assertAlmostEqual(validation_f1(first), .80)

    def test_missing_or_invalid_metrics_cannot_select_a_checkpoint(self):
        with self.assertRaises(KeyError):
            validation_f1({"overall": {"f1": .9}})
        for bad in (float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                validation_f1({"clean": {"f1": bad}, "weak": {"f1": .8}})

    def test_resume_does_not_mix_historical_selection_criteria(self):
        for checkpoint in ({}, {"selection_metric": "overall_f1"}):
            with self.assertRaises(ValueError):
                validate_selection_resume(checkpoint)
        validate_selection_resume({"selection_metric": SELECTION_METRIC})


if __name__ == "__main__":
    unittest.main()
