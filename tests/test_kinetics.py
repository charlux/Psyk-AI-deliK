"""Unit tests for the dimensionless kinetics control curve."""
import unittest

from psychaidelique.kinetics import (
    DEFAULT_PROFILES,
    KineticProfile,
    dose_adjusted_profile,
)


class KineticProfileTests(unittest.TestCase):
    def test_zero_step_equals_baseline(self):
        p = KineticProfile("test", 2, 5, 3, intensity=0.8, baseline=0.1)
        self.assertAlmostEqual(p.value_at(0), 0.1)

    def test_onset_is_monotonic_until_plateau(self):
        p = KineticProfile("test", 2, 8, 4)
        values = [p.value_at(x) for x in range(9)]
        self.assertEqual(values, sorted(values))

    def test_decay_is_monotonic_after_plateau(self):
        p = KineticProfile("test", 2, 5, 3)
        values = [p.value_at(x) for x in range(5, 15)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_dose_scales_intensity_not_time_constants(self):
        p = KineticProfile("test", 2, 5, 3, intensity=0.8)
        scaled = dose_adjusted_profile(p, 0.5)
        self.assertAlmostEqual(scaled.intensity, 0.4)
        self.assertEqual(scaled.onset_tau, p.onset_tau)
        self.assertEqual(scaled.decay_tau, p.decay_tau)

    def test_invalid_dose_rejected(self):
        p = DEFAULT_PROFILES["lsd"]
        with self.assertRaises(ValueError):
            dose_adjusted_profile(p, 1.1)

    def test_invalid_time_rejected(self):
        with self.assertRaises(ValueError):
            DEFAULT_PROFILES["lsd"].value_at(-1)

    def test_profiles_are_available(self):
        self.assertIn("dmt_inhaled", DEFAULT_PROFILES)
        self.assertIn("ayahuasca_oral", DEFAULT_PROFILES)


if __name__ == "__main__":
    unittest.main()
