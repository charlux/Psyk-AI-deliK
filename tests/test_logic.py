# tests/test_logic.py
import unittest
from psychaidelique.profiles import PsychedelicLibrary

class TestConsciousness(unittest.TestCase):
    def test_entropy_scaling(self):
        lib = PsychedelicLibrary()
        # Dose 0 : Entropie doit être 1.0 (Neutre)
        res_sobriety = lib.get_dose_response("lsd", 0.0)
        self.assertEqual(res_sobriety["current_entropy"], 1.0)
        
        # Dose 1.0 LSD : Doit atteindre 2.2
        res_full = lib.get_dose_response("lsd", 1.0)
        self.assertAlmostEqual(res_full["current_entropy"], 2.2)

if __name__ == "__main__":
    unittest.main()
