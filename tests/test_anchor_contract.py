from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from refer_or_clear.anchor_contract import contract_for


class AnchorContractTests(unittest.TestCase):
    def test_tracks_have_distinct_architectures_and_input_contracts(self):
        contracts = [contract_for(track) for track in ("R", "D", "N")]
        self.assertEqual(len({item.architecture_id for item in contracts}), 3)
        self.assertEqual(contracts[0].input_shape, (3, 299, 299))
        self.assertEqual(contracts[1].input_shape, (4, 512, 800))
        self.assertEqual(contracts[2].input_shape, (4, 512, 800))
        self.assertEqual([item.mask_weight for item in contracts], [0.0, 0.0, 3.0])

    def test_batch_statistics_never_enable_batched_inference(self):
        self.assertEqual(contract_for("D").bn_policy, "batch_stats")
        self.assertEqual(contract_for("D").eval_batch_size, 1)
        self.assertEqual(contract_for("N").bn_policy, "running")

    def test_contracts_are_immutable_and_unknown_tracks_fail(self):
        with self.assertRaises(FrozenInstanceError):
            contract_for("R").track = "N"
        with self.assertRaisesRegex(ValueError, "Unknown pilot track"):
            contract_for("typo")


if __name__ == "__main__":
    unittest.main()
