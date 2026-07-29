import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import moe_plan


class MoePlanTests(unittest.TestCase):
    def write_csv(self, path: Path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def test_canonical_ggml_type_names(self):
        self.assertEqual(moe_plan.canonical_ggml_type_name("Q5_K"), "q5_K")
        self.assertEqual(moe_plan.canonical_ggml_type_name("Q4_0"), "q4_0")
        self.assertEqual(moe_plan.canonical_ggml_type_name("IQ4_XS"), "iq4_xs")
        self.assertEqual(moe_plan.canonical_ggml_type_name("BF16"), "bf16")

    def test_selects_best_hits_per_byte_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.csv"
            placement = root / "placement.csv"
            out = root / "plan.json"
            report = root / "report.csv"
            self.write_csv(
                stats,
                ["layer", "expert", "hits", "sampled_hits", "layer_share", "coverage", "sample_rate"],
                [[0, 0, 100, 10, 0.5, 0.1, 10], [0, 1, 20, 2, 0.1, 0.1, 10],
                 [1, 0, 80, 8, 0.4, 0.1, 10], [1, 1, 70, 7, 0.35, 0.1, 10]],
            )
            self.write_csv(
                placement,
                ["layer", "tensor", "expert_first", "expert_last", "storage", "buffer", "device",
                 "size_mib", "size_bytes", "n_experts", "expert_stride_bytes"],
                [[0, "blk.0.ffn_gate_exps.weight", 0, 1, "RAM", "CPU", "CPU", "0", 200, 2, 100],
                 [1, "blk.1.ffn_gate_exps.weight", 0, 1, "RAM", "CPU", "CPU", "0", 400, 2, 200]],
            )
            rc = moe_plan.main(["--stats", str(stats), "--placement", str(placement),
                                "--vram-budget-bytes", "300", "--output", str(out), "--report", str(report)])
            self.assertEqual(rc, 0)
            plan = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(plan["schema_version"], 1)
            self.assertEqual(plan["selected_expert_count"], 2)
            self.assertEqual(plan["selected_bytes"], 300)

    def test_exact_placement_creates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / "stats.csv"
            placement = root / "placement.csv"
            out = root / "plan.json"
            self.write_csv(
                stats,
                ["layer", "expert", "hits", "sampled_hits", "layer_share", "coverage", "sample_rate"],
                [[0, 0, 10, 1, .5, .5, 10], [0, 1, 10, 1, .5, .5, 10]],
            )
            header = ["layer", "tensor", "expert_first", "expert_last", "storage", "buffer", "device",
                      "size_mib", "size_bytes", "n_experts", "expert_stride_bytes", "type",
                      "ne0", "ne1", "ne2", "ne3", "nb0", "nb1", "nb2", "nb3"]
            rows = [[0, "blk.0.ffn_gate_exps.weight", 0, 1, "RAM", "CPU", "CPU", "0", 200, 2, 100, "q5_K",
                     64, 2, 2, 1, 176, 176, 100, 200]]
            self.write_csv(placement, header, rows)
            rc = moe_plan.main(["--stats", str(stats), "--placement", str(placement),
                                "--require-manifest", "--vram-budget-bytes", "100", "--output", str(out)])
            self.assertEqual(rc, 0)
            plan = json.loads(out.read_text())
            self.assertEqual(plan["schema_version"], 2)
            self.assertEqual(plan["tensor_manifest"][0]["type"], "q5_K")
            self.assertEqual(plan["tensor_manifest"][0]["nb"], [176, 176, 100, 200])

    def test_sampled_model_fingerprint_detects_tail_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"a" * (moe_plan.FINGERPRINT_SAMPLE_BYTES + 32))
            first = moe_plan.sampled_model_fingerprint(path)
            data = bytearray(path.read_bytes())
            data[-1] = ord("b")
            path.write_bytes(data)
            second = moe_plan.sampled_model_fingerprint(path)
            self.assertNotEqual(first, second)

    def test_missing_stats_row_is_cold_not_fatal(self):
        stats = [moe_plan.ExpertStats(0, 0, 10, 1, 1.0, 0.1, 10)]
        sizes = {(0, 0): 100, (0, 1): 100}
        candidates = moe_plan.build_candidates(stats, sizes)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1].hits, 0)

    def test_rejects_inconsistent_ranges(self):
        rows = [
            moe_plan.PlacementTensor(0, "gate", 0, 1, 200, 100, "", "", ""),
            moe_plan.PlacementTensor(0, "up", 0, 2, 300, 100, "", "", ""),
        ]
        with self.assertRaises(moe_plan.PlanError):
            moe_plan.logical_expert_sizes(rows)


if __name__ == "__main__":
    unittest.main()
