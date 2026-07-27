import csv
import json
import tempfile
import unittest
from pathlib import Path

import moe_plan


class MoePlanTests(unittest.TestCase):
    def write_csv(self, path: Path, header, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

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
                [
                    [0, 0, 100, 10, 0.5, 0.1, 10],
                    [0, 1, 20, 2, 0.1, 0.1, 10],
                    [1, 0, 80, 8, 0.4, 0.1, 10],
                    [1, 1, 70, 7, 0.35, 0.1, 10],
                ],
            )
            self.write_csv(
                placement,
                [
                    "layer", "tensor", "expert_first", "expert_last", "storage", "buffer", "device",
                    "size_mib", "size_bytes", "n_experts", "expert_stride_bytes",
                ],
                [
                    [0, "blk.0.ffn_gate_exps.weight", 0, 1, "RAM", "CPU", "CPU", "0", 200, 2, 100],
                    [1, "blk.1.ffn_gate_exps.weight", 0, 1, "RAM", "CPU", "CPU", "0", 400, 2, 200],
                ],
            )
            rc = moe_plan.main([
                "--stats", str(stats),
                "--placement", str(placement),
                "--vram-budget-bytes", "300",
                "--output", str(out),
                "--report", str(report),
            ])
            self.assertEqual(rc, 0)
            plan = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(plan["selected_expert_count"], 2)
            self.assertEqual(plan["selected_bytes"], 300)
            selected = {
                (layer["layer"], expert)
                for layer in plan["layers"]
                for expert in layer["gpu_experts"]
            }
            self.assertEqual(selected, {(0, 0), (1, 0)})
            self.assertAlmostEqual(plan["estimated_gpu_hit_rate"], 180 / 270)

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
