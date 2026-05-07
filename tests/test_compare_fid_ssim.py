import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError:
    np = None
    Image = None

if np is not None and Image is not None:
    from scripts.compare_fid_ssim import (
        build_visual_rows,
        compute_asr_delta,
        compute_fid_with_fallback,
        infer_log_path,
        mean_ssim,
        parse_final_asr,
        prepare_metric_dirs,
        render_grid,
        select_largest_change_indices,
        write_summary,
    )


@unittest.skipUnless(np is not None and Image is not None, "numpy and pillow are required")
class CompareFidSsimTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "runs").mkdir()

        self.baseline_run = self.root / "runs" / "mt_baseline_fixed_demo"
        self.candidate_run = self.root / "runs" / "mt_fft_candidate_demo"
        (self.baseline_run / "image_samples").mkdir(parents=True)
        (self.candidate_run / "image_samples").mkdir(parents=True)

        self._write_image(self.baseline_run / "image_samples" / "test_0_0_orig.png", 10)
        self._write_image(self.baseline_run / "image_samples" / "test_1_0_orig.png", 20)
        self._write_image(self.baseline_run / "image_samples" / "test_0_2_clip_demo_5_ngen6.png", 12)
        self._write_image(self.baseline_run / "image_samples" / "test_1_2_clip_demo_3_ngen6.png", 25)
        self._write_image(self.baseline_run / "image_samples" / "test_1_2_clip_demo_5_ngen6.png", 22)
        self._write_image(self.baseline_run / "image_samples" / "test_2_2_clip_demo_5_ngen6.png", 30)

        self._write_image(self.candidate_run / "image_samples" / "test_0_2_clip_demo_5_ngen6.png", 14)
        self._write_image(self.candidate_run / "image_samples" / "test_1_2_clip_demo_5_ngen6.png", 28)
        self._write_image(self.candidate_run / "image_samples" / "test_2_2_clip_demo_5_ngen6.png", 32)

        (self.root / "runs" / "baseline.log").write_text(
            "ASR in FAR@0.1: 0.9100, ASR in FAR@0.01: 0.5100, ASR in FAR@0.001: 0.1100\n"
            "ASR in FAR@0.1: 0.9500, ASR in FAR@0.01: 0.7800, ASR in FAR@0.001: 0.2200\n"
        )
        (self.root / "runs" / "fft.log").write_text(
            "ASR in FAR@0.1: 0.9600, ASR in FAR@0.01: 0.7900, ASR in FAR@0.001: 0.2200\n"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_image(self, path: Path, value: int):
        array = np.full((16, 16, 3), value, dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array, mode="RGB").save(path)

    def test_parse_final_asr_uses_last_match(self):
        metrics = parse_final_asr(self.root / "runs" / "baseline.log")
        self.assertEqual(
            metrics,
            {"far_0.1": 0.95, "far_0.01": 0.78, "far_0.001": 0.22},
        )

    def test_infer_log_path_prefers_known_names(self):
        inferred = infer_log_path(self.root, "mt_baseline_fixed", None, "baseline")
        self.assertEqual(inferred, self.root / "runs" / "baseline.log")

    def test_prepare_metric_dirs_matches_common_triplets(self):
        prepared = prepare_metric_dirs(
            self.root,
            baseline_tag="mt_baseline_fixed",
            candidate_tag="mt_fft_candidate",
            work_dir=None,
        )
        self.assertEqual(prepared["common_indices"], [0, 1])
        self.assertTrue((prepared["orig_dir"] / "000.png").exists())
        self.assertTrue((prepared["baseline_dir"] / "001.png").exists())
        self.assertTrue((prepared["candidate_dir"] / "001.png").exists())

    def test_ssim_and_visual_outputs_are_generated(self):
        prepared = prepare_metric_dirs(
            self.root,
            baseline_tag="mt_baseline_fixed",
            candidate_tag="mt_fft_candidate",
            work_dir=None,
        )
        ssim_metrics = mean_ssim(prepared["orig_dir"], prepared["baseline_dir"])
        self.assertEqual(ssim_metrics["count"], 2)

        fixed_rows = build_visual_rows(
            [0, 1],
            prepared["orig_dir"],
            prepared["baseline_dir"],
            prepared["candidate_dir"],
        )
        fixed_grid = prepared["metrics_root"] / "fixed_grid.png"
        render_grid(fixed_rows, fixed_grid, "candidate")
        self.assertTrue(fixed_grid.exists())

        largest = select_largest_change_indices(
            prepared["common_indices"],
            prepared["baseline_dir"],
            prepared["candidate_dir"],
            top_k=2,
        )
        self.assertEqual(len(largest), 2)
        largest_rows = build_visual_rows(
            largest,
            prepared["orig_dir"],
            prepared["baseline_dir"],
            prepared["candidate_dir"],
        )
        largest_grid = prepared["metrics_root"] / "largest_change_grid.png"
        render_grid(largest_rows, largest_grid, "candidate")
        self.assertTrue(largest_grid.exists())

    def test_write_summary_outputs_json(self):
        summary_path = self.root / "summary.json"
        payload = {
            "asr": {
                "baseline": parse_final_asr(self.root / "runs" / "baseline.log"),
                "candidate": parse_final_asr(self.root / "runs" / "fft.log"),
                "delta": compute_asr_delta(
                    parse_final_asr(self.root / "runs" / "baseline.log"),
                    parse_final_asr(self.root / "runs" / "fft.log"),
                ),
            }
        }
        write_summary(summary_path, payload)
        loaded = json.loads(summary_path.read_text())
        self.assertEqual(loaded["asr"]["candidate"]["far_0.1"], 0.96)

    def test_fid_fallback_uses_next_available_dims(self):
        with mock.patch("scripts.compare_fid_ssim.compute_fid") as mocked_compute_fid:
            mocked_compute_fid.side_effect = [
                ValueError("Imaginary component 1.0"),
                0.42,
            ]
            value, used_dims, attempted = compute_fid_with_fallback(
                dir_a=self.root,
                dir_b=self.root,
                batch_size=4,
                device="cpu",
                dims_candidates=[2048, 192, 64],
            )

        self.assertEqual(value, 0.42)
        self.assertEqual(used_dims, 192)
        self.assertEqual(attempted, [2048, 192])


if __name__ == "__main__":
    unittest.main()
