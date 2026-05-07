import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim


ASR_RE = re.compile(
    r"ASR in FAR@0\.1: ([0-9.]+), ASR in FAR@0\.01: ([0-9.]+), ASR in FAR@0\.001: ([0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ASR, SSIM, FID, and visuals for DiffAM baseline vs candidate runs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("diffam_ab_results"),
        help="Root folder containing checkpoint/ and runs/ from the saved experiment.",
    )
    parser.add_argument(
        "--baseline-tag",
        default="mt_baseline_fixed",
        help="Prefix used to locate the baseline run directory under runs/.",
    )
    parser.add_argument(
        "--candidate-tag",
        "--fft-tag",
        dest="candidate_tag",
        default="mt_fft_pilot",
        help="Prefix used to locate the candidate run directory under runs/.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Optional path to the baseline log file. Defaults to runs/baseline.log when present.",
    )
    parser.add_argument(
        "--candidate-log",
        "--fft-log",
        dest="candidate_log",
        type=Path,
        default=None,
        help="Optional path to the candidate log file. Defaults to runs/fft.log when present.",
    )
    parser.add_argument(
        "--real-makeup-dir",
        type=Path,
        default=None,
        help="Path to MT-dataset/images/makeup for standard FID. If omitted, only pairwise diagnostic FID is computed.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional output folder for prepared metric images. Defaults to <root>/metrics_eval.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for FID feature extraction.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for FID.",
    )
    parser.add_argument(
        "--pairwise-fid-dims",
        default="auto",
        help="Inception feature dims for diagnostic pairwise FID. Use auto, 64, 192, 768, or 2048.",
    )
    parser.add_argument(
        "--fixed-indices",
        default="0,10,25,50,75,99",
        help="Comma-separated sample indices to show in the fixed visual grid.",
    )
    return parser.parse_args()


def find_run(root: Path, tag: str) -> Path:
    matches = sorted((root / "runs").glob(f"{tag}*"))
    if not matches:
        raise FileNotFoundError(f"No run found for tag {tag!r} under {root / 'runs'}")
    return matches[-1]


def infer_log_path(root: Path, tag: str, explicit_path: Path | None, role: str) -> Path | None:
    if explicit_path is not None:
        resolved = explicit_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Log path does not exist: {resolved}")
        return resolved

    direct_names = []
    if role == "baseline":
        direct_names.extend(["baseline.log", f"{tag}.log"])
    else:
        direct_names.extend(["candidate.log", "fft.log", f"{tag}.log"])

    for name in direct_names:
        candidate = root / "runs" / name
        if candidate.exists():
            return candidate

    token_matches = []
    tag_tokens = [token for token in re.split(r"[_-]+", tag) if token and token != "mt"]
    for path in sorted((root / "runs").glob("*.log")):
        name = path.name.lower()
        if role in name:
            return path
        if any(token.lower() in name for token in tag_tokens):
            token_matches.append(path)

    if len(token_matches) == 1:
        return token_matches[0]
    return None


def parse_final_asr(log_path: Path | None):
    if log_path is None:
        return None

    matches = ASR_RE.findall(log_path.read_text())
    if not matches:
        raise RuntimeError(f"No ASR metrics found in {log_path}")

    far_01, far_001, far_0001 = (float(value) for value in matches[-1])
    return {
        "far_0.1": far_01,
        "far_0.01": far_001,
        "far_0.001": far_0001,
    }


def compute_asr_delta(baseline_asr, candidate_asr):
    if baseline_asr is None or candidate_asr is None:
        return None

    return {
        key: candidate_asr[key] - baseline_asr[key]
        for key in baseline_asr
    }


def latest_test_images(run_dir: Path) -> dict[int, Path]:
    image_dir = run_dir / "image_samples"
    latest: dict[int, tuple[int, Path]] = {}
    pattern = re.compile(r"test_(\d+)_2_clip_.*_(\d+)_ngen\d+\.png$")
    for path in image_dir.glob("test_*_2_clip_*.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        idx = int(match.group(1))
        it_out = int(match.group(2))
        if idx not in latest or it_out > latest[idx][0]:
            latest[idx] = (it_out, path)
    return {idx: value[1] for idx, value in latest.items()}


def original_test_images(run_dir: Path) -> dict[int, Path]:
    image_dir = run_dir / "image_samples"
    originals: dict[int, Path] = {}
    pattern = re.compile(r"test_(\d+)_0_orig\.png$")
    for path in image_dir.glob("test_*_0_orig.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        originals[int(match.group(1))] = path
    return originals


def prepare_metric_dirs(root: Path, baseline_tag: str, candidate_tag: str, work_dir: Path | None):
    baseline_run = find_run(root, baseline_tag)
    candidate_run = find_run(root, candidate_tag)

    baseline_map = latest_test_images(baseline_run)
    candidate_map = latest_test_images(candidate_run)
    original_map = original_test_images(baseline_run)

    common = sorted(set(baseline_map) & set(candidate_map) & set(original_map))
    if not common:
        raise RuntimeError("No common original/baseline/candidate test image triplets found.")

    metrics_root = work_dir or (root / "metrics_eval")
    orig_dir = metrics_root / "original_test"
    baseline_dir = metrics_root / "baseline_final"
    candidate_dir = metrics_root / "candidate_final"

    for folder in (orig_dir, baseline_dir, candidate_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    for idx in common:
        file_name = f"{idx:03d}.png"
        shutil.copy2(original_map[idx], orig_dir / file_name)
        shutil.copy2(baseline_map[idx], baseline_dir / file_name)
        shutil.copy2(candidate_map[idx], candidate_dir / file_name)

    return {
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "metrics_root": metrics_root,
        "orig_dir": orig_dir,
        "baseline_dir": baseline_dir,
        "candidate_dir": candidate_dir,
        "common_indices": common,
    }


def mean_ssim(dir_a: Path, dir_b: Path):
    values = []
    for path_a in sorted(dir_a.glob("*.png")):
        path_b = dir_b / path_a.name
        if not path_b.exists():
            continue
        img_a = np.array(Image.open(path_a).convert("RGB"))
        img_b = np.array(Image.open(path_b).convert("RGB"))
        values.append(ssim(img_a, img_b, channel_axis=-1, data_range=255))

    if not values:
        raise RuntimeError(f"No matching image pairs between {dir_a} and {dir_b}")

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": len(values),
    }


def resolve_device(choice: str) -> str:
    if choice in {"cpu", "cuda"}:
        return choice

    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_fid_dims(choice: str, n_images: int) -> int:
    if choice != "auto":
        dims = int(choice)
        if dims not in {64, 192, 768, 2048}:
            raise ValueError("--pairwise-fid-dims must be one of auto, 64, 192, 768, 2048")
        return dims

    for dims in (2048, 768, 192, 64):
        if n_images > dims:
            return dims
    return 64


def compute_fid(
    dir_a: Path,
    dir_b: Path,
    batch_size: int,
    device: str,
    dims: int,
) -> float:
    import torch  # noqa: F401
    from pytorch_fid import fid_score

    return float(
        fid_score.calculate_fid_given_paths(
            [str(dir_a), str(dir_b)],
            batch_size=batch_size,
            device=device,
            dims=dims,
        )
    )


def compute_fid_with_fallback(
    dir_a: Path,
    dir_b: Path,
    batch_size: int,
    device: str,
    dims_candidates,
):
    attempted = []
    last_error = None
    for dims in dims_candidates:
        attempted.append(dims)
        try:
            value = compute_fid(dir_a, dir_b, batch_size, device, dims)
            return value, dims, attempted
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"FID failed for dims candidates {attempted}: {last_error}"
    ) from last_error


def parse_fixed_indices(raw_value: str):
    indices = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        indices.append(int(chunk))
    return indices


def load_rgb(path: Path):
    return Image.open(path).convert("RGB")


def load_abs_diff(path_a: Path, path_b: Path):
    arr_a = np.array(load_rgb(path_a), dtype=np.int16)
    arr_b = np.array(load_rgb(path_b), dtype=np.int16)
    diff = np.abs(arr_a - arr_b).astype(np.uint8)
    return Image.fromarray(diff, mode="RGB")


def mean_abs_diff(path_a: Path, path_b: Path):
    arr_a = np.array(load_rgb(path_a), dtype=np.int16)
    arr_b = np.array(load_rgb(path_b), dtype=np.int16)
    return float(np.abs(arr_a - arr_b).mean())


def render_grid(rows, output_path: Path, candidate_label: str):
    if not rows:
        return None

    font = ImageFont.load_default()
    images = [load_rgb(rows[0]["original"]), load_rgb(rows[0]["baseline"]), load_rgb(rows[0]["candidate"])]
    cell_w, cell_h = images[0].size
    headers = ["Original", "Baseline", candidate_label, "Abs Diff"]
    left_pad = 70
    top_pad = 40
    header_pad = 28
    row_gap = 18
    col_gap = 12
    width = left_pad + (4 * cell_w) + (3 * col_gap) + 20
    height = top_pad + header_pad + len(rows) * cell_h + max(0, len(rows) - 1) * row_gap + 20
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for col, header in enumerate(headers):
        x = left_pad + col * (cell_w + col_gap)
        draw.text((x, 8), header, fill="black", font=font)

    for row_idx, row in enumerate(rows):
        y = top_pad + header_pad + row_idx * (cell_h + row_gap)
        draw.text((10, y + 8), f"idx {row['idx']}", fill="black", font=font)
        visuals = [
            load_rgb(row["original"]),
            load_rgb(row["baseline"]),
            load_rgb(row["candidate"]),
            load_abs_diff(row["baseline"], row["candidate"]),
        ]
        for col, visual in enumerate(visuals):
            x = left_pad + col * (cell_w + col_gap)
            canvas.paste(visual, (x, y))

    canvas.save(output_path)
    return output_path


def build_visual_rows(indices, orig_dir: Path, baseline_dir: Path, candidate_dir: Path):
    rows = []
    for idx in indices:
        file_name = f"{idx:03d}.png"
        orig_path = orig_dir / file_name
        baseline_path = baseline_dir / file_name
        candidate_path = candidate_dir / file_name
        if not (orig_path.exists() and baseline_path.exists() and candidate_path.exists()):
            continue
        rows.append(
            {
                "idx": idx,
                "original": orig_path,
                "baseline": baseline_path,
                "candidate": candidate_path,
            }
        )
    return rows


def select_largest_change_indices(common_indices, baseline_dir: Path, candidate_dir: Path, top_k=6):
    scored = []
    for idx in common_indices:
        file_name = f"{idx:03d}.png"
        baseline_path = baseline_dir / file_name
        candidate_path = candidate_dir / file_name
        if not (baseline_path.exists() and candidate_path.exists()):
            continue
        scored.append((mean_abs_diff(baseline_path, candidate_path), idx))

    scored.sort(reverse=True)
    return [idx for _, idx in scored[:top_k]]


def write_summary(summary_path: Path, payload):
    summary_path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    baseline_log = infer_log_path(root, args.baseline_tag, args.baseline_log, "baseline")
    candidate_log = infer_log_path(root, args.candidate_tag, args.candidate_log, "candidate")
    baseline_asr = parse_final_asr(baseline_log)
    candidate_asr = parse_final_asr(candidate_log)

    prepared = prepare_metric_dirs(
        root=root,
        baseline_tag=args.baseline_tag,
        candidate_tag=args.candidate_tag,
        work_dir=args.work_dir.resolve() if args.work_dir else None,
    )

    orig_dir = prepared["orig_dir"]
    baseline_dir = prepared["baseline_dir"]
    candidate_dir = prepared["candidate_dir"]
    common = prepared["common_indices"]

    ssim_metrics = {
        "original_vs_baseline": mean_ssim(orig_dir, baseline_dir),
        "original_vs_candidate": mean_ssim(orig_dir, candidate_dir),
        "baseline_vs_candidate": mean_ssim(baseline_dir, candidate_dir),
    }

    device = resolve_device(args.device)
    pairwise_dims = resolve_fid_dims(args.pairwise_fid_dims, len(common))
    pairwise_fid = {
        "used_dims": pairwise_dims,
        "original_vs_baseline": compute_fid(orig_dir, baseline_dir, args.batch_size, device, pairwise_dims),
        "original_vs_candidate": compute_fid(orig_dir, candidate_dir, args.batch_size, device, pairwise_dims),
        "baseline_vs_candidate": compute_fid(baseline_dir, candidate_dir, args.batch_size, device, pairwise_dims),
    }
    pairwise_fid["delta_candidate_minus_baseline"] = (
        pairwise_fid["original_vs_candidate"] - pairwise_fid["original_vs_baseline"]
    )

    standard_fid = None
    if args.real_makeup_dir is not None:
        real_makeup_dir = args.real_makeup_dir.resolve()
        if not real_makeup_dir.exists():
            raise FileNotFoundError(f"Real makeup directory does not exist: {real_makeup_dir}")
        dims_candidates = [2048, 192, 64]
        orig_value, orig_dims, orig_attempted = compute_fid_with_fallback(
            real_makeup_dir, orig_dir, args.batch_size, device, dims_candidates
        )
        baseline_value, baseline_dims, baseline_attempted = compute_fid_with_fallback(
            real_makeup_dir, baseline_dir, args.batch_size, device, dims_candidates
        )
        candidate_value, candidate_dims, candidate_attempted = compute_fid_with_fallback(
            real_makeup_dir, candidate_dir, args.batch_size, device, dims_candidates
        )
        standard_fid = {
            "real_makeup_dir": str(real_makeup_dir),
            "original_vs_real_makeup": orig_value,
            "baseline_vs_real_makeup": baseline_value,
            "candidate_vs_real_makeup": candidate_value,
            "delta_candidate_minus_baseline": candidate_value - baseline_value,
            "used_dims": {
                "original": orig_dims,
                "baseline": baseline_dims,
                "candidate": candidate_dims,
            },
            "attempted_dims": {
                "original": orig_attempted,
                "baseline": baseline_attempted,
                "candidate": candidate_attempted,
            },
        }

    fixed_indices = [idx for idx in parse_fixed_indices(args.fixed_indices) if idx in common]
    if not fixed_indices:
        fixed_indices = common[: min(6, len(common))]
    largest_change_indices = select_largest_change_indices(common, baseline_dir, candidate_dir)

    candidate_label = prepared["candidate_run"].name
    fixed_grid_path = prepared["metrics_root"] / "fixed_grid.png"
    largest_change_grid_path = prepared["metrics_root"] / "largest_change_grid.png"
    render_grid(
        build_visual_rows(fixed_indices, orig_dir, baseline_dir, candidate_dir),
        fixed_grid_path,
        candidate_label,
    )
    render_grid(
        build_visual_rows(largest_change_indices, orig_dir, baseline_dir, candidate_dir),
        largest_change_grid_path,
        candidate_label,
    )

    summary = {
        "baseline_run": str(prepared["baseline_run"]),
        "candidate_run": str(prepared["candidate_run"]),
        "baseline_log": str(baseline_log) if baseline_log else None,
        "candidate_log": str(candidate_log) if candidate_log else None,
        "image_count": len(common),
        "asr": {
            "baseline": baseline_asr,
            "candidate": candidate_asr,
            "delta": compute_asr_delta(baseline_asr, candidate_asr),
        },
        "ssim": ssim_metrics,
        "pairwise_fid": pairwise_fid,
        "standard_fid": standard_fid,
        "visuals": {
            "fixed_indices": fixed_indices,
            "largest_change_indices": largest_change_indices,
            "fixed_grid": str(fixed_grid_path),
            "largest_change_grid": str(largest_change_grid_path),
        },
    }
    write_summary(prepared["metrics_root"] / "summary.json", summary)

    print("Prepared metric folders:")
    print(f"  original : {orig_dir}")
    print(f"  baseline : {baseline_dir}")
    print(f"  candidate: {candidate_dir}")
    print(f"  images   : {len(common)}")
    if baseline_asr is not None and candidate_asr is not None:
        print()
        print("ASR")
        print(
            f"  baseline : {baseline_asr['far_0.1']:.4f} / {baseline_asr['far_0.01']:.4f} / {baseline_asr['far_0.001']:.4f}"
        )
        print(
            f"  candidate: {candidate_asr['far_0.1']:.4f} / {candidate_asr['far_0.01']:.4f} / {candidate_asr['far_0.001']:.4f}"
        )
        delta = compute_asr_delta(baseline_asr, candidate_asr)
        print(
            f"  delta    : {delta['far_0.1']:+.4f} / {delta['far_0.01']:+.4f} / {delta['far_0.001']:+.4f}"
        )
    print()
    print("SSIM")
    print(
        f"  original vs baseline : {ssim_metrics['original_vs_baseline']['mean']:.4f} +/- {ssim_metrics['original_vs_baseline']['std']:.4f}"
    )
    print(
        f"  original vs candidate: {ssim_metrics['original_vs_candidate']['mean']:.4f} +/- {ssim_metrics['original_vs_candidate']['std']:.4f}"
    )
    print(
        f"  baseline vs candidate: {ssim_metrics['baseline_vs_candidate']['mean']:.4f} +/- {ssim_metrics['baseline_vs_candidate']['std']:.4f}"
    )
    print(
        f"  delta (candidate-baseline): {ssim_metrics['original_vs_candidate']['mean'] - ssim_metrics['original_vs_baseline']['mean']:+.4f}"
    )
    print()
    print(f"Pairwise FID (device={device}, dims={pairwise_dims})")
    print(f"  original vs baseline : {pairwise_fid['original_vs_baseline']:.4f}")
    print(f"  original vs candidate: {pairwise_fid['original_vs_candidate']:.4f}")
    print(f"  baseline vs candidate: {pairwise_fid['baseline_vs_candidate']:.4f}")
    print(f"  delta (candidate-baseline): {pairwise_fid['delta_candidate_minus_baseline']:+.4f}")
    if standard_fid is None:
        print()
        print("Standard FID vs real makeup skipped: pass --real-makeup-dir /path/to/MT-dataset/images/makeup")
    else:
        print()
        print("Standard FID vs real makeup")
        print(f"  original : {standard_fid['original_vs_real_makeup']:.4f} (dims {standard_fid['used_dims']['original']})")
        print(f"  baseline : {standard_fid['baseline_vs_real_makeup']:.4f} (dims {standard_fid['used_dims']['baseline']})")
        print(f"  candidate: {standard_fid['candidate_vs_real_makeup']:.4f} (dims {standard_fid['used_dims']['candidate']})")
        print(f"  delta    : {standard_fid['delta_candidate_minus_baseline']:+.4f}")
    print()
    print(f"Fixed grid saved to: {fixed_grid_path}")
    print(f"Largest-change grid saved to: {largest_change_grid_path}")
    print(f"Summary saved to: {prepared['metrics_root'] / 'summary.json'}")


if __name__ == "__main__":
    main()
