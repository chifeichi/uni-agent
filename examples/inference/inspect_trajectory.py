"""Decode and compare trajectories saved in ``trajectory.npz``.

Example:

    python examples/inference/inspect_trajectory.py \
        /path/to/session/trajectory.npz \
        --tokenizer /path/to/Qwen3.5-4B

The script writes complete decoded prompts, responses, and mask-separated
segments next to the NPZ file.  It also compares each trajectory prompt with
the other trajectories' full token streams, which helps distinguish a retry
fork from a context rewrite or compaction.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROMPT_KEY = re.compile(r"^traj(\d+)_prompt_ids$")


@dataclass(frozen=True)
class SavedTrajectory:
    index: int
    prompt_ids: np.ndarray
    response_ids: np.ndarray
    response_mask: np.ndarray
    response_logprobs: np.ndarray | None

    @property
    def full_ids(self) -> np.ndarray:
        return np.concatenate((self.prompt_ids, self.response_ids))


def _one_dimensional(array: Any, *, dtype: Any) -> np.ndarray:
    return np.asarray(array, dtype=dtype).reshape(-1)


def _load_trajectories(path: Path) -> list[SavedTrajectory]:
    with np.load(path, allow_pickle=False) as archive:
        indices = sorted(
            int(match.group(1))
            for key in archive.files
            if (match := _PROMPT_KEY.fullmatch(key)) is not None
        )
        if not indices:
            raise ValueError(f"{path} does not contain any traj*_prompt_ids arrays")

        trajectories = []
        for index in indices:
            prompt_key = f"traj{index}_prompt_ids"
            response_key = f"traj{index}_response_ids"
            mask_key = f"traj{index}_response_mask"
            missing = [key for key in (prompt_key, response_key, mask_key) if key not in archive.files]
            if missing:
                raise ValueError(f"trajectory {index} is missing arrays: {', '.join(missing)}")

            prompt_ids = _one_dimensional(archive[prompt_key], dtype=np.int64)
            response_ids = _one_dimensional(archive[response_key], dtype=np.int64)
            response_mask = _one_dimensional(archive[mask_key], dtype=np.int8)
            if len(response_ids) != len(response_mask):
                raise ValueError(
                    f"trajectory {index} has {len(response_ids)} response tokens "
                    f"but {len(response_mask)} mask entries"
                )
            unexpected_mask_values = sorted(set(response_mask.tolist()) - {0, 1})
            if unexpected_mask_values:
                raise ValueError(f"trajectory {index} has unexpected response-mask values: {unexpected_mask_values}")

            logprobs_key = f"traj{index}_response_logprobs"
            response_logprobs = (
                _one_dimensional(archive[logprobs_key], dtype=np.float32) if logprobs_key in archive.files else None
            )
            if response_logprobs is not None and len(response_logprobs) != len(response_ids):
                raise ValueError(
                    f"trajectory {index} has {len(response_ids)} response tokens "
                    f"but {len(response_logprobs)} logprobs"
                )

            trajectories.append(
                SavedTrajectory(
                    index=index,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    response_logprobs=response_logprobs,
                )
            )
    return trajectories


def _decode(tokenizer: Any, token_ids: np.ndarray) -> str:
    if len(token_ids) == 0:
        return ""
    return tokenizer.decode(
        token_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    if len(mask) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1]])
    ends = np.r_[starts[1:], len(mask)]
    return [(int(start), int(end), int(mask[start])) for start, end in zip(starts, ends, strict=True)]


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    text = text.replace("\x00", "\\0")
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return f"{text[:head]}\n... <{len(text) - limit} characters omitted> ...\n{text[-tail:]}"


def _common_prefix_length(left: np.ndarray, right: np.ndarray) -> int:
    shared_length = min(len(left), len(right))
    if shared_length == 0:
        return 0
    mismatches = np.flatnonzero(left[:shared_length] != right[:shared_length])
    return int(mismatches[0]) if len(mismatches) else shared_length


def _write_trajectory_files(
    trajectory: SavedTrajectory,
    *,
    tokenizer: Any,
    output_dir: Path,
) -> list[tuple[int, int, int, str]]:
    prefix = output_dir / f"traj_{trajectory.index}"
    prompt_text = _decode(tokenizer, trajectory.prompt_ids)
    response_text = _decode(tokenizer, trajectory.response_ids)
    full_text = _decode(tokenizer, trajectory.full_ids)
    prefix.with_name(f"{prefix.name}_prompt.txt").write_text(prompt_text, encoding="utf-8")
    prefix.with_name(f"{prefix.name}_response.txt").write_text(response_text, encoding="utf-8")
    prefix.with_name(f"{prefix.name}_full.txt").write_text(full_text, encoding="utf-8")

    decoded_runs = []
    sections = []
    for run_index, (start, end, mask_value) in enumerate(_mask_runs(trajectory.response_mask)):
        kind = "MODEL_GENERATED" if mask_value == 1 else "NON_MODEL_CONTEXT"
        text = _decode(tokenizer, trajectory.response_ids[start:end])
        decoded_runs.append((start, end, mask_value, text))
        sections.append(
            f"{'=' * 88}\n"
            f"segment={run_index} kind={kind} response_tokens=[{start}:{end}] "
            f"full_tokens=[{len(trajectory.prompt_ids) + start}:{len(trajectory.prompt_ids) + end}] "
            f"token_count={end - start}\n"
            f"{'=' * 88}\n"
            f"{text}\n"
        )
    prefix.with_name(f"{prefix.name}_segments.txt").write_text("\n".join(sections), encoding="utf-8")
    return decoded_runs


def _comparison_report(
    trajectories: list[SavedTrajectory],
    *,
    tokenizer: Any,
    comparison_window: int,
) -> list[str]:
    if len(trajectories) < 2:
        return ["Only one trajectory was saved; there are no sibling trajectories to compare."]

    lines = [
        "PAIRWISE FORK ANALYSIS",
        "A target prompt matching a source full-stream prefix is strong evidence of a replay/fork.",
        "A short common prefix is more consistent with compaction or another history rewrite.",
        "",
    ]
    for source in trajectories:
        for target in trajectories:
            if source.index == target.index:
                continue
            common = _common_prefix_length(source.full_ids, target.prompt_ids)
            denominator = max(1, len(target.prompt_ids))
            coverage = common / denominator
            source_remaining = len(source.full_ids) - common
            target_remaining = len(target.prompt_ids) - common
            lines.append(
                f"traj {target.index} prompt vs traj {source.index} full stream: "
                f"common_prefix={common}/{len(target.prompt_ids)} ({coverage:.2%}), "
                f"source_tokens_after_split={source_remaining}, "
                f"target_tokens_after_split={target_remaining}"
            )

            if common == len(target.prompt_ids) and len(target.prompt_ids) <= len(source.full_ids):
                lines.append(
                    "  interpretation: target prompt is an exact prefix of the source trajectory; "
                    "this strongly resembles a replay from an earlier assistant boundary."
                )
            elif coverage >= 0.98:
                lines.append(
                    "  interpretation: prompts are nearly prefix-identical; likely replay with small "
                    "rendering/canonicalization drift."
                )
            elif coverage < 0.5:
                lines.append(
                    "  interpretation: substantial early history divergence; compaction or another "
                    "history rewrite is more likely than an exact retry."
                )
            else:
                lines.append("  interpretation: partial shared history; inspect the split windows below.")

            window_start = max(0, common - comparison_window)
            source_window = source.full_ids[window_start : common + comparison_window]
            target_window = target.prompt_ids[window_start : common + comparison_window]
            lines.append(f"  source around split:\n{_decode(tokenizer, source_window)}")
            lines.append(f"  target around split:\n{_decode(tokenizer, target_window)}")
            lines.append("")
    return lines


def _metadata_text(npz_path: Path) -> str | None:
    metadata_path = npz_path.with_name("trajectory.json")
    if not metadata_path.exists():
        return None
    try:
        return json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
    except (OSError, json.JSONDecodeError):
        return metadata_path.read_text(encoding="utf-8", errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_npz", type=Path, help="Path to a framework trajectory.npz file.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer name or local model/tokenizer directory used for the rollout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Decoded output directory (default: <NPZ directory>/trajectory_decoded).",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=400,
        help="Maximum decoded characters shown for each mask segment in report.txt (default: 400).",
    )
    parser.add_argument(
        "--comparison-window",
        type=int,
        default=64,
        help="Tokens decoded on each side of a pairwise trajectory split (default: 64).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    npz_path = args.trajectory_npz.expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"trajectory NPZ does not exist: {npz_path}")
    if args.preview_chars < 0:
        raise ValueError("--preview-chars must be non-negative")
    if args.comparison_window < 0:
        raise ValueError("--comparison-window must be non-negative")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else npz_path.parent / f"{npz_path.stem}_decoded"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required to decode token IDs; install it or run this script in the rollout environment"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    trajectories = _load_trajectories(npz_path)

    report = [f"source={npz_path}", f"tokenizer={args.tokenizer}", f"trajectories={len(trajectories)}", ""]
    metadata = _metadata_text(npz_path)
    if metadata is not None:
        report.extend(["TRAJECTORY METADATA", metadata, ""])

    report.append("TRAJECTORY CONTENT SUMMARY")
    for trajectory in trajectories:
        decoded_runs = _write_trajectory_files(trajectory, tokenizer=tokenizer, output_dir=output_dir)
        model_tokens = int(trajectory.response_mask.sum())
        context_tokens = len(trajectory.response_ids) - model_tokens
        model_runs = sum(mask_value == 1 for _, _, mask_value, _ in decoded_runs)
        report.extend(
            [
                f"traj {trajectory.index}: prompt={len(trajectory.prompt_ids)} "
                f"response={len(trajectory.response_ids)} full={len(trajectory.full_ids)} "
                f"model={model_tokens} non_model={context_tokens} model_runs={model_runs}",
            ]
        )
        if trajectory.response_logprobs is not None and len(trajectory.response_logprobs):
            report.append(
                f"  logprobs: min={float(trajectory.response_logprobs.min()):.6f} "
                f"mean={float(trajectory.response_logprobs.mean()):.6f} "
                f"max={float(trajectory.response_logprobs.max()):.6f}"
            )
        for run_index, (start, end, mask_value, text) in enumerate(decoded_runs):
            kind = "MODEL_GENERATED" if mask_value == 1 else "NON_MODEL_CONTEXT"
            report.append(
                f"  segment {run_index}: {kind} response=[{start}:{end}] tokens={end - start}\n"
                f"{_preview(text, args.preview_chars)}"
            )
        report.append("")

    report.extend(
        _comparison_report(
            trajectories,
            tokenizer=tokenizer,
            comparison_window=args.comparison_window,
        )
    )
    report_text = "\n".join(report).rstrip() + "\n"
    report_path = output_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Complete decoded trajectories written to: {output_dir}")


if __name__ == "__main__":
    main()
