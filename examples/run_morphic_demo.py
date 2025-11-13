#!/usr/bin/env python3
"""
Quick sanity check for the Morphic Frames-to-Video integration.

This script generates a single video using the `morphic-wan2.2-i2v` model and
stores the result inside the structured VMEvalKit output directory layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

from vmevalkit.runner.inference import InferenceRunner


def _build_question_metadata(
    image: Path,
    domain: Optional[str],
    final_image: Optional[Path],
) -> Dict[str, Any]:
    question_id = image.stem
    metadata: Dict[str, Any] = {
        "id": question_id,
        "domain": domain or "custom",
    }
    if final_image:
        metadata["final_image_path"] = str(final_image.resolve())
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a sample video with the Morphic frames-to-video model."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the first frame image (PNG/JPG).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
        help="Text prompt for the video.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/morphic-demo",
        type=Path,
        help="Base directory for structured outputs.",
    )
    parser.add_argument(
        "--domain",
        default="custom",
        type=str,
        help="Optional domain label for metadata (default: custom).",
    )
    parser.add_argument(
        "--final-image",
        type=Path,
        default=None,
        help="Optional reference final frame to pass as `--img_end`.",
    )
    parser.add_argument(
        "--middle-images",
        type=Path,
        nargs="*",
        default=None,
        help="Optional intermediate frames passed to Morphic.",
    )
    parser.add_argument(
        "--middle-timestamps",
        type=float,
        nargs="*",
        default=None,
        help="Timestamps (0-1) for each intermediate frame.",
    )
    parser.add_argument(
        "--save-as",
        type=str,
        default=None,
        help="Optional custom filename for the generated video.",
    )

    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"Input image not found: {args.image}")

    if args.middle_timestamps and (
        not args.middle_images or len(args.middle_timestamps) != len(args.middle_images)
    ):
        parser.error(
            "If you provide --middle-timestamps you must also supply "
            "the same number of paths via --middle-images."
        )

    question_meta = _build_question_metadata(
        image=args.image,
        domain=args.domain,
        final_image=args.final_image,
    )

    runner = InferenceRunner(output_dir=str(args.output_dir))

    morphic_kwargs: Dict[str, Any] = {}
    if args.final_image:
        morphic_kwargs["final_image"] = str(args.final_image.resolve())
    if args.middle_images:
        morphic_kwargs["middle_images"] = [str(path.resolve()) for path in args.middle_images]
    if args.middle_timestamps:
        morphic_kwargs["middle_timestamps"] = [float(ts) for ts in args.middle_timestamps]
    if args.save_as:
        morphic_kwargs["output_filename"] = args.save_as

    result = runner.run(
        model_name="morphic-wan2.2-i2v",
        image_path=str(args.image.resolve()),
        text_prompt=args.prompt,
        question_data=question_meta,
        **morphic_kwargs,
    )

    if result.get("success"):
        print("\n✅ Morphic generation succeeded!")
        print(f"Video: {result.get('video_path')}")
        print(f"Inference folder: {result.get('inference_dir')}")
    else:
        print("\n❌ Morphic generation failed.")
        print(result.get("error", "Unknown error"))


if __name__ == "__main__":
    main()


