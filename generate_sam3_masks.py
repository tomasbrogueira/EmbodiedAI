import argparse
from pathlib import Path

import numpy as np
from sam3.model_builder import build_sam3_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAM3 video segmentation masks."
    )

    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help=(
            "Base output directory. Masks will be saved in a "
            "<VIDEO_NAME>_masks subdirectory."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Text prompt, e.g. "path", "floor", or "ground".',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = args.video_path.expanduser().resolve()
    output_root = args.output_folder.expanduser()
    prompt = args.prompt

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    video_name = video_path.stem

    # Preserve the original output directory naming convention.
    out_dir = output_root / f"{video_name}_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAM3 VIDEO SEGMENTATION")
    print("=" * 60)
    print(f"Video:  {video_path}")
    print(f"Prompt: {prompt}")
    print(f"Output: {out_dir.resolve()}")
    print("=" * 60)

    predictor = build_sam3_video_predictor()
    session_id = None

    try:
        # Start video session.
        session = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(video_path),
            }
        )

        session_id = session["session_id"]

        # Add the prompt on the first frame.
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
            }
        )

        # Propagate segmentation through the entire video.
        frames_saved = 0

        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
            }
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]

            output_path = (
                out_dir
                / f"{video_name}_{prompt}_frame{frame_idx:06d}.npz"
            )

            np.savez_compressed(
                output_path,
                **outputs,
            )

            frames_saved += 1
            print(f"Saved frame {frame_idx}: {output_path.name}")

        print("\nSegmentation complete.")
        print(f"Frames saved: {frames_saved}")
        print(f"Output directory: {out_dir.resolve()}")

    finally:
        # Close the session even if propagation encounters an error.
        if session_id is not None:
            predictor.handle_request(
                request={
                    "type": "close_session",
                    "session_id": session_id,
                }
            )


if __name__ == "__main__":
    main()