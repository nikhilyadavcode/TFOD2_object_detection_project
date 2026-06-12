from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from tfod2_demo.detector import TFOD2Detector
from tfod2_demo.drawing import draw_detections
from tfod2_demo.io_utils import list_images, read_image_rgb, write_image_rgb, write_json
from tfod2_demo.labels import load_label_map
from tfod2_demo.model_zoo import available_models, ensure_model, get_model_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_LABELS = PROJECT_ROOT / "assets" / "labels" / "coco_label_map.pbtxt"
DEFAULT_MODEL = "ssd_mobilenet_v2"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfod2-demo",
        description="TensorFlow Object Detection 2 tasks for VS Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup-model", help="Download a TFOD2 model zoo SavedModel.")
    _add_model_args(setup)
    setup.set_defaults(handler=handle_setup_model)

    image = subparsers.add_parser("detect-image", help="Run object detection on one image.")
    _add_inference_args(image)
    image.add_argument("--image", type=Path, required=True, help="Input image path.")
    image.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "detected_image.jpg")
    image.add_argument("--json", type=Path, default=PROJECT_ROOT / "output" / "detected_image.json")
    image.set_defaults(handler=handle_detect_image)

    batch = subparsers.add_parser("detect-batch", help="Run object detection on a folder of images.")
    _add_inference_args(batch)
    batch.add_argument("--input", type=Path, default=PROJECT_ROOT / "sample_images")
    batch.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "batch")
    batch.add_argument("--json", type=Path, default=PROJECT_ROOT / "output" / "batch_results.json")
    batch.set_defaults(handler=handle_detect_batch)

    video = subparsers.add_parser("detect-video", help="Run object detection on a video file.")
    _add_inference_args(video)
    video.add_argument("--video", type=Path, required=True, help="Input video path.")
    video.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "detected_video.mp4")
    video.add_argument("--display", action="store_true", help="Preview frames while processing.")
    video.set_defaults(handler=handle_detect_video)

    webcam = subparsers.add_parser("detect-webcam", help="Run live object detection from a webcam.")
    _add_inference_args(webcam)
    webcam.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    webcam.set_defaults(handler=handle_detect_webcam)

    benchmark = subparsers.add_parser("benchmark", help="Measure average inference time on one image.")
    _add_inference_args(benchmark)
    benchmark.add_argument("--image", type=Path, required=True)
    benchmark.add_argument("--runs", type=int, default=20)
    benchmark.set_defaults(handler=handle_benchmark)

    return parser


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=available_models())
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--min-score", type=float, default=0.5)


def handle_setup_model(args: argparse.Namespace) -> None:
    model_dir = ensure_model(args.model, args.models_dir)
    print(f"Model is ready: {model_dir}")


def handle_detect_image(args: argparse.Namespace) -> None:
    detector, labels = _load_detector_and_labels(args)
    image_rgb = read_image_rgb(args.image)
    result = detector.detect(image_rgb)
    annotated = draw_detections(image_rgb, result.boxes, result.classes, result.scores, labels, args.min_score)

    write_image_rgb(args.output, annotated)
    write_json(args.json, result.as_records(labels, args.min_score))
    print(f"Saved annotated image: {args.output}")
    print(f"Saved JSON results: {args.json}")


def handle_detect_batch(args: argparse.Namespace) -> None:
    detector, labels = _load_detector_and_labels(args)
    images = list_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input}")

    all_results: dict[str, list[dict[str, object]]] = {}
    for image_path in images:
        image_rgb = read_image_rgb(image_path)
        result = detector.detect(image_rgb)
        annotated = draw_detections(image_rgb, result.boxes, result.classes, result.scores, labels, args.min_score)

        output_path = args.output_dir / image_path.name
        write_image_rgb(output_path, annotated)
        all_results[str(image_path)] = result.as_records(labels, args.min_score)
        print(f"Processed {image_path} -> {output_path}")

    write_json(args.json, all_results)
    print(f"Saved batch JSON results: {args.json}")


def handle_detect_video(args: argparse.Namespace) -> None:
    detector, labels = _load_detector_and_labels(args)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        frame_count = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(frame_rgb)
            annotated_rgb = draw_detections(
                frame_rgb, result.boxes, result.classes, result.scores, labels, args.min_score
            )
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(annotated_bgr)
            frame_count += 1

            if args.display:
                cv2.imshow("TFOD2 Video Detection", annotated_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        print(f"Saved detected video: {args.output} ({frame_count} frames)")
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()


def handle_detect_webcam(args: argparse.Namespace) -> None:
    detector, labels = _load_detector_and_labels(args)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    print("Webcam detection is running. Press q in the preview window to quit.")
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(frame_rgb)
            annotated_rgb = draw_detections(
                frame_rgb, result.boxes, result.classes, result.scores, labels, args.min_score
            )
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("TFOD2 Webcam Detection", annotated_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def handle_benchmark(args: argparse.Namespace) -> None:
    detector, _ = _load_detector_and_labels(args)
    image_rgb = read_image_rgb(args.image)

    detector.detect(image_rgb)
    start = time.perf_counter()
    for _ in range(args.runs):
        detector.detect(image_rgb)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / args.runs) * 1000
    print(f"Average inference time over {args.runs} runs: {avg_ms:.2f} ms")


def _load_detector_and_labels(args: argparse.Namespace) -> tuple[TFOD2Detector, dict[int, str]]:
    model_dir = get_model_dir(args.model, args.models_dir)
    if not (model_dir / "saved_model.pb").exists():
        print(f"Model not found locally. Downloading {args.model} first...")
        model_dir = ensure_model(args.model, args.models_dir)
    labels = load_label_map(args.labels)
    detector = TFOD2Detector(model_dir)
    return detector, labels


if __name__ == "__main__":
    main()

