#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Simulation Runner
=======================
Replaces the live camera with one or more video files, runs the full
YOLO-E → bc_model pipeline on every frame, draws the command arrows /
detection overlays, and saves the result as a new video (and optional
per-frame PNGs).

No robot, no serial port, no pygame required.

Usage examples
--------------
# Single video, real models
python video_sim.py \
    --input  ./videos/case1.mp4 \
    --yoloe-model  ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
    --target "black spot" \
    --confidence-threshold 0.01

# Multiple videos in one go
python video_sim.py \
    --input ./videos/case1.mp4 ./videos/case2.mp4 \
    --yoloe-model  ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
    --target "black spot"

# Quick smoke-test with mock models (no GPU / checkpoints needed)
python video_sim.py --input ./videos/test.mp4 --mock-all --target "polyp"

# Compare bc_model vs proportional controller side-by-side
python video_sim.py \
    --input /Users/weiyuqing/Desktop/dataset/video_09/video.mp4 \
    --yoloe-model ./checkpoints/best_blackpoint_base.pt \
    --control-checkpoint ./checkpoints/bc_model.pt \
    --target "black spot" \
    --control-mode both --p-gain-x 0.6 --p-gain-y 0.6

# Save individual frames as PNG as well
python video_sim.py --input ./videos/case1.mp4 --mock-all --save-frames

Output
------
For each input video  <name>.mp4  a folder  ./sim_output/<name>_<timestamp>/
is created containing:
    annotated.mp4   – full annotated video
    frames/         – (optional) per-frame PNG files
    data.csv        – per-frame detection / action log
"""

import argparse
import os
import sys
import time
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# ── local imports ────────────────────────────────────────────────────────────
from yoloe_combined_config import get_config_from_args
from yoloe_perception_interface import (
    YOLOEPerceptionInterface, MockYOLOEPerceptionInterface,
    DetectionResult, create_yoloe_perception_interface,
)
from control_interface import (
    ControlInterface, MockControlInterface, ProportionalController,
    ControlAction, create_control_interface,
)
from safety_manager import SafetyManager, SafetyState, SafetyOutput
from data_recorder import VisualizationRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Angle utility
# ─────────────────────────────────────────────────────────────────────────────

MIN_ACTION_MAGNITUDE = 0.1  # ignore frames where either action is near-zero

def angle_between_actions(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """
    Return the signed angle in degrees FROM vector a TO vector b.
    Positive = b is counter-clockwise from a.
    Returns None if either vector magnitude is below MIN_ACTION_MAGNITUDE.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < MIN_ACTION_MAGNITUDE or norm_b < MIN_ACTION_MAGNITUDE:
        return None
    a_u = a / norm_a
    b_u = b / norm_b
    cross = a_u[0] * b_u[1] - a_u[1] * b_u[0]
    dot   = np.clip(a_u[0] * b_u[0] + a_u[1] * b_u[1], -1.0, 1.0)
    return float(np.degrees(np.arctan2(cross, dot)))


def print_angle_stats(angles: List[float], out_path: Path) -> None:
    """Print summary statistics and ASCII histogram, and save to a txt file."""
    if not angles:
        print("[AngleStats] No valid angle samples collected.")
        return

    arr = np.array(angles)
    mean   = float(np.mean(arr))
    std    = float(np.std(arr))
    median = float(np.median(arr))
    mn     = float(np.min(arr))
    mx     = float(np.max(arr))
    n      = len(arr)

    # ASCII histogram (9 bins from -180 to 180)
    bins = np.linspace(-180, 180, 10)
    counts, edges = np.histogram(arr, bins=bins)
    bar_max = max(counts) if max(counts) > 0 else 1
    bar_width = 30

    lines = []
    lines.append("=" * 55)
    lines.append("  Angle Difference Stats  (model → P-controller)")
    lines.append("=" * 55)
    lines.append(f"  Samples : {n}")
    lines.append(f"  Mean    : {mean:+.1f}°")
    lines.append(f"  Std dev : {std:.1f}°")
    lines.append(f"  Median  : {median:+.1f}°")
    lines.append(f"  Min/Max : {mn:+.1f}° / {mx:+.1f}°")
    lines.append("")
    lines.append("  Distribution:")
    for i, cnt in enumerate(counts):
        bar = "█" * int(cnt / bar_max * bar_width)
        lines.append(f"  [{edges[i]:+7.1f}°, {edges[i+1]:+7.1f}°)  {bar} {cnt}")
    lines.append("=" * 55)

    text = "\n".join(lines)
    print("\n" + text + "\n")

    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"  Angle stats saved → {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run bc_model on video file(s) and save annotated output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / output
    p.add_argument("--input", nargs="+", required=True,
                   help="Path(s) to input video file(s)")
    p.add_argument("--output-dir", default="./sim_output",
                   help="Root directory for all outputs (default: ./sim_output)")
    p.add_argument("--save-frames", action="store_true",
                   help="(Ignored — frames are always saved to frames/ subfolder)")

    # YOLO-E
    p.add_argument("--yoloe-model", default="best_blackpoint_base.pt",
                   help="Path to YOLO-E model checkpoint")
    p.add_argument("--target", default="polyp",
                   help='Text prompt for detection, e.g. "black spot"')
    p.add_argument("--confidence-threshold", type=float, default=0.25)

    # Control
    p.add_argument("--control-checkpoint", default=None,
                   help="Path to bc_model checkpoint (.pt)")
    p.add_argument("--control-mode",
                   choices=["bc_model", "proportional", "both"],
                   default="bc_model",
                   help="Which controller to run/display")
    p.add_argument("--p-gain-x", type=float, default=0.6)
    p.add_argument("--p-gain-y", type=float, default=0.6)

    # Mock / debug
    p.add_argument("--mock-perception", action="store_true")
    p.add_argument("--mock-control",    action="store_true")
    p.add_argument("--mock-all",        action="store_true",
                   help="Use mock models (no GPU / checkpoints needed)")

    # Heatmap
    p.add_argument("--show-heatmap", action="store_true",
                   help="Overlay YOLO-E attention heatmap")

    # Playback
    p.add_argument("--fps-override", type=float, default=None,
                   help="Override output video FPS (default: use input video FPS)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after this many frames per video (useful for quick tests)")
    p.add_argument("--display", action="store_true",
                   help="Show annotated frames in an OpenCV window while processing")

    # Fake mode / camera args required by get_config_from_args
    p.add_argument("--mode",      default="simulation")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-logging", action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers (extend the base VisualizationRecorder.draw_visualization)
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame(
    frame: np.ndarray,
    detection: DetectionResult,
    action: ControlAction,
    action_p: Optional[ControlAction],
    safety: SafetyOutput,
    target_name: str,
    frame_idx: int,
    fps: float,
    control_mode: str,
    show_heatmap: bool,
) -> np.ndarray:
    """Annotate a single frame with detections, arrows, and HUD text."""
    # Base visualization (bbox, crosshair, command arrow, status text)
    vis = VisualizationRecorder.draw_visualization(
        frame, detection, action, safety, show_heatmap
    )

    h, w = vis.shape[:2]
    cx, cy = w // 2, h // 2

    # ── draw proportional-controller arrow (orange) when in 'both' mode ──
    if control_mode == "both" and action_p is not None and action_p.valid:
        scale = 50
        px = int(cx - action_p.action[0] * scale)
        py = int(cy - action_p.action[1] * scale)
        cv2.arrowedLine(vis, (cx, cy), (px, py), (255, 165, 0), 2, tipLength=0.3)
        cv2.putText(vis, f"P ({action_p.action[0]:.2f},{action_p.action[1]:.2f})",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

    # ── HUD strip at bottom ──
    y = h - 30
    cv2.rectangle(vis, (0, y - 5), (w, h), (0, 0, 0), -1)

    label_map = {"bc_model": "bc_model", "proportional": "P-Ctrl", "both": "bc_model(sent)"}
    ctrl_label = label_map.get(control_mode, control_mode)

    hud = (f"Frame:{frame_idx}  FPS:{fps:.1f}  Target:{target_name}  "
           f"{ctrl_label}:({action.action[0]:.2f},{action.action[1]:.2f})  "
           f"Conf:{detection.confidence:.2f}")
    cv2.putText(vis, hud, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scale_detection(detection, src_w: int, src_h: int, dst_w: int, dst_h: int):
    """
    Return a copy of DetectionResult with bbox/center scaled from
    (src_w x src_h) model space to (dst_w x dst_h) output space.
    All other fields are passed through unchanged.
    """
    if src_w == dst_w and src_h == dst_h:
        return detection  # nothing to do

    sx = dst_w / src_w
    sy = dst_h / src_h

    new_bbox   = None
    new_center = None

    if detection.bbox is not None:
        x, y, w, h = detection.bbox
        new_bbox = np.array([x * sx, y * sy, w * sx, h * sy])

    if detection.center is not None:
        cx, cy = detection.center
        new_center = np.array([cx * sx, cy * sy])

    # Build a new DetectionResult with scaled coordinates using direct construction
    from yoloe_perception_interface import DetectionResult
    return DetectionResult(
        bbox=new_bbox,
        center=new_center,
        confidence=detection.confidence,
        detection_score=detection.detection_score,
        no_detection=detection.no_detection,
        status=detection.status,
        lost_count=detection.lost_count,
        inference_time=detection.inference_time,
        fps=detection.fps,
        heatmap=detection.heatmap,
        class_name=detection.class_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-video processor
# ─────────────────────────────────────────────────────────────────────────────

class VideoSimulator:
    def __init__(self, args):
        self.args = args
        self.config = get_config_from_args(args)

        # resolve mock flags
        self.use_mock_perception = args.mock_all or args.mock_perception
        self.use_mock_control    = args.mock_all or args.mock_control

        # guard: if no checkpoint, fall back to mock silently
        if not self.use_mock_control and not self.use_mock_perception:
            if not args.control_checkpoint:
                print("[VideoSim] No --control-checkpoint given → using mock control")
                self.use_mock_control = True
            elif not os.path.exists(args.control_checkpoint):
                print(f"[VideoSim] Checkpoint not found: {args.control_checkpoint} → mock")
                self.use_mock_control = True

        self._init_models()

    # ── model init ─────────────────────────────────────────────────────────

    def _init_models(self):
        print("\n[VideoSim] Initialising models…")

        if self.use_mock_perception:
            self.perception = MockYOLOEPerceptionInterface(
                target_classes=self.args.target)
            print("  Perception : MockYOLOEPerceptionInterface")
        else:
            self.perception = YOLOEPerceptionInterface(
                model_path=self.args.yoloe_model,
                target_classes=self.args.target,
                confidence_threshold=self.args.confidence_threshold,
                device=self.config.device,
            )
            print(f"  Perception : YOLOEPerceptionInterface ({self.args.yoloe_model})")

        self.control = create_control_interface(self.config, use_mock=self.use_mock_control)
        print(f"  Control    : {'Mock' if self.use_mock_control else 'bc_model'}")

        self.control_p = None
        if self.args.control_mode in ("proportional", "both"):
            self.control_p = ProportionalController(
                kp_x=self.args.p_gain_x,
                kp_y=self.args.p_gain_y,
                img_width=self.config.camera.width,
                img_height=self.config.camera.height,
            )
            print(f"  P-Ctrl     : kp_x={self.args.p_gain_x} kp_y={self.args.p_gain_y}")

        self.safety = SafetyManager(self.config)
        print("[VideoSim] Models ready.\n")

    # ── pipeline ───────────────────────────────────────────────────────────

    def _run_pipeline(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[DetectionResult, ControlAction, Optional[ControlAction], SafetyOutput]:
        detection = self.perception.detect(frame, return_heatmap=self.args.show_heatmap)

        ctrl_mode = self.args.control_mode
        action_p  = None

        if ctrl_mode in ("bc_model", "both"):
            action = self.control.compute_action(
                frame=frame, detection_result=detection,
                timestamp=timestamp, deterministic=True,
            )
        else:
            # will be set below
            action = None

        if ctrl_mode in ("proportional", "both") and self.control_p is not None:
            action_p = self.control_p.compute_action(
                frame=frame, detection_result=detection, timestamp=timestamp
            )

        if ctrl_mode == "proportional":
            action = action_p

        # safety filter (simulation → commands are logged but not sent)
        flipped = action.action * np.array([-1.0, -1.0])
        safety = self.safety.process(
            raw_action=flipped,
            detection_valid=not detection.no_detection,
            confidence=detection.confidence,
            timestamp=timestamp,
        )

        return detection, action, action_p, safety

    # ── single video ───────────────────────────────────────────────────────

    def process_video(self, video_path: str):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"[VideoSim] ERROR: file not found → {video_path}")
            return

        # ── output directory ──
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.args.output_dir) / f"{video_path.stem}_{timestamp_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)  # always save frames

        # ── open input ──
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[VideoSim] ERROR: cannot open {video_path}")
            return

        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out_fps   = self.args.fps_override if self.args.fps_override else src_fps
        src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output at original video resolution
        out_w, out_h = src_w, src_h

        # Model inference size (what the pipeline expects internally)
        mdl_w = self.config.camera.width
        mdl_h = self.config.camera.height

        print(f"[VideoSim] Input       : {video_path.name}  {src_w}x{src_h} @ {src_fps:.1f} fps  "
              f"({total_fr} frames)")
        print(f"[VideoSim] Model size  : {mdl_w}x{mdl_h}  |  Output size: {out_w}x{out_h}")

        # ── output video writer ──
        out_video_path = out_dir / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            print("[VideoSim] WARNING: VideoWriter failed to open. Trying XVID…")
            out_video_path = out_dir / "annotated.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, out_fps, (out_w, out_h))

        # ── CSV log ──
        csv_path = out_dir / "data.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_fields = [
            "frame_idx", "timestamp",
            "detection_valid", "conf",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "center_x", "center_y",
            "error_x", "error_y",
            "action_m1", "action_m2",
            "action_p_m1", "action_p_m2",
            "angle_deg",
            "safety_state", "safety_gain",
            "det_fps",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        # ── process loop ──
        frame_idx   = 0
        perf_times  = []
        start_time  = time.time()
        fps_display = 0.0
        angle_samples: List[float] = []   # collect per-frame angle differences

        print(f"[VideoSim] Processing → {out_dir}")
        print("           Press 'q' to abort early.\n")

        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if self.args.max_frames and frame_idx >= self.args.max_frames:
                print(f"[VideoSim] Reached --max-frames {self.args.max_frames}, stopping.")
                break

            t0 = time.time()

            # ── model inference: resize to model input size ──
            frame_mdl_bgr = cv2.resize(bgr, (mdl_w, mdl_h))
            frame_mdl_rgb = cv2.cvtColor(frame_mdl_bgr, cv2.COLOR_BGR2RGB)

            timestamp = frame_idx / src_fps

            # ── pipeline (runs at model resolution) ──
            detection, action, action_p, safety = self._run_pipeline(frame_mdl_rgb, timestamp)

            # ── draw annotations onto original-resolution frame ──
            # Scale detection bbox/center from model space → original space
            detection_out = _scale_detection(detection, mdl_w, mdl_h, out_w, out_h)

            frame_orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            annotated_rgb = draw_frame(
                frame=frame_orig_rgb,
                detection=detection_out,
                action=action,
                action_p=action_p,
                safety=safety,
                target_name=self.args.target,
                frame_idx=frame_idx,
                fps=fps_display,
                control_mode=self.args.control_mode,
                show_heatmap=self.args.show_heatmap,
            )

            # ── write output video (BGR) ──
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

            # ── always save individual frames ──
            cv2.imwrite(
                str(frames_dir / f"frame_{frame_idx:06d}.png"),
                cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
            )

            # ── CSV row ──
            bbox   = detection.bbox   if detection.bbox   is not None else np.zeros(4)
            center = detection.center if detection.center is not None else np.zeros(2)
            error  = detection.get_pixel_error() if detection.center is not None else np.zeros(2)

            # ── angle difference (both mode only) ──
            angle = None
            if self.args.control_mode == "both" and action_p is not None \
                    and action.valid and action_p.valid:
                angle = angle_between_actions(action.action, action_p.action)
                if angle is not None:
                    angle_samples.append(angle)
                    print(f"[AngleDiff] frame {frame_idx:>5}  {angle:+.1f}°  "
                          f"model={action.action}  p={action_p.action}")

            csv_writer.writerow({
                "frame_idx":      frame_idx,
                "timestamp":      f"{timestamp:.4f}",
                "detection_valid": int(not detection.no_detection),
                "conf":           f"{detection.confidence:.4f}",
                "bbox_x":  f"{bbox[0]:.1f}", "bbox_y": f"{bbox[1]:.1f}",
                "bbox_w":  f"{bbox[2]:.1f}", "bbox_h": f"{bbox[3]:.1f}",
                "center_x":f"{center[0]:.1f}", "center_y": f"{center[1]:.1f}",
                "error_x": f"{error[0]:.1f}",  "error_y":  f"{error[1]:.1f}",
                "action_m1": f"{action.action[0]:.4f}",
                "action_m2": f"{action.action[1]:.4f}",
                "action_p_m1": f"{action_p.action[0]:.4f}" if action_p else "",
                "action_p_m2": f"{action_p.action[1]:.4f}" if action_p else "",
                "angle_deg": f"{angle:.2f}" if angle is not None else "",
                "safety_state": safety.state.value,
                "safety_gain":  f"{safety.applied_gain:.3f}",
                "det_fps":      f"{detection.fps:.1f}",
            })

            # ── optional live display ──
            if self.args.display:
                cv2.imshow("VideoSim", cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[VideoSim] Aborted by user.")
                    break

            # ── timing / progress ──
            elapsed = time.time() - t0
            perf_times.append(elapsed)
            if len(perf_times) > 30:
                perf_times.pop(0)
            fps_display = 1.0 / (sum(perf_times) / len(perf_times))

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == 1:
                pct = (frame_idx / total_fr * 100) if total_fr > 0 else 0
                print(f"  [{pct:5.1f}%] frame {frame_idx:>5}"
                      f"  proc_fps={fps_display:.1f}"
                      f"  conf={detection.confidence:.2f}"
                      f"  cmd=({action.action[0]:.2f},{action.action[1]:.2f})")

        # ── cleanup ──
        cap.release()
        writer.release()
        csv_file.close()
        if self.args.display:
            cv2.destroyAllWindows()

        wall = time.time() - start_time
        print(f"\n[VideoSim] Done  ({frame_idx} frames in {wall:.1f}s)")
        print(f"  Annotated video : {out_video_path}")
        print(f"  Frames dir      : {frames_dir}  ({frame_idx} PNGs)")
        print(f"  CSV log         : {csv_path}")

        # ── angle difference summary ──
        if angle_samples:
            stats_path = out_dir / "angle_stats.txt"
            print_angle_stats(angle_samples, stats_path)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("VideoSim – bc_model inference on video files")
    print("=" * 60)
    print(f"  Target          : {args.target}")
    print(f"  Control mode    : {args.control_mode}")
    print(f"  Mock perception : {args.mock_all or args.mock_perception}")
    print(f"  Mock control    : {args.mock_all or args.mock_control}")
    print(f"  Output dir      : {args.output_dir}")
    print(f"  Videos          : {args.input}")
    print("=" * 60 + "\n")

    sim = VideoSimulator(args)

    for video_path in args.input:
        sim.process_video(video_path)

    print("[VideoSim] All videos processed.")


if __name__ == "__main__":
    main()