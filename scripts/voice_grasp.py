"""
voice_grasp.py — 语音指令版短轴夹取主程序
========================================
基于 scripts/main.py 的普通短轴夹取流程，在等待阶段加入语音识别：
  1. 初始化机械臂、夹爪、相机、YOLO，移动到预备位
  2. 后台线程循环录音 + SenseVoice ASR + 语义解析
  3. 主线程收到语音目标后，动态注入 YOLO open-vocabulary 类别
  4. 主线程采帧、筛选目标、估计短轴夹取姿态并执行
  5. G 仍可手动夹取当前最佳目标，Q/ESC 退出
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.camera import make_camera  # noqa: E402
from drivers.robot.rebot_arm import RebotArm  # noqa: E402
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye  # noqa: E402
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps, select_best_grasp  # noqa: E402
from utils.transforms import (  # noqa: E402
    canonicalize_parallel_gripper_tcp_rotation,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
)
from utils.voice.asr import model_cache_status, transcribe_with_timeout  # noqa: E402
from utils.voice.config import load_voice_config  # noqa: E402
from utils.voice.recorder import record_wav  # noqa: E402
from utils.voice.semantic import parse_command  # noqa: E402
from utils.yolo_utils import load_yolo  # noqa: E402


@dataclass(frozen=True)
class VoiceRequest:
    text: str
    target_class: str


class VoiceListener:
    def __init__(
        self,
        voice_cfg: dict[str, Any],
        out_queue: "queue.Queue[VoiceRequest]",
        busy_event: threading.Event,
    ) -> None:
        self._cfg = voice_cfg
        self._queue = out_queue
        self._busy_event = busy_event
        self._recognition_paused = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="voice-listener", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        _clear_status()

    def resume(self) -> None:
        self._recognition_paused.clear()
        _clear_status()

    def pause(self, message: str | None = None) -> None:
        was_paused = self._recognition_paused.is_set()
        self._recognition_paused.set()
        _clear_status()
        if message and not was_paused:
            print(f"{message}  [Keys] R=重新监听 Q/ESC=退出")

    def is_paused(self) -> bool:
        return self._recognition_paused.is_set()

    def _run(self) -> None:
        input_cfg = self._cfg["input"]
        mode = str(input_cfg.get("mode", "record")).lower()
        if mode == "text":
            self._handle_text(str(input_cfg.get("text") or ""))
            return

        while not self._stop.is_set():
            if self._recognition_paused.is_set():
                time.sleep(0.1)
                continue
            if self._busy_event.is_set():
                _status("[Voice] paused while grasping")
                time.sleep(0.1)
                continue
            try:
                if mode == "audio":
                    audio_value = input_cfg.get("audio")
                    if not audio_value:
                        print("[Voice] input.mode=audio but input.audio is empty")
                        return
                    self._handle_audio(Path(str(audio_value)).expanduser())
                    return

                if mode != "record":
                    print(f"[Voice] unsupported input.mode={mode}")
                    return

                seconds = float(self._cfg["recording"].get("seconds", 3.0))
                with tempfile.NamedTemporaryFile(prefix="voice_grasp_", suffix=".wav", delete=False) as tmp:
                    audio_path = Path(tmp.name)
                if self._cfg["recording"].get("countdown", True):
                    _countdown()
                _status(f"[Voice REC] start {seconds:.0f}s")
                record_wav(audio_path, seconds=seconds)
                _status("[Voice REC] end")
                if self._stop.is_set():
                    try:
                        audio_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    return
                if input_cfg.get("record_only", False):
                    _clear_status()
                    print(f"[Voice] recorded: {audio_path}")
                    return
                if self._recognition_paused.is_set() or self._busy_event.is_set():
                    self.pause("[Voice] manual action active; discarded recorded audio.")
                    try:
                        audio_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                self._handle_audio(audio_path)
                try:
                    audio_path.unlink(missing_ok=True)
                except OSError:
                    pass
            except Exception as exc:
                self.pause(f"[Voice] {exc}")
            time.sleep(0.2)

    def _handle_audio(self, audio_path: Path) -> None:
        asr_cfg = self._cfg["asr"]
        model_dir = str(asr_cfg.get("model_dir", "iic/SenseVoiceSmall"))
        cache_status = model_cache_status(model_dir)
        if cache_status.get("checked") and not cache_status.get("complete") and not asr_cfg.get("allow_incomplete_model", False):
            raise RuntimeError(f"SenseVoice model cache incomplete: {cache_status}")

        _status("[Voice ASR] running")
        transcript = transcribe_with_timeout(
            audio_path,
            model_dir=model_dir,
            device=asr_cfg.get("device"),
            language=str(asr_cfg.get("language", "zh")),
            timeout_s=float(asr_cfg.get("timeout_s", 120.0)),
            verbose=bool(asr_cfg.get("verbose", False)),
        )
        _status("[Voice ASR] done")
        self._handle_text(transcript["clean_text"])

    def _handle_text(self, text: str) -> None:
        intent = parse_command(text)
        _clear_status()
        if intent.action != "grasp" or intent.target is None or not intent.target.yolo_phrase:
            self.pause(f"[Voice] no valid target action={intent.action} text={text}")
            return
        request = VoiceRequest(text=text, target_class=intent.target.yolo_phrase)
        self._queue.put(request)
        self.pause(f"[Voice] target={request.target_class} text={request.text}")


def _move_ready(robot: RebotArm, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    robot.move_to(
        float(ready_cfg.get("x", 0.25)),
        float(ready_cfg.get("y", 0.0)),
        float(ready_cfg.get("z", 0.35)),
        float(ready_cfg.get("roll", 0.0)),
        float(ready_cfg.get("pitch", 1.2)),
        float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    robot.wait_motion(duration)


def _cam_to_base(T_hand_eye: np.ndarray, robot: RebotArm, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(robot.get_tcp_pose(), T_hand_eye, cfg)


def _execute_grasp(
    robot: RebotArm,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Grasp] --dry-run: 跳过执行")
        return False

    print("[Grasp] 打开夹爪...")
    robot.open_gripper()

    print("[Grasp] 移动到预夹取位...")
    if not robot.move_to(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Grasp] 预夹取 IK 失败，中止")
        return False
    robot.wait_motion(2.0)

    print("[Grasp] 移动到夹取位...")
    if not robot.move_to(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Grasp] 夹取 IK 失败，中止")
        return False
    robot.wait_motion(1.5)

    print("[Grasp] 夹取中...")
    ok = robot.grasp()
    print("[Grasp] 夹取成功，力控保持中" if ok else "[Grasp] 空夹取")

    print("[Grasp] 返回预备位...")
    _move_ready(robot, ready_cfg)
    return ok


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if best is not None:
        x_m, y_m, z_m = best.position.tolist()
        best_text = (
            f"best={best.class_name} conf={best.conf:.2f} "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
        )
        cv2.putText(
            display,
            best_text,
            (10, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 255, 140),
            2,
        )
    return display


def _print_best_grasp(grasp: GraspPose, tag: str) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print(f"\n[{tag}] 当前夹取:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _ensure_yolo_class(model: Any, yolo_opts: dict[str, Any], target_class: str) -> None:
    classes = list(yolo_opts.get("custom_classes", []))
    if target_class in classes:
        return
    classes.append(target_class)
    yolo_opts["custom_classes"] = classes
    if hasattr(model, "set_classes"):
        try:
            model.set_classes(classes)
            print(f"[Voice] YOLO classes += {target_class}")
        except Exception as exc:
            print(f"[Voice] YOLO set_classes failed: {exc}")


def _select_grasp_for_target(grasps: list[GraspPose], target_class: Optional[str]) -> Optional[GraspPose]:
    if not target_class:
        return select_best_grasp(grasps)
    target_norm = target_class.casefold()
    candidates = [
        grasp
        for grasp in grasps
        if grasp.is_valid
        and (
            grasp.class_name.casefold() == target_norm
            or target_norm in grasp.class_name.casefold()
            or grasp.class_name.casefold() in target_norm
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda grasp: grasp.conf)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _countdown() -> None:
    for value in (3, 2, 1):
        _status(f"[Voice REC] {value}")
        time.sleep(1)


def _status(message: str) -> None:
    if sys.stderr.isatty():
        print(f"\r{message:<56}", end="", file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def _clear_status() -> None:
    if sys.stderr.isatty():
        print("\r" + " " * 80 + "\r", end="", file=sys.stderr, flush=True)


def _run_grasp_once(
    *,
    tag: str,
    target_class: Optional[str],
    cam: Any,
    model: Any,
    yolo_opts: dict[str, Any],
    K: np.ndarray,
    depth_quantile: float,
    T_hand_eye: Optional[np.ndarray],
    cfg: dict[str, Any],
    robot: RebotArm,
    pregrasp_offset_m: float,
    insertion_depth_m: float,
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> tuple[bool, Optional[np.ndarray], list[GraspPose], Optional[GraspPose]]:
    if target_class:
        _ensure_yolo_class(model, yolo_opts, target_class)

    print(f"\n[{tag}] 采帧并估计夹取姿态 target={target_class or 'best'}...")
    snap_color, snap_depth = cam.get_frame()
    if snap_color is None or snap_depth is None:
        print(f"[{tag}] 采帧失败")
        return False, None, [], None

    snap_results = model.predict(
        snap_color,
        verbose=False,
        device=yolo_opts.get("device", "cpu"),
        conf=float(yolo_opts.get("conf", 0.25)),
        iou=float(yolo_opts.get("iou", 0.45)),
    )
    snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)
    best = _select_grasp_for_target(snap_grasps, target_class)
    if best is None:
        print(f"[{tag}] 未找到有效夹取候选 target={target_class or 'best'}")
        return False, snap_color, snap_grasps, None

    _print_best_grasp(best, tag)
    if T_hand_eye is None:
        print(f"[{tag}] 手眼标定不可用，无法执行夹取")
        return False, snap_color, snap_grasps, best

    T_cam2base = _cam_to_base(T_hand_eye, robot, cfg)
    grasp6d, pre6d = transform_grasp_pose_to_base(
        best.position,
        best.tcp_rotation,
        T_cam2base,
        pregrasp_offset_m,
        insertion_depth_m,
    )
    ok = _execute_grasp(robot, grasp6d, pre6d, ready_cfg, dry_run=dry_run)
    return ok, snap_color, snap_grasps, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="语音指令版短轴估计机械臂夹取主程序")
    parser.add_argument("--voice-config", default=None, help="语音与夹取运行配置 YAML，默认 config/voice.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    voice_cfg = load_voice_config(_resolve_project_path(args.voice_config) if args.voice_config else None)
    voice_grasp_cfg = voice_cfg.get("grasp", {})
    cfg = load_config(_resolve_project_path(voice_grasp_cfg.get("config", "config/default.yaml")))
    dry_run = not bool(voice_grasp_cfg.get("execute", False))

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        repo_root=robot_cfg.get("repo_root"),
        gripper_config=robot_cfg.get("gripper"),
        control_config=robot_cfg.get("control"),
    )
    robot.connect(enable=True)
    robot.init_gripper()

    print("[Robot] 移动到预备位置...")
    _move_ready(robot, ready_cfg)

    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
    if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
        print("[WARN] 手眼标定不可用或非 eye_in_hand，夹取执行将被禁用")
        T_hand_eye = None

    print(f"=== 相机: {cfg.get('camera', {}).get('type')} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(15)
    K = cam.K.astype(np.float32)

    yolo_cfg = cfg.get("yolo", {})
    gp_cfg = cfg.get("grasp_pipeline", {})
    grasp_cfg = gp_cfg.get("grasp", {})

    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
    insertion_depth_m = float(grasp_cfg.get("insertion_depth_m", 0.0))
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
    infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

    print(f"=== 加载 YOLO: {model_name} ===")
    model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

    voice_queue: "queue.Queue[VoiceRequest]" = queue.Queue()
    grasp_busy = threading.Event()
    voice_listener = VoiceListener(voice_cfg, voice_queue, grasp_busy)
    voice_listener.start()

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0
    last_voice_target: Optional[str] = None

    window_name = "Voice Grasp - Ordinary Grasp"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=手动夹取当前最佳  R=重新监听  Q/ESC=退出\n")

    try:
        while True:
            voice_request: Optional[VoiceRequest] = None
            try:
                voice_request = voice_queue.get_nowait()
            except queue.Empty:
                pass

            if voice_request is not None:
                last_voice_target = voice_request.target_class
                grasp_busy.set()
                try:
                    ok, snap_color, snap_grasps, best = _run_grasp_once(
                        tag="Voice",
                        target_class=voice_request.target_class,
                        cam=cam,
                        model=model,
                        yolo_opts=yolo_opts,
                        K=K,
                        depth_quantile=depth_quantile,
                        T_hand_eye=T_hand_eye,
                        cfg=cfg,
                        robot=robot,
                        pregrasp_offset_m=pregrasp_offset_m,
                        insertion_depth_m=insertion_depth_m,
                        ready_cfg=ready_cfg,
                        dry_run=dry_run,
                    )
                finally:
                    grasp_busy.clear()
                if snap_color is not None:
                    last_display = _render_display(snap_color, snap_grasps, best, "VOICE SNAPSHOT")
                    last_grasps = snap_grasps
                    frozen = True

            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K, depth_quantile=depth_quantile)

            voice_status = last_voice_target or ("press-R" if voice_listener.is_paused() else "listening")
            status = (
                f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | "
                f"voice={voice_status} | G=manual R=listen Q=quit"
            )
            best_live = select_best_grasp(last_grasps)
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_live, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                last_voice_target = None
                voice_listener.resume()
                continue

            if key in (ord("g"), ord("G")):
                voice_listener.pause("[Voice] manual action selected; voice recognition paused.")
                grasp_busy.set()
                try:
                    ok, snap_color, snap_grasps, best = _run_grasp_once(
                        tag="G",
                        target_class=None,
                        cam=cam,
                        model=model,
                        yolo_opts=yolo_opts,
                        K=K,
                        depth_quantile=depth_quantile,
                        T_hand_eye=T_hand_eye,
                        cfg=cfg,
                        robot=robot,
                        pregrasp_offset_m=pregrasp_offset_m,
                        insertion_depth_m=insertion_depth_m,
                        ready_cfg=ready_cfg,
                        dry_run=dry_run,
                    )
                finally:
                    grasp_busy.clear()
                if snap_color is not None:
                    last_display = _render_display(snap_color, snap_grasps, best, "SNAPSHOT")
                    last_grasps = snap_grasps
                    frozen = True

    finally:
        voice_listener.stop()
        print("\n[退出] 释放夹爪并回零...")
        try:
            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")
        robot.disconnect(safe_home=False)
        cam.close()
        cv2.destroyAllWindows()
        print("已退出。")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
