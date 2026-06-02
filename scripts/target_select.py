"""
target_select.py - OBB 交互式目标选择夹取脚本
=====================================
流程：
  1. 机械臂 + 夹爪使能，移动到预备位
  2. 实时相机预览 + YOLO 检测 + OBB 短轴夹取姿态估计
  3. Enter/Space：冻结当前帧，缓存所有识别物体和对应夹取估计
  4. 冻结后按数字键选择目标夹取，R 恢复实时预览，Q/ESC 退出
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (PROJECT_ROOT,):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from drivers.camera import make_camera
from drivers.robot.rebot_arm import RebotArm
from utils.camera_utils import load_config, load_hand_eye
from utils.ordinary_grasp import GraspPose, draw_grasp, estimate_grasps
from utils.transforms import (
    canonicalize_parallel_gripper_tcp_rotation,
    rotation_matrix_to_euler_zyx,
    transform_grasp_pose_to_base,
)
from utils.yolo_utils import load_yolo


@dataclass
class FrozenSnapshot:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    results: list[Any]
    grasps: list[GraspPose]
    selectable: list[GraspPose]
    display: np.ndarray


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


def _cam_to_base(T_hand_eye: np.ndarray, robot: RebotArm) -> np.ndarray:
    return robot.get_tcp_pose() @ T_hand_eye


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


def _valid_grasps(grasps: list[GraspPose]) -> list[GraspPose]:
    return sorted([grasp for grasp in grasps if grasp.is_valid], key=lambda grasp: grasp.conf, reverse=True)


def _draw_selectable_index(display: np.ndarray, grasp: GraspPose, index: int) -> None:
    x1, y1, _, _ = grasp.bbox_xyxy
    origin = (max(0, x1 + 6), max(26, y1 + 26))
    cv2.circle(display, origin, 16, (0, 255, 80), -1, cv2.LINE_AA)
    cv2.putText(display, str(index), (origin[0] - 8, origin[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    selectable: list[GraspPose],
    status_text: str,
    frozen: bool = False,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        draw_grasp(display, grasp)

    for index, grasp in enumerate(selectable[:9], start=1):
        _draw_selectable_index(display, grasp, index)

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    if frozen:
        cv2.putText(display, "[FROZEN]", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 2, cv2.LINE_AA)
    return display


def _print_snapshot(snapshot: FrozenSnapshot) -> None:
    print("\n[Select] 已冻结当前帧，候选目标如下：")
    if not snapshot.selectable:
        print("  无有效夹取候选。按 R 重新采帧，或 Q 退出。")
        return
    for index, grasp in enumerate(snapshot.selectable[:9], start=1):
        x1, y1, x2, y2 = grasp.bbox_xyxy
        x_m, y_m, z_m = grasp.position.tolist()
        print(
            f"  {index}. {grasp.class_name} conf={grasp.conf:.2f} "
            f"center={grasp.center_px} bbox=({x1},{y1},{x2},{y2}) "
            f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={grasp.jaw_width_m * 100:.1f}cm"
        )


def _print_selected_grasp(index: int, grasp: GraspPose) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print(f"\n[Select] 选择目标 #{index}:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def _make_snapshot(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    results: list[Any],
    K: np.ndarray,
    depth_quantile: float,
) -> FrozenSnapshot:
    grasps = estimate_grasps(results, depth_mm, K, depth_quantile=depth_quantile)
    selectable = _valid_grasps(grasps)
    status = "FROZEN | 1-9=select/grasp  R=reset  Q/ESC=quit"
    display = _render_display(color_bgr, grasps, selectable, status, frozen=True)
    return FrozenSnapshot(
        color_bgr=color_bgr.copy(),
        depth_mm=depth_mm.copy(),
        results=results,
        grasps=grasps,
        selectable=selectable,
        display=display,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBB 交互式目标选择夹取脚本")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只估计夹取姿态，不移动机械臂")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )

    print("=== 初始化机械臂 ===")
    robot = RebotArm(
        config_path=robot_cfg.get("config_path"),
        urdf_path=robot_cfg.get("urdf_path"),
        repo_root=robot_cfg.get("repo_root"),
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
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
    infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

    print(f"=== 加载 YOLO: {model_name} ===")
    model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    last_selectable: list[GraspPose] = []
    frozen_snapshot: Optional[FrozenSnapshot] = None
    frozen = False
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Select - OBB Grasp"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys] Enter/Space=冻结当前帧  1-9=选择夹取  R=恢复  Q/ESC=退出\n")

    try:
        while True:
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
                last_selectable = _valid_grasps(last_grasps)

            if frozen and frozen_snapshot is not None:
                display = frozen_snapshot.display.copy()
            else:
                status = f"LIVE {fps_value:.1f}fps | Enter=freeze  1-9=select after freeze  R=reset  Q=quit"
                display = _render_display(color_bgr, last_grasps, last_selectable, status, frozen=False)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                frozen_snapshot = None
                continue

            if key in (13, 10, ord(" ")):
                print("\n[Select] 冻结当前帧并缓存识别/夹取信息...")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[Select] 采帧失败")
                    continue
                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                frozen_snapshot = _make_snapshot(snap_color, snap_depth, snap_results, K, depth_quantile)
                frozen = True
                _print_snapshot(frozen_snapshot)
                continue

            if frozen and frozen_snapshot is not None and ord("1") <= key <= ord("9"):
                select_index = key - ord("0")
                if select_index > len(frozen_snapshot.selectable):
                    print(f"[Select] 编号 {select_index} 不存在")
                    continue

                selected = frozen_snapshot.selectable[select_index - 1]
                _print_selected_grasp(select_index, selected)

                if T_hand_eye is None:
                    print("[Select] 手眼标定不可用，无法执行夹取")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, robot)
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    selected.position,
                    selected.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                    float(grasp_cfg.get("insertion_depth_m", 0.0)),
                )
                _execute_grasp(robot, grasp6d, pre6d, ready_cfg, dry_run=args.dry_run)

    finally:
        print("\n[退出] 释放夹爪并回零...")
        try:
            robot.release_gripper()
            robot.safe_home()
        except Exception as exc:
            print(f"[退出] {exc}")
        robot.disconnect()
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
